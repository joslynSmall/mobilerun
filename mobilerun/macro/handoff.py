"""Agent handoff for guarded macro replay divergence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mobilerun-macro")

# Heavy per-action snapshots are stripped when persisting the agent's
# resolution actions into a divergence case.
_STRIP_ACTION_KEYS = ("pre_state", "post_state", "screen")


def _slim_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slimmed = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        slimmed.append({k: v for k, v in action.items() if k not in _STRIP_ACTION_KEYS})
    return slimmed


def _newest_trajectory_dir(base: Path, before: set) -> Optional[Path]:
    try:
        candidates = [
            p
            for p in base.iterdir()
            if p.is_dir() and p.name not in before and (p / "macro.json").exists()
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _capture_agent_trajectory(
    trajectory_base: Optional[str],
) -> Dict[str, Any]:
    """Best-effort snapshot helper factory state is captured by the caller.

    The actual diffing happens in run_agent_handoff; this just resolves the
    base directory for trajectory output.
    """
    if trajectory_base:
        base = Path(trajectory_base)
    else:
        base = Path.cwd() / "trajectories"
    return {"base": base}


async def run_agent_handoff(
    *,
    goal: str,
    device_serial: Optional[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    divergence: Dict[str, Any],
    current_state: Dict[str, Any],
    remaining_actions: List[Dict[str, Any]],
    case_context: Optional[List[str]] = None,
    result_sink: Optional[Dict[str, Any]] = None,
) -> bool:
    """Invoke the normal Mobilerun agent after macro replay diverges.

    When ``result_sink`` is given, it is filled with the agent's trajectory
    folder and slimmed resolution actions so the caller can persist the
    handoff as a divergence case::

        {"agent_trajectory": str | None, "agent_actions": [...]}
    """
    from mobilerun.cli.main import run_command

    _ = divergence, current_state, remaining_actions

    effective_goal = goal
    if case_context:
        precedent = "\n".join(f"- {line}" for line in case_context)
        effective_goal = (
            f"{goal}\n\nKnown past divergences in similar states and how "
            f"they were resolved (follow the precedent when applicable):\n"
            f"{precedent}"
        )

    base = _capture_agent_trajectory(None)["base"]
    try:
        before = {p.name for p in base.iterdir() if p.is_dir()}
    except OSError:
        before = set()

    ok = await run_command(
        effective_goal,
        config_path=config_path,
        device=device_serial,
        provider=provider,
        model=model,
        save_trajectory="step",
    )

    if result_sink is not None:
        newest = _newest_trajectory_dir(base, before)
        if newest is not None:
            result_sink["agent_trajectory"] = str(newest)
            try:
                macro = json.loads((newest / "macro.json").read_text(encoding="utf-8"))
                result_sink["agent_actions"] = _slim_actions(macro.get("actions", []))
            except Exception as e:
                logger.debug("Could not read handoff agent macro: %s", e)
                result_sink["agent_actions"] = []
        else:
            result_sink["agent_trajectory"] = None
            result_sink["agent_actions"] = []

    return ok
