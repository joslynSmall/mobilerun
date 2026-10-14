"""Divergence case library for macro replay.

When anchored/coordinate replay cannot proceed (state mismatch or anchor
unresolvable) and an agent takes over, the handoff is recorded as a case::

    diverged step + original snapshot  ->  resolution (how the agent fixed it)

Cases live next to the macro (``<macro_folder>/resolutions.json``) so they
travel with it. On the next divergence, matching cases are enumerated and
passed to the agent as precedent, and operators can list them via
``mobilerun macro cases <path>`` to pick a resolution for a known situation.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mobilerun.macro.matcher import compare_states

logger = logging.getLogger("mobilerun-macro")

RESOLUTIONS_FILENAME = "resolutions.json"


def resolutions_path_for(macro_path: str) -> Path:
    """Return the resolutions file for a macro file or trajectory folder."""
    p = Path(macro_path)
    folder = p if p.is_dir() else p.parent
    return folder / RESOLUTIONS_FILENAME


def load_cases(macro_path: str) -> List[Dict[str, Any]]:
    """Load recorded divergence cases for a macro (empty list if none)."""
    path = resolutions_path_for(macro_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read resolutions file %s: %s", path, e)
        return []
    cases = data.get("cases") if isinstance(data, dict) else None
    return cases if isinstance(cases, list) else []


def record_case(
    macro_path: str,
    *,
    diverged_step: int,
    action: Dict[str, Any],
    divergence: Dict[str, Any],
    snapshot: Dict[str, Any],
    resolution: Dict[str, Any],
) -> Optional[str]:
    """Append a divergence case. Returns the case_id, or None on failure."""
    path = resolutions_path_for(macro_path)
    cases = load_cases(macro_path)
    case_id = uuid.uuid4().hex[:8]
    cases.append(
        {
            "case_id": case_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "diverged_step": diverged_step,
            "action_type": action.get("action_type", action.get("type", "unknown")),
            "action": {k: v for k, v in action.items() if k != "pre_state"},
            "divergence": divergence,
            "snapshot": snapshot,
            "resolution": resolution,
        }
    )
    try:
        path.write_text(
            json.dumps(
                {"cases": cases},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Could not write resolutions file %s: %s", path, e)
        return None
    logger.info("Recorded divergence case %s (step %s)", case_id, diverged_step)
    return case_id


def find_matching_cases(
    macro_path: str,
    current_snapshot: Dict[str, Any],
    *,
    threshold: float = 0.85,
    diverged_step: Optional[int] = None,
    limit: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Enumerate recorded cases whose snapshot matches the current state.

    Returns ``[(score, case), ...]`` sorted by score descending. When
    ``diverged_step`` is given, cases from the same step rank first.
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for case in load_cases(macro_path):
        snapshot = case.get("snapshot") or {}
        try:
            result = compare_states(snapshot, current_snapshot, threshold=threshold)
        except Exception:
            continue
        if not result.matches:
            continue
        bonus = 0.0
        if diverged_step is not None and case.get("diverged_step") == diverged_step:
            bonus = 0.05
        scored.append((min(1.0, result.score + bonus), case))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def format_case_summary(score: float, case: Dict[str, Any]) -> str:
    """One-line human summary for listing / agent context."""
    resolution = case.get("resolution") or {}
    return (
        f"[{case.get('case_id')}] step {case.get('diverged_step')} "
        f"{case.get('action_type')} (match {score:.2f}): "
        f"{resolution.get('mode', '?')} - {resolution.get('summary', '')[:100]}"
    )
