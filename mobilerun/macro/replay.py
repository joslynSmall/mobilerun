"""
Macro Replay Module - Replay recorded UI automation sequences.

This module provides functionality to load and replay macro JSON files
that were generated during MobileAgent trajectory recording.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from mobilerun_core_local.driver.android import AndroidDriver

from mobilerun.agent.utils.trajectory import Trajectory
from mobilerun.macro.anchor import resolve_anchor_with_screen
from mobilerun.macro.handoff import run_agent_handoff
from mobilerun.macro.matcher import StateMatchResult, compare_states
from mobilerun.macro.resolutions import (
    find_matching_cases,
    format_case_summary,
    record_case,
)
from mobilerun.macro.state import check_macro_schema_version, normalize_ui_state
from mobilerun.tools.filters import DetailedFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.provider import AndroidStateProvider

logger = logging.getLogger("mobilerun-macro")

# Reverse map for legacy key_press macro entries
_KEYCODE_TO_BUTTON = {4: "back", 3: "home", 66: "enter"}


class MacroPlayer:
    """
    A class for loading and replaying Mobilerun macro sequences.

    This player can execute recorded UI actions (taps, swipes, text input, key presses)
    on Android devices using AndroidDriver.
    """

    def __init__(
        self,
        device_serial: str = None,
        delay_between_actions: float = 1.0,
        on_mismatch: str = "stop",
        state_timeout: float = 5.0,
        state_threshold: float = 0.85,
        state_poll_interval: float = 0.5,
        handoff_runner=None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        resolve_mode: str = "anchors",
    ):
        """
        Initialize the MacroPlayer.

        Args:
            device_serial: Serial number of the target device. If None, will use first available device.
            delay_between_actions: Delay in seconds between each action (default: 1.0s)
            resolve_mode: "anchors" re-resolves element anchors against the
                live UI before tapping (falls back to recorded coordinates
                only for anchor-less v2 actions); "coords" always uses the
                recorded coordinates (legacy behavior).
        """
        if resolve_mode not in ("anchors", "coords"):
            raise ValueError("resolve_mode must be 'anchors' or 'coords'")
        self.device_serial = device_serial
        self.delay_between_actions = delay_between_actions
        self.on_mismatch = on_mismatch
        self.state_timeout = state_timeout
        self.state_threshold = state_threshold
        self.state_poll_interval = state_poll_interval
        self.handoff_runner = handoff_runner or run_agent_handoff
        self.provider = provider
        self.model = model
        self.config_path = config_path
        self.resolve_mode = resolve_mode
        self.last_divergence: Optional[Dict[str, Any]] = None
        # Per-step execution ledger: [{step, action_type, mode, ...}].
        # mode is anchor/coords/agent/skip (+ strategy / reason details).
        self.step_results: list = []
        self.state_provider = None
        self.driver: AndroidDriver | None = None
        self.credential_manager = None
        self._credential_load_error: Optional[str] = None

    async def _initialize_driver(self) -> AndroidDriver:
        """Initialize AndroidDriver for the target device."""
        if self.driver is None:
            self.driver = AndroidDriver(serial=self.device_serial)
            await self.driver.connect()
            logger.info(f"🤖 Initialized driver for device: {self.device_serial}")
        return self.driver

    def load_macro_from_file(self, macro_file_path: str) -> Dict[str, Any]:
        """
        Load macro data from a JSON file.

        Args:
            macro_file_path: Path to the macro JSON file

        Returns:
            Dictionary containing the macro data
        """
        return Trajectory.load_macro_sequence(macro_file_path)

    def load_macro_from_folder(self, trajectory_folder: str) -> Dict[str, Any]:
        """
        Load macro data from a trajectory folder.

        Args:
            trajectory_folder: Path to the trajectory folder containing macro.json

        Returns:
            Dictionary containing the macro data
        """
        return Trajectory.load_macro_sequence(trajectory_folder)

    def _load_credential_manager(self):
        if self.credential_manager is not None:
            return self.credential_manager

        from mobilerun.config_manager.loader import ConfigLoader
        from mobilerun.credential_manager import FileCredentialManager

        config_path = self.config_path
        if config_path is None:
            user_config_path = ConfigLoader.get_user_config_path()
            if not user_config_path.exists():
                self._credential_load_error = "no Mobilerun config file was found"
                return None
            config_path = str(user_config_path)

        try:
            config = ConfigLoader.load(config_path)
            credential_manager = FileCredentialManager(config.credentials)
        except Exception as e:
            self._credential_load_error = str(e)
            logger.debug("Failed to load credentials for macro replay: %s", e)
            return None

        if not credential_manager.secrets:
            self._credential_load_error = "credentials are disabled or empty"
            return None

        self.credential_manager = credential_manager
        self._credential_load_error = None
        return self.credential_manager

    async def get_current_state_snapshot(self) -> Dict[str, Any]:
        driver = await self._initialize_driver()
        if self.state_provider is None and isinstance(driver, AndroidDriver):
            self.state_provider = AndroidStateProvider(
                driver,
                tree_filter=DetailedFilter(),
                tree_formatter=IndexedFormatter(),
                use_normalized=False,
            )
        if self.state_provider is not None:
            try:
                return normalize_ui_state(await self.state_provider.get_state())
            except Exception as e:
                logger.debug("Falling back to raw driver state: %s", e)
        raw_state = await driver.get_ui_tree()
        return normalize_ui_state(raw_state)

    async def wait_for_pre_state(
        self, pre_state: Optional[Dict[str, Any]], step_number: int
    ) -> tuple[bool, Optional[Dict[str, Any]], Optional[StateMatchResult]]:
        if not pre_state:
            return True, None, None

        deadline = time.monotonic() + self.state_timeout
        last_state = None
        last_result = None

        while True:
            current_state = await self.get_current_state_snapshot()
            result = compare_states(pre_state, current_state, self.state_threshold)
            if result.matches:
                return True, current_state, result

            last_state = current_state
            last_result = result
            if time.monotonic() >= deadline:
                logger.error(
                    "State mismatch before step %s: %s", step_number, result.reason
                )
                return False, last_state, last_result

            await asyncio.sleep(self.state_poll_interval)

    async def replay_action(
        self, action: Dict[str, Any], current_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Replay a single action.

        Args:
            action: Action dictionary containing type and parameters

        Returns:
            True if action was executed successfully, False otherwise
        """
        driver = await self._initialize_driver()
        action_type = action.get("action_type", action.get("type", "unknown"))

        try:
            if action_type == "start_app":
                package = action.get("package")
                activity = action.get("activity", None)
                await driver.start_app(package, activity)
                return True

            elif action_type == "tap":
                x = action.get("x", 0)
                y = action.get("y", 0)
                logger.info(f"🫰 Tapping at ({x}, {y})")
                await driver.tap(x, y)
                return True

            elif action_type == "swipe":
                start_x = action.get("start_x", 0)
                start_y = action.get("start_y", 0)
                end_x = action.get("end_x", 0)
                end_y = action.get("end_y", 0)
                duration_ms = action.get("duration_ms", 300)

                logger.info(
                    f"👆 Swiping from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_ms}ms"
                )
                await driver.swipe(start_x, start_y, end_x, end_y, duration_ms)
                # Additional wait after swipe for UI to settle
                await asyncio.sleep(2)
                return True

            elif action_type == "drag":
                start_x = action.get("start_x", 0)
                start_y = action.get("start_y", 0)
                end_x = action.get("end_x", 0)
                end_y = action.get("end_y", 0)
                duration = action.get(
                    "duration", action.get("duration_ms", 300) / 1000.0
                )

                logger.info(
                    f"👆 Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})"
                )
                await driver.drag(start_x, start_y, end_x, end_y, duration)
                return True

            elif action_type == "input_text":
                text = action.get("text", "")
                clear = action.get("clear", False)
                logger.info(f"⌨️  Inputting text: '{text}'")
                await driver.input_text(text, clear)
                return True

            elif action_type == "type_secret":
                secret_id = action.get("secret_id")
                clear = action.get("clear", False)
                if not secret_id:
                    logger.error("❌ Secret macro action is missing secret_id")
                    return False

                credential_manager = self._load_credential_manager()
                if credential_manager is None:
                    detail = (
                        f": {self._credential_load_error}"
                        if self._credential_load_error
                        else ""
                    )
                    logger.error(
                        "❌ Cannot replay secret '%s'; no credentials are available%s",
                        secret_id,
                        detail,
                    )
                    return False

                try:
                    secret_value = await credential_manager.resolve_key(secret_id)
                except Exception as e:
                    logger.error("❌ Secret '%s' is not available: %s", secret_id, e)
                    return False

                logger.info("⌨️  Inputting secret: '%s'", secret_id)
                await driver.input_text(secret_value, clear)
                return True

            elif action_type == "key_press":
                keycode = action.get("keycode", 0)
                button = _KEYCODE_TO_BUTTON.get(keycode)
                if button:
                    logger.info(f"🔘 Pressing button: {button}")
                    await driver.press_button(button)
                else:
                    logger.warning(f"⚠️  Unknown keycode {keycode}, skipping")
                return True

            elif action_type == "button_press":
                button = action.get("button", "")
                logger.info(f"🔘 Pressing button: {button}")
                await driver.press_button(button)
                return True

            elif action_type == "back":
                logger.info("⬅️  Pressing back button")
                await driver.press_button("back")
                return True

            elif action_type == "wait":
                duration = action.get("duration", 1.0)
                logger.info(f"⏳ Waiting for {duration} seconds")
                await asyncio.sleep(duration)
                return True

            else:
                logger.warning(f"⚠️  Unknown action type: {action_type}")
                return False

        except Exception as e:
            logger.error(f"❌ Error executing action {action_type}: {e}")
            return False

    def _resolve_action(
        self, action: Dict[str, Any], current_state: Optional[Dict[str, Any]]
    ) -> tuple:
        """Resolve an anchored action against the live UI.

        Returns ``(action_to_execute, info)`` where info describes how the
        coordinates were chosen: ``{"mode": "anchor", "strategy": ...}``,
        ``{"mode": "coords", ...}``, or
        ``{"mode": "unresolvable", "reason": ...}`` when an anchor exists
        but matches nothing (caller treats this as a divergence).
        """
        anchor = action.get("anchor")
        if (
            self.resolve_mode != "anchors"
            or not anchor
            or action.get("action_type") != "tap"
            or not current_state
        ):
            return action, {"mode": "coords", "reason": "no anchor or coords mode"}
        nodes = current_state.get("nodes") or []
        hit = resolve_anchor_with_screen(
            anchor,
            nodes,
            (action.get("pre_state") or {}).get("screen"),
            current_state.get("screen"),
        )
        if hit is None:
            return action, {"mode": "unresolvable", "reason": "anchor matched nothing"}
        (x, y), strategy = hit
        resolved = dict(action)
        resolved["x"], resolved["y"] = x, y
        logger.info(f"   🎯 Anchor resolved via {strategy}: ({x}, {y})")
        return resolved, {"mode": "anchor", "strategy": strategy}

    async def _agent_handoff_with_case(
        self,
        *,
        description: str,
        macro_data: Dict[str, Any],
        macro_source_path: Optional[str],
        action: Dict[str, Any],
        step: int,
        action_type: str,
        divergence: Dict[str, Any],
        current_state: Dict[str, Any],
    ) -> bool:
        """Hand off to the agent, enumerating + recording resolution cases."""
        matched: list = []
        if macro_source_path:
            try:
                matched = find_matching_cases(
                    macro_source_path, current_state, diverged_step=step
                )
            except Exception as e:
                logger.debug("Case lookup failed: %s", e)
        if matched:
            logger.warning(f"📚 Found {len(matched)} known case(s) for this state:")
            for score, case in matched:
                logger.warning(f"   {format_case_summary(score, case)}")
        else:
            logger.info("📚 No known cases for this state; recording a new one")

        case_context = [format_case_summary(s, c) for s, c in matched]
        sink: Dict[str, Any] = {}
        try:
            ok = await self.handoff_runner(
                goal=description,
                device_serial=self.device_serial,
                provider=self.provider,
                model=self.model,
                config_path=self.config_path,
                divergence=divergence,
                current_state=current_state,
                remaining_actions=macro_data["actions"][step - 1 :],
                case_context=case_context,
                result_sink=sink,
            )
        except TypeError:
            # Custom handoff_runner without the extended kwargs.
            ok = await self.handoff_runner(
                goal=description,
                device_serial=self.device_serial,
                provider=self.provider,
                model=self.model,
                config_path=self.config_path,
                divergence=divergence,
                current_state=current_state or {},
                remaining_actions=macro_data["actions"][step - 1 :],
            )
        self.step_results.append(
            {
                "step": step,
                "action_type": action_type,
                "mode": "agent",
                "success": bool(ok),
                "precedent_cases": [c.get("case_id") for _, c in matched],
            }
        )
        if macro_source_path:
            try:
                record_case(
                    macro_source_path,
                    diverged_step=step,
                    action=action,
                    divergence=divergence,
                    snapshot=current_state,
                    resolution={
                        "mode": "agent",
                        "success": bool(ok),
                        "summary": f"agent handoff ({'ok' if ok else 'failed'})",
                        "agent_trajectory": sink.get("agent_trajectory"),
                        "agent_actions": sink.get("agent_actions"),
                        "precedent_cases": [c.get("case_id") for _, c in matched],
                    },
                )
            except Exception as e:
                logger.debug("Case recording failed: %s", e)
        return bool(ok)

    async def replay_macro(
        self,
        macro_data: Dict[str, Any],
        start_from_step: int = 0,
        max_steps: Optional[int] = None,
        macro_source_path: Optional[str] = None,
    ) -> bool:
        """
        Replay a complete macro sequence.

        Args:
            macro_data: Macro data dictionary loaded from JSON
            start_from_step: Step number to start from (0-based, default: 0)
            max_steps: Maximum number of steps to execute (default: all)
            macro_source_path: Macro file or trajectory folder path. Used to
                load/enumerate divergence cases in resolutions.json.

        Returns:
            True if all actions were executed successfully, False otherwise
        """
        if not macro_data or "actions" not in macro_data:
            logger.error("❌ Invalid macro data - no actions found")
            return False
        check_macro_schema_version(macro_data)

        actions = macro_data["actions"]
        description = macro_data.get("description", "Unknown macro")
        total_actions = len(actions)

        # Apply start_from_step and max_steps filters
        if start_from_step > 0:
            actions = actions[start_from_step:]
            logger.info(f"📍 Starting from step {start_from_step + 1}")

        if max_steps is not None:
            actions = actions[:max_steps]
            logger.info(f"🎯 Limiting to {max_steps} steps")

        logger.info(f"🎬 Starting macro replay: '{description}'")
        logger.info(f"📊 Total actions to execute: {len(actions)} / {total_actions}")

        success_count = 0
        failed_count = 0

        for i, action in enumerate(actions, start=start_from_step + 1):
            action_type = action.get("action_type", action.get("type", "unknown"))
            description_text = action.get("description", "")

            logger.info(f"\n📍 Step {i}/{total_actions}: {action_type}")
            if description_text:
                logger.info(f"   Description: {description_text}")

            state_matched, current_state, match_result = await self.wait_for_pre_state(
                action.get("pre_state"), i
            )
            if not state_matched:
                self.last_divergence = {
                    "step": i,
                    "action_type": action_type,
                    "reason": match_result.reason if match_result else "state mismatch",
                    "score": match_result.score if match_result else None,
                }
                if self.on_mismatch == "agent":
                    logger.warning("🤖 Handing off to agent after macro divergence")
                    return await self._agent_handoff_with_case(
                        description=description,
                        macro_data=macro_data,
                        macro_source_path=macro_source_path,
                        action=action,
                        step=i,
                        action_type=action_type,
                        divergence=self.last_divergence,
                        current_state=current_state or {},
                    )
                return False

            # Anchor resolution (schema v3): re-resolve the element against
            # the live UI; unresolvable anchors are a divergence.
            exec_action, resolve_info = self._resolve_action(action, current_state)
            if resolve_info["mode"] == "unresolvable":
                self.last_divergence = {
                    "step": i,
                    "action_type": action_type,
                    "reason": "anchor unresolvable: "
                    + resolve_info.get("reason", "no match"),
                    "score": None,
                    "anchor": action.get("anchor"),
                }
                logger.warning(
                    f"🎯 Anchor unresolvable at step {i}; "
                    f"recorded coords kept as fallback but not trusted"
                )
                if self.on_mismatch == "agent":
                    logger.warning("🤖 Handing off to agent after anchor divergence")
                    return await self._agent_handoff_with_case(
                        description=description,
                        macro_data=macro_data,
                        macro_source_path=macro_source_path,
                        action=action,
                        step=i,
                        action_type=action_type,
                        divergence=self.last_divergence,
                        current_state=current_state or {},
                    )
                return False

            # Execute the action
            success = await self.replay_action(
                exec_action, current_state=current_state
            )
            self.step_results.append(
                {
                    "step": i,
                    "action_type": action_type,
                    "mode": resolve_info["mode"],
                    "strategy": resolve_info.get("strategy"),
                    "success": success,
                }
            )

            if success:
                success_count += 1
                logger.info("   ✅ Action completed successfully")
            else:
                failed_count += 1
                logger.error("   ❌ Action failed")

            # Wait between actions (except for the last one)
            if i < start_from_step + len(actions):
                logger.debug(f"   ⏳ Waiting {self.delay_between_actions}s...")
                await asyncio.sleep(self.delay_between_actions)

        # Summary
        total_executed = success_count + failed_count
        success_rate = (
            (success_count / total_executed * 100) if total_executed > 0 else 0
        )

        logger.info("\n🎉 Macro replay completed!")
        logger.info(
            f"📊 Success: {success_count}/{total_executed} ({success_rate:.1f}%)"
        )

        if failed_count > 0:
            logger.warning(f"⚠️  Failed actions: {failed_count}")

        if self.step_results:
            modes: Dict[str, int] = {}
            for entry in self.step_results:
                key = entry.get("mode", "?")
                strategy = entry.get("strategy")
                if strategy:
                    key = f"{key}:{strategy}"
                modes[key] = modes.get(key, 0) + 1
            breakdown = ", ".join(f"{k}×{v}" for k, v in sorted(modes.items()))
            logger.info(f"🧭 Step modes: {breakdown}")

        return failed_count == 0


# Utility functions for convenience


async def replay_macro_file(
    macro_file_path: str,
    device_serial: str = None,
    delay_between_actions: float = 1.0,
    start_from_step: int = 0,
    max_steps: Optional[int] = None,
    on_mismatch: str = "stop",
    state_timeout: float = 5.0,
    state_threshold: float = 0.85,
    resolve_mode: str = "anchors",
) -> bool:
    """
    Convenience function to replay a macro from a file.

    Args:
        macro_file_path: Path to the macro JSON file
        device_serial: Target device serial (optional)
        delay_between_actions: Delay between actions in seconds
        start_from_step: Step to start from (0-based)
        max_steps: Maximum steps to execute
        resolve_mode: "anchors" or "coords" (see MacroPlayer)

    Returns:
        True if replay was successful, False otherwise
    """
    player = MacroPlayer(
        device_serial=device_serial,
        delay_between_actions=delay_between_actions,
        on_mismatch=on_mismatch,
        state_timeout=state_timeout,
        state_threshold=state_threshold,
        resolve_mode=resolve_mode,
    )

    try:
        macro_data = player.load_macro_from_file(macro_file_path)
        return await player.replay_macro(
            macro_data,
            start_from_step=start_from_step,
            max_steps=max_steps,
            macro_source_path=macro_file_path,
        )
    except Exception as e:
        logger.error(f"❌ Error replaying macro file {macro_file_path}: {e}")
        return False


async def replay_macro_folder(
    trajectory_folder: str,
    device_serial: str = None,
    delay_between_actions: float = 1.0,
    start_from_step: int = 0,
    max_steps: Optional[int] = None,
    on_mismatch: str = "stop",
    state_timeout: float = 5.0,
    state_threshold: float = 0.85,
    resolve_mode: str = "anchors",
) -> bool:
    """
    Convenience function to replay a macro from a trajectory folder.

    Args:
        trajectory_folder: Path to the trajectory folder containing macro.json
        device_serial: Target device serial (optional)
        delay_between_actions: Delay between actions in seconds
        start_from_step: Step to start from (0-based)
        max_steps: Maximum steps to execute
        resolve_mode: "anchors" or "coords" (see MacroPlayer)

    Returns:
        True if replay was successful, False otherwise
    """
    player = MacroPlayer(
        device_serial=device_serial,
        delay_between_actions=delay_between_actions,
        on_mismatch=on_mismatch,
        state_timeout=state_timeout,
        state_threshold=state_threshold,
        resolve_mode=resolve_mode,
    )

    try:
        macro_data = player.load_macro_from_folder(trajectory_folder)
        return await player.replay_macro(
            macro_data,
            start_from_step=start_from_step,
            max_steps=max_steps,
            macro_source_path=trajectory_folder,
        )
    except Exception as e:
        logger.error(f"❌ Error replaying macro folder {trajectory_folder}: {e}")
        return False
