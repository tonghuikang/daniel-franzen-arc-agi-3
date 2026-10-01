"""Action names and the formatting of action-related values for the prompt.

Covers the mapping between model-facing labels and engine actions, the list of
currently valid actions as the model sees it, the reason an action sequence
stopped early and the sentence explaining that stop, and the 1-indexed action
number shown in logs and prompts.
"""
from __future__ import annotations

from typing import Any, Iterable


ENGINE_TO_MODEL_ACTION = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
    "ACTION7": "UNDO",
    "RESET": "RESET",
}

MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}


def to_model_action(name: str | None) -> str:
    raw = str(name or "").strip().upper()

    return ENGINE_TO_MODEL_ACTION.get(raw, raw)


def to_engine_action(name: str | None) -> str | None:
    raw = str(name or "").strip().upper()
    if not raw:
        return None
    if raw in ENGINE_TO_MODEL_ACTION:
        return raw

    return MODEL_TO_ENGINE_ACTION.get(raw)


def to_model_actions(names: Iterable[str] | None) -> list[str]:
    resolved: list[str] = []
    for name in names or []:
        label = to_model_action(name)
        if label and label not in resolved:
            resolved.append(label)

    return resolved


def format_valid_action_line(valid_actions: list[str] | None) -> str:
    names = to_model_actions(valid_actions)
    if not names:
        return "unknown"

    return ", ".join(names)


def terminal_action_reason(result: dict[str, Any]) -> str | None:
    if result.get("run_complete"):
        return "run_complete"
    if result.get("game_over"):
        return "game_over"
    if result.get("level_completed"):
        return "level_completed"
    if result.get("done"):
        return "done"

    return None


def terminal_action_stop_detail(reason: str | None) -> str:
    if reason == "run_complete":
        return "No further actions were executed because the run is already complete."
    if reason == "game_over":
        return (
            "No further actions were executed because the previous action reached GAME_OVER; "
            "the runner will auto-reset before the next analyzer turn."
        )
    if reason == "level_completed":
        return (
            "No further actions were executed because the previous action completed a level; "
            "re-ground on the new scene before acting again."
        )
    if reason == "done":
        return "No further actions were executed because the environment reported done."

    return "No further actions were executed because the previous action reached a terminal state."


def display_action_number(action_num: int) -> int:
    return max(1, int(action_num) + 1)
