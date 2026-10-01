"""Structured runtime state shared with created Python tools."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"
ANIMATION_HISTORY_DEPTH = 16


@dataclass
class RuntimeStateIoSection:
    """Accumulated cost of one direction (write or read) of the state file."""

    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def record(self, seconds: float) -> None:
        self.count += 1
        self.total_seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_seconds": round(self.total_seconds, 3),
            "max_seconds": round(self.max_seconds, 3),
        }


@dataclass
class RuntimeStateIoStats:
    """Per-state-file totals of `write_runtime_state` / `load_runtime_state`,
    collected for the whole game and reported once in its `game_end` metrics."""

    write: RuntimeStateIoSection = field(default_factory=RuntimeStateIoSection)
    read: RuntimeStateIoSection = field(default_factory=RuntimeStateIoSection)

    def to_payload(self) -> dict[str, Any]:
        return {"write": self.write.to_payload(), "read": self.read.to_payload()}


_io_stats_by_path: dict[Path, RuntimeStateIoStats] = {}


def runtime_state_io_stats(path: Path) -> RuntimeStateIoStats:
    return _io_stats_by_path.setdefault(path, RuntimeStateIoStats())


def pop_runtime_state_io_stats(path: Path) -> RuntimeStateIoStats:
    """Return the totals for `path` and forget them, so the next game on the
    same path starts from zero."""

    return _io_stats_by_path.pop(path, RuntimeStateIoStats())


@dataclass(frozen=True)
class Frame:
    """One board as the agent sees it.

    `animation` holds the frames the action that produced this frame played, `grid`
    itself last, with frames the engine held across ticks collapsed and the rest
    sampled down to a fixed cap; `frame_count` is how many of them there are, so the
    two always agree and both say the action did not animate when it is 1.
    """

    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int
    animation: tuple[tuple[tuple[int, ...], ...], ...] = ()
    frame_count: int = 1

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    """One executed action and the board it produced.

    `result` is the compact outcome of that action and `plan` the one-line intent the
    model passed when it issued the action, so a later turn can search the whole game
    for score changes, dead actions and what it was trying at the time, long after the
    turn that acted has left the context window.
    """

    action: str
    frame: Frame
    result: dict[str, Any] = field(default_factory=dict)
    plan: str = ""


@dataclass(frozen=True)
class GameOver:
    """One game over and the life of actions that ended in it, the last of them the action
    the game over followed.

    A life is the run of actions since the level was entered or since the previous game over
    on it, which is the span a level's own action budget follows. `frame` is the board the
    game over was declared on, carrying the animation that action played, and only the newest
    game over of a level keeps it. On the older ones it is `None`, because their boards are
    still in `history`, which the state file keeps whole, and only the animation is theirs
    alone to lose.
    """

    actions: tuple[str, ...]
    frame: Frame | None


@dataclass(frozen=True)
class RuntimeState:
    """The runtime state as one value, read from the state file or built in memory.

    `game_overs` holds the game overs on the level the game currently stands on, oldest
    first, and `actions_this_life` counts the actions executed since that level last
    started. Both start over when a level is cleared, because a board the game has left
    is one the agent can no longer act on, which is also why a game over carries no level of
    its own. Every entry belongs to the level the game currently stands on.
    """

    current_frame: Frame | None
    history: list[HistoryEntry]
    game_overs: tuple[GameOver, ...] = ()
    actions_this_life: int = 0


def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells: list[int] = []
        for cell in row:
            try:
                cells.append(int(cell))
            except (TypeError, ValueError):
                cells.append(0)
        rows.append(tuple(cells))
    return tuple(rows)


def changed_pixel_count(
    previous_grid: tuple[tuple[int, ...], ...],
    current_grid: tuple[tuple[int, ...], ...],
) -> int:
    return sum(
        1
        for previous_row, current_row in zip(previous_grid, current_grid)
        for previous_cell, current_cell in zip(previous_row, current_row)
        if previous_cell != current_cell
    )


def total_pixel_count(grid: tuple[tuple[int, ...], ...]) -> int:
    return sum(len(row) for row in grid)


CHANGED_REGION_LIMIT = 4


def changed_regions(
    previous_grid: tuple[tuple[int, ...], ...],
    current_grid: tuple[tuple[int, ...], ...],
    *,
    limit: int = CHANGED_REGION_LIMIT,
) -> tuple[list[dict[str, Any]], int]:
    """Group the cells that differ into 4-connected regions, largest first.

    Returns the largest `limit` regions plus how many regions there are in total,
    so a caller can tell a few moved objects from many scattered changes.
    """
    changed_cells = {
        (row_index, col_index)
        for row_index, (previous_row, current_row) in enumerate(zip(previous_grid, current_grid))
        for col_index, (previous_cell, current_cell) in enumerate(zip(previous_row, current_row))
        if previous_cell != current_cell
    }
    regions: list[dict[str, Any]] = []
    while changed_cells:
        pending = [changed_cells.pop()]
        region_cells = list(pending)
        while pending:
            row_index, col_index = pending.pop()
            for neighbor in (
                (row_index - 1, col_index),
                (row_index + 1, col_index),
                (row_index, col_index - 1),
                (row_index, col_index + 1),
            ):
                if neighbor in changed_cells:
                    changed_cells.discard(neighbor)
                    pending.append(neighbor)
                    region_cells.append(neighbor)
        row_indices = [row_index for row_index, _ in region_cells]
        col_indices = [col_index for _, col_index in region_cells]
        regions.append(
            {
                "bbox": [min(row_indices), min(col_indices), max(row_indices), max(col_indices)],
                "pixel_count": len(region_cells),
            }
        )
    regions.sort(key=lambda region: (-region["pixel_count"], region["bbox"]))

    return regions[:limit], len(regions)


def frame_from_payload(payload: Any) -> Frame | None:
    if not isinstance(payload, dict):
        return None
    try:
        step = max(0, int(payload.get("step", 0) or 0))
    except (TypeError, ValueError):
        step = 0
    try:
        level = max(1, int(payload.get("level", 1) or 1))
    except (TypeError, ValueError):
        level = 1
    try:
        frame_count = max(1, int(payload.get("frame_count", 1) or 1))
    except (TypeError, ValueError):
        frame_count = 1
    raw_animation = payload.get("animation")
    animation = (
        tuple(normalize_grid(raw_grid) for raw_grid in raw_animation)
        if isinstance(raw_animation, list)
        else ()
    )

    return Frame(
        grid=normalize_grid(payload.get("grid")),
        step=step,
        level=level,
        animation=animation,
        frame_count=frame_count,
    )


def frame_to_payload(frame: Frame | None, *, include_animation: bool = True) -> dict[str, Any] | None:
    if frame is None:
        return None
    payload: dict[str, Any] = {
        "grid": [list(row) for row in frame.grid],
        "step": frame.step,
        "level": frame.level,
    }
    if include_animation and frame.animation:
        payload["frame_count"] = frame.frame_count
        payload["animation"] = [[list(row) for row in grid] for grid in frame.animation]

    return payload


def history_entry_from_payload(payload: Any) -> HistoryEntry | None:
    if not isinstance(payload, dict):
        return None
    frame = frame_from_payload(payload.get("frame"))
    if frame is None:
        return None
    raw_result = payload.get("result")

    return HistoryEntry(
        action=str(payload.get("action", "")).strip(),
        frame=frame,
        result=dict(raw_result) if isinstance(raw_result, dict) else {},
        plan=str(payload.get("plan", "") or ""),
    )


def history_entry_to_payload(entry: HistoryEntry, *, include_animation: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": entry.action,
        "frame": frame_to_payload(entry.frame, include_animation=include_animation),
    }
    if entry.result:
        payload["result"] = dict(entry.result)
    if entry.plan:
        payload["plan"] = entry.plan

    return payload


def game_over_from_payload(payload: Any) -> GameOver | None:
    if not isinstance(payload, dict):
        return None
    frame = frame_from_payload(payload.get("frame"))
    raw_actions = payload.get("actions")
    actions = tuple(str(item) for item in raw_actions) if isinstance(raw_actions, list) else ()

    return GameOver(actions=actions, frame=frame)


def game_over_to_payload(game_over: GameOver, *, include_frame: bool = True) -> dict[str, Any]:
    return {
        "actions": list(game_over.actions),
        "frame": frame_to_payload(game_over.frame) if include_frame else None,
    }


def load_runtime_state(path: Path) -> RuntimeState:
    started_at = time.perf_counter()
    if not path.exists():
        return RuntimeState(current_frame=None, history=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_frame = frame_from_payload(payload.get("current_frame"))
    history_entries = [
        entry
        for raw_entry in payload.get("history", [])
        for entry in [history_entry_from_payload(raw_entry)]
        if entry is not None
    ]
    game_overs = tuple(
        game_over
        for raw_game_over in payload.get("game_overs", [])
        for game_over in [game_over_from_payload(raw_game_over)]
        if game_over is not None
    )
    try:
        actions_this_life = max(0, int(payload.get("actions_this_life", 0) or 0))
    except (TypeError, ValueError):
        actions_this_life = 0
    runtime_state_io_stats(path).read.record(time.perf_counter() - started_at)

    return RuntimeState(
        current_frame=current_frame,
        history=history_entries,
        game_overs=game_overs,
        actions_this_life=actions_this_life,
    )


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
    game_overs: list[GameOver],
    actions_this_life: int,
) -> None:
    """Persist the state the Python tool reads. Only the newest
    `ANIMATION_HISTORY_DEPTH` history entries keep their animation frames, and only the
    newest game over of a level keeps its board at all, which bounds the file the harness
    rewrites after every action against a level the agent keeps dying on. An older entry
    drops its `frame_count` with its frames, so a reloaded frame never claims frames it no
    longer has. The newest game over keeps its frames however old the history entry it came
    from is, because the animation of the death is what that entry loses first and the agent
    may still be reading it many actions later."""
    started_at = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    first_animated_index = len(history) - ANIMATION_HISTORY_DEPTH
    newest_game_over_index = len(game_overs) - 1
    payload = {
        "current_frame": frame_to_payload(current_frame),
        "history": [
            history_entry_to_payload(entry, include_animation=index >= first_animated_index)
            for index, entry in enumerate(history)
        ],
        "game_overs": [
            game_over_to_payload(game_over, include_frame=index == newest_game_over_index)
            for index, game_over in enumerate(game_overs)
        ],
        "actions_this_life": int(actions_this_life),
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(path)
    runtime_state_io_stats(path).write.record(time.perf_counter() - started_at)


def _frame_as_stored(frame: Frame, *, include_animation: bool) -> Frame:
    """`frame` as `frame_to_payload` followed by `frame_from_payload` returns it. A frame
    stored without animation, or with an empty one, comes back with no animation and a
    `frame_count` of 1."""
    if include_animation and frame.animation:
        return frame
    if not frame.animation and frame.frame_count == 1:
        return frame

    return replace(frame, animation=(), frame_count=1)


def build_runtime_state(
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
    game_overs: list[GameOver],
    actions_this_life: int,
) -> RuntimeState:
    """The state `write_runtime_state` would persist, as `load_runtime_state` would return it,
    built in memory without the file. History entries older than the newest
    `ANIMATION_HISTORY_DEPTH` lose their animation and every game over but the newest loses
    its board, exactly as the file drops them, so the Python tool sees the same values either
    way. Entries and frames the file would keep whole are shared, not copied."""
    first_animated_index = len(history) - ANIMATION_HISTORY_DEPTH
    newest_game_over_index = len(game_overs) - 1
    stored_history = [
        entry
        if stored_frame is entry.frame
        else replace(entry, frame=stored_frame)
        for index, entry in enumerate(history)
        for stored_frame in [_frame_as_stored(entry.frame, include_animation=index >= first_animated_index)]
    ]
    stored_game_overs = tuple(
        GameOver(
            actions=tuple(game_over.actions),
            frame=(
                _frame_as_stored(game_over.frame, include_animation=True)
                if index == newest_game_over_index and game_over.frame is not None
                else None
            ),
        )
        for index, game_over in enumerate(game_overs)
    )

    return RuntimeState(
        current_frame=(
            _frame_as_stored(current_frame, include_animation=True) if current_frame is not None else None
        ),
        history=stored_history,
        game_overs=stored_game_overs,
        actions_this_life=max(0, int(actions_this_life)),
    )
