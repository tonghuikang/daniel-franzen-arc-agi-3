"""Rendering of runtime-state grid data for the model.

The Python tool sandbox receives the current frame and the history as JSON
payloads and rebuilds its own read-only views from them, so the raw `Frame`
objects never cross into it. These helpers build those payloads and the text
description of changed areas that goes into the prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from inference.agent.python_tool_sandbox import EncodedJson
from inference.agent.runtime_state import Frame, GameOver, HistoryEntry


@dataclass(frozen=True)
class _AsciiFrameView:
    ascii: str
    step: int
    level: int
    shape: tuple[int, int]

    def __str__(self) -> str:
        rows, cols = self.shape
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

    __repr__ = __str__


def _to_ascii_frame_view(frame: Frame | None) -> _AsciiFrameView | None:
    if frame is None:
        return None
    return _AsciiFrameView(
        ascii=frame.ascii,
        step=frame.step,
        level=frame.level,
        shape=frame.shape,
    )


def ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:
    """Send ordinary frames as ASCII, with numeric grids decoded lazily in the sandbox.
    Animation frames keep their numeric grids and render ASCII only when read."""
    view = _to_ascii_frame_view(frame)
    if view is None:
        return None
    payload: dict[str, Any] = {
        "ascii": view.ascii,
        "step": view.step,
        "level": view.level,
        "shape": [int(view.shape[0]), int(view.shape[1])],
    }
    if frame.frame_count > 1:
        payload["frame_count"] = frame.frame_count
    if frame.animation:
        payload["animation"] = [
            {
                "step": view.step,
                "level": view.level,
                "shape": [len(grid), max((len(row) for row in grid), default=0)],
                "grid": [list(row) for row in grid],
            }
            for grid in frame.animation
        ]

    return payload


def ascii_history_entry_view_payload(entry: HistoryEntry) -> dict[str, Any]:
    """The sandbox's view of one history entry."""
    entry_payload: dict[str, Any] = {"action": entry.action, "frame": ascii_frame_view_payload(entry.frame)}
    if entry.result:
        entry_payload["result"] = dict(entry.result)
    if entry.plan:
        entry_payload["plan"] = entry.plan

    return entry_payload


class HistoryViewEncoder:
    """The sandbox's view of the history as JSON text, kept between calls of one game.

    Each entry is encoded once and its text reused while the entry at that position stays
    equal, so a call encodes only the entries appended since the previous call and the ones
    that just lost their animation, instead of rendering and encoding the whole history. An
    entry is compared by identity first, and by value only when the object changed, which
    stays cheap because a trimmed entry shares its grids with the one it replaces. The
    identity check relies on an entry never changing once it is in the history, `result`
    included, which holds because the solver builds each entry whole when it appends it."""

    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []
        self._fragments: list[str] = []

    def encode(self, history_entries: list[HistoryEntry]) -> EncodedJson:
        del self._entries[len(history_entries):]
        del self._fragments[len(history_entries):]
        for index, entry in enumerate(history_entries):
            if index < len(self._entries):
                cached_entry = self._entries[index]
                if cached_entry is entry or cached_entry == entry:
                    continue
                self._entries[index] = entry
                self._fragments[index] = self._encode_entry(entry)
            else:
                self._entries.append(entry)
                self._fragments.append(self._encode_entry(entry))

        return EncodedJson("[" + ",".join(self._fragments) + "]")

    @staticmethod
    def _encode_entry(entry: HistoryEntry) -> str:
        return json.dumps(ascii_history_entry_view_payload(entry), ensure_ascii=False)


def ascii_game_over_view_payload(game_overs: tuple[GameOver, ...]) -> list[dict[str, Any]]:
    """The sandbox's view of the game overs on the current level, oldest first. Only the
    newest carries the board the game over was declared on, so its animation travels with
    it while the payload stays the same size however often the agent dies on the level."""
    return [
        {
            "actions": list(game_over.actions),
            "frame": ascii_frame_view_payload(game_over.frame),
        }
        for game_over in game_overs
    ]


def format_changed_regions(regions: Any) -> str:
    if not isinstance(regions, list):
        return ""
    rendered: list[str] = []
    for region in regions:
        bbox = region.get("bbox") if isinstance(region, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        row_min, col_min, row_max, col_max = bbox
        pixel_count = region.get("pixel_count")
        cell_label = "cell" if pixel_count == 1 else "cells"
        rendered.append(f"rows {row_min}-{row_max} cols {col_min}-{col_max} ({pixel_count} {cell_label})")

    return "; ".join(rendered)
