"""What an action's animation showed.

The engine may answer one action with several frames. The last of them is the board
the game continues from; the ones before it are the animation a player would have
watched. This module keeps a bounded sample of those frames for the Python tool and
classifies that same sample for the action result, so the summary never describes a
frame the agent cannot read. An action that ended a level is summarized as nothing at all,
because the board it settled at belongs to the next level and the frames on the way there
describe a board that is gone.

Summary types, in the order they are tested:

- ``blink``: the kept frames alternate between two boards, one of them the settled one.
- ``tween``: one sprite (matched by its segmentation ``hash``) slides in a straight line
  from where it was first seen to its position on the settled board.
- ``path``: one sprite moves through positions that do not form such a line, or is gone
  from the settled board.
- ``other``: anything else, which ``bbox`` locates and nothing here names.

Every summary carries ``frame_count`` and ``bbox`` whatever its type, so the count of frames
and the region they touched never depend on the classification landing. Every field is read
off the kept frames, off the boards before and after the action, or off both, and the agent
holds all three. ``frame_count`` counts the kept frames, which are the same frames
``current_frame.animation_frames`` holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import _object_hash

Grid = tuple[tuple[int, ...], ...]
Position = tuple[int, int]

ANIMATION_FRAME_CAP = 16
_MIN_TRACK_POINTS = 3
PATH_POINT_LIMIT = 8

_ORTHOGONAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Animation:
    """An action's animation as the agent gets it: the frames the runtime state keeps,
    the last of them the board the action settled at, and the summary for the action
    result, `None` when the action showed nothing beyond that board."""

    frames: tuple[Grid, ...]
    summary: dict[str, Any] | None


@dataclass(frozen=True)
class _Component:
    """One 4-connected same-color object of a frame, hashed and colored exactly like the
    segmentation nodes the agent reads, positioned by its bounding box's top-left."""

    hash: str
    color: str
    pixel_count: int
    position: Position


@dataclass(frozen=True)
class _Track:
    """One object followed through the animation: its component in the board before the
    action (when present there), in every kept transient frame, and in the board the
    action settled at (when present there)."""

    component: _Component
    before_position: Position | None
    transient_positions: tuple[Position, ...]
    after_position: Position | None


def describe_animation(
    engine_frames: Sequence[Grid],
    before_grid: Grid,
    *,
    level_completed: bool,
) -> Animation:
    """Reduce the frames an action returned to the ones the runtime state keeps, and
    describe those same frames for the action result.

    `before_grid` is the board the action started from, which the agent still holds as
    `previous_frame`, so the summary never rests on anything the agent cannot check.

    An action that ended a level keeps its frames and gets no summary. The engine reports the
    level change itself, and the board the action settled at is the next level's, so a reading
    of the frames would name objects on a board the agent no longer has.
    """
    kept = _sample_frames_evenly(_collapse_held_frames(engine_frames), ANIMATION_FRAME_CAP)
    if len(kept) < 2:
        return Animation(frames=(), summary=None)
    if level_completed:
        return Animation(frames=kept, summary=None)

    return Animation(frames=kept, summary=_summarize(kept, before_grid))


def _collapse_held_frames(engine_frames: Sequence[Grid]) -> tuple[Grid, ...]:
    """The frames with back-to-back repeats collapsed, since a frame repeated across ticks
    is the engine holding one image rather than showing a new one. Order is preserved, so
    the board the action settled at stays last.

    This is the only pass a short animation gets, because `_sample_frames_evenly` returns
    anything already under the cap untouched. It also decides what counts as an animation
    at all, a held image collapsing to the single frame the caller reads as none. Frames
    equal to the settled board are kept wherever they sit between other frames, because
    that alternation is what a blink is.
    """
    kept: list[Grid] = []
    for grid in engine_frames:
        if not kept or grid != kept[-1]:
            kept.append(grid)

    return tuple(kept)


def _sample_frames_evenly(frames: tuple[Grid, ...], limit: int) -> tuple[Grid, ...]:
    """At most `limit` frames at equal index spacing, keeping the first and the last, and never
    two equal frames in a row.

    Collapsing has already dropped the repeats that sit next to each other in time; the ones
    left are the repeats thinning creates, an animation alternating between two looks aliasing
    away to one still board whenever the stride is even. So a target landing on the frame just
    taken walks forward to the next one that differs. The settled board is kept whatever it
    looks like and the frame before it gives way instead, since the sample is what every
    summary field is measured on and it has to hold the change.
    """
    if len(frames) <= limit:
        return frames
    last_index = len(frames) - 1
    picked = [0]
    for position in range(1, limit - 1):
        target = round(position * last_index / (limit - 1))
        index = _find_changed_index(frames, max(target, picked[-1] + 1), last_index, frames[picked[-1]])
        if index is not None:
            picked.append(index)
    if len(picked) > 1 and frames[last_index] == frames[picked[-1]]:
        picked.pop()
    picked.append(last_index)

    return tuple(frames[index] for index in picked)


def _find_changed_index(
    frames: tuple[Grid, ...],
    start_index: int,
    stop_index: int,
    previous_grid: Grid,
) -> int | None:
    """The first index in `[start_index, stop_index)` whose frame differs from `previous_grid`,
    or `None` when the sample would have to run into the settled board to find one."""
    for index in range(start_index, stop_index):
        if frames[index] != previous_grid:
            return index

    return None


def _summarize(kept: tuple[Grid, ...], before_grid: Grid) -> dict[str, Any]:
    """Classify the animation from the frames the agent gets, the last of them the board
    the action settled at and the ones before it its transient frames.

    `frame_count` and `bbox` are on every summary, so the count of frames and the region
    they touched are there whatever the type turns out to be."""
    settled_grid = kept[-1]
    transient = kept[:-1]
    summary: dict[str, Any] = {
        "frame_count": len(kept),
        "bbox": _compute_changed_bbox(transient, settled_grid),
    }
    blink = _detect_blink(kept)
    if blink is not None:
        summary.update(blink)

        return summary
    sprite = _track_moving_sprite(before_grid, transient, settled_grid)
    if sprite is not None:
        summary.update(_describe_sprite(sprite))

        return summary
    summary["type"] = "other"

    return summary


def _detect_blink(kept: tuple[Grid, ...]) -> dict[str, Any] | None:
    """A blink is the kept frames alternating between the settled board and one other board.

    They are already free of back-to-back repeats, so two distinct boards over three or more
    frames can only alternate. `cycles` counts the kept frames showing the other board, which
    is every cycle the agent can see and fewer than the engine played when the animation was
    thinned to the cap.
    """
    settled_grid = kept[-1]
    distinct_grids = set(kept)
    if len(kept) < 3 or len(distinct_grids) != 2:
        return None
    other_grid = next(grid for grid in distinct_grids if grid != settled_grid)

    return {
        "type": "blink",
        "cycles": sum(1 for grid in kept if grid == other_grid),
    }


def _compute_changed_bbox(grids: Sequence[Grid], reference_grid: Grid) -> list[int] | None:
    """Bounding box of every cell that differs from `reference_grid` in any of `grids`.

    Neither collapsing nor thinning leaves two equal frames next to each other, so for frames
    of one shape there is always a box. `None` is left for frames that differ only outside the
    shape they share, which the engine does not produce and the comparison cannot see.
    """
    rows: list[int] = []
    cols: list[int] = []
    for grid in grids:
        for row_index, (row, reference_row) in enumerate(zip(grid, reference_grid)):
            for col_index, (cell, reference_cell) in enumerate(zip(row, reference_row)):
                if cell != reference_cell:
                    rows.append(row_index)
                    cols.append(col_index)
    if not rows:
        return None

    return [min(rows), min(cols), max(rows), max(cols)]


def _track_moving_sprite(
    before_grid: Grid,
    transient: tuple[Grid, ...],
    settled_grid: Grid,
) -> list[_Track] | None:
    """Follow the one sprite the animation moves, as the tracks of its parts.

    Components present at the same place before and after the action are static scenery and
    are ignored in every frame. Among the rest, a moving object is a hash that appears exactly
    once in every kept transient frame and passes through at least `_MIN_TRACK_POINTS` distinct
    positions, so a single jump between two places does not count as motion. Components that
    move in lockstep are taken for parts of one sprite, whether or not they touch. More than
    one sprite, or none, is not a single track.
    """
    before_components = _find_components(before_grid)
    after_components = _find_components(settled_grid)
    static = {(component.hash, component.position) for component in before_components} & {
        (component.hash, component.position) for component in after_components
    }
    candidates = _find_single_components_per_frame(transient, static)
    tracks = [
        _build_track(transient_components, before_components, after_components, static)
        for transient_components in candidates.values()
    ]
    moving_tracks = [track for track in tracks if len(_collect_track_points(track)) >= _MIN_TRACK_POINTS]
    sprites = _group_lockstep_tracks(moving_tracks)
    if len(sprites) != 1:
        return None

    return sprites[0]


def _find_components(grid: Grid) -> list[_Component]:
    """4-connected same-color components of `grid`, hashed exactly like the segmentation
    nodes the agent reads, positioned by their bounding box's top-left."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    visited = [[False] * width for _ in range(height)]
    components: list[_Component] = []
    for start_row in range(height):
        for start_col in range(width):
            if visited[start_row][start_col]:
                continue
            value = grid[start_row][start_col]
            cells = _flood_fill(grid, visited, start_row, start_col, value)
            color = ARC_COLOR_CHARS[max(0, min(15, int(value)))]
            components.append(
                _Component(
                    hash=_object_hash(cells, color),
                    color=color,
                    pixel_count=len(cells),
                    position=(min(row for row, _ in cells), min(col for _, col in cells)),
                )
            )

    return components


def _flood_fill(
    grid: Grid,
    visited: list[list[bool]],
    start_row: int,
    start_col: int,
    value: int,
) -> set[Position]:
    """The cells of the 4-connected same-value region containing `(start_row, start_col)`,
    marking each of them visited on the way."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    cells: set[Position] = set()
    pending = [(start_row, start_col)]
    visited[start_row][start_col] = True
    while pending:
        row, col = pending.pop()
        cells.add((row, col))
        for row_offset, col_offset in _ORTHOGONAL_OFFSETS:
            next_row, next_col = row + row_offset, col + col_offset
            if 0 <= next_row < height and 0 <= next_col < width and not visited[next_row][next_col] and grid[next_row][next_col] == value:
                visited[next_row][next_col] = True
                pending.append((next_row, next_col))

    return cells


def _find_single_components_per_frame(
    transient: tuple[Grid, ...],
    static: set[tuple[str, Position]],
) -> dict[str, tuple[_Component, ...]]:
    """Hashes that occur exactly once outside the static scenery in every kept transient
    frame, mapped to that component in each frame, in frame order."""
    per_frame: list[dict[str, list[_Component]]] = []
    for grid in transient:
        dynamic: dict[str, list[_Component]] = {}
        for component in _find_components(grid):
            if (component.hash, component.position) in static:
                continue
            dynamic.setdefault(component.hash, []).append(component)
        per_frame.append(dynamic)
    shared_hashes = set.intersection(*[set(dynamic) for dynamic in per_frame])

    return {
        hash_value: tuple(dynamic[hash_value][0] for dynamic in per_frame)
        for hash_value in sorted(shared_hashes)
        if all(len(dynamic[hash_value]) == 1 for dynamic in per_frame)
    }


def _build_track(
    transient_components: tuple[_Component, ...],
    before_components: list[_Component],
    after_components: list[_Component],
    static: set[tuple[str, Position]],
) -> _Track:
    """One hash's track, its ends taken from the boards before and after the action.

    Either board can hold several copies of one shape, and the copy that belongs to this track
    is the one nearest the animation position beside it, a board being one engine tick from the
    frame next to it. No copy at all means the object was not there, which is the track
    genuinely having no such end.
    """
    hash_value = transient_components[0].hash
    before_copies = _find_hash_copies(before_components, hash_value, static)
    after_copies = _find_hash_copies(after_components, hash_value, static)

    return _Track(
        component=(after_copies or before_copies or list(transient_components))[0],
        before_position=_pick_nearest_position(before_copies, transient_components[0].position),
        transient_positions=tuple(component.position for component in transient_components),
        after_position=_pick_nearest_position(after_copies, transient_components[-1].position),
    )


def _find_hash_copies(
    components: list[_Component],
    hash_value: str,
    static: set[tuple[str, Position]],
) -> list[_Component]:
    """The components carrying `hash_value`, the ones that sat still through the action passed
    over as scenery unless every copy did, which is an object that ended where it began rather
    than a piece of the background."""
    copies = [component for component in components if component.hash == hash_value]
    moved = [component for component in copies if (component.hash, component.position) not in static]

    return moved or copies


def _pick_nearest_position(matches: list[_Component], anchor: Position) -> Position | None:
    """The position of the match nearest `anchor`, `None` when there is no match or when two of
    them are equally near and picking one would be a guess between copies."""
    if not matches:
        return None
    ranked = sorted(
        (abs(match.position[0] - anchor[0]) + abs(match.position[1] - anchor[1]), match.position)
        for match in matches
    )
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None

    return ranked[0][1]


def _collect_track_points(track: _Track) -> list[Position]:
    """The track's positions in time order, without back-to-back repeats."""
    raw_points = [
        position
        for position in (track.before_position, *track.transient_positions, track.after_position)
        if position is not None
    ]
    points: list[Position] = []
    for position in raw_points:
        if not points or position != points[-1]:
            points.append(position)

    return points


def _group_lockstep_tracks(tracks: list[_Track]) -> list[list[_Track]]:
    """Group tracks whose transient positions share the same frame-to-frame displacement.

    A single transient frame carries no frame-to-frame displacement, so nothing there can be
    shown to move in lockstep and every track stands alone. Without that, objects on unrelated
    paths would all share the one empty displacement and be read as parts of one sprite.
    """
    groups: dict[tuple[tuple[Position, ...], int], list[_Track]] = {}
    for index, track in enumerate(tracks):
        origin = track.transient_positions[0]
        displacement = tuple((row - origin[0], col - origin[1]) for row, col in track.transient_positions)
        stands_alone = len(displacement) < 2
        groups.setdefault((displacement, index if stands_alone else -1), []).append(track)

    return list(groups.values())


def _describe_sprite(parts: list[_Track]) -> dict[str, Any]:
    """Describe a sprite by its largest part; `parts` is reported when there were several,
    since the description then covers only that one of them.

    The identity keys are the segmentation node's own `hash`, `color` and `pixel_count`, so the
    sprite can be matched against `segmentation['nodes']` without translating a second name."""
    track = max(parts, key=lambda part: part.component.pixel_count)
    points = _collect_track_points(track)
    identity: dict[str, Any] = {
        "hash": track.component.hash,
        "color": track.component.color,
        "pixel_count": track.component.pixel_count,
    }
    if len(parts) > 1:
        identity["parts"] = len(parts)
    if track.after_position is not None and _is_straight_monotonic(points):
        return {"type": "tween", **identity, "from": list(points[0]), "to": list(points[-1])}

    return {
        "type": "path",
        **identity,
        "points": [list(point) for point in _sample_positions_evenly(points, PATH_POINT_LIMIT)],
    }


def _sample_positions_evenly(positions: list[Position], limit: int) -> list[Position]:
    """At most `limit` positions at equal index spacing, always keeping the first and the last."""
    if len(positions) <= limit:
        return list(positions)
    last_index = len(positions) - 1
    indices = sorted({round(position * last_index / (limit - 1)) for position in range(limit)})

    return [positions[index] for index in indices]


def _is_straight_monotonic(points: list[Position]) -> bool:
    """Whether every point lies on the segment from the first to the last point, visited in
    order without turning back."""
    start, end = points[0], points[-1]
    direction = (end[0] - start[0], end[1] - start[1])
    if direction == (0, 0):
        return False
    previous_progress = 0
    for row, col in points[1:]:
        offset = (row - start[0], col - start[1])
        if direction[0] * offset[1] != direction[1] * offset[0]:
            return False
        progress = direction[0] * offset[0] + direction[1] * offset[1]
        if progress < previous_progress or progress > direction[0] ** 2 + direction[1] ** 2:
            return False
        previous_progress = progress

    return True
