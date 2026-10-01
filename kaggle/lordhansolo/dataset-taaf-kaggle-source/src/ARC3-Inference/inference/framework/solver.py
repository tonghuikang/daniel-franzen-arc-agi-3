"""TAAF solver adapter for the existing tool-using harness."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import arcengine
import taaf.game
from taaf.solver import Solver

from inference.agent.action import (
    to_engine_action,
    to_model_action,
    to_model_actions,
)
from inference.agent.animation import describe_animation
from inference.agent.metrics_log import GameMetricsRecorder
from inference.agent.run_paths import GamePaths, metrics_dir
from inference.agent.runtime_state import (
    Frame,
    GameOver,
    HistoryEntry,
    RuntimeState,
    build_runtime_state,
    changed_pixel_count,
    changed_regions,
    pop_runtime_state_io_stats,
    total_pixel_count,
)
from inference.agent.tool_agent import ToolAgent, resolve_analyzer_model
from inference.agent.transcript_log import write_transcript_html
from inference.framework.kaggle import (
    DEFAULT_QWEN_MODEL_SOURCE,
    DEFAULT_SERVED_MODEL_NAME,
    DEFAULT_VLLM_DRAFT_VOCAB,
    DEFAULT_VLLM_DTYPE,
    DEFAULT_VLLM_ENABLE_PREFIX_CACHING,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_GDN_DECODE_KERNEL,
    DEFAULT_VLLM_INDEXER_KV_DTYPE,
    DEFAULT_VLLM_KV_CACHE_DTYPE,
    DEFAULT_VLLM_KV_CACHE_GIB,
    DEFAULT_VLLM_MAMBA_SSM_CACHE_DTYPE,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_VLLM_MAX_NUM_SEQS,
    DEFAULT_VLLM_MTP_TOKENS,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_PREFIX_CACHE_RETENTION_INTERVAL,
    DEFAULT_VLLM_PREFIX_MATCH_UNIT,
    DEFAULT_VLLM_PROFILE,
    DEFAULT_VLLM_QUANTIZATION,
    DEFAULT_VLLM_RUNTIME_DATASET_SOURCE,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    DuckKaggleVllmConfig,
    duck_kaggle_dataset_sources,
    duck_kaggle_model_sources,
    duck_kaggle_setup_command,
    duck_kaggle_teardown_command,
)
from inference.framework.vllm_metrics import (
    VLLM_METRICS_FILENAME,
    VllmMetricsScraper,
    VllmMetricsTarget,
    metrics_url_from_base_url,
)
from inference.utils.env import get_env_bool, get_env_int, get_env_str
from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    reset_raw_events_sidecar,
)

log = logging.getLogger(__name__)

AnalyzerFactory = Callable[[taaf.game.Game, int], Any]

ANALYZER_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS = 120.0
SOFT_DEADLINE_GAME_MARGIN_SECONDS = 60.0
_LOCAL_SERVER_PROCESS_ENV_KEYS = (
    "LOCAL_ANALYZER_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_ANALYZER_BASE_URL",
    "OPENAI_BASE_URL",
    "LOCAL_ANALYZER_PROVIDER",
    "OPENAI_PROVIDER",
)


@dataclass
class _LocalServerRuntime:
    index: int
    repo_dir: Path
    api_key_file: Path
    env_overrides: dict[str, str]
    base_url: str
    api_key: str = ""


def _analyzer_reported_tokens(analyzer: Any) -> int:
    value = (
        getattr(analyzer, "generated_tokens", None)
        if hasattr(analyzer, "generated_tokens")
        else getattr(analyzer, "total_tokens", 0)
    )
    return max(0, int(value or 0))


def _analyzer_reported_uncached_input_tokens(analyzer: Any) -> int:
    return max(0, int(getattr(analyzer, "uncached_input_tokens", 0) or 0))


def _grid_from_frame_data(data: Any) -> tuple[tuple[int, ...], ...]:
    rows = data.tolist() if hasattr(data, "tolist") else data

    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _grid_from_state(state: taaf.game.GameState | None) -> tuple[tuple[int, ...], ...]:
    if state is None:
        return ()

    return _grid_from_frame_data(state.frame.data)


_TAAF_GREEN_FRAME_COLOR = 14


def _engine_frames_from_state(state: taaf.game.GameState) -> list[tuple[tuple[int, ...], ...]]:
    """Every frame the engine returned for the action, as grids, the board it settled at last.

    TAAF prepends a synthetic all-green frame when a level transition came back
    as a single frame (see ``Game.execute_action``). That frame is TAAF's, not the
    game's, so a level transition whose only animation frame is all green is
    treated as having none.
    """
    raw_frames = list(state.raw.frame)
    if state.just_won_level and len(raw_frames) == 2 and _is_uniform_grid(raw_frames[0], _TAAF_GREEN_FRAME_COLOR):
        raw_frames = raw_frames[1:]

    return [_grid_from_frame_data(frame) for frame in raw_frames]


def _is_uniform_grid(data: Any, color: int) -> bool:
    return all(int(cell) == color for row in _grid_from_frame_data(data) for cell in row)


def _level_number(game: taaf.game.Game) -> int:
    state = game.current_state
    completed = int(state.levels_completed)
    if state.won:
        return max(1, int(game.number_of_levels))
    return max(1, min(int(game.number_of_levels), completed + 1))


def _engine_action_names(game: taaf.game.Game) -> list[str]:
    names: list[str] = []
    for action_id in game.current_state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:
            continue
        if name == "RESET":  # RESET is solver-driven.
            continue
        if name not in names:
            names.append(name)
    return names


def _model_mouse_action_data(
    action_data: dict[str, Any] | None = None,
) -> dict[str, int]:
    data = action_data or {}
    return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}


def _format_action_display(
    action_name: str, action_data: dict[str, Any] | None = None
) -> str:
    if action_name == "ACTION6":
        data = _model_mouse_action_data(action_data)
        return f"MOUSE(row={data['row']}, col={data['col']})"
    return to_model_action(action_name)


def _is_engine_game_over(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.GAME_OVER


def _is_run_complete(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.WIN


@dataclass
class _HarnessGameSession:
    solver: "HarnessSolver"
    game: taaf.game.Game
    analyzer: Any
    game_index: int
    pass_index: int
    state_path: Path
    transcript_path: Path
    analysis_html_relpath: str
    stop_event: threading.Event
    viewer_data_path: Path
    started_at: float = field(default_factory=time.monotonic)
    history_entries: list[HistoryEntry] = field(default_factory=list)
    life_actions: list[str] = field(default_factory=list)
    level_game_overs: list[GameOver] = field(default_factory=list)
    viewer_events: list[dict[str, Any]] = field(default_factory=list)
    analysis_step: int = 0
    last_engine_action: str | None = None
    token_baseline: int = 0
    uncached_token_baseline: int = 0
    metrics: GameMetricsRecorder = field(init=False, repr=False)
    _viewer_events_flushed: int = field(default=0, init=False, repr=False)
    _strategy_audit_interval_seconds: float | None = field(init=False, repr=False)
    _strategy_audit_started_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.metrics = GameMetricsRecorder(
            self.state_path, started_at=self.started_at, enabled=self.diagnostics
        )
        initial_runtime_limit = self.effective_max_runtime_s()
        self._strategy_audit_interval_seconds = (
            initial_runtime_limit * 0.25 if initial_runtime_limit is not None and initial_runtime_limit > 0 else None
        )
        self._strategy_audit_started_at = self.started_at

    def _should_audit_strategy(self) -> bool:
        """Use the fixed initial budget, independent of actions, lives and model claims."""

        return (
            self._strategy_audit_interval_seconds is not None
            and time.monotonic() - self._strategy_audit_started_at >= self._strategy_audit_interval_seconds
        )

    @property
    def diagnostics(self) -> bool:
        """Whether this game writes its optional diagnostics (transcript, viewer
        payload, analysis HTML, metrics); off under TAAF's ``minimal_diagnostics``."""
        return not self.solver.minimal_diagnostics

    def current_frame(self) -> Frame:
        """The board the game stands at. Right after an action that is the frame
        the action appended to the history, so its animation travels with it;
        otherwise a fresh frame built from the engine state."""
        grid = _grid_from_state(self.game.current_state)
        latest_frame = self.history_entries[-1].frame if self.history_entries else None
        if latest_frame is not None and latest_frame.grid == grid and latest_frame.step == self.action_count:
            return latest_frame

        return Frame(
            grid=grid,
            step=self.action_count,
            level=_level_number(self.game),
        )

    def runtime_state(self) -> RuntimeState:
        """The state the analyzer and its Python tool read, built from this session in memory."""

        return build_runtime_state(
            current_frame=self.current_frame(),
            history=self.history_entries,
            game_overs=self.level_game_overs,
            actions_this_life=len(self.life_actions),
        )

    def record_life_action(
        self,
        action_display: str,
        frame: Frame,
        *,
        is_reset: bool,
        game_over: bool,
        level_cleared: bool,
    ) -> None:
        """Track the actions of the current life and close it when it ends.

        A life is the run of actions since the level was entered or since the previous game
        over on it, which is the span a level's own action budget follows. The harness RESET
        that follows a game over is not part of any life, and clearing a level drops the game
        overs with it, because they describe a board the game has left.

        A cleared level is settled before a game over, so an action that does both leaves
        nothing of the old level behind. Nothing in the engine rules that pair out, since
        `next_level` moves the score while `lose` moves the state, and a game over on the
        board the game just left is not one the next life can act on anyway.
        """
        if is_reset:
            return
        self.life_actions.append(action_display)
        if level_cleared:
            self.life_actions = []
            self.level_game_overs = []
            self._strategy_audit_started_at = time.monotonic()
        elif game_over:
            self.level_game_overs.append(GameOver(actions=tuple(self.life_actions), frame=frame))
            self.life_actions = []

    def seed_initial_history(self) -> None:
        if not self.history_entries:
            self.history_entries.append(
                HistoryEntry(action="", frame=self.current_frame())
            )

    @property
    def action_count(self) -> int:
        run = self.game.game_run
        return len(run.history) if run is not None else 0

    def effective_max_runtime_s(self) -> float | None:
        """Per-game budget, capped by the experiment soft deadline when the
        deadline would land first."""
        per_game_limit = self.solver.max_runtime_s_per_game
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is None:
            return per_game_limit
        elapsed = max(0.0, time.monotonic() - self.started_at)
        soft_limit = elapsed + max(
            0.0, soft_remaining - SOFT_DEADLINE_GAME_MARGIN_SECONDS
        )
        if per_game_limit is None:
            return soft_limit

        return min(per_game_limit, soft_limit)

    def runtime_limit_reached(self) -> bool:
        runtime_limit = self.effective_max_runtime_s()
        if runtime_limit is None:
            return False

        return (time.monotonic() - self.started_at) >= runtime_limit

    def get_time_remaining_seconds(self) -> int | None:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        runtime_limit = self.effective_max_runtime_s()
        if runtime_limit is None:

            return None

        return round(max(0.0, runtime_limit - elapsed))

    def request_timeout_seconds(self) -> float | None:
        """Timeout for one model call: the configured analyzer timeout capped by
        the per-game and soft-deadline budgets left right now. The analyzer calls
        this before every request, so later requests in a turn cannot overrun
        the deadline by the time the earlier ones took."""
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.get_time_remaining_seconds()
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        return max(0.1, min(candidates))

    def should_stop(self) -> bool:
        run = self.game.game_run
        if run is None or run.state != "playing":
            return True
        if self.stop_event.is_set():
            return True
        if _is_run_complete(self.game):
            return True
        if self.runtime_limit_reached():
            return True
        if (
            self.solver.max_actions_per_game is not None
            and self.action_count >= self.solver.max_actions_per_game
        ):
            return True
        return False

    def play(self) -> None:
        run = self.game.game_run
        assert run is not None, "TAAF starts games before invoking the solver."
        run.solver_analysis_html = self.analysis_html_relpath
        if self.diagnostics:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.touch(exist_ok=True)
        self.token_baseline = _analyzer_reported_tokens(self.analyzer)
        self.uncached_token_baseline = _analyzer_reported_uncached_input_tokens(self.analyzer)
        self.seed_initial_history()
        self._append_initial_viewer_event()
        self.write_viewer_payload()
        self.metrics.game_start(
            level=_level_number(self.game),
            action=self.action_count,
            number_of_levels=int(run.number_of_levels),
            runtime_limit_seconds=self.effective_max_runtime_s(),
            max_actions=self.solver.max_actions_per_game,
        )
        try:
            retry_analysis_step: int | None = None
            analysis_attempt = 0
            while not self.should_stop():
                if (
                    _is_engine_game_over(self.game)
                    and self.last_engine_action != "RESET"
                ):
                    self._execute_auto_reset()
                    continue

                if retry_analysis_step is None:
                    self.analysis_step += 1
                    analysis_step = self.analysis_step
                    analysis_attempt = 1
                else:
                    analysis_step = retry_analysis_step
                    analysis_attempt += 1

                transcript_before = self._read_transcript_bytes()
                strategy_audit_due = self._should_audit_strategy()
                analysis_started_at = time.monotonic()
                try:
                    result = self.analyzer.analyze(
                        self.state_path,
                        self.action_count,
                        valid_actions=_engine_action_names(self.game),
                        step_env=self.step_env,
                        transcript_path=self.transcript_path,
                        analysis_step=analysis_step,
                        analysis_attempt=analysis_attempt,
                        request_timeout_seconds=self.request_timeout_seconds,
                        should_stop=self.should_stop,
                        runtime_state_source=self.runtime_state,
                        strategy_audit_due=strategy_audit_due,
                    )
                finally:
                    transcript_delta = self._transcript_delta_since(transcript_before)
                    if transcript_delta.strip():
                        self._append_analysis_viewer_event(
                            analysis_step, transcript_delta
                        )
                        self.write_viewer_payload()
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    retry_analysis_step = analysis_step
                    self.metrics.analyzer_retry(
                        level=_level_number(self.game),
                        action=self.action_count,
                        analysis_step=analysis_step,
                        analysis_attempt=analysis_attempt,
                        backoff_seconds=ANALYZER_RETRY_BACKOFF_SECONDS,
                    )
                    if self.should_stop():
                        break
                    time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
                    continue

                if strategy_audit_due:
                    self._strategy_audit_started_at = max(self._strategy_audit_started_at, analysis_started_at)
                retry_analysis_step = None
                if getattr(result, "yielded_control", False):
                    retry_analysis_step = analysis_step
                    continue
                if not result.step_executed:
                    continue
        except Exception as exc:
            if run.final_score is None:
                run.solver_note = f"error: {type(exc).__name__}: {exc}"
                if run.state == "playing":
                    run.state = "crashed"
                self._finish_if_needed()
        finally:
            total_tokens = _analyzer_reported_tokens(self.analyzer)
            if run.solver_note is None:
                run.solver_note = f"tokens={total_tokens}"
            self._finish_if_needed()
            self._append_game_end_metrics()
            self._write_analysis_html()
            self.write_viewer_payload()

    def _game_stop_reason(self) -> str:
        if _is_run_complete(self.game):
            return "run_complete"
        if self.runtime_limit_reached():
            return "runtime_limit"
        if (
            self.solver.max_actions_per_game is not None
            and self.action_count >= self.solver.max_actions_per_game
        ):
            return "max_actions"
        if self.stop_event.is_set():
            return "cancelled"

        return "other"

    def _append_game_end_metrics(self) -> None:
        run = self.game.game_run
        if run is None:
            return
        self.metrics.game_end(
            level=_level_number(self.game),
            action=self.action_count,
            stop_reason=self._game_stop_reason(),
            state=run.state,
            final_score=run.final_score,
            levels_completed=int(run.levels_completed),
            number_of_levels=int(run.number_of_levels),
            actions_per_level=list(run.actions_per_level),
            base_actions_per_level=run.base_actions_per_level,
            analysis_steps=self.analysis_step,
            generated_tokens=_analyzer_reported_tokens(self.analyzer),
            uncached_input_tokens=_analyzer_reported_uncached_input_tokens(self.analyzer),
            solver_note=run.solver_note,
            runtime_state_io=pop_runtime_state_io_stats(self.state_path).to_payload(),
        )

    def _finish_if_needed(self) -> None:
        """Finish the TAAF run once, handing it the tokens the analyzer spent
        after the last executed action so TAAF's per-game totals match the
        analyzer's session counters instead of dropping the trailing turns."""
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            self.game.finish_game(
                generated_tokens=max(0, _analyzer_reported_tokens(self.analyzer) - self.token_baseline),
                uncached_input_tokens=max(
                    0, _analyzer_reported_uncached_input_tokens(self.analyzer) - self.uncached_token_baseline
                ),
            )

    def _write_analysis_html(self) -> None:
        if self.solver.job_dir is None or not self.diagnostics:
            return
        write_transcript_html(
            self.transcript_path,
            self.solver.job_dir / self.analysis_html_relpath,
            f"{self.game.game_run.game_id if self.game.game_run else self.game_index} analysis",
        )

    def _read_transcript_bytes(self) -> bytes:
        try:
            return self.transcript_path.read_bytes()
        except OSError:
            return b""

    def _transcript_delta_since(self, previous_transcript: bytes) -> str:
        try:
            current_size = self.transcript_path.stat().st_size
            previous_size = len(previous_transcript)
            with self.transcript_path.open("rb") as file:
                if current_size >= previous_size:
                    current_prefix = file.read(previous_size)
                    if current_prefix == previous_transcript:
                        return file.read().decode("utf-8", errors="replace").strip()
                    file.seek(0)
                return file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _base_viewer_event(self, frame: Frame) -> dict[str, Any]:
        run = self.game.game_run
        raw_state = self.game.current_state.raw.state
        return {
            "board": [list(row) for row in frame.grid],
            "board_ascii": frame.ascii,
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "level": frame.level,
            "run_status": run.state if run is not None else "playing",
        }

    def _append_initial_viewer_event(self) -> None:
        if self.viewer_events:
            return
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "initial",
                "title": "Initial State",
                "action_num": self.action_count,
                "analysis_step": None,
                "action_display": "RESET",
            }
        )

    def _append_analysis_viewer_event(
        self, analysis_step: int, transcript: str
    ) -> None:
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "analysis",
                "title": f"Analysis Step {analysis_step}",
                "action_num": self.action_count,
                "analysis_step": analysis_step,
                "transcript": transcript,
            }
        )

    def _append_action_viewer_event(
        self, payload: dict[str, Any], frame: Frame
    ) -> None:
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "action",
                "title": f"Action {int(payload.get('action_num') or self.action_count)}",
                "action_num": int(payload.get("action_num") or self.action_count),
                "analysis_step": self.analysis_step,
                "action_name": payload.get("action_name"),
                "action_display": payload.get("action_display"),
                "changed_pixels": payload.get("changed_pixels"),
                "done": payload.get("done"),
                "level_completed": payload.get("level_completed"),
                "game_over": payload.get("game_over"),
                "run_complete": payload.get("run_complete"),
                "batch_index": payload.get("batch_index"),
                "batch_size": payload.get("batch_size"),
            }
        )

    def write_viewer_payload(self) -> None:
        if self.solver.job_dir is None or not self.diagnostics:
            return
        self.viewer_data_path.parent.mkdir(parents=True, exist_ok=True)
        run = self.game.game_run
        last_event_source = next(
            (
                event
                for event in reversed(self.viewer_events)
                if event.get("type") == "action"
            ),
            self.viewer_events[-1] if self.viewer_events else {},
        )
        last_event = dict(last_event_source)
        last_event.pop("board", None)
        last_event.pop("board_ascii", None)
        last_event.pop("transcript", None)
        payload = {
            "game_id": run.game_id if run is not None else str(self.game_index),
            "agent_name": self.solver.label,
            "status": run.state if run is not None else "playing",
            "pass_index": self.pass_index,
            "pass_label": str(self.pass_index),
            "eventCount": len(self.viewer_events),
            "lastEvent": last_event,
            "viewer_steps": [],
            "replay_url": self.analysis_html_relpath,
        }
        if run is not None:
            payload.update(
                {
                    "levels_completed": run.levels_completed,
                    "total_levels": run.number_of_levels,
                    "actions_per_level": list(run.actions_per_level),
                    "final_score": run.final_score,
                }
            )
        if self._viewer_events_flushed == 0:
            reset_raw_events_sidecar(self.viewer_data_path)
        append_raw_events_sidecar(
            self.viewer_data_path, self.viewer_events[self._viewer_events_flushed :]
        )
        self._viewer_events_flushed = len(self.viewer_events)
        tmp_path = self.viewer_data_path.with_suffix(
            f"{self.viewer_data_path.suffix}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.viewer_data_path)

    def _normalize_actions(
        self, arguments: dict[str, Any]
    ) -> tuple[list[arcengine.ActionInput] | None, str | None]:
        has_single = bool(str(arguments.get("action", "")).strip())
        has_batch = arguments.get("actions") is not None
        if has_single and has_batch:
            return None, "Use either `action` or `actions`, not both."

        if has_batch:
            raw_actions = arguments.get("actions")
            if not isinstance(raw_actions, list):
                return None, "`actions` must be a JSON array of action objects."
            if not raw_actions:
                return None, "`actions` must contain at least one action."
        else:
            if not has_single:
                return None, "step_env requires `action` or `actions`."
            raw_actions = [
                {
                    "action": arguments.get("action"),
                    "row": arguments.get("row"),
                    "col": arguments.get("col"),
                }
            ]

        actions: list[arcengine.ActionInput] = []
        for index, raw_action in enumerate(raw_actions, start=1):
            if not isinstance(raw_action, dict):
                return None, f"Action {index} must be a JSON object."
            action_name = to_engine_action(raw_action.get("action"))
            if not action_name:
                return (
                    None,
                    f"Unknown action at index {index}: {raw_action.get('action')!r}",
                )
            action_id = arcengine.GameAction.from_name(action_name)
            data: dict[str, Any] = {}
            if action_id == arcengine.GameAction.ACTION6:
                try:
                    row = max(0, min(63, int(raw_action["row"])))
                    column = max(0, min(63, int(raw_action["col"])))
                    data = {
                        "x": column,
                        "y": row,
                    }
                except (KeyError, TypeError, ValueError):
                    return (
                        None,
                        f"MOUSE action at index {index} requires integer row and col arguments.",
                    )
            actions.append(arcengine.ActionInput(id=action_id, data=data))
        return actions, None

    def _error_payload(self, message: str) -> dict[str, Any]:
        return {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "changed_pixels": 0,
            "total_pixels": total_pixel_count(_grid_from_state(self.game.current_state)),
            "largest_changed_regions": [],
            "changed_region_count": 0,
        }

    def _terminal_payload(
        self, requested_actions: list[arcengine.ActionInput]
    ) -> dict[str, Any]:
        raw_state = self.game.current_state.raw.state
        is_game_over = raw_state == arcengine.GameState.GAME_OVER
        is_win = raw_state == arcengine.GameState.WIN
        requested = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        stop_reason = (
            "run_complete" if is_win else "game_over" if is_game_over else "stopped"
        )
        return {
            "executed": False,
            "error": "No action was executed because the current game state is terminal or stopping.",
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "valid_actions": [],
            "changed_pixels": 0,
            "total_pixels": total_pixel_count(_grid_from_state(self.game.current_state)),
            "largest_changed_regions": [],
            "changed_region_count": 0,
            "done": is_win,
            "level_completed": False,
            "game_over": is_game_over,
            "run_complete": is_win,
            "batched": len(requested_actions) > 1,
            "requested_count": len(requested_actions),
            "executed_count": 0,
            "requested_actions": requested,
            "executed_actions": [],
            "stopped_early": True,
            "stop_reason": stop_reason,
        }

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        plan = str(arguments.get("plan", "") or "")
        requested_actions, error = self._normalize_actions(arguments)
        if error is not None or requested_actions is None:
            return self._error_payload(error or "Could not parse action request.")
        if self.should_stop() or _is_engine_game_over(self.game):
            return self._terminal_payload(requested_actions)

        batch_previous_grid = _grid_from_state(self.game.current_state)
        executed_payloads: list[dict[str, Any]] = []
        stop_reason: str | None = None
        batch_size = len(requested_actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]

        for batch_index, action in enumerate(requested_actions, start=1):
            if self.should_stop():
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.current_state.available_actions:
                message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                if executed_payloads:
                    stop_reason = "invalid_action"
                    break
                return self._error_payload(message)

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    flush_viewer_payload=False,
                    plan=plan,
                )
            except Exception as exc:
                if executed_payloads:
                    stop_reason = "action_error"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")
            executed_payloads.append(payload)

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break

        if not executed_payloads:
            return self._error_payload("No action was executed.")

        final_payload = dict(executed_payloads[-1])
        final_payload["batched"] = batch_size > 1
        final_payload["requested_count"] = batch_size
        final_payload["executed_count"] = len(executed_payloads)
        final_payload["requested_actions"] = requested_displays
        final_payload["executed_actions"] = [
            str(item.get("action_display") or item.get("action_name") or "")
            for item in executed_payloads
        ]
        final_payload["per_step_changed_pixels"] = [
            int(item.get("changed_pixels") or 0) for item in executed_payloads
        ]
        final_payload["per_step_frames"] = [
            int(item.get("frame_count") or 1) for item in executed_payloads
        ]
        final_grid = _grid_from_state(self.game.current_state)
        final_payload["changed_pixels"] = changed_pixel_count(
            batch_previous_grid, final_grid
        )
        final_payload["total_pixels"] = total_pixel_count(final_grid)
        final_regions, final_region_count = changed_regions(batch_previous_grid, final_grid)
        final_payload["largest_changed_regions"] = final_regions
        final_payload["changed_region_count"] = final_region_count
        final_payload["stopped_early"] = len(executed_payloads) < batch_size
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
        self.write_viewer_payload()
        return final_payload

    def _append_level_cleared_metrics(self, cleared_level: int) -> None:
        run = self.game.game_run
        actions_per_level = list(run.actions_per_level) if run is not None else []
        level_actions = actions_per_level[cleared_level - 1] if len(actions_per_level) >= cleared_level else None
        self.metrics.level_cleared(cleared_level, action=self.action_count, level_actions=level_actions)

    def _execute_auto_reset(self) -> None:
        """Issue the RESET that follows a GAME_OVER, retrying a failed engine call until it succeeds
        or the game has to stop.

        In a competition rerun the RESET is an HTTP call to Kaggle's gateway. ``arc_agi`` swallows
        every transport or HTTP failure, 4xx included, and GameAPI turns it into a RuntimeError, so
        a transient failure cannot be told apart from a permanent one. A RuntimeError the engine
        never recorded is therefore retried every ``ANALYZER_RETRY_BACKOFF_SECONDS``, and the game
        stops quietly when its own budget runs out mid-retry. Retrying is never worse than giving
        up. A crash scores the current level and every later one as zero, while a game that
        resumes can still clear them. Holding the slot costs nothing, since every game gets its
        full budget either way. TAAF validation errors are deterministic and raise at once.

        Retries can cost actions. When the gateway applied a RESET and only its reply was lost,
        the competition server leaves the fresh level as it is on the next RESET but still counts
        that RESET as a reset and an action on the level, in the same play. Every attempt the
        server applies without its reply arriving adds one more such action on the current level
        only, so their number grows with the length of the outage. The auto_reset metric is
        recorded only for the RESET the engine accepted."""
        level = _level_number(self.game)
        action_count = self.action_count
        action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        while True:
            try:
                self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
            except RuntimeError as exc:
                if self.action_count != action_count:
                    raise
                if self.should_stop():
                    return
                log.warning("auto reset failed at action %d, retrying: %s", action_count, exc)
                time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
            else:
                self.metrics.auto_reset(level=level, action=action_count, reason="game_over")

                return

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
        plan: str = "",
    ) -> dict[str, Any]:
        previous_grid = _grid_from_state(self.game.current_state)
        previous_completed = int(self.game.current_state.levels_completed)
        uncached_input_tokens = 0
        if generated_tokens is None:
            current_tokens = _analyzer_reported_tokens(self.analyzer)
            generated_tokens = max(0, current_tokens - self.token_baseline)
            self.token_baseline = current_tokens
            current_uncached_tokens = _analyzer_reported_uncached_input_tokens(self.analyzer)
            uncached_input_tokens = max(0, current_uncached_tokens - self.uncached_token_baseline)
            self.uncached_token_baseline = current_uncached_tokens

        new_state = self.game.execute_action(
            action, generated_tokens=generated_tokens, uncached_input_tokens=uncached_input_tokens
        )
        self.last_engine_action = action.id.name
        action_display = _format_action_display(action.id.name, dict(action.data))
        current_grid = _grid_from_state(new_state)
        engine_frames = _engine_frames_from_state(new_state)
        animation = describe_animation(
            engine_frames,
            previous_grid,
            level_completed=bool(new_state.just_won_level),
        )
        current_frame = Frame(
            grid=current_grid,
            step=self.action_count,
            level=_level_number(self.game),
            animation=animation.frames,
            frame_count=max(1, len(animation.frames)),
        )
        completed = int(new_state.levels_completed)
        raw_state = new_state.raw.state
        level_completed = bool(
            new_state.just_won_level and raw_state != arcengine.GameState.WIN
        )
        changed_pixels = changed_pixel_count(previous_grid, current_grid)
        self.history_entries.append(
            HistoryEntry(
                action=action_display,
                frame=current_frame,
                result={
                    "score": completed,
                    "changed_pixels": changed_pixels,
                    "level_completed": level_completed,
                    "game_over": raw_state == arcengine.GameState.GAME_OVER,
                },
                plan=plan,
            )
        )
        self.record_life_action(
            action_display,
            current_frame,
            is_reset=action.id == arcengine.GameAction.RESET,
            game_over=raw_state == arcengine.GameState.GAME_OVER,
            level_cleared=completed > previous_completed,
        )

        if completed > previous_completed:
            self._append_level_cleared_metrics(previous_completed + 1)
        regions, region_count = changed_regions(previous_grid, current_grid)
        payload = {
            "executed": True,
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": completed,
            "state": raw_state.name,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixel_count(current_grid),
            "largest_changed_regions": regions,
            "changed_region_count": region_count,
            "done": raw_state == arcengine.GameState.WIN,
            "level_completed": level_completed,
            "game_over": raw_state == arcengine.GameState.GAME_OVER,
            "run_complete": raw_state == arcengine.GameState.WIN,
            "action_name": action.id.name,
            "action_data": (
                _model_mouse_action_data(action.data)
                if action.id == arcengine.GameAction.ACTION6
                else dict(action.data)
            ),
            "action_display": action_display,
            "batch_index": batch_index,
            "batch_size": batch_size,
            "frame_count": current_frame.frame_count,
        }
        if animation.summary is not None:
            payload["animation"] = animation.summary
        self._append_action_viewer_event(payload, current_frame)
        if flush_viewer_payload:
            self.write_viewer_payload()
        return payload


@dataclass
class HarnessSolver(Solver):
    """Run the existing tool-using harness as a TAAF ``Solver``.

    Honours TAAF's ``minimal_diagnostics`` (Kaggle submission mode): the games
    write no transcript, prompt snapshot, request log, metrics JSONL, viewer
    payload, analysis HTML or vLLM metrics scrape.
    """

    label: str = "HarnessSolver"
    model: str = ""
    analyzer_timeout: float | None = 120.0
    max_actions_per_game: int | None = None
    max_runtime_s_per_game: float | None = None
    concurrency: int = 16
    save_request_logs: bool = False
    start_local_server: bool = False
    local_server_config: str = ""
    local_server_api_key_file: str = ""
    local_server_repo_dir: str = ""
    local_server_port: int | None = None
    local_server_tensor_parallel_size: int | None = None
    local_server_count: int = 1
    kaggle_enable_vllm: bool = field(default=True, repr=False)
    kaggle_runtime_dataset_source: str = field(
        default=DEFAULT_VLLM_RUNTIME_DATASET_SOURCE, repr=False
    )
    kaggle_model_source: str = field(default=DEFAULT_QWEN_MODEL_SOURCE, repr=False)
    kaggle_served_model_name: str = field(default=DEFAULT_SERVED_MODEL_NAME, repr=False)
    kaggle_vllm_port: int = field(
        default_factory=lambda: get_env_int("KAGGLE_VLLM_PORT", DEFAULT_VLLM_PORT),
        repr=False,
    )
    kaggle_vllm_max_model_len: int = field(
        default_factory=lambda: get_env_int("KAGGLE_VLLM_MAX_MODEL_LEN", DEFAULT_VLLM_MAX_MODEL_LEN),
        repr=False,
    )
    kaggle_vllm_tensor_parallel_size: int = field(
        default_factory=lambda: get_env_int(
            "KAGGLE_VLLM_TENSOR_PARALLEL_SIZE", DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
        ),
        repr=False,
    )
    kaggle_vllm_kv_cache_gib: int = field(
        default_factory=lambda: get_env_int("KAGGLE_VLLM_KV_CACHE_GIB", DEFAULT_VLLM_KV_CACHE_GIB),
        repr=False,
    )
    kaggle_vllm_mtp_tokens: int = field(
        default_factory=lambda: get_env_int("KAGGLE_VLLM_MTP_TOKENS", DEFAULT_VLLM_MTP_TOKENS),
        repr=False,
    )
    kaggle_vllm_max_num_seqs: int = field(
        default_factory=lambda: get_env_int("KAGGLE_VLLM_MAX_NUM_SEQS", DEFAULT_VLLM_MAX_NUM_SEQS),
        repr=False,
    )
    kaggle_vllm_dtype: str = field(
        default_factory=lambda: get_env_str("KAGGLE_VLLM_DTYPE", DEFAULT_VLLM_DTYPE),
        repr=False,
    )
    kaggle_vllm_gpu_memory_utilization: str = field(
        default_factory=lambda: get_env_str(
            "KAGGLE_VLLM_GPU_MEMORY_UTILIZATION", DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
        ),
        repr=False,
    )
    kaggle_vllm_kv_cache_dtype: str = field(
        default_factory=lambda: get_env_str("KAGGLE_VLLM_KV_CACHE_DTYPE", DEFAULT_VLLM_KV_CACHE_DTYPE),
        repr=False,
    )
    kaggle_vllm_indexer_kv_dtype: str = field(
        default_factory=lambda: get_env_str("KAGGLE_VLLM_INDEXER_KV_DTYPE", DEFAULT_VLLM_INDEXER_KV_DTYPE),
        repr=False,
    )
    kaggle_vllm_mamba_ssm_cache_dtype: str = field(
        default_factory=lambda: get_env_str(
            "KAGGLE_VLLM_MAMBA_SSM_CACHE_DTYPE", DEFAULT_VLLM_MAMBA_SSM_CACHE_DTYPE
        ),
        repr=False,
    )
    kaggle_vllm_max_num_batched_tokens: int = field(
        default_factory=lambda: get_env_int(
            "KAGGLE_VLLM_MAX_NUM_BATCHED_TOKENS", DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS
        ),
        repr=False,
    )
    kaggle_vllm_enable_prefix_caching: bool = field(
        default_factory=lambda: get_env_bool(
            "KAGGLE_VLLM_ENABLE_PREFIX_CACHING", DEFAULT_VLLM_ENABLE_PREFIX_CACHING
        ),
        repr=False,
    )
    kaggle_vllm_prefix_match_unit: int = field(
        default_factory=lambda: get_env_int(
            "KAGGLE_VLLM_PREFIX_MATCH_UNIT", DEFAULT_VLLM_PREFIX_MATCH_UNIT
        ),
        repr=False,
    )
    kaggle_vllm_prefix_cache_retention_interval: int = field(
        default_factory=lambda: get_env_int(
            "KAGGLE_VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
            DEFAULT_VLLM_PREFIX_CACHE_RETENTION_INTERVAL,
        ),
        repr=False,
    )
    kaggle_vllm_quantization: str = field(
        default_factory=lambda: get_env_str("KAGGLE_VLLM_QUANTIZATION", DEFAULT_VLLM_QUANTIZATION),
        repr=False,
    )
    kaggle_vllm_moe_backend: str = field(
        default_factory=lambda: get_env_str("KAGGLE_VLLM_MOE_BACKEND", ""),
        repr=False,
    )
    kaggle_vllm_gdn_decode_kernel: str = field(
        default_factory=lambda: get_env_str(
            "KAGGLE_VLLM_GDN_DECODE_KERNEL", DEFAULT_VLLM_GDN_DECODE_KERNEL
        ),
        repr=False,
    )
    kaggle_vllm_profile: bool = field(
        default_factory=lambda: get_env_bool("KAGGLE_VLLM_PROFILE", DEFAULT_VLLM_PROFILE),
        repr=False,
    )
    kaggle_vllm_draft_vocab: str = field(
        default_factory=lambda: get_env_str("KAGGLE_VLLM_DRAFT_VOCAB", DEFAULT_VLLM_DRAFT_VOCAB),
        repr=False,
    )
    cancel_drain_timeout_s: float = DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS
    analyzer_factory: AnalyzerFactory | None = field(
        default=None, repr=False, compare=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _local_server_started: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _local_server_env_overrides: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _local_server_cwd: str = field(default="", init=False, repr=False, compare=False)
    _local_server_api_key: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_server_base_url: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_servers: list[_LocalServerRuntime] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    _local_server_original_env: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    # Custom pool sized to self.concurrency: asyncio.to_thread routes onto
    # Python's default executor, capped at min(32, cpu+4) — which would
    # silently cap real concurrency below self.concurrency.
    _worker_pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False, compare=False)
    _vllm_metrics_scraper: VllmMetricsScraper | None = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["analyzer_factory"] = None
        state.pop("_stop_event", None)
        state.pop("_local_server_started", None)
        state.pop("_local_server_env_overrides", None)
        state.pop("_local_server_cwd", None)
        state.pop("_local_server_api_key", None)
        state.pop("_local_server_base_url", None)
        state.pop("_local_servers", None)
        state.pop("_local_server_original_env", None)
        state.pop("_worker_pool", None)
        state.pop("_vllm_metrics_scraper", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._local_server_started = False
        self._local_server_env_overrides = {}
        self._local_server_cwd = ""
        self._local_server_api_key = ""
        self._local_server_base_url = ""
        self._local_servers = []
        self._local_server_original_env = {}
        self._worker_pool = None
        self._vllm_metrics_scraper = None

    def __deepcopy__(self, memo: dict[int, Any]) -> "HarnessSolver":
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_stop_event":
                object.__setattr__(new, key, threading.Event())
            elif key == "analyzer_factory":
                object.__setattr__(new, key, value)
            elif key == "_local_servers":
                object.__setattr__(new, key, [])
            elif key == "_local_server_original_env":
                object.__setattr__(new, key, {})
            elif key in ("_worker_pool", "_vllm_metrics_scraper"):
                object.__setattr__(new, key, None)
            else:
                object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    @property
    def kaggle_dataset_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_dataset_sources(self._kaggle_vllm_config())

    @property
    def kaggle_model_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_model_sources(self._kaggle_vllm_config())

    @property
    def kaggle_setup_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_setup_command(self._kaggle_vllm_config())]

    @property
    def kaggle_teardown_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_teardown_command()]

    def _kaggle_vllm_config(self) -> DuckKaggleVllmConfig:
        return DuckKaggleVllmConfig(
            runtime_dataset_source=self.kaggle_runtime_dataset_source,
            model_source=self.kaggle_model_source,
            served_model_name=self.kaggle_served_model_name,
            vllm_port=self.kaggle_vllm_port,
            max_model_len=self.kaggle_vllm_max_model_len,
            tensor_parallel_size=self.kaggle_vllm_tensor_parallel_size,
            kv_cache_gib=self.kaggle_vllm_kv_cache_gib,
            mtp_tokens=self.kaggle_vllm_mtp_tokens,
            max_num_seqs=self.kaggle_vllm_max_num_seqs,
            dtype=self.kaggle_vllm_dtype,
            gpu_memory_utilization=self.kaggle_vllm_gpu_memory_utilization,
            kv_cache_dtype=self.kaggle_vllm_kv_cache_dtype,
            indexer_kv_dtype=self.kaggle_vllm_indexer_kv_dtype,
            mamba_ssm_cache_dtype=self.kaggle_vllm_mamba_ssm_cache_dtype,
            max_num_batched_tokens=self.kaggle_vllm_max_num_batched_tokens,
            enable_prefix_caching=self.kaggle_vllm_enable_prefix_caching,
            prefix_match_unit=self.kaggle_vllm_prefix_match_unit,
            prefix_cache_retention_interval=self.kaggle_vllm_prefix_cache_retention_interval,
            quantization=self.kaggle_vllm_quantization,
            moe_backend=self.kaggle_vllm_moe_backend,
            gdn_decode_kernel=self.kaggle_vllm_gdn_decode_kernel,
            profile=self.kaggle_vllm_profile,
            draft_vocab=self.kaggle_vllm_draft_vocab,
        )

    def _setup(self) -> None:
        if self.start_local_server:
            self._start_local_servers()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.concurrency)),
            thread_name_prefix="harness-game",
        )
        self._start_vllm_metrics_scraper()

    def _teardown(self) -> None:
        if self._vllm_metrics_scraper is not None:
            self._vllm_metrics_scraper.stop()
            self._vllm_metrics_scraper = None
        if self._local_server_started:
            self._stop_local_servers()
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=False)
            self._worker_pool = None

    def _vllm_metrics_targets(self) -> list[VllmMetricsTarget]:
        """Servers started by this solver, else the server the analyzer itself
        resolves for `self.model` when its provider is vLLM (Kaggle starts vLLM
        outside the solver). Reusing the analyzer's resolver keeps host
        normalization and env precedence identical to the requests being measured."""
        if self._local_servers:
            return [
                VllmMetricsTarget(
                    server_index=runtime.index,
                    metrics_url=metrics_url_from_base_url(runtime.base_url),
                    api_key=runtime.api_key,
                )
                for runtime in self._local_servers
                if runtime.base_url
            ]
        try:
            analyzer_model = resolve_analyzer_model(self.model)
        except ValueError:
            return []
        if analyzer_model.provider != "vllm" or not analyzer_model.base_url:
            return []

        return [
            VllmMetricsTarget(
                server_index=0,
                metrics_url=metrics_url_from_base_url(analyzer_model.base_url),
                api_key=os.environ.get("LOCAL_ANALYZER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            )
        ]

    def _start_vllm_metrics_scraper(self) -> None:
        if self.job_dir is None or self.minimal_diagnostics:
            return
        targets = self._vllm_metrics_targets()
        if not targets:
            return
        self._vllm_metrics_scraper = VllmMetricsScraper(
            targets=targets,
            output_path=metrics_dir(self.job_dir) / VLLM_METRICS_FILENAME,
        )
        self._vllm_metrics_scraper.start()

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._stop_event.clear()
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
        pass_indices_by_game_id: dict[str, int] = {}
        loop = asyncio.get_running_loop()
        pool = self._worker_pool

        async def run_one(index: int, pass_index: int, game: taaf.game.Game) -> None:
            async with semaphore:
                args = (game, index, pass_index, self._local_server_for_game_index(index))
                if pool is not None:
                    await loop.run_in_executor(pool, functools.partial(self._play_one, *args))
                else:
                    # _setup wasn't called (direct test invocation).
                    await asyncio.to_thread(self._play_one, *args)

        tasks: list[asyncio.Task[None]] = []
        for index, game in enumerate(games):
            game_id = game.game_run.game_id if game.game_run is not None else str(index)
            pass_index = pass_indices_by_game_id.get(game_id, 0)
            pass_indices_by_game_id[game_id] = pass_index + 1
            tasks.append(asyncio.create_task(run_one(index, pass_index, game)))
        try:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )
        except asyncio.CancelledError:
            self._stop_event.set()
            await self._drain_game_tasks(tasks)
            self._finish_remaining(games)
            raise

    async def _drain_game_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        timeout = max(0.0, float(self.cancel_drain_timeout_s))
        if timeout == 0.0:
            return
        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    def _start_local_servers(self) -> None:
        server_count = self._resolved_local_server_count()
        started: list[_LocalServerRuntime] = []
        self._capture_local_server_process_env()
        try:
            for server_index in range(server_count):
                runtime = self._local_server_settings(
                    server_index=server_index, server_count=server_count
                )
                print(
                    "Starting local inference server inside solver setup "
                    f"(server {server_index + 1}/{server_count})"
                )
                subprocess.run(
                    ["make", "server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=True,
                )
                if runtime.api_key_file.is_file():
                    runtime.api_key = runtime.api_key_file.read_text(
                        encoding="utf-8"
                    ).strip()
                started.append(runtime)
        except Exception:
            self._local_servers = started
            self._local_server_started = bool(started)
            with contextlib.suppress(Exception):
                self._stop_local_servers()
            raise

        self._local_servers = started
        self._local_server_started = bool(started)
        if started:
            first = started[0]
            self._local_server_cwd = str(first.repo_dir)
            self._local_server_env_overrides = first.env_overrides
            self._local_server_api_key = first.api_key
            self._local_server_base_url = first.base_url
            if first.api_key:
                os.environ["LOCAL_ANALYZER_API_KEY"] = first.api_key
                os.environ["OPENAI_API_KEY"] = first.api_key
            if first.base_url:
                os.environ["LOCAL_ANALYZER_BASE_URL"] = first.base_url
                os.environ["OPENAI_BASE_URL"] = first.base_url
                os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
                os.environ["OPENAI_PROVIDER"] = "vllm"

    def _stop_local_servers(self) -> None:
        runtimes = list(reversed(self._local_servers))
        if not runtimes and self._local_server_env_overrides:
            repo_dir = (
                Path(self._local_server_cwd)
                if self._local_server_cwd
                else self._local_server_repo_dir()
            )
            runtimes = [
                _LocalServerRuntime(
                    index=0,
                    repo_dir=repo_dir,
                    api_key_file=Path(
                        self._local_server_env_overrides.get("SERVER_API_KEY_FILE", "")
                    ),
                    env_overrides=self._local_server_env_overrides,
                    base_url=self._local_server_base_url,
                    api_key=self._local_server_api_key,
                )
            ]
        try:
            for runtime in runtimes:
                subprocess.run(
                    ["make", "stop-server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=False,
                )
        finally:
            self._local_servers = []
            self._local_server_started = False
            self._restore_local_server_process_env()

    def _capture_local_server_process_env(self) -> None:
        self._local_server_original_env = {
            key: os.environ.get(key) for key in _LOCAL_SERVER_PROCESS_ENV_KEYS
        }

    def _restore_local_server_process_env(self) -> None:
        if not self._local_server_original_env:
            return
        for key, value in self._local_server_original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._local_server_original_env = {}

    def _local_server_settings(
        self, *, server_index: int, server_count: int
    ) -> _LocalServerRuntime:
        config_path = self.local_server_config.strip()
        if not config_path:
            raise ValueError(
                "local_server_config is required when start_local_server is enabled."
            )

        repo_dir = self._local_server_repo_dir()
        run_dir = (self.job_dir or Path.cwd()).resolve()
        api_key_file = self._local_server_api_key_path(
            server_index=server_index,
            server_count=server_count,
            run_dir=run_dir,
        )
        pid_path = run_dir / (
            "server.pid" if server_count <= 1 else f"server-{server_index}.pid"
        )
        log_path = run_dir / (
            "server.log" if server_count <= 1 else f"server-{server_index}.log"
        )
        port = self._local_server_port(server_index=server_index)
        base_url = f"http://127.0.0.1:{port}/v1" if port is not None else ""
        env_overrides = {
            "CONFIG_PATH": config_path,
            "SERVER_API_KEY_FILE": str(api_key_file),
            "SERVER_PID": str(pid_path),
            "SERVER_LOG": str(log_path),
            "SERVER_TAIL_ON_WAIT": "true",
            "UV_PROJECT_ENVIRONMENT": str(repo_dir / ".venv"),
        }
        venv_python = self._local_server_venv_python(repo_dir)
        if venv_python is not None:
            env_overrides["SERVER_VENV_PYTHON"] = str(venv_python)
            env_overrides["PYTHON"] = str(venv_python)
        if port is not None:
            env_overrides.update(
                {
                    "SERVER_PORT": str(port),
                    "LOCAL_ANALYZER_BASE_URL": base_url,
                    "OPENAI_BASE_URL": base_url,
                    "LOCAL_ANALYZER_PROVIDER": "vllm",
                    "OPENAI_PROVIDER": "vllm",
                }
            )
        if self.local_server_tensor_parallel_size is not None:
            env_overrides["SERVER_TENSOR_PARALLEL_SIZE"] = str(
                int(self.local_server_tensor_parallel_size)
            )
        if server_count > 1:
            env_overrides["CUDA_VISIBLE_DEVICES"] = (
                self._cuda_visible_device_for_server(server_index)
            )
        return _LocalServerRuntime(
            index=server_index,
            repo_dir=repo_dir,
            api_key_file=api_key_file,
            env_overrides=env_overrides,
            base_url=base_url,
        )

    def _resolved_local_server_count(self) -> int:
        if not self.start_local_server:
            return 0
        return max(1, int(self.local_server_count or 1))

    def _local_server_port(self, *, server_index: int) -> int | None:
        if self.local_server_port is None:
            return None
        return int(self.local_server_port) + int(server_index)

    def _local_server_api_key_path(
        self, *, server_index: int, server_count: int, run_dir: Path
    ) -> Path:
        default_name = (
            "server-api-key" if server_count <= 1 else f"server-{server_index}-api-key"
        )
        base_path = self._resolve_local_server_path(
            self.local_server_api_key_file, default=run_dir / default_name
        )
        if server_count <= 1 or not str(self.local_server_api_key_file or "").strip():
            return base_path
        suffix = base_path.suffix
        stem = base_path.name[: -len(suffix)] if suffix else base_path.name
        return base_path.with_name(f"{stem}-{server_index}{suffix}")

    def _cuda_visible_device_for_server(self, server_index: int) -> str:
        visible_devices = [
            device.strip()
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if server_index < len(visible_devices):
            return visible_devices[server_index]
        return str(server_index)

    def _local_server_repo_dir(self) -> Path:
        repo_dir = (
            Path(self.local_server_repo_dir).expanduser()
            if self.local_server_repo_dir
            else Path(__file__).parents[2]
        )
        repo_dir = repo_dir.resolve()
        if not repo_dir.is_dir():
            raise ValueError(f"local_server_repo_dir does not exist: {repo_dir}")
        return repo_dir

    def _local_server_venv_python(self, repo_dir: Path) -> Path | None:
        repo_venv_python = repo_dir / ".venv" / "bin" / "python"
        if repo_venv_python.is_file():
            return repo_venv_python
        return None

    def _resolve_local_server_path(self, raw_value: str, *, default: Path) -> Path:
        raw = str(raw_value or "").strip()
        if not raw:
            return default.resolve()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        return (self._local_server_repo_dir() / path).resolve()

    def _local_server_env(
        self, overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides or self._local_server_env_overrides)
        return env

    def soft_time_remaining_seconds(self) -> float | None:
        if self.soft_end_time is None:
            return None
        now = (
            datetime.now(self.soft_end_time.tzinfo)
            if self.soft_end_time.tzinfo
            else datetime.now()
        )
        return max(0.0, (self.soft_end_time - now).total_seconds())

    def _local_server_for_game_index(
        self, game_index: int
    ) -> _LocalServerRuntime | None:
        if not self._local_servers:
            return None
        return self._local_servers[int(game_index) % len(self._local_servers)]

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        if self.analyzer_factory is not None:
            return self.analyzer_factory(game, index)
        return ToolAgent(
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            diagnostics=not self.minimal_diagnostics,
            api_key=(
                local_server.api_key
                if local_server is not None
                else self._local_server_api_key
            )
            or None,
            base_url=(
                local_server.base_url
                if local_server is not None
                else self._local_server_base_url
            )
            or None,
            provider="vllm" if local_server is not None else None,
        )

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        try:
            assert game.game_run is not None
            run = game.game_run
            paths = GamePaths.for_game(
                self.job_dir or Path.cwd() / "taaf_harness_artifacts", run.game_id, pass_index
            )
            paths.state_path.parent.mkdir(parents=True, exist_ok=True)
            analyzer = self._make_analyzer(game, index, local_server)
            session = _HarnessGameSession(
                solver=self,
                game=game,
                analyzer=analyzer,
                game_index=index,
                pass_index=pass_index,
                state_path=paths.state_path,
                transcript_path=paths.transcript_path,
                analysis_html_relpath=paths.analysis_html_relpath,
                stop_event=self._stop_event,
                viewer_data_path=paths.viewer_data_path,
            )
            session.play()
        except Exception as exc:
            self._finish_after_error(game, exc)

    def _finish_remaining(self, games: list[taaf.game.Game]) -> None:
        for game in games:
            run = game.game_run
            if run is not None and run.final_score is None:
                try:
                    if self._stop_event.is_set() and run.state == "playing":
                        run.state = "cancelled"
                    game.finish_game()
                except Exception:
                    pass

    def _finish_after_error(self, game: taaf.game.Game, exc: Exception) -> None:
        run = game.game_run
        if run is None or run.final_score is not None:
            return
        run.solver_note = f"error: {type(exc).__name__}: {exc}"
        if run.state == "playing":
            run.state = "crashed"
        with contextlib.suppress(Exception):
            game.finish_game()
        if run.final_score is None:
            with contextlib.suppress(Exception):
                run.final_score = run._compute_final_score()
