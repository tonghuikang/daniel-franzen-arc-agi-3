"""Per-game structured metrics: ``<run>/metrics/<game>_metrics.jsonl``, shared by
the solver (:class:`GameMetricsRecorder`) and the analyzer
(:class:`AgentMetricsRecorder`). The run-level vLLM scrape next to it is
``inference.framework.vllm_metrics``.

Every record carries ``event``, ``ts`` (wall clock) and ``game``; the solver adds
``level``, ``action`` and ``elapsed_seconds`` since the game started, the
analyzer adds ``level``, ``action``, ``analysis_step`` and ``analysis_attempt``.
``action`` is off by one between the two: solver events carry the number of
actions executed so far, analyzer events the 1-indexed number of the action the
turn works towards (executed + 1, the ``action=N`` of the transcript header).
Analyzer ``level`` and ``action`` are fixed at turn start, so the rows of a turn
that clears a level still carry the level it started on.

Solver events:

- ``game_start`` — ``number_of_levels``, ``runtime_limit_seconds``, ``max_actions``.
- ``level_cleared`` — ``level`` is the cleared level (the engine already reports
  the next one), ``level_seconds``, ``level_actions``.
- ``auto_reset`` — ``reason``.
- ``analyzer_retry`` — ``analysis_step``, ``analysis_attempt``, ``backoff_seconds``.
- ``game_end`` — ``stop_reason``, ``state``, ``final_score``, ``levels_completed``,
  ``number_of_levels``, ``actions_per_level``, ``base_actions_per_level``,
  ``analysis_steps``, ``generated_tokens``, ``uncached_input_tokens``, ``solver_note``,
  ``runtime_state_io`` — ``write`` / ``read`` sections of the tool runtime state
  JSON over the whole game, each with ``count``, ``total_seconds``, ``max_seconds``.

Analyzer events, one turn per ``analyze()`` call. ``request``, ``tool_call`` and
``action`` join on ``request_index`` / ``tool_index`` / ``action_call_index``;
``action`` rows precede the ``tool_call`` that produced them because each is
written when it happens:

- ``request`` — per model call: ``latency_seconds``, ``message_count``,
  ``estimated_prompt_tokens``, ``finish_reason``, ``tool_call_count``,
  ``tool_calls_recovered_from_markup``, ``malformed_argument_count``,
  ``reasoning_chars`` / ``content_chars``, ``reasoning_tokens_est`` /
  ``content_tokens_est`` and the usage token counts (``prompt_tokens``,
  ``completion_tokens``, ``total_tokens``, ``cached_prompt_tokens``).
- ``request_error`` — ``latency_seconds``, ``message_count``, ``error``,
  ``context_length_error``.
- ``tool_call`` — ``tool``, ``latency_seconds``, ``code_chars``,
  ``action_call_count``, ``step_executed``, ``result_chars``, ``error``.
- ``action`` — per ``action(...)`` call from the sandbox: ``action_call_index``,
  ``latency_seconds``, ``requested_count``, ``executed_count``, ``executed``,
  ``actions``, ``changed_pixels``, ``per_step_changed_pixels``, ``noop_count``
  (executed steps whose board did not change), ``level_completed``,
  ``game_over``, ``stop_reason``, ``error``.
- ``compaction`` — one per compaction pass beyond the untouched fast path:
  ``phase`` (``turn_start`` / ``request`` / ``overflow_retry`` / ``persist``),
  ``request_ceiling_tokens``, ``dropped_blocks``, ``messages_before`` / ``_after``,
  ``user_turns_before`` / ``_after``, ``history_tokens_after``,
  ``request_tokens_before`` / ``_after``.
  ``removed_saved_module_sections`` counts messages whose old catalog was stripped
  before dropping blocks, including messages subsequently dropped. It distinguishes
  catalog-only compaction from an unchanged pass when ``dropped_blocks`` is zero.
- ``turn`` — exactly one per turn, always last: ``outcome`` (``step_executed``,
  ``yielded``, ``no_action``, ``request_error``, ``error``), ``yield_reason``,
  ``step_executed``, ``duration_seconds``, ``model_seconds`` / ``tool_seconds`` /
  ``action_seconds``, ``request_count``, ``tool_call_count``,
  ``action_call_count``, ``executed_action_count``, ``noop_action_count``,
  ``history_messages``, ``history_estimated_tokens``, ``last_prompt_tokens``.

Metrics are pure observability: a failed open or write is logged and disables
the recorder until it is reopened — every turn for the analyzer, every event
for the solver, so a transient failure costs one turn or one event and can
never reach the game loop. A recorder built with ``enabled=False`` (TAAF's
``minimal_diagnostics``) returns from every event at once and never opens the file.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.agent.log_text import trim_log_text
from inference.agent.run_paths import GamePaths
from inference.agent.token_estimate import estimate_tokens

log = logging.getLogger(__name__)

_ERROR_MAX_CHARS = 300
_REQUEST_ERROR_MAX_CHARS = 500


def append_metrics_record(log_path: Path, record: dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=True))
        output_file.write("\n")


def _first_nonnegative_int(usage: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        try:
            return max(0, int(usage[key]))
        except (KeyError, TypeError, ValueError):
            continue

    return None


def usage_token_counts(usage: dict[str, Any] | None) -> dict[str, int | None]:
    """Extract prompt/completion/total/cached token counts from an OpenAI-style
    usage payload, tolerating the field aliases different servers use."""
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_prompt_tokens": None,
        }
    prompt_details = usage.get("prompt_tokens_details")
    cached_prompt_tokens = (
        _first_nonnegative_int(prompt_details, ("cached_tokens",))
        if isinstance(prompt_details, dict)
        else None
    )

    return {
        "prompt_tokens": _first_nonnegative_int(usage, ("prompt_tokens", "input_tokens")),
        "completion_tokens": _first_nonnegative_int(usage, ("completion_tokens", "output_tokens", "generated_tokens")),
        "total_tokens": _first_nonnegative_int(usage, ("total_tokens",)),
        "cached_prompt_tokens": cached_prompt_tokens,
    }


def _turn_outcome(step_executed: bool, yielded_control_reason: str | None) -> str:
    if step_executed:
        return "step_executed"
    if yielded_control_reason is not None:
        return "yielded"

    return "no_action"


class _MetricsRecorder:
    """Guarded append-only writer for one game's metrics JSONL. Any failure is
    logged and disables the recorder until the next `_open`. Subclasses return
    from every public method at once when the recorder is not `enabled`."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._log_path: Path | None = None

    def _open(self, state_path: Path) -> None:
        try:
            log_path = GamePaths.from_state_path(state_path).metrics_log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("metrics log unavailable, disabling: %s", exc)
            self._log_path = None

            return
        self._log_path = log_path

    def _append(self, event: str, record: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        try:
            append_metrics_record(self._log_path, {"event": event, "ts": round(time.time(), 3), **record})
        except Exception as exc:  # noqa: BLE001
            log.warning("metrics write failed for %s event, disabling: %s", event, exc)
            self._log_path = None


class GameMetricsRecorder(_MetricsRecorder):
    """Solver-side events for one game; `level` and `action` are passed by the
    caller because they change under the recorder's feet. The log is reopened
    before every event (there are only a handful per game), so a transient
    write failure loses that event alone."""

    def __init__(self, state_path: Path, *, started_at: float, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._state_path = state_path
        self._game = GamePaths.from_state_path(state_path).game_stem
        self._started_at = started_at
        self._level_started_at = started_at

    def _append_game_event(self, event: str, *, level: int, action: int, payload: dict[str, Any]) -> None:
        self._open(self._state_path)
        self._append(
            event,
            {
                "game": self._game,
                "level": level,
                "action": action,
                "elapsed_seconds": round(max(0.0, time.monotonic() - self._started_at), 3),
                **payload,
            },
        )

    def game_start(
        self,
        *,
        level: int,
        action: int,
        number_of_levels: int,
        runtime_limit_seconds: float | None,
        max_actions: int | None,
    ) -> None:
        if not self._enabled:
            return
        self._level_started_at = time.monotonic()
        self._append_game_event(
            "game_start",
            level=level,
            action=action,
            payload={
                "number_of_levels": number_of_levels,
                "runtime_limit_seconds": runtime_limit_seconds,
                "max_actions": max_actions,
            },
        )

    def level_cleared(self, cleared_level: int, *, action: int, level_actions: int | None) -> None:
        """Emit `level_cleared` for the level just finished and restart the
        per-level clock."""
        if not self._enabled:
            return
        now = time.monotonic()
        self._append_game_event(
            "level_cleared",
            level=cleared_level,
            action=action,
            payload={
                "level_seconds": round(max(0.0, now - self._level_started_at), 3),
                "level_actions": level_actions,
            },
        )
        self._level_started_at = now

    def auto_reset(self, *, level: int, action: int, reason: str) -> None:
        if not self._enabled:
            return
        self._append_game_event("auto_reset", level=level, action=action, payload={"reason": reason})

    def analyzer_retry(
        self,
        *,
        level: int,
        action: int,
        analysis_step: int,
        analysis_attempt: int,
        backoff_seconds: float,
    ) -> None:
        if not self._enabled:
            return
        self._append_game_event(
            "analyzer_retry",
            level=level,
            action=action,
            payload={
                "analysis_step": analysis_step,
                "analysis_attempt": analysis_attempt,
                "backoff_seconds": backoff_seconds,
            },
        )

    def game_end(
        self,
        *,
        level: int,
        action: int,
        stop_reason: str,
        state: str,
        final_score: float | None,
        levels_completed: int,
        number_of_levels: int,
        actions_per_level: list[int],
        base_actions_per_level: list[int] | None,
        analysis_steps: int,
        generated_tokens: int,
        uncached_input_tokens: int,
        solver_note: str | None,
        runtime_state_io: dict[str, Any],
    ) -> None:
        if not self._enabled:
            return
        self._append_game_event(
            "game_end",
            level=level,
            action=action,
            payload={
                "stop_reason": stop_reason,
                "state": state,
                "final_score": final_score,
                "levels_completed": levels_completed,
                "number_of_levels": number_of_levels,
                "actions_per_level": actions_per_level,
                "base_actions_per_level": base_actions_per_level,
                "analysis_steps": analysis_steps,
                "generated_tokens": generated_tokens,
                "uncached_input_tokens": uncached_input_tokens,
                "solver_note": solver_note,
                "runtime_state_io": runtime_state_io,
            },
        )


@dataclass
class _TurnTotals:
    """Per-turn accumulators behind the `turn` event, fed by the request,
    tool-call and action paths."""
    request_count: int = 0
    model_seconds: float = 0.0
    tool_call_count: int = 0
    tool_seconds: float = 0.0
    action_call_count: int = 0
    action_seconds: float = 0.0
    executed_action_count: int = 0
    noop_action_count: int = 0


class AgentMetricsRecorder(_MetricsRecorder):
    """Analyzer-side events. `begin_turn` fixes the game/level/action context and
    resets the totals; `begin_tool_call` fixes the request/tool indices that the
    `action` events emitted from inside the sandbox are joined on. The turn
    totals are fed at the measurement points (`request_started`,
    `request_returned`, `tool_call`, `action`), not by the event emitters, so a
    turn that dies between the model returning and the `request` event still
    accounts for the call."""

    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._turn_context: dict[str, Any] = {}
        self._tool_context: dict[str, Any] = {}
        self._totals = _TurnTotals()

    def begin_turn(
        self,
        state_path: Path,
        *,
        level: int | None,
        action: int,
        analysis_step: int | None,
        analysis_attempt: int | None,
    ) -> None:
        if not self._enabled:
            return
        self._open(state_path)
        self._totals = _TurnTotals()
        self._tool_context = {}
        self._turn_context = {
            "game": GamePaths.from_state_path(state_path).game_stem,
            "level": level,
            "action": action,
            "analysis_step": analysis_step,
            "analysis_attempt": analysis_attempt,
        }

    def begin_tool_call(self, *, request_index: int, tool_index: int) -> None:
        if not self._enabled:
            return
        self._tool_context = {"request_index": request_index, "tool_index": tool_index}

    def _append_turn_event(self, event: str, payload: dict[str, Any]) -> None:
        self._append(event, {**self._turn_context, **payload})

    def trace_request_payload(self, payload: dict[str, Any]) -> str | None:
        from uuid import uuid4

        from inference.framework.cache_diagnostics import hash_json, is_cache_diagnostics_enabled

        if not self._enabled or not is_cache_diagnostics_enabled():
            return None
        try:
            request_id = f"arc3-{uuid4().hex}"
            self._append_turn_event("cache_request", {
                "request_id": f"chatcmpl-{request_id}",
                "request_index": self._totals.request_count,
                "message_hashes": [hash_json(message) for message in payload.get("messages", [])],
                "message_roles": [message.get("role") for message in payload.get("messages", [])],
                "tools_hash": hash_json(payload.get("tools")),
                "chat_template_kwargs_hash": hash_json(payload.get("chat_template_kwargs")),
            })
        except Exception:
            log.exception("Cache request diagnostic failed")

            return None

        return request_id

    def request_started(self) -> None:
        if not self._enabled:
            return
        self._totals.request_count += 1

    def request_returned(self, latency_seconds: float) -> None:
        if not self._enabled:
            return
        self._totals.model_seconds += latency_seconds

    def request(
        self,
        *,
        request_index: int,
        latency_seconds: float,
        messages: list[dict[str, Any]],
        estimated_prompt_tokens: int,
        finish_reason: str,
        tool_calls: list[dict[str, Any]],
        tool_calls_recovered_from_markup: bool,
        malformed_argument_errors: list[str],
        reasoning: str,
        content: str,
        usage: dict[str, Any] | None,
    ) -> None:
        if not self._enabled:
            return
        self._append_turn_event(
            "request",
            {
                "request_index": request_index,
                "latency_seconds": round(latency_seconds, 3),
                "message_count": len(messages),
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "finish_reason": finish_reason,
                "tool_call_count": len(tool_calls),
                "tool_calls_recovered_from_markup": tool_calls_recovered_from_markup,
                "malformed_argument_count": len(malformed_argument_errors),
                "reasoning_chars": len(reasoning),
                "content_chars": len(content),
                "reasoning_tokens_est": estimate_tokens(reasoning) if reasoning else 0,
                "content_tokens_est": estimate_tokens(content) if content else 0,
                **usage_token_counts(usage),
            },
        )

    def request_error(
        self,
        *,
        request_index: int,
        latency_seconds: float,
        messages: list[dict[str, Any]],
        error: str,
        context_length_error: bool,
    ) -> None:
        if not self._enabled:
            return
        self._append_turn_event(
            "request_error",
            {
                "request_index": request_index,
                "latency_seconds": round(latency_seconds, 3),
                "message_count": len(messages),
                "error": trim_log_text(error, max_chars=_REQUEST_ERROR_MAX_CHARS),
                "context_length_error": context_length_error,
            },
        )

    def tool_call(
        self,
        *,
        tool: str,
        latency_seconds: float,
        arguments: dict[str, Any],
        action_call_count: int,
        step_executed: bool,
        result: str,
        error: str | None,
    ) -> None:
        if not self._enabled:
            return
        self._totals.tool_call_count += 1
        self._totals.tool_seconds += latency_seconds
        self._append_turn_event(
            "tool_call",
            {
                **self._tool_context,
                "tool": tool,
                "latency_seconds": round(latency_seconds, 3),
                "code_chars": len(str(arguments.get("code", "") or "")),
                "action_call_count": action_call_count,
                "step_executed": step_executed,
                "result_chars": len(result),
                "error": trim_log_text(error, max_chars=_ERROR_MAX_CHARS) if error else None,
            },
        )

    def action(self, compact_payload: dict[str, Any], *, requested_actions: list[dict[str, Any]], latency_seconds: float) -> None:
        """One `action` event per `action(...)` call, folded into the turn totals.
        A no-op is an executed step whose board did not change."""
        if not self._enabled:
            return
        executed = bool(compact_payload.get("executed"))
        executed_count = int(compact_payload.get("executed_count") or (1 if executed else 0))
        per_step_changed_pixels = compact_payload.get("per_step_changed_pixels")
        if not isinstance(per_step_changed_pixels, list):
            per_step_changed_pixels = [compact_payload.get("changed_pixels")] if executed else []
        noop_count = sum(1 for changed in per_step_changed_pixels if changed == 0)
        error = compact_payload.get("error")
        self._totals.action_call_count += 1
        self._totals.action_seconds += latency_seconds
        self._totals.executed_action_count += executed_count
        self._totals.noop_action_count += noop_count
        self._append_turn_event(
            "action",
            {
                **self._tool_context,
                "action_call_index": self._totals.action_call_count,
                "latency_seconds": round(latency_seconds, 3),
                "requested_count": len(requested_actions),
                "executed_count": executed_count,
                "executed": executed,
                "actions": compact_payload.get("executed_actions") or [],
                "changed_pixels": compact_payload.get("changed_pixels"),
                "per_step_changed_pixels": per_step_changed_pixels,
                "noop_count": noop_count,
                "level_completed": bool(compact_payload.get("level_completed")),
                "game_over": bool(compact_payload.get("game_over")),
                "stop_reason": compact_payload.get("stop_reason"),
                "error": trim_log_text(str(error), max_chars=_ERROR_MAX_CHARS) if error else None,
            },
        )

    def compaction(
        self,
        *,
        phase: str,
        request_ceiling_tokens: int,
        request_target_tokens: int,
        dropped_blocks: int,
        removed_saved_module_sections: int,
        messages_before: list[dict[str, Any]],
        messages_after: list[dict[str, Any]],
        history_after: list[dict[str, Any]],
        user_turns_before: int,
        user_turns_after: int,
        request_tokens_before: int,
        request_tokens_after: int,
    ) -> None:
        """The message lists arrive raw so counting them and the `estimate_tokens`
        pass over the trimmed history are paid only once past the gate. The remaining
        counts stay the caller's, because each is either read before the drop loop
        runs or is the value the ceiling check itself acted on, which recomputing here
        would leave free to disagree with the decision it records."""
        if not self._enabled:
            return
        self._append_turn_event(
            "compaction",
            {
                "phase": phase,
                "request_ceiling_tokens": request_ceiling_tokens,
                "request_target_tokens": request_target_tokens,
                "dropped_blocks": dropped_blocks,
                "removed_saved_module_sections": removed_saved_module_sections,
                "messages_before": len(messages_before),
                "messages_after": len(messages_after),
                "user_turns_before": user_turns_before,
                "user_turns_after": user_turns_after,
                "history_tokens_after": estimate_tokens(history_after),
                "request_tokens_before": request_tokens_before,
                "request_tokens_after": request_tokens_after,
            },
        )

    def turn(
        self,
        *,
        outcome: str | None,
        turn_started_at: float,
        step_executed: bool,
        yielded_control_reason: str | None,
        history_messages: list[dict[str, Any]],
        last_prompt_tokens: int,
    ) -> None:
        """The `turn` summary: wall-clock split between model, tool and action
        time plus the counts accumulated over the turn. `outcome` defaults to
        what `step_executed` / `yielded_control_reason` imply."""
        if not self._enabled:
            return
        totals = self._totals
        self._append_turn_event(
            "turn",
            {
                "outcome": outcome or _turn_outcome(step_executed, yielded_control_reason),
                "yield_reason": yielded_control_reason,
                "step_executed": step_executed,
                "duration_seconds": round(time.monotonic() - turn_started_at, 3),
                "model_seconds": round(totals.model_seconds, 3),
                "tool_seconds": round(totals.tool_seconds, 3),
                "action_seconds": round(totals.action_seconds, 3),
                "request_count": totals.request_count,
                "tool_call_count": totals.tool_call_count,
                "action_call_count": totals.action_call_count,
                "executed_action_count": totals.executed_action_count,
                "noop_action_count": totals.noop_action_count,
                "history_messages": len(history_messages),
                "history_estimated_tokens": estimate_tokens(history_messages),
                "last_prompt_tokens": last_prompt_tokens,
            },
        )
