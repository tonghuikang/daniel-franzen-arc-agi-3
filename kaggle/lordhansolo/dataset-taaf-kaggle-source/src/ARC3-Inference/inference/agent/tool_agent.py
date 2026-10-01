"""Direct OpenAI-compatible tool-calling analyzer for ARC puzzle runs."""
from __future__ import annotations

import ast
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import requests

from inference.agent.prompts import (
    GAME_OVERVIEW_ADDENDUM,
    GENERAL_GAME_STRATEGY_ADDENDUM,
    PYTHON_ADDENDUM,
    ROLE_PREAMBLE,
    STRATEGY_AUDIT_PROMPT,
    STRUCTURED_RUNTIME_STATE_ADDENDUM,
    MULTIMODAL_CONTEXT_ADDENDUM,
    VISUAL_GAME_ADDENDUM,
)

from inference.agent.model_response import (
    contains_tool_call_markup,
    extract_reasoning_text,
    normalize_message_content,
    recover_tool_calls_from_markup,
    select_reasoning_field,
    strip_tool_call_markup,
)

from inference.agent.grid_view import (
    HistoryViewEncoder,
    ascii_frame_view_payload,
    ascii_game_over_view_payload,
    format_changed_regions,
)

from inference.agent.action import (
    display_action_number,
    format_valid_action_line,
    terminal_action_reason,
    terminal_action_stop_detail,
    to_model_actions,
)

from inference.agent.metrics_log import AgentMetricsRecorder, usage_token_counts

from inference.agent.token_estimate import estimate_tokens

from inference.agent.transcript_log import TurnLog

from inference.agent.vision_context import (
    current_grid_image_enabled,
    current_grid_image_part,
)

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import (
    Frame,
    GameOver,
    HistoryEntry,
    RuntimeState,
    changed_pixel_count,
    changed_regions,
    load_runtime_state,
    total_pixel_count,
)
from inference.utils.env import get_env_bool, get_env_float, get_env_int
from inference.utils.openai_compat import build_chat_payload, build_headers

log = logging.getLogger(__name__)


_LOCAL_ANALYZER_MODEL_ID = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
_LOCAL_ANALYZER_BASE_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
_DEFAULT_ANALYZER_MODEL = os.environ.get(
    "INFERENCE_ANALYZER_MODEL",
    _LOCAL_ANALYZER_MODEL_ID,
)
_LOCAL_ANALYZER_MAX_OUTPUT = get_env_int("LOCAL_ANALYZER_MAX_OUTPUT", 0)
_LOCAL_ANALYZER_CONTEXT_WINDOW = get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 73728)
_LOCAL_ANALYZER_TARGET_CONTEXT = get_env_int("LOCAL_ANALYZER_TARGET_CONTEXT", 0)
_LOCAL_ANALYZER_TIMEOUT = get_env_float("LOCAL_ANALYZER_TIMEOUT", 0.0)
_LOCAL_ANALYZER_TOOL_STEPS = get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
_LOCAL_ANALYZER_TOOL_TIMEOUT = get_env_int("LOCAL_ANALYZER_TOOL_TIMEOUT", 30)
_LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = get_env_int("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024)
_LOCAL_ANALYZER_YIELD_SECONDS = get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
_LOCAL_ANALYZER_ENABLE_THINKING = get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
_LOCAL_ANALYZER_REASONING_EFFORT = os.environ.get("LOCAL_ANALYZER_REASONING_EFFORT", "xhigh")
_LOCAL_ANALYZER_TEMPERATURE = get_env_float("LOCAL_ANALYZER_TEMPERATURE", 1.0)
_LOCAL_ANALYZER_TOP_P = get_env_float("LOCAL_ANALYZER_TOP_P", 0.95)
_LOCAL_ANALYZER_TOP_K = get_env_int("LOCAL_ANALYZER_TOP_K", 20)
_LOCAL_ANALYZER_SEED = get_env_int("LOCAL_ANALYZER_SEED", -1)
_OPTIONAL_ACTION_RESULT_KEYS = (
    "per_step_changed_pixels",
    "per_step_frames",
    "animation",
    "action_display",
    "stop_reason",
)
_UNRENDERED_ACTION_RESULT_KEYS = (
    "largest_changed_regions",
    "changed_region_count",
    "total_pixels",
    "action_display",
    "valid_actions",
    "state",
)
_REQUEST_SAFETY_MARGIN_TOKENS = 512
_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
_MAX_YIELD_BUDGET_MULTIPLIER = 3
_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30
_MAX_PLAN_CHARS = 500

_PYTHON_TOOL_DESCRIPTION = (
    "Run one ephemeral Python snippet against the preloaded game state, and execute real environment "
    "actions from inside it by calling `action(actions)`. The runtime variables, the frame and "
    "segmentation API, and the action semantics are described in the system prompt."
)


def _normalized_plan(value: Any) -> str:
    """The `plan` parameter as it is stored on every action of the call: one line, capped, so
    the persisted history stays searchable without growing with the model's prose."""
    text = " ".join(str(value or "").split())
    if len(text) <= _MAX_PLAN_CHARS:
        return text

    return f"{text[:_MAX_PLAN_CHARS].rstrip()}..."


def _request_tool_choice(tools: list[dict[str, Any]] | None) -> str | None:
    return "auto" if tools else None


def _did_action_animate(action_result: dict[str, Any]) -> bool:
    """Whether the action result carries the `animation` dict the model sees in `last_action_result`."""
    return isinstance(action_result.get("animation"), dict)


def _count_executed_actions(current_frame: Frame | None, action_num: int) -> int:
    return max(current_frame.step if current_frame is not None else 0, max(0, action_num))


def _render_game_over_prompt_lines(game_overs: tuple[GameOver, ...]) -> list[str]:
    """The line naming the game overs the current level has already produced, empty while it
    has produced none. Each count includes the action the game over followed, so it is how
    many actions the run spent rather than how many it survived. The counts are what separates
    a level that ends on a fixed action budget, where every run ends at the same count, from
    one that ends on something the board did, where the counts differ."""
    if not game_overs:
        return []
    counts = ", ".join(str(len(game_over.actions)) for game_over in game_overs)

    return [
        f"Game overs on this level so far: {len(game_overs)}. "
        f"Each ended a run of this many executed actions: {counts}. "
        "The newest one's board is in `game_overs`."
    ]


def _describe_saved_callable(
    module_name: str, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    """Describe the source interface with a bounded signature and first docstring paragraph.

    Source and default expressions are never executed. Each part is limited separately so
    a large default cannot crowd the function's description out of the catalog.
    """
    signature = f"{module_name}.{node.name}"
    if isinstance(node, ast.ClassDef):
        signature += " (class)"
    else:
        signature += f"({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        if isinstance(node, ast.AsyncFunctionDef):
            signature = f"async {signature}"
    if len(signature) > 160:
        signature = signature[:157] + "..."
    docstring = (ast.get_docstring(node) or "").strip()
    summary = " ".join(docstring.split("\n\n")[0].split())
    if len(summary) > 160:
        summary = summary[:157] + "..."

    return f"{signature} — {summary}" if summary else signature


def _describe_saved_modules(modules: dict[str, Any]) -> str:
    """List public source interfaces with signatures and short docstring summaries.

    The names alone survive a compaction while the turns that wrote the module do not, so
    without the interface the model cannot tell what it already built and either rebuilds it
    or reaches for a name it never bound."""
    described = []
    for name in sorted(modules):
        entry = modules.get(name)
        source = str(entry.get("source") or "") if isinstance(entry, dict) else ""
        try:
            body = ast.parse(source).body
        except (SyntaxError, ValueError):
            described.append(name)
            continue
        public = [
            _describe_saved_callable(name, node)
            for node in body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        ]
        described.extend(public or [name])

    return "\n".join(described) or "none"


def _json_deep_copy(value: Any) -> Any:
    """Deep-copy a JSON-serializable value via a JSON round-trip.

    Not interchangeable with `copy.deepcopy`: the round-trip also normalizes
    JSON-incompatible shapes (tuples become lists, keys become strings), which
    callers rely on for payloads that mirror the wire format.
    """
    return json.loads(json.dumps(value))


def _strip_saved_modules_from_text(text: str) -> str:
    """Remove the trailing catalog, preserving the optional grid image caption.

    Catalog entries are single lines. A blank line separates the image caption
    appended by `_build_user_message` from the catalog.
    """
    before, marker, catalog = text.partition("\nSaved Python modules:\n")
    if not marker:

        return text
    _, separator, after = catalog.partition("\n\n")

    return before + separator + after


def _build_system_prompt(*, tool_output_tokens: int) -> str:
    """Assemble the analyzer's system prompt, context before reference.

    The order runs from what the game is, through what its boards look like, to the one tool
    and then the runtime API inside it, because the API section is the bulk of the prompt and
    reads as an appendix rather than as something to work through. It also names the tool it
    documents, so it follows the sections that introduce it. The multimodal section closes the
    reference because it contrasts the attached image with `current_frame`, which only means
    something once the frame view has been described. The strategy section comes last because
    a third of it names runtime variables and result keys, which makes it guidance on the
    reference above rather than more context, and because it is the part meant to be acted on
    this turn.
    """
    prompt = ROLE_PREAMBLE
    prompt += GAME_OVERVIEW_ADDENDUM
    prompt += VISUAL_GAME_ADDENDUM
    prompt += PYTHON_ADDENDUM.format(tool_output_tokens=tool_output_tokens)
    prompt += STRUCTURED_RUNTIME_STATE_ADDENDUM
    if current_grid_image_enabled():
        prompt += MULTIMODAL_CONTEXT_ADDENDUM
    prompt += GENERAL_GAME_STRATEGY_ADDENDUM

    return prompt


@dataclass(frozen=True)
class AnalyzerModelConfig:
    provider: str
    base_url: str
    model_id: str


@dataclass(frozen=True)
class AnalyzerTurnResult:
    step_executed: bool
    retryable_failure: bool = False
    reasoning: str = ""
    yielded_control: bool = False


@dataclass(frozen=True)
class _ToolDispatchResult:
    content: str
    step_executed: bool = False
    action_call_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class _CompactedRequest:
    messages: list[dict[str, Any]]
    estimated_input_tokens: int


@dataclass
class _SandboxActionContext:
    state_path: Path
    fallback_valid_actions: list[str]
    terminal_action_result: dict[str, Any] | None = None


def _host_accessible_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname != "host.docker.internal":
        return base_url
    netloc = "127.0.0.1"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _resolve_analyzer_provider(primary_env_var: str, fallback_env_var: str) -> str:
    """Resolve the provider from the first env var that is set, defaulting to "vllm".

    The fallback variable applies only when the primary one is absent; a primary
    set to an empty or blank value resolves to "vllm", not to the fallback.
    """
    provider = os.environ.get(primary_env_var, os.environ.get(fallback_env_var, "vllm")).strip().lower()
    if not provider:
        return "vllm"

    return provider


def resolve_analyzer_model(model: str) -> AnalyzerModelConfig:
    requested = (model or "").strip()
    lowered = requested.lower()
    if lowered in {"local", "local-qwen", "qwen-local", "qwen"}:
        configured_base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL).strip()
        if not configured_base_url:
            raise ValueError("LOCAL_ANALYZER_BASE_URL must be set for the local analyzer preset.")

        provider = _resolve_analyzer_provider("LOCAL_ANALYZER_PROVIDER", "OPENAI_PROVIDER")
        model_id = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "").strip() or _LOCAL_ANALYZER_MODEL_ID.strip()
        if not model_id:
            raise ValueError("LOCAL_ANALYZER_MODEL_ID must be set for the local analyzer preset.")
        return AnalyzerModelConfig(
            provider=provider,
            base_url=_host_accessible_base_url(configured_base_url),
            model_id=model_id,
        )

    if not requested:
        requested = _LOCAL_ANALYZER_MODEL_ID.strip()
    if not requested:
        raise ValueError(
            "Analyzer model id is required. Set analyzer.model_id in config, pass --model, "
            "or set LOCAL_ANALYZER_MODEL_ID / INFERENCE_ANALYZER_MODEL."
        )

    provider = _resolve_analyzer_provider("OPENAI_PROVIDER", "LOCAL_ANALYZER_PROVIDER")
    base_url = _host_accessible_base_url(
        os.environ.get("OPENAI_BASE_URL", os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL)).strip()
    )
    if not base_url:
        raise ValueError("OPENAI_BASE_URL or LOCAL_ANALYZER_BASE_URL must be set for direct model ids.")
    return AnalyzerModelConfig(provider=provider, base_url=base_url, model_id=requested)


def _is_context_length_error(exc: BaseException) -> bool:
    """Whether the server rejected the request for not fitting its context, in any of the
    wordings vLLM's renderer, vLLM's engine and llama.cpp use. An unrecognized wording makes
    the caller retry a byte-identical request instead of shedding history."""
    message = str(exc).lower()
    return (
        "maximum context length" in message
        or "reduce the length of the input prompt" in message
        or "parameter=input_tokens" in message
        or '"param":"input_tokens"' in message
        or "longer than the maximum model length" in message
        or "exceed_context_size_error" in message
    )


def _control_yield_reason(
    should_stop: Callable[[], bool] | None,
    yield_budget_seconds: float | None,
    turn_started_at: float,
    display_action_num: int,
) -> str | None:
    if should_stop is not None:
        try:
            if should_stop():
                return "stop_requested"
        except Exception as exc:
            log.warning("analyzer stop check failed at action %d: %s", display_action_num, exc)
    if yield_budget_seconds is not None and (time.monotonic() - turn_started_at) >= yield_budget_seconds:
        return "turn_time_budget"

    return None


@dataclass
class _ChatCompletionResult:
    message: dict[str, Any]
    finish_reason: str = ""
    usage: dict[str, Any] | None = None


class ToolAgent:
    """Direct tool-calling analyzer compatible with OpenAI-style endpoints."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_ANALYZER_MODEL,
        timeout: Optional[float] = None,
        save_request_logs: bool = False,
        diagnostics: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        resolved_model = resolve_analyzer_model(model)
        if base_url is not None or provider is not None:
            resolved_model = AnalyzerModelConfig(
                provider=str(provider or resolved_model.provider).strip() or resolved_model.provider,
                base_url=(
                    _host_accessible_base_url(str(base_url).strip())
                    if base_url is not None and str(base_url).strip()
                    else resolved_model.base_url
                ),
                model_id=resolved_model.model_id,
            )
        self._model = resolved_model
        configured_timeout = _LOCAL_ANALYZER_TIMEOUT if timeout is None else timeout
        self._timeout = None if configured_timeout is None or configured_timeout <= 0 else float(configured_timeout)
        self._api_key = str(api_key or "").strip()
        self._tool_steps = None if _LOCAL_ANALYZER_TOOL_STEPS <= 0 else max(1, _LOCAL_ANALYZER_TOOL_STEPS)
        self._python_timeout = min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))
        self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(_LOCAL_ANALYZER_YIELD_SECONDS)
        configured_max_output = _LOCAL_ANALYZER_MAX_OUTPUT
        self._max_output_tokens = None if configured_max_output <= 0 else max(1, configured_max_output)
        self._reply_reserve_tokens = self._max_output_tokens or 512
        self._tool_output_tokens = max(64, _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS)
        self._tool_output_chars = max(256, self._tool_output_tokens * 4)
        self._save_request_logs = bool(save_request_logs)
        self._diagnostics = bool(diagnostics)
        self._system_prompt = _build_system_prompt(
            tool_output_tokens=self._tool_output_tokens,
        )
        self._request_safety_margin_tokens = _REQUEST_SAFETY_MARGIN_TOKENS
        self._context_budget_tokens = max(
            1024,
            _LOCAL_ANALYZER_CONTEXT_WINDOW - self._reply_reserve_tokens - self._request_safety_margin_tokens,
        )
        if _LOCAL_ANALYZER_TARGET_CONTEXT < 0 or _LOCAL_ANALYZER_TARGET_CONTEXT >= self._context_budget_tokens:
            raise ValueError(
                "LOCAL_ANALYZER_TARGET_CONTEXT must be 0 (disabled) or positive and below "
                f"the effective input budget of {self._context_budget_tokens} tokens."
            )
        self._context_target_tokens = _LOCAL_ANALYZER_TARGET_CONTEXT or self._context_budget_tokens
        self._history_messages: list[dict[str, Any]] = []
        self._session_state_path: Path | None = None
        self._session_total_tokens = 0
        self._session_generated_tokens = 0
        self._session_uncached_input_tokens = 0
        self._last_prompt_tokens = 0
        self._step_env_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._runtime_state_source: Callable[[], RuntimeState] | None = None
        self._turn_log: TurnLog | None = None
        self._current_valid_actions: list[str] = []
        self._last_step_summary: dict[str, Any] | None = None
        self._last_turn_executed = False
        self._last_action_result: dict[str, Any] | None = None
        self._python_modules: dict[str, Any] = {}
        self._history_view_encoder = HistoryViewEncoder()
        self._turn_plan = ""
        self._consecutive_yielded_turns = 0
        self._metrics = AgentMetricsRecorder(enabled=self._diagnostics)

    def _headers(self) -> dict[str, str]:
        api_key = (
            self._api_key
            or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        site_url = os.environ.get("LOCAL_ANALYZER_SITE_URL", "").strip()
        app_name = os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Agent Harness").strip()
        return build_headers(
            provider=self._model.provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        )

    def _ensure_session(self, state_path: Path) -> None:
        session_state_path = state_path.resolve()
        if self._session_state_path != session_state_path:
            self._session_state_path = session_state_path
            self._history_messages = []
            self._session_total_tokens = 0
            self._session_generated_tokens = 0
            self._session_uncached_input_tokens = 0
            self._last_prompt_tokens = 0
            self._last_step_summary = None
            self._last_turn_executed = False
            self._last_action_result = None
            self._python_modules = {}
            self._history_view_encoder = HistoryViewEncoder()

    @property
    def total_tokens(self) -> int:
        return max(0, int(self._session_total_tokens))

    @property
    def generated_tokens(self) -> int:
        return max(0, int(self._session_generated_tokens))

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, int(self._session_uncached_input_tokens))

    def _accumulate_usage_tokens(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        token_counts = usage_token_counts(usage)
        self._session_generated_tokens += token_counts["completion_tokens"] or 0
        if token_counts["prompt_tokens"] is not None:
            self._last_prompt_tokens = token_counts["prompt_tokens"]
            self._session_uncached_input_tokens += max(
                0, token_counts["prompt_tokens"] - (token_counts["cached_prompt_tokens"] or 0)
            )
        if token_counts["total_tokens"] is not None:
            self._session_total_tokens += token_counts["total_tokens"]
            return
        self._session_total_tokens += (token_counts["prompt_tokens"] or 0) + (token_counts["completion_tokens"] or 0)

    def _summarize_step_sequence(
        self,
        action_results: list[dict[str, Any]],
        *,
        before_frame: Frame | None,
        after_frame: Frame | None,
    ) -> dict[str, Any] | None:
        if not action_results:
            return None
        executed_results = [item for item in action_results if item.get("executed")]
        if not executed_results:
            return None

        total_executed = 0
        per_step_changed_total = 0
        executed_actions: list[str] = []
        for item in executed_results:
            count = item.get("executed_count")
            try:
                parsed = int(count) if count is not None else 1
            except (TypeError, ValueError):
                parsed = 1
            total_executed += max(1, parsed)
            per_step_counts = item.get("per_step_changed_pixels")
            if not isinstance(per_step_counts, list):
                per_step_counts = [item.get("changed_pixels")]
            for changed_count in per_step_counts:
                try:
                    per_step_changed_total += max(0, int(changed_count or 0))
                except (TypeError, ValueError):
                    continue
            action_names = item.get("executed_actions")
            if isinstance(action_names, list):
                executed_actions.extend(str(name).strip() for name in action_names if str(name).strip())
            else:
                fallback_action = str(item.get("action_display") or "").strip()
                if fallback_action:
                    executed_actions.append(fallback_action)

        before_grid = before_frame.grid if before_frame is not None else ()
        after_grid = after_frame.grid if after_frame is not None else ()

        last = executed_results[-1]
        regions, region_count = changed_regions(before_grid, after_grid)

        return {
            "executed_count": total_executed,
            "executed_actions": executed_actions,
            "level": last.get("level"),
            "level_transition": any(bool(item.get("level_completed")) for item in executed_results),
            "run_complete": any(bool(item.get("run_complete")) for item in executed_results),
            "game_over": any(bool(item.get("game_over")) for item in executed_results),
            "changed_pixels": changed_pixel_count(before_grid, after_grid),
            "per_step_changed_total": per_step_changed_total,
            "animated": _did_action_animate(executed_results[-1]),
            "total_pixels": total_pixel_count(after_grid),
            "largest_changed_regions": regions,
            "changed_region_count": region_count,
        }

    def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:
        image_part = current_grid_image_part(current_frame)
        if image_part is None:
            return {"role": "user", "content": user_prompt}

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{user_prompt}\n\nCurrent grid image, level {current_frame.level}, step {current_frame.step}:"},
                image_part,
            ],
        }


    def _build_user_prompt(
        self,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Frame | None = None,
        history_entries: list[HistoryEntry] | None = None,
        game_overs: tuple[GameOver, ...] = (),
        actions_this_life: int = 0,
        previous_step_summary: dict[str, Any] | None = None,
        previous_turn_executed: bool = True,
        strategy_audit_due: bool = False,
    ) -> str:
        """Build the turn prompt with the saved-module catalog as its final block.

        History compaction relies on this ordering to remove only the catalog.
        """
        history_entries = history_entries or []
        executed_actions = _count_executed_actions(current_frame, action_num)
        current_level = current_frame.level if current_frame is not None else 1
        summary_level = None
        if previous_step_summary is not None:
            try:
                summary_level = int(previous_step_summary.get("level"))
            except (TypeError, ValueError):
                summary_level = None
        if summary_level is not None:
            current_level = max(current_level, summary_level)
        observed_max_level = max(
            [current_level, *[entry.frame.level for entry in history_entries if entry.frame is not None]],
            default=current_level,
        )
        lines: list[str] = []
        if previous_step_summary:
            count = previous_step_summary.get("executed_count")
            try:
                normalized_count = int(count) if count is not None else None
            except (TypeError, ValueError):
                normalized_count = None
            action_label = "action" if normalized_count == 1 else "actions"
            if previous_turn_executed:
                lines.append(f"The code executed {normalized_count or 0} {action_label} in the previous sequence.")
                rendered_actions = [str(name).strip() for name in previous_step_summary.get("executed_actions") or [] if str(name).strip()]
                if rendered_actions:
                    action_prefix = f"Executed actions (last 10 of {len(rendered_actions)}):" if len(rendered_actions) > 10 else "Executed actions:"
                    lines.append(f"{action_prefix} {', '.join(rendered_actions[-10:])}.")
                else:
                    lines.append("Executed actions: none.")
            else:
                lines.append("The previous turn executed no action.\n")
            changed_pixels = previous_step_summary.get("changed_pixels") or 0
            total_pixels = previous_step_summary.get("total_pixels") or 0
            if previous_turn_executed and total_pixels:
                if changed_pixels:
                    lines.append(f"Across the whole turn, those actions changed {changed_pixels} of {total_pixels} board cells.")
                    listed_regions = previous_step_summary.get("largest_changed_regions") or []
                    regions_text = format_changed_regions(listed_regions)
                    if regions_text:
                        region_count = previous_step_summary.get("changed_region_count") or 0
                        region_label = "connected region" if region_count == 1 else "connected regions"
                        if region_count > len(listed_regions):
                            lines.append(f"The change spans {region_count} {region_label}; the largest are {regions_text}.")
                        else:
                            lines.append(f"The change spans {region_count} {region_label}: {regions_text}.")
                elif previous_step_summary.get("per_step_changed_total"):
                    lines.append("Across the whole turn, those actions changed 0 board cells, but individual actions did change the board along the way, so the net effect is zero.")
                elif previous_step_summary.get("animated"):
                    lines.append("Across the whole turn, those actions changed 0 board cells, but the last action played an animation (`last_action_result['animation']`).")
                else:
                    lines.append("Across the whole turn, those actions changed 0 board cells, so the board is identical to the one they started from.")
            if not previous_turn_executed:
                lines.append("You are still on the same level.")
            elif previous_step_summary.get("run_complete"):
                lines.append("You have completed the run!")
            elif previous_step_summary.get("game_over"):
                lines.append(
                    "The last of those actions caused a game over, and the harness then executed a RESET, counted as one more action. "
                    "The level is back at its starting board, so the changes above are no longer visible. The board at the game over is `previous_frame` this turn and `game_overs[-1].frame` until the next game over."
                )
            elif previous_step_summary.get("level_transition"):
                lines.append("You have progressed to a new level!")
            else:
                lines.append("You are still on the same level.")
        elif (current_frame is not None and current_frame.step > 0) or action_num > 0:
            lines.append("No previous action sequence was captured.")
        else:
            lines.append("No previous sequence has been executed yet.")
        level_label = f"level {current_level}"
        if observed_max_level > current_level:
            level_label += f" out of observed max level {observed_max_level} so far"
        state_line = (
            f"Current state: {level_label}; current_frame.step = {executed_actions}; "
            f"actions executed since this level last started = {actions_this_life}."
        )
        lines.extend(
            [
                state_line,
                *_render_game_over_prompt_lines(game_overs),
                f"Valid actions right now: {format_valid_action_line(valid_actions)}.",
            ]
        )
        if strategy_audit_due:
            lines.append(STRATEGY_AUDIT_PROMPT)
        lines.append(f"Saved Python modules:\n{_describe_saved_modules(self._python_modules)}")

        return "\n".join(lines)

    def _tools(self, state_path: Path) -> list[dict[str, Any]]:
        self._ensure_session(state_path)
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _PYTHON_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to run. The snippet is ephemeral and is not saved across tool calls."
                                ),
                            },
                            "plan": {
                                "type": "string",
                                "description": (
                                    "One line naming what the actions in this call are meant to achieve and what "
                                    "result would confirm it. It is stored on every action it executes and stays "
                                    "readable from `history` for the rest of the game, so a later turn can see what "
                                    "you were testing at that point. Include it on every `python` call whose code calls "
                                    f"`action(...)`. Keep it under {_MAX_PLAN_CHARS} characters."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
        enable_thinking: bool | None = None,
        max_output_tokens: int | None = None,
        temperature: float = _LOCAL_ANALYZER_TEMPERATURE,
        top_p: float = _LOCAL_ANALYZER_TOP_P,
        top_k: int = _LOCAL_ANALYZER_TOP_K,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        reasoning_effort: str | None = None,
    ) -> _ChatCompletionResult:
        """One chat completion against the analyzer's server.

        `max_output_tokens` overrides the configured analyzer cap for a single request, and
        the sampling keywords default to the analyzer's configured values, so one request can
        be sent with a length bound or a sampling profile the playing agent's turns do not
        want. `reasoning_effort` (`xhigh`, `medium` or `low`) is sent through
        `chat_template_kwargs` and overrides the server's default for this one request; `None`
        falls back to `LOCAL_ANALYZER_REASONING_EFFORT` (default `xhigh`)."""
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=self._max_output_tokens if max_output_tokens is None else max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING) if enable_thinking is None else bool(enable_thinking),
            tools=tools,
            tool_choice=_request_tool_choice(tools),
            seed=_LOCAL_ANALYZER_SEED,
            reasoning_effort=reasoning_effort or _LOCAL_ANALYZER_REASONING_EFFORT,
        )
        if self._diagnostics and self._model.provider == "vllm":
            diagnostic_request_id = self._metrics.trace_request_payload(payload)
            if diagnostic_request_id is not None:
                payload["request_id"] = diagnostic_request_id
        response = requests.post(
            f"{self._model.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=request_timeout_seconds if request_timeout_seconds is not None else self._timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            message = f"{exc}"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message) from exc
        if getattr(response, "status_code", 200) >= 400:
            detail = response.text.strip()
            message = f"{response.status_code} Error"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message)
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise requests.RequestException("server returned no choices")
        choice = choices[0]
        return _ChatCompletionResult(
            message=choice.get("message", {}),
            finish_reason=str(choice.get("finish_reason", "") or ""),
            usage=payload.get("usage"),
        )

    def _trim_tool_text(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._tool_output_chars:
            return text, False
        omitted = len(text) - self._tool_output_chars
        return f"{text[:self._tool_output_chars]}\n... [truncated {omitted} chars]", True

    def _render_tool_payload(self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()) -> str:
        result = dict(payload)
        truncated = False
        for field in truncate_fields:
            value = result.get(field)
            if isinstance(value, str):
                result[field], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        if truncated:
            result["truncated"] = True
            result["truncation_note"] = (
                f"Tool output was cut off to stay within the ~{self._tool_output_tokens}-token response budget."
            )
        return json.dumps(result, separators=(",", ":"))

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, dict):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(f"Action {index} uses legacy MOUSE x/y fields; use row and col.")
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized

    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "executed": bool(payload.get("executed")),
            "action_num": payload.get("action_num"),
            "level": payload.get("level"),
            "score": payload.get("score"),
            "state": payload.get("state"),
            "valid_actions": payload.get("valid_actions", []),
            "changed_pixels": payload.get("changed_pixels"),
            "total_pixels": payload.get("total_pixels"),
            "largest_changed_regions": payload.get("largest_changed_regions"),
            "changed_region_count": payload.get("changed_region_count"),
            "done": bool(payload.get("done")),
            "level_completed": bool(payload.get("level_completed")),
            "game_over": bool(payload.get("game_over")),
            "run_complete": bool(payload.get("run_complete")),
            "action_display": payload.get("action_display") or payload.get("action_name"),
        }
        executed_actions = payload.get("executed_actions")
        if isinstance(executed_actions, list) and executed_actions:
            compact["executed_actions"] = [str(action).strip() for action in executed_actions if str(action).strip()]
        elif compact.get("action_display"):
            compact["executed_actions"] = [str(compact["action_display"]).strip()]
        batch_size = int(payload.get("requested_count") or payload.get("executed_count") or 1)
        if batch_size > 1 or bool(payload.get("stopped_early")):
            compact["requested_count"] = payload.get("requested_count", batch_size)
            compact["executed_count"] = payload.get("executed_count", batch_size)
            compact["stopped_early"] = bool(payload.get("stopped_early"))
            per_step_changed_pixels = payload.get("per_step_changed_pixels")
            if isinstance(per_step_changed_pixels, list):
                compact["per_step_changed_pixels"] = [
                    int(changed_count or 0) for changed_count in per_step_changed_pixels
                ]
            per_step_frames = payload.get("per_step_frames")
            if isinstance(per_step_frames, list) and any(int(count or 1) > 1 for count in per_step_frames):
                compact["per_step_frames"] = [int(count or 1) for count in per_step_frames]
        if _did_action_animate(payload):
            compact["animation"] = dict(payload["animation"])
        if payload.get("stop_reason"):
            compact["stop_reason"] = payload.get("stop_reason")
        if payload.get("stop_detail"):
            compact["stop_detail"] = payload.get("stop_detail")
        if payload.get("error"):
            compact["error"] = payload.get("error")

        return compact

    def _load_runtime_state(self, state_path: Path) -> RuntimeState:
        """The runtime state from the in-memory source the solver passed to `analyze`, or from the
        state file when there is none."""
        if self._runtime_state_source is not None:
            return self._runtime_state_source()

        return load_runtime_state(state_path)

    def _serialized_runtime_state(
        self,
        state_path: Path,
        fallback_valid_actions: list[str],
        *,
        next_valid_actions: list[str] | None = None,
        last_action_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The runtime state as the model's Python sees it, reloaded from its source."""
        refreshed_state = self._load_runtime_state(state_path)
        current_frame_payload = ascii_frame_view_payload(refreshed_state.current_frame)
        if isinstance(next_valid_actions, list):
            sanitized_actions = [str(item).strip() for item in next_valid_actions if str(item).strip()]
        else:
            sanitized_actions = list(fallback_valid_actions)
        persisted_action_result = (
            last_action_result
            if isinstance(last_action_result, dict)
            else self._last_action_result
        )

        return {
            "current_frame": current_frame_payload,
            "game_overs": ascii_game_over_view_payload(refreshed_state.game_overs),
            "history": self._history_view_encoder.encode(refreshed_state.history),
            "valid_actions": sanitized_actions,
            "last_action_result": self._sandbox_action_result(persisted_action_result),
        }

    @staticmethod
    def _sandbox_action_result(action_result: dict[str, Any] | None) -> dict[str, Any]:
        """The action result as the model's Python sees it. Keys that a compact result carries
        only when there is something to report are filled with `None` so that indexing them never
        raises; before any call the result stays `{}`. It is what `action(...)` returns and what
        `last_action_result` holds."""
        if not isinstance(action_result, dict):
            return {}
        sandbox_result = dict(action_result)
        for key in _OPTIONAL_ACTION_RESULT_KEYS:
            sandbox_result.setdefault(key, None)

        return sandbox_result

    @staticmethod
    def _trim_action_result(action_result: dict[str, Any]) -> dict[str, Any]:
        """The action result as the tool-result text prints it. Fields the user turn already
        reports on its own and fields that never vary are left out. The model's Python still
        sees every field on `action(...)` and on `last_action_result`."""

        return {
            key: value
            for key, value in action_result.items()
            if key not in _UNRENDERED_ACTION_RESULT_KEYS
        }

    def _handle_sandbox_action(
        self,
        context: _SandboxActionContext,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._step_env_callback is None:
            raise RuntimeError("action(actions) is not available in this session.")
        normalized_actions = self._normalize_python_actions(actions)
        previous_terminal_result = context.terminal_action_result
        if previous_terminal_result is not None:
            reason = terminal_action_reason(previous_terminal_result) or "terminal_state"
            compact_payload = {
                "executed": False,
                "action_num": previous_terminal_result.get("action_num"),
                "level": previous_terminal_result.get("level"),
                "score": previous_terminal_result.get("score"),
                "state": previous_terminal_result.get("state"),
                "valid_actions": [],
                "changed_pixels": 0,
                "total_pixels": previous_terminal_result.get("total_pixels"),
                "largest_changed_regions": [],
                "changed_region_count": 0,
                "done": bool(previous_terminal_result.get("done")),
                "level_completed": bool(previous_terminal_result.get("level_completed")),
                "game_over": bool(previous_terminal_result.get("game_over")),
                "run_complete": bool(previous_terminal_result.get("run_complete")),
                "requested_count": len(normalized_actions),
                "executed_count": 0,
                "stopped_early": True,
                "stop_reason": f"previous_{reason}",
                "stop_detail": terminal_action_stop_detail(reason),
            }
            self._last_action_result = dict(compact_payload)
            if self._diagnostics:
                self._metrics.action(compact_payload, requested_actions=normalized_actions, latency_seconds=0.0)

            return {
                "action_result": self._sandbox_action_result(compact_payload),
                "state": self._serialized_runtime_state(
                    context.state_path,
                    context.fallback_valid_actions,
                    next_valid_actions=[],
                    last_action_result=compact_payload,
                ),
            }
        action_started_at = time.monotonic()
        raw_payload = self._step_env_callback({"actions": normalized_actions, "plan": self._turn_plan})
        action_latency_seconds = time.monotonic() - action_started_at
        if not isinstance(raw_payload, dict):
            raise RuntimeError("action(actions) did not return a JSON-like payload.")
        compact_payload = self._compact_action_result(raw_payload)
        if self._diagnostics:
            self._metrics.action(
                compact_payload, requested_actions=normalized_actions, latency_seconds=action_latency_seconds
            )
        next_valid_actions = raw_payload.get("valid_actions")
        if isinstance(next_valid_actions, list):
            self._current_valid_actions = to_model_actions(next_valid_actions)
        if compact_payload.get("executed") and terminal_action_reason(compact_payload):
            context.terminal_action_result = compact_payload
        self._last_action_result = dict(compact_payload)

        return {
            "action_result": self._sandbox_action_result(compact_payload),
            "state": self._serialized_runtime_state(
                context.state_path,
                context.fallback_valid_actions,
                next_valid_actions=next_valid_actions if isinstance(next_valid_actions, list) else None,
                last_action_result=compact_payload,
            ),
        }

    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        self._turn_plan = _normalized_plan(arguments.get("plan"))
        code = str(arguments.get("code", "")).rstrip()
        if not code:
            return _ToolDispatchResult(json.dumps({"error": "python requires a non-empty `code` string."}, separators=(",", ":")), error="empty_code")
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            return _ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, separators=(",", ":")), error=f"syntax_error: {exc}")

        current_frame = self._load_runtime_state(state_path).current_frame
        valid_actions = list(to_model_actions(self._current_valid_actions))
        action_context = _SandboxActionContext(
            state_path=state_path,
            fallback_valid_actions=valid_actions,
        )

        sandbox_result = run_sandboxed_python(
            code=code,
            timeout_seconds=self._python_timeout,
            initial_state=self._serialized_runtime_state(state_path, valid_actions),
            action_handler=partial(self._handle_sandbox_action, action_context),
            modules=self._python_modules,
        )

        if not sandbox_result.get("error") and isinstance(sandbox_result.get("modules"), dict):
            self._python_modules = sandbox_result["modules"]

        action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        payload: dict[str, Any] = {"tool": "python"}
        if sandbox_result.get("workspace_events"):
            payload["workspace"] = sandbox_result["workspace_events"]
        rendered_stdout = str(sandbox_result.get("stdout", "") or "")
        rendered_error = str(sandbox_result.get("error", "") or "")
        if rendered_error:
            payload["error"] = rendered_error
            payload["workspace_status"] = "No module changes from this tool call were committed. Previously saved modules are unchanged."
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
        else:
            payload["returncode"] = 0
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
            elif sandbox_result.get("result") is not None:
                payload["result"] = sandbox_result.get("result")
        if action_results:
            if len(action_results) == 1:
                payload["action_result"] = self._trim_action_result(action_results[-1])
            else:
                payload["action_result"] = {
                    "action_calls": len(action_results),
                    "last_action_result": self._trim_action_result(action_results[-1]),
                }

        step_executed = any(bool(item.get("executed")) for item in action_results)
        if step_executed:
            after_frame = self._load_runtime_state(state_path).current_frame
            self._last_step_summary = self._summarize_step_sequence(
                action_results,
                before_frame=current_frame,
                after_frame=after_frame,
            )
        return _ToolDispatchResult(
            self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
            step_executed=step_executed,
            action_call_count=len(action_results),
            error=rendered_error or None,
        )

    def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        if name == "python":
            return self._run_python_tool(state_path, arguments)
        error_message = (
            "save_module is a function inside the python tool. Call it from Python code."
            if name == "save_module" else f"Unknown tool: {name}"
        )

        return _ToolDispatchResult(json.dumps({"error": error_message}, separators=(",", ":")), error=f"unknown_tool: {name}")

    def _estimate_request_input_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        payload: dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _request_tool_choice(tools)
        return estimate_tokens(payload)

    def _drop_oldest_history_block(self, history: list[dict[str, Any]], *, preserve_recent: int) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        if history[0].get("role") == "user" and self._count_user_turns(history) <= 1:
            return False
        first = history.pop(0)
        first_role = str(first.get("role", "")).strip()
        if first_role in {"assistant", "tool"}:
            while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
                history.pop(0)
            return True
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        while history and history[0].get("role") != "user" and len(history) > preserve_recent:
            history.pop(0)
        return True

    def _strip_historical_saved_modules(self, history: list[dict[str, Any]], *, preserve_recent: int) -> int:
        """Strip old catalogs and count changed messages, preserving the newest catalog.

        Changed messages and text parts are replaced rather than mutated so
        caller-owned history and the before-compaction snapshot stay untouched.
        """
        latest_catalog_index = -1
        for index in range(len(history) - 1, -1, -1):
            message = history[index]
            if message.get("role") != "user":
                continue
            content = message.get("content")
            text_parts = [content] if isinstance(content, str) else [
                part["text"] for part in content or [] if part.get("type") == "text"
            ]
            if any("\nSaved Python modules:\n" in text for text in text_parts):
                latest_catalog_index = index
                break
        removed_sections = 0
        for index in range(min(latest_catalog_index, len(history) - preserve_recent)):
            message = history[index]
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                stripped_content = _strip_saved_modules_from_text(content)
            elif isinstance(content, list):
                stripped_content = [
                    {**part, "text": _strip_saved_modules_from_text(part["text"])}
                    if part.get("type") == "text" else part
                    for part in content
                ]
            else:
                continue
            if stripped_content != content:
                history[index] = {**message, "content": stripped_content}
                removed_sections += 1

        return removed_sections

    def _keep_recent_history_turns(
        self,
        messages: list[dict[str, Any]],
        *,
        max_turns: int,
    ) -> list[dict[str, Any]]:
        if max_turns <= 0 or not messages:
            return []
        kept_reversed: list[dict[str, Any]] = []
        assistant_turns = 0
        for message in reversed(messages):
            kept_reversed.append(message)
            if str(message.get("role", "")).strip() == "assistant":
                assistant_turns += 1
                if assistant_turns >= max_turns:
                    break
        kept = list(reversed(kept_reversed))
        while kept and str(kept[0].get("role", "")).strip() == "tool":
            kept.pop(0)

        return kept

    def _drop_until_first_user_message(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = list(history)
        while trimmed and str(trimmed[0].get("role", "")).strip() != "user":
            trimmed.pop(0)
        return trimmed

    def _count_user_turns(self, history: list[dict[str, Any]]) -> int:
        return sum(1 for message in history if str(message.get("role", "")).strip() == "user")

    def _extend_to_turn_start(self, history: list[dict[str, Any]], kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Grow `kept` back to the user message that opened the turn it starts inside.

        The cap counts assistant replies, and a turn holds one per tool call, so it
        lands mid-turn on any turn where the model called `python` more than once.
        Walking back to the user message keeps that turn whole. Trimming forward
        instead would drop the rest of it, which is the opposite of the intent."""
        if not kept or str(kept[0].get("role", "")).strip() == "user":
            return kept
        start_index = len(history) - len(kept)
        while start_index > 0 and str(history[start_index - 1].get("role", "")).strip() != "user":
            start_index -= 1
        if start_index == 0:
            return kept

        return history[start_index - 1:]

    def _persistent_history_messages(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Carry history within the token budget into the next turn.

        A configured compaction target replaces the separate assistant-turn cap,
        so short turns do not shift the prefix while below the token ceiling.
        Without a target, retain the legacy 30-reply cap."""
        compacted = self._compact_messages_for_context(messages, tools=tools, phase="persist").messages
        if not compacted:
            return []
        compacted_history = compacted[1:]
        if self._context_target_tokens < self._context_budget_tokens:

            return self._drop_until_first_user_message(compacted_history)
        kept = self._keep_recent_history_turns(
            compacted_history,
            max_turns=_PERSISTENT_HISTORY_ASSISTANT_TURNS,
        )

        return self._drop_until_first_user_message(self._extend_to_turn_start(compacted_history, kept))

    def _compact_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        phase: str,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
        force: bool = False,
    ) -> _CompactedRequest:
        """Once the input ceiling is exceeded, strip old catalogs, then drop turns.

        The ceiling is the configured context window minus the reply reserve and
        safety margin. Below it, keep history even when it exceeds the target.
        Above it, remove saved-module catalogs from earlier user messages first,
        preserving the newest user catalog. Then remove oldest turns until the
        estimated whole request reaches `_context_target_tokens`, leaving room
        for subsequent requests to grow.

        The bound is not guaranteed. The newest user turn and `preserve_recent`
        win over meeting it, and the estimate the loop
        trims against can undershoot what the server counts. That undershoot is what
        the retry after a server rejection is for. It lowers the bound by
        `extra_safety_tokens` and passes `force`, which drops at least one removable
        block even if stripping catalogs makes the estimate fit, and keeps the
        pass on the logged path.

        Retained messages stay unchanged below the ceiling, so appending a turn
        preserves the prefix.

        Every pass beyond the untouched fast path emits a `compaction` metrics event,
        even when no block could be dropped, so over-budget requests stay visible.
        `phase` names the call site (`turn_start`, `request`, `overflow_retry`,
        `persist`) so the events can be told apart in the log.
        """
        if not messages:
            return _CompactedRequest([], 0)
        system_message = messages[0]
        history = list(messages[1:])
        request_messages = [system_message, *history]
        preserve_recent = max(0, preserve_recent)
        request_ceiling_tokens = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        request_target_tokens = min(self._context_target_tokens, request_ceiling_tokens)
        request_tokens_before = self._estimate_request_input_tokens(request_messages, tools=tools)
        if not force and request_tokens_before <= request_ceiling_tokens:
            return _CompactedRequest(request_messages, request_tokens_before)

        user_turns_before = self._count_user_turns(history)
        removed_saved_module_sections = self._strip_historical_saved_modules(history, preserve_recent=preserve_recent)
        dropped_blocks = 0
        while history and (
            (force and dropped_blocks == 0)
            or self._estimate_request_input_tokens([system_message, *history], tools=tools) > request_target_tokens
        ):
            if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                break
            dropped_blocks += 1
        history = self._drop_until_first_user_message(history)
        compacted_messages = [system_message, *history]
        request_tokens_after = self._estimate_request_input_tokens(compacted_messages, tools=tools)
        if self._diagnostics:
            self._metrics.compaction(
                phase=phase,
                request_ceiling_tokens=request_ceiling_tokens,
                request_target_tokens=request_target_tokens,
                dropped_blocks=dropped_blocks,
                removed_saved_module_sections=removed_saved_module_sections,
                messages_before=request_messages,
                messages_after=compacted_messages,
                history_after=history,
                user_turns_before=user_turns_before,
                user_turns_after=self._count_user_turns(history),
                request_tokens_before=request_tokens_before,
                request_tokens_after=request_tokens_after,
            )

        return _CompactedRequest(compacted_messages, request_tokens_after)

    def _render_analyzer_status(
        self,
        *,
        yield_budget_seconds: float | None,
        step_executed: bool,
        status_message: str,
    ) -> str:
        """The transcript's closing status block for one turn. The caller renders it
        only when diagnostics are on, because the transcript would drop the result
        anyway and `estimate_tokens` walks the whole history to build it."""

        return (
            f"model: {self._model.model_id}\n"
            f"base_url: {self._model.base_url}\n"
            f"max_output_tokens: {self._max_output_tokens if self._max_output_tokens is not None else 'server default'}\n"
            f"reply_reserve_tokens: {self._reply_reserve_tokens}\n"
            f"context_budget_tokens: {self._context_budget_tokens}\n"
            f"context_target_tokens: {self._context_target_tokens}\n"
            f"request_safety_margin_tokens: {self._request_safety_margin_tokens}\n"
            f"tool_output_tokens: {self._tool_output_tokens}\n"
            f"yield_seconds: {yield_budget_seconds if yield_budget_seconds is not None else 'disabled'}\n"
            f"consecutive_yielded_turns: {self._consecutive_yielded_turns}\n"
            f"available_tools: python\n"
            f"python_timeout_seconds: {self._python_timeout}\n"
            f"history_messages: {len(self._history_messages)}\n"
            f"history_estimated_tokens: {estimate_tokens(self._history_messages)}\n"
            f"last_prompt_tokens: {self._last_prompt_tokens}\n"
            f"step_executed: {step_executed}\n"
            f"message: {status_message}"
        )

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        analysis_attempt: int | None = None,
        transcript_updated: Callable[[str], None] | None = None,
        request_timeout_seconds: Callable[[], float | None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        runtime_state_source: Callable[[], RuntimeState] | None = None,
        strategy_audit_due: bool = False,
    ) -> AnalyzerTurnResult | None:
        if runtime_state_source is None and not state_path.exists():
            return None
        self._ensure_session(state_path)
        self._step_env_callback = step_env
        self._runtime_state_source = runtime_state_source
        self._current_valid_actions = to_model_actions(valid_actions)

        runtime_state = self._load_runtime_state(state_path)
        current_frame = runtime_state.current_frame
        user_prompt = self._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=runtime_state.history,
            game_overs=runtime_state.game_overs,
            actions_this_life=runtime_state.actions_this_life,
            previous_step_summary=self._last_step_summary,
            previous_turn_executed=self._last_turn_executed,
            strategy_audit_due=strategy_audit_due,
        )
        display_action_num = display_action_number(action_num)
        if self._diagnostics:
            self._metrics.begin_turn(
                state_path,
                level=current_frame.level if current_frame is not None else None,
                action=display_action_num,
                analysis_step=analysis_step,
                analysis_attempt=analysis_attempt,
            )
        turn_log = TurnLog(
            state_path,
            transcript_path=transcript_path,
            display_action_num=display_action_num,
            analysis_step=analysis_step,
            analysis_attempt=analysis_attempt,
            model_id=self._model.model_id,
            base_url=self._model.base_url,
            save_request_logs=self._save_request_logs,
            enabled=self._diagnostics,
            on_update=transcript_updated,
        )
        self._turn_log = turn_log
        user_message = self._build_user_message(user_prompt, current_frame)
        turn_log.append_system_message(self._system_prompt)
        turn_log.append_user_message(user_message["content"])

        previous_history_messages = list(self._history_messages)
        preserve_history = True
        turn_started_at = time.monotonic()
        messages: list[dict[str, Any]] = self._compact_messages_for_context(
            [{"role": "system", "content": self._system_prompt}, *self._history_messages, user_message],
            tools=self._tools(state_path),
            phase="turn_start",
            preserve_recent=1,
        ).messages
        step_executed = False
        captured_reasoning = ""
        yielded_control_reason: str | None = None
        turn_outcome: str | None = None
        yield_budget_seconds = (
            None
            if self._yield_seconds is None
            else self._yield_seconds
            * min(1 + self._consecutive_yielded_turns, _MAX_YIELD_BUDGET_MULTIPLIER)
        )

        try:
            turn_count = 0
            while self._tool_steps is None or turn_count < self._tool_steps:
                yielded_control_reason = _control_yield_reason(
                    should_stop, yield_budget_seconds, turn_started_at, display_action_num
                )
                if yielded_control_reason is not None:
                    break
                turn_count += 1
                tools = self._tools(state_path)
                tool_choice = _request_tool_choice(tools)
                compacted_request = self._compact_messages_for_context(messages, tools=tools, phase="request")
                messages = compacted_request.messages
                turn_log.record_request(messages=messages, tools=tools, tool_choice=tool_choice, request_index=turn_count)
                turn_log.write_prompt_snapshot()
                request_started_at = time.monotonic()
                if self._diagnostics:
                    self._metrics.request_started()
                try:
                    request_kwargs: dict[str, Any] = {"tools": tools}
                    if request_timeout_seconds is not None:
                        request_kwargs["request_timeout_seconds"] = request_timeout_seconds()
                    turn_log.append_request_snapshot(event="request")
                    result = self._chat_completion(messages, **request_kwargs)
                    request_latency_seconds = time.monotonic() - request_started_at
                    if self._diagnostics:
                        self._metrics.request_returned(request_latency_seconds)
                    self._accumulate_usage_tokens(result.usage)
                    turn_log.append_request_snapshot(event="response", finish_reason=result.finish_reason)
                except requests.RequestException as exc:
                    request_latency_seconds = time.monotonic() - request_started_at
                    context_length_error = _is_context_length_error(exc)
                    if self._diagnostics:
                        self._metrics.request_returned(request_latency_seconds)
                        self._metrics.request_error(
                            request_index=turn_count,
                            latency_seconds=request_latency_seconds,
                            messages=messages,
                            error=str(exc),
                            context_length_error=context_length_error,
                        )
                    if not context_length_error:
                        raise
                    trimmed_messages = self._compact_messages_for_context(
                        messages,
                        tools=tools,
                        phase="overflow_retry",
                        extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
                        force=True,
                    ).messages
                    if trimmed_messages == messages:
                        raise
                    turn_log.append(
                        "ANALYZER STATUS",
                        "context_overflow_recovered: compacted older history after server rejected the request as too long.",
                    )
                    messages = trimmed_messages
                    continue
                raw_reasoning = extract_reasoning_text(result.message)
                raw_content = normalize_message_content(result.message.get("content", ""))
                tool_calls = _json_deep_copy(result.message.get("tool_calls") or [])
                tool_call_markup_in_text = contains_tool_call_markup(raw_reasoning, raw_content)
                recovered_tool_calls_from_markup = False
                if not tool_calls and tool_call_markup_in_text:
                    tool_calls = recover_tool_calls_from_markup(raw_reasoning, raw_content)
                    recovered_tool_calls_from_markup = bool(tool_calls)
                reasoning = strip_tool_call_markup(raw_reasoning) if tool_call_markup_in_text else raw_reasoning
                content = strip_tool_call_markup(raw_content) if tool_call_markup_in_text else raw_content
                malformed_argument_errors: list[str] = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip() or "unknown"
                    raw_arguments = function.get("arguments", "{}")
                    if isinstance(raw_arguments, str):
                        try:
                            json.loads(raw_arguments)
                        except json.JSONDecodeError as exc:
                            malformed_argument_errors.append(f"{tool_name}: invalid JSON arguments ({exc})")
                turn_log.append_model_response_meta(
                    finish_reason=result.finish_reason,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_markup_in_text=tool_call_markup_in_text,
                    recovered_tool_calls_from_markup=recovered_tool_calls_from_markup,
                    malformed_argument_errors=malformed_argument_errors,
                )
                if self._diagnostics:
                    self._metrics.request(
                        request_index=turn_count,
                        latency_seconds=request_latency_seconds,
                        messages=messages,
                        estimated_prompt_tokens=compacted_request.estimated_input_tokens,
                        finish_reason=result.finish_reason,
                        tool_calls=tool_calls,
                        tool_calls_recovered_from_markup=recovered_tool_calls_from_markup,
                        malformed_argument_errors=malformed_argument_errors,
                        reasoning=reasoning,
                        content=content,
                        usage=result.usage,
                    )
                assistant_message: dict[str, Any] = {"role": "assistant"}

                if reasoning:
                    captured_reasoning = reasoning
                    turn_log.append("THINKING", reasoning)
                    assistant_message[select_reasoning_field(result.message)] = reasoning

                if not tool_calls:
                    if content:
                        turn_log.append("ASSISTANT", content)
                        assistant_message["content"] = content
                    elif reasoning:
                        assistant_message["content"] = None

                    if content or reasoning:
                        messages.append(assistant_message)
                    yielded_control_reason = _control_yield_reason(
                        should_stop, yield_budget_seconds, turn_started_at, display_action_num
                    )
                    if yielded_control_reason is not None:
                        break
                    followup_prefix = "You have not called the tool yet, so nothing has run this turn. "
                    if result.finish_reason == "length":
                        output_limit = (
                            f" ({self._max_output_tokens} tokens)" if self._max_output_tokens is not None else ""
                        )
                        followup_prefix = (
                            f"Your reply was cut off at the output token limit{output_limit} before you called a tool, "
                            "so nothing was executed. "
                        )
                    elif tool_call_markup_in_text:
                        followup_prefix = (
                            "You did not call a tool. `<tool_call>` markup inside reasoning or assistant text is not "
                            "parsed as a tool call, so nothing was executed. "
                        )
                    followup_prompt = f"{followup_prefix}Call the `python` tool now."
                    turn_log.append("USER PROMPT", followup_prompt)
                    messages.append({"role": "user", "content": followup_prompt})
                    continue

                if content:
                    turn_log.append("ASSISTANT", content)
                    assistant_message["content"] = content
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                for tool_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip()
                    raw_args = function.get("arguments", "{}")
                    try:
                        if isinstance(raw_args, str):
                            arguments = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            arguments = _json_deep_copy(raw_args)
                        else:
                            arguments = {}
                    except json.JSONDecodeError:
                        arguments = {}
                    turn_log.append_tool_call(tool_name, raw_args, arguments)
                    if self._diagnostics:
                        self._metrics.begin_tool_call(request_index=turn_count, tool_index=tool_index)
                    tool_started_at = time.monotonic()
                    dispatch = self._dispatch_tool(state_path, tool_name, arguments)
                    if self._diagnostics:
                        self._metrics.tool_call(
                            tool=tool_name,
                            latency_seconds=time.monotonic() - tool_started_at,
                            arguments=arguments,
                            action_call_count=dispatch.action_call_count,
                            step_executed=dispatch.step_executed,
                            result=dispatch.content,
                            error=dispatch.error,
                        )
                    if dispatch.step_executed:
                        step_executed = True
                    turn_log.append_tool_result(tool_name, dispatch.content)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": dispatch.content,
                        }
                    )
                    if dispatch.step_executed:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                    yielded_control_reason = _control_yield_reason(
                        should_stop, yield_budget_seconds, turn_started_at, display_action_num
                    )
                    if yielded_control_reason is not None:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                if yielded_control_reason is not None:
                    break
                if step_executed:
                    break

        except requests.RequestException as exc:
            turn_log.append("ANALYZER STATUS", f"request_error: {exc}")
            preserve_history = False
            turn_log.write_prompt_snapshot()
            log.warning("analyzer request failed at action %d: %s", display_action_num, exc)
            turn_outcome = "request_error"

            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=captured_reasoning)
        except Exception as exc:
            turn_log.append("ANALYZER STATUS", f"error: {exc}")
            preserve_history = False
            turn_log.write_prompt_snapshot()
            log.warning("analyzer failed at action %d: %s", display_action_num, exc)
            turn_outcome = "error"

            return None
        finally:
            if preserve_history:
                self._history_messages = self._persistent_history_messages(messages, tools=self._tools(state_path))
            else:
                self._history_messages = previous_history_messages
            self._last_turn_executed = step_executed
            self._step_env_callback = None
            self._runtime_state_source = None
            self._turn_log = None
            self._current_valid_actions = []
            if self._diagnostics:
                self._metrics.turn(
                    outcome=turn_outcome,
                    turn_started_at=turn_started_at,
                    step_executed=step_executed,
                    yielded_control_reason=yielded_control_reason,
                    history_messages=self._history_messages,
                    last_prompt_tokens=self._last_prompt_tokens,
                )

        if step_executed:
            self._consecutive_yielded_turns = 0
        elif yielded_control_reason == "turn_time_budget":
            self._consecutive_yielded_turns += 1

        if step_executed:
            status_message = "Step executed."
        elif yielded_control_reason is not None:
            status_message = f"Yielded control to solver: {yielded_control_reason}."
        else:
            status_message = "No action(...) call was captured."

        if self._diagnostics:
            turn_log.append(
                "ANALYZER STATUS",
                self._render_analyzer_status(
                    yield_budget_seconds=yield_budget_seconds,
                    step_executed=step_executed,
                    status_message=status_message,
                ),
            )
        turn_log.write_prompt_snapshot()

        return AnalyzerTurnResult(
            step_executed=step_executed,
            reasoning=captured_reasoning,
            yielded_control=yielded_control_reason is not None,
        )
