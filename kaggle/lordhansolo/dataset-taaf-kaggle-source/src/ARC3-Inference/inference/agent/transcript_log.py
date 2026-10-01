"""The run's human-readable logs.

`TurnLog` is what one analyzer turn writes for people: the per-game transcript
(`transcripts/<game>.txt`), the latest-model-call snapshot in `prompts/` and,
when enabled, the request JSONL. The rest renders messages, tool calls and tool
results into plain text and turns a transcript into the analysis HTML. Where
the files live is `inference.agent.run_paths`.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Callable

from inference.agent.log_text import trim_log_text
from inference.agent.model_response import normalize_tool_call_arguments
from inference.agent.run_paths import GamePaths

_RESPONSE_META_MAX_CHARS = 4000


def append_transcript_section(log_path: Path, label: str, content: str) -> None:
    rendered_content = content.strip()
    if not rendered_content:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{label}]\n")
        f.write(rendered_content)
        f.write("\n\n")


def render_transcript_section(label: str, content: str) -> str:
    rendered_content = content.strip()
    if not rendered_content:
        return ""
    return f"[{label}]\n{rendered_content}\n\n"


def render_model_response_meta(
    *,
    finish_reason: str,
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    tool_call_markup_in_text: bool,
    recovered_tool_calls_from_markup: bool,
    malformed_argument_errors: list[str],
) -> str:
    lines = [
        f"finish_reason: {finish_reason or '(empty)'}",
        f"tool_call_count: {len(tool_calls)}",
        f"content_chars: {len(content)}",
        f"reasoning_chars: {len(reasoning)}",
        f"tool_call_markup_in_text: {'yes' if tool_call_markup_in_text else 'no'}",
        f"tool_calls_recovered_from_markup: {'yes' if recovered_tool_calls_from_markup else 'no'}",
    ]
    if malformed_argument_errors:
        lines.append("tool_call_argument_issues:")
        lines.extend(f"- {issue}" for issue in malformed_argument_errors)
    if tool_calls:
        lines.append("raw_tool_calls:")
        raw_tool_calls = json.dumps(tool_calls, indent=2, ensure_ascii=True)
        lines.append(trim_log_text(raw_tool_calls, max_chars=_RESPONSE_META_MAX_CHARS))

    return "\n".join(lines)


def _render_tool_parameter_text(value: Any) -> str:
    if isinstance(value, str):
        return value.rstrip("\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def render_tool_call_markup(tool_name: str, arguments: Any) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
    try:
        parsed_arguments = normalize_tool_call_arguments(arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""

    lines = ["<tool_call>", f"<function={name}>"]
    for parameter_name, parameter_value in parsed_arguments.items():
        lines.append(f"<parameter={parameter_name}>")
        rendered_value = _render_tool_parameter_text(parameter_value)
        if rendered_value:
            lines.extend(rendered_value.splitlines())
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_verbatim_message_text(content: Any) -> str:
    """Message content exactly as sent to the model; image blocks are replaced
    by a placeholder since their data URI carries no readable information."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                parts.append("[IMAGE]")
        return "\n".join(parts)
    if content is None:
        return ""

    return str(content)


def _render_prompt_log_message(message: dict[str, Any]) -> str:
    """Render one request message verbatim for the prompt snapshot, so the log
    shows the model input 1:1 rather than a cleaned-up, human-readable view."""
    role = str(message.get("role", "")).strip().upper() or "UNKNOWN"
    header = f"[{role}]"
    tool_call_id = str(message.get("tool_call_id", "")).strip()
    if role == "TOOL" and tool_call_id:
        header = f"[TOOL RESULT: {tool_call_id}]"
    blocks = [header]

    content = _render_verbatim_message_text(message.get("content", ""))
    if content:
        blocks.append(content)

    reasoning = message.get("reasoning")
    if reasoning in (None, ""):
        reasoning = message.get("reasoning_content", "")
    reasoning = _render_verbatim_message_text(reasoning)
    if reasoning:
        blocks.append("[REASONING]")
        blocks.append(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = str(function.get("name", "")).strip() or "unknown"
            blocks.append(f"[ASSISTANT TOOL CALL: {name}]")
            tool_call_id = str(tool_call.get("id", "")).strip()
            if tool_call_id:
                blocks.append(f"id: {tool_call_id}")
            raw_arguments = function.get("arguments", "{}")
            blocks.append("arguments:")
            blocks.append(raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments, ensure_ascii=True))

    return "\n".join(blocks)


def _append_request_snapshot(
    log_path: Path,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    event: str | None = None,
    tool_choice: str | None = None,
    finish_reason: str | None = None,
    analysis_step: int | None = None,
    analysis_attempt: int | None = None,
    action: int | None = None,
    request_index_within_turn: int | None = None,
) -> None:
    payload = {
        "messages": messages,
        "tools": tools or [],
    }
    if event:
        payload["event"] = event
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if finish_reason is not None:
        payload["finish_reason"] = str(finish_reason)
    if analysis_step is not None:
        payload["analysis_step"] = analysis_step
    if analysis_attempt is not None:
        payload["analysis_attempt"] = analysis_attempt
    if action is not None:
        payload["action"] = action
    if request_index_within_turn is not None:
        payload["request_index_within_turn"] = request_index_within_turn
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=True,
            )
        )
        f.write("\n")


def _write_prompt_log_snapshot(
    log_path: Path,
    *,
    model_id: str,
    base_url: str,
    display_action_num: int,
    analysis_step: int | None,
    request_index: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
    transcript: str,
) -> None:
    rendered_messages = "\n\n".join(_render_prompt_log_message(message) for message in messages)
    rendered_tools: list[str] = []
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name", "")).strip() or "unknown"
        description = str(function.get("description", "")).strip()
        if description:
            rendered_tools.append(f"- {name}: {description}")
        else:
            rendered_tools.append(f"- {name}")
    analysis_label = str(analysis_step) if analysis_step is not None else "n/a"
    transcript_text = transcript.strip()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LATEST MODEL CALL SNAPSHOT\n")
        f.write(f"model: {model_id}\n")
        f.write(f"base_url: {base_url}\n")
        f.write(f"analysis_step: {analysis_label}\n")
        f.write(f"action: {display_action_num}\n")
        f.write(f"request_index_within_turn: {request_index}\n")
        f.write(f"message_count: {len(messages)}\n")
        f.write(f"tool_choice: {tool_choice or '(none)'}\n")
        f.write("\n[AVAILABLE TOOLS]\n")
        f.write("\n".join(rendered_tools) if rendered_tools else "(none)")
        f.write("\n\n[MODEL INPUT]\n")
        f.write(rendered_messages.strip())
        f.write("\n\n[TURN TRANSCRIPT SO FAR]\n")
        f.write(transcript_text)
        f.write("\n")


class TurnLog:
    """One analyzer turn's human-readable logs.

    The transcript is appended section by section to the per-game analyzer text
    file (header written on construction) and mirrored to an optional listener
    with the full rendered text after every append. `record_request` captures
    the request about to be sent; `write_prompt_snapshot` overwrites the
    `prompts/` file with that request plus the transcript so far, and
    `append_request_snapshot` appends it to the request JSONL when request logs
    are enabled. The `append_*` methods take the raw model output and render
    it only once past the gate. With `enabled=False` every method returns at
    once and nothing is rendered or written.
    """

    def __init__(
        self,
        state_path: Path,
        *,
        transcript_path: Path | None,
        display_action_num: int,
        analysis_step: int | None,
        analysis_attempt: int | None,
        model_id: str,
        base_url: str,
        save_request_logs: bool,
        enabled: bool = True,
        on_update: Callable[[str], None] | None = None,
    ) -> None:
        paths = GamePaths.from_state_path(state_path)
        self._enabled = enabled
        self._log_path = transcript_path or (state_path.parent / f"{state_path.stem}_analyzer.txt")
        self._prompt_log_path = paths.prompt_log_path
        self._request_log_path = paths.request_log_path if save_request_logs else None
        self._display_action_num = display_action_num
        self._analysis_step = analysis_step
        self._analysis_attempt = analysis_attempt
        self._model_id = model_id
        self._base_url = base_url
        self._on_update = on_update
        self._request_messages: list[dict[str, Any]] | None = None
        self._request_tools: list[dict[str, Any]] | None = None
        self._request_tool_choice: str | None = None
        self._request_index = 0
        step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
        header = f"\n--- {step_label}action={display_action_num} | {time.strftime('%H:%M:%S')} | tool-agent ---\n"
        if enabled:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(header)
        self._parts = [header]

    def append(self, label: str, content: str) -> None:
        if not self._enabled:
            return
        append_transcript_section(self._log_path, label, content)
        self._parts.append(render_transcript_section(label, content))
        if self._on_update is not None:
            self._on_update(self.rendered())

    def append_model_response_meta(
        self,
        *,
        finish_reason: str,
        reasoning: str,
        content: str,
        tool_calls: list[dict[str, Any]],
        tool_call_markup_in_text: bool,
        recovered_tool_calls_from_markup: bool,
        malformed_argument_errors: list[str],
    ) -> None:
        if not self._enabled:
            return
        self.append(
            "MODEL RESPONSE META",
            render_model_response_meta(
                finish_reason=finish_reason,
                reasoning=reasoning,
                content=content,
                tool_calls=tool_calls,
                tool_call_markup_in_text=tool_call_markup_in_text,
                recovered_tool_calls_from_markup=recovered_tool_calls_from_markup,
                malformed_argument_errors=malformed_argument_errors,
            ),
        )

    def append_tool_call(self, tool_name: str, raw_arguments: Any, arguments: dict[str, Any]) -> None:
        """Log the call as the model wrote it; fall back to the parsed
        arguments when the raw form does not render."""
        if not self._enabled:
            return
        rendered_tool_call = render_tool_call_markup(tool_name, raw_arguments)
        self.append(
            f"TOOL CALL: {tool_name}",
            rendered_tool_call or (json.dumps(arguments, indent=2) if arguments else "{}"),
        )

    def append_system_message(self, content: str) -> None:
        if not self._enabled:
            return
        self.append("SYSTEM PROMPT", content)

    def append_user_message(self, content: Any) -> None:
        """Log the user message as the model receives it, image placeholder included."""
        if not self._enabled:
            return
        self.append("USER PROMPT", _render_verbatim_message_text(content))

    def append_tool_result(self, tool_name: str, content: str) -> None:
        """Log the tool result string exactly as it is sent back to the model."""
        if not self._enabled:
            return
        self.append(f"TOOL RESULT: {tool_name}", content)

    def rendered(self) -> str:
        return "".join(self._parts)

    def record_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | None,
        request_index: int,
    ) -> None:
        """Snapshot the request as sent (JSON round trip, so later mutation of
        the live message list cannot leak into the logs)."""
        if not self._enabled:
            return
        self._request_messages = json.loads(json.dumps(messages))
        self._request_tools = json.loads(json.dumps(tools))
        self._request_tool_choice = tool_choice
        self._request_index = request_index

    def write_prompt_snapshot(self) -> None:
        if not self._enabled:
            return
        if self._request_messages is None:
            return
        _write_prompt_log_snapshot(
            self._prompt_log_path,
            model_id=self._model_id,
            base_url=self._base_url,
            display_action_num=self._display_action_num,
            analysis_step=self._analysis_step,
            request_index=self._request_index,
            messages=self._request_messages,
            tools=self._request_tools,
            tool_choice=self._request_tool_choice,
            transcript=self.rendered(),
        )

    def append_request_snapshot(self, *, event: str, finish_reason: str | None = None) -> None:
        if not self._enabled:
            return
        if self._request_log_path is None or self._request_messages is None:
            return
        _append_request_snapshot(
            self._request_log_path,
            messages=self._request_messages,
            tools=self._request_tools,
            event=event,
            tool_choice=self._request_tool_choice,
            finish_reason=finish_reason,
            analysis_step=self._analysis_step,
            analysis_attempt=self._analysis_attempt,
            action=self._display_action_num,
            request_index_within_turn=self._request_index,
        )


def write_transcript_html(transcript_path: Path, html_path: Path, title: str) -> None:
    if not transcript_path.exists():
        return
    html_path.parent.mkdir(parents=True, exist_ok=True)
    text = transcript_path.read_text(encoding="utf-8")
    body = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
        "padding:20px;max-width:1100px;margin:0 auto;line-height:1.4;}"
        "h1{color:#fff;}pre{white-space:pre-wrap;background:#111;padding:16px;border-radius:6px;"
        "border:1px solid #333;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        "</body></html>\n"
    )
    html_path.write_text(body, encoding="utf-8")
