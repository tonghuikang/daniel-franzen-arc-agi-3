"""Helpers for provider-specific OpenAI-compatible requests."""
from __future__ import annotations

from typing import Any


def normalize_provider(value: str | None) -> str:
    provider = str(value or "").strip().lower()
    if provider in {"", "openai", "openai-compatible", "compat"}:
        return "vllm"
    if provider in {"openrouter", "router"}:
        return "openrouter"
    return provider


def build_headers(
    *,
    provider: str,
    api_key: str,
    referer: str = "",
    title: str = "",
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    normalized_provider = normalize_provider(provider)
    if normalized_provider == "openrouter":
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    return headers


def build_chat_payload(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    seed: int | None = None,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    if top_k > 0:
        payload["top_k"] = top_k
    if seed is not None and seed >= 0:
        payload["seed"] = seed
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    normalized_provider = normalize_provider(provider)
    if normalized_provider in {"vllm", "llama.cpp"}:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if reasoning_effort:
            payload["chat_template_kwargs"]["reasoning_effort"] = reasoning_effort
    if normalized_provider == "llama.cpp":
        payload["repeat_penalty"] = repetition_penalty

    return payload
