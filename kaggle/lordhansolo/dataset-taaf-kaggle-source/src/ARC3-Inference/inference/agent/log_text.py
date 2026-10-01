"""Text truncation shared by the human-readable logs and the metrics JSONL."""
from __future__ import annotations


def trim_log_text(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    omitted = len(stripped) - max_chars

    return f"{stripped[:max_chars].rstrip()}\n... [truncated {omitted} chars]"
