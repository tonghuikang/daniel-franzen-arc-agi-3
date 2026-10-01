"""Cheap local estimate of how many tokens a request payload will cost.

Used to decide when history must be compacted, so it has to run on every turn
without calling the server's tokenizer. Text is approximated from its rendered
length; image parts are priced from their PNG dimensions instead of the length
of their base64 payload, which would otherwise dwarf the real vision cost.

Text is priced with one ratio per kind of content, because a request carries several
that tokenize at very different rates. Measured against the model's own tokenizer over
the transcripts of run 20260907_213605_91c5dca, tool results average 2.1 rendered
characters per token, user turns and tool-call code average 3.2, and the system prompt
averages 4.36. Reasoning was priced with the user turns until it was measured on its
own and came out denser and far more variable than they are; it now carries its own
ratio, and `_extract_reasoning_texts` records what it was measured at.

The ratios below sit under those averages on purpose. A whole request averages its
messages, but the spread survives: at the measured averages the estimate lands between
0.88 and 1.04 of the true count, and the trim gate reads it as a stand-in for the
server's own count, so an undershoot spends the generation headroom that keeps a long
reply from being truncated. Tool results carry that spread, from 1.2 to 5.7 characters
per token depending on how dense the dump is. Set here at roughly the fifth percentile
of the per-request ratio, the estimate is an upper bound for 95 percent of requests and
runs about 10 percent high on a median one.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any, cast

_ESTIMATED_CHARS_PER_TOKEN = 2.9
_ESTIMATED_TOOL_RESULT_CHARS_PER_TOKEN = 1.8
_ESTIMATED_REASONING_CHARS_PER_TOKEN = 2.5
_ESTIMATED_SYSTEM_PROMPT_CHARS_PER_TOKEN = 4.3
_REASONING_FIELDS = ("reasoning", "reasoning_content")
_VISION_PATCH_PIXELS = 32
_DEFAULT_IMAGE_PART_TOKENS = 96


def _png_data_url_dimensions(url: str) -> tuple[int, int] | None:
    marker = "base64,"
    marker_index = url.find(marker)
    if marker_index < 0:
        return None
    header_start = marker_index + len(marker)
    try:
        header = base64.b64decode(url[header_start:header_start + 64], validate=False)
    except (ValueError, binascii.Error):
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        return None

    return width, height


def _estimate_image_part_tokens(part: dict[Any, Any]) -> int:
    """Vision tokens for one image part, not the length of its base64 payload."""
    image_url: Any = part.get("image_url")
    url = str(image_url.get("url", "")) if isinstance(image_url, dict) else ""
    dimensions = _png_data_url_dimensions(url)
    if dimensions is None:
        return _DEFAULT_IMAGE_PART_TOKENS
    width, height = dimensions
    patch_columns = max(1, round(width / _VISION_PATCH_PIXELS))
    patch_rows = max(1, round(height / _VISION_PATCH_PIXELS))

    return patch_columns * patch_rows


def _replace_image_parts(value: Any) -> tuple[Any, int]:
    """Strip base64 image payloads out of `value`, returning their token cost separately."""
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        if mapping.get("type") == "image_url":
            return {"type": "image_url"}, _estimate_image_part_tokens(mapping)
        replaced_mapping: dict[Any, Any] = {}
        image_tokens = 0
        for key, item in mapping.items():
            replaced_mapping[key], item_tokens = _replace_image_parts(item)
            image_tokens += item_tokens
        return replaced_mapping, image_tokens
    if isinstance(value, list):
        items = cast(list[Any], value)
        replaced_items: list[Any] = []
        image_tokens = 0
        for item in items:
            replaced_item, item_tokens = _replace_image_parts(item)
            replaced_items.append(replaced_item)
            image_tokens += item_tokens
        return replaced_items, image_tokens
    return value, 0


def _extract_messages_with_role(value: Any, role: str, extracted: list[Any]) -> Any:
    """Move the messages whose role is `role` out of `value` into `extracted`.

    Run this after `_replace_image_parts`, so that an image reached through one of those
    messages is still priced from its dimensions rather than from the length of its base64
    payload.

    Such a message reached through a dict value rather than a list is replaced by `None`
    and costs the four characters it renders as. A list entry that is already `None` is
    dropped instead, which the message lists this runs on never hold.
    """
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        if str(mapping.get("role", "")).strip() == role:
            extracted.append(mapping)
            return None
        replaced_mapping: dict[Any, Any] = {}
        for key, item in mapping.items():
            replaced_mapping[key] = _extract_messages_with_role(item, role, extracted)
        return replaced_mapping
    if isinstance(value, list):
        replaced_items: list[Any] = []
        for item in cast(list[Any], value):
            replaced_item = _extract_messages_with_role(item, role, extracted)
            if replaced_item is not None:
                replaced_items.append(replaced_item)
        return replaced_items
    return value


def _extract_tool_messages(value: Any, tool_messages: list[Any]) -> Any:
    """Move the tool results out of `value`, so they can be priced at their own ratio."""

    return _extract_messages_with_role(value, "tool", tool_messages)


def _extract_system_messages(value: Any, system_messages: list[Any]) -> Any:
    """Move the system prompt out of `value`, so it can be priced at its own ratio."""

    return _extract_messages_with_role(value, "system", system_messages)


def _extract_reasoning_texts(value: Any, reasoning_texts: list[str]) -> Any:
    """Move the assistant reasoning out of `value`, so it can be priced at its own ratio.

    Every assistant message the analyzer appends echoes the model's own reasoning back, so
    by mid-game it is the largest block of a request and the one whose density moves the
    estimate most. It is nothing like the English prose the general ratio was fitted to.
    The same tokenizer puts plain prose at 4.19 characters per token and a passage of board
    rows and coordinates at 2.20; reasoning about a board sits between, pooling to 3.03,
    3.02 and 2.75 over the 1638 reasoning blocks of runs 20260908_222615_9f3886a,
    20260909_182039_d910bac and 20260909_204728_6ef8283, with a per-block fifth percentile
    near 2.2 and a median near 3.1.

    The ratio is set from whole requests rather than from single blocks, since a request
    holds many and averages their spread away. Swept over the 502 prompt snapshots this
    repository has kept, the share of requests the estimate undershoots runs 23 percent at
    2.9, 7.8 at 2.6, 5.8 at 2.5, 2.0 at 2.3 and 1.4 at 2.2, while the median request goes
    from 6 percent high to 20. At 2.5 the estimate covers 94 percent of requests and runs
    13 percent high on a median one, which is as close as one number comes to the two aims
    stated above. Run this after the tool and system messages are out, so their own
    reasoning-shaped fields, if a server ever sends one, are already priced.
    """
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        replaced_mapping: dict[Any, Any] = {}
        for key, item in mapping.items():
            if key in _REASONING_FIELDS and isinstance(item, str):
                reasoning_texts.append(item)
                continue
            replaced_mapping[key] = _extract_reasoning_texts(item, reasoning_texts)

        return replaced_mapping
    if isinstance(value, list):
        return [_extract_reasoning_texts(item, reasoning_texts) for item in cast(list[Any], value)]

    return value


def _render_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def estimate_tokens(value: Any) -> int:
    stripped, image_tokens = _replace_image_parts(value)
    tool_messages: list[Any] = []
    system_messages: list[Any] = []
    reasoning_texts: list[str] = []
    remaining_value = _extract_tool_messages(stripped, tool_messages)
    remaining_value = _extract_system_messages(remaining_value, system_messages)
    remaining_value = _extract_reasoning_texts(remaining_value, reasoning_texts)
    text_tokens = math.ceil(len(_render_json(remaining_value)) / _ESTIMATED_CHARS_PER_TOKEN)
    if tool_messages:
        text_tokens += math.ceil(len(_render_json(tool_messages)) / _ESTIMATED_TOOL_RESULT_CHARS_PER_TOKEN)
    if system_messages:
        text_tokens += math.ceil(len(_render_json(system_messages)) / _ESTIMATED_SYSTEM_PROMPT_CHARS_PER_TOKEN)
    if reasoning_texts:
        reasoning_chars = sum(len(text) for text in reasoning_texts)
        text_tokens += math.ceil(reasoning_chars / _ESTIMATED_REASONING_CHARS_PER_TOKEN)

    return max(1, text_tokens + image_tokens)
