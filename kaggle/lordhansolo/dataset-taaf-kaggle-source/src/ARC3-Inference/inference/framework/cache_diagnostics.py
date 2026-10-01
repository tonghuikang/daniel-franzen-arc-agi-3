"""Test-only cache tracing, also installed as a standalone vLLM general plugin."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

_ORIGINAL_LOOKUP = None
_ORIGINAL_EVICT = None
_ORIGINAL_SCHEDULE = None
_FAILED = False

ENGINE_STEP_COLUMNS = (
    "monotonic", "running", "decode_requests", "decode_tokens",
    "prefill_requests", "prefill_tokens", "context_tokens",
)
ENGINE_STEP_BATCH = 50
_ENGINE_STEP_ROWS: list[list[float]] = []


def is_test_run(environ=None) -> bool:
    environment = os.environ if environ is None else environ

    return (
        environment.get("TAAF_RUN_AS_SUBMISSION", "").strip().lower() == "0"
        and environment.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
        in {"", "0", "false"}
    )


def is_cache_diagnostics_enabled() -> bool:
    return is_test_run() and os.environ.get("ARC3_CACHE_DIAGNOSTICS") == "1"


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def encode_hash(value) -> str:
    return value.hex() if isinstance(value, bytes) else str(value)


def append_cache_record(event: str, **fields) -> None:
    global _FAILED
    if _FAILED or not is_cache_diagnostics_enabled():
        return
    try:
        directory = Path(os.environ["ARC3_CACHE_DIAGNOSTICS_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"vllm_cache_diagnostics_{os.getpid()}.jsonl"
        record = {"event": event, "ts": time.time(), "pid": os.getpid(), **fields}
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        _FAILED = True
        logging.getLogger(__name__).exception("Cache diagnostic writer disabled")


def summarize_group_blocks(coordinator) -> list[dict]:
    """Count the pool blocks every KV cache group holds at this moment.

    This is what separates a group's share of the fixed per-request cost from the
    part that grows with context, which the lookup's own `free_blocks` only gives
    as a total. Align-mode Mamba groups leave a null placeholder wherever a retired
    state block used to sit, and several requests share one cached prefix block, so
    both are collapsed by counting distinct non-null block ids rather than the
    length of each request's block list. Summed over the groups these counts and
    `free_blocks` have to account for `total_blocks`.
    """
    summary = []
    for group_id, group_manager in enumerate(getattr(coordinator, "single_type_managers", ())):
        block_ids = set()
        requests = 0
        for blocks in group_manager.req_to_blocks.values():
            held = {block.block_id for block in blocks if not block.is_null}
            block_ids |= held
            requests += bool(held)
        summary.append({
            "group": group_id, "kind": type(group_manager.kv_cache_spec).__name__,
            "block_size": group_manager.block_size, "requests": requests,
            "blocks": len(block_ids),
        })

    return summary


def record_cache_lookup(manager, request, result) -> None:
    started = time.monotonic()
    coordinator = manager.coordinator
    per_group = []
    if manager.prefix_cache_lookup_enabled(request):
        for spec, group_ids, manager_class, use_eagle in getattr(coordinator, "attention_groups", ()):
            for drop_eagle in (False, True) if use_eagle else (False,):
                _, hit = manager_class.find_longest_cache_hit(
                    block_hashes=request.block_hashes,
                    max_length=request.num_tokens - 1,
                    kv_cache_group_ids=group_ids,
                    block_pool=manager.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle,
                    alignment_tokens=coordinator._cache_hit_alignment_tokens,
                )
                per_group.append({
                    "groups": group_ids, "kind": type(spec).__name__,
                    "block_size": spec.block_size, "drop_eagle": drop_eagle,
                    "hit_tokens": hit,
                })
    tokens = request.prompt_token_ids or []
    token_chunks = None
    if not getattr(request, "_arc3_cache_prompt_logged", False):
        token_chunks = [hash_json(tokens[offset:offset + 256]) for offset in range(0, len(tokens), 256)]
        request._arc3_cache_prompt_logged = True
    client_request = re.match(r"chatcmpl-arc3-[0-9a-f]{32}", request.request_id)
    append_cache_record(
        "cache_lookup", request_id=request.request_id,
        client_request_id=client_request.group(0) if client_request else None,
        prompt_tokens=len(tokens), computed_tokens=request.num_computed_tokens,
        hit_tokens=result[1], shared_prefix_boundary=result[2],
        hash_block_size=manager.block_pool.hash_block_size,
        block_hashes=[encode_hash(value) for value in request.block_hashes],
        token_chunk_size=256,
        token_chunk_hashes=token_chunks,
        per_group=per_group, kv_usage=manager.block_pool.get_usage(),
        free_blocks=manager.block_pool.get_num_free_blocks(),
        total_blocks=manager.block_pool.num_gpu_blocks,
        group_blocks=summarize_group_blocks(coordinator),
        prefix_caching_enabled=manager.enable_caching,
        retention_interval=getattr(coordinator, "retention_interval", None),
        diagnostic_seconds_before_write=time.monotonic() - started,
    )


def trace_cache_lookup(manager, request):
    global _FAILED
    result = _ORIGINAL_LOOKUP(manager, request)
    if is_cache_diagnostics_enabled() and not _FAILED:
        try:
            record_cache_lookup(manager, request, result)
        except Exception:
            _FAILED = True
            logging.getLogger(__name__).exception("Cache lookup diagnostic failed")

    return result


def trace_cache_eviction(pool, block):
    global _FAILED
    hashes = []
    if is_cache_diagnostics_enabled() and not _FAILED:
        try:
            from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id

            block_hashes = []
            if block.block_hash is not None:
                block_hashes.append(block.block_hash)
            block_hashes.extend(pool.cached_block_hashes_by_block.get(block.block_id, ()))
            hashes = [
                {"hash": encode_hash(get_block_hash(value)), "group": get_group_id(value)}
                for value in block_hashes
            ]
        except Exception:
            _FAILED = True
            logging.getLogger(__name__).exception("Cache eviction diagnostic failed")
    result = _ORIGINAL_EVICT(pool, block)
    if result and hashes:
        append_cache_record("cache_eviction", block_id=block.block_id, hashes=hashes)

    return result


def summarize_engine_step(scheduler, scheduler_output) -> list[float]:
    """Describe one engine step with the batch composition its cost scales on.

    A request is prefilling when the step started below the end of its prompt,
    which also covers a final chunk as narrow as a decode. Context tokens sum
    the computed tokens of every scheduled request and stand for the KV the step
    reads, counted after the scheduler advanced them past this step.
    """
    decode_requests = 0
    decode_tokens = 0
    prefill_requests = 0
    prefill_tokens = 0
    context_tokens = 0
    for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests[request_id]
        context_tokens += request.num_computed_tokens
        if request.num_computed_tokens - scheduled < request.num_prompt_tokens:
            prefill_requests += 1
            prefill_tokens += scheduled
        else:
            decode_requests += 1
            decode_tokens += scheduled

    return [
        round(time.monotonic(), 6), len(scheduler.running),
        decode_requests, decode_tokens, prefill_requests, prefill_tokens, context_tokens,
    ]


def flush_engine_steps() -> None:
    """Write the buffered steps, stamping the record right after the last row.

    Nothing flushes a partial batch, so the batch bounds how much of a run's
    tail a crash or a kill takes with it. The record's own wall-clock `ts`
    belongs to its last row, which is what anchors these monotonic timestamps
    against the wall-clock records of cache lookups.
    """
    if not _ENGINE_STEP_ROWS:
        return
    rows = list(_ENGINE_STEP_ROWS)
    _ENGINE_STEP_ROWS.clear()
    append_cache_record("engine_steps", columns=list(ENGINE_STEP_COLUMNS), rows=rows)


def trace_engine_step(scheduler, *args, **kwargs):
    """Time steps by their spacing, which is what the engine sustains under load."""
    global _FAILED
    scheduler_output = _ORIGINAL_SCHEDULE(scheduler, *args, **kwargs)
    if (
        is_cache_diagnostics_enabled()
        and not _FAILED
        and scheduler_output.total_num_scheduled_tokens > 0
    ):
        try:
            _ENGINE_STEP_ROWS.append(summarize_engine_step(scheduler, scheduler_output))
            if len(_ENGINE_STEP_ROWS) >= ENGINE_STEP_BATCH:
                flush_engine_steps()
        except Exception:
            _FAILED = True
            logging.getLogger(__name__).exception("Engine step diagnostic failed")

    return scheduler_output


def install_cache_diagnostics() -> None:
    """Leave competition processes untouched, even if the plugin is present."""
    global _ORIGINAL_LOOKUP, _ORIGINAL_EVICT, _ORIGINAL_SCHEDULE
    if not is_cache_diagnostics_enabled() or _ORIGINAL_LOOKUP is not None:
        return
    try:
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.scheduler import Scheduler

        _ORIGINAL_LOOKUP = KVCacheManager.get_computed_blocks
        _ORIGINAL_EVICT = BlockPool._maybe_evict_cached_block
        _ORIGINAL_SCHEDULE = Scheduler.schedule
        KVCacheManager.get_computed_blocks = trace_cache_lookup
        BlockPool._maybe_evict_cached_block = trace_cache_eviction
        Scheduler.schedule = trace_engine_step
        append_cache_record("cache_diagnostics_installed")
    except Exception:
        logging.getLogger(__name__).exception("Cache diagnostics could not be installed")
