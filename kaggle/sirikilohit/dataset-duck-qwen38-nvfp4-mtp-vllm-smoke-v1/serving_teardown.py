"""Capture final vLLM evidence and stop only processes owned by this run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

EXPECTED_WORKING_DIR = Path("/kaggle/working")
RUNTIME_ROOT = Path("/tmp/qwen38-flash-next-vllm-runtime")
SYSTEM_CACHE_ROOTS = (
    Path("/tmp/qwen38-flash-next-vllm-cache"),
    Path("/tmp/qwen38-flash-next-vllm-compile-cache"),
    Path("/tmp/qwen38-flash-next-vllm-tmp"),
)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1234
ENDPOINT_TIMEOUT_SECONDS = 1.0
TERM_GRACE_SECONDS = 3.0
KILL_GRACE_SECONDS = 2.0
PORT_CLOSE_GRACE_SECONDS = 1.0
GPU_QUERY_TIMEOUT_SECONDS = 3.0
PROCESS_POLL_SECONDS = 0.1
MAX_ENDPOINT_BYTES = 16 * 1024 * 1024
MAX_REQUIRED_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_FULL_LOG_BYTES = 64 * 1024 * 1024
LOG_SAMPLE_BYTES = 4 * 1024 * 1024
MAX_EXPLICIT_WAIT_SECONDS = (
    (2 * ENDPOINT_TIMEOUT_SECONDS)
    + TERM_GRACE_SECONDS
    + KILL_GRACE_SECONDS
    + PORT_CLOSE_GRACE_SECONDS
    + GPU_QUERY_TIMEOUT_SECONDS
    + (3 * PROCESS_POLL_SECONDS)
)
IDENTITY_SCHEMA_VERSION = 1
IDENTITY_REQUIRED_FIELDS = {
    "schema_version",
    "backend",
    "boot_id",
    "host",
    "port",
    "pid",
    "start_ticks",
    "pgid",
    "sid",
    "argv",
    "argv_sha256",
    "vllm_tuning",
    "started_epoch",
    "phase",
    "snapshot_epoch",
    "ready_epoch",
    "workers",
}
IDENTITY_ALLOWED_FIELDS = IDENTITY_REQUIRED_FIELDS
VLLM_TUNING_FIELDS = {
    "schema_version",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "kv_cache_dtype",
    "kv_cache_memory_bytes",
    "max_cudagraph_capture_size",
    "max_num_batched_tokens",
    "max_num_batched_tokens_requested",
    "max_num_seqs",
    "moe_backend",
    "mtp_dynamic_batch_schedule",
    "mtp_index_share_for_iteration",
    "mtp_speculative_tokens",
    "omp_num_threads",
    "overridden",
}
VLLM_TUNING_OVERRIDE_FIELDS = {
    "TAAF_VLLM_MAX_NUM_SEQS",
    "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS",
    "TAAF_VLLM_OMP_THREADS",
    "TAAF_VLLM_MTP_TOKENS",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES",
    "TAAF_VLLM_KV_CACHE_DTYPE",
    "TAAF_VLLM_ENABLE_PREFIX_CACHING",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE",
    "TAAF_VLLM_MOE_BACKEND",
    "TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION",
    "TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE",
}
MTP_DYNAMIC_BATCH_SCHEDULE_ENV = "TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE"
MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES = 96 * 1024**3
MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES = 64 * 1024**2
WORKER_FIELDS = {
    "pid",
    "start_ticks",
    "ppid",
    "pgid",
    "sid",
    "cmdline_sha256",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TORCH_SHM_MANAGER_CMDLINE_SHA256 = (
    "fd3a41c6faf7553a76d60949dea5632c423de414c9c0e9494171ea942543f428"
)


def working_paths(working_dir: Path) -> dict[str, Path | tuple[Path, ...]]:
    return {
        "server_log": working_dir / "vllm-openai-server.log",
        "server_identity": working_dir / "vllm-server-identity.json",
        "result": working_dir / "vllm-server-teardown.json",
        "final_metrics": working_dir / "vllm-metrics-final.prom",
        "final_models": working_dir / "vllm-models-final.json",
        "score": working_dir / "score.json",
        "submission": working_dir / "submission.parquet",
        "cache_roots": (
            working_dir / "vllm-cache",
            working_dir / "vllm-compile-cache",
        ),
    }


def validate_working_dir() -> Path:
    raw = os.environ.get("TAAF_KAGGLE_WORKING_DIR", "")
    if not raw:
        raise RuntimeError("TAAF_KAGGLE_WORKING_DIR is missing")
    supplied = Path(raw).resolve(strict=True)
    expected = EXPECTED_WORKING_DIR.resolve(strict=True)
    if supplied != expected:
        raise RuntimeError(f"wrong Kaggle working root: {supplied} != {expected}")
    if not supplied.is_dir():
        raise RuntimeError(f"Kaggle working root is not a directory: {supplied}")
    return supplied


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def argv_sha256(argv: list[str]) -> str:
    encoded = json.dumps(
        argv, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def proc_stat_record(pid: int) -> dict[str, Any] | None:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw_stat.rfind(")")
        if close < 0:
            return None
        fields = raw_stat[close + 2 :].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def proc_record(pid: int, *, read_environ: bool = True) -> dict[str, Any] | None:
    try:
        stat = proc_stat_record(pid)
        if stat is None:
            return None
        raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        raw_environ = b""
        if read_environ:
            try:
                raw_environ = Path(f"/proc/{pid}/environ").read_bytes()
            except (FileNotFoundError, OSError):
                raw_environ = b""
        argv = [
            item.decode("utf-8", errors="replace")
            for item in raw_cmdline.rstrip(b"\x00").split(b"\x00")
            if item
        ]
        return {
            **stat,
            "comm": Path(f"/proc/{pid}/comm")
            .read_text(encoding="utf-8", errors="replace")
            .strip(),
            "argv": argv,
            "cmdline": " ".join(argv),
            "cmdline_sha256": sha256_bytes(raw_cmdline),
            "server_env_marker": all(
                marker in raw_environ
                for marker in (
                    b"VLLM_PLE_CPU_OFFLOAD=1\x00",
                    b"VLLM_WORKER_MULTIPROC_METHOD=spawn\x00",
                    b"VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8=1\x00",
                )
            ),
        }
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def proc_stat_table() -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        record = proc_stat_record(int(item.name))
        if record is not None:
            records[int(item.name)] = record
    return records


def descendant_ids_many(
    roots: set[int], records: dict[int, dict[str, Any]]
) -> set[int]:
    found = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if int(record["ppid"]) in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def owned_process_table(identity: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Read fingerprints only for saved, descendant, session, or group candidates."""
    stats = proc_stat_table()
    seeds = {int(identity["pid"])} | {
        int(worker["pid"]) for worker in identity["workers"]
    }
    candidates = descendant_ids_many(seeds, stats)
    candidates.update(
        pid
        for pid, record in stats.items()
        if int(record["sid"]) == int(identity["sid"])
        or int(record["pgid"]) == int(identity["pgid"])
    )
    records: dict[int, dict[str, Any]] = {}
    for pid in sorted(candidates):
        record = proc_record(pid)
        if record is not None:
            records[pid] = record
    return records


def global_marker_records() -> list[dict[str, Any]]:
    """Run one light final marker pass without reading every process environment."""
    matches: list[dict[str, Any]] = []
    for pid in sorted(proc_stat_table()):
        if pid == os.getpid():
            continue
        record = proc_record(pid, read_environ=False)
        if record is None or record["state"] == "Z":
            continue
        reasons = marker_reasons(record)
        if not reasons and (
            "python" in str(record.get("comm", "")).lower()
            or "spawn_main" in str(record.get("cmdline", "")).lower()
        ):
            enriched = proc_record(pid)
            if enriched is not None:
                record = enriched
                reasons = marker_reasons(record)
        if reasons:
            matches.append({**public_record(record), "marker_reasons": reasons})
    return matches


def exact_signal_match(record: dict[str, Any], saved: dict[str, Any]) -> bool:
    return (
        record["state"] != "Z"
        and int(record["pid"]) == int(saved["pid"])
        and int(record["start_ticks"]) == int(saved["start_ticks"])
        and int(record["pgid"]) == int(saved["pgid"])
        and int(record["sid"]) == int(saved["sid"])
        and record["cmdline_sha256"] == saved["cmdline_sha256"]
    )


def allowed_detached_worker(
    record: dict[str, Any],
    workers_by_pid: dict[int, dict[str, Any]],
    root_pgid: int,
    root_sid: int,
) -> bool:
    pid = int(record.get("pid", -1))
    parent = workers_by_pid.get(int(record.get("ppid", -1)))
    return (
        pid > 1
        and parent is not None
        and int(parent.get("pgid", -1)) == root_pgid
        and int(parent.get("sid", -1)) == root_sid
        and int(record.get("pgid", -1)) == pid
        and int(record.get("sid", -1)) == pid
        and record.get("cmdline_sha256") == TORCH_SHM_MANAGER_CMDLINE_SHA256
    )


def descendant_ids(root_pid: int, records: dict[int, dict[str, Any]]) -> set[int]:
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if record["ppid"] in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: record[key]
        for key in (
            "pid",
            "state",
            "ppid",
            "pgid",
            "sid",
            "start_ticks",
            "comm",
            "cmdline",
            "cmdline_sha256",
        )
    }
    result["server_env_marker"] = bool(record.get("server_env_marker", False))
    return result


def validate_identity_document(
    value: Any, boot_id: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["identity_is_not_an_object"]
    keys = set(value)
    missing = sorted(IDENTITY_REQUIRED_FIELDS - keys)
    unknown = sorted(keys - IDENTITY_ALLOWED_FIELDS)
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
    if unknown:
        errors.append(f"unknown_fields:{','.join(unknown)}")
    if errors:
        return None, errors

    errors.extend(validate_vllm_tuning_document(value["vllm_tuning"]))

    if type(value["schema_version"]) is not int or value["schema_version"] != IDENTITY_SCHEMA_VERSION:
        errors.append("wrong_schema_version")
    if value["backend"] != "vllm":
        errors.append("wrong_backend")
    if value["boot_id"] != boot_id:
        errors.append("boot_id_mismatch")
    if value["host"] != SERVER_HOST:
        errors.append("host_mismatch")
    if type(value["port"]) is not int or value["port"] != SERVER_PORT:
        errors.append("port_mismatch")
    for field in ("pid", "start_ticks", "pgid", "sid"):
        if type(value[field]) is not int or value[field] <= 0:
            errors.append(f"invalid_{field}")
    if (
        type(value["pid"]) is int
        and type(value["pgid"]) is int
        and type(value["sid"]) is int
        and (value["pgid"] != value["pid"] or value["sid"] != value["pid"])
    ):
        errors.append("root_is_not_its_own_session_and_group_leader")
    if type(value["started_epoch"]) not in (int, float) or value["started_epoch"] <= 0:
        errors.append("invalid_started_epoch")
    if value["phase"] not in ("starting", "ready"):
        errors.append("invalid_phase")
    if type(value["snapshot_epoch"]) not in (int, float) or value["snapshot_epoch"] <= 0:
        errors.append("invalid_snapshot_epoch")
    if (
        type(value["started_epoch"]) in (int, float)
        and type(value["snapshot_epoch"]) in (int, float)
        and value["snapshot_epoch"] < value["started_epoch"]
    ):
        errors.append("snapshot_precedes_start")
    if value["phase"] == "starting":
        if value["ready_epoch"] is not None:
            errors.append("starting_phase_has_ready_epoch")
    elif value["phase"] == "ready":
        if type(value["ready_epoch"]) not in (int, float) or value["ready_epoch"] <= 0:
            errors.append("invalid_ready_epoch")
        elif (
            type(value["started_epoch"]) in (int, float)
            and value["ready_epoch"] < value["started_epoch"]
        ):
            errors.append("ready_precedes_start")
        elif (
            type(value["snapshot_epoch"]) in (int, float)
            and value["ready_epoch"] > value["snapshot_epoch"]
        ):
            errors.append("ready_follows_snapshot")

    argv = value["argv"]
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        errors.append("invalid_argv")
    else:
        if not any("vllm" in item.lower() for item in argv):
            errors.append("argv_is_not_vllm")
        if value["argv_sha256"] != argv_sha256(argv):
            errors.append("argv_sha256_mismatch")
    if not isinstance(value["argv_sha256"], str) or not SHA256_PATTERN.fullmatch(value["argv_sha256"]):
        errors.append("invalid_argv_sha256")

    workers = value["workers"]
    if not isinstance(workers, list) or not workers:
        errors.append("missing_workers")
    else:
        seen: set[int] = set()
        for index, worker in enumerate(workers):
            prefix = f"worker_{index}"
            if not isinstance(worker, dict) or set(worker) != WORKER_FIELDS:
                errors.append(f"{prefix}_schema_mismatch")
                continue
            for field in ("pid", "start_ticks", "ppid", "pgid", "sid"):
                if type(worker[field]) is not int or worker[field] < (0 if field == "ppid" else 1):
                    errors.append(f"{prefix}_invalid_{field}")
            fingerprint = worker["cmdline_sha256"]
            if not isinstance(fingerprint, str) or not SHA256_PATTERN.fullmatch(fingerprint):
                errors.append(f"{prefix}_invalid_cmdline_sha256")
            if type(worker.get("pid")) is int:
                if worker["pid"] in seen:
                    errors.append(f"{prefix}_duplicate_pid")
                seen.add(worker["pid"])
        roots = [
            worker
            for worker in workers
            if isinstance(worker, dict)
            and worker.get("pid") == value["pid"]
            and worker.get("start_ticks") == value["start_ticks"]
            and worker.get("pgid") == value["pgid"]
            and worker.get("sid") == value["sid"]
        ]
        if len(roots) != 1:
            errors.append("root_worker_identity_missing")
        if value["phase"] == "starting" and len(workers) != 1:
            errors.append("starting_phase_must_contain_only_root")
        workers_by_pid = {
            int(worker["pid"]): worker
            for worker in workers
            if isinstance(worker, dict) and type(worker.get("pid")) is int
        }
        if any(
            isinstance(worker, dict)
            and worker.get("sid") != value["sid"]
            and not allowed_detached_worker(
                worker, workers_by_pid, int(value["pgid"]), int(value["sid"])
            )
            for worker in workers
        ):
            errors.append("unrecognized_detached_worker")
    return (value if not errors else None), errors


def validate_vllm_tuning_document(value: Any) -> list[str]:
    """Validate persisted tuning without trusting it for process ownership."""
    if not isinstance(value, dict):
        return ["vllm_tuning_is_not_an_object"]
    errors: list[str] = []
    missing = sorted(VLLM_TUNING_FIELDS - set(value))
    unknown = sorted(set(value) - VLLM_TUNING_FIELDS)
    if missing:
        errors.append(f"vllm_tuning_missing_fields:{','.join(missing)}")
    if unknown:
        errors.append(f"vllm_tuning_unknown_fields:{','.join(unknown)}")
    if errors:
        return errors

    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        errors.append("vllm_tuning_wrong_schema_version")
    for field in (
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "mtp_index_share_for_iteration",
    ):
        if type(value[field]) is not bool:
            errors.append(f"vllm_tuning_invalid_{field}")
    if not isinstance(value["kv_cache_dtype"], str) or value[
        "kv_cache_dtype"
    ] not in {"auto", "fp8"}:
        errors.append("vllm_tuning_invalid_kv_cache_dtype")
    if (
        value["moe_backend"] is not None
        and value["moe_backend"] != "flashinfer_b12x"
    ):
        errors.append("vllm_tuning_invalid_moe_backend")

    integer_bounds = {
        "kv_cache_memory_bytes": (0, MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES),
        "max_cudagraph_capture_size": (0, 256),
        "max_num_batched_tokens": (1, 1_048_576),
        "max_num_batched_tokens_requested": (1, 1_048_576),
        "max_num_seqs": (1, 256),
        "mtp_speculative_tokens": (0, 4),
        "omp_num_threads": (1, 64),
    }
    for field, (minimum, maximum) in integer_bounds.items():
        item = value[field]
        if type(item) is not int or not minimum <= item <= maximum:
            errors.append(f"vllm_tuning_invalid_{field}")
    kv_bytes = value["kv_cache_memory_bytes"]
    if type(kv_bytes) is int and 0 < kv_bytes < MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES:
        errors.append("vllm_tuning_invalid_kv_cache_memory_bytes")

    dynamic_schedule = value["mtp_dynamic_batch_schedule"]
    mtp_tokens = value["mtp_speculative_tokens"]
    if dynamic_schedule is not None:
        try:
            _validate_mtp_dynamic_batch_schedule(
                dynamic_schedule,
                max_speculative_tokens=(
                    mtp_tokens if type(mtp_tokens) is int and mtp_tokens > 0 else 0
                ),
            )
        except RuntimeError:
            errors.append("vllm_tuning_invalid_mtp_dynamic_batch_schedule")
    if (
        value["mtp_index_share_for_iteration"] is True
        and (type(mtp_tokens) is not int or mtp_tokens <= 0)
    ):
        errors.append("vllm_tuning_mtp_index_share_without_mtp")

    if (
        type(value["enable_chunked_prefill"]) is bool
        and type(value["max_num_batched_tokens"]) is int
        and type(value["max_num_batched_tokens_requested"]) is int
    ):
        if (
            value["enable_chunked_prefill"]
            and value["max_num_batched_tokens"]
            != value["max_num_batched_tokens_requested"]
        ):
            errors.append("vllm_tuning_chunked_batch_tokens_mismatch")
        if (
            not value["enable_chunked_prefill"]
            and value["max_num_batched_tokens"]
            < value["max_num_batched_tokens_requested"]
        ):
            errors.append("vllm_tuning_no_chunk_batch_tokens_mismatch")

    overridden = value["overridden"]
    if not isinstance(overridden, dict):
        errors.append("vllm_tuning_overridden_is_not_an_object")
    else:
        missing_overrides = sorted(VLLM_TUNING_OVERRIDE_FIELDS - set(overridden))
        unknown_overrides = sorted(set(overridden) - VLLM_TUNING_OVERRIDE_FIELDS)
        if missing_overrides:
            errors.append(
                "vllm_tuning_overridden_missing_fields:"
                + ",".join(missing_overrides)
            )
        if unknown_overrides:
            errors.append(
                "vllm_tuning_overridden_unknown_fields:"
                + ",".join(unknown_overrides)
            )
        if any(type(item) is not bool for item in overridden.values()):
            errors.append("vllm_tuning_overridden_values_are_not_boolean")
    return errors


def _validate_mtp_dynamic_batch_schedule(
    value: Any, *, max_speculative_tokens: int
) -> list[list[int]]:
    """Validate the exact dynamic-MTP identity shape written by setup."""
    if not isinstance(value, list) or not value:
        raise RuntimeError(
            f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} must be a non-empty JSON list."
        )
    resolved: list[list[int]] = []
    previous_maximum = 0
    for index, row in enumerate(value):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or any(type(item) is not int for item in row)
        ):
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] must contain exactly "
                "three integers: [minimum_batch_size, maximum_batch_size, "
                "num_speculative_tokens]."
            )
        minimum_batch_size, maximum_batch_size, speculative_tokens = row
        if minimum_batch_size < 1 or maximum_batch_size < minimum_batch_size:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] has an invalid batch "
                f"range: {row!r}."
            )
        if minimum_batch_size <= previous_maximum:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} batch ranges must be strictly "
                "increasing and non-overlapping."
            )
        if not 1 <= speculative_tokens <= max_speculative_tokens:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] speculative tokens "
                f"must be from 1 through {max_speculative_tokens}, found "
                f"{speculative_tokens}."
            )
        resolved.append(
            [minimum_batch_size, maximum_batch_size, speculative_tokens]
        )
        previous_maximum = maximum_batch_size
    return resolved


def marker_reasons(record: dict[str, Any]) -> list[str]:
    comm = str(record.get("comm", "")).lower()
    cmdline = str(record.get("cmdline", "")).lower()
    scrubbed = cmdline.replace("vllm_serving_teardown.py", "teardown.py")
    reasons: list[str] = []
    if comm.startswith("vllm::") or "vllm::" in comm:
        reasons.append("vllm_process_title")
    if "vllm.entrypoints" in scrubbed or re.search(r"(?:^|\s)vllm(?:\s+serve|$)", scrubbed):
        reasons.append("vllm_command")
    if str(RUNTIME_ROOT).lower() in scrubbed:
        reasons.append("vllm_runtime_root")
    if record.get("server_env_marker") is True:
        reasons.append("vllm_server_environment")
    if (
        comm == "torch_shm_manager"
        and record.get("cmdline_sha256") == TORCH_SHM_MANAGER_CMDLINE_SHA256
    ):
        reasons.append("torch_shm_manager")
    if any(marker in f"{comm} {scrubbed}" for marker in ("ple_offload", "pleworker", "ple_worker", "ple-worker")):
        reasons.append("ple_worker")
    return sorted(set(reasons))


def scan_ownership(
    identity: dict[str, Any], records: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    saved_by_pid = {int(item["pid"]): item for item in identity["workers"]}
    root = records.get(int(identity["pid"]))
    root_live = False
    root_conflict: dict[str, Any] | None = None
    if root is not None and root["state"] != "Z":
        saved_root = saved_by_pid[int(identity["pid"])]
        if (
            int(root["start_ticks"]) == int(identity["start_ticks"])
            and int(root["pgid"]) == int(identity["pgid"])
            and int(root["sid"]) == int(identity["sid"])
            and root["cmdline_sha256"] == saved_root["cmdline_sha256"]
        ):
            root_live = True
        else:
            root_conflict = public_record(root)

    saved_matches: dict[int, dict[str, Any]] = {}
    saved_conflicts: list[dict[str, Any]] = []
    saved_pid_reuse: list[dict[str, Any]] = []
    for pid, saved in saved_by_pid.items():
        record = records.get(pid)
        if record is None or record["state"] == "Z":
            continue
        if int(record["start_ticks"]) != int(saved["start_ticks"]):
            saved_pid_reuse.append(public_record(record))
            continue
        stable = (
            int(record["pgid"]) == int(saved["pgid"])
            and int(record["sid"]) == int(saved["sid"])
            and record["cmdline_sha256"] == saved["cmdline_sha256"]
        )
        if stable:
            saved_matches[pid] = record
        else:
            saved_conflicts.append(public_record(record))

    descendants: set[int] = set()
    if root_live:
        descendants = descendant_ids(int(identity["pid"]), records)
    session_matches = {
        pid: record
        for pid, record in records.items()
        if record["state"] != "Z"
        and int(record["sid"]) == int(identity["sid"])
    }
    group_only = {
        pid: record
        for pid, record in records.items()
        if record["state"] != "Z"
        and int(record["pgid"]) == int(identity["pgid"])
        and int(record["sid"]) != int(identity["sid"])
    }

    authorized: dict[int, dict[str, Any]] = {}
    if root_conflict is None and not saved_conflicts:
        for pid in descendants:
            if pid in records and records[pid]["state"] != "Z":
                authorized[pid] = records[pid]
        authorized.update(saved_matches)
        authorized.update(session_matches)
    authorized.pop(os.getpid(), None)

    marker_matches: dict[int, dict[str, Any]] = {}
    for pid, record in records.items():
        if pid == os.getpid() or record["state"] == "Z":
            continue
        reasons = marker_reasons(record)
        if reasons:
            marker_matches[pid] = {**public_record(record), "marker_reasons": reasons}
    marker_only = {
        pid: record for pid, record in marker_matches.items() if pid not in authorized
    }
    suspect = {
        **{pid: public_record(record) for pid, record in group_only.items()},
        **marker_only,
    }
    return {
        "root_live": root_live,
        "root_conflict": root_conflict,
        "saved_matches": [public_record(saved_matches[pid]) for pid in sorted(saved_matches)],
        "saved_conflicts": saved_conflicts,
        "saved_pid_reuse": saved_pid_reuse,
        "session_matches": [public_record(session_matches[pid]) for pid in sorted(session_matches)],
        "group_only_suspects": [public_record(group_only[pid]) for pid in sorted(group_only)],
        "authorized": authorized,
        "authorized_records": [public_record(authorized[pid]) for pid in sorted(authorized)],
        "marker_matches": [marker_matches[pid] for pid in sorted(marker_matches)],
        "marker_only_suspects": [marker_only[pid] for pid in sorted(marker_only)],
        "suspect_records": [suspect[pid] for pid in sorted(suspect)],
    }


def fetch(url: str, timeout: float = ENDPOINT_TIMEOUT_SECONDS) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    previous_handler: Any = None
    alarm_installed = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")

    def deadline_exceeded(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"endpoint deadline exceeded after {timeout:.3f}s")

    if alarm_installed:
        previous_handler = signal.signal(signal.SIGALRM, deadline_exceeded)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_ENDPOINT_BYTES + 1)
        if len(body) > MAX_ENDPOINT_BYTES:
            raise RuntimeError(
                f"endpoint body exceeds {MAX_ENDPOINT_BYTES} bytes: {url}"
            )
        return body
    finally:
        if alarm_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def port_open(timeout: float = PROCESS_POLL_SECONDS) -> bool:
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def gpu_rows() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=GPU_QUERY_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - return a bounded GPU query error
        return [{"query_error": repr(exc)}]
    if completed.returncode != 0:
        return [
            {
                "query_error": f"nvidia-smi exited {completed.returncode}",
                "stderr": completed.stderr[-4000:],
            }
        ]
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        record = proc_record(pid, read_environ=False)
        rows.append(
            {
                "pid": pid,
                "process_name": parts[1],
                "used_memory_mib": parts[2],
                "gpu_uuid": parts[3],
                "start_ticks": record["start_ticks"] if record else None,
                "comm": record["comm"] if record else "",
                "cmdline": record["cmdline"] if record else "",
            }
        )
    return rows


def count_log_markers(text: str) -> dict[str, int]:
    lowered = text.lower()
    patterns = {
        "http_4xx": r'(?:http/1\.[01]"\s+4\d\d\b|status(?:_code)?[=: ]+4\d\d\b)',
        "http_5xx": r'(?:http/1\.[01]"\s+5\d\d\b|status(?:_code)?[=: ]+5\d\d\b)',
        "parser_error": r"(?:parser error|parse error|failed to parse|invalid tool call|parser.{0,40}error|error.{0,40}parser)",
        "context_error": r"(?:context length|maximum model length|too many tokens|input too long|max_model_len)",
        "timeout": r"(?:timed out|timeouterror|timeout error)",
        "out_of_memory": r"(?:out of memory|outofmemoryerror|cuda oom|cudnn_status_alloc_failed)",
        "cuda_error": r"(?:cuda error|cudaerror)",
        "crash": r"(?:segmentation fault|engine core failed|engine dead|core dumped|fatal error)",
    }
    counts = {name: len(re.findall(pattern, lowered)) for name, pattern in patterns.items()}
    counts.update(
        {
            "traceback": lowered.count("traceback (most recent call last)"),
            "mtp": lowered.count("mtp"),
            "speculative": lowered.count("speculative"),
            "draft": lowered.count("draft"),
            "accepted": lowered.count("accepted"),
            "acceptance_rate": lowered.count("acceptance rate"),
            "ple": lowered.count("ple"),
            "nvfp4": lowered.count("nvfp4"),
            "fp8": lowered.count("fp8"),
            "prefix_cache": lowered.count("prefix cache"),
        }
    )
    return counts


def capture_log(path: Path, prefix: str, result: dict[str, Any]) -> None:
    if not path.exists():
        result[f"server_log_{prefix}_exists"] = False
        return
    size = path.stat().st_size
    if size <= MAX_FULL_LOG_BYTES:
        raw = path.read_bytes()
        result[f"server_log_{prefix}_sha256"] = sha256_bytes(raw)
        result[f"server_log_{prefix}_hash_scope"] = "full"
    else:
        with path.open("rb") as handle:
            prefix_bytes = handle.read(LOG_SAMPLE_BYTES)
            handle.seek(max(0, size - LOG_SAMPLE_BYTES))
            tail_bytes = handle.read(LOG_SAMPLE_BYTES)
        raw = prefix_bytes + b"\n<BOUNDED_LOG_SAMPLE>\n" + tail_bytes
        result[f"server_log_{prefix}_sha256"] = None
        result[f"server_log_{prefix}_hash_scope"] = "omitted_over_limit"
        result[f"server_log_{prefix}_prefix_sha256"] = sha256_bytes(prefix_bytes)
        result[f"server_log_{prefix}_tail_sha256"] = sha256_bytes(tail_bytes)
    text = raw.decode("utf-8", errors="replace")
    result[f"server_log_{prefix}_exists"] = True
    result[f"server_log_{prefix}_bytes"] = size
    result[f"server_log_{prefix}_scanned_bytes"] = len(raw)
    result[f"server_log_{prefix}_counts"] = count_log_markers(text)
    result[f"server_log_{prefix}_tail"] = "\n".join(text.splitlines()[-200:])


def capture_endpoints(
    metrics_path: Path, models_path: Path, result: dict[str, Any]
) -> None:
    metrics_tmp = metrics_path.with_name(f".{metrics_path.name}.tmp")
    models_tmp = models_path.with_name(f".{models_path.name}.tmp")
    metrics_tmp.unlink(missing_ok=True)
    models_tmp.unlink(missing_ok=True)
    result["endpoint_capture_epoch"] = time.time()
    try:
        metrics = fetch(
            f"http://{SERVER_HOST}:{SERVER_PORT}/metrics",
            timeout=ENDPOINT_TIMEOUT_SECONDS,
        )
        metrics_tmp.write_bytes(metrics)
        metrics_tmp.replace(metrics_path)
        result["final_metrics_bytes"] = len(metrics)
        result["final_metrics_sha256"] = sha256_bytes(metrics)
    except Exception as exc:  # noqa: BLE001 - preserve teardown after capture failure
        metrics_tmp.unlink(missing_ok=True)
        result["final_metrics_error"] = repr(exc)
    try:
        raw_models = fetch(
            f"http://{SERVER_HOST}:{SERVER_PORT}/v1/models",
            timeout=ENDPOINT_TIMEOUT_SECONDS,
        )
        models = json.loads(raw_models.decode("utf-8"))
        write_json(models_tmp, models)
        models_tmp.replace(models_path)
        result["final_models_sha256"] = sha256_file(models_path)
        result["final_models_bytes"] = models_path.stat().st_size
    except Exception as exc:  # noqa: BLE001 - preserve teardown after capture failure
        models_tmp.unlink(missing_ok=True)
        result["final_models_error"] = repr(exc)


def artifact_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    if not path.is_file():
        return {"exists": True, "is_file": False}
    result = {
        "exists": True,
        "is_file": True,
        "bytes": path.stat().st_size,
    }
    if result["bytes"] > MAX_REQUIRED_ARTIFACT_BYTES:
        result["sha256"] = None
        result["hash_error"] = "artifact_exceeds_bounded_hash_limit"
    else:
        result["sha256"] = sha256_file(path)
    return result


def required_artifacts_preserved(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> bool:
    return all(
        not snapshot.get("exists", False)
        or (bool(snapshot.get("sha256")) and after.get(name) == snapshot)
        for name, snapshot in before.items()
    )


def signal_exact(
    records: dict[int, dict[str, Any]], sig: signal.Signals
) -> list[dict[str, Any]]:
    signalled: list[dict[str, Any]] = []
    for pid in sorted(records, reverse=True):
        saved = records[pid]
        current = proc_record(pid, read_environ=False)
        if current is None or not exact_signal_match(current, saved):
            continue
        try:
            os.kill(pid, sig)
            signalled.append(
                {
                    "pid": pid,
                    "start_ticks": saved["start_ticks"],
                    "signal": int(sig),
                }
            )
        except ProcessLookupError:
            pass
    return signalled


def wait_for_exact_exit(
    records: dict[int, dict[str, Any]], timeout_seconds: float
) -> dict[int, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    survivors: dict[int, dict[str, Any]] = {}
    while True:
        survivors = {}
        for pid, saved in records.items():
            current = proc_record(pid, read_environ=False)
            if current is not None and exact_signal_match(current, saved):
                survivors[pid] = current
        if not survivors:
            return {}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return survivors
        time.sleep(min(PROCESS_POLL_SECONDS, remaining))


def stop_owned_processes(
    identity: dict[str, Any], result: dict[str, Any]
) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    historical: set[tuple[int, int]] = set()
    before = scan_ownership(identity, owned_process_table(identity))
    result["process_scan_before_term"] = {
        key: value for key, value in before.items() if key != "authorized"
    }
    if before["root_conflict"] or before["saved_conflicts"]:
        result["signal_blocked_by_identity_conflict"] = True
        return before, historical

    term_targets = before["authorized"]
    for record in term_targets.values():
        historical.add((int(record["pid"]), int(record["start_ticks"])))
    result["sigterm"] = signal_exact(term_targets, signal.SIGTERM)
    term_started = time.monotonic()
    term_survivors = wait_for_exact_exit(term_targets, TERM_GRACE_SECONDS)
    result["term_wait_seconds"] = time.monotonic() - term_started
    result["term_wait_bound_seconds"] = TERM_GRACE_SECONDS
    result["term_exact_survivors"] = [
        public_record(term_survivors[pid]) for pid in sorted(term_survivors)
    ]

    after_term = scan_ownership(identity, owned_process_table(identity))
    result["process_scan_after_term"] = {
        key: value for key, value in after_term.items() if key != "authorized"
    }
    result["sigkill"] = []
    if not after_term["root_conflict"] and not after_term["saved_conflicts"]:
        kill_targets = after_term["authorized"]
        for record in after_term["authorized"].values():
            historical.add((int(record["pid"]), int(record["start_ticks"])))
        result["sigkill"] = signal_exact(kill_targets, signal.SIGKILL)
        kill_started = time.monotonic()
        kill_survivors = wait_for_exact_exit(kill_targets, KILL_GRACE_SECONDS)
        result["kill_wait_seconds"] = time.monotonic() - kill_started
        result["kill_wait_bound_seconds"] = KILL_GRACE_SECONDS
        result["kill_exact_survivors"] = [
            public_record(kill_survivors[pid]) for pid in sorted(kill_survivors)
        ]

    final = scan_ownership(identity, owned_process_table(identity))
    result["process_scan_after_kill"] = {
        key: value for key, value in final.items() if key != "authorized"
    }
    return final, historical


def classify_vllm_gpu_rows(
    rows: list[dict[str, Any]], historical: set[tuple[int, int]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    historical_pids = {pid for pid, _ in historical}
    for row in rows:
        if "query_error" in row:
            continue
        pid = int(row["pid"])
        process_text = f"{row.get('process_name', '')} {row.get('comm', '')} {row.get('cmdline', '')}".lower()
        exact_owned = (
            row.get("start_ticks") is not None
            and (pid, int(row["start_ticks"])) in historical
        )
        stale_owned_pid = row.get("start_ticks") is None and pid in historical_pids
        marker = any(
            item in process_text
            for item in ("vllm", "pleworker", "ple_worker", "qwen3_8_flash_next")
        )
        if exact_owned or stale_owned_pid or marker:
            result.append(
                {
                    **row,
                    "classification": {
                        "exact_owned": exact_owned,
                        "stale_owned_pid": stale_owned_pid,
                        "marker": marker,
                    },
                }
            )
    return result


def run_self_tests() -> None:
    argv = ["/usr/bin/python", "-m", "vllm.entrypoints.openai.api_server"]
    digest = argv_sha256(argv)
    assert SHA256_PATTERN.fullmatch(digest)
    assert digest == argv_sha256(list(argv))
    assert marker_reasons(
        {
            "comm": "VLLM::PLEWorker",
            "cmdline": "python -c spawn_main",
        }
    ) == ["ple_worker", "vllm_process_title"]
    assert marker_reasons(
        {
            "comm": "torch_shm_manager",
            "cmdline": "torch_shm_manager",
            "cmdline_sha256": TORCH_SHM_MANAGER_CMDLINE_SHA256,
        }
    ) == ["torch_shm_manager"]
    assert not marker_reasons(
        {
            "comm": "python",
            "cmdline": "python /bundle/vllm_serving_teardown.py --self-test",
        }
    )
    records = {
        10: {"ppid": 1},
        11: {"ppid": 10},
        12: {"ppid": 11},
        99: {"ppid": 1},
    }
    assert descendant_ids(10, records) == {10, 11, 12}
    assert descendant_ids_many({10}, records) == {10, 11, 12}
    identity = {
        "schema_version": 1,
        "backend": "vllm",
        "boot_id": "boot",
        "host": SERVER_HOST,
        "port": SERVER_PORT,
        "pid": 10,
        "start_ticks": 20,
        "pgid": 10,
        "sid": 10,
        "argv": argv,
        "argv_sha256": digest,
        "vllm_tuning": {
            "schema_version": 1,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "kv_cache_dtype": "auto",
            "kv_cache_memory_bytes": 5 * 1024**3,
            "max_cudagraph_capture_size": 4,
            "max_num_batched_tokens": 8192,
            "max_num_batched_tokens_requested": 8192,
            "max_num_seqs": 4,
            "moe_backend": None,
            "mtp_dynamic_batch_schedule": None,
            "mtp_index_share_for_iteration": False,
            "mtp_speculative_tokens": 3,
            "omp_num_threads": 1,
            "overridden": {
                name: name
                not in {
                    "TAAF_VLLM_MOE_BACKEND",
                    "TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION",
                    "TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE",
                }
                for name in VLLM_TUNING_OVERRIDE_FIELDS
            },
        },
        "started_epoch": 1.0,
        "phase": "ready",
        "snapshot_epoch": 2.0,
        "ready_epoch": 1.5,
        "workers": [
            {
                "pid": 10,
                "start_ticks": 20,
                "ppid": 1,
                "pgid": 10,
                "sid": 10,
                "cmdline_sha256": "a" * 64,
            },
            {
                "pid": 11,
                "start_ticks": 21,
                "ppid": 10,
                "pgid": 10,
                "sid": 10,
                "cmdline_sha256": "b" * 64,
            },
        ],
    }
    validated, errors = validate_identity_document(identity, "boot")
    assert validated is identity and not errors
    bad_tuning = json.loads(json.dumps(identity))
    bad_tuning["vllm_tuning"]["max_num_seqs"] = True
    assert validate_identity_document(bad_tuning, "boot") == (
        None,
        ["vllm_tuning_invalid_max_num_seqs"],
    )
    unknown_tuning = json.loads(json.dumps(identity))
    unknown_tuning["vllm_tuning"]["surprise"] = 1
    assert validate_identity_document(unknown_tuning, "boot") == (
        None,
        ["vllm_tuning_unknown_fields:surprise"],
    )
    broken = dict(identity)
    broken["argv_sha256"] = "0" * 64
    assert validate_identity_document(broken, "boot")[0] is None
    starting = dict(identity)
    starting.update(
        {
            "phase": "starting",
            "snapshot_epoch": 1.0,
            "ready_epoch": None,
            "workers": [identity["workers"][0]],
        }
    )
    assert validate_identity_document(starting, "boot")[0] is starting
    detached_identity = json.loads(json.dumps(identity))
    detached_identity["workers"].append(
        {
            "pid": 12,
            "start_ticks": 22,
            "ppid": 11,
            "pgid": 12,
            "sid": 12,
            "cmdline_sha256": TORCH_SHM_MANAGER_CMDLINE_SHA256,
        }
    )
    assert validate_identity_document(detached_identity, "boot")[0] is detached_identity
    bad_detached = json.loads(json.dumps(detached_identity))
    bad_detached["workers"][-1]["cmdline_sha256"] = "f" * 64
    assert validate_identity_document(bad_detached, "boot")[0] is None
    root_record = {
        "pid": 10,
        "state": "S",
        "ppid": 1,
        "pgid": 10,
        "sid": 10,
        "start_ticks": 20,
        "comm": "python",
        "cmdline": "python -m vllm.entrypoints.openai.api_server",
        "cmdline_sha256": "a" * 64,
    }
    worker_record = {
        "pid": 11,
        "state": "S",
        "ppid": 10,
        "pgid": 10,
        "sid": 10,
        "start_ticks": 21,
        "comm": "python",
        "cmdline": "python -c spawn_main",
        "cmdline_sha256": "b" * 64,
    }
    detached_record = {
        "pid": 12,
        "state": "S",
        "ppid": 11,
        "pgid": 12,
        "sid": 12,
        "start_ticks": 22,
        "comm": "torch_shm_manager",
        "cmdline": "torch_shm_manager",
        "cmdline_sha256": TORCH_SHM_MANAGER_CMDLINE_SHA256,
    }
    assert exact_signal_match(worker_record, worker_record)
    changed_command = dict(worker_record)
    changed_command["cmdline_sha256"] = "d" * 64
    assert not exact_signal_match(changed_command, worker_record)
    changed_session = dict(worker_record)
    changed_session["sid"] = 99
    assert not exact_signal_match(changed_session, worker_record)
    original_proc_record = globals()["proc_record"]
    try:
        globals()["proc_record"] = lambda _pid, read_environ=False: worker_record
        assert wait_for_exact_exit({11: worker_record}, 0) == {11: worker_record}
        globals()["proc_record"] = lambda _pid, read_environ=False: changed_command
        assert wait_for_exact_exit({11: worker_record}, 0) == {}
    finally:
        globals()["proc_record"] = original_proc_record
    marker_only = {
        "pid": 99,
        "state": "S",
        "ppid": 1,
        "pgid": 99,
        "sid": 99,
        "start_ticks": 199,
        "comm": "VLLM::stale",
        "cmdline": "python -c other",
        "cmdline_sha256": "c" * 64,
    }
    live_scan = scan_ownership(
        identity, {10: root_record, 11: worker_record, 99: marker_only}
    )
    assert set(live_scan["authorized"]) == {10, 11}
    assert [item["pid"] for item in live_scan["marker_only_suspects"]] == [99]
    detached_scan = scan_ownership(
        detached_identity,
        {10: root_record, 11: worker_record, 12: detached_record},
    )
    assert set(detached_scan["authorized"]) == {10, 11, 12}
    starting_scan = scan_ownership(starting, {10: root_record, 11: worker_record})
    assert set(starting_scan["authorized"]) == {10, 11}
    dead_root_scan = scan_ownership(identity, {11: worker_record})
    assert set(dead_root_scan["authorized"]) == {11}
    conflict = dict(root_record)
    conflict["start_ticks"] = 200
    conflict_scan = scan_ownership(identity, {10: conflict, 11: worker_record})
    assert conflict_scan["root_conflict"] and not conflict_scan["authorized"]
    assert classify_vllm_gpu_rows(
        [
            {
                "pid": 11,
                "start_ticks": 21,
                "process_name": "python",
                "comm": "python",
                "cmdline": "python -c spawn_main",
            }
        ],
        {(11, 21)},
    )
    log_counts = count_log_markers(
        'HTTP/1.1" 500 parser error maximum context length CUDA out of memory Engine core failed'
    )
    assert all(
        log_counts[name] >= 1
        for name in ("http_5xx", "parser_error", "context_error", "out_of_memory", "crash")
    )
    assert MAX_EXPLICIT_WAIT_SECONDS == 11.3
    assert MAX_EXPLICIT_WAIT_SECONDS <= 20.0
    before_artifacts = {
        "score.json": {
            "exists": True,
            "is_file": True,
            "bytes": 12,
            "sha256": "1" * 64,
        },
        "submission.parquet": {"exists": False},
    }
    assert required_artifacts_preserved(
        before_artifacts,
        {
            "score.json": dict(before_artifacts["score.json"]),
            "submission.parquet": {
                "exists": True,
                "is_file": True,
                "bytes": 99,
                "sha256": "2" * 64,
            },
        },
    )
    changed_artifacts = {
        "score.json": {**before_artifacts["score.json"], "bytes": 13},
        "submission.parquet": {"exists": False},
    }
    assert not required_artifacts_preserved(before_artifacts, changed_artifacts)
    print("VLLM_TEARDOWN_SELF_TEST_OK", flush=True)


def main() -> None:
    teardown_started = time.monotonic()
    working_dir = validate_working_dir()
    paths = working_paths(working_dir)
    server_log = paths["server_log"]
    server_identity = paths["server_identity"]
    result_path = paths["result"]
    final_metrics = paths["final_metrics"]
    final_models = paths["final_models"]
    score_path = paths["score"]
    submission_path = paths["submission"]
    assert isinstance(server_log, Path)
    assert isinstance(server_identity, Path)
    assert isinstance(result_path, Path)
    assert isinstance(final_metrics, Path)
    assert isinstance(final_models, Path)
    assert isinstance(score_path, Path)
    assert isinstance(submission_path, Path)
    cache_roots = paths["cache_roots"]
    assert isinstance(cache_roots, tuple)

    result_path.unlink(missing_ok=True)
    result: dict[str, Any] = {
        "timestamp_epoch": time.time(),
        "working_dir": str(working_dir),
        "working_dir_validated": True,
        "identity_exists": server_identity.exists(),
        "port_open_before": port_open(),
        "max_explicit_wait_seconds": MAX_EXPLICIT_WAIT_SECONDS,
        "recursive_cleanup_skipped": True,
    }
    required_before = {
        "score.json": artifact_snapshot(score_path),
        "submission.parquet": artifact_snapshot(submission_path),
    }
    result["required_artifacts_before"] = required_before
    capture_endpoints(final_metrics, final_models, result)

    identity: dict[str, Any] | None = None
    identity_errors: list[str] = []
    if not server_identity.exists():
        identity_errors.append("identity_file_missing")
    else:
        try:
            raw_identity = json.loads(server_identity.read_text(encoding="utf-8"))
            identity, identity_errors = validate_identity_document(
                raw_identity, current_boot_id()
            )
        except Exception as exc:  # noqa: BLE001 - invalid identity must block signals
            identity_errors.append(f"identity_read_error:{exc!r}")
    result["identity_valid"] = identity is not None
    result["identity_errors"] = identity_errors

    final_scan: dict[str, Any] = {
        "authorized": {},
        "authorized_records": [],
        "suspect_records": [],
        "marker_matches": [],
        "marker_only_suspects": [],
        "group_only_suspects": [],
        "root_conflict": None,
        "saved_conflicts": [],
    }
    historical: set[tuple[int, int]] = set()
    if identity is not None:
        for worker in identity["workers"]:
            historical.add((int(worker["pid"]), int(worker["start_ticks"])))
        final_scan, signalled = stop_owned_processes(identity, result)
        historical.update(signalled)
    else:
        result["signals_blocked"] = "identity_not_valid"

    port_wait_started = time.monotonic()
    port_deadline = port_wait_started + PORT_CLOSE_GRACE_SECONDS
    while time.monotonic() < port_deadline and port_open():
        remaining = port_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(PROCESS_POLL_SECONDS, remaining))
    result["port_wait_seconds"] = time.monotonic() - port_wait_started
    result["port_wait_bound_seconds"] = PORT_CLOSE_GRACE_SECONDS
    result["port_closed"] = not port_open()

    if identity is not None:
        final_scan = scan_ownership(identity, owned_process_table(identity))
        result["process_scan_final_gate"] = {
            key: value for key, value in final_scan.items() if key != "authorized"
        }
    full_marker_scan = global_marker_records()
    gpu_after = gpu_rows()
    gpu_pids = {
        int(row["pid"])
        for row in gpu_after
        if "query_error" not in row and "pid" in row
    }
    cpu_only_marker_survivors = [
        record for record in full_marker_scan if int(record["pid"]) not in gpu_pids
    ]
    vllm_gpu_after = classify_vllm_gpu_rows(gpu_after, historical)
    result["full_proc_marker_survivors"] = full_marker_scan
    result["cpu_only_vllm_ple_marker_survivors"] = cpu_only_marker_survivors
    result["gpu_rows_after"] = gpu_after
    result["gpu_query_error_after"] = any("query_error" in row for row in gpu_after)
    result["vllm_gpu_rows_after"] = vllm_gpu_after
    capture_log(server_log, "final", result)

    required_after = {
        "score.json": artifact_snapshot(score_path),
        "submission.parquet": artifact_snapshot(submission_path),
    }
    artifacts_preserved = required_artifacts_preserved(
        required_before, required_after
    )
    result["required_artifacts_after"] = required_after
    result["required_artifacts_preserved"] = artifacts_preserved

    root_or_saved_conflict = bool(
        final_scan.get("root_conflict") or final_scan.get("saved_conflicts")
    )
    cpu_survivors = bool(
        final_scan.get("authorized_records")
        or final_scan.get("suspect_records")
        or full_marker_scan
    )
    metrics_preserved = final_metrics.is_file() and bool(
        result.get("final_metrics_sha256")
    )
    result["final_metrics_preserved"] = metrics_preserved
    shutdown_gate = all(
        (
            identity is not None,
            not root_or_saved_conflict,
            result["port_closed"],
            not cpu_survivors,
            not result["gpu_query_error_after"],
            not vllm_gpu_after,
            metrics_preserved,
            artifacts_preserved,
        )
    )
    result["shutdown_ok"] = shutdown_gate
    result["ephemeral_cleanup_deferred_to_kaggle_vm"] = {
        str(root): root.exists() or root.is_symlink()
        for root in (RUNTIME_ROOT, *SYSTEM_CACHE_ROOTS, *cache_roots)
    }
    result["teardown_elapsed_seconds"] = time.monotonic() - teardown_started
    result["completed_epoch"] = time.time()
    write_json(result_path, result)
    summary = {
        "shutdown_ok": shutdown_gate,
        "port_closed": result["port_closed"],
        "metrics_preserved": metrics_preserved,
        "required_artifacts_preserved": artifacts_preserved,
        "cpu_survivors": len(full_marker_scan),
        "vllm_gpu_survivors": len(vllm_gpu_after),
        "elapsed_seconds": result["teardown_elapsed_seconds"],
        "result": str(result_path),
    }
    print("VLLM_SERVER_TEARDOWN", json.dumps(summary, sort_keys=True), flush=True)
    if not shutdown_gate:
        raise RuntimeError("vLLM teardown did not reach the bounded terminal gate")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        run_self_tests()
    elif sys.argv[1:]:
        raise SystemExit(f"unsupported arguments: {sys.argv[1:]}")
    else:
        main()
