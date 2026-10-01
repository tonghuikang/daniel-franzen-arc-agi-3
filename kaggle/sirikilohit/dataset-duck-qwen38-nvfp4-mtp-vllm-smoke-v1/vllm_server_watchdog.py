"""Fail-closed health watchdog for the owned Kaggle vLLM server.

The watchdog reuses the serving setup module's ownership, stop, start, and
readiness functions.  It never sends a signal itself and never launches a
replacement until the setup contract proves that the old server is gone and
the listening port is closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class WatchdogConfig:
    interval_seconds: float = 15.0
    request_timeout_seconds: int = 5
    failure_threshold: int = 4
    max_restart_attempts: int = 2

    def validate(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.max_restart_attempts < 0:
            raise ValueError("max_restart_attempts cannot be negative")


class WatchdogError(RuntimeError):
    """The watchdog could not prove that a restart was safe."""


class WatchdogStopped(WatchdogError):
    """Notebook shutdown cancelled watchdog work."""


def _raise_if_stopping(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise WatchdogStopped("Notebook shutdown cancelled the vLLM restart")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_event(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _identity(setup: ModuleType) -> dict[str, Any]:
    path = Path(setup.SERVER_IDENTITY)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise WatchdogError(f"Cannot read the owned server identity: {exc}") from exc
    if not isinstance(value, dict):
        raise WatchdogError("The owned server identity is not a JSON object")
    return value


def _model_dir_from_identity(setup: ModuleType, identity: dict[str, Any]) -> Path:
    argv = identity.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise WatchdogError("The owned server identity has no valid argv")
    try:
        serve_index = argv.index("serve")
        model_dir = Path(argv[serve_index + 1])
    except (ValueError, IndexError) as exc:
        raise WatchdogError("The owned server identity has no vLLM serve model path") from exc
    expected_commands = (
        setup.server_command(model_dir, enable_chunked_prefill=False),
        setup.server_command(model_dir, enable_chunked_prefill=True),
    )
    if argv not in expected_commands:
        raise WatchdogError("The saved server argv differs from the current serving contract")
    if not model_dir.is_dir():
        raise WatchdogError(f"The saved model directory no longer exists: {model_dir}")
    return model_dir


def health(setup: ModuleType, timeout_seconds: int) -> tuple[bool, str]:
    """Check the cheap endpoint and require the exact served model identity."""

    try:
        response = setup.request_json(
            f"{setup.BASE_URL}/models", timeout=timeout_seconds
        )
        rows = response.get("data") or []
        ids = [row.get("id") for row in rows if isinstance(row, dict)]
        expected = [setup.SERVED_MODEL_NAME]
        if ids != expected:
            return False, f"wrong_models:{ids!r}"
        return True, "ok"
    except Exception as exc:  # A failed health request is data, not a watchdog crash.
        return False, f"{type(exc).__name__}:{exc}"


def _wait_for_server_cancelable(
    setup: ModuleType,
    identity: dict[str, Any],
    stop_event: threading.Event | None,
    request_timeout_seconds: int,
) -> dict[str, Any]:
    """Wait for exact readiness while allowing immediate notebook shutdown."""

    started = time.monotonic()
    deadline = started + int(setup.SERVER_READY_TIMEOUT)
    last_reason = "not_checked"
    while time.monotonic() < deadline:
        _raise_if_stopping(stop_event)
        if not setup.alive(int(identity["pid"]), int(identity["start_ticks"])):
            tail = setup.tail_log(300) if hasattr(setup, "tail_log") else ""
            raise RuntimeError(f"vLLM exited before restart readiness.\n{tail}")
        ready, last_reason = health(setup, request_timeout_seconds)
        if ready:
            _raise_if_stopping(stop_event)
            return {"ready_seconds": time.monotonic() - started}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_seconds = min(5.0, remaining)
        if stop_event is None:
            time.sleep(sleep_seconds)
        elif stop_event.wait(sleep_seconds):
            _raise_if_stopping(stop_event)
    tail = setup.tail_log(300) if hasattr(setup, "tail_log") else ""
    raise TimeoutError(
        f"Timed out waiting for replacement vLLM: {last_reason}.\n{tail}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _server_log_archives(setup: ModuleType) -> list[dict[str, Any]]:
    working_dir = Path(setup.WORKING_DIR)
    rows: list[tuple[int, Path]] = []
    for path in working_dir.glob("vllm-openai-server.restart-*.log"):
        index = path.name.removeprefix(
            "vllm-openai-server.restart-"
        ).removesuffix(".log")
        if index.isdigit() and path.is_file() and not path.is_symlink():
            rows.append((int(index), path))
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for _, path in sorted(rows)
    ]


def _rotate_server_log(setup: ModuleType) -> dict[str, Any] | None:
    """Atomically retain the newest two pre-replacement server logs."""

    source = Path(setup.SERVER_LOG)
    working_dir = Path(setup.WORKING_DIR)
    if source.parent != working_dir or source.name != "vllm-openai-server.log":
        raise WatchdogError(f"Unexpected server log path: {source}")
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_file():
        raise WatchdogError(f"The server log is not a regular owned file: {source}")

    archives: list[tuple[int, Path]] = []
    for path in working_dir.glob("vllm-openai-server.restart-*.log"):
        match = path.name.removeprefix("vllm-openai-server.restart-").removesuffix(
            ".log"
        )
        if match.isdigit() and path.is_file() and not path.is_symlink():
            archives.append((int(match), path))
    archives.sort()
    # Delete before replace so even an interrupted rotation never retains more
    # than two large server logs.
    while len(archives) >= 2:
        _, oldest = archives.pop(0)
        oldest.unlink()
    next_index = (max((index for index, _ in archives), default=0) + 1)
    destination = working_dir / f"vllm-openai-server.restart-{next_index}.log"
    os.replace(source, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def _validate_stopped_cleanup(cleanup: dict[str, Any], context: str) -> None:
    if not cleanup.get("stopped", False):
        raise WatchdogError(f"{context} did not reach a stopped gate: {cleanup}")
    if cleanup.get("identity_rejection_reasons"):
        raise WatchdogError(
            f"{context} rejected the saved ownership identity: "
            f"{cleanup['identity_rejection_reasons']}"
        )
    if cleanup.get("survivors"):
        raise WatchdogError(f"{context} left owned survivors: {cleanup['survivors']}")


def _launch_replacement(
    setup: ModuleType,
    model_dir: Path,
    server_env: dict[str, str],
    stop_event: threading.Event | None,
    request_timeout_seconds: int,
    *,
    enable_chunked_prefill: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Launch one mode and always stop it if readiness or shutdown fails."""

    _raise_if_stopping(stop_event)
    archive = _rotate_server_log(setup)
    _raise_if_stopping(stop_event)
    try:
        identity = setup.start_server(
            model_dir,
            server_env,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        _raise_if_stopping(stop_event)
        readiness = _wait_for_server_cancelable(
            setup,
            identity,
            stop_event,
            request_timeout_seconds,
        )
        _raise_if_stopping(stop_event)
        ready_identity = setup.update_ready_identity(identity)
        _raise_if_stopping(stop_event)
        return ready_identity, readiness, archive
    except BaseException as exc:
        failed_cleanup = setup.stop_owned_server()
        try:
            _validate_stopped_cleanup(failed_cleanup, "Failed replacement cleanup")
        except WatchdogError:
            raise WatchdogError(
                "The replacement failed and could not be stopped safely: "
                f"{failed_cleanup}"
            ) from exc
        raise


def restart_owned_server(
    setup: ModuleType,
    *,
    request_timeout_seconds: int = 5,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Restart once after proving the exact old server no longer exists.

    The setup module owns all process matching and signaling.  This function
    only coordinates its existing contract and adds a second no-duplicate gate
    before launch.
    """

    _raise_if_stopping(stop_event)
    healthy, reason = health(setup, request_timeout_seconds)
    _raise_if_stopping(stop_event)
    if healthy:
        return {"restarted": False, "reason": "healthy_on_restart_recheck"}

    identity = _identity(setup)
    model_dir = _model_dir_from_identity(setup, identity)
    saved_pid = int(identity.get("pid", -1))
    saved_ticks = int(identity.get("start_ticks", -1))
    # Rebuild the pinned launch environment before stopping the old server so
    # this small check does not add to restart downtime.
    server_env, _environment_check = setup.runtime_environment(
        deep_preload_validation=False
    )
    _raise_if_stopping(stop_event)

    cleanup = setup.stop_owned_server()
    _validate_stopped_cleanup(cleanup, "Owned server cleanup")
    if cleanup.get("port_open") or setup.port_open():
        raise WatchdogError("The vLLM port is still open after owned cleanup")

    live_saved = setup.proc_record(saved_pid) if saved_pid > 1 else None
    if (
        live_saved is not None
        and live_saved.get("state") != "Z"
        and int(live_saved.get("start_ticks", -1)) == saved_ticks
    ):
        raise WatchdogError("The exact old vLLM root still lives after cleanup")

    _raise_if_stopping(stop_event)
    try:
        ready_identity, readiness, _archive = _launch_replacement(
            setup,
            model_dir,
            server_env,
            stop_event,
            request_timeout_seconds,
            enable_chunked_prefill=True,
        )
    except WatchdogStopped:
        raise
    except BaseException as exc:
        raise WatchdogError(
            "The chunked-prefill replacement failed after exact cleanup; "
            f"error={type(exc).__name__}:{exc}"
        ) from exc
    return {
        "restarted": True,
        "trigger": reason,
        "old_pid": saved_pid,
        "new_pid": int(ready_identity["pid"]),
        "ready_seconds": float(readiness["ready_seconds"]),
        "chunked_prefill_enabled": True,
        "replacement_launches": 1,
        "preserved_server_logs": _server_log_archives(setup),
        "completed_epoch": time.time(),
    }


class ServerWatchdog:
    """Single-threaded watchdog with a non-overlapping restart section."""

    def __init__(
        self,
        setup: ModuleType,
        config: WatchdogConfig = WatchdogConfig(),
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        config.validate()
        self.setup = setup
        self.config = config
        self.stop_event = stop_event or threading.Event()
        working_dir = Path(setup.WORKING_DIR)
        self.events_path = working_dir / "vllm-watchdog-events.jsonl"
        self.status_path = working_dir / "vllm-watchdog-status.json"
        self._restart_lock = threading.Lock()
        self.restart_attempts = 0

    def stop(self, _signal_number: int = int(signal.SIGTERM)) -> None:
        self.stop_event.set()

    def _event(self, event: str, **fields: Any) -> None:
        value = {"event": event, "epoch": time.time(), **fields}
        _append_event(self.events_path, value)
        _write_json(self.status_path, value)

    def run(self) -> None:
        failures = 0
        self._event("watchdog_started", config=self.config.__dict__)
        try:
            while not self.stop_event.is_set():
                healthy, reason = health(
                    self.setup, self.config.request_timeout_seconds
                )
                if healthy:
                    if failures:
                        self._event("health_recovered", prior_failures=failures)
                    failures = 0
                else:
                    failures += 1
                    self._event(
                        "health_failed",
                        consecutive_failures=failures,
                        reason=reason,
                    )
                    if failures >= self.config.failure_threshold:
                        if self.restart_attempts >= self.config.max_restart_attempts:
                            self._event(
                                "restart_limit_reached",
                                restart_attempts=self.restart_attempts,
                            )
                            return
                        with self._restart_lock:
                            self.restart_attempts += 1
                            try:
                                result = restart_owned_server(
                                    self.setup,
                                    request_timeout_seconds=(
                                        self.config.request_timeout_seconds
                                    ),
                                    stop_event=self.stop_event,
                                )
                            except WatchdogStopped:
                                self._event(
                                    "restart_cancelled",
                                    restart_attempt=self.restart_attempts,
                                )
                                return
                            except Exception as exc:
                                self._event(
                                    "restart_failed",
                                    restart_attempt=self.restart_attempts,
                                    error_type=type(exc).__name__,
                                    error=str(exc),
                                    traceback=traceback.format_exc(),
                                )
                            else:
                                self._event(
                                    "restart_complete",
                                    restart_attempt=self.restart_attempts,
                                    result=result,
                                )
                        failures = 0
                self.stop_event.wait(self.config.interval_seconds)
        finally:
            self._event("watchdog_stopped", restart_attempts=self.restart_attempts)


_background_lock = threading.Lock()
_background_watchdog: tuple[ServerWatchdog, threading.Thread] | None = None
_background_stopping = False


def start_background(
    setup: ModuleType,
    config: WatchdogConfig = WatchdogConfig(),
) -> tuple[ServerWatchdog, threading.Thread]:
    """Start exactly one watchdog thread in the notebook process."""

    global _background_watchdog
    with _background_lock:
        if _background_stopping:
            raise WatchdogError("The vLLM watchdog is stopping for notebook teardown")
        if _background_watchdog is not None and _background_watchdog[1].is_alive():
            raise WatchdogError("A vLLM watchdog thread is already running")
        controller = ServerWatchdog(setup, config)
        thread = threading.Thread(
            target=controller.run,
            name="vllm-server-watchdog",
            daemon=True,
        )
        thread.start()
        _background_watchdog = (controller, thread)
        return controller, thread


def stop_background(timeout_seconds: float = 30.0) -> None:
    """Stop the notebook watchdog before the normal serving teardown runs."""

    global _background_stopping, _background_watchdog
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with _background_lock:
        current = _background_watchdog
        if current is not None:
            _background_stopping = True
    if current is None:
        return
    controller, thread = current
    controller.stop(int(signal.SIGTERM))
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise WatchdogError("The vLLM watchdog did not stop before teardown")
    with _background_lock:
        if _background_watchdog is current:
            _background_watchdog = None
            _background_stopping = False


def load_setup(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("taaf_vllm_serving_setup", path)
    if spec is None or spec.loader is None:
        raise WatchdogError(f"Cannot load serving setup module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--request-timeout-seconds", type=int, default=5)
    parser.add_argument("--failure-threshold", type=int, default=4)
    parser.add_argument("--max-restart-attempts", type=int, default=2)
    args = parser.parse_args(argv)
    setup = load_setup(args.setup)
    controller = ServerWatchdog(
        setup,
        WatchdogConfig(
            interval_seconds=args.interval_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            failure_threshold=args.failure_threshold,
            max_restart_attempts=args.max_restart_attempts,
        ),
    )
    signal.signal(signal.SIGTERM, lambda signum, _frame: controller.stop(signum))
    signal.signal(signal.SIGINT, lambda signum, _frame: controller.stop(signum))
    controller.run()


if __name__ == "__main__":
    main()
