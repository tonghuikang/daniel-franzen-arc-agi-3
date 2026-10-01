"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

from inference.agent import python_workspace as _python_workspace
from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""
    GRID_DECODE_TABLE = b""

    __SEGMENTATION_SOURCE__

    __WORKSPACE_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "AssertionError",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "ZeroDivisionError",
        "ImportError",
        "StopIteration",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, step, level, shape, grid, ascii=None, animation_frames=(), frame_count=1):
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self.animation_frames = list(animation_frames)
            self.frame_count = frame_count
            self._numeric_grid = grid
            self._ascii = None if ascii is None else str(ascii)
            self._encoded_grid = self._ascii if grid is None and self.shape[0] else None
            self._rows = None
            self._segmentation = None

        @property
        def _grid(self):
            '''Decode once, splitting rows before color 10 becomes a newline byte.'''
            if self._numeric_grid is None:
                self._numeric_grid = [
                    list(row.translate(GRID_DECODE_TABLE))
                    for row in self._encoded_grid.encode("ascii").split(b"\n")
                ] if self._encoded_grid is not None else []
                self._encoded_grid = None

            return self._numeric_grid

        @_grid.setter
        def _grid(self, grid):
            self._numeric_grid = grid
            self._encoded_grid = None

        def __copy__(self):
            '''Keep shallow copies sharing their grid even before its first read.'''
            duplicate = object.__new__(type(self))
            duplicate.__dict__ = self.__dict__.copy()
            duplicate._grid = self._grid

            return duplicate

        @property
        def ascii(self):
            if self._ascii is None:
                self._ascii = _grid_ascii(self._grid)
            return self._ascii

        @property
        def rows(self):
            if self._rows is None:
                self._rows = tuple(self.ascii.split("\n"))

            return self._rows

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def crop(self, row_min, col_min, row_max, col_max):
            return _render_crop_text(self.rows, row_min, col_min, row_max, col_max)

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    def _short_plan(plan):
        return plan if len(plan) <= 40 else plan[:40] + "..."


    class HistoryEntryView:
        def __init__(self, *, action, frame, result=None, plan=""):
            self.action = action
            self.frame = frame
            self.result = dict(result) if isinstance(result, dict) else {}
            self.plan = plan

        def __str__(self):
            return (
                f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame}, "
                f"result={self.result}, plan={_short_plan(self.plan)!r})"
            )

        __repr__ = __str__


    class GameOverView:
        def __init__(self, *, actions, frame):
            self.actions = list(actions)
            self.action_count = len(self.actions)
            self.frame = frame

        def __str__(self):
            return f"GameOverView(action_count={self.action_count}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result, plan=""):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}
            self.plan = plan

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame}, "
                f"result={self.result}, plan={_short_plan(self.plan)!r})"
            )

        __repr__ = __str__


    def _grid_ascii(grid):
        return "\n".join(
            "".join(COLOR_CHARS[max(0, min(15, int(value)))] for value in row) for row in grid
        )


    def _render_crop_text(rows, row_min, col_min, row_max, col_max):
        '''The board region with inclusive bounds, clipped to the board, as text with a
        two-line column ruler (tens digit above ones digit) and the row number before each row.'''
        row_min, col_min, row_max, col_max = int(row_min), int(col_min), int(row_max), int(col_max)
        if row_min > row_max or col_min > col_max:
            raise ValueError(
                "crop takes (row_min, col_min, row_max, col_max) with each min at most its max, "
                f"got rows {row_min}..{row_max} and cols {col_min}..{col_max}"
            )
        height = len(rows)
        width = len(rows[0]) if height else 0
        row_min = max(0, min(row_min, height - 1))
        row_max = max(row_min, min(row_max, height - 1))
        col_min = max(0, min(col_min, width - 1))
        col_max = max(col_min, min(col_max, width - 1))
        label_width = len(str(row_max))
        columns = range(col_min, col_max + 1)
        tens_ruler = " " * label_width + " " + "".join(str(col // 10 % 10) for col in columns)
        ones_ruler = " " * label_width + " " + "".join(str(col % 10) for col in columns)
        lines = [tens_ruler, ones_ruler]
        for row in range(row_min, row_max + 1):
            lines.append(f"{row:>{label_width}} {rows[row][col_min:col_max + 1]}")

        return "\n".join(lines)


    def _animation_frame_from_payload(payload):
        return FrameView(
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid"),
            ascii=payload.get("ascii"),
            animation_frames=[
                _animation_frame_from_payload(item)
                for item in payload.get("animation") or []
                if isinstance(item, dict)
            ],
            frame_count=int(payload.get("frame_count", 1) or 1),
        )


    def _game_overs_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                GameOverView(
                    actions=[str(item) for item in entry.get("actions") or []],
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                    result=entry.get("result"),
                    plan=str(entry.get("plan", "") or ""),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result=entry.result,
                    plan=entry.plan,
                )
            )
        if transitions and isinstance(last_action_result, dict) and last_action_result:
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _explain_missing_tool_name(exc):
        if not isinstance(exc, NameError) or _WORKSPACE is None:
            return ""
        missing = str(getattr(exc, "name", "") or "")
        if missing == "module":
            return "`module` is bound only inside a test string. Tool code reaches a saved module with load_module('name')."
        if missing in _WORKSPACE.modules:
            return f"'{missing}' is a saved module. Bind it first, for example {missing} = load_module('{missing}')."

        return ""


    def _describe_nested_error(text):
        lines = str(text).splitlines()
        if lines[:1] == ["Traceback (most recent call last):"]:
            lines = lines[1:]
        detail = [
            line for line in lines
            if not line.startswith('  File "') and not line.startswith("  [")
        ]
        if detail and ": " in detail[0]:
            detail[0] = detail[0].split(": ", 1)[1]

        return "\n".join(detail)


    MAX_TRACEBACK_FRAMES = 24


    def _format_frames(frames):
        '''Keep the entry and the failing end of a deep call chain, so the exception line and the
        notes after it still fit the tool output budget instead of being cut off ahead of them.'''
        lines = [f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}' for frame in frames]
        if len(lines) <= MAX_TRACEBACK_FRAMES:
            return lines
        kept = MAX_TRACEBACK_FRAMES // 2

        return lines[:kept] + [f"  [{len(lines) - 2 * kept} frames omitted]"] + lines[-kept:]


    def _sanitize_exception(exc, validation=False):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [
            frame for frame in extracted
            if frame.filename == "<python_tool>" or frame.filename.startswith(("<module:", "<tests:"))
        ]
        lines = []
        if user_frames:
            lines.extend(_format_frames(user_frames))
        elif not validation:
            lines.extend(
                f'  File "<python_tool>", line {frame.lineno}, in {frame.name}'
                for frame in extracted[-1:]
            )
        nested = str(getattr(exc, "nested_traceback", "") or "").splitlines()
        if nested:
            lines.extend(nested[1:] if nested[:1] == ["Traceback (most recent call last):"] else nested)
        else:
            lines.append(f"{exc.__class__.__name__}: {exc}")
        if lines and lines[0].startswith('  File "'):
            lines.insert(0, "Traceback (most recent call last):")
        try:
            notes = getattr(exc, "__notes__", [])
            if not isinstance(notes, (list, tuple)):
                notes = [notes] if notes is not None else []
            lines.extend(str(note) for note in notes)
        except Exception:
            pass

        return "\n".join(lines)


    class _ForbiddenModule:
        def __init__(self, name):
            self._name = name

        def __getattr__(self, attribute):
            raise ImportError(f"Module '{self._name}' is not allowed in the sandbox.")

        def __repr__(self):
            return f"<unavailable module '{self._name}'>"


    _WORKSPACE = None


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root in SAFE_MODULES:
            return builtins.__import__(name, globals, locals, fromlist, level)
        saved = _WORKSPACE.find_module(root) if _WORKSPACE is not None else None
        if saved is not None:
            return saved

        return _ForbiddenModule(str(name))


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_AS", None), 768 << 20),
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
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
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def _report_validation_failure(error):
        _send({"type": "validation_failure", "error": error})


    def _request_module_validation(name, source, tests, test_states=None, modules=None):
        _send({
            "type": "validate_module", "name": name, "source": source, "tests": tests,
            "test_states": test_states if test_states is not None else [None] * len(tests),
            "modules": modules or {},
        })
        reply = _recv()
        if reply.get("type") != "validation_result":
            raise RuntimeError("Invalid validation response from sandbox host.")
        if reply.get("stdout"):
            print(reply["stdout"], end="")
        if reply.get("error"):
            exception_class = getattr(builtins, reply.get("error_type", ""), RuntimeError)
            if not isinstance(exception_class, type) or not issubclass(exception_class, Exception):
                exception_class = RuntimeError
            detail = _describe_nested_error(reply["error"])
            try:
                error = exception_class(detail)
            except Exception:
                error = RuntimeError(detail)
            error.nested_traceback = reply["error"]
            raise error

        return {**reply["result"], "test_states": reply["test_states"]}


    def main():
        initial = _recv()
        global COLOR_CHARS, GRID_DECODE_TABLE
        COLOR_CHARS = str(initial.get("color_chars") or "")
        GRID_DECODE_TABLE = bytes.maketrans(COLOR_CHARS.encode("ascii"), bytes(range(len(COLOR_CHARS))))
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["game_overs"] = _game_overs_from_payload(state_payload.get("game_overs"))
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result

        def action(actions, **unsupported):
            if unsupported:
                named = ", ".join(f"`{name}`" for name in sorted(unsupported))
                belongs = "belongs" if len(unsupported) == 1 else "belong"
                raise TypeError(
                    f"action() takes the list of actions and nothing else. "
                    f"{named} {belongs} beside `code` on the `python` tool call, not here."
                )
            normalized_actions = _normalize_actions(actions)
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            return action_result

        global _WORKSPACE

        runtime_globals["action"] = action
        _refresh_state(initial.get("state") or {})
        workspace = PythonWorkspace(
            initial.get("modules") or {}, runtime_globals, _request_module_validation, _refresh_state, SAFE_MODULES,
            report_validation_failure=_report_validation_failure if initial.get("validation") is not None else None,
        )
        _WORKSPACE = workspace
        runtime_globals.update({
            "save_module": workspace.save_module,
            "load_module": workspace.load_module,
            "read_module": workspace.read_module,
        })

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                if initial.get("validation") is not None:
                    runtime_globals["result"] = workspace.run_tests(**initial["validation"])
                else:
                    exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "modules": workspace.modules,
                    "workspace_events": workspace.events,
                }
            )
        except Exception as exc:
            hint = "" if initial.get("validation") is not None else _explain_missing_tool_name(exc)
            if hint:
                try:
                    BaseException.add_note(exc, hint)
                except Exception:
                    pass
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc, validation=initial.get("validation") is not None),
                    "error_type": type(exc).__name__,
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)).replace(
    "__WORKSPACE_SOURCE__\n", inspect.getsource(_python_workspace)
)


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


@dataclass(frozen=True)
class EncodedJson:
    """JSON text that goes into a message to the sandbox as it is, so a payload the host keeps
    encoded between calls is not encoded again for every message."""

    text: str


def _append_json_parts(value: Any, parts: list[str]) -> None:
    """Append the JSON text of `value` to `parts`. Dicts are walked so that an `EncodedJson`
    inside them is appended verbatim, and every other value is encoded with `json.dumps`."""
    if isinstance(value, EncodedJson):
        parts.append(value.text)
    elif isinstance(value, dict):
        parts.append("{")
        for index, (key, item) in enumerate(value.items()):
            parts.append(f"{',' if index else ''}{json.dumps(str(key), ensure_ascii=False)}:")
            _append_json_parts(item, parts)
        parts.append("}")
    else:
        parts.append(json.dumps(value, ensure_ascii=False))


def _encode_json(value: Any) -> str:
    """`value` as JSON text, with every `EncodedJson` inside it spliced in verbatim. The parts
    are joined once, so a large spliced text is copied once rather than at every level."""
    parts: list[str] = []
    _append_json_parts(value, parts)

    return "".join(parts)


def _wait_for_pipe(selector: selectors.BaseSelector, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not selector.select(timeout=remaining):
        raise TimeoutError


def _send_json_line(handle: Any, payload: dict[str, Any], *, deadline: float) -> None:
    pending = memoryview((_encode_json(payload) + "\n").encode("utf-8"))
    with selectors.DefaultSelector() as selector:
        selector.register(handle, selectors.EVENT_WRITE)
        while pending:
            _wait_for_pipe(selector, deadline)
            try:
                written = os.write(handle.fileno(), pending)
            except BlockingIOError:
                continue
            pending = pending[written:]


def _read_sandbox_stdout(handle: Any, pending: bytearray, *, deadline: float) -> bytes | None:
    search_from = 0
    with selectors.DefaultSelector() as selector:
        selector.register(handle, selectors.EVENT_READ)
        while True:
            separator = pending.find(b"\n", search_from)
            if separator >= 0:
                line = bytes(pending[:separator])
                del pending[:separator + 1]

                return line
            _wait_for_pipe(selector, deadline)
            try:
                chunk = os.read(handle.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                line = bytes(pending) if pending else None
                pending.clear()

                return line
            search_from = len(pending)
            pending.extend(chunk)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """Signal the owned group before any wait/poll can release the leader's PID."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[bytes], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: float,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]] | None,
    modules: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    reported_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if reported_timeout_seconds is None:
        reported_timeout_seconds = timeout_seconds
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        sandbox_state = initial_state
        forbidden_request_error = ""
        validation_failure = ""
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        try:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))
            pending_stdout = bytearray()
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            _send_json_line(
                process.stdin,
                {
                    "code": code,
                    "timeout_seconds": timeout_seconds,
                    "sandbox_cwd": sandbox_dir,
                    "state": initial_state,
                    "color_chars": ARC_COLOR_CHARS,
                    "modules": modules or {},
                    "validation": validation,
                },
                deadline=deadline,
            )

            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                line = _read_sandbox_stdout(process.stdout, pending_stdout, deadline=deadline)
                if line is None:

                    return {
                        "error": "Sandbox process exited unexpectedly.",
                        "stdout": "",
                        "action_results": list(host_action_results),
                    }

                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("Expected a protocol object")
                except ValueError:

                    return {
                        "error": "Sandbox process returned an invalid response.",
                        "stdout": "",
                        "action_results": list(host_action_results),
                    }

                msg_type = str(message.get("type", "")).strip()
                if msg_type == "validation_failure" and validation is not None:
                    validation_failure = str(message["error"])
                    continue
                if msg_type == "validate_module":
                    if validation is not None:
                        forbidden_request_error = "Validation cannot start another validation."
                        validation_result = {"error": forbidden_request_error}
                    else:
                        test_states = list(message["test_states"])
                        if None in test_states:
                            snapshot = _python_workspace.encode_test_state(json.loads(_encode_json(sandbox_state)))
                            test_states = [state if state is not None else snapshot for state in test_states]
                        validation_result = run_sandboxed_python(
                            code="",
                            timeout_seconds=max(0.0, deadline - time.monotonic()),
                            initial_state={},
                            action_handler=None,
                            modules=message.get("modules") or {},
                            validation={
                                **{key: message.get(key) for key in ("name", "source", "tests")},
                                "test_states": test_states,
                            },
                            reported_timeout_seconds=reported_timeout_seconds,
                        )
                        if not validation_result.get("error"):
                            validation_result["test_states"] = test_states
                        if validation_result.get("timed_out"):

                            return {**validation_result, "action_results": list(host_action_results)}
                        validation_failure = str(validation_result.get("error") or "")
                    _send_json_line(process.stdin, {**validation_result, "type": "validation_result"}, deadline=deadline)
                    continue

                if msg_type == "action":
                    if action_handler is None:
                        forbidden_request_error = "Validation cannot execute environment actions."
                        _send_json_line(process.stdin, {"type": "action_error", "error": forbidden_request_error}, deadline=deadline)
                        continue
                    try:
                        action_result_payload = action_handler(list(message.get("actions") or []))
                    except Exception:  # noqa: BLE001
                        _send_json_line(
                            process.stdin,
                            {
                                "type": "action_error",
                                "error": "action failed in sandbox host.",
                            },
                            deadline=deadline,
                        )
                        continue
                    raw_action_result = action_result_payload.get("action_result") or {}
                    if isinstance(raw_action_result, dict):
                        host_action_results.append(dict(raw_action_result))
                    sandbox_state = action_result_payload.get("state") or {}
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_result",
                            "action_result": raw_action_result,
                            "state": sandbox_state,
                        },
                        deadline=deadline,
                    )
                    continue

                if msg_type in {"final", "error"}:

                    return {
                        "stdout": str(message.get("stdout", "") or ""),
                        "result": message.get("result"),
                        "error": forbidden_request_error or str(message.get("error", "") or ""),
                        "error_type": "RuntimeError" if forbidden_request_error else message.get("error_type", ""),
                        "action_results": list(message.get("action_results") or host_action_results),
                        "modules": message.get("modules"),
                        "workspace_events": message.get("workspace_events", []),
                    }

                return {
                    "error": "Sandbox process returned an unknown message type.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }
        except TimeoutError:
            error = f"Tool timed out after {reported_timeout_seconds}s"
            if validation_failure:
                validation_status = (
                    "Validation was incomplete. Last reported failures follow."
                    if validation is not None else "Last completed validation failed."
                )
                error += f"\n{validation_status}\n{validation_failure}"

            return {
                "error": error,
                "timed_out": True,
                "stdout": "",
                "action_results": list(host_action_results),
            }
        except (BrokenPipeError, ConnectionResetError):

            return {
                "error": "Sandbox process disconnected before the tool finished.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
        finally:
            _kill_process_group(process)
            _wait_for_process_exit(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                with suppress(OSError):
                    stream.close()
