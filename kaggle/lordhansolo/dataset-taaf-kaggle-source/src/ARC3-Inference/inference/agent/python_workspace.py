"""Per-game Python source checked against frozen observations before replacement.

This module is also embedded in the isolated Python subprocess. Only source and
test strings and serialized test states cross the process boundary.
"""

import base64
import gc
import json
import sys
import zlib
from types import ModuleType


MAX_FAILED_CHECKS = 20


def encode_test_state(state):
    """Keep frozen grids compact when passing modules between subprocesses."""
    serialized = json.dumps(state, separators=(",", ":")).encode("utf-8")

    return base64.b64encode(zlib.compress(serialized)).decode("ascii")


def decode_test_state(state):
    """Return fresh observations for one regression suite."""

    return json.loads(zlib.decompress(base64.b64decode(state)))


class PythonWorkspace:
    """Keep reusable code separate from live game state and real actions."""

    def __init__(self, modules, runtime_globals, validate_module, restore_state, reserved_names=(), report_validation_failure=None):
        self.modules = modules
        self.events = []
        self._runtime_globals = runtime_globals
        self._validate_module = validate_module
        self._restore_state = restore_state
        self._reserved_names = frozenset(reserved_names)
        self._loaded_modules = {}
        self._check_count = 0
        self._failed_checks = []
        self._failed_check_count = 0
        self._suite_check_count = 0
        self._test_suite_index = 0
        self._report_validation_failure = report_validation_failure
        self._refresh_names()

    def _refresh_names(self):
        self._runtime_globals["saved_modules"] = tuple(sorted(self.modules))

    def _execute_source(self, name, source):
        """Register the module before running its source, the way `sys.modules` does.

        Source that reaches back for its own name, directly or around a cycle of modules that
        load each other, then binds the half-built module instead of loading the name again and
        recursing until the stack runs out. A module whose source raises leaves no entry
        behind, so the cache never holds a half-built module past this call."""
        module = ModuleType(name)
        namespace = module.__dict__
        namespace["__builtins__"] = dict(self._runtime_globals["__builtins__"])
        namespace["load_module"] = self.load_module
        self._loaded_modules[name] = module
        try:
            exec(compile(source, f"<module:{name}>", "exec"), namespace, namespace)
        except Exception:
            self._loaded_modules.pop(name, None)
            raise

        return module

    def _raised_in_test_code(self, error):
        """Whether the innermost frame is a test string, the only scope where `module` is bound.

        A module function that reaches for a name of its own raises the same NameError, and
        advice about test code would send the reader to the wrong file."""
        innermost = error.__traceback__
        while innermost is not None and innermost.tb_next is not None:
            innermost = innermost.tb_next

        return innermost is not None and innermost.tb_frame.f_code.co_filename.startswith("<tests:")

    def _explain_missing_name(self, error, name):
        """Point a test's NameError at `module`, the only binding that reaches the source under test."""
        if not isinstance(error, NameError) or not self._raised_in_test_code(error):
            return ""
        missing = str(getattr(error, "name", "") or "")
        if missing == name:
            return " Test code reaches the module under test as `module`, never by its own name."
        module = self._loaded_modules.get(name)
        if missing and module is not None and hasattr(module, missing):
            reference = f"module.{missing}(...)" if callable(getattr(module, missing)) else f"module.{missing}"
            return f" Test code reaches that name as `{reference}`."

        return ""

    def _describe_label_misuse(self, actual, expected, label):
        """Name the likely misuse when a two-argument check compares a boolean with text.

        The comparison still reports its counterexample, because a module that returns a boolean
        where the test expects a string is refuted by exactly the same call."""
        if not label and isinstance(actual, bool) and isinstance(expected, str):
            return " A description belongs in the third argument, check(actual, expected, label)."

        return ""

    def _check_equal(self, actual, expected, label=""):
        """Remember failed or interrupted comparisons even when test code catches them."""
        self._check_count += 1
        self._suite_check_count += 1
        prefix = f"Test suite {self._test_suite_index}. Check {self._suite_check_count}"
        try:
            if actual == expected:

                return
            detail = (
                f"{prefix} {str(label)[:160]} failed. "
                f"Expected {repr(expected)[:300]}, observed {repr(actual)[:300]}."
                f"{self._describe_label_misuse(actual, expected, label)}"
            ) if len(self._failed_checks) < MAX_FAILED_CHECKS else ""
        except BaseException:
            self._record_failed_check(f"{prefix} could not compare values.")
            raise
        self._record_failed_check(detail)

    def _record_failed_check(self, detail):
        """Keep bounded examples and publish them before subsequent test code can hang."""
        self._failed_check_count += 1
        if len(self._failed_checks) >= MAX_FAILED_CHECKS:

            return
        frame = sys._getframe(2)
        while frame is not None and not frame.f_code.co_filename.startswith("<tests:"):
            frame = frame.f_back
        if frame is not None:
            detail += f' File "{frame.f_code.co_filename}", line {frame.f_lineno}.'
        del frame
        self._failed_checks.append(detail)
        if self._report_validation_failure is not None:
            self._report_validation_failure(self._describe_failed_checks(partial=True))

    def _describe_failed_checks(self, partial=False, stop_reason=""):
        """Put the summary and any stop reason ahead of the counterexamples that tool output trimming cuts first."""
        omitted = self._failed_check_count - len(self._failed_checks)
        if partial:
            summary = (
                f"At least {self._failed_check_count} checks failed. "
                f"Showing the first {len(self._failed_checks)} recorded failures. "
                "Later checks and failures may be missing from this partial report."
            )
        else:
            summary = f"{self._failed_check_count} of {self._check_count} checks failed."
            if omitted:
                summary += f" Showing the first {len(self._failed_checks)} failures; {omitted} omitted."
        if stop_reason:
            summary += f"\n{stop_reason}"

        return (
            summary + "\nUse module functions with explicit observation arguments. "
            "This failed save made no workspace changes. When a failed suite pins an "
            "interface you meant to change, save again with replace_tests=True and the tests "
            "the new interface deserves.\n" + "\n".join(self._failed_checks)
        )

    def save_module(self, name, source, tests="", replace_tests=False):
        """Run old and new tests with no action callback, then replace the source.

        `replace_tests` drops the suites the module already carries, which is how an
        interface change gets past the regression net the older suites form. The whole save
        still applies or does not, so a refused replacement leaves the stored module alone."""
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError("Module name must be a public Python identifier.")
        if name in self._reserved_names:
            raise ValueError(f"Module name '{name}' is taken by an importable module. Pick another name.")
        if not isinstance(source, str) or not source.strip() or not isinstance(tests, str):
            raise ValueError("Pass nonempty source and a test string.")
        if not isinstance(replace_tests, bool):
            raise ValueError("replace_tests must be True or False.")
        stored = {} if replace_tests else self.modules.get(name, {})
        suites = list(stored.get("tests", []))
        test_states = list(stored.get("test_states", []))
        if tests.strip() and tests not in suites:
            suites.append(tests)
            test_states.append(None)
        if not suites:
            raise ValueError("A save needs at least one test suite using check(actual, expected).")
        validation = self._validate_module(name, source, suites, test_states, dict(self.modules))
        self.modules[name] = {"source": source, "tests": suites, "test_states": validation.pop("test_states")}
        self._loaded_modules.pop(name, None)
        self._refresh_names()
        event = {"saved": name, **validation}
        self.events.append(event)

        return event

    def run_tests(self, name, source, tests, test_states):
        """Run in a separate process whose host cannot execute environment actions.

        Each suite decodes its snapshot into fresh observations, so its namespace can
        share those objects without copying the history again. Mutations stay within
        that suite's observations and never reach the live tool process.

        Clear loaded modules before each suite so dependencies also start fresh.
        Imports within a suite still share module instances, including cycles.
        Between suites, drop observations and collect cycles before allocating the
        next snapshot and modules, so old module payloads do not accumulate. Discard
        modules reloaded by finalizers too, and retain the old suite's check context
        until cleanup finishes."""
        self._check_count = 0
        self._failed_checks = []
        self._failed_check_count = 0
        for index, suite in enumerate(tests, start=1):
            try:
                self._loaded_modules.clear()
                if index > 1:
                    self._restore_state({})
                    gc.collect()
                    self._loaded_modules.clear()
                self._test_suite_index = index
                self._suite_check_count = 0
                self._restore_state(decode_test_state(test_states[index - 1]))
                module = self._execute_source(name, source)
                namespace = {
                    key: self._runtime_globals[key]
                    for key in (
                        "current_frame", "previous_frame", "history", "transitions",
                        "last_transition", "game_overs", "last_action_result", "valid_actions",
                    )
                }
                namespace.update({
                    "__builtins__": dict(self._runtime_globals["__builtins__"]),
                    "module": module,
                    "load_module": self.load_module,
                    "check": self._check_equal,
                })
                previous_count = self._check_count
                exec(compile(suite, f"<tests:{name}:{index}>", "exec"), namespace, namespace)
                if self._check_count == previous_count:
                    raise ValueError(f"Test suite {index} executed no check(actual, expected).")
                del namespace, module
            except Exception as error:
                if self._failed_checks:
                    try:
                        stop_reason = (
                            f"Validation stopped in test suite {index} after the failed checks. "
                            f"{type(error).__name__}: {str(error)[:300]}{self._explain_missing_name(error, name)}"
                        )
                    except Exception:
                        stop_reason = ""
                    failure = AssertionError(self._describe_failed_checks(stop_reason=stop_reason))
                    raise failure.with_traceback(error.__traceback__) from None
                try:
                    BaseException.add_note(
                        error,
                        f"Module {name!r}, test suite {index} failed.{self._explain_missing_name(error, name)} "
                        "Use module functions with explicit observation arguments. "
                        "This failed save made no workspace changes. When the suite that failed pins an "
                        "interface you meant to change, save again with replace_tests=True and the tests "
                        "the new interface deserves."
                    )
                except Exception:
                    pass
                raise

        if self._failed_checks:
            raise AssertionError(self._describe_failed_checks())

        return {"checks_passed": self._check_count, "test_suites": len(tests)}

    def _require_module(self, name):
        """Refuse an unknown name with the names that do exist.

        The turn prompt lists each saved name with the callables its source defines, which is
        easy to copy whole, so the name a caller passes can carry that suffix."""
        if name not in self.modules:
            raise ValueError(f"No saved module named {name!r}. Saved modules are {tuple(sorted(self.modules))}.")

        return self.modules[name]

    def load_module(self, name):
        """Load lazily into an isolated namespace; callers pass current state explicitly."""
        if name not in self._loaded_modules:
            self._execute_source(name, self._require_module(name)["source"])

        return self._loaded_modules[name]

    def find_module(self, name):
        """Return the module an import of this name resolves to, or None when no such module exists."""
        if name in self._loaded_modules:
            return self._loaded_modules[name]
        if name in self.modules:
            return self.load_module(name)

        return None

    def read_module(self, name):
        """Return editable source and test text without exposing the stored records."""
        record = self._require_module(name)

        return {"source": record["source"], "tests": list(record["tests"])}
