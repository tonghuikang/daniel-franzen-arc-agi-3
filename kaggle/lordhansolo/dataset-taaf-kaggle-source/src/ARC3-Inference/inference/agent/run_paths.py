"""Where a run keeps each game's files.

The solver builds every path forwards from the run directory and the game stem
``<game>_p<pass>``; the analyzer and the metrics recorders only receive the
runtime-state path, so its inverse lives here too, next to the layout it undoes::

    <run>/artifacts/<stem>_tool_runtime_state.json   the handle; the harness passes the state in memory and never writes it
    <run>/artifacts/<stem>_viewer_data.json
    <run>/transcripts/<stem>.txt
    <run>/solver_analysis/<stem>.html
    <run>/prompts/<stem>.log                         latest model-call snapshot
    <run>/<stem>_requests.jsonl                      request log, when enabled
    <run>/metrics/<stem>_metrics.jsonl
    <run>/metrics/vllm_server.jsonl                  run-level, see vllm_metrics

Every per-game file is named after the stem alone, so the paths are pure
functions of the run root and the stem: nothing here looks at the filesystem,
and a game's files can never move while it plays.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from inference.agent.runtime_state import RUNTIME_STATE_FILENAME

_ARTIFACTS_DIR = "artifacts"
_TRANSCRIPTS_DIR = "transcripts"
_ANALYSIS_DIR = "solver_analysis"
_PROMPTS_DIR = "prompts"
_METRICS_DIR = "metrics"
_RUNTIME_STATE_SUFFIX = f"_{RUNTIME_STATE_FILENAME}"
_GAME_ID_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def metrics_dir(run_root: Path) -> Path:
    return run_root / _METRICS_DIR


@dataclass(frozen=True)
class GamePaths:
    run_root: Path
    game_stem: str

    @classmethod
    def for_game(cls, run_root: Path, game_id: str, pass_index: int) -> "GamePaths":
        return cls(run_root=run_root, game_stem=f"{_GAME_ID_UNSAFE_CHARS.sub('_', game_id)}_p{pass_index}")

    @classmethod
    def from_state_path(cls, state_path: Path) -> "GamePaths":
        """Inverse of `state_path`: the run root is two levels up and the stem
        is the filename without the runtime-state suffix."""
        return cls(
            run_root=state_path.parent.parent,
            game_stem=state_path.name.removesuffix(_RUNTIME_STATE_SUFFIX),
        )

    @property
    def state_path(self) -> Path:
        return self.run_root / _ARTIFACTS_DIR / f"{self.game_stem}{_RUNTIME_STATE_SUFFIX}"

    @property
    def viewer_data_path(self) -> Path:
        return self.run_root / _ARTIFACTS_DIR / f"{self.game_stem}_viewer_data.json"

    @property
    def transcript_path(self) -> Path:
        return self.run_root / _TRANSCRIPTS_DIR / f"{self.game_stem}.txt"

    @property
    def analysis_html_relpath(self) -> str:
        return f"{_ANALYSIS_DIR}/{self.game_stem}.html"

    @property
    def metrics_log_path(self) -> Path:
        return metrics_dir(self.run_root) / f"{self.game_stem}_metrics.jsonl"

    @property
    def prompt_log_path(self) -> Path:
        return self.run_root / _PROMPTS_DIR / f"{self.game_stem}.log"

    @property
    def request_log_path(self) -> Path:
        return self.run_root / f"{self.game_stem}_requests.jsonl"
