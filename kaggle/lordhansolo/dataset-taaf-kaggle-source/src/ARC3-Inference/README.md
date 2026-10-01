# The Duck 🦆

The Duck is the ARC3 inference harness in this repo: a tool-using solver that
plays ARC-AGI-3 games through TAAF.

It ties together:

- TAAF `Benchmark` / `GameAPI` execution
- a local OpenAI-compatible vLLM server, or OpenRouter
- the duck's single `python` tool with per-game saved modules
- structured run artifacts for scoring, viewing, and trace export

The Python package lives under `inference/`. The run viewer lives under
`viewer/`.

## Quick Start

You need Python 3.12 and `uv`.

```bash
make install
```

Run with the default local vLLM config:

```bash
make server
make interactive
```

Submit the default Slurm run:

```bash
make sbatch
```

Run through OpenRouter instead:

```bash
export OPENROUTER_API_KEY=<your-openrouter-api-key>
CONFIG_PATH=configs/inference.openrouter.json make interactive
```

Open the viewer:

```bash
make view
```

If your runs are under the checked-in default experiment root, point the viewer
there:

```bash
make view VIEW_RUNS_DIR=/shared/arc_3_results/$USER
```

## What The Duck Does

For each TAAF game run, the harness starts a `HarnessSolver`. The solver gives
the duck the latest game state, valid actions, history, and a Python tool. The
duck inspects the board, writes small bits of code to reason about it, and calls
`action(...)` from inside Python to execute real game actions.

The duck can use:

- `current_frame.ascii` for a compact symbolic grid
- `current_frame.segmentation` for connected components, object hashes,
  boundaries, containment, and adjacency
- `history`, `previous_frame`, `transitions`, and `last_transition` for
  before/after reasoning
- `valid_actions` for the current action set
- `last_action_result` for fields such as `changed_pixels`, `total_pixels`,
  `largest_changed_regions`, `changed_region_count`, `level_completed`, `game_over`,
  and `run_complete`

The raw numeric grid is intentionally hidden from the Python tool. The preferred
view is `current_frame.segmentation`; `current_frame.ascii` is there for small
local checks.

### Checked Python modules (MVP)

The Python tool can save reusable source with small regression tests. This is
enabled in the duck by default and needs no additional model requests. For
example, one tool call can save and use a planner's goal predicate:

```python
save_module(
    "goals",
    "def is_goal(mask, count):\n    return mask == (1 << count) - 1\n",
    "check(module.is_goal(3, 2), True)\ncheck(module.is_goal(1, 2), False)",
)
goals = load_module("goals")
result = goals.is_goal(3, 2)
```

`save_module(name, source, tests='', replace_tests=False)` reruns all previous
test strings and appends new ones before replacing the module. Each suite must
execute at least one `check(actual, expected, label='')`; a mismatch rejects
the update. Tests call the tested module through `module`, including
assignments such as `module.value = 7`. Test variables and runtime
observations are separate from the module's globals, just as they are when
using `load_module`.
Pass observations explicitly to module functions. Validation uses a separate subprocess
whose host rejects environment actions, including requests reached through
Python introspection. Each new test string captures a compressed snapshot of the
host's runtime state when it is added, including history and available animation
frames. Old tests always use their original observations; new tests use the state
at the new save. Repeating the same test string keeps its original snapshot.
These snapshots consume host memory and subprocess transfer time, but are not
included in model prompts or `read_module` results. Test actual module outputs
against observed transitions and exercise the planner and goal predicate;
`check(True, True)` provides no validation. Passing comparisons confirms only
the supplied examples.

`load_module(name)` loads functions and constants into a separate namespace.
Pass live frames and other state explicitly to functions. `saved_modules`
lists available names and `read_module(name)` returns source and test strings.
An update can edit the saved source programmatically instead of printing and
regenerating it. When intentionally changing the interface, save with
`replace_tests=True`, which drops the suites the module already carries and
keeps only the ones passed with it. A save applies in full or not at all, so a
refused replacement leaves the stored module exactly as it was.

Module source and test strings both receive `load_module`, so one module can
build on another and a test can compare against a module it did not save.
`import <saved name>` returns the same object. Asking for the module being
saved returns the source under test rather than the stored copy, whether the
question comes from its own tests or from a module that loads it. Each module
is registered before its source runs, the way `sys.modules` is, so source that
reaches back for its own name, directly or around a cycle, binds the
half-built module instead of loading the name again; source that raises leaves
no entry for the next load to find.

Source, tests and their snapshots survive turns, context trimming, game overs and level changes
within the same agent session. Live Python variables do not persist. A new
game starts empty; process restarts do not restore this in-memory workspace.
All module changes commit only when the entire tool call succeeds. Uncaught errors or
timeouts discard those changes, while real actions already executed remain
executed. Tests run only on save, within the existing tool timeout. Each turn
prompt lists the saved names with the public callables their sources define;
save results appear under `workspace` in tool results. This MVP does not
require a complete simulator or automatically validate action batches.

The model-facing actions are:

- `UP`, `DOWN`, `LEFT`, `RIGHT`
- `SPACE`
- `MOUSE(row=..., col=...)`

`MOUSE` uses `row` and `col`. Legacy `x` / `y` mouse fields are rejected.

Every Python tool call starts fresh. It can import a small allowlist of standard
library modules, print compact summaries, assign a final value to `result`, and
call `action(...)` once or many times. The tool call timeout defaults to 30
seconds.

## Configuration

The local vLLM profile uses three context thresholds. `server.max_model_len`
is 81,920 tokens for input plus output. `shared.context_window` is 73,728;
the agent subtracts its reply reserve and safety margin, giving a 72,704-token
estimated input ceiling with the default unlimited-output setting.
`analyzer.target_context` is 55,296 estimated input tokens, including the
system prompt, tools and retained history. Only after crossing the ceiling does
the agent remove oldest turns until it reaches this target. The newest user
turn is always retained, so an oversized current turn can exceed the target.
Below the ceiling, retained user messages and tool-call arguments remain
unchanged to preserve the reusable prefix. Once compaction runs, it removes
complete oldest turns and leaves the retained ones untouched.

A positive `target_context` replaces the separate 30-assistant-reply history
cap, allowing the prefix to remain stable between token-based cuts. Omitting
the setting or using `0` preserves the previous ceiling-based trimming and
30-reply cap. The setting is exported as `LOCAL_ANALYZER_TARGET_CONTEXT`,
including in the generated Kaggle setup. These token estimates are not a
guarantee of server-side prompt length; overflow retries remain enabled.

The main config is strict JSON. Comments are not supported.

- `configs/inference.json` is the default local-vLLM / Slurm config.
- `configs/inference.openrouter.json` uses OpenRouter.
- `configs/eval.json` selects runs for `make eval`.
- `configs/significance.json` selects score files for `make significance`.

Use a different config with:

```bash
CONFIG_PATH=/path/to/config.json make interactive
CONFIG_PATH=/path/to/config.json make sbatch
```

Useful sections in `configs/inference.json`:

- `shared.*`: model name, base URL, provider, and context window.
- `experiments.root_dir`: where timestamped run directories are written.
- `environment.*`: games, tags, passes, concurrency, and runtime limits.
- `deployment.*`: inline vs Slurm and source repos bundled into Slurm jobs.
- `deployment.slurm.*`: GPU, walltime, partition, local-server startup, and
  extra `sbatch` flags.
- Kaggle runs are configured by CLI/Make overrides because the notebook slug
  and source dataset are usually per run.
- `server.*`: vLLM model-serving settings.
- `analyzer.*`: duck sampling/tool settings. This key is still named
  `analyzer` for compatibility with existing code and configs.
- `chat.*`: direct chat probing with `make chat`.
- `viewer.port`: default viewer port.
- `multimodal.*`: image context for the current grid.

The checked-in default config currently runs the official tagged game set with
20 passes, 45 minutes per game, and `concurrent_jobs=16`. On Slurm it requests
two B200 GPUs and starts one local vLLM server per GPU. In that mode
`concurrent_jobs` is interpreted per GPU/server, so the effective concurrency is
32.

## Running Games

Run inline in the current process:

```bash
make interactive
```

Submit to Slurm:

```bash
make sbatch
```

Launch the validated duck harness on Kaggle:

```bash
make kaggle-duck \
  RUN_NAME=duck-harness-20260527 \
  KAGGLE_KERNEL_SLUG=taaf-duck-harness-20260527 \
  KAGGLE_DATASET_REF=driessmit1/taaf-kaggle-source-duck-harness-20260527
```

Name a run:

```bash
make sbatch RUN_NAME=baseline-qwen
```

Run one game or short-prefix:

```bash
make interactive GAME=taps N_PASSES=1 MAX_RUNTIME_MINUTES=10
```

Run the official tag set with a whole-experiment cap:

```bash
make sbatch GAME=[] GAME_TAGS=official MAX_EXPERIMENT_RUNTIME_MINUTES=360
```

Run the duck locally against TAAF's competition Arcade simulator:

```bash
make interactive \
  GAME=[] GAME_TAGS=official \
  SIMULATE_COMPETITION_ARCADE=true \
  COMPETITION_CLONE_RUNS=110 \
  N_PASSES=1
```

The simulator is inline-only. `COMPETITION_CLONE_RUNS=110` repeats the selected
25 official games with unique competition-safe IDs, which catches submission
Arcade issues without waiting for a Kaggle rerun.

Kaggle test runs automatically collect additional prefix-cache diagnostics:

- `metrics/*_metrics.jsonl`: `cache_request` records connect the game and request
  index to the server request ID and hash each message, tool schema and template options.
- `metrics/vllm_cache_diagnostics_<pid>.jsonl`: actual server token hashes in
  256-token chunks, multimodal-aware cache block hashes, independent cache-group
  hits with/without the MTP block drop, the effective hit and evicted block hashes.
  Token chunk hashes are recorded on the first lookup of each request; subsequent
  lookups retain the request ID. Each server process has a separate file.
- `metrics/vllm_server.jsonl`: also receives vLLM's sampled KV block lifetime,
  idle-time and reuse-gap metrics (10% sampling).
- `metrics/host_memory.jsonl`: samples available host RAM every 15 seconds from
  server startup through inference and restarts. `available_bytes` includes the
  cgroup limit; the record also shows host and cgroup availability separately.

The hooks are installed only for explicit test mode (`TAAF_RUN_AS_SUBMISSION=0`).
Competition reruns veto them even if the test flag or diagnostic environment
variables are set. Missing or unrecognized mode values disable the new diagnostics.
Submission processes do not install the plugin or enable the extra server metrics;
the plugin and agent independently check the mode before collecting anything.
Tracing adds test-only CPU/I/O overhead and does not change cache allocation or
the selected hit. The runtime dataset does not need to be rebuilt: test setup
installs the observational plugin into the freshly unpacked runtime.

For a bounded PyTorch CPU/CUDA trace of the vLLM worker, opt in on a test run:

```bash
make kaggle-duck KAGGLE_VLLM_PROFILE=true KAGGLE_RUN_AS_SUBMISSION=false
```

The profiler starts after the API smoke test, skips five engine iterations,
captures twenty, and writes compressed traces under `vllm-profile/` in the run
output. It is disabled by default. Submission mode, a real Kaggle competition
rerun, or an unknown run mode vetoes both `--profiler-config` and the profiling
endpoint call even when `KAGGLE_VLLM_PROFILE=true` is present.

Common overrides:

- `GAME`: one game, comma-separated games, or a JSON list.
- `GAME_TAGS`: include tags such as `official`.
- `EXCLUDE_GAME_TAGS`: exclude tags.
- `N_PASSES`: TAAF passes per selected game.
- `CONCURRENT_JOBS`: TAAF concurrency. With Slurm local servers this is per
  GPU/server.
- `MAX_ACTIONS`: optional per-game action cap.
- `MAX_RUNTIME_MINUTES`: per-game wall-clock cap.
- `MAX_EXPERIMENT_RUNTIME_MINUTES` or `MAX_EXPERIMENT_RUNTIME_HOURS`: whole-run
  wall-clock budget. If the per-game cap is unset, the runner derives it from
  the number of games, passes, and effective concurrency.
- `EXPERIMENTS_DIR`: base directory for timestamped runs.
- `EXPERIMENT_DIR`: exact output directory for one run.

List resolved official games without running them:

```bash
uv run --no-sync inference-taaf-run --include-tags official --list-games
```

## Local vLLM

Start the server:

```bash
make server
```

Check or stop it:

```bash
make check-server
make stop-server
```

The default local base URL is `http://127.0.0.1:1234/v1`. `make server`
generates a local server API key unless `SERVER_REQUIRE_API_KEY=false`.

On cluster machines, the Makefile moves Hugging Face, Torch, Triton, and related
caches under `/shared/<user>` when that directory exists.

## Slurm Flow

`make sbatch` runs through TAAF's Slurm deployment. The job directory includes
the run config, generated Slurm script, dependency override file, benchmark
artifacts, diagnostics, and logs.

For local-vLLM Slurm runs, the solver starts local server processes inside the
allocation before the duck begins playing. Each run gets its own API key and
run-scoped localhost ports, so a run either talks to its own server or fails
fast. The server is started from the bundled `src/ARC3-Inference` snapshot in
the run directory, using the worker's per-run virtualenv.

`deployment.source_repos` is bundled into the Slurm job directory. The worker
uses a generated dependency override file so it installs those bundled repos
instead of fetching private dependencies from GitHub.

## Kaggle Flow

`make kaggle-duck` runs through TAAF's Kaggle deployment. It packages the
current TAAF and ARC3-Inference sources into a Kaggle source dataset, pushes a
private Kaggle notebook, and attaches the duck solver's declared model and vLLM
wheelhouse datasets. The duck solver owns the Kaggle setup hooks, so the
launcher only chooses the run shape.

By default the target uses the 16 public games from the duck harness validation,
`model=local`, 16 concurrent games, 75 minutes per game, and a 90-minute Kaggle
runtime. Add `DEPLOYMENT_WAIT=true` if you want the command to block and pull
the finished Kaggle output back into the run directory.

The equivalent direct CLI form is:

```bash
uv run --no-sync inference-taaf-run \
  --deployment-target kaggle \
  --kaggle-duck-public-harness \
  --agent duck-harness \
  --model local \
  --run-name duck-harness-20260527 \
  --kaggle-kernel-slug taaf-duck-harness-20260527 \
  --kaggle-dataset-ref driessmit1/taaf-kaggle-source-duck-harness-20260527 \
  --max-runtime-minutes 75 \
  --max-experiment-runtime-minutes 90 \
  --concurrent-jobs 16 \
  --analyzer-timeout 900
```

## Run Artifacts

Each run writes a timestamped directory under `experiments.root_dir`, or under
`EXPERIMENTS_DIR` / `EXPERIMENT_DIR` when overridden.

Important files include:

- `run_config.json`: resolved games, passes, concurrency, runtime caps, model,
  Slurm settings, and hardware metadata.
- `benchmark.json`: saved TAAF benchmark and per-game `GameRun` state.
- `diagnostics.html`: TAAF diagnostics.
- `artifacts/*_viewer_data.json`: compact viewer payloads.
- `artifacts/*_events.jsonl`: append-only full viewer event sidecars.
- duck transcript HTML/text files linked from the viewer.
- `stdout.log` and `stderr.log` for Slurm jobs.
- `<game>_p<pass>_requests.jsonl` files when `analyzer.save_request_logs` is true.

## Viewer

Start the viewer on the default port from `configs/inference.json`:

```bash
make view
```

Override the port:

```bash
make view VIEW_PORT=8012
```

Point at a run root:

```bash
make view VIEW_RUNS_DIR=/shared/arc_3_results/$USER
```

Point at one exact run:

```bash
make view VIEW_RUN_DIR=/shared/arc_3_results/$USER/<run-name>
```

The viewer shows run summaries, per-game progress, boards, actions,
level transitions, and the duck's transcript.

## Scoring

Score one run directory:

```bash
make score_run SCORE_RUN_DIR=/path/to/run
```

Evaluate runs from `configs/eval.json`:

```bash
make eval
```

Write a score file somewhere specific:

```bash
make score_run SCORE_RUN_DIR=/path/to/run SCORE_OUTPUT_PATH=docs/candidate-score.json
```

The scorer reads TAAF `benchmark.json`, uses persisted `final_score` values when
present, and otherwise asks TAAF's `GameRun` scorer to compute the score from
the saved state. It writes `evaluation.json` plus the lightweight `score.json`
format used by significance checks.

## Significance

Compare a candidate score file against a current best:

```bash
make significance \
  BASELINE_SCORE=docs/current-best-score.json \
  CANDIDATE_SCORE=docs/candidate-score.json
```

Or configure those paths in `configs/significance.json` and run:

```bash
make significance
```

The comparison aligns by `game_id`, averages repeated trials within each game,
and uses games as the paired unit. It checks runtime budget, hardware, dataset
metadata, and trial counts before reporting whether the candidate passes the
internal-highscore threshold:

```text
P(true_delta > 0 | results) >= 0.90
```

The output also includes win rate, a bootstrap 90% interval, and TAAF paired
test p-values as robustness checks.

## Trace Export

Export machine-readable per-episode duck traces:

```bash
make traces
```

For runs outside `runs/`, call the tool directly:

```bash
uv run --no-sync inference-traces --runs-dir /shared/arc_3_results/$USER
```

Traces are written in live-chat `messages` format. They preserve assistant
reasoning, tool calls, compact tool results, actions, scores, and level
transitions linked back to message indices.

## Useful Commands

- `make install`: create `.venv` and install all locked dependencies.
- `make prepare-ci`: run Ruff and the test suite.
- `make server`: start local vLLM.
- `make interactive`: run through TAAF inline deployment.
- `make sbatch`: submit through TAAF Slurm deployment.
- `make chat PROMPT="..."`: send a direct chat probe to the configured model.
- `make view`: serve the run viewer.
- `make score_run SCORE_RUN_DIR=...`: score one saved run.
- `make eval`: score runs selected by an eval config.
- `make significance`: compare two score files.
- `make traces`: export trace JSON.
- `make zip`: zip the local `runs/` directory.
- `make clean`: remove local `runs/` artifacts.

## Repo Map

- `inference/framework/run.py`: CLI entry point and TAAF deployment setup.
- `inference/framework/solver.py`: TAAF solver adapter, action execution,
  viewer events, transcripts, and local-server orchestration.
- `inference/agent/tool_agent.py`: OpenAI-compatible tool-calling duck.
- `inference/agent/python_tool_sandbox.py`: isolated Python tool runtime.
- `inference/utils/segmentation.py`: connected-component board segmentation.
- `inference/tools/eval.py`: TAAF score export.
- `inference/tools/significance.py`: paired score comparison.
- `inference/tools/traces.py`: trace export.
- `viewer/`: local browser UI for saved runs.
- `tests/`: unit coverage for config, duck runtime, TAAF runner, viewer,
  scoring, significance, and traces.
