# Serving capacity comparison: three ARC-AGI-3 Milestone 2 notebooks

All three run the Tufa ARC-AGI Framework (TAAF) harness, serve Qwen3.8-Flash-Next, and
target a single RTX PRO 6000 (96 GB).

For the one-table overview, see [SUMMARY.md](SUMMARY.md).

## Overview

| | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Public LB score (2026-09-30) | 27.89 | 23.84 | 22.53 |
| Kernel | `arc-agi-3-milestone-2-solution` | `arc-agi-3-milestone-2` | `arc-agi-3-duck-18-1gc-submit` |
| Local copy | `kaggle/dfranzen/` | `kaggle/lordhansolo/` | `kaggle/sirikilohit/` |
| Cells (code) | 23 (11) | 10 (9) | 50 (27) |
| Serving config location | inline, cells 4 / 12 / 16 | dataset bundle | inline, cells 3 / 9 / 11 / 16 |
| Server | [SGLang Pennyroyal 2.5.3](https://www.kaggle.com/datasets/dfranzen/pennyroyal-v253) | [vLLM 0.29.1rc1 nightly e975732](https://www.kaggle.com/datasets/lordhansolo/vllm-main-e975732-arc3) | [SGLang Pennyroyal 2.5.0](https://www.kaggle.com/datasets/sirikilohit/sglang-penny-build-qwen) |
| Quantisation | AutoRound W4A16 | NVFP4 + FP8 mixed | AutoRound W4A16 + FP8 PLE |
| Speculative decoding | NEXTN, 3 steps (num-draft-tokens 4) | MTP, 3 draft tokens | NEXTN, 3 steps (num-draft-tokens 4) |

Notes

- dfranzen's notebook is this repo's competition notebook; its 499 KB harness patch is cell 2.
- lordhansolo's harness links point at its source bundle copied into `kaggle/lordhansolo/`,
  because it has no GitHub repo and Kaggle cannot deep-link nested dataset files.
- Server links point at the Kaggle dataset that ships each runtime. sirikilohit additionally
  borrows the CUDA 13 toolkit from `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`.
- lordhansolo's serving config is not in the notebook. It lives in `setup_commands.json`
  inside the dataset `lordhansolo/taaf-kaggle-source`.
- dfranzen adds an FR-Spec 64k hot-token map to its draft; lordhansolo uses a 32k draft vocab.
- The top score belongs to the build with the fewest server slots and the longest per-slot
  context, not the one with the most sequences or the largest raw context length.

## 1. Token capacity

| Setting | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Server context length | 139,264 | 147,072 | 69,632 |
| Harness context window | 131,072 | 127,488 | 69,632 |
| Reply reservation | 12,288 + 512 | 512 + 512 | 512 + 512 + 4,096 |
| Prompt budget before trim | ~118k | ~126k | 65,536 |
| Trim rule | drop down to ~59k (58 Ki drain) | drop down to 81,536 | watermarks 57k → 45k |
| Assistant-turn cap | 150, drain 30 | none (off when a target is set) | lifted |
| Headroom, window → server | 8,192 | 19,584 | 0 |
| Prefill chunk | 8,192 | 2,048 | 4,096 |
| Max prefill tokens | 16,384 | n/a | n/a |
| Board image size | 640 px | 256 px | 256 px |
| Tokens charged per board | ~402 | ~66 | 66 |
| Tool output cap | 3,072 | 1,024 | 1,024 |
| Worst case tokens, all slots full | 1.39 M | 2.06 M | 1.11 M |
| Steady-state prompt | ~60k → ~118k | ≤ ~81k | 45k → 57k |

Notes

- dfranzen: window is (116+12)·1024 and server context is (116+12+8)·1024. After hitting
  the budget the harness drains a further 58k of old history so the retained prefix stays
  cache-hot for many requests. Diff, game-over and animation images are also attached.
- lordhansolo: the 81,536 target comes from `LOCAL_ANALYZER_TARGET_CONTEXT`. Once the local
  estimate passes ~126k, the harness strips old saved-module catalogs and drops the oldest turns
  until the estimate is at or below 81,536. Setting a target also disables the 30-turn cap.
- sirikilohit: M79 caps prompts at 65,536 using exact server token counts; M84 trims from
  57,344 down to 45,056; M74 charges each board image 66 tokens.
- Worst case is server context × server slot cap.

## 2. Maximum number of sequences

| Setting | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Server slot cap | 10 | 14 | 16 |
| Boot guard on slots | assert only | none | raise if < 15 |
| Games admitted (competition) | 110 | 14 | 28 |
| Games streaming at once | 10 | 14 | 16 |
| Admission control | priority gate | none | UCB slices, 2,400 s |
| Per-game runtime | 31,920 s shared | 3,918 s | 7,920 s |
| Notebook budget | 32,400 s | 2,100 s (see caveats) | 32,400 s |
| CUDA graph batch sizes | 1–10, max 10 | up to 56 | default |
| Schedule policy | lpm | async | default |
| Yield rule per turn | 2,048 generated tokens | 150 s | none |
| Tool steps per turn | unlimited | unlimited | unlimited |

Notes

- dfranzen admits all 110 games to the harness but a priority gate keeps 10 streams active,
  ranked by progress, actions and tokens spent, with tail fade in the last 20% of time.
- lordhansolo's 56 is ceil(14 × (1 + 3 MTP) / 8) × 8.
- sirikilohit's cell 3 still carries a stale vLLM profile (8 sequences, 1 MTP token, 10 GiB
  KV) and cell 16 prints `kv=10GiB mtp=1`. SGLang ignores those variables; the effective values
  are the launch arguments in cell 11.

## 3. KV cache settings

| Setting | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| KV dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 |
| Draft KV dtype | fp8_e4m3 | n/a | default |
| Indexer KV dtype | n/a | fp8 | n/a |
| SSM state dtype | bf16 | bf16 | bf16 |
| Mamba cache size | 60 | n/a | 80 |
| Pool sizing | static 0.96 | util 0.98, profiled | static 0.97 |
| Pinned KV bytes | none | none (auto) | none |
| Page / block size | 64 | 128 match unit | 64 |
| Prefix caching | radix | on, align mode | radix |
| Mamba radix strategy | extra_buffer | align via patches | extra_buffer |
| Host cache tier | none | none | 48 GB |
| MTP draft cache mode | none | recover SSM | none + patch |
| Offloaded weights | PLE | PLE + embed_tokens | PLE |
| KV pool asserted at boot | no | no | ≥ 330k tokens |

Notes

- lordhansolo validates six patched vLLM files by hash before enabling fine-grained prefix
  caching in Mamba align mode.
- sirikilohit patches the SGLang KV configurator to stop reserving draft SSM scratch, and its
  cell 8 quotes a 727,168-token pool for the earlier boot recipe.
- dfranzen uses longest-prefix-match scheduling so requests sharing a prefix run together.

## 4. Other differences that affect throughput

| | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Linear-attention backend | flashinfer | flashinfer prefill, triton decode | flashinfer |
| MoE backend | auto (Marlin for W4A16) | auto (flashinfer_cutlass for NVFP4) | auto (Marlin for W4A16), draft on flashinfer_cutlass |
| Reasoning kept in history | reasoning_content | preserve_thinking, xhigh | reasoning_content |
| Temperature | 0.7 | 0.6 | 0.6 |
| Server watchdog | 1,800 s SGLang | restart script | 3-strike exit |
| Startup deadline | 12 min, then start anyway | 3,600 s | 1,500 s |
| Weight prefetch | page-cache precache | NFS shard prefetcher | 16-thread prefetch |
| Save & Run games | 10 | 25 | 25 |

Notes

- dfranzen warms up with one RESET per game while the server is still loading, and the
  harness retries HTTP for 900 s, so Kaggle never sees an idle notebook.
- sirikilohit also patches parallel PLE row copies and restores compiled-kernel caches.

## 5. Patches

| | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Harness delivery | git patch applied at setup | forked source in bundle | runtime monkey-patches |
| Harness patch size | 16 files, +8,405 / −197 | 40 files, +5,246 / −2,337 | 15 patch cells |
| Largest harness change | tool_agent.py +5,339 | tool_agent.py rewrite, 1,745 lines vs 2,063 stock | M90 harness port |
| New harness modules | 6 | 11 | 0 (inline) |
| Server patches | 5 | 32 overlay files | 4 |
| Server patch delivery | baked into wheelhouse | hash-pinned overlay tar | applied to site-packages at boot |
| Upstream fixes ported | 0 | 9 vLLM PRs | 0 |
| Custom server fixes | 5 | 16 | 4 |

Notes

- dfranzen (harness): cell 2 holds a 499 KB patch applied with `git apply` onto an unmodified
  copy of Tufa's bundle. New modules are animation, frame_diff, retained_functions,
  noop_repeat_guard, priority_scheduler and a pace-reference JSON. Per the write-up the shipped
  features are calibrated token estimates, image-size-aware accounting, blockwise drain trimming,
  bounded admission with priority scheduling and tail fade, animation and diff access from
  Python, larger board images, an UNDO action, retained functions and imports, and stale-state
  and no-op guards. Structured world-model sections are switched off.
- dfranzen (server): five patches in `serving/build_bundle_pennyroyal.sh`: low-M BF16 GEMM
  (from Olympie), speculative-state budget correction (from Ratsimbazafy), a Marlin scale dtype
  fix, prefix-cache retention (sparse Mamba prefill checkpoints, LRU refresh on final unlock),
  and bounded checkpoint prefetch with one-shard lookahead. The checkpoint was also resharded.
- dfranzen (did not help): summarisation, resume-prompt variants, composite animation images,
  extra verification hints and guards, and more elaborate priority rules.
- lordhansolo (harness): a fork of ARC3-Inference at commit `ca1bd02` on branch
  `solution-improve-qwen38-flash-next` ("Let the agent act without full certainty"), shipped as
  source rather than a patch. Against Tufa's bundle it rewrites tool_agent, solver and the
  sandbox, adds a 1,475-line Kaggle launcher, and adds modules for per-game checked Python
  modules, token estimation, model-response parsing, animation, grid rendering, transcript and
  metrics logs, a vLLM Prometheus scraper and cache diagnostics. The line counts include
  uv.lock and README changes.
- lordhansolo (server): the dataset ships a 32-file overlay over the vLLM nightly image, pinned
  by SHA256 and installed by an applier that refuses modified targets. It carries nine ports of
  vLLM main PRs and sixteen custom fixes, among them Mamba align-state retention and prompt-tail
  state for multi-turn prefix reuse, an MTP checkpoint weight prefilter (4 of 97 shards), a
  pruned draft vocabulary, pinned-host input embeddings, SM120 small BF16 GEMMs for decode rows
  40 to 64, exact CUDA graphs at 44 and 52 tokens for 14 sequences with MTP3, and GDN RecoverSSM
  for MTP verification. The validation notes say rc2 was tested on an RTX PRO 4000 laptop and
  never benchmarked on the RTX PRO 6000 before upload.
- sirikilohit (server): four source patches applied to the unpacked SGLang site-packages at
  boot, each anchored on an exact code snippet and compiled before use: drop the draft SSM
  scratch reservation from the KV budget, prefetch checkpoints once, copy PLE rows in parallel,
  and create GPTQ-Marlin MoE scales in the model dtype. Boot also merges Intel W4A16 experts
  with FP8 PLE tables, builds a draft-only MTP folder (M108), and restores compiled-kernel caches.
- sirikilohit (harness): patch cells wrap ToolAgent and session methods at import time, each
  with a self-test gate that raises if the wrap is inert. Installed: ACTION7 fix, reasoning
  ledger (M24), two-tier scheduler (M43, superseded by M72), multi-request turns and sticky
  sandbox (M44), world-model capture and code memory (M59), image token cost (M74), intermediate
  animation frames (M76), exact token-count guard (M79), context watermarks (M84), context
  repair, incremental state file (M86), the Wang/Ludvig harness port (M90), solved-level memory
  (M85), and the one-pool UCB scheduler (M72). Their write-up credits the last jump (14.49 to
  22.53) to FP8 KV plus longer history, and lists instruction blocks, fresh-context restarts,
  temperature changes, expert pruning, higher-resolution images and fine-tuning as failures.

## 6. Measured throughput

From each notebook's Save & Run server log on 2026-09-30. These are demo runs, not the
scored competition reruns.

| Metric | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Run shape | 10 games, 25 min each | 25 games, 14 at once | 25 games, 20 min |
| Peak decode tok/s | 946 | 1,135 | 1,159 |
| p90 decode tok/s | 833 | 1,056 | 1,039 |
| Median decode tok/s | 722 | 976 | 899 |
| Samples | 991 | 90 | 540 |
| Running requests at peak | 10 | 14 | 16 |
| Median accepted draft length | 2.7 | 2.9 | 2.6 |
| KV pool (tokens) | 1,011,264 | 1,417,100 | 1,004,288 |
| KV usage at peak | not logged | 91% | not logged |
| Prefix cache hit rate | not logged | median 91% | not logged |
| Job-wide generated tok/s | 588 | 933 | 688 |
| Notebook start to ready | 531 s | 544 s | 615 s |
| Server launch to ready | 478 s | 430 s | 536 s |
| Weight load within that | 269 s | 240 s + 13 s draft | 215 s + 47 s draft |
| Demo mean score | 36.56 | 5.28 | 6.89 |

Notes

- Time to first request is the kernel-log timestamp of the first successful health or
  model-list response, counted from notebook start. Before the server launch, dfranzen
  installs its wheelhouse (53 s), lordhansolo unpacks and overlays the vLLM image (114 s),
  and sirikilohit unpacks SGLang plus a CUDA toolkit (79 s).
- SGLang logs one line per decode batch, so the dfranzen and sirikilohit peaks are single-batch
  readings. vLLM logs a 10 s average, so lordhansolo's peak is already smoothed.
- Job-wide tok/s is total generated tokens over the whole notebook wallclock, including server
  boot and idle time, so it understates steady-state throughput for every run.
- Peak throughput tracks the number of concurrent requests more than the server: 10, 14 and 16
  slots give 946, 1,135 and 1,159 tok/s. Per-sequence decode speed is therefore highest for
  dfranzen at about 95 tok/s per stream, versus about 81 and 72.
- lordhansolo's vLLM reports 19.69 GiB available for KV after profiling, giving 1,417,100 tokens
  and 9.64 full-context requests. Its draft acceptance rate was 63 to 71 percent.
- sirikilohit's boot log warns that the FP8 KV cache has no scaling factors
  (`fp8_unscaled_warning=True`).
- Demo mean scores are not comparable: dfranzen played 10 of the 25 public games, the other two
  played all 25 under different time limits.

## Caveats

- lordhansolo's values come from `lordhansolo/taaf-kaggle-source` version 305, uploaded
  2026-09-30 13:11:22 UTC, the same minute the linked notebook version started its Save & Run,
  so it is the bundle attached to that version, which the scored rerun reuses. Its pickled
  target still names an older kernel slug and a 2,100 s budget.
- dfranzen's budget formula (window − reply − 512) and image-token estimate are read from
  this repo's `ARC3-Inference`, which matches the notebook's bundled patch.
- Scores are the public leaderboard values shown on each Kaggle page on 2026-09-30.
- No notebook ships saved outputs, so measured KV pool size, cache hit rate and achieved
  concurrency are not recorded.

## Sources

- Save & Run logs via `kaggle kernels output`: dfranzen `serve.log`, lordhansolo
  `vllm-openai-server.log` and `metrics/vllm_server.jsonl`, sirikilohit `sglang-main.log`
- lordhansolo patch docs: `PATCH_README.md`, `VALIDATION.md`, `PATCH_IDENTITY.json` in `lordhansolo/vllm-main-e975732-arc3`
- sirikilohit write-up: `LohitSiriki/arc-agi-3-milestone2-solution/WRITEUP.md`
- `kaggle/dfranzen/arc-agi-3-milestone-2-solution.ipynb` cells 0, 4, 12, 16, 20
- `kaggle/lordhansolo/arc-agi-3-milestone-2.ipynb` plus `setup_commands.json`, `preamble.txt`, `benchmark_initial.pkl`, `deploy_target.pkl` from `lordhansolo/taaf-kaggle-source`
- `kaggle/sirikilohit/arc-agi-3-duck-18-1gc-submit.ipynb` cells 3, 9, 11, 16, 33, 35, 44, 45
