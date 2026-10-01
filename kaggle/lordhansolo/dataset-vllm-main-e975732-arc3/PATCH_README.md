# ARC3 e975732 — exact CUDA graphs for C14/MTP3

This local runtime bundle builds on `vllm-main-e975732-arc3-gdn-recoverssm`.
The six Docker image layers and all existing GDN RecoverSSM code are identical
to that pinned bundle. The changed vLLM files are `vllm/config/vllm.py`,
`vllm/models/qwen4_exp/nvidia/arc3_sm120_gemm.py` and the ported vLLM main
fixes listed under "Ported vLLM main fixes".

For Qwen4Exp with tensor parallel size 1, `max_num_seqs=14`, a uniform decode
query length of 4 (MTP3), and a CUDA graph capture ceiling of at least 56,
the configuration adds exact capture sizes 44 and 52. The existing 48 and 56
captures remain. Other configurations keep the GDN bundle's capture sizes.
The Kaggle serving configuration and maximum context length are unchanged.

These two sizes can avoid padding 11- and 13-request decode batches to 48 and
56 tokens. The actual gain depends on the distribution of active batch sizes.
They use the existing CUDA graph memory pool, but may increase its high-water
mark and capture time. Neither the KV pool delta nor the throughput gain has
been measured on the RTX PRO 6000.

The rc3 SM120 GEMM serves only the rows in `GEMM_ROWS`, which had no 52. An
exact M52 graph would have sent 13-request steps from the custom GEMM at M56
back to the original linear method, so M52 joins `GEMM_ROWS`. The rc3 plans
are per projection, not per row count, and M52 runs the same kernels as M48
and M56. The partial GEMM has the same launch grid at all three; the split-K
reduction grid grows with the row count. Model weights, KV format and MTP are
unchanged. The 11- and 13-request batches run different row counts, so their
outputs are not bitwise identical to the padded graphs.

## Build and validation

From `ARC3-Inference/`:

```bash
.venv/bin/python patches/vllm-main-e975732-arc3/build_runtime_patch_bundle.py
.venv/bin/python patches/vllm-main-e975732-arc3/validate_runtime_dataset.py
```

The flat dataset is written to `builds/vllm-main-e975732-arc3/dataset`, with
Kaggle dataset id `lordhansolo/vllm-main-e975732-arc3`, which the Kaggle
launcher mounts and pins by overlay SHA256. The installer in the dataset
accepts the pinned stock, rc1, rc2, rc3 and GDN source hashes, verifies all
overlay members before writing, and is safe to run twice.

## Ported vLLM main fixes

Pinned since rc1 (overlay `702a3b91`):

- `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`: PLE prefetch ids stay out
  of the CUDA graph pool (vllm#58489, `48d8880d0`).
- `vllm/models/qwen4_exp/common/qsa_cache.py`: QSA key views no longer hold the
  profiling KV cache (vllm#58961, `a2be4d3cc`).
- `vllm/v1/worker/gpu_worker.py`: `max_split_size_mb` 20 during `profile_run`,
  with the allocator settings restored afterwards (vllm#58430, `6491f481a`).

Added in rc2 (overlay `e3a6fe0d`), not pinned yet:

- `vllm/v1/attention/backends/mamba_attn.py`,
  `vllm/v1/attention/backends/short_conv_attn.py` and
  `vllm/models/qwen4_exp/nvidia/ple_layer.py`: PLE short-conv metadata is built
  without the causal-conv1d metadata and without the GPU sync of mixed batches
  (vllm#58114, `20b52e9f5`).
- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`: a one-token prompt tail
  padded with placeholder drafts builds as a spec-decode row, so GDN does not
  fold the placeholders into its state (vllm#58434, `7dbd0a8d2`).
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`: draft slots padded
  onto a step that starts in the prefill are rejected (vllm#58784,
  `fedbc3b56`).
- `vllm/model_executor/layers/fused_moe/routed_experts.py`: `load_weights`
  indexes the expert mapping by expert id (vllm#58720, `ad6817b68`).

All files except `ngram_embedding.py` and `mamba_hybrid.py` are new overlay
members over stock image files. The builder reads their stock bytes from the
image layers and checks every port base against its pinned digest. The AMD
`ple_layer.py` change of vllm#58114 is not ported.

## Release candidates

This is the main patch package; release candidates are builds of it, not
separate packages or datasets. Change `source/` and the builder here, then:

```bash
.venv/bin/python patches/vllm-main-e975732-arc3/build_runtime_patch_bundle.py --rc N
.venv/bin/python patches/vllm-main-e975732-arc3/validate_runtime_dataset.py --rc N
```

An rc is written to `builds/vllm-main-e975732-arc3-rcN`, with its dataset in
`dataset/` and its identity and review copies in `review/`. It is uploaded as
a new version of `lordhansolo/vllm-main-e975732-arc3`, with the same artifact
and applier names. `PATCH_IDENTITY.json`, `patched/` and `diffs/` in this
folder are the pinned build, which the Kaggle launcher and its tests check.
Once an rc is uploaded, repin `overlay_sha256` in
`inference/framework/kaggle.py` and rerun the builder without `--rc` to
refresh them.

`source/` holds the changed vLLM sources. The builder regenerates `patched/`,
`diffs/`, `PATCH_IDENTITY.json` and `APPLIED_FILES.json` for review. The diffs
are relative to the GDN bundle. See `VALIDATION.md` for the local checks.

The GPU tests of the GEMM module (moved from rc3) run with the extracted pinned
runtime, carrying this overlay, on `PYTHONPATH`:

```bash
.venv/bin/python -m pytest -q --noconftest patches/vllm-main-e975732-arc3/tests/test_sm120_gemm.py
```
