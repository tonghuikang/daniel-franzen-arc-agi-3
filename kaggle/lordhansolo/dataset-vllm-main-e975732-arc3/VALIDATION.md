# e975732 ARC3 bundle validation

## rc2 — four more vLLM main fixes, 2026-09-30

Overlay SHA256 `e3a6fe0d9f010bc1fb66e53e7dc6fde43272824d5200da5a49b5c4456065e0cb`,
32 files. Eleven differ from the GDN bundle: the rc1 changes plus vllm#58114
(three files), #58434, #58784 and #58720. Five are new members over stock
image files.

- The validator passed, including installation, idempotency and rejection of
  a modified target.
- On the e975732 runtime with this overlay, `tests/models/qwen4_exp/test_ple.py`,
  `tests/v1/worker/test_mamba_hybrid_model_state.py`,
  `tests/v1/spec_decode/test_rejection_sampler_utils.py` and
  `tests/kernels/moe/test_moe_weight_loading_padded.py` from e975732 plus the
  upstream test changes passed 205 of 205;
  `tests/v1/attention/test_gdn_metadata_builder.py` and
  `test_mamba_update_block_table.py` passed 22 of 22 with and without rc2.
- With the rc1 overlay, 11 of the new upstream tests fail and the rejection
  sampler test does not import. Among the failures, the mixed spec/prefill
  PLE batch on CUDA raises `called a synchronizing CUDA operation`.

## rc1 — ported vLLM main fixes, 2026-09-30

Overlay SHA256 `702a3b9171e48f9d6fe97edc127572169ccdd0c2d56fbfdaffb16c7e7575d3f0`,
27 files. Five differ from the GDN bundle: the two C14 files below and the
three ports (vllm#58489, #58961, #58430). `qsa_cache.py` and `gpu_worker.py`
are new members over stock image files, read from image layer 27.

- The validator passed, including installation, idempotency and rejection of
  a modified target.
- The shipped applier installed the overlay onto a local e975732 runtime
  carrying an older overlay, and a second run changed nothing.
- On that runtime `tests/models/qwen4_exp/test_ple.py`,
  `test_qsa_reference.py` and `tests/v1/worker/test_gpu_worker.py` from
  e975732 plus the three upstream test changes passed 143 of 143. Without the
  ports, 11 of the 12 new upstream tests fail.
- The uploaded Kaggle version has the same 13 file names and sizes as the
  local build, and its manifest carries this overlay.

The KV pool change and whether production takes the PIECEWISE path of
vllm#58489 remain to be read from the first Kaggle run
(`Available KV cache memory`, 19.6 GiB before).

## Exact C14 CUDA graphs, overlay ff4686ad — 2026-09-30

The local build and validator passed. The overlay SHA256 is
`ff4686ad9b616293bc153821c73a3c980b3c559356d8d0239d1f2d9545ce103c`.
Of its 25 files, 23 have the same bytes as the pinned GDN bundle; only
`vllm/config/vllm.py` and `vllm/models/qwen4_exp/nvidia/arc3_sm120_gemm.py`
differ.

The local validator checks all six image layer sizes and SHA256 digests,
overlay membership and per-file digests, the exact two-file source change,
installation onto the GDN bundle, idempotent installation, and rejection of a
modified target before any file is written. The build script also compiles the
changed vLLM modules. Installation onto the stock files extracted from the
image layers wrote all 25 files, a second run wrote none, and every installed
file matched the overlay.

The two new sizes are present only when Qwen4Exp runs TP1, MTP3, C14 with a
capture ceiling of at least 56. With a 56-token ceiling, the capture list is
`[1, 2, 4, 8, 16, 24, 32, 40, 44, 48, 52, 56]`.

## GEMM at M52

Local GPU: RTX PRO 4000 Blackwell Laptop, SM120, with the pinned e975732
runtime carrying this overlay. The rc3 suite `test_sm120_gemm.py` passed all
97 tests in 51 s. It covers eager and CUDA graph replay outputs against FP32
for the seven plans at M=40/44/48/52/56/64, custom GEMM selection at those
rows, the stock fallbacks and the installation scope.

Over five seeds per plan, the worst relative RMS against FP32 at M52 is
0.00166–0.00169, the same as at M48 and M56 and never above torch BF16 at the
same row count. Synthetic weights; not a model quality test.

No RTX PRO 6000 server benchmark or competition wave has been run for this
bundle. The M52 GEMM time, the speed gain and any change in available KV
blocks remain to be measured on the target machine. The local checks do not
establish bitwise equality of a full model run.
