# ARC-AGI-3 Milestone 2: comparing the top three notebooks

This is a fork of [Daniel Franzen's ARC-AGI-3 Milestone 2 solution](https://github.com/da-fr/arc-agi-3-solution) for the ARC Prize 2026 ARC-AGI-3 competition. It adds a side-by-side look at how the top three notebooks (dfranzen, lordhansolo, sirikilohit) serve their model.

All three run Tufa Labs' Duck harness on the Tufa ARC-AGI Framework (TAAF), serve Qwen3.8-Flash-Next, and fit on a single 96 GB RTX PRO 6000. They differ in how they split that GPU memory between model weights, KV and Mamba cache, and CUDA graphs. They also differ in quantization, speculative decoding, inference server, and serving patches.

[kaggle/COMPARISON.md](kaggle/COMPARISON.md) has the full breakdown, with links to the exact lines in each notebook and source bundle. Copies of all three notebooks and their source datasets are in [kaggle/](kaggle/).

The comparison is also posted as a [discussion on the Kaggle competition forum](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792).

The original solver code, serving runtime, and write-up from Daniel Franzen are still in this repo ([WRITEUP.md](WRITEUP.md), [ARC3-Inference/](ARC3-Inference/), [serving/](serving/)).
