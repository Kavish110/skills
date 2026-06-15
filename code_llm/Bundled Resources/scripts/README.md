This folder contains minimal example scripts used by the `code_llm` skill.

Files:
- `train_example.py` — minimal PyTorch training loop for a tiny decoder-only model.
- `eval_example.py` — simple evaluation harness computing perplexity and token-level accuracy on a dataset.
- `quantize_example.py` — notes and a small wrapper showing how to apply post-training quantization (conceptual; requires bitsandbytes/GPTQ).
- `inference_bench.py` — latency and throughput micro-benchmark for a loaded model.

Usage: run each script with `python <script>` after installing dependencies from `Bundled Resources/assets/requirements.txt`.
