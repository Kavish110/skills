QLoRA / 4-bit fine-tuning (Dettmers et al., 2023)

- ArXiv: https://arxiv.org/abs/2306.08907
- Summary: Combines 4-bit model quantization with low-rank adapters to make fine-tuning of large models feasible on smaller GPUs.
- What worked: Provides large memory savings and preserves fine-tuned accuracy in many settings.
- What didn't: Requires careful choice of quantization method and may need dequantization or special inference support.
