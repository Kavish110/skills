GPTQ / AWQ (post-training quantization methods)

- ArXiv: https://arxiv.org/abs/2310.03060
- Summary: Specialized post-training quantization methods for transformer weights that preserve accuracy when reducing precision aggressively.
- What worked: Achieves very low-bit inference with small accuracy loss for many large language models.
- What didn't: Sensitive to implementation details and model weight distributions; not all models transfer smoothly.
