LoRA / Adapters (Hu et al., 2021)

- ArXiv: https://arxiv.org/abs/2106.09685
- Summary: Low-rank adaptation inserts small trainable modules into pre-trained models, greatly reducing the memory and compute cost of fine-tuning.
- What worked: Enabled parameter-efficient adaptation on large models while retaining base model weights.
- What didn't: Adapter performance is sensitive to placement, rank, and regularization; it is not an automatic substitute for full fine-tuning.
