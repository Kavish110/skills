# Diffusion and Flow-Matching Optimizers

- **Papers:**
  - "Denoising Diffusion Probabilistic Models" — Ho et al., 2020
  - "Score-Based Generative Modeling through Stochastic Differential Equations" — Song et al., 2021
  - "Flow Matching for Generative Modeling" — Lipman et al., 2022
- **Links:**
  - https://arxiv.org/abs/2006.11239
  - https://arxiv.org/abs/2011.13456
  - https://arxiv.org/abs/2206.00364

## Summary

Diffusion model training has become heavily dependent on adaptive optimizers such as Adam and AdamW, combined with precise noise schedules, gradient clipping, and reliable LR decay. Flow matching introduces a different generative training objective based on matching continuous vector fields, which also benefits from adaptive optimizers and careful normalization.

## What worked

- AdamW with cosine or linear decay and warmup is a strong default for diffusion and score-based models.
- Stability is improved by gradient norm clipping, loss scaling, and constant-parameter weight decay.
- Flow matching models can use the same optimizer families as diffusion models, though they often require smaller learning rates and tighter conditioning on time embeddings.
- Using a separate optimizer for denoiser and guidance networks can improve sample quality in advanced pipelines.

## What did not work

- Large diffusion models are sensitive to optimizer state corruption in fp16/bfloat16; mixed-precision training must be validated carefully.
- Using a generic LR schedule without warmup can destabilize early training.
- Flow matching objectives may diverge if the optimizer interacts poorly with the vector field parametrization, especially for higher-resolution data.
- Excessive adaptive momentum without proper decay can lead to oversmoothing and slow sample diversity.
