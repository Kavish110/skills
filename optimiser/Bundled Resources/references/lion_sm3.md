# LION and SM3 Optimizers for Large Models

- **Papers:**
  - "Symbolic Discovery of Optimization Algorithms" — Bello et al., 2021 (LION family inspiration)
  - "Scaling Memory-Efficient Optimizers for Transformers" — Gupta et al., 2022 (SM3)
- **Links:**
  - https://arxiv.org/abs/2110.03603
  - https://arxiv.org/abs/2201.00599

## Summary

LION is a modern optimizer that combines sign-based updates with momentum to reduce sensitivity to step size and weight decay. It has shown strong results on large transformer training and generative model fine-tuning.

SM3 is a memory-efficient adaptive optimizer that stores only factorized second-moment estimates. It is designed for extremely large embedding tables and transformer dense layers where full moment accumulators are prohibitively expensive.

## What worked

- LION often matches or outperforms AdamW on language and vision tasks while using simpler momentum behavior.
- LION can be more stable for large-batch transformer pretraining and can reduce the need for aggressive LR warmup.
- SM3 is effective for models with massive sparse embeddings and large vocabularies, enabling very large-scale training on limited-memory hardware.
- SM3 is a strong option when parameter counts exceed GPU memory budgets for Adam/AdamW.

## What did not work

- LION can still require task-specific tuning of weight decay and learning rate schedules.
- SM3 may train slower per step due to increased computation and can underperform AdamW on small dense models.
- Both optimizers are less widely supported in off-the-shelf libraries than AdamW, so implementation and mixed-precision support must be validated.
