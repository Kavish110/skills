# Decoupled Weight Decay and Adaptive Training

- **Papers:**
  - "Decoupled Weight Decay Regularization" — Loshchilov & Hutter, 2017
  - "Adaptive Learning Rates with Sublinear Memory Cost" — Shazeer & Stern, 2018
- **Links:**
  - https://arxiv.org/abs/1711.05101
  - https://arxiv.org/abs/1804.04235

## Summary

AdamW showed that classic Adam interacts poorly with L2 weight decay because the decay term is applied inside the adaptive moment update. Decoupling weight decay from the gradient step makes regularization behave like SGD and improves generalization on transformers and large-scale pretraining.

AdaFactor introduced a memory-efficient adaptive optimizer that tracks second-moment statistics using factorized approximation. It is especially useful when training billion-parameter models on constrained hardware.

## What worked

- AdamW is now the default optimizer for transformer-based LLMs and diffusion models.
- Decoupling weight decay reduces overfitting and improves validation quality at scale.
- AdaFactor enables training large models with much lower GPU memory usage than Adam/AdamW.
- Relative step size and warmup schedules in AdaFactor produce stable convergence for language models.

## What did not work

- AdamW still requires careful learning rate schedules and warmup; naive settings can diverge in large-batch training.
- AdamW alone does not solve instability for very deep diffusion or flow-matching networks; gradient clipping and loss scaling are still needed.
- AdaFactor can underperform AdamW on dense vision and small-scale tasks when not tuned.
- Aggressive factorization in AdaFactor may reduce optimizer fidelity for unusual parameter shapes or mixed-precision updates.
