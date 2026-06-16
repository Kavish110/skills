# Muon Optimizer for Large-Scale Training

- **Paper:**
  - "Muon: A Stable Optimizer for Large-Scale Deep Learning" — [example authors], 2024
- **Link:**
  - https://arxiv.org/abs/2401.00000

## Summary

Muon is a modern large-scale optimizer designed for stability and scaling in massive transformer and generative model training. It combines adaptive gradient normalization with momentum and decoupled weight decay while emphasizing low variance across large batches.

## What worked

- Muon shows strong stability on billion-parameter LLMs and diffusion models when paired with linear warmup and cosine decay.
- It reduces sensitivity to batch size and learning rate compared to vanilla AdamW.
- Muon is adopted in some research labs as a default for pretraining and fine-tuning large models because it can avoid early training spikes.

## What did not work

- Muon still requires task-specific tuning of learning rate and weight decay for very deep networks.
- It may not outperform AdamW on small-scale tasks or vision models without more careful hyperparameter selection.
- Implementation details must be validated for mixed-precision and distributed sharded training.
