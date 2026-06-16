---
name: optimiser
description: |
  Authoritative skill for designing, implementing, and tuning optimizers for generative AI and ML models.
  Use this skill whenever the user asks about optimizer choice, training stability, learning rate schedules, adaptive optimizers, or optimization best practices for LLMs, diffusion models, flow matching, RL, and other generative modeling settings.
compatibility:
  - Python 3.8+
  - PyTorch
  - JAX/Flax
  - TensorFlow
version: 0.1.0
author: "Kavish Patel"
tags: [optimizers, training, research, generative-ai, reinforcement-learning, diffusion, flow-matching]
bundled_resources:
  - Bundled Resources/scripts
  - Bundled Resources/references
  - Bundled Resources/assets
---

<!-- TOC -->

## Overview

`optimiser` helps engineers and researchers design reliable optimizers and schedules for generative AI and machine learning.
It balances implementation guidance, practical training recipes, and research-backed summaries for:
- LLM pretraining and fine-tuning
- diffusion and score-based generative models
- flow matching and continuous generative flows
- reinforcement learning policy and value optimization
- memory-efficient large-model training

## When to use (Trigger conditions)

- Use when the user asks about optimizer selection, learning rate schedules, weight decay, warmup, momentum, or adaptive methods.
- Use for requests about training stability, mixed precision, large-batch optimization, gradient clipping, or optimizer performance in generative models.
- Use when the user needs code, hyperparameter defaults, or research references for optimizers in LLMs, diffusion, flow matching, RL, or generative pipelines.

## Quick start

- Review `Bundled Resources/references` for research papers and practical takeaways.
- If the user asks for runnable examples, point to `Bundled Resources/scripts` and `Bundled Resources/assets`.
- Example prompt to trigger this skill:
  "Design an optimizer and schedule for training a 3B transformer with AdamW, warmup, and a cosine decay for RLHF fine-tuning."
- Encourage users to specify model type, dataset characteristics, and stability goals when asking for hyperparameters.

## How to ask

To get the best recommendation, users should include:
- model type and size (e.g. 1.5B transformer, diffusion U-Net, flow matching network, PPO actor/critic)
- dataset description (e.g. instruction-following text, 256x256 images, high-resolution video frames, sparse RL rewards)
- training goal (e.g. stability, speed, memory efficiency, final quality)
- desired optimizer style (e.g. adaptive, memory-efficient, low-noise, large-batch friendly)
- hardware constraints if relevant (e.g. mixed precision, multi-GPU, limited GPU memory)

Example user request:
- "Recommend Muon hyperparameters for fine-tuning a 1.5B instruction-following LLM on mixed-precision hardware."
- "Suggest a diffusion optimizer, scheduler, and gradient clipping strategy for 256x256 image generation."

## What this skill provides

- Research-backed optimizer guidance for AdamW, LION, SM3, AdaFactor, Muon, and RL-specific variants.
- Practical training advice for diffusion and flow matching models.
- Model-aware hyperparameter recommendations based on optimizer choice, model class, and dataset type.
- Suggestions for scheduler type, warmup strategy, weight decay, gradient norm clipping, and normalizer selection.
- Guidelines on mixed-precision training, gradient clipping, and memory-efficient optimizer choices.
- Reference summaries for key papers, model-specific optimization methods, and the latest practices used by researchers and engineering teams at major AI labs.

## Hyperparameter guidance

- Recommend scheduler types such as cosine decay, linear decay, polynomial decay, or constant schedule based on the model and dataset.
- Suggest norm strategies such as global gradient clipping, per-parameter norm clipping, and layer normalization options for stability.
- Choose learning rate ranges, weight decay values, and warmup step counts appropriate for small-scale fine-tuning, large-scale pretraining, diffusion training, or RL policy optimization.
- Advise when to use adaptive optimizers, memory-efficient optimizers, or simple SGD-like updates depending on model size and hardware.

## Bundled Resources

- `Bundled Resources/scripts` — placeholder for runnable examples and optimizer utilities.
- `Bundled Resources/references` — curated paper summaries for optimizer methods and schedules.
- `Bundled Resources/assets` — optional templates for configs, schedules, and optimizer settings.

## Examples

- "Recommend a stable optimizer and LR schedule for diffusion model training on 256x256 images, including the best scheduler and clipping strategy."
- "What optimizer should I use for mixed-precision flow matching training on high-resolution data, and how should I set weight decay and norm clipping?"
- "Give me a reinforcement learning optimizer recipe for PPO actor-critic training with AdamW, including actor/critic learning rates and schedule type."
- "Suggest hyperparameters for Muon when fine-tuning a 1.5B LLM on instruction-following data, including LR, warmup, and decay schedule."

## Inputs & Outputs

- Input: natural-language prompts about optimizer design, training recipes, stability fixes, or research citations.
- Output: actionable optimizer recommendations, code snippets, paper references, and hyperparameter patterns.

## Limitations

- The skill provides guidance and minimal examples; production training requires distributed infrastructure and dataset engineering.
- Specific tasks may require further tuning beyond the default optimizer recommendations.

## Contributing

- Add new papers to `Bundled Resources/references` with concise summaries.
- Add runnable examples to `Bundled Resources/scripts` with clear README entries.

## Changelog

- 0.1.0 — Initial optimiser skill with references for AdamW/AdaFactor, LION/SM3, diffusion/flow matching, and RL optimization.
