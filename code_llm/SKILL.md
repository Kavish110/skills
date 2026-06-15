---
name: code_llm
description: |
  Authoritative skill for building, training, evaluating, and deploying large language models (LLMs). Use this skill whenever the user asks about designing LLM architectures, training recipes, evaluation metrics, inference optimizations, or quantization — even if they only mention one of those topics. Be pushy: prefer this skill for any LLM-related engineering or research request, including benchmarking, ablation ideas, or production deployment tips.
compatibility:
  - Python 3.8+
  - PyTorch (preferred examples)
  - JAX/Flax (notes included)
  - TensorFlow (notes included)
  - Optional: bitsandbytes, GPTQ, Hugging Face Transformers, accelerate
version: 0.1.0
author: "Kavish Patel"
tags: [llm, training, inference, quantization, evaluation, research, engineering]
bundled_resources:
  - Bundled Resources/scripts
  - Bundled Resources/references
  - Bundled Resources/assets
---

<!-- TOC -->

## Overview

`code_llm` helps engineers and researchers design, implement, evaluate, and deploy Large Language Models(LLMs). It balances practical runnable examples (training loops, evaluation harnesses, quantization demos, inference benchmarking) with concise research summaries that state "what worked" and "what did not" for important approaches.

## When to use (Trigger conditions)

- Use whenever the user asks about: LLM model architecture, scaling, data curation, training recipes, instruction tuning, RLHF, evaluation metrics, inference latency/throughput, model quantization, or deployment.  
- Use when the user proposes experiments, asks why results diverged from papers, or requests code for training/evaluation/quantization a Large Language Model.

## Quick start

- Read the Bundled Resources/README files for scripts and references.
- Example prompt to trigger this skill:

  "I want to train a 7B LLM for instruction following — give me a reproducible training recipe, evaluation plan, and quantization strategy."  

## What this skill provides

- Research summaries and short takeaways (scaling laws, Chinchilla, Deeper/Longer models, instruction tuning, RLHF, distillation, LoRA/Adapters, MoE).  
- Minimal runnable example scripts for: training, evaluation, quantization, and inference benchmarking.  
- Prompt and config templates for instruction-tuning and evaluation manifests.  
- Practical notes on reproducibility, hyperparameter defaults, and typical failure modes.

## Bundled Resources

- `Bundled Resources/scripts` — runnable example scripts and a README describing usage.
- `Bundled Resources/references` — curated paper summaries with "what worked/what didn't".
- `Bundled Resources/assets` — prompt templates, example configs, and small dataset manifests.

## Examples

- "Give me a training recipe for a 3B decoder-only model on webtext + instruction-turns, with checkpoints and evaluation plan."
- "How do I quantize a 7B model for 8-bit inference while preserving instruction-following quality?"

## Inputs & Outputs

- Input: Natural-language prompts or structured requests specifying model size, dataset, desired latency, or evaluation metrics.  
- Output: Actionable steps, code snippets, file references inside bundled resources, and recommended readings.

## Limitations

- Bundled scripts are educational and minimal — they do not include model weights.  
- For production deployments, additional engineering (distributed training, secure data handling, CI/CD) is required.

## Contributing

- Add new papers to `Bundled Resources/references` as short summaries.
- Add scripts to `Bundled Resources/scripts` with clear README entries.

## Changelog

- 0.1.0 — Initial version with examples, references, and templates.
