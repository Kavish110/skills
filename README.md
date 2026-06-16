# 🧠 AI/ML Coding Skills

A curated collection of **skills** — structured knowledge files that give AI coding assistants (like Claude) authoritative, opinionated guidance for implementing, training, and deploying AI/ML models. Think of each skill as a senior ML engineer embedded in your assistant: it knows what works, what doesn't, and hands you runnable code rather than generic advice.

---

## What is a "skill"?

A skill is a structured markdown file that an AI coding assistant reads before answering questions in a given domain. It encodes:

- **When to activate** — trigger conditions and example prompts
- **Curated knowledge** — research summaries, design patterns, common failure modes
- **Runnable resources** — scripts, configs, and templates the assistant can reference
- **Practical defaults** — hyperparameters, tooling choices, and reproducibility notes

Skills make the assistant *domain-aware* rather than generically helpful — it stops hedging and starts recommending.

---

## Current Skills

| Skill | Status | Description |
|---|---|---|
| `code-llm` | ✅ Available | Architecture, training recipes, evaluation, and inference for Large Language Models |
| `optimisers` | ✅ Available | Optimiser selection, hyperparameter tuning, schedulers, and training stability |

---

## Roadmap

Skills planned in rough priority order:

| Skill | Status | Scope |
|---|---|---|
| `diffusion` | 🔜 Planned | Diffusion model theory, DDPM/DDIM/EDM, training and sampling |
| `flow-matching` | 🔜 Planned | Continuous normalizing flows, rectified flow, conditional flow matching |
| `quantisation` | 🔜 Planned | PTQ, QAT, GPTQ, AWQ, mixed-precision, and deployment targets |
| `jax-llm` | 💡 Exploring | JAX/Flax equivalents of the LLM skill |
| `jax-diffusion` | 💡 Exploring | JAX/Flax equivalents of the diffusion + flow matching skills |

---

## Repo Structure

```
.
├── README.md
└── skills/
    ├── code-llm/
    │   ├── SKILL.md
    │   └── Bundled Resources/
    │       ├── scripts/       # Runnable training, eval, and inference examples
    │       ├── references/    # Paper summaries — what worked and what didn't
    │       └── assets/        # Prompt templates, configs, dataset manifests
    └── optimisers/
        ├── SKILL.md
        └── Bundled Resources/
            ├── scripts/
            ├── references/
            └── assets/
```

New skills follow the same layout: a `SKILL.md` at the root of the folder plus a `Bundled Resources/` directory for scripts, references, and assets.

---

## Using a Skill

Point your AI assistant at the relevant `SKILL.md` before asking ML questions. With Claude, you can do this by uploading the file or placing it in a project. The skill header defines when the assistant should activate it automatically.

Example prompts that trigger the `code-llm` skill:

```
"Give me a training recipe for a 3B decoder-only model with an evaluation plan."
"How do I apply LoRA to a 7B model for instruction fine-tuning?"
"Quantize this model to 8-bit without wrecking perplexity."
```

---

## Contributing

Contributions are welcome — whether that's a new skill, additional paper summaries, or improved scripts.

**Adding a new skill:**

1. Create a folder under `skills/<skill-name>/`
2. Write a `SKILL.md` following the existing format (frontmatter, overview, trigger conditions, bundled resources)
3. Add runnable examples to `Bundled Resources/scripts/` with a short README
4. Add paper summaries to `Bundled Resources/references/` — keep them brief: *what was claimed, what held up, what didn't*
5. Open a PR with a one-line description of what the skill covers

**Improving an existing skill:**

- New papers go in `Bundled Resources/references/` as short summaries
- Script improvements should be self-contained and reproducible
- Bump the version in the `SKILL.md` frontmatter

---

## Design Philosophy

- **Opinionated over comprehensive.** A skill that says "use AdamW with cosine decay as your default" is more useful than one that lists every optimiser.
- **Runnable over theoretical.** Every skill should ship at least one script you can actually execute.
- **Framework-honest.** PyTorch is the primary target for most skills; JAX support is called out explicitly where it exists.
- **Failure-aware.** Paper summaries note what *didn't* work, not just what did.

---

## Author

**Kavish Patel**

---

## License

MIT
