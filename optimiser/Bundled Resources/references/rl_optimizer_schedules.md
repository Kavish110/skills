# Reinforcement Learning Optimizers and Schedules

- **Papers:**
  - "Proximal Policy Optimization Algorithms" — Schulman et al., 2017
  - "Deep Reinforcement Learning that Matters" — Henderson et al., 2017
  - "On Layer Normalization in the Transformer Architecture" — Xiong et al., 2020
- **Links:**
  - https://arxiv.org/abs/1707.06347
  - https://arxiv.org/abs/1709.06560
  - https://arxiv.org/abs/2002.04745

## Summary

Reinforcement learning optimizer design emphasizes stability across non-stationary reward signals, policy gradient variance, and value function approximation. Common best practices include AdamW-style adaptive updates, clipped gradient norms, learning rate schedules tuned to rollout batch size, and separate optimizers for actor and critic networks.

## What worked

- PPO with Adam/AdamW and linear learning rate decay remains the industry standard for policy optimization.
- Gradient clipping, reward normalization, and value function coefficient tuning are essential for stable training.
- Using a smaller learning rate for critic/value networks than for actor/policy networks often improves convergence.
- Adaptive optimizers perform well in RL when paired with consistent batch sizes and schedule warmup.

## What did not work

- Overly large adaptive learning rates can cause catastrophic policy collapse.
- Fixed schedules without decay often lead to poor final performance in continuous-control tasks.
- Mixing optimizer families across actor and critic without careful tuning can destabilize training.
- Using Adam without weight decay may reduce generalization for large RL policy networks.
