RLHF (Reinforcement Learning from Human Feedback)

- ArXiv: https://arxiv.org/abs/2203.02155
- Summary: Uses human preference data to train a reward model and optimize the base model via policy optimization, aligning outputs with human preferences.
- What worked: Improves instruction-following and helpfulness relative to supervised fine-tuning alone.
- What didn't: Can introduce reward model overfitting and undesirable optimization artifacts if the reward model is poor.
