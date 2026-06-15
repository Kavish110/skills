## Instruction-tuning prompt templates

- Simple instruction-completion:

  Instruction: {instruction}
  Input: {input}
  Response:

- System + few-shot template:

  System: You are a helpful assistant.
  Example 1:
  Q: {example_q1}
  A: {example_a1}
  ---
  Now follow this instruction:
  Instruction: {instruction}
  Response:

## Evaluation prompts

- Use zero-shot instruction evaluation and store both model response and reference answer for human scoring.
