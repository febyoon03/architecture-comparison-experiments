# Architecture Comparison Experiments

*Built for AI Programming, March–June 2025. Uploaded to GitHub in August 2026.*

Two small experiments, same question each time: does the bigger/fancier architecture actually earn its extra parameters on this specific task, or does a simpler model get there just as well?

## Pieces

- [`6dof-mlp-vs-transformer/`](6dof-mlp-vs-transformer/) — MLP vs. Transformer encoder-decoder on 6-DoF robot arm forward kinematics regression. MLP wins on position with 19x fewer parameters (12,326 vs 234,050); also catches and reports a teacher-forcing evaluation leak in the Transformer's numbers.
- [`binary-sequence-classification/`](binary-sequence-classification/) — Transformer vs. LSTM vs. SimpleRNN vs. Dense vs. PoolingDense on a synthetic counting task. All five reach 100% accuracy; the smallest, best-matched-to-the-task model (PoolingDense, 2,241 params) wins on cost once accuracy saturates.

## Shared theme

In both cases, model capacity and task fit turned out to matter more than model capacity alone. Forward kinematics is a purely feedforward function of the joint angles, and the MLP — architecturally the better match — won on that task's harder axis (position). The counting task's real symmetry is permutation-invariance (only the *count* of 1s matters, not their order), and the model built around that symmetry (PoolingDense) was also the cheapest one. Bigger and more general-purpose isn't automatically better when the task has a specific structure a simpler model can be built to exploit directly.

See each piece's own README for full methodology, results tables, and how to run.
