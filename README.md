# Architecture Comparison Experiments

*Built for AI Programming, March–June 2025. Uploaded to GitHub in August 2026.*

Two small experiments, both asking the same question in a different setting: does the bigger, fancier architecture actually earn its extra parameters on this specific task, or does a simpler model get there just as well?

## The two experiments

**[`6dof-mlp-vs-transformer/`](6dof-mlp-vs-transformer/)** compares an MLP against a Transformer encoder-decoder on a 6-DoF robot arm forward-kinematics regression task. The MLP wins on position accuracy while using 19 times fewer parameters (12,326 vs 234,050). This experiment also catches and reports a teacher-forcing evaluation leak in the Transformer's numbers.

**[`binary-sequence-classification/`](binary-sequence-classification/)** compares a Transformer, an LSTM, a SimpleRNN, a Dense model, and a PoolingDense model on a synthetic counting task. All five models reach 100% accuracy, so once accuracy is no longer a differentiator, the smallest model that best fits the task's structure, PoolingDense at just 2,241 parameters, wins on cost.

## The shared theme

In both experiments, how well a model's structure actually fits the task turned out to matter more than raw model size. Forward kinematics is a purely feedforward function of the joint angles, and the MLP, which is architecturally the better fit, won on the harder of the two output types (position). The counting task's real underlying symmetry is permutation-invariance: only the *count* of ones matters, not their order. The one model built around that exact symmetry, PoolingDense, also turned out to be the cheapest one to run. Bigger and more general-purpose isn't automatically better when a task has a specific structure that a simpler, better-matched model can exploit directly.

See each experiment's own README for its full methodology, results, and how to run it.
