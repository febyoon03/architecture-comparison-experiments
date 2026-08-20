# Binary Sequence Classification: Do You Need the Bigger Model?

Five architectures compared on a synthetic but genuinely non-trivial task: given a length-20 binary sequence, predict whether it has more than 10 ones.

## Motivation

This task has a perfect closed-form solution (`sum(x) > 10`) — a "count oracle" gets 100% by construction. The question isn't whether a model *can* solve it, but how much model it actually takes to learn that rule from examples, and whether the fancier architectures (attention, recurrence) earn their extra parameters and training time over a much simpler baseline.

## Approach

Five learned models, all on the same 10,000/2,000/2,000 train/val/test split, same seed, same 8-epoch budget with early stopping (patience 3):

- **Transformer** — token + trainable positional embedding, single encoder block, mean-pooled.
- **LSTM** — single recurrent layer.
- **SimpleRNN** — single vanilla-RNN layer.
- **Dense** — position-aware fully-connected (flattens the sequence, so it isn't permutation-invariant).
- **PoolingDense** — permutation-invariant: pools before the dense layers, so it structurally can't see position, only the count — which is the actual decision rule.

Two non-learned baselines: majority-class prediction, and the count oracle (`sum(x) > threshold`, the ground-truth rule — this one is guaranteed 100% and mainly serves as a sanity check that the label logic isn't buggy).

One correctness detail worth keeping: with `BinaryCrossentropy(from_logits=True)`, Keras's default `metrics=["accuracy"]` silently uses a 0.5 threshold, which is wrong for logits (the correct decision boundary is 0.0). The script demonstrates this directly on a small logits example before training anything, and uses the corrected `BinaryAccuracy(threshold=0.0)` throughout.

## Results

| Model | Params | Train time | Epochs | Test acc | Test AUC | Boundary acc |
|---|---|---|---|---|---|---|
| Majority baseline | 0 | — | — | 59.85% | — | — |
| Count oracle | 0 | — | — | 100% | — | — |
| PoolingDense | 2,241 | 4.7s | 4 | 100% | 1.0 | 100% |
| Transformer | 9,281 | 16.4s | 4 | 100% | 1.0 | 100% |
| SimpleRNN | 6,337 | 24.5s | 6 | 100% | 1.0 | 100% |
| LSTM | 24,961 | 30.1s | 6 | 100% | 1.0 | 100% |
| Dense | 41,153 | 6.2s | 4 | 100% | 1.0 | 100% |

"Boundary acc" is accuracy restricted to sequences with 9, 10, or 11 ones — the cases nearest the decision threshold, and the only place a model could plausibly still fail even at 100% overall accuracy.

**Takeaway:** every learned model reaches 100% test accuracy on this task, including boundary cases. The smallest model here (PoolingDense, 2,241 params) is also one of the fastest to train, and its permutation-invariant structure is actually a better match to the task's real symmetry (the label only depends on the *count* of ones, not their positions) than the larger, position-aware Dense model. Once every model saturates on accuracy, parameter count and training time — not accuracy — become the only real differentiators, and the smallest, best-matched-to-the-task model wins on both.

## Tech stack

Python, Keras 3 (TensorFlow or PyTorch backend — auto-detected).

## How to run

```bash
pip install keras tensorflow numpy matplotlib
python sequence_classification_comparison.py
```
Runs all 5 models by default; restrict with `--models Transformer,PoolingDense` etc. Writes `comparison_results.json`, `training_curves.png`, and `accuracy_by_count.png` to `--output-dir` (default: current directory).

## Limitations / what's next

- All five models hit the accuracy ceiling on this task, so it doesn't actually separate model quality — a harder or noisier version of the same task (e.g. a tighter threshold margin, or label noise) would likely show more differentiation between architectures and is a natural next step.
- Training time here is wall-clock on whatever machine ran it, not a controlled hardware benchmark — treat the relative ordering as informative, not the absolute seconds.
