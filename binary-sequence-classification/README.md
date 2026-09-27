# Binary Sequence Classification: Do You Need the Bigger Model?

Five different model architectures compared on a synthetic but genuinely non-trivial task: given a length-20 binary sequence, predict whether it has more than 10 ones.

## Motivation

This task actually has a perfect closed-form solution (`sum(x) > 10`), so a simple "count oracle" gets 100% correct by construction. The real question isn't whether a model *can* solve it, but how much model it actually takes to learn that rule from examples, and whether fancier architectures like attention and recurrence actually earn their extra parameters and training time over a much simpler baseline.

## Approach

Five learned models were compared, all using the same 10,000/2,000/2,000 train/validation/test split, the same random seed, and the same 8-epoch training budget with early stopping (patience of 3).

**Transformer.** Uses token and trainable positional embeddings, a single encoder block, and mean pooling.

**LSTM.** A single recurrent layer.

**SimpleRNN.** A single plain (vanilla) RNN layer.

**Dense.** A position-aware, fully-connected model. It flattens the sequence, so it is not permutation-invariant.

**PoolingDense.** A permutation-invariant model. It pools the sequence before the dense layers, so structurally it can't see position at all, only the total count, which is actually the real decision rule.

Two non-learned baselines were also included for comparison: a simple majority-class prediction, and the count oracle itself (`sum(x) > threshold`, the actual ground-truth rule). The oracle is guaranteed 100% by definition, and mainly acts as a sanity check that the labeling logic isn't buggy.

One correctness detail worth calling out: with `BinaryCrossentropy(from_logits=True)`, Keras's default `metrics=["accuracy"]` silently uses a 0.5 threshold, which is actually wrong when working with logits (the correct decision boundary there is 0.0). The script demonstrates this directly on a small logits example before training anything, and uses the corrected `BinaryAccuracy(threshold=0.0)` throughout the rest of the comparison.

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

"Boundary acc" means accuracy measured only on sequences with 9, 10, or 11 ones, the cases closest to the decision threshold, and the only place a model could plausibly still fail even while sitting at 100% overall accuracy.

**Takeaway.** Every learned model reaches 100% test accuracy on this task, including on the boundary cases. The smallest model here, PoolingDense at just 2,241 parameters, is also one of the fastest to train, and its permutation-invariant structure is actually a better match to the real symmetry of the task (the label only depends on the *count* of ones, not their positions) than the larger, position-aware Dense model. Once every model reaches the same accuracy
