# 6-DoF Forward Kinematics: MLP vs. Transformer

The same regression task, tried with two different architectures: predict an ABB IRB2400 robot arm's end-effector pose (x, y, z, yaw, pitch, roll) from its 6 joint angles.

## Motivation

Forward kinematics is a deterministic, purely feedforward function of the joint angles. There's no sequential dependency between joint 1 and joint 2, the way there is between word 1 and word 2 of a sentence. On paper, a Transformer encoder-decoder is a real architectural mismatch for this task. This experiment checks whether that mismatch actually costs anything in practice, and how much.

## Approach

**MLP.** A simple stack of dense layers (128, then 64, then 32, then 6 outputs), with batch normalization before each ReLU, trained directly on the 6-to-6 mapping using Keras/TensorFlow.

**Transformer.** An encoder-decoder that treats the 6 joint angles and the 6 pose outputs as sequences, built in PyTorch. It uses sinusoidal positional encoding on the encoder side, and a learned start token (in scaled target space, not zero) on the decoder side. It's trained with teacher forcing, but evaluated with true autoregressive decoding, since that's what the model would actually have to do at inference time, with no ground truth available to feed it.

For both models, position error (x, y, z, in mm) and orientation error (yaw, pitch, roll, in radians, using circular distance) are always reported separately, never averaged together into one combined "mm/rad" number. Those are two different physical quantities, and combining them would hide which one a model is actually struggling with.

Also for both models, the train/test split, the scaler fitting, and the un-scaling of predictions back to real units are all done train-only and kept isolated from the test set, so no test information leaks into training.

## Results

| | MLP | Transformer (autoregressive) |
|---|---|---|
| Params | 12,326 | 234,050 |
| Test MAE position | **114.0mm** | 170.9mm |
| Test MAE orientation | 1.204 rad | 1.251 rad |

**Takeaway.** The MLP wins clearly on position, using 19 times fewer parameters. The architectural mismatch really does cost the Transformer something in practice here, not just in theory. Orientation results are close between the two, but both are still fairly poor in absolute terms, at roughly 1.2 radians average error on yaw, pitch, and roll. With only 8,000 random poses and a 20-30 epoch training budget, there simply isn't enough training signal yet for either model to handle the harder orientation axes well.

**A second, arguably more important finding.** The Transformer's teacher-forced evaluation (160.3mm position MAE) looks meaningfully better than its true autoregressive evaluation (170.9mm), because teacher forcing feeds the model ground-truth previous outputs that it won't actually have available at real inference time. The script measures both numbers and prints the gap explicitly, so the better-looking but misleading number can't get reported by accident. Only the autoregressive number reflects the model's real forward-kinematics accuracy.

## Tech stack

For the MLP: Python, TensorFlow/Keras, pandas, and NumPy. For the Transformer: Python, PyTorch, pandas, and NumPy.

## How to run it

Both scripts expect a file called `datasetIRB2400.csv`, a publicly available ABB IRB2400 forward-kinematics dataset that is not included in this repo. Place it in this folder, or point to it with `--csv` or the `IRB2400_CSV` environment variable.

```bash
pip install tensorflow pandas numpy matplotlib   # for the MLP
python mlp_6dof_forward_kinematics.py --csv path/to/datasetIRB2400.csv

pip install torch pandas numpy matplotlib        # for the Transformer
python transformer_6dof_forward_kinematics.py --csv path/to/datasetIRB2400.csv
```

## Limitations and what's next

With only 8,000 rows and a short training budget (30 epochs for the MLP, up to 30 epochs with early stopping for the Transformer), this is a pilot-scale experiment rather than a fully converged final model for either architecture. The point here is the relative comparison between the two, not the absolute MAE numbers.
