# 6-DoF Forward Kinematics: MLP vs. Transformer

Same regression task, two architectures: predict an ABB IRB2400 robot arm's end-effector pose (x, y, z, yaw, pitch, roll) from its 6 joint angles.

## Motivation

Forward kinematics is a deterministic, purely feedforward function of the joint angles — there's no sequential dependency between q1 and q2 the way there is between word 1 and word 2 of a sentence. A Transformer encoder-decoder is a real architectural mismatch for this task on paper; this experiment checks whether that mismatch actually costs anything in practice, and by how much.

## Approach

- **MLP:** Dense(128)→Dense(64)→Dense(32)→Dense(6), BatchNorm before each ReLU, trained directly on the 6→6 mapping (Keras/TensorFlow).
- **Transformer:** encoder-decoder over the 6 joint angles and 6 pose outputs treated as sequences (PyTorch), with sinusoidal positional encoding on the encoder side and a learned BOS token (in scaled target space, not zero) for the decoder. Trained with teacher forcing; evaluated with true autoregressive decoding, since that's what the model would actually have to do at inference time with no ground truth to feed it.
- **Both models:** MAE for position (x,y,z, in mm) and orientation (yaw,pitch,roll, in rad, using circular distance) are always reported separately, never averaged into one "mm/rad" number — those are different physical quantities and collapsing them hides which one a model is actually failing at.
- **Both models:** the CSV split, scaler fitting, and target un-scaling are all train-only / test-isolated — no leakage of test statistics into training.

## Results

| | MLP | Transformer (autoregressive) |
|---|---|---|
| Params | 12,326 | 234,050 |
| Test MAE position | **114.0mm** | 170.9mm |
| Test MAE orientation | 1.204 rad | 1.251 rad |

**Takeaway:** the MLP wins clearly on position with 19x fewer parameters — the architectural mismatch does cost the Transformer something real here, not just in theory. Orientation is close between the two, but both are still poor in absolute terms (~1.2 rad average error on yaw/pitch/roll) — 8,000 random poses and a 20-30 epoch budget just isn't enough training signal for the harder orientation axes, for either model.

**A second, arguably more important finding:** the Transformer's teacher-forced evaluation (position MAE 160.3mm) looks meaningfully better than its true autoregressive evaluation (170.9mm) — because teacher forcing feeds the model ground-truth previous outputs it won't have at real inference time. The script measures both and prints the gap explicitly so that number can't get reported by accident; only the autoregressive number is the real FK accuracy.

## Tech stack

MLP: Python, TensorFlow/Keras, pandas, NumPy. Transformer: Python, PyTorch, pandas, NumPy.

## How to run

Both scripts expect `datasetIRB2400.csv` (ABB IRB2400 forward-kinematics dataset — this is a publicly available robotics dataset, not included in this repo; place it in this folder, or point to it via `--csv` or the `IRB2400_CSV` environment variable).

```bash
pip install tensorflow pandas numpy matplotlib   # for the MLP
python mlp_6dof_forward_kinematics.py --csv path/to/datasetIRB2400.csv

pip install torch pandas numpy matplotlib        # for the Transformer
python transformer_6dof_forward_kinematics.py --csv path/to/datasetIRB2400.csv
```

## Limitations / what's next

- 8,000 rows and a short training budget (30 MLP epochs / up to 30 Transformer epochs with early stopping) is a pilot scale, not a converged final model for either architecture — the *relative* comparison is the point, not the absolute MAE.
- Orientation error is high for both models; a longer training run or an orientation-specific loss weighting would be the natural next step before drawing conclusions about the orientation axes specifically.
- The two scripts use different data splits (MLP: 64/16/20, Transformer: 80/10/10) because that's what each was independently built with — the same 8,000-row source file underlies both, so this doesn't bias the comparison, but it's worth flagging for exact reproducibility.

## Credits

Built independently for a course assignment (Assignment 1: MLP, Assignment 2: Transformer). No external code reused beyond TensorFlow/PyTorch themselves.
