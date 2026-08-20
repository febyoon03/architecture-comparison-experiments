"""
Assignment 1 (revised): MLP 6-DoF forward kinematics

Maps ABB IRB2400 joint angles (q1..q6) to end-effector pose
(x, y, z, yaw, pitch, roll).

This file is a single training program. Ownership:
  load_and_validate  – CSV I/O and schema checks
  split_indices      – isolation of test before any statistics
  Standardizer       – train-only mean/std, inverse transform, persistence
  build_mlp          – 6 → 6 static regressor
  train_and_evaluate – fit, checkpoint, physical-unit metrics

Evaluation is always inverse-transformed. Position (x,y,z) and
orientation (yaw,pitch,roll) are never collapsed into one “mm/rad” number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# TensorFlow is imported after the CLI so `--help` works without a GPU spin-up.
SEED = 42
INPUT_COLS = ["q1_in", "q2_in", "q3_in", "q4_in", "q5_in", "q6_in"]
TARGET_COLS = ["x", "y", "z", "yaw", "pitch", "roll"]
POS_IDX = (0, 1, 2)
ORI_IDX = (3, 4, 5)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
@dataclass
class MLPConfig:
    csv_path: str
    seed: int = SEED
    train_ratio: float = 0.8
    val_ratio_within_train: float = 0.2  # → 64 / 16 / 20 of all rows
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    initial_lr: float = 1e-3
    cosine_alpha: float = 0.1
    loss: str = "mse"  # "mse" | "huber"
    huber_delta: float = 1.0
    hidden: Tuple[int, ...] = (128, 64, 32)
    checkpoint: str = "best_mlp_6dof.keras"
    scaler_path: str = "best_mlp_6dof_scaler.json"
    history_plot: str = "mlp_loss_curve.png"
    circular_ori: bool = True


def resolve_csv_path(cli_path: Optional[str]) -> str:
    """Prefer argv, then env, then local files. Never assume Colab Drive."""
    candidates = [
        cli_path,
        os.environ.get("IRB2400_CSV"),
        "./datasetIRB2400.csv",
        os.path.join(os.path.dirname(__file__), "datasetIRB2400.csv"),
        "/content/drive/MyDrive/datasetIRB2400.csv",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "CSV not found. Pass --csv PATH or set IRB2400_CSV. "
        f"Looked at: {[c for c in candidates if c]}"
    )


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
def load_and_validate(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in INPUT_COLS + TARGET_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Have: {df.columns.tolist()}")

    if df.empty:
        raise ValueError("CSV has no rows.")

    X = df[INPUT_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLS].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        bad = int((~np.isfinite(X)).any(axis=1).sum() + (~np.isfinite(y)).any(axis=1).sum())
        raise ValueError(f"Non-finite values in X or y (affected rows ~{bad}).")

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"X {X.shape}  y {y.shape}")
    return X, y


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio_within_train: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle once. Test is carved out before any scaler statistics."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train_full = int(n * train_ratio)
    train_full, test_idx = idx[:n_train_full], idx[n_train_full:]
    n_val = int(len(train_full) * val_ratio_within_train)
    # Last slice of the shuffled train block → matches Keras validation_split
    # semantics we replace: explicit arrays instead of a silent tail split.
    val_idx = train_full[:n_val]
    train_idx = train_full[n_val:]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(f"Degenerate split: train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Split train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# 3. Standardizer (owns the physical ↔ training map)
# ---------------------------------------------------------------------------
class Standardizer:
    """Per-column z-score. Fit on train only. std==0 is rejected, not silenced."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, a: np.ndarray) -> "Standardizer":
        mean = a.mean(axis=0).astype(np.float64)
        std = a.std(axis=0).astype(np.float64)
        if np.any(std < self.eps):
            dead = np.where(std < self.eps)[0].tolist()
            raise ValueError(f"Near-zero std on columns {dead}; cannot scale.")
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        self._check()
        return ((a.astype(np.float64) - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        self._check()
        return (a.astype(np.float64) * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> dict:
        self._check()
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist(), "eps": self.eps}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        obj = cls(eps=float(d.get("eps", 1e-8)))
        obj.mean_ = np.asarray(d["mean"], dtype=np.float64)
        obj.std_ = np.asarray(d["std"], dtype=np.float64)
        return obj

    def _check(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer used before fit / load.")


def save_bundle(path: str, x_scaler: Standardizer, y_scaler: Standardizer, cfg: MLPConfig) -> None:
    payload = {
        "x": x_scaler.to_dict(),
        "y": y_scaler.to_dict(),
        "input_cols": INPUT_COLS,
        "target_cols": TARGET_COLS,
        "seed": cfg.seed,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote scaler bundle → {path}")


# ---------------------------------------------------------------------------
# 4. Metrics (physical units)
# ---------------------------------------------------------------------------
def circular_abs(delta: np.ndarray) -> np.ndarray:
    """Smallest angle |Δ| in radians. Safe even if the dataset is in degrees
    only if you convert first — this assignment treats orientation as rad."""
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def report_mae(y_true: np.ndarray, y_pred: np.ndarray, circular_ori: bool) -> dict:
    err = y_true.astype(np.float64) - y_pred.astype(np.float64)
    abs_err = np.abs(err)
    if circular_ori:
        abs_err[:, list(ORI_IDX)] = circular_abs(err[:, list(ORI_IDX)])

    pos = abs_err[:, list(POS_IDX)].mean()
    ori = abs_err[:, list(ORI_IDX)].mean()
    per = abs_err.mean(axis=0)

    print("\nTest metrics (physical units, all test samples):")
    print(f"  MAE position  (x,y,z mean)     : {pos:.6f}")
    print(f"  MAE orientation (yaw,pitch,roll): {ori:.6f}  "
          f"{'(circular)' if circular_ori else '(linear)'}")
    for name, value in zip(TARGET_COLS, per):
        print(f"  MAE {name:5s}: {value:.6f}")
    print("  Note: do not average position and orientation into one number.")
    return {
        "mae_position": float(pos),
        "mae_orientation": float(ori),
        "mae_per_axis": {k: float(v) for k, v in zip(TARGET_COLS, per)},
    }


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------
def build_mlp(tf, input_dim: int, hidden: Sequence[int], seed: int):
    """Dense-BN-ReLU stacks. BN before ReLU so the nonlinearity sees
    normalized pre-activations. Last layer is linear (regression)."""
    initializer = tf.keras.initializers.HeNormal(seed=seed)
    inp = tf.keras.Input(shape=(input_dim,), name="joints")
    x = inp
    for i, units in enumerate(hidden):
        x = tf.keras.layers.Dense(
            units,
            kernel_initializer=initializer,
            name=f"dense_{i}_{units}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{i}")(x)
    out = tf.keras.layers.Dense(6, name="pose")(x)
    return tf.keras.Model(inp, out, name="mlp_fk")


def make_loss(tf, cfg: MLPConfig):
    if cfg.loss == "mse":
        return "mse"
    if cfg.loss == "huber":
        return tf.keras.losses.Huber(delta=cfg.huber_delta)
    raise ValueError(f"Unknown loss {cfg.loss}")


# ---------------------------------------------------------------------------
# 6. Train / eval
# ---------------------------------------------------------------------------
def set_seeds(tf, seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_and_evaluate(cfg: MLPConfig) -> dict:
    import tensorflow as tf
    import matplotlib.pyplot as plt

    set_seeds(tf, cfg.seed)

    X, y = load_and_validate(cfg.csv_path)
    train_idx, val_idx, test_idx = split_indices(
        len(X), cfg.train_ratio, cfg.val_ratio_within_train, cfg.seed
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    x_scaler = Standardizer().fit(X_train)
    y_scaler = Standardizer().fit(y_train)
    save_bundle(cfg.scaler_path, x_scaler, y_scaler, cfg)

    X_train_s, y_train_s = x_scaler.transform(X_train), y_scaler.transform(y_train)
    X_val_s, y_val_s = x_scaler.transform(X_val), y_scaler.transform(y_val)
    X_test_s = x_scaler.transform(X_test)

    steps_per_epoch = max(1, int(np.ceil(len(X_train_s) / cfg.batch_size)))
    decay_steps = steps_per_epoch * cfg.epochs
    print(f"steps/epoch={steps_per_epoch}  cosine decay_steps={decay_steps}")

    model = build_mlp(tf, input_dim=6, hidden=cfg.hidden, seed=cfg.seed)
    model.summary()

    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.initial_lr,
        decay_steps=decay_steps,
        alpha=cfg.cosine_alpha,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=make_loss(tf, cfg),
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            cfg.checkpoint,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    print("Training finished (best val_loss weights restored).")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel(cfg.loss)
    plt.title("MLP FK training loss (standardized pose space)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.history_plot, dpi=150)
    plt.close()
    print(f"Wrote {cfg.history_plot}")

    y_pred = y_scaler.inverse(model.predict(X_test_s, verbose=0))
    metrics = report_mae(y_test, y_pred, circular_ori=cfg.circular_ori)
    metrics["checkpoint"] = cfg.checkpoint
    metrics["scaler"] = cfg.scaler_path
    return metrics


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> MLPConfig:
    p = argparse.ArgumentParser(description="MLP 6-DoF FK regressor")
    p.add_argument("--csv", default=None, help="Path to datasetIRB2400.csv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--loss", choices=("mse", "huber"), default="mse")
    p.add_argument("--no-circular-ori", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    return MLPConfig(
        csv_path=resolve_csv_path(args.csv),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss=args.loss,
        circular_ori=not args.no_circular_ori,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    train_and_evaluate(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
