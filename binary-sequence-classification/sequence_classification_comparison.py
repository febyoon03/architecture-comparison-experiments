"""
Binary Sequence Classification Comparison

Task
    Length-20 binary sequences. Label = 1 iff the number of 1s > THRESHOLD.

Models
    Transformer encoder, LSTM, SimpleRNN, position-aware Dense,
    permutation-invariant PoolingDense, plus non-learned baselines.

This script is written for Keras 3 (TensorFlow or Torch backend).
Set KERAS_BACKEND before importing keras if you are not using TensorFlow.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Backend must be chosen before importing keras.
if "KERAS_BACKEND" not in os.environ:
    # Prefer TensorFlow when present (matches the original homework stack).
    try:
        import tensorflow  # noqa: F401

        os.environ["KERAS_BACKEND"] = "tensorflow"
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    seq_len: int = 20
    threshold: int = 10  # y = 1 iff count(1) > threshold
    n_train: int = 10_000
    n_val: int = 2_000
    n_test: int = 2_000
    embed_dim: int = 32
    rnn_units: int = 64
    ff_dim: int = 64
    num_heads: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-3
    seed: int = 42
    patience: int = 3
    output_dir: str = "."


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def expected_positive_rate(seq_len: int, threshold: int) -> float:
    """Exact Binomial(seq_len, 0.5) P(X > threshold)."""
    from math import comb

    total = 1 << seq_len
    return sum(comb(seq_len, k) for k in range(threshold + 1, seq_len + 1)) / total


def generate_split(
    n_samples: int, seq_len: int, threshold: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (int32), y (float32), and per-row counts (int32)."""
    x = rng.integers(0, 2, size=(n_samples, seq_len), dtype=np.int32)
    counts = x.sum(axis=1).astype(np.int32)
    y = (counts > threshold).astype(np.float32)
    return x, y, counts


def describe_split(name: str, y: np.ndarray, counts: np.ndarray) -> None:
    print(
        f"{name:<6} n={len(y):5d}  P(y=1)={y.mean():.4f}  "
        f"count mean={counts.mean():.2f}  "
        f"boundary share (count in {{9,10,11}})="
        f"{np.isin(counts, [9, 10, 11]).mean():.3f}"
    )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class AddPositionalEmbedding(layers.Layer):
    """Trainable positional embedding added to token embeddings.

    Kept as a Layer so the embedding weights are always part of the model
    graph, unlike calling Embedding() on a raw constant tensor at build time.
    """

    def __init__(self, seq_len: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_embedding = layers.Embedding(
            input_dim=seq_len, output_dim=embed_dim, name="positional_embedding"
        )

    def call(self, x):
        positions = keras.ops.arange(self.seq_len)
        return x + self.pos_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return config


class TransformerEncoderBlock(layers.Layer):
    """Single pre-norm-style residual encoder block.

    key_dim is embed_dim // num_heads so each head is a split of the
    model width. Keras still projects the concatenated heads back to
    embed_dim, which keeps the residual add shape-safe.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
            ],
            name="ffn",
        )
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop_1 = layers.Dropout(dropout)
        self.drop_2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.ln_1(x + self.drop_1(attn, training=training))
        ffn = self.ffn(x, training=training)
        return self.ln_2(x + self.drop_2(ffn, training=training))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Compile (shared — this is where the original accuracy bug lived)
# ---------------------------------------------------------------------------


def compile_binary_classifier(model: keras.Model, learning_rate: float) -> keras.Model:
    """Compile a logits model with a matching accuracy threshold.

    BinaryCrossentropy(from_logits=True) is correct for a Dense(1) with no
    sigmoid. The default metric string "accuracy" uses BinaryAccuracy with
    threshold=0.5, which is the probability decision boundary. Logits should
    be thresholded at 0.0 (sigmoid(0)=0.5).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
            keras.metrics.AUC(name="auc", from_logits=True),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_transformer(cfg: Config) -> keras.Model:
    inputs = layers.Input(shape=(cfg.seq_len,), dtype="int32")
    x = layers.Embedding(2, cfg.embed_dim, name="token_embedding")(inputs)
    x = AddPositionalEmbedding(cfg.seq_len, cfg.embed_dim)(x)
    x = TransformerEncoderBlock(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, name="logits")(x)
    return compile_binary_classifier(
        keras.Model(inputs, outputs, name="Transformer"), cfg.learning_rate
    )


def build_lstm(cfg: Config) -> keras.Model:
    inputs = layers.Input(shape=(cfg.seq_len,), dtype="int32")
    x = layers.Embedding(2, cfg.embed_dim)(inputs)
    x = layers.LSTM(cfg.rnn_units)(x)
    outputs = layers.Dense(1, name="logits")(x)
    return compile_binary_classifier(
        keras.Model(inputs, outputs, name="LSTM"), cfg.learning_rate
    )


def build_simple_rnn(cfg: Config) -> keras.Model:
    inputs = layers.Input(shape=(cfg.seq_len,), dtype="int32")
    x = layers.Embedding(2, cfg.embed_dim)(inputs)
    x = layers.SimpleRNN(cfg.rnn_units)(x)
    outputs = layers.Dense(1, name="logits")(x)
    return compile_binary_classifier(
        keras.Model(inputs, outputs, name="SimpleRNN"), cfg.learning_rate
    )


def build_dense(cfg: Config) -> keras.Model:
    """Position-aware MLP: Flatten keeps a separate weight per timestep."""
    inputs = layers.Input(shape=(cfg.seq_len,), dtype="int32")
    x = layers.Embedding(2, cfg.embed_dim)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, name="logits")(x)
    return compile_binary_classifier(
        keras.Model(inputs, outputs, name="Dense"), cfg.learning_rate
    )


def build_pooling_dense(cfg: Config) -> keras.Model:
    """Permutation-invariant MLP: mean-pool tokens, then classify.

    This is the architecture that matches the task (counting is order-free).
    """
    inputs = layers.Input(shape=(cfg.seq_len,), dtype="int32")
    x = layers.Embedding(2, cfg.embed_dim)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, name="logits")(x)
    return compile_binary_classifier(
        keras.Model(inputs, outputs, name="PoolingDense"), cfg.learning_rate
    )


MODEL_BUILDERS = {
    "Transformer": build_transformer,
    "LSTM": build_lstm,
    "SimpleRNN": build_simple_rnn,
    "Dense": build_dense,
    "PoolingDense": build_pooling_dense,
}


# ---------------------------------------------------------------------------
# Non-learned baselines
# ---------------------------------------------------------------------------


def majority_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> tuple[int, float]:
    """Predict the majority class observed on train."""
    label = int(y_train.mean() >= 0.5)
    acc = float((y_test == label).mean())
    return label, acc


def oracle_accuracy(counts: np.ndarray, y: np.ndarray, threshold: int) -> float:
    """The label rule itself. Test accuracy must be 1.0 if labels are consistent."""
    pred = (counts > threshold).astype(np.float32)
    return float((pred == y).mean())


def evaluate_by_count(
    counts: np.ndarray, y_true: np.ndarray, y_pred_label: np.ndarray
) -> dict[int, dict[str, float]]:
    """Accuracy broken down by number of 1s. Failures should concentrate at the threshold."""
    out: dict[int, dict[str, float]] = {}
    for k in range(int(counts.min()), int(counts.max()) + 1):
        mask = counts == k
        if not mask.any():
            continue
        out[k] = {
            "n": int(mask.sum()),
            "acc": float((y_pred_label[mask] == y_true[mask]).mean()),
            "true_rate": float(y_true[mask].mean()),
        }
    return out


def logits_to_label(logits: np.ndarray) -> np.ndarray:
    return (logits.reshape(-1) > 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Train / evaluate one model
# ---------------------------------------------------------------------------


def train_and_evaluate(
    name: str,
    model: keras.Model,
    cfg: Config,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    counts_test,
) -> dict:
    print("\n" + "=" * 64)
    print(f"Model: {name}")
    print("=" * 64)
    model.summary()
    n_params = int(model.count_params())
    print(f"Trainable parameters: {n_params:,}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.patience,
            restore_best_weights=True,
            mode="max",
        )
    ]

    t0 = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.perf_counter() - t0

    test_metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    logits = model.predict(x_test, batch_size=cfg.batch_size, verbose=0)
    pred = logits_to_label(logits)
    by_count = evaluate_by_count(counts_test, y_test, pred)

    boundary = np.isin(counts_test, [cfg.threshold - 1, cfg.threshold, cfg.threshold + 1])
    boundary_acc = float((pred[boundary] == y_test[boundary]).mean()) if boundary.any() else float("nan")

    result = {
        "params": n_params,
        "train_time_s": round(train_time, 2),
        "epochs_run": len(history.history["loss"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["accuracy"]),
        "test_auc": float(test_metrics["auc"]),
        "boundary_acc": boundary_acc,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "acc_by_count": {str(k): v for k, v in by_count.items()},
    }
    print(
        f"\n[{name}] params={n_params:,}  time={train_time:.1f}s  "
        f"epochs={result['epochs_run']}  test_acc={result['test_acc']:.4f}  "
        f"test_auc={result['test_auc']:.4f}  boundary_acc={boundary_acc:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: dict, majority_acc: float, oracle_acc: float) -> None:
    print("\n" + "=" * 64)
    print("Final comparison")
    print("=" * 64)
    header = (
        f"{'model':<14} {'params':>10} {'time_s':>8} {'epochs':>6} "
        f"{'test_acc':>10} {'test_auc':>10} {'boundary':>10}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'Majority':<14} {'0':>10} {'—':>8} {'—':>6} "
        f"{majority_acc:>10.4f} {'—':>10} {'—':>10}"
    )
    print(
        f"{'CountOracle':<14} {'0':>10} {'—':>8} {'—':>6} "
        f"{oracle_acc:>10.4f} {'—':>10} {'—':>10}"
    )
    for name, r in results.items():
        print(
            f"{name:<14} {r['params']:>10,} {r['train_time_s']:>8.1f} "
            f"{r['epochs_run']:>6} {r['test_acc']:>10.4f} "
            f"{r['test_auc']:>10.4f} {r['boundary_acc']:>10.4f}"
        )


def maybe_plot(results: dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, r in results.items():
        axes[0].plot(r["history"]["loss"], label=name)
        axes[1].plot(r["history"]["val_accuracy"], label=name)
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Val accuracy")
    axes[1].set_xlabel("epoch")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")

    # Accuracy by count for each model
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        ks = sorted(int(k) for k in r["acc_by_count"])
        accs = [r["acc_by_count"][str(k)]["acc"] for k in ks]
        ax.plot(ks, accs, marker="o", label=name)
    ax.axvline(10, color="gray", ls="--", label="threshold")
    ax.set_xlabel("number of 1s in the sequence")
    ax.set_ylabel("test accuracy")
    ax.set_title("Where models fail (should dip at the decision boundary)")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "accuracy_by_count.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Metric self-check (documents the original bug)
# ---------------------------------------------------------------------------


def demonstrate_accuracy_threshold_bug() -> None:
    """Show why metrics=['accuracy'] is wrong for logits."""
    y_true = np.array([[1.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    # Logits that are correct if thresholded at 0, but two sit in (0, 0.5).
    y_logit = np.array([[0.2], [2.0], [-0.2], [-2.0]], dtype=np.float32)

    wrong = keras.metrics.BinaryAccuracy(threshold=0.5)
    right = keras.metrics.BinaryAccuracy(threshold=0.0)
    wrong.update_state(y_true, y_logit)
    right.update_state(y_true, y_logit)
    print("Accuracy-threshold self-check on logits [0.2, 2, -0.2, -2], labels [1,1,0,0]")
    print(f"  threshold=0.5 (original metrics=['accuracy']): {float(wrong.result()):.2f}")
    print(f"  threshold=0.0 (correct for logits):            {float(right.result()):.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Compare sequence classifiers on a counting task")
    p.add_argument("--seq-len", type=int, default=Config.seq_len)
    p.add_argument("--threshold", type=int, default=Config.threshold)
    p.add_argument("--n-train", type=int, default=Config.n_train)
    p.add_argument("--n-val", type=int, default=Config.n_val)
    p.add_argument("--n-test", type=int, default=Config.n_test)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--output-dir", type=str, default=Config.output_dir)
    p.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_BUILDERS),
        help="Comma-separated subset of: " + ",".join(MODEL_BUILDERS),
    )
    args = p.parse_args()
    cfg = Config(
        seq_len=args.seq_len,
        threshold=args.threshold,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return cfg, [m.strip() for m in args.models.split(",") if m.strip()]


def main() -> None:
    cfg, model_names = parse_args()
    unknown = [n for n in model_names if n not in MODEL_BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown models {unknown}. Choose from {list(MODEL_BUILDERS)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Data")
    print("=" * 64)
    print(f"Keras {keras.__version__} backend={keras.config.backend()}")
    print(f"Task: seq_len={cfg.seq_len}, y=1 iff sum(x) > {cfg.threshold}")
    print(f"Theoretical P(y=1) = {expected_positive_rate(cfg.seq_len, cfg.threshold):.6f}")
    print("Majority-class accuracy if always predicting 0 ≈ 0.5881 for these defaults.")

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    x_train, y_train, counts_train = generate_split(cfg.n_train, cfg.seq_len, cfg.threshold, rng)
    x_val, y_val, counts_val = generate_split(cfg.n_val, cfg.seq_len, cfg.threshold, rng)
    x_test, y_test, counts_test = generate_split(cfg.n_test, cfg.seq_len, cfg.threshold, rng)

    describe_split("Train", y_train, counts_train)
    describe_split("Val", y_val, counts_val)
    describe_split("Test", y_test, counts_test)

    maj_label, maj_acc = majority_accuracy(y_train, y_test)
    ora_acc = oracle_accuracy(counts_test, y_test, cfg.threshold)
    print(f"Majority baseline: always predict {maj_label} → test_acc={maj_acc:.4f}")
    print(f"Count oracle:      sum > {cfg.threshold}     → test_acc={ora_acc:.4f}")
    if ora_acc != 1.0:
        raise RuntimeError("Label rule and y are inconsistent — data bug")

    demonstrate_accuracy_threshold_bug()

    results = {}
    for name in model_names:
        set_seed(cfg.seed)
        model = MODEL_BUILDERS[name](cfg)
        results[name] = train_and_evaluate(
            name,
            model,
            cfg,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            counts_test,
        )
        # Free graph / optimizer state between models.
        del model
        keras.backend.clear_session()

    print_summary(results, maj_acc, ora_acc)
    maybe_plot(results, output_dir)

    payload = {
        "config": asdict(cfg),
        "backend": keras.config.backend(),
        "majority_acc": maj_acc,
        "oracle_acc": ora_acc,
        "theoretical_p_positive": expected_positive_rate(cfg.seq_len, cfg.threshold),
        "results": results,
    }
    out_json = output_dir / "comparison_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
