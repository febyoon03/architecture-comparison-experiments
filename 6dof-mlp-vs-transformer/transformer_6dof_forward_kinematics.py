"""
Assignment 2 (revised): Transformer encoder–decoder 6-DoF FK

Same task as the MLP: q1..q6 → (x, y, z, yaw, pitch, roll).

Why a sequence model at all
  Encoder PE on q1..q6 is defensible (serial kinematic chain).
  Causal decoding of pose components is a course constraint, not physics:
  x does not generate y the way token t generates token t+1.

What this file therefore does
  Train with teacher forcing (standard seq2seq).
  Select checkpoints and report TEST numbers with autoregressive decode.
  Also print teacher-forced MAE as a diagnostic so the gap is visible.

BOS is a learned scalar in scaled-target space, not 0 (which would be the
per-axis mean after StandardScaler).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

SEED = 42
INPUT_COLS = [f"q{i}_in" for i in range(1, 7)]
TARGET_COLS = ["x", "y", "z", "yaw", "pitch", "roll"]
POS_IDX = (0, 1, 2)
ORI_IDX = (3, 4, 5)


# ---------------------------------------------------------------------------
# 1. Config / device / seeds
# ---------------------------------------------------------------------------
@dataclass
class TFConfig:
    csv_path: str
    seed: int = SEED
    batch_size: int = 512
    epochs: int = 30
    patience: int = 7
    lr: float = 1e-3
    weight_decay: float = 1e-5
    d_model: int = 64
    nhead: int = 4
    num_enc: int = 2
    num_dec: int = 2
    ff: int = 256
    dropout: float = 0.1
    seq_len: int = 6
    checkpoint: str = "best_transformer_6dof.pth"
    scaler_path: str = "best_transformer_6dof_scalers.json"
    plot_path: str = "transformer_mae_curve.png"
    circular_ori: bool = True
    num_workers: int = 0


def resolve_csv_path(cli_path: Optional[str]) -> str:
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


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
class SixDoFDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in INPUT_COLS + TARGET_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Have: {df.columns.tolist()}")
    if df.empty:
        raise ValueError("CSV has no rows.")
    x = df[INPUT_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLS].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values in X or y.")
    print(f"Loaded {len(df)} rows from {path}")
    return x, y


class Standardizer:
    """Train-only z-score. Inverse-transform is required for any published MAE."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, a: np.ndarray, name: str) -> "Standardizer":
        mean = a.mean(axis=0).astype(np.float64)
        std = a.std(axis=0).astype(np.float64)
        if np.any(std < self.eps):
            raise ValueError(f"{name}: near-zero std on columns {np.where(std < self.eps)[0].tolist()}")
        self.mean_, self.std_ = mean, std
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        return ((a.astype(np.float64) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        return (a.astype(np.float64) * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}


def dump_scalers(path: str, x_scaler: Standardizer, y_scaler: Standardizer, cfg: TFConfig) -> None:
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


def split_80_10_10(
    x: np.ndarray, y: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))
    n = len(x)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_i, val_i, test_i = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    if min(len(train_i), len(val_i), len(test_i)) == 0:
        raise ValueError("Degenerate 80/10/10 split")
    return x[train_i], x[val_i], x[test_i], y[train_i], y[val_i], y[test_i]


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sin/cos PE interleaving.")
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class SmallTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_enc: int = 2,
        num_dec: int = 2,
        ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.src_emb = nn.Linear(1, d_model)
        self.tgt_emb = nn.Linear(1, d_model)
        # Learned BOS in *scaled target* space. Not zero (the column mean).
        self.bos = nn.Parameter(torch.zeros(1))
        self.pos_src = PositionalEncoding(d_model, seq_len, dropout)
        # tgt length during train is seq_len (BOS + y[:-1]); during AR it grows 1..seq_len
        self.pos_tgt = PositionalEncoding(d_model, seq_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def bos_token(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.bos.expand(batch, 1).to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        src: (B, src_len) joint angles, already scaled
        tgt: (B, tgt_len) decoder prefix (BOS + previously generated / GT tokens)
        returns: (B, tgt_len) one prediction per prefix position
        """
        src_e = self.pos_src(self.src_emb(src.unsqueeze(-1)))
        tgt_e = self.pos_tgt(self.tgt_emb(tgt.unsqueeze(-1)))
        mask = self.causal_mask(tgt.size(1), src.device)
        out = self.transformer(src=src_e, tgt=tgt_e, tgt_mask=mask)
        return self.fc_out(out).squeeze(-1)

    @torch.no_grad()
    def infer(self, src: torch.Tensor) -> torch.Tensor:
        """Open-loop decode: each step only sees its own previous predictions."""
        self.eval()
        b = src.size(0)
        ys = self.bos_token(b, src.device)
        preds: List[torch.Tensor] = []
        for _ in range(self.seq_len):
            out = self.forward(src, ys)
            nxt = out[:, -1:]
            preds.append(nxt)
            ys = torch.cat([ys, nxt], dim=1)
        return torch.cat(preds, dim=1)


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def circular_abs(delta: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def physical_errors(
    pred_s: np.ndarray,
    true_s: np.ndarray,
    y_scaler: Standardizer,
    circular_ori: bool,
) -> np.ndarray:
    pred = y_scaler.inverse_transform(pred_s)
    true = y_scaler.inverse_transform(true_s)
    err = true - pred
    abs_err = np.abs(err)
    if circular_ori:
        abs_err[:, list(ORI_IDX)] = circular_abs(err[:, list(ORI_IDX)])
    return abs_err


def summarize_abs(abs_err: np.ndarray, title: str) -> Dict[str, float]:
    pos = float(abs_err[:, list(POS_IDX)].mean())
    ori = float(abs_err[:, list(ORI_IDX)].mean())
    per = abs_err.mean(axis=0)
    print(f"\n{title}")
    print(f"  MAE position     : {pos:.6f}")
    print(f"  MAE orientation  : {ori:.6f}")
    for name, value in zip(TARGET_COLS, per):
        print(f"  MAE {name:5s}: {float(value):.6f}")
    return {
        "mae_position": pos,
        "mae_orientation": ori,
        **{f"mae_{k}": float(v) for k, v in zip(TARGET_COLS, per)},
    }


def collect_preds(
    model: SmallTransformer,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """mode: 'ar' (checkpoint metric) or 'tf' (diagnostic teacher forcing)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            if mode == "ar":
                pred = model.infer(src)
            elif mode == "tf":
                tgt_in = torch.cat([model.bos_token(src.size(0), device), tgt[:, :-1]], dim=1)
                pred = model(src, tgt_in)
            else:
                raise ValueError(mode)
            preds.append(pred.cpu().numpy())
            trues.append(tgt.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


# ---------------------------------------------------------------------------
# 5. Train / test
# ---------------------------------------------------------------------------
def train(
    model: SmallTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    y_scaler: Standardizer,
    cfg: TFConfig,
    device: torch.device,
) -> Tuple[float, List[float], List[float], List[float]]:
    best_mae = float("inf")
    wait = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_ar_maes: List[float] = []
    mse = nn.MSELoss()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_loss = []
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = torch.cat([model.bos_token(src.size(0), device), tgt[:, :-1]], dim=1)
            optimizer.zero_grad(set_to_none=True)
            pred = model(src, tgt_in)
            loss = mse(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss.append(float(loss.item()))

        # Validation: teacher-forced loss (optimizes the train objective)
        # and autoregressive MAE (what we actually care about for FK).
        model.eval()
        val_loss_items = []
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = torch.cat([model.bos_token(src.size(0), device), tgt[:, :-1]], dim=1)
                pred = model(src, tgt_in)
                val_loss_items.append(float(mse(pred, tgt).item()))

        pred_s, true_s = collect_preds(model, val_loader, device, mode="ar")
        abs_err = physical_errors(pred_s, true_s, y_scaler, cfg.circular_ori)
        # Monitor the mean of position and orientation *separately reported*;
        # selection uses position+orientation average only as a single scalar
        # for early stopping, not as a published “mm/rad” score.
        ar_pos = float(abs_err[:, list(POS_IDX)].mean())
        ar_ori = float(abs_err[:, list(ORI_IDX)].mean())
        ar_score = ar_pos + ar_ori

        scheduler.step()

        avg_t = float(np.mean(ep_loss))
        avg_v = float(np.mean(val_loss_items))
        train_losses.append(avg_t)
        val_losses.append(avg_v)
        val_ar_maes.append(ar_score)

        print(
            f"[Epoch {ep:03d}] train_mse={avg_t:.4f}  val_tf_mse={avg_v:.4f}  "
            f"val_AR_pos={ar_pos:.4f}  val_AR_ori={ar_ori:.4f}"
        )

        if ar_score < best_mae:
            best_mae = ar_score
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "seq_len": cfg.seq_len,
                        "d_model": cfg.d_model,
                        "nhead": cfg.nhead,
                        "num_enc": cfg.num_enc,
                        "num_dec": cfg.num_dec,
                        "ff": cfg.ff,
                        "dropout": cfg.dropout,
                    },
                },
                cfg.checkpoint,
            )
            print(f"  → checkpoint {cfg.checkpoint}  score(pos+ori)={best_mae:.4f}")
        else:
            wait += 1
            if wait >= cfg.patience:
                print("Early stopping (no AR val improvement)")
                break

    return best_mae, train_losses, val_losses, val_ar_maes


def evaluate_split(
    model: SmallTransformer,
    loader: DataLoader,
    y_scaler: Standardizer,
    device: torch.device,
    circular_ori: bool,
    split_name: str,
) -> Dict[str, float]:
    pred_ar, true_ar = collect_preds(model, loader, device, mode="ar")
    pred_tf, true_tf = collect_preds(model, loader, device, mode="tf")
    ar = summarize_abs(
        physical_errors(pred_ar, true_ar, y_scaler, circular_ori),
        f"{split_name} AUTOREGRESSIVE (this is the FK number)",
    )
    tfm = summarize_abs(
        physical_errors(pred_tf, true_tf, y_scaler, circular_ori),
        f"{split_name} teacher-forced (diagnostic only; uses GT prefix)",
    )
    return {"ar": ar, "teacher_forced": tfm}


def load_checkpoint(model: SmallTransformer, path: str, device: torch.device) -> None:
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> TFConfig:
    p = argparse.ArgumentParser(description="Transformer 6-DoF FK regressor")
    p.add_argument("--csv", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-circular-ori", action="store_true")
    args = p.parse_args(argv)
    return TFConfig(
        csv_path=resolve_csv_path(args.csv),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        circular_ori=not args.no_circular_ori,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import matplotlib.pyplot as plt

    cfg = parse_args(argv)
    set_seeds(cfg.seed)
    device = get_device()
    print("DEVICE:", device)

    x, y = load_dataset(cfg.csv_path)
    x_train, x_val, x_test, y_train, y_val, y_test = split_80_10_10(x, y, cfg.seed)
    print(f"Split train/val/test = {len(x_train)}/{len(x_val)}/{len(x_test)}")

    x_scaler = Standardizer().fit(x_train, "X")
    y_scaler = Standardizer().fit(y_train, "Y")
    dump_scalers(cfg.scaler_path, x_scaler, y_scaler, cfg)

    x_train_s, x_val_s, x_test_s = (
        x_scaler.transform(x_train),
        x_scaler.transform(x_val),
        x_scaler.transform(x_test),
    )
    y_train_s, y_val_s, y_test_s = (
        y_scaler.transform(y_train),
        y_scaler.transform(y_val),
        y_scaler.transform(y_test),
    )

    train_loader = DataLoader(
        SixDoFDataset(x_train_s, y_train_s),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(SixDoFDataset(x_val_s, y_val_s), batch_size=cfg.batch_size)
    test_loader = DataLoader(SixDoFDataset(x_test_s, y_test_s), batch_size=cfg.batch_size)

    model = SmallTransformer(
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_enc=cfg.num_enc,
        num_dec=cfg.num_dec,
        ff=cfg.ff,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train(model, train_loader, val_loader, optimizer, scheduler, y_scaler, cfg, device)

    load_checkpoint(model, cfg.checkpoint, device)
    evaluate_split(model, test_loader, y_scaler, device, cfg.circular_ori, "TEST")

    # Recompute val AR curve already stored during train; re-plot that list
    # is not returned to main in this version — plot last-run file if present
    # by re-evaluating is wasteful. The train() printout is the record.
    # Save a one-shot test-bar is more honest than a fake curve.
    pred_ar, true_ar = collect_preds(model, test_loader, device, mode="ar")
    abs_err = physical_errors(pred_ar, true_ar, y_scaler, cfg.circular_ori)
    plt.figure(figsize=(8, 4))
    plt.bar(TARGET_COLS, abs_err.mean(axis=0))
    plt.ylabel("MAE (physical units)")
    plt.title("Test autoregressive MAE by axis")
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=150)
    plt.close()
    print(f"Wrote {cfg.plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
