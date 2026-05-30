"""
Evaluate the trained model on the held-out val set and compare against the 2D
table baseline. Honest metrics: log-loss, Brier score, AUC, calibration.

The 2D table baseline predicts P(direction-side wins) at minute T+10 conditioned
on (minute, |move| bucket). We convert that to P(YES wins) using the observed
direction at T+10.

Usage:
    python3 nn/evaluate.py
"""

import os
import csv
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TSWinPredictor

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "dataset.npz")
CKPT_PATH  = os.path.join(ROOT, "checkpoints", "best.pt")
TABLE_PATH = os.path.join(os.path.dirname(ROOT), "data", "logs", "minute_analysis_2d.csv")

# Must match analyze_minutes_2d.py / simulate_dh.py buckets.
_2D_BUCKETS = [
    (0.000, 0.05), (0.050, 0.10), (0.100, 0.20),
    (0.200, 0.50), (0.500, float("inf")),
]
_2D_BUCKET_LABELS = [
    "0.00-0.05%", "0.05-0.10%", "0.10-0.20%", "0.20-0.50%", "0.50%+",
]
VAL_FRAC = 0.20
DECISION_MINUTE = 10  # must match train.py


def load_2d_table():
    table = {}
    label_to_idx = {lbl: i for i, lbl in enumerate(_2D_BUCKET_LABELS)}
    with open(TABLE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            m = int(row["minute"]); bi = label_to_idx.get(row["bucket"])
            if bi is None: continue
            table[(m, bi)] = float(row["win_rate"])
    return table


def bucket_for_pct(pct):
    for i, (lo, hi) in enumerate(_2D_BUCKETS):
        if lo <= pct < hi:
            return i
    return len(_2D_BUCKETS) - 1


def calibration_bins(p, y, nbins=10):
    bin_edges = np.linspace(0, 1, nbins + 1)
    bins = []
    for i in range(nbins):
        in_bin = (p >= bin_edges[i]) & (p < bin_edges[i+1] if i < nbins-1
                                        else p <= bin_edges[i+1])
        if in_bin.sum() == 0: continue
        bins.append({
            "range":  f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
            "n":      int(in_bin.sum()),
            "p_mean": float(p[in_bin].mean()),
            "y_mean": float(y[in_bin].mean()),
        })
    return bins


def main():
    # ---- Load data and replay time-based split ----
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts)
    X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n_val = int(len(y) * VAL_FRAC); n_train = len(y) - n_val
    Xva, mva, yva, tsva = X[n_train:], mask[n_train:], y[n_train:], ts[n_train:]
    # Truncate at decision minute (same as training)
    Xva  = Xva.copy();  mva = mva.copy()
    Xva[:, DECISION_MINUTE+1:, :] = 0.0
    mva[:, DECISION_MINUTE+1:]    = False
    print(f"Val set: {len(yva)} windows  (truncated at T+{DECISION_MINUTE})")
    print(f"  YES win rate (base): {yva.mean():.4f}")

    # ---- NN predictions ----
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    std  = np.array(ckpt["feature_std"],  dtype=np.float32)
    Xn = ((Xva - mean) / std).astype(np.float32)
    model = TSWinPredictor(n_features=ckpt["n_features"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xn), torch.from_numpy(mva))
        p_nn = torch.sigmoid(logits).cpu().numpy()

    # ---- 2D table predictions ----
    # We score the 2D table the same way it's used in the trader: at T+10,
    # look up win rate for (minute=10, magnitude bucket). Convert to P(YES).
    # Feature index 0 is btc_return_from_t0 (raw, not standardized in Xva).
    table = load_2d_table()
    p_table = np.full(len(yva), 0.5, dtype=np.float32)
    for i in range(len(yva)):
        if not mva[i, 10]:
            continue  # no data at T+10, fall back to base rate
        ret_t10 = Xva[i, 10, 0]   # raw cumulative return
        pct = abs(ret_t10) * 100
        bi  = bucket_for_pct(pct)
        win_rate = table.get((10, bi), yva.mean())   # P(direction wins)
        direction_up = ret_t10 > 0
        # P(YES) = P(direction wins) if up else (1 - that)
        p_table[i] = win_rate if direction_up else (1.0 - win_rate)

    # ---- Metrics ----
    def report(label, p):
        # Clip to avoid log(0)
        p_c = np.clip(p, 1e-6, 1-1e-6)
        ll  = log_loss(yva, p_c)
        br  = brier_score_loss(yva, p_c)
        try:
            au = roc_auc_score(yva, p_c)
        except ValueError:
            au = float("nan")
        acc = ((p_c >= 0.5).astype(np.float32) == yva).mean()
        print(f"  {label:<10s}  log-loss={ll:.4f}  brier={br:.4f}  "
              f"AUC={au:.4f}  acc={acc:.4f}")
        return ll, br, au, acc

    print(f"\n── Metrics on held-out val set ──────────────────────────────")
    print(f"  {'model':<10s}  {'log-loss':>9s}  {'brier':>9s}  {'AUC':>5s}  {'acc':>5s}")
    base_p = np.full_like(p_nn, yva.mean())
    report("base rate", base_p)
    nn_metrics    = report("NN", p_nn)
    table_metrics = report("2D table", p_table)

    print(f"\n── Calibration (NN) ──")
    for b in calibration_bins(p_nn, yva):
        print(f"  p in {b['range']}  n={b['n']:>4}  "
              f"predicted={b['p_mean']:.3f}  observed={b['y_mean']:.3f}")
    print(f"\n── Calibration (2D table) ──")
    for b in calibration_bins(p_table, yva):
        print(f"  p in {b['range']}  n={b['n']:>4}  "
              f"predicted={b['p_mean']:.3f}  observed={b['y_mean']:.3f}")

    print(f"\nSummary:")
    print(f"  NN beats 2D on log-loss?  {nn_metrics[0] < table_metrics[0]}")
    print(f"  NN beats 2D on AUC?       {nn_metrics[2] > table_metrics[2]}")


if __name__ == "__main__":
    main()
