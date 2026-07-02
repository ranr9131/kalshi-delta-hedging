"""
Compute Sharpe for NN14 (best fair-value model) using the same walkforward
simulation as walkforward_v2p.py, but with per-window logging.

Outputs:
  - sharpe_eval_trades.csv  (one row per acted window: ts, pnl, wagered)
  - prints daily/annualized Sharpe under multiple capital assumptions
"""

import os, sys, csv, math, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TSWinPredictor

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "dataset_v2p.npz")
OUT_CSV   = os.path.join(ROOT, "sharpe_eval_trades.csv")
SEED      = 0

# Match walkforward_v2p.py exactly
BATCH_SIZE = 128
LR, WD     = 1e-3, 1e-4
EPOCHS     = 40
PATIENCE   = 8
MIN_DEC, MAX_DEC = 4, 13
D_MODEL, N_HEADS, N_LAYERS, DIM_FF, DROPOUT = 32, 4, 2, 64, 0.1
MINUTES_LATE = list(range(10, 14))

BASE_STAKE = 100.0
MIN_BET    = 5.0
SLIP_C     = 4
MIN_EDGE_C = 10
RH_TRIGGER = 10.0
MAX_HEDGE_F = 0.80
MAX_FILL   = 0.97
MAX_LEGS   = 2
FEE_KEEP   = 0.93
SIG_K, SIG_C, SIG_MAX = 20.0, 0.10, 3.0
MISPR_K, MISPR_MAX    = 8.0, 2.0


def f_btc(p):  return SIG_MAX / (1.0 + math.exp(-SIG_K * (p - SIG_C)))
def g_misp(m): return MISPR_MAX / (1.0 + math.exp(-MISPR_K * m))
def time_decay(minute):
    if minute < 7:  return 0.4
    if minute < 10: return 0.8
    return 1.2


class TruncDataset(Dataset):
    def __init__(self, X, mask, y):
        self.X, self.mask, self.y = X, mask, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        X = self.X[idx].copy(); m = self.mask[idx].copy()
        dec = random.randint(MIN_DEC, MAX_DEC)
        X[dec+1:, :] = 0.0; m[dec+1:] = False
        return (torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(m),
                torch.tensor(self.y[idx], dtype=torch.float32))


def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def train_nn14(X_train, mask_train, y_train):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = pick_device()
    n_features = 14
    X_train = X_train[:, :, :14]
    Xn = X_train.copy(); Xn[:, MIN_DEC+1:, :] = 0.0
    flat = Xn.reshape(-1, n_features)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32) + 1e-6
    Xnorm = ((X_train - mean) / std).astype(np.float32)
    ds = TruncDataset(Xnorm, mask_train, y_train)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = TSWinPredictor(n_features=n_features, d_model=D_MODEL, n_heads=N_HEADS,
                           n_layers=N_LAYERS, dim_feedforward=DIM_FF, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()
    best_loss = float("inf"); best_state = None; patience = 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0; n = 0
        for X, mk, y in loader:
            X, mk, y = X.to(device), mk.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X, mk)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y); n += len(y)
        epoch_loss = total / n
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE: break
    model.load_state_dict(best_state)
    return model.to("cpu").eval(), mean, std


def predict_minutes(model, mean, std, Xte, mte):
    Xte = Xte[:, :, :14]
    Xn = ((Xte - mean) / std).astype(np.float32)
    p = np.zeros((len(Xte), 15), dtype=np.float32)
    with torch.no_grad():
        for m in MINUTES_LATE:
            Xt = Xn.copy(); mt = mte.copy()
            Xt[:, m+1:, :] = 0.0; mt[:, m+1:] = False
            logits = model(torch.from_numpy(Xt), torch.from_numpy(mt))
            p[:, m] = torch.sigmoid(logits).numpy()
    return p


def simulate_window(X, mask, y, p_yes_minutes):
    resolved_yes = (y == 1.0)
    yes_bets = []; no_bets = []
    yes_exp = no_exp = 0.0
    yes_c = no_c = 0.0
    legs = 0
    for m in MINUTES_LATE:
        if legs >= MAX_LEGS: break
        if not mask[m]: continue
        ret = float(X[m, 0]); kal = float(X[m, 4])
        if not (0.01 < kal < 0.99) or ret == 0: continue
        up = ret > 0; pct = abs(ret) * 100
        slip = SLIP_C / 100.0
        yes_fill = min(MAX_FILL, kal + slip)
        no_fill  = min(MAX_FILL, (1 - kal) + slip)
        fair_y = float(p_yes_minutes[m])
        fair = fair_y if up else (1 - fair_y)
        mispr = fair - (yes_fill if up else no_fill)
        if mispr * 100 < MIN_EDGE_C:
            target = 0.0
        else:
            target = BASE_STAKE * f_btc(pct) * g_misp(mispr) * time_decay(m)
        if up:
            gap = max(0.0, target - yes_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and yes_fill < MAX_FILL:
                yes_bets.append((gap, yes_fill))
                yes_exp += gap; yes_c += gap / yes_fill; legs += 1
        else:
            gap = max(0.0, target - no_exp)
            if gap >= MIN_BET and legs < MAX_LEGS and no_fill < MAX_FILL:
                no_bets.append((gap, no_fill))
                no_exp += gap; no_c += gap / no_fill; legs += 1
        if m >= 10 and legs < MAX_LEGS:
            if up:
                if no_exp >= RH_TRIGGER and no_c > 0 and yes_fill <= MAX_HEDGE_F:
                    hedge = no_c * yes_fill
                    if hedge >= MIN_BET:
                        yes_bets.append((hedge, yes_fill))
                        yes_exp += hedge; yes_c += hedge / yes_fill; legs += 1
            else:
                if yes_exp >= RH_TRIGGER and yes_c > 0 and no_fill <= MAX_HEDGE_F:
                    hedge = yes_c * no_fill
                    if hedge >= MIN_BET:
                        no_bets.append((hedge, no_fill))
                        no_exp += hedge; no_c += hedge / no_fill; legs += 1
    pnl_yes = sum((1-fp)*(s/fp)*FEE_KEEP if resolved_yes else -s for s, fp in yes_bets)
    pnl_no  = sum((1-fp)*(s/fp)*FEE_KEEP if not resolved_yes else -s for s, fp in no_bets)
    return pnl_yes + pnl_no, yes_exp + no_exp


def main():
    z = np.load(DATA_PATH, allow_pickle=True)
    X, mask, y, ts = z["X"], z["mask"], z["y"], z["ts"]
    order = np.argsort(ts); X, mask, y, ts = X[order], mask[order], y[order], ts[order]
    n = len(y)
    print(f"Dataset: {n} windows | features={X.shape[-1]}")
    print(f"Range: {datetime.fromtimestamp(int(ts[0]), tz=timezone.utc).date()} "
          f"-> {datetime.fromtimestamp(int(ts[-1]), tz=timezone.utc).date()}\n")

    fold_starts = [int(n * f) for f in [0.40, 0.55, 0.70, 0.85]]
    fold_size = int(n * 0.15)
    folds = []
    for i, s in enumerate(fold_starts):
        e = min(s + fold_size, n)
        if s <= 100 or e - s < 100: continue
        folds.append((f"F{i+1}", s, e))

    rows = []
    for label, s, e in folds:
        d0 = datetime.fromtimestamp(int(ts[s]), tz=timezone.utc).date()
        d1 = datetime.fromtimestamp(int(ts[e-1]), tz=timezone.utc).date()
        print(f"Fold {label}: train n={s}, test {d0} -> {d1} (n={e-s})", flush=True)
        Xte, mte, yte = X[s:e], mask[s:e], y[s:e]
        tste = ts[s:e]

        t0 = time.time()
        model, mean, std = train_nn14(X[:s], mask[:s], y[:s])
        p14 = predict_minutes(model, mean, std, Xte, mte)
        print(f"  trained NN14 in {time.time()-t0:.0f}s", flush=True)

        acted = wins = 0; total_pnl = total_wag = 0.0
        for i in range(len(yte)):
            pnl, wag = simulate_window(Xte[i], mte[i], yte[i], p14[i])
            if wag > 0:
                acted += 1
                if pnl > 0: wins += 1
                rows.append({"fold": label, "ts": int(tste[i]),
                             "pnl": pnl, "wagered": wag})
            total_pnl += pnl; total_wag += wag
        roi = total_pnl/total_wag*100 if total_wag else 0
        print(f"  acted={acted} win%={wins/acted*100:.1f} "
              f"wag=${total_wag:.0f} pnl=${total_pnl:+.0f} ROI={roi:+.2f}%\n",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} trades -> {OUT_CSV}\n")

    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.date
    daily = df.groupby("date").agg(pnl=("pnl", "sum"),
                                   wagered=("wagered", "sum"),
                                   trades=("pnl", "size")).reset_index()
    print(f"Trading days: {len(daily)}")
    print(f"Total PnL: ${df['pnl'].sum():+,.0f}")
    print(f"Total wagered: ${df['wagered'].sum():,.0f}")
    print(f"ROI on wagered: {df['pnl'].sum()/df['wagered'].sum()*100:+.2f}%")
    print(f"Trades: {len(df)}")
    print(f"Win rate: {(df['pnl']>0).mean()*100:.1f}%")
    print(f"\nDaily PnL: mean=${daily['pnl'].mean():.2f}, "
          f"std=${daily['pnl'].std():.2f}, "
          f"min=${daily['pnl'].min():.0f}, max=${daily['pnl'].max():.0f}")

    print("\n--- Sharpe (annualized, sqrt(365), zero risk-free) ---")
    for label, capital in [
        ("$10k account", 10_000),
        ("$25k account", 25_000),
        ("$50k account", 50_000),
        ("$100k account", 100_000),
    ]:
        daily_ret = daily["pnl"] / capital
        sharpe = daily_ret.mean() / daily_ret.std() * math.sqrt(365)
        ann_ret = daily_ret.mean() * 365 * 100
        ann_vol = daily_ret.std() * math.sqrt(365) * 100
        print(f"  {label}: ann_ret={ann_ret:+.1f}%  ann_vol={ann_vol:.1f}%  Sharpe={sharpe:.2f}")

    # Capital-invariant Sharpe based on per-trade ROI
    print("\n--- Per-trade ROI Sharpe (independent of capital) ---")
    df["roi"] = df["pnl"] / df["wagered"]
    trades_per_year = len(df) / (daily["date"].nunique() / 365.0)
    sharpe_t = df["roi"].mean() / df["roi"].std() * math.sqrt(trades_per_year)
    print(f"  trades/year (annualized): {trades_per_year:.0f}")
    print(f"  mean trade ROI: {df['roi'].mean()*100:+.2f}%  std: {df['roi'].std()*100:.2f}%")
    print(f"  Sharpe (trade-level, annualized): {sharpe_t:.2f}")


if __name__ == "__main__":
    main()
