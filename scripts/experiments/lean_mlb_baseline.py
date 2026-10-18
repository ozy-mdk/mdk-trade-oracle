#!/usr/bin/env python
"""Step 3.1 — lean MLB-flow baseline, per stock vs pooled (isolated experiment, read-only on the database).

Fixed design (chosen on the 2023-2024 discovery ICs, never tuned on the holdout):
    features   MLB buy share, MLB net-flow intensity, total institutional net share, MLB late (W4+W5) net share,
               each z-scored per stock against its own trailing 252 sessions (knowable at the close of d)
    target     adj close d -> close d+1 (the published T+1 target; graded on exactly this)
    model      Ridge(alpha=10), refit at every origin on rows whose label is realized by the origin close,
               trailing 504 sessions (min 126); "pooled" fits one model on all six stocks with the same date rule
    calls      all-days sign; selective = score beyond the trailing 80/20 (or 90/10) percentile of the model's own
               past scores for that stock
Grading: strict hit (moved > 0.25% in the called direction) vs random (shuffled predictions) for all-days, and vs
the stock's base rate of that move for selective calls; per-stock edges; block-bootstrap CIs; discovery vs holdout.

Usage:
    .venv/bin/python scripts/experiments/lean_mlb_baseline.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from feature_diagnostics import window_flow_features  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
OUT_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "lean_mlb"
RAW_FEATURES = ["feat_mlb_buy_share_today", "feat_mlb_turnover_intensity", "feat_total_inst_net_share_today", "mlb_late_share"]
Z_WINDOW = 252
TRAIN_MAX, TRAIN_MIN = 504, 126
ALPHA = 10.0
DB = 0.25
EVAL_START, DISCOVERY_END = "2023-01-01", "2024-12-31"
SCORE_WINDOW = 252  # trailing own-score history for selective thresholds
N_BOOT, BLOCK, N_PERM = 1000, 10, 300


def load_panel() -> pd.DataFrame:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price, extract_3pillar_time_series

    db = PostgresManager(read_only=True)
    frames = []
    for sym in FOCUS:
        df = extract_3pillar_time_series(db, sym, lookback_days=5000).sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        wf = window_flow_features(db, sym)
        wf["trade_date"] = pd.to_datetime(wf["trade_date"])
        df = df.merge(wf, on="trade_date", how="left")
        tt = df["total_turnover_tl"].replace(0, np.nan)
        df["mlb_late_share"] = ((df["cand_mlb_w4_flow_tl"].fillna(0) + df["cand_mlb_w5_flow_tl"].fillna(0)) / tt * 100).clip(-100, 100)
        c = adjusted_price(df, "close")
        df["y"] = (c.shift(-1) / c - 1) * 100
        df["label_realized_date"] = df["trade_date"].shift(-1)  # the close that realizes the label of row d
        for f in RAW_FEATURES:
            x = df[f].astype(float)
            mu = x.rolling(Z_WINDOW, min_periods=126).mean()
            sd = x.rolling(Z_WINDOW, min_periods=126).std().replace(0, np.nan)
            df[f"z_{f}"] = ((x - mu) / sd).clip(-5, 5)
        df["symbol"] = sym
        frames.append(df[["symbol", "trade_date", "label_realized_date", "y", *[f"z_{f}" for f in RAW_FEATURES]]])
    return pd.concat(frames, ignore_index=True)


def walk_forward(panel: pd.DataFrame, pooled: bool) -> pd.DataFrame:
    zc = [f"z_{f}" for f in RAW_FEATURES]
    p = panel.dropna(subset=zc).copy()
    origins = sorted(p.loc[p["trade_date"] >= EVAL_START, "trade_date"].unique())
    out = []
    groups = [None] if pooled else FOCUS
    for g in groups:
        sub = p if pooled else p[p["symbol"] == g]
        for origin in origins:
            today = sub[sub["trade_date"] == origin]
            if today.empty:
                continue
            hist = sub[(sub["label_realized_date"] <= origin) & sub["y"].notna()]
            # trailing TRAIN_MAX sessions per stock
            hist = hist.groupby("symbol", group_keys=False).tail(TRAIN_MAX)
            if hist.groupby("symbol").size().min() < TRAIN_MIN:
                continue
            assert hist["label_realized_date"].max() <= origin
            X, y = hist[zc].to_numpy(), hist["y"].to_numpy()
            if pooled:  # stock-specific intercepts: demean y per stock
                y = y - hist.groupby("symbol")["y"].transform("mean").to_numpy()
            m = Ridge(alpha=ALPHA).fit(X, y)
            for _, r in today.iterrows():
                out.append({"symbol": r["symbol"], "trade_date": r["trade_date"], "score": float(m.predict(r[zc].to_numpy(dtype=float)[None, :])[0]),
                            "y": r["y"]})
    res = pd.DataFrame(out).dropna(subset=["y"]).sort_values(["symbol", "trade_date"])
    res["variant"] = "pooled" if pooled else "per_stock"
    # selective thresholds from the model's own past scores (strictly before d)
    for q in (0.8, 0.9):
        hi = res.groupby("symbol")["score"].transform(lambda s: s.shift(1).rolling(SCORE_WINDOW, min_periods=60).quantile(q))
        lo = res.groupby("symbol")["score"].transform(lambda s: s.shift(1).rolling(SCORE_WINDOW, min_periods=60).quantile(1 - q))
        res[f"call_q{int(q * 100)}"] = np.where(res["score"] > hi, 1, np.where(res["score"] < lo, -1, 0))
    return res


def block_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(v)
    if n < 2 * BLOCK:
        return (np.nan, np.nan)
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, int(np.ceil(n / BLOCK))))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]
    return tuple(np.percentile(v[idx].mean(axis=1), [2.5, 97.5]))


def evaluate(res: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    res = res.assign(period=np.where(res["trade_date"] <= pd.Timestamp(DISCOVERY_END), "discovery", "holdout"))
    rows = []
    for (variant, period), g in res.groupby(["variant", "period"]):
        for sym, gs in list(g.groupby("symbol")) + [("ALL6", g)]:
            y, s = gs["y"].to_numpy(), gs["score"].to_numpy()
            sign = np.sign(s)
            strict_all = (sign * y > DB).astype(float)
            perm = np.mean([((np.sign(rng.permutation(s)) * y) > DB).mean() for _ in range(N_PERM)])
            lo, hi = block_ci(strict_all * 100, rng)
            row = {"variant": variant, "period": period, "symbol": sym, "n": len(gs),
                   "all_strict_pct": strict_all.mean() * 100, "all_ci_lo": lo, "all_ci_hi": hi, "all_random_pct": perm * 100,
                   "all_sign_acc_pct": (np.sign(s) == np.sign(y)).mean() * 100}
            p_up = gs.groupby("symbol")["y"].transform(lambda v: (v > DB).mean()).to_numpy()
            p_dn = gs.groupby("symbol")["y"].transform(lambda v: (v < -DB).mean()).to_numpy()
            for q in ("q80", "q90"):
                call = gs[f"call_{q}"].to_numpy()
                k = call != 0
                hit = (call[k] * y[k] > DB).astype(float)
                base = np.where(call[k] > 0, p_up[k], p_dn[k])
                clo, chi = block_ci(hit * 100 - base * 100, rng) if k.sum() >= 2 * BLOCK else (np.nan, np.nan)
                row.update({f"{q}_coverage_pct": k.mean() * 100, f"{q}_n": int(k.sum()), f"{q}_hit_pct": hit.mean() * 100 if k.any() else np.nan,
                            f"{q}_base_pct": base.mean() * 100 if k.any() else np.nan, f"{q}_edge_pp": (hit.mean() - base.mean()) * 100 if k.any() else np.nan,
                            f"{q}_edge_ci_lo": clo, f"{q}_edge_ci_hi": chi})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    res = pd.concat([walk_forward(panel, pooled=False), walk_forward(panel, pooled=True)], ignore_index=True)
    res.to_parquet(OUT_DIR / "predictions.parquet")
    ev = evaluate(res)
    ev.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    for period in ("discovery", "holdout"):
        e = ev[ev["period"] == period].set_index(["variant", "symbol"])
        print(f"\n=== {period.upper()} — all-days calls (strict %, random = shuffled scores) ===")
        print(e[["n", "all_strict_pct", "all_ci_lo", "all_ci_hi", "all_random_pct", "all_sign_acc_pct"]].round(1).to_string())
        for q in ("q80", "q90"):
            print(f"\n--- {period} — selective calls {q} (hit vs the stock's base rate of that move) ---")
            print(e[[f"{q}_coverage_pct", f"{q}_n", f"{q}_hit_pct", f"{q}_base_pct", f"{q}_edge_pp", f"{q}_edge_ci_lo", f"{q}_edge_ci_hi"]].round(1).to_string())
    print(f"\nwrote {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
