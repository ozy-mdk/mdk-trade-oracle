#!/usr/bin/env python
"""Does next-day direction concentrate on extreme institutional-flow days? (isolated, read-only)

Per stock: on session d, call UP if the flow feature is above its trailing-252-session q-quantile (known at the close
of d, thresholds shifted one session), DOWN if below the (1-q)-quantile; otherwise no call. Graded on the model target
(adj close d -> close d+1) and, separately, on the next session open -> close. Strict hit = moved beyond 0.25% in the
called direction. Baseline = unconditional frequency of that move direction over the same sessions.
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
FEATURES = ["feat_mlb_turnover_intensity", "feat_total_inst_net_share_today", "feat_mlb_buy_share_today", "feat_mlb_w5_share"]
DB = 0.25


def main() -> None:
    from mdk_trading_oracle.core.db import PostgresManager
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import adjusted_price, extract_3pillar_time_series

    db = PostgresManager(read_only=True)
    rows = []
    for sym in FOCUS:
        df = extract_3pillar_time_series(db, sym, lookback_days=5000).sort_values("trade_date").reset_index(drop=True)
        c, o = adjusted_price(df, "close"), adjusted_price(df, "open")
        targets = {"close_to_close": (c.shift(-1) / c - 1) * 100, "next_open_to_close": (c.shift(-1) / o.shift(-1) - 1) * 100}
        date = pd.to_datetime(df["trade_date"])
        for f in FEATURES:
            x = df[f]
            for q in (0.8, 0.9):
                hi = x.rolling(252, min_periods=126).quantile(q).shift(1)
                lo = x.rolling(252, min_periods=126).quantile(1 - q).shift(1)
                call = np.where(x > hi, 1, np.where(x < lo, -1, 0))
                for tname, y in targets.items():
                    for i in np.where((call != 0) & (date >= "2023-01-01") & y.notna())[0]:
                        rows.append({"symbol": sym, "feature": f, "q": q, "target": tname, "period": "discovery" if date[i] <= pd.Timestamp("2024-12-31") else "holdout",
                                     "call": int(call[i]), "hit": bool(y[i] * call[i] > DB), "y": float(y[i])})
                    base = y[(date >= "2023-01-01") & y.notna()]
                    rows.append({"symbol": sym, "feature": "_base", "q": q, "target": tname, "period": "all",
                                 "call": 0, "hit": np.nan, "y": np.nan, "p_up": float((base > DB).mean()), "p_dn": float((base < -DB).mean())})
    r = pd.DataFrame(rows)
    base = r[r["feature"] == "_base"].groupby(["symbol", "target"])[["p_up", "p_dn"]].mean()
    calls = r[r["feature"] != "_base"].drop(columns=["p_up", "p_dn"]).join(base, on=["symbol", "target"])
    calls["base_hit"] = np.where(calls["call"] > 0, calls["p_up"], calls["p_dn"])
    out = calls.groupby(["target", "feature", "q", "period"]).agg(n=("hit", "size"), hit_pct=("hit", "mean"), base_pct=("base_hit", "mean"),
                                                                  stocks_beating_base=("symbol", lambda s: 0)).reset_index()
    per_stock = calls.groupby(["target", "feature", "q", "period", "symbol"]).apply(lambda g: g["hit"].mean() - g["base_hit"].mean()).rename("edge").reset_index()
    out["stocks_beating_base"] = out.apply(lambda r_: int((per_stock[(per_stock.target == r_.target) & (per_stock.feature == r_.feature) & (per_stock.q == r_.q) & (per_stock.period == r_.period)]["edge"] > 0).sum()), axis=1)
    out["hit_pct"] *= 100
    out["base_pct"] *= 100
    out["edge_pp"] = out["hit_pct"] - out["base_pct"]
    out["se_pp"] = np.sqrt(out["base_pct"] / 100 * (1 - out["base_pct"] / 100) / out["n"]) * 100
    pd.set_option("display.width", 200)
    print(out.round(1).to_string(index=False))
    out.to_csv(os.path.expanduser("~/data/mdk_oracle/experiments/extreme_flow_check.csv"), index=False)


if __name__ == "__main__":
    main()
