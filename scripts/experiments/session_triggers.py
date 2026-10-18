#!/usr/bin/env python
"""Operational form of the 11:30 session signal under end-of-day data delivery (isolated experiment, read-only on DB).

Broker-level data arrives only after the close, so a forecast for session N may use data through the close of N-1.
The session signal score = z(yesterday MLB intensity) + z(morning move) needs one live input, the morning move
(open -> 11:30 of N), which is public price information the trader sees on screen. Everything else (yesterday's MLB
intensity, trailing z-score statistics, call thresholds) is known at the N-1 close. So after the close we publish
conditional triggers for N:
    at 11:30: BUY if price >= open * (1 + up_trigger_pct / 100); SELL if price <= open * (1 + down_trigger_pct / 100)
which reproduce the backtested calls exactly (checked by --check).

Modes:
    --check     equivalence of trigger calls vs backtested calls; end-of-day-only variant (no live input)
    --forward   write triggers for the next session (frozen: never overwritten) and grade any earlier trigger
                files whose session data has arrived -> ~/data/mdk_oracle/experiments/intraday_flow/forward_log/

Variants published in the forward log (frozen 2026-10-18):
    session_v1        z(yesterday MLB intensity) + z(morning move), trailing 252, call beyond 80/20 of past scores
    session_bank_wl   watchlist: session_v1 + z(yesterday bank/BIST30 MLB flow)  (rejected on discovery, re-tested live)
    bank_pair_wl      watchlist: long highest / short lowest relative MLB-flow bank of 7, next open -> close
                      (strong in discovery, faded in holdout; re-tested live)
    eod_v1            no live input: z(yesterday MLB intensity) beyond its trailing 80/20; trade open -> close
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intraday_flow_experiment import DB, DISCOVERY_END, EVAL_START, OUT_DIR, build_panel, engineer  # noqa: E402

WINDOW, MIN_P, Q = 252, 126, 0.8
Z_CLIP = 5.0
FORWARD_DIR = OUT_DIR / "forward_log"
CONFIG_VERSION = "frozen-2026-10-18: window=252, q=0.8, z_clip=5"


def trailing(g: pd.core.groupby.SeriesGroupBy, how: str, q: float | None = None) -> pd.Series:
    r = lambda x: x.shift(1).rolling(WINDOW, min_periods=MIN_P)  # noqa: E731  (strictly before the row)
    if how == "mean":
        return g.transform(lambda x: r(x).mean())
    if how == "std":
        return g.transform(lambda x: r(x).std())
    return g.transform(lambda x: r(x).quantile(q))


def add_state(p: pd.DataFrame, extra: str | None = None) -> pd.DataFrame:
    """Trailing statistics, score, thresholds and trigger levels per row (row = session N)."""
    p = p.sort_values(["symbol", "trade_date"]).copy()
    feats = ["mlb_net_yday_intensity", "morning_ret"] + ([extra] if extra else [])
    for f in feats:
        g = p.groupby("symbol")[f]
        p[f"mu_{f}"], p[f"sd_{f}"] = trailing(g, "mean"), trailing(g, "std")
        p[f"z_{f}"] = ((p[f] - p[f"mu_{f}"]) / p[f"sd_{f}"]).clip(-Z_CLIP, Z_CLIP)
    known = p["z_mlb_net_yday_intensity"] + (p[f"z_{extra}"] if extra else 0.0)  # known at the previous close
    p["known_part"] = known
    p["score"] = known + p["z_morning_ret"]
    gs = p.groupby("symbol")["score"]
    p["hi"], p["lo"] = trailing(gs, "q", Q), trailing(gs, "q", 1 - Q)
    p["call"] = np.where(p["score"] > p["hi"], 1, np.where(p["score"] < p["lo"], -1, 0))
    # trigger levels on the morning move: score > hi  <=>  z_morning > hi - known
    up_z, dn_z = p["hi"] - known, p["lo"] - known
    p["up_trigger_pct"] = np.where(up_z < Z_CLIP, p["mu_morning_ret"] + p["sd_morning_ret"] * up_z, np.nan)
    p["down_trigger_pct"] = np.where(dn_z > -Z_CLIP, p["mu_morning_ret"] + p["sd_morning_ret"] * dn_z, np.nan)
    # end-of-day-only variant: yesterday's MLB intensity alone
    gy = p.groupby("symbol")["z_mlb_net_yday_intensity"]
    p["eod_hi"], p["eod_lo"] = trailing(gy, "q", Q), trailing(gy, "q", 1 - Q)
    zy = p["z_mlb_net_yday_intensity"]
    p["eod_call"] = np.where(zy > p["eod_hi"], 1, np.where(zy < p["eod_lo"], -1, 0))
    return p


def trigger_call(morning_ret: pd.Series, up: pd.Series, dn: pd.Series) -> np.ndarray:
    return np.where(morning_ret > up, 1, np.where(morning_ret < dn, -1, 0))


def summarize(call: pd.Series, y: pd.Series, symbol: pd.Series) -> dict[str, float]:
    k = (call != 0) & y.notna()
    sign_ok = np.sign(call[k]) == np.sign(y[k])
    pnl = call[k] * y[k]
    base = np.where(call[k] > 0, (y > DB).mean(), (y < -DB).mean())
    per_stock = pd.Series(sign_ok.to_numpy(), index=symbol[k].to_numpy()).groupby(level=0).mean()
    return {"coverage_pct": k.sum() / y.notna().sum() * 100, "n_calls": int(k.sum()), "sign_acc_pct": sign_ok.mean() * 100,
            "edge_pp": ((pnl > DB).mean() - base.mean()) * 100, "mean_ret_pct": pnl.mean(),
            "t_stat": pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl)) if len(pnl) > 2 else np.nan, "stocks_acc_gt_50": int((per_stock > 0.5).sum())}


def with_context(p: pd.DataFrame) -> pd.DataFrame:
    from session_signal_extensions import daily_context

    ctx = daily_context()[["symbol", "trade_date", "group_mlb_yday", "group_mlb_today"]]
    p = p.merge(ctx, on=["symbol", "trade_date"], how="left").sort_values(["symbol", "trade_date"])
    # next-session placeholder rows are not in silver yet: their "yesterday" is the last close's group flow
    prev_today = p.groupby("symbol")["group_mlb_today"].shift(1)
    p["group_mlb_yday"] = p["group_mlb_yday"].fillna(prev_today)
    return p.drop(columns="group_mlb_today")


def check() -> None:
    pd.set_option("display.width", 250)
    p, _ = engineer(pd.read_parquet(OUT_DIR / "tick_panel.parquet"))
    p = add_state(p.dropna(subset=["rest_1130"]))
    from session_signal_robustness import signal

    ref = signal(p[[c for c in p.columns if not c.startswith(("mu_", "sd_", "z_", "hi", "lo", "call", "score", "known", "up_", "down_", "eod_"))]])
    m = p.merge(ref[["symbol", "trade_date", "call"]].rename(columns={"call": "ref_call"}), on=["symbol", "trade_date"])
    m["trig_call"] = trigger_call(m["morning_ret"], m["up_trigger_pct"], m["down_trigger_pct"])
    print(f"trigger calls == backtested calls on {(m['trig_call'] == m['ref_call']).mean() * 100:.2f}% of {len(m)} sessions "
          f"(differences only where a z-score hits the +/-{Z_CLIP} clip: {int((m['trig_call'] != m['ref_call']).sum())})")
    print("median trigger levels (holdout): up {:+.2f}%  down {:+.2f}%  (morning move vs open)".format(
        m.loc[m["trade_date"] > DISCOVERY_END, "up_trigger_pct"].median(), m.loc[m["trade_date"] > DISCOVERY_END, "down_trigger_pct"].median()))

    p["open_to_close"] = (p["close_price"] / p["open_price"] - 1) * 100
    p["cc"] = (p["adj_close_price"] / p.groupby("symbol")["adj_close_price"].shift(1) - 1) * 100  # close N-1 -> close N
    p["period"] = np.where(p["trade_date"] < EVAL_START, None, np.where(p["trade_date"] <= DISCOVERY_END, "discovery", "holdout"))
    rows = []
    for per, g in p.dropna(subset=["period"]).groupby("period"):
        rows.append({"variant": "session_v1 (11:30 -> close)", "period": per, **summarize(g["call"], g["rest_1130"], g["symbol"])})
        rows.append({"variant": "eod_v1 (open -> close)", "period": per, **summarize(g["eod_call"], g["open_to_close"], g["symbol"])})
        rows.append({"variant": "eod_v1 (close -> close)", "period": per, **summarize(g["eod_call"], g["cc"], g["symbol"])})
    print("\n=== operational variants (all inputs known at the previous close, except the trader-observed morning move) ===")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def next_session_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Placeholder rows for the next session after each symbol's last date (prices unknown, history known)."""
    last = raw.sort_values("trade_date").groupby("symbol").tail(1)
    nxt = last[["symbol", "trade_date"]].copy()
    nxt["trade_date"] = nxt["trade_date"] + pd.offsets.BDay(1)
    nxt["last_time"] = pd.Timestamp("2000-01-01 18:09:59").time()  # passes the half-day filter
    return nxt


def publish_bank_pair(session: pd.Timestamp, restate: str | None) -> None:
    """Watchlist bank_pair_wl (end-of-day only): long the bank with the highest relative MLB flow z-score at the last
    close, short the lowest; graded on next-session open -> close spread. Frozen like the triggers."""
    from bank_relative_value import BANKS, load

    d = load()
    last = d["trade_date"].max()
    t = d[d["trade_date"] == last].copy()
    if t["symbol"].nunique() == len(BANKS) and t["z_flow"].notna().all():
        t["rel"] = t["z_flow"] - t["z_flow"].mean()
        t = t.sort_values("rel")
        row = pd.DataFrame([{"trade_date": session, "long": t.iloc[-1]["symbol"], "short": t.iloc[0]["symbol"],
                             "gap": t.iloc[-1]["rel"] - t.iloc[0]["rel"], "as_of_close": str(last.date()), "config_version": CONFIG_VERSION}])
        path = FORWARD_DIR / f"bank_pair_{session.date()}.csv"
        if path.exists() and not restate:
            print(f"{path.name} already published; not overwritten (frozen)")
        else:
            row.to_csv(path, index=False)
            print(f"published {path.name}: long {row['long'].iloc[0]} / short {row['short'].iloc[0]} (open -> close)")
    graded = []
    oc = d.assign(oc=(d["close_price"] / d["open_price"] - 1) * 100).set_index(["symbol", "trade_date"])["oc"]
    for f in sorted(FORWARD_DIR.glob("bank_pair_*.csv")):
        r = pd.read_csv(f, parse_dates=["trade_date"]).iloc[0]
        if (r["long"], r["trade_date"]) in oc.index and (r["short"], r["trade_date"]) in oc.index:
            graded.append({"trade_date": r["trade_date"], "long": r["long"], "short": r["short"],
                           "spread_pct": oc[(r["long"], r["trade_date"])] - oc[(r["short"], r["trade_date"])]})
    if graded:
        g = pd.DataFrame(graded)
        g.to_parquet(FORWARD_DIR / "bank_pair_ledger.parquet")
        print(f"bank_pair_wl forward: {len(g)} sessions, hit {(g['spread_pct'] > 0).mean() * 100:.1f}%, mean spread {g['spread_pct'].mean():+.2f}%")


def forward(restate: str | None = None) -> None:
    FORWARD_DIR.mkdir(parents=True, exist_ok=True)
    raw = build_panel()
    raw = pd.concat([raw, next_session_rows(raw)], ignore_index=True)
    p, _ = engineer(raw)
    p = with_context(p)
    base = add_state(p)
    bank = add_state(p, extra="group_mlb_yday")
    session = base["trade_date"].max()
    out = base[base["trade_date"] == session][["symbol", "trade_date", "z_mlb_net_yday_intensity", "up_trigger_pct", "down_trigger_pct", "eod_call"]].copy()
    b = bank[bank["trade_date"] == session].set_index("symbol")
    out["bank_wl_up_trigger_pct"] = out["symbol"].map(b["up_trigger_pct"])
    out["bank_wl_down_trigger_pct"] = out["symbol"].map(b["down_trigger_pct"])
    out["as_of_close"] = str(raw.loc[raw["trade_date"] < session, "trade_date"].max().date())
    out["config_version"] = CONFIG_VERSION
    path = FORWARD_DIR / f"triggers_{session.date()}.csv"
    if path.exists() and not restate:
        print(f"{path.name} already published; not overwritten (frozen; use --restate before the session opens)")
    else:
        if path.exists():
            with open(FORWARD_DIR / "restatements.log", "a", encoding="utf-8") as log:
                log.write(f"{pd.Timestamp.now(tz='Europe/Istanbul').isoformat()} restated {path.name}: {restate}\n")
        out.to_csv(path, index=False)
        print(f"published {path}")
    print(out.round(2).to_string(index=False))

    publish_bank_pair(session, restate)

    # grade every published trigger file whose session is now in the data
    graded = []
    realized = p.set_index(["symbol", "trade_date"])
    for f in sorted(FORWARD_DIR.glob("triggers_*.csv")):
        t = pd.read_csv(f, parse_dates=["trade_date"])
        t = t[t["trade_date"] < session]
        if t.empty:
            continue
        r = t.join(realized[["morning_ret", "rest_1130", "open_price", "close_price"]], on=["symbol", "trade_date"]).dropna(subset=["rest_1130"])
        if r.empty:
            continue
        r["session_call"] = trigger_call(r["morning_ret"], r["up_trigger_pct"], r["down_trigger_pct"])
        r["bank_wl_call"] = trigger_call(r["morning_ret"], r["bank_wl_up_trigger_pct"], r["bank_wl_down_trigger_pct"])
        r["open_to_close"] = (r["close_price"] / r["open_price"] - 1) * 100
        graded.append(r)
    if graded:
        g = pd.concat(graded, ignore_index=True)
        g.to_parquet(FORWARD_DIR / "forward_ledger.parquet")
        rows = [{"variant": "session_v1", **summarize(g["session_call"], g["rest_1130"], g["symbol"])},
                {"variant": "session_bank_wl", **summarize(g["bank_wl_call"], g["rest_1130"], g["symbol"])},
                {"variant": "eod_v1", **summarize(g["eod_call"], g["open_to_close"], g["symbol"])}]
        print(f"\n=== forward track: {g['trade_date'].nunique()} sessions graded ===")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    else:
        print("\nno published session has realized data yet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--restate", default=None, help="reason; overwrite the next session's published triggers (logged)")
    a = ap.parse_args()
    if a.check:
        check()
    if a.forward:
        forward(a.restate)


if __name__ == "__main__":
    main()
