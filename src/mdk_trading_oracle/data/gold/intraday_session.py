"""Gold layer: Intraday Session Forecaster (11:30 -> close), tournament, published signals and graded ledger.

Tables
    gold_intraday_candidate_results  every candidate's out-of-sample forecast at the realized morning move (tournament input)
    gold_intraday_session_signals    the champion's published grid, levels and trader numbers per stock and session
                                     (frozen once written; source LIVE = published after the previous close,
                                     SIMULATED = walk-forward history)
    gold_intraday_session_ledger     each published signal graded against what happened (same source label)

Modes
    simulate_history(db, symbols, start)  walk-forward history (all candidates, monthly crowning, signals, ledger)
    update_gold_intraday_session(db)      nightly: results + grading for new sessions, then publish the next session
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.models import intraday_session as m

logger = logging.getLogger("mdk_oracle.data.gold.intraday_session")

FOCUS = ["ASELS", "GARAN", "AKBNK", "YKBNK", "ISCTR", "TTKOM"]
SIM_START = "2022-07-01"
SIM_REFIT_EVERY = 5  # simulation refits regression candidates every 5 sessions (always on data before the forecast day)  # first crowning needs CROWN_SESSIONS of candidate results; published history starts ~2023-01


def ensure_tables(db: PostgresManager) -> None:
    db.execute("""
    CREATE TABLE IF NOT EXISTS gold_intraday_candidate_results (
        symbol VARCHAR(16) NOT NULL,
        session_date DATE NOT NULL,
        candidate VARCHAR(48) NOT NULL,
        forecast_pct DOUBLE PRECISION,
        actual_pct DOUBLE PRECISION,
        morning_move_pct DOUBLE PRECISION,
        sigma_pct DOUBLE PRECISION,
        traded BOOLEAN,
        calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, session_date, candidate)
    );
    CREATE TABLE IF NOT EXISTS gold_intraday_session_signals (
        symbol VARCHAR(16) NOT NULL,
        session_date DATE NOT NULL,
        as_of_close DATE,
        source VARCHAR(16) NOT NULL,
        candidate VARCHAR(48),
        model VARCHAR(32),
        feature_set VARCHAR(32),
        training_window INTEGER,
        crowned_on DATE,
        crown_hit_rate_pct DOUBLE PRECISION,
        crown_trades INTEGER,
        grid_forecast_pct DOUBLE PRECISION[],
        buy_level_pct DOUBLE PRECISION,
        sell_level_pct DOUBLE PRECISION,
        sigma_pct DOUBLE PRECISION,
        win_typ_pct DOUBLE PRECISION,
        win_q75_pct DOUBLE PRECISION,
        loss_typ_pct DOUBLE PRECISION,
        record_hit_rate_pct DOUBLE PRECISION,
        record_trades INTEGER,
        record_avg_after_cost_pct DOUBLE PRECISION,
        published_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, session_date)
    );
    CREATE TABLE IF NOT EXISTS gold_intraday_session_ledger (
        symbol VARCHAR(16) NOT NULL,
        session_date DATE NOT NULL,
        source VARCHAR(16) NOT NULL,
        candidate VARCHAR(48),
        open_price DOUBLE PRECISION,
        price_1130 DOUBLE PRECISION,
        close_price DOUBLE PRECISION,
        morning_move_pct DOUBLE PRECISION,
        call INTEGER,
        forecast_pct DOUBLE PRECISION,
        actual_pct DOUBLE PRECISION,
        hit BOOLEAN,
        abs_error_pp DOUBLE PRECISION,
        graded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, session_date)
    );
    """)


def universe(db: PostgresManager) -> list[str]:
    try:
        active = db.query_df("SELECT DISTINCT symbol FROM bronze_bist30_membership WHERE is_active = TRUE")["symbol"].tolist()
    except Exception:
        active = []
    return sorted(set(active) | set(FOCUS))


def _result_rows(sym: str, row: pd.Series, grids: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    out = []
    for name, grid in grids.items():
        f = m.forecast_at(grid, float(row["morning_ret"]))
        out.append({"symbol": sym, "session_date": row["trade_date"], "candidate": name, "forecast_pct": f, "actual_pct": float(row["target"]),
                    "morning_move_pct": float(row["morning_ret"]), "sigma_pct": float(row["sigma_pct"]), "traded": m.trades(f, name)})
    return out


def _signal_row(sym: str, row: pd.Series, champ: m.Champion, grid: np.ndarray | None, results: pd.DataFrame, source: str) -> dict[str, Any]:
    c = m.parse_candidate(champ.name)
    up, dn = m.levels(grid, champ.name) if grid is not None else (None, None)
    nums = m.trader_numbers(results, champ.name, row["trade_date"], float(row["sigma_pct"]))
    return {"symbol": sym, "session_date": row["trade_date"].date(), "as_of_close": row["prev_date"].date(), "source": source,
            "candidate": champ.name, "model": c["model"], "feature_set": c["features"], "training_window": c["window"],
            "crowned_on": champ.crowned_on.date(), "crown_hit_rate_pct": champ.hit_rate_pct, "crown_trades": champ.trades,
            "grid_forecast_pct": [round(float(x), 4) for x in grid] if grid is not None else None,
            "buy_level_pct": up, "sell_level_pct": dn, "sigma_pct": round(float(row["sigma_pct"]), 4) if pd.notna(row["sigma_pct"]) else None,
            "win_typ_pct": nums.get("win_typ_pct"), "win_q75_pct": nums.get("win_q75_pct"), "loss_typ_pct": nums.get("loss_typ_pct"),
            "record_hit_rate_pct": nums.get("record_hit_rate_pct"), "record_trades": nums.get("record_trades"),
            "record_avg_after_cost_pct": nums.get("record_avg_after_cost_pct")}


def grade(signal: dict[str, Any], row: pd.Series) -> dict[str, Any] | None:
    """Grade a stored signal against the realized session (reads the frozen grid at the realized morning move)."""
    if not bool(row["usable"]) or signal.get("grid_forecast_pct") is None:
        return None
    f = m.forecast_at(np.asarray(signal["grid_forecast_pct"], dtype=float), float(row["morning_ret"]))
    traded = m.trades(f, signal["candidate"])
    call = int(np.sign(f)) if traded else 0
    actual = float(row["target"])
    return {"symbol": signal["symbol"], "session_date": signal["session_date"], "source": signal["source"], "candidate": signal["candidate"],
            "open_price": float(row["open_price"]), "price_1130": float(row["price_1130"]), "close_price": float(row["close_price"]),
            "morning_move_pct": float(row["morning_ret"]), "call": call, "forecast_pct": f if traded else 0.0, "actual_pct": actual,
            "hit": bool(np.sign(f) == np.sign(actual)) if traded else None, "abs_error_pp": abs((f if traded else 0.0) - actual)}


def simulate_symbol(args: tuple[str, str, str | None]) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Walk-forward history for one stock: candidate results, monthly crowning, published grid and grading per day."""
    sym, start, end = args
    db = PostgresManager(read_only=True)
    panel = m.build_panel(db, sym)
    if panel.empty:
        return pd.DataFrame(), [], []
    rule = m.rule_state(panel)
    results: list[dict] = []
    by_cand: dict[str, list[dict]] = {}
    empty = pd.DataFrame(columns=["candidate", "session_date", "forecast_pct", "actual_pct", "sigma_pct", "traded"])
    signals, ledger = [], []
    champ, month = None, None
    cache: dict = {}
    idxs = panel.index[(panel["trade_date"] >= start) & ((panel["trade_date"] <= end) if end else True)]
    for i in idxs:
        row = panel.loc[i]
        day = row["trade_date"]
        if month != (day.year, day.month):  # monthly crowning on results strictly before the month
            month = (day.year, day.month)
            champ = m.crown(pd.DataFrame(results) if results else empty, pd.Timestamp(day.year, day.month, 1))
        grids = m.candidate_grids(panel, i, rule, cache=cache, refit_every=SIM_REFIT_EVERY)
        champ_hist = pd.DataFrame(by_cand.get(champ.name, [])) if by_cand.get(champ.name) else empty  # results before `day`
        sig = _signal_row(sym, row, champ, grids.get(champ.name), champ_hist, "SIMULATED")
        signals.append(sig)
        g = grade(sig, row)
        if g:
            ledger.append(g)
        if bool(row["usable"]):
            for r in _result_rows(sym, row, grids):
                results.append(r)
                by_cand.setdefault(r["candidate"], []).append(r)
    logger.info(f"{sym}: simulated {len(idxs)} sessions")
    return pd.DataFrame(results), signals, ledger


def _write(db: PostgresManager, symbols: list[str], results: pd.DataFrame, signals: list[dict], ledger: list[dict], source: str) -> None:
    import polars as pl

    syms_in = ", ".join(f"'{s}'" for s in symbols)
    if not results.empty:
        days = results["session_date"].min()
        db.execute(f"DELETE FROM gold_intraday_candidate_results WHERE symbol IN ({syms_in}) AND session_date >= '{pd.Timestamp(days).date()}';")
        r = results.copy()
        r["session_date"] = pd.to_datetime(r["session_date"]).dt.date
        db.copy_df_to_table(pl.from_pandas(r[["symbol", "session_date", "candidate", "forecast_pct", "actual_pct", "morning_move_pct", "sigma_pct", "traded"]]),
                            "gold_intraday_candidate_results")
    if source == "SIMULATED":
        db.execute(f"DELETE FROM gold_intraday_session_signals WHERE symbol IN ({syms_in}) AND source = 'SIMULATED';")
        db.execute(f"DELETE FROM gold_intraday_session_ledger WHERE symbol IN ({syms_in}) AND source = 'SIMULATED';")
    insert_signals(db, signals, overwrite=False)
    insert_ledger(db, ledger)


SIGNAL_COLS = ["symbol", "session_date", "as_of_close", "source", "candidate", "model", "feature_set", "training_window", "crowned_on",
               "crown_hit_rate_pct", "crown_trades", "grid_forecast_pct", "buy_level_pct", "sell_level_pct", "sigma_pct", "win_typ_pct",
               "win_q75_pct", "loss_typ_pct", "record_hit_rate_pct", "record_trades", "record_avg_after_cost_pct"]
LEDGER_COLS = ["symbol", "session_date", "source", "candidate", "open_price", "price_1130", "close_price", "morning_move_pct", "call",
               "forecast_pct", "actual_pct", "hit", "abs_error_pp"]


def insert_signals(db: PostgresManager, signals: list[dict], overwrite: bool) -> int:
    """Frozen: an existing (symbol, session) signal is kept unless `overwrite` (an explicit restatement)."""
    if not signals:
        return 0
    cols = ", ".join(SIGNAL_COLS)
    vals = ", ".join(f"%({c})s" for c in SIGNAL_COLS)
    conflict = ("DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in SIGNAL_COLS[2:]) + ", published_at = CURRENT_TIMESTAMP") if overwrite else "DO NOTHING"
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO gold_intraday_session_signals ({cols}) VALUES ({vals}) ON CONFLICT (symbol, session_date) {conflict}", signals)
    return len(signals)


def insert_ledger(db: PostgresManager, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = ", ".join(LEDGER_COLS)
    vals = ", ".join(f"%({c})s" for c in LEDGER_COLS)
    upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in LEDGER_COLS[2:])
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO gold_intraday_session_ledger ({cols}) VALUES ({vals}) ON CONFLICT (symbol, session_date) DO UPDATE SET {upd}, graded_at = CURRENT_TIMESTAMP", rows)
    return len(rows)


def simulate_history(db: PostgresManager, symbols: list[str] | None = None, start: str = SIM_START, end: str | None = None, workers: int = 8) -> dict[str, Any]:
    """Walk-forward history for `symbols` (default: universe). Replaces SIMULATED rows; LIVE rows are never touched."""
    ensure_tables(db)
    symbols = symbols or universe(db)
    t0 = datetime.now()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"  # one thread per worker process
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(simulate_symbol, [(s, start, end) for s in symbols]))
    results = pd.concat([o[0] for o in out if not o[0].empty], ignore_index=True) if any(not o[0].empty for o in out) else pd.DataFrame()
    signals = [s for o in out for s in o[1]]
    ledger = [r for o in out for r in o[2]]
    _write(db, symbols, results, signals, ledger, "SIMULATED")
    return {"symbols": len(symbols), "candidate_results": len(results), "signals": len(signals), "ledger": len(ledger),
            "elapsed_sec": (datetime.now() - t0).total_seconds()}


def _load_results(db: PostgresManager, sym: str) -> pd.DataFrame:
    r = db.query_df("SELECT candidate, session_date, forecast_pct, actual_pct, sigma_pct, traded FROM gold_intraday_candidate_results WHERE symbol = %s", (sym,))
    r["session_date"] = pd.to_datetime(r["session_date"])
    return r


def update_symbol(db: PostgresManager, sym: str, restate: bool = False) -> dict[str, Any]:
    """Nightly for one stock: candidate results and grading for new sessions, then publish the next session (LIVE)."""
    base = m.build_panel(db, sym)
    if base.empty:
        return {"symbol": sym, "status": "no_data"}
    last_day = base["trade_date"].max()
    next_day = last_day + pd.offsets.BDay(1)
    panel = m.build_panel(db, sym, next_date=next_day)
    rule = m.rule_state(panel)
    results = _load_results(db, sym)
    done = set(results["session_date"]) if not results.empty else set()
    new_rows = []
    for i in panel.index[panel["usable"] & ~panel["trade_date"].isin(done) & (panel["trade_date"] >= SIM_START)]:
        new_rows.extend(_result_rows(sym, panel.loc[i], m.candidate_grids(panel, i, rule)))
    if new_rows:
        import polars as pl

        nr = pd.DataFrame(new_rows)
        nr["session_date"] = pd.to_datetime(nr["session_date"]).dt.date
        db.copy_df_to_table(pl.from_pandas(nr[["symbol", "session_date", "candidate", "forecast_pct", "actual_pct", "morning_move_pct", "sigma_pct", "traded"]]),
                            "gold_intraday_candidate_results")
        results = _load_results(db, sym)
    # grade stored signals whose session has arrived
    sig = db.query_df("SELECT * FROM gold_intraday_session_signals WHERE symbol = %s AND session_date NOT IN "
                      "(SELECT session_date FROM gold_intraday_session_ledger WHERE symbol = %s)", (sym, sym))
    graded = []
    by_date = panel.set_index("trade_date")
    for s in sig.to_dict("records"):
        d = pd.Timestamp(s["session_date"])
        if d in by_date.index:
            g = grade(s, by_date.loc[d])
            if g:
                graded.append(g)
    insert_ledger(db, graded)
    # publish the next session
    row = panel.iloc[-1]
    champ = m.crown(results, pd.Timestamp(next_day.year, next_day.month, 1))
    grid = m.candidate_grids(panel, len(panel) - 1, rule, names=[champ.name]).get(champ.name)
    published = insert_signals(db, [_signal_row(sym, row, champ, grid, results, "LIVE")], overwrite=restate)
    return {"symbol": sym, "new_results_sessions": len({r["session_date"] for r in new_rows}), "graded": len(graded),
            "published_for": str(next_day.date()), "champion": champ.name, "published": published}


def update_gold_intraday_session(db: PostgresManager, symbols: list[str] | None = None, restate: bool = False) -> dict[str, Any]:
    ensure_tables(db)
    symbols = symbols or universe(db)
    t0 = datetime.now()
    out = []
    for s in symbols:
        try:
            out.append(update_symbol(db, s, restate=restate))
        except Exception as e:  # one stock must not stop the others
            logger.warning(f"Intraday session update failed for {s}: {e}")
            out.append({"symbol": s, "status": f"error: {e}"})
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"Intraday session forecaster updated for {len(symbols)} stocks in {elapsed:.1f}s")
    return {"status": "success", "symbols": out, "elapsed_sec": elapsed}
