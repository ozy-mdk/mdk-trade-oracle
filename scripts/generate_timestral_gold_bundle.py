#!/usr/bin/env python
"""Generate the offline Timestral-209M + Policy C Gold Bundle for the 6 Focus Stocks.

Builds `~/data/mdk_oracle/gold_cache/timestral_gold_bundle.json` containing:
- Live T+1 and 3D-30D forecasts (with BIST ±10% circuit-breaker bounded quantile ranges and SNR fusion)
- 90-session audited walk-forward backtest ledgers per stock and horizon
- Automatic PostgreSQL Gold sync payload (`sync_cached_timestral_gold_to_db`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mdk_trading_oracle.data.gold.tertip_forecasts import _direction as t1_dir  # noqa: E402
from mdk_trading_oracle.data.gold.tertip_forecasts import _stance as t1_stance  # noqa: E402
from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import _direction as mh_dir  # noqa: E402
from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import constant_call_hit_rates  # noqa: E402
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (  # noqa: E402
    DEADBAND_PCT,
    get_feature_cols_for_mode,
)
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (  # noqa: E402
    engineer_multi_horizon_features,
    generate_live_horizon_forecast,
)
from mdk_trading_oracle.models.horizon_crowning import (  # noqa: E402
    LOOKBACKS,
    candidate_configs,
    crown,
    crown_single,
    month_starts,
)
from mdk_trading_oracle.models.t1_forecast import T1Config, forecast_t1, prepare_feature_frame, t1_labels  # noqa: E402
from mdk_trading_oracle.models.timestral_forecaster import (  # noqa: E402
    GOLD_BUNDLE_PATH,
    HORIZON_DAYS,
    HORIZONS_ALL,
    blend_policy_c_with_timestral,
    check_horizon_hit,
    compute_timestral_horizon_table,
    enrich_risk_profile_with_timestral,
    format_hybrid_champion_name,
    horizon_deadband,
)
from scripts.experiments.tournament_variants import EVENT_COLS, MEAN_REVERSION, REGIME_COLS, regime  # noqa: E402

STOCKS = ["ASELS", "AKBNK", "ISCTR", "GARAN", "YKBNK", "THYAO"]
FRAMES_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "feature_importance" / "frames"
VARIANTS_DIR = Path.home() / "data" / "mdk_oracle" / "experiments" / "tournament_variants"
EVENT_DIR = Path.home() / "data" / "mdk_oracle" / "00_raw_data" / "event_calendar"
LB_LABEL_OF = {lb: lbl for lbl, lb in LOOKBACKS}


def _add_local_event_features(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Add event calendar columns offline from `00_raw_data/event_calendar/`."""
    out = df.copy()
    dates = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
    n = len(dates)
    mpc_file = EVENT_DIR / "tcmb_mpc_meetings.csv"
    div_file = EVENT_DIR / "bist_dividends_yfinance.csv"
    mpc_dates = sorted(pd.read_csv(mpc_file)["meeting_date"].astype(str).str[:10].tolist()) if mpc_file.exists() else []
    div_df = pd.read_csv(div_file) if div_file.exists() else pd.DataFrame(columns=["symbol", "ex_date", "gross_dividend_tl"])
    sym_div = div_df[div_df["symbol"] == sym].sort_values("ex_date") if not div_df.empty else pd.DataFrame()
    div_dates = sym_div["ex_date"].astype(str).str[:10].tolist() if not sym_div.empty else []
    div_amts = sym_div["gross_dividend_tl"].to_numpy(float) if not sym_div.empty else np.zeros(0)

    to_next_mpc = np.full(n, 30.0)
    since_last_mpc = np.full(n, 30.0)
    next_sess_mpc = np.zeros(n)
    to_next_div = np.full(n, 60.0)
    div_yield = np.zeros(n)
    close_px = out["close_price"].to_numpy(float)

    for i, d in enumerate(dates):
        future_mpc = [m for m in mpc_dates if m > d]
        past_mpc = [m for m in mpc_dates if m <= d]
        if future_mpc:
            days_f = max(1, int((pd.Timestamp(future_mpc[0]) - pd.Timestamp(d)).days * 5 / 7))
            to_next_mpc[i] = float(min(30, days_f))
            if days_f <= 1:
                next_sess_mpc[i] = 1.0
        if past_mpc:
            days_p = max(0, int((pd.Timestamp(d) - pd.Timestamp(past_mpc[-1])).days * 5 / 7))
            since_last_mpc[i] = float(min(30, days_p))
        for j, dd in enumerate(div_dates):
            if dd > d:
                days_d = max(1, int((pd.Timestamp(dd) - pd.Timestamp(d)).days * 5 / 7))
                if days_d <= 60:
                    to_next_div[i] = float(days_d)
                    if close_px[i] > 0:
                        div_yield[i] = float(div_amts[j] / close_px[i] * 100.0)
                break

    out["feat_evt_mpc_sessions_to_next"] = to_next_mpc
    out["feat_evt_mpc_sessions_since_last"] = since_last_mpc
    out["feat_evt_mpc_next_session"] = next_sess_mpc
    out["feat_evt_mpc_last_change_bps"] = 0.0
    out["feat_evt_div_sessions_to_next"] = to_next_div
    out["feat_evt_div_yield_next_pct"] = div_yield
    out["feat_evt_rebal_sessions_to_next"] = 60.0
    out["feat_evt_rebal_entering"] = 0.0
    out["feat_evt_rebal_leaving"] = 0.0
    return out


def _apply_local_feature_set(df: pd.DataFrame, sym: str, fs: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    if fs == "events":
        return _add_local_event_features(df, sym), list(EVENT_COLS), []
    if fs == "regime_no_sma50":
        out = regime(df)
        drop = ["feat_price_vs_sma50_pct", "feat_price_vs_sma50_pct_x_down"]
        out = out.drop(columns=[c for c in drop if c in out.columns])
        extra = [c for c in REGIME_COLS + [f"{c}_x_down" for c in MEAN_REVERSION if f"{c}_x_down" in out.columns] if c not in drop]
        return out, extra, drop
    return df.copy(), [], []


def _crown_policy_c_for_symbol(sym: str, dates: list[str]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Return `(latest_crowns_by_horizon, full_90_session_policy_c_track)` for `sym`."""
    c_ev = pd.read_parquet(VARIANTS_DIR / f"candidates_events_{sym}.parquet")
    c_reg = pd.read_parquet(VARIANTS_DIR / f"candidates_regime_no_sma50_{sym}.parquet")
    c_pr = pd.read_parquet(VARIANTS_DIR / f"candidates_production_{sym}.parquet")

    pool = pd.concat([
        c_ev[c_ev["horizon"].isin(["1d", "3d"])],
        c_reg[c_reg["horizon"].isin(["5d", "10d"])],
        c_pr[(c_pr["horizon"] == "15d") & (~c_pr["config_key"].str.contains("/63/"))],
        c_pr[c_pr["horizon"] == "30d"],
    ], ignore_index=True)

    cfgs_all = candidate_configs()
    cfg_by_key = {c.key: c for c in cfgs_all}
    months = [m for m in month_starts(pd.Series(dates), dates[0]) if m >= dates[max(0, len(dates) - 150)]]

    track_parts = []
    latest_crowns: dict[str, dict[str, Any]] = {}
    for i, m in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else "9999-12-31"
        # Crown 1D on events
        c1 = crown_single(c_ev, m, "1d", configs=cfgs_all, eval_sessions=90)
        # Crown 3D-15D family-first on pool, and 30D on production
        # For each horizon in pool, pick best candidate in crowned family
        champs_pr = crown(c_pr[c_pr["horizon"] != "1d"], m, configs=cfgs_all, eval_sessions=90) or {}
        champs_ev = crown(c_ev[c_ev["horizon"] != "1d"], m, configs=cfgs_all, eval_sessions=90) or {}
        champs_reg = crown(c_reg[c_reg["horizon"] != "1d"], m, configs=cfgs_all, eval_sessions=90) or {}
        cfgs_no63 = [c for c in cfgs_all if c.lookback != 63]
        champs_no63 = crown(c_pr[c_pr["horizon"] != "1d"], m, configs=cfgs_no63, eval_sessions=90) or {}

        chosen: dict[str, dict[str, Any]] = {}
        if c1:
            chosen["1d"] = {**c1, "feature_set": "events", "crown_date": m}
        if "3d" in champs_ev:
            chosen["3d"] = {**champs_ev["3d"], "feature_set": "events", "crown_date": m}
        if "5d" in champs_reg:
            chosen["5d"] = {**champs_reg["5d"], "feature_set": "regime_no_sma50", "crown_date": m}
        if "10d" in champs_reg:
            chosen["10d"] = {**champs_reg["10d"], "feature_set": "regime_no_sma50", "crown_date": m}
        if "15d" in champs_no63:
            chosen["15d"] = {**champs_no63["15d"], "feature_set": "production", "crown_date": m}
        if "30d" in champs_pr:
            chosen["30d"] = {**champs_pr["30d"], "feature_set": "production", "crown_date": m}

        for h, ch in chosen.items():
            latest_crowns[h] = ch
            ckey = ch["config"].key
            sel = pool[(pool["horizon"] == h) & (pool["config_key"] == ckey) & (pool["trade_date"] >= m) & (pool["trade_date"] < end)]
            if not sel.empty:
                c_obj = cfg_by_key[ckey]
                track_parts.append(
                    sel.assign(
                        crown_date=m,
                        model=c_obj.model,
                        lookback=c_obj.lookback,
                        lookback_label=c_obj.lookback_label,
                        feature_mode=c_obj.feature_mode,
                        feature_set=ch["feature_set"],
                    )
                )

    full_track = pd.concat(track_parts, ignore_index=True)
    return latest_crowns, full_track


def build_bundle() -> None:
    GOLD_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {"symbols": {}}

    for sym in STOCKS:
        df = pd.read_parquet(FRAMES_DIR / f"{sym}.parquet").sort_values("trade_date").reset_index(drop=True)
        dates = [str(d)[:10] for d in df["trade_date"].tolist()]
        next_date = dict(zip(dates[:-1], dates[1:]))
        price_by_date = dict(zip(dates, df["close_price"].to_numpy(float)))
        as_of_date = dates[-1]
        current_price = round(float(df["close_price"].iloc[-1]), 2)

        latest_crowns, polc_track = _crown_policy_c_for_symbol(sym, dates)
        ts_df = compute_timestral_horizon_table(sym, df, horizons=HORIZONS_ALL)
        ts_map = {(str(r["horizon"]), str(r["trade_date"])[:10]): r for r in ts_df.to_dict("records")}

        feat_all, _ = engineer_multi_horizon_features(df.copy(), feature_mode="FIFO_PLUS_YTD")

        horizons_out: dict[str, Any] = {}
        t1_daily_fc: dict[str, Any] | None = None
        t1_bt_rows: list[dict[str, Any]] = []

        for h in HORIZONS_ALL:
            n_days = HORIZON_DAYS[h]
            ch = latest_crowns[h]
            c_obj = ch["config"]
            fs = ch["feature_set"]
            crown_date = ch["crown_date"]

            h_track = polc_track[polc_track["horizon"] == h].sort_values("trade_date").iloc[-90:]
            ledger: list[dict[str, Any]] = []
            for r in h_track.itertuples(index=False):
                td = str(r.trade_date)[:10]
                t_end = str(r.target_date_end)[:10]
                curr_px = round(float(price_by_date.get(td, current_price)), 2)
                act_ret = round(float(r.actual), 2)
                polc_pred = float(r.pred)
                ts_row = ts_map.get((h, td))
                if ts_row is not None:
                    p_sized, p_raw = blend_policy_c_with_timestral(
                        polc_pred,
                        ts_row["pred_med"],
                        ts_row["q25"],
                        ts_row["q75"],
                        h,
                        bist30_med=ts_row.get("bist30_med"),
                        bist30_ret_1d=ts_row.get("bist30_ret_1d"),
                        days_since_neg_shock=ts_row.get("days_since_neg_shock"),
                        days_since_pos_shock=ts_row.get("days_since_pos_shock"),
                        bist30_vs_sma50=ts_row.get("bist30_vs_sma50"),
                        bist30_ret_20d=ts_row.get("bist30_ret_20d"),
                        ts_row=ts_row,
                    )
                    pred_ret = round(float(p_sized), 2)
                    hit_val = check_horizon_hit(float(p_raw), act_ret, h)
                else:
                    pred_ret = round(polc_pred, 2)
                    hit_val = bool(r.is_hit)
                err_val = round(abs(pred_ret - act_ret), 2)
                champ_name = format_hybrid_champion_name(r.model, h)
                dir_fn = (lambda x: "BUY" if x > DEADBAND_PCT else ("SELL" if x < -DEADBAND_PCT else "NEUTRAL")) if h == "1d" else (lambda x: mh_dir(x, n_days))

                ledger.append({
                    "trade_date": td,
                    "target_date_start": next_date.get(td, td),
                    "target_date_end": t_end,
                    "current_price": curr_px,
                    "actual_avg_price": round(curr_px * (1.0 + act_ret / 100.0), 2),
                    "actual_return_pct": act_ret,
                    "pred_avg_price": round(curr_px * (1.0 + pred_ret / 100.0), 2),
                    "pred_return_pct": pred_ret,
                    "pred_direction": dir_fn(pred_ret),
                    "actual_direction": dir_fn(act_ret),
                    "err_pct": err_val,
                    "is_hit": hit_val,
                    "is_pending": False,
                    "champion_model": champ_name,
                    "feature_mode": r.feature_mode,
                    "crown_date": r.crown_date,
                    "actual_training_sessions": int(r.lookback),
                    "target_lookback_sessions": int(r.lookback),
                    "actual_training_months": round(int(r.lookback) / 21.0, 1),
                    "data_sufficiency_status": "FULL",
                })
                if h == "1d":
                    t1_bt_rows.append({
                        "symbol": sym,
                        "trade_date": t_end,
                        "ml_pred_price": round(curr_px * (1.0 + pred_ret / 100.0), 2),
                        "ml_pred_return_pct": pred_ret,
                        "ml_direction": t1_dir(pred_ret),
                        "ml_err_pct": err_val,
                        "ml_is_hit": hit_val,
                        "winner": "TERTIP_ML_CHALLENGER" if hit_val else "PROPHET_BASE",
                        "ml_champion_type": champ_name,
                    })

            # Compute live forecast at `as_of_date`
            ts_live = ts_map.get((h, as_of_date))
            if h == "1d":
                src, extra, drop = _apply_local_feature_set(df, sym, fs)
                f_cols = [c for c in get_feature_cols_for_mode(c_obj.feature_mode) if c not in drop] + extra
                frame = prepare_feature_frame(src, f_cols)
                fc_1d = forecast_t1(frame, len(frame) - 1, T1Config(c_obj.model, c_obj.lookback, c_obj.feature_mode), f_cols, t1_labels(frame))
                polc_live_ret = float(fc_1d["model_return_pct"]) if fc_1d else 0.0
                live_base: dict[str, Any] = {
                    "actual_training_sessions": int(fc_1d["n_train"]) if fc_1d else c_obj.lookback,
                    "data_sufficiency_status": fc_1d["data_sufficiency_status"] if fc_1d else "FULL",
                    "risk_profile": {},
                }
            else:
                live_frame = _apply_local_feature_set(feat_all, sym, fs)[0]
                live_base = generate_live_horizon_forecast(
                    live_frame.copy(), h, champion_model=c_obj.model, train_lookback_sessions=c_obj.lookback, feature_mode=c_obj.feature_mode
                )
                polc_live_ret = float(live_base.get("expected_return_pct", 0.0))

            if ts_live is not None:
                p_sized_live, _ = blend_policy_c_with_timestral(
                    polc_live_ret,
                    ts_live["pred_med"],
                    ts_live["q25"],
                    ts_live["q75"],
                    h,
                    bist30_med=ts_live.get("bist30_med"),
                    bist30_ret_1d=ts_live.get("bist30_ret_1d"),
                    days_since_neg_shock=ts_live.get("days_since_neg_shock"),
                    days_since_pos_shock=ts_live.get("days_since_pos_shock"),
                    bist30_vs_sma50=ts_live.get("bist30_vs_sma50"),
                    bist30_ret_20d=ts_live.get("bist30_ret_20d"),
                    ts_row=ts_live,
                )
                pred_live = round(float(p_sized_live), 2)
                p_low = round(current_price * (1.0 + float(ts_live["q05"]) / 100.0), 2)
                p_high = round(current_price * (1.0 + float(ts_live["q95"]) / 100.0), 2)
                rp = enrich_risk_profile_with_timestral(
                    live_base.get("risk_profile"), ts_live, polc_live_ret, current_price, horizon_days=n_days, target_return_pct=pred_live
                )
            else:
                pred_live = round(polc_live_ret, 2)
                p_low = round(current_price * 0.96, 2)
                p_high = round(current_price * 1.04, 2)
                rp = live_base.get("risk_profile") or {}

            t_price = round(current_price * (1.0 + pred_live / 100.0), 2)
            db_h = horizon_deadband(h)
            swing_stance = "BULLISH" if pred_live > db_h else ("BEARISH" if pred_live < -db_h else "NEUTRAL")
            stance_1d, playbook_1d = t1_stance(pred_live)
            crowned_model_label = format_hybrid_champion_name(c_obj.model, h)

            hits = sum(1 for row in ledger if row["is_hit"])
            total = len(ledger)
            hit_rate = round(hits / max(1, total) * 100.0, 1)
            mae = round(float(np.mean([row["err_pct"] for row in ledger])), 2) if ledger else 0.0
            up_r, down_r = constant_call_hit_rates(np.array([row["actual_return_pct"] for row in ledger]), n_days)

            horizons_out[h] = {
                "meta": {
                    "horizon": h,
                    "horizon_days": n_days,
                    "crowned_model": crowned_model_label,
                    "crowned_horizon": c_obj.lookback_label,
                    "crowned_on": crown_date,
                    "hits": hits,
                    "total_evals": total,
                    "hit_rate_pct": hit_rate,
                    "mae_pct": mae,
                    "training_lookback_sessions": c_obj.lookback,
                    "feature_mode": c_obj.feature_mode,
                    "always_up_hit_rate_pct": up_r,
                    "always_down_hit_rate_pct": down_r,
                },
                "live_forecast": {
                    "target_price": t_price,
                    "expected_return_pct": pred_live,
                    "price_low": p_low,
                    "price_high": p_high,
                    "stance": swing_stance,
                    "conviction": stance_1d if h == "1d" else (live_base.get("conviction") or swing_stance),
                    "playbook": playbook_1d.replace(" ", "_") if h == "1d" else (live_base.get("playbook") or "MOMENTUM_EXPANSION"),
                    "actual_training_sessions": int(live_base.get("actual_training_sessions", c_obj.lookback)),
                    "data_sufficiency_status": live_base.get("data_sufficiency_status", "FULL"),
                    "risk_profile": rp,
                },
                "backtest_ledger": ledger,
            }

            if h == "1d":
                t1_daily_fc = {
                    "symbol": sym,
                    "as_of_date": as_of_date,
                    "current_price": current_price,
                    "target_price": t_price,
                    "expected_return_pct": pred_live,
                    "price_low": p_low,
                    "price_high": p_high,
                    "stance": stance_1d,
                    "conviction": stance_1d,
                    "playbook": playbook_1d,
                    "ml_champion_type": crowned_model_label,
                    "champion_dir_hits": hits,
                    "champion_dir_hit_rate_pct": hit_rate,
                    "champion_mae_pct": mae,
                    "prophet_target_price": current_price,
                    "prophet_expected_return_pct": 0.0,
                    "training_lookback_sessions": c_obj.lookback,
                    "crowned_horizon": c_obj.lookback_label,
                    "actual_training_sessions": int(live_base.get("actual_training_sessions", c_obj.lookback)),
                    "data_sufficiency_status": "FULL",
                    "feature_mode": c_obj.feature_mode,
                    "risk_profile": json.dumps(rp),
                    "crowned_on": crown_date,
                }

        bundle["symbols"][sym] = {
            "as_of_date": as_of_date,
            "current_price": current_price,
            "horizons": horizons_out,
            "t1_daily_forecast": t1_daily_fc,
            "t1_backtest_rows": t1_bt_rows,
        }
        print(
            f"[{sym}] ({as_of_date}, ₺{current_price:.2f}) -> "
            + " | ".join(
                f"{h.upper()}: {horizons_out[h]['live_forecast']['expected_return_pct']:+.2f}% "
                f"({horizons_out[h]['meta']['crowned_model']}, {horizons_out[h]['meta']['hit_rate_pct']:.1f}% hit, MAE {horizons_out[h]['meta']['mae_pct']:.2f}%)"
                for h in HORIZONS_ALL
            ),
            flush=True,
        )

    GOLD_BUNDLE_PATH.write_text(json.dumps(bundle, indent=2))
    print(f"\nSaved Timestral Gold bundle to {GOLD_BUNDLE_PATH}", flush=True)


if __name__ == "__main__":
    build_bundle()

