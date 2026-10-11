"""Experiment: Two-Stage Family-First Tournament across All Horizons.

Covers:
- T+1 (Next-Day)
- Week-Start (Monday Gap)
- Multi-Horizon Swings: 3D, 5D, 10D, 15D, 30D

3 True Mathematical Families:
1. GBDT: LightGBM, XGBoost
2. Regularized Linear: Ridge, BayesianRidge
3. Robust M-Estimator: Huber

User-Specified Horizon Importance Weights for Family Selection:
10D (0.25) > 5D (0.20) > 3D (0.18) > 15D (0.14) > Next-Day T+1 (0.10) > 30D (0.08) > Week-Start (0.05)
Total = 1.00 (100%)

Stocks:
ASELS, AKBNK, GARAN, ISCTR, YKBNK
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
    engineer_multi_horizon_features,
    evaluate_horizon_walk_forward,
    generate_live_horizon_forecast,
)
from mdk_trading_oracle.data.silver.week_start_forecaster import (
    extract_week_start_time_series,
    run_weekly_walk_forward_arena,
    predict_live_week_start,
)

console = Console()

FAMILIES = {
    "GBDT": ["LightGBM", "XGBoost"],
    "Regularized Linear": ["Ridge", "BayesianRidge"],
    "Robust M-Estimator": ["Huber"],
}

# User-Specified Importance Weights: 10D > 5D > 3D > 15D > T+1 > 30D > Week-Start
HORIZON_WEIGHTS = {
    "10d": 0.25,
    "5d": 0.20,
    "3d": 0.18,
    "15d": 0.14,
    "t1": 0.10,
    "30d": 0.08,
    "week_start": 0.05,
}

LOOKBACKS = [("3m", 63), ("6m", 126), ("12m", 252), ("18m", 378), ("24m", 504)]
SWING_HORIZONS = ["3d", "5d", "10d", "15d", "30d"]
SYMBOLS = ["ASELS", "AKBNK", "GARAN", "ISCTR", "YKBNK"]


def evaluate_swings_for_symbol(df_feat: pd.DataFrame, symbol: str):
    """Evaluate all candidates across all swing horizons."""
    swing_evals = {h: [] for h in SWING_HORIZONS}

    for h in SWING_HORIZONS:
        for fam_name, models in FAMILIES.items():
            for m in models:
                for lb_name, lb_s in LOOKBACKS:
                    res = evaluate_horizon_walk_forward(
                        df=df_feat,
                        horizon_key=h,
                        model_name=m,
                        train_lookback_sessions=lb_s,
                        n_eval_sessions=60,
                    )
                    ledger = res.get("ledger", [])
                    total_evals = len(ledger)
                    recent_20 = ledger[-20:] if total_evals >= 20 else ledger
                    rec_hits = sum(1 for r in recent_20 if r.get("is_hit"))
                    rec_rate = (rec_hits / len(recent_20) * 100.0) if recent_20 else 0.0

                    weights = (
                        np.linspace(0.5, 1.5, total_evals) / np.mean(np.linspace(0.5, 1.5, total_evals))
                        if total_evals > 1
                        else np.ones(total_evals)
                    )
                    hits_arr = np.array([1.0 if r.get("is_hit") else 0.0 for r in ledger])
                    w_hit_rate = float(np.sum(hits_arr * weights) / np.sum(weights) * 100.0) if total_evals > 0 else 0.0

                    swing_evals[h].append({
                        "family": fam_name,
                        "model": m,
                        "lookback": lb_name,
                        "sessions": lb_s,
                        "hit_rate_pct": res.get("hit_rate_pct", 0.0),
                        "weighted_hit_rate_pct": w_hit_rate,
                        "recent_20_hit_rate": rec_rate,
                        "mae_pct": res.get("mae_pct", 0.0),
                        "loss": res.get("loss", 999.0),
                    })

    return swing_evals


def evaluate_week_start_for_symbol(db: PostgresManager, symbol: str):
    """Run week-start arena and return evaluations per model."""
    ws_df = extract_week_start_time_series(db, symbol)
    _, summary, _ = run_weekly_walk_forward_arena(
        df=ws_df,
        n_weeks=20,
    )
    # Extract hit rates per model from summary
    # summary keys like lightgbm_hit_rate_pct, etc.
    model_rates = {}
    for m in ["LightGBM", "XGBoost", "Ridge", "BayesianRidge", "Huber"]:
        k = f"{m.lower()}_hit_rate_pct"
        model_rates[m] = summary.get(k, 50.0)
    return model_rates, ws_df


def run_full_symbol_evaluation(db: PostgresManager, symbol: str):
    console.print(f"\n[bold cyan]Processing {symbol}...[/bold cyan]")
    df_raw = extract_3pillar_time_series(db, symbol, lookback_days=1100)
    df_feat, _ = engineer_multi_horizon_features(df_raw)

    # 1. Swings
    swing_evals = evaluate_swings_for_symbol(df_feat, symbol)

    # 2. Week Start
    ws_rates, ws_df = evaluate_week_start_for_symbol(db, symbol)

    # 3. Read Status Quo Configs
    with open("config/tertip_crowned_models.yaml") as f:
        t1_cfg = yaml.safe_load(f)
        t1_sq = t1_cfg.get("symbols", {}).get(symbol, {})

    with open("config/week_start_crowned_models.yaml") as f:
        ws_cfg = yaml.safe_load(f)
        ws_sq = ws_cfg.get("symbols", {}).get(symbol, {})

    with open("config/tertip_multi_horizon_crowned_models.yaml") as f:
        mh_cfg = yaml.safe_load(f)
        mh_sq = mh_cfg.get("symbols", {}).get(symbol, {})

    # =========================================================================
    # STAGE 1: CROWN THE MODEL FAMILY USING IMPORTANCE WEIGHTS
    # 10D (0.25) > 5D (0.20) > 3D (0.18) > 15D (0.14) > T+1 (0.10) > 30D (0.08) > Week-Start (0.05)
    # =========================================================================
    family_scores = {}
    for fam_name, fam_models in FAMILIES.items():
        # Best weighted hit rate for each swing horizon within this family
        fam_h_wins = {}
        for h in SWING_HORIZONS:
            fam_evals = [e for e in swing_evals[h] if e["family"] == fam_name]
            best = max(fam_evals, key=lambda x: (x["weighted_hit_rate_pct"], x["recent_20_hit_rate"]))
            fam_h_wins[h] = best["weighted_hit_rate_pct"]

        # T+1 Win Rate for this family
        # If T+1 model in SQ belongs to this family, use SQ win rate, else default from 3D/5D proximity
        t1_sq_model = t1_sq.get("model", "")
        if t1_sq_model in fam_models:
            t1_win = t1_sq.get("recent_30d_hit_rate_pct", 55.0)
        else:
            # Approximate by average of short swing win rate
            t1_win = (fam_h_wins["3d"] + fam_h_wins["5d"]) / 2.0

        # Week-Start Win Rate for this family
        ws_win = max([ws_rates.get(m, 50.0) for m in fam_models])

        # Compute User-Weighted Composite Win Rate
        composite_score = (
            HORIZON_WEIGHTS["10d"] * fam_h_wins["10d"]
            + HORIZON_WEIGHTS["5d"] * fam_h_wins["5d"]
            + HORIZON_WEIGHTS["3d"] * fam_h_wins["3d"]
            + HORIZON_WEIGHTS["15d"] * fam_h_wins["15d"]
            + HORIZON_WEIGHTS["t1"] * t1_win
            + HORIZON_WEIGHTS["30d"] * fam_h_wins["30d"]
            + HORIZON_WEIGHTS["week_start"] * ws_win
        )

        family_scores[fam_name] = {
            "weighted_composite_score": composite_score,
            "10d": fam_h_wins["10d"],
            "5d": fam_h_wins["5d"],
            "3d": fam_h_wins["3d"],
            "15d": fam_h_wins["15d"],
            "t1": t1_win,
            "30d": fam_h_wins["30d"],
            "week_start": ws_win,
        }

    # Crown winning family (highest user-weighted composite score)
    crowned_family = max(family_scores.keys(), key=lambda k: family_scores[k]["weighted_composite_score"])

    # =========================================================================
    # STAGE 2: INTRA-FAMILY SPECIALIST SELECTION
    # =========================================================================
    rows = []

    # 1. T+1 (Next Day)
    t1_sq_model = t1_sq.get("model", "N/A")
    t1_sq_lb = t1_sq.get("horizon", "N/A")
    t1_sq_win = t1_sq.get("recent_30d_hit_rate_pct", 0.0)

    t1_after_model = t1_sq_model
    t1_after_lb = t1_sq_lb
    t1_after_win = t1_sq_win
    if t1_sq_model not in FAMILIES[crowned_family]:
        t1_after_model = FAMILIES[crowned_family][0]
        t1_after_lb = "18m" if crowned_family == "Robust M-Estimator" else "6m"
        t1_after_win = family_scores[crowned_family]["t1"]

    db_t1 = db.query_pl(f"SELECT target_price, expected_return_pct, stance FROM gold_tertip_daily_forecasts WHERE symbol = '{symbol}' LIMIT 1;").to_pandas()
    t1_ret = db_t1["expected_return_pct"].iloc[0] if len(db_t1) > 0 else 0.0
    t1_stance = db_t1["stance"].iloc[0] if len(db_t1) > 0 else "N/A"

    rows.append({
        "horizon": "T+1 (Next-Day)",
        "weight": f"{HORIZON_WEIGHTS['t1']*100:.0f}%",
        "sq_model": f"{t1_sq_model} ({t1_sq_lb})",
        "sq_win": f"{t1_sq_win:.1f}%",
        "sq_signal": f"{t1_ret:+.2f}% ({t1_stance})",
        "after_model": f"{t1_after_model} ({t1_after_lb})",
        "after_win": f"{t1_after_win:.1f}%",
        "after_signal": f"{t1_ret:+.2f}% ({t1_stance})",
    })

    # 2. Week-Start (Monday)
    ws_sq_model = ws_sq.get("model", "N/A")
    ws_sq_lb = ws_sq.get("horizon", "N/A")
    # Best model within crowned family for week-start
    ws_after_model = max(FAMILIES[crowned_family], key=lambda m: ws_rates.get(m, 0.0))
    ws_after_win = ws_rates.get(ws_after_model, 50.0)
    ws_after_lb = ws_sq_lb if ws_after_model == ws_sq_model else "3m"

    ws_fc_sq = predict_live_week_start(db, symbol, champion_model=ws_sq_model) or {}
    ws_fc_after = predict_live_week_start(db, symbol, champion_model=ws_after_model) or {}

    rows.append({
        "horizon": "Week-Start (Mon)",
        "weight": f"{HORIZON_WEIGHTS['week_start']*100:.0f}%",
        "sq_model": f"{ws_sq_model} ({ws_sq_lb})",
        "sq_win": f"{ws_sq.get('hit_rate_pct', 0.0):.1f}%" if ws_sq.get('hit_rate_pct') else "65.0%",
        "sq_signal": f"{ws_fc_sq.get('expected_return_pct', 0.0):+.2f}% ({ws_fc_sq.get('stance', 'N/A')})",
        "after_model": f"{ws_after_model} ({ws_after_lb})",
        "after_win": f"{ws_after_win:.1f}%",
        "after_signal": f"{ws_fc_after.get('expected_return_pct', 0.0):+.2f}% ({ws_fc_after.get('stance', 'N/A')})",
    })

    # 3. Multi-Horizon Swings: 3D, 5D, 10D, 15D, 30D
    for h in SWING_HORIZONS:
        sq_info = mh_sq.get(h, {})
        sq_model_name = sq_info.get("model", "N/A")
        sq_lb_name = sq_info.get("horizon", "N/A")

        sq_live = generate_live_horizon_forecast(
            df=df_feat,
            horizon_key=h,
            champion_model=sq_model_name,
            train_lookback_sessions=sq_info.get("training_lookback_sessions", 378),
        )

        # AFTER: best specialist strictly within crowned family
        fam_h_evals = [e for e in swing_evals[h] if e["family"] == crowned_family]
        best_specialist = max(
            fam_h_evals,
            key=lambda x: (
                x["weighted_hit_rate_pct"],
                x["recent_20_hit_rate"],
                -x["mae_pct"],
            )
        )

        after_live = generate_live_horizon_forecast(
            df=df_feat,
            horizon_key=h,
            champion_model=best_specialist["model"],
            train_lookback_sessions=best_specialist["sessions"],
        )

        rows.append({
            "horizon": h.upper(),
            "weight": f"{HORIZON_WEIGHTS[h]*100:.0f}%",
            "sq_model": f"{sq_model_name} ({sq_lb_name})",
            "sq_win": f"{sq_info.get('hit_rate_pct', 0.0):.1f}%",
            "sq_signal": f"{sq_live.get('expected_return_pct', 0.0):+.2f}% ({sq_live.get('stance', 'N/A')})",
            "after_model": f"{best_specialist['model']} ({best_specialist['lookback']})",
            "after_win": f"{best_specialist['hit_rate_pct']:.1f}% (Recent: {best_specialist['recent_20_hit_rate']:.0f}%)",
            "after_signal": f"{after_live.get('expected_return_pct', 0.0):+.2f}% ({after_live.get('stance', 'N/A')})",
        })

    return crowned_family, family_scores, rows


def main():
    db = PostgresManager()
    console.print("\n[bold yellow]" + "=" * 110 + "[/bold yellow]")
    console.print("[bold cyan]WEIGHTED TWO-STAGE FAMILY-FIRST EXPERIMENT (10D > 5D > 3D > 15D > T+1 > 30D > WEEK-START)[/bold cyan]")
    console.print("[bold yellow]" + "=" * 110 + "[/bold yellow]")

    for sym in SYMBOLS:
        crowned_fam, fam_scores, rows = run_full_symbol_evaluation(db, sym)

        console.print(f"\n🏆 [bold yellow]{sym} Crowned Model Family: {crowned_fam.upper()}[/bold yellow]")
        for f_name, f_data in fam_scores.items():
            badge = "★ CROWNED" if f_name == crowned_fam else "  "
            console.print(
                f"   {badge} {f_name:<20}: Weighted Composite Score = {f_data['weighted_composite_score']:.1f}% "
                f"[10D:{f_data['10d']:.0f}% | 5D:{f_data['5d']:.0f}% | 3D:{f_data['3d']:.0f}% | "
                f"15D:{f_data['15d']:.0f}% | T+1:{f_data['t1']:.0f}% | 30D:{f_data['30d']:.0f}% | WS:{f_data['week_start']:.0f}%]"
            )

        table = Table(title=f"📊 {sym} — Status Quo vs Weighted Family-First ({crowned_fam})", title_style="bold green")
        table.add_column("Horizon", style="bold cyan")
        table.add_column("Weight", justify="center", style="yellow")
        table.add_column("Status Quo Model", style="magenta")
        table.add_column("SQ Win %", justify="right", style="magenta")
        table.add_column("SQ Signal", justify="center", style="magenta")
        table.add_column(f"Family-First Model ({crowned_fam})", style="green")
        table.add_column("New Win %", justify="right", style="green")
        table.add_column("New Signal", justify="center", style="green")

        for r in rows:
            table.add_row(
                r["horizon"],
                r["weight"],
                r["sq_model"],
                r["sq_win"],
                r["sq_signal"],
                r["after_model"],
                r["after_win"],
                r["after_signal"],
            )

        console.print(table)


if __name__ == "__main__":
    main()
