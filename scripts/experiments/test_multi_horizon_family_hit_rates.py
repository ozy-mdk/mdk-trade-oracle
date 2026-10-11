"""Experiment: Multi-Horizon Family-First Tournament with Deep Retrospective Hit Rate Audit.

Multi-Horizon Models ONLY (3D, 5D, 10D, 15D, 30D).
Importance Weights: 10D (35%) > 5D (25%) > 3D (20%) > 15D (12%) > 30D (8%).

For each horizon, compares Before (Status Quo) vs After (Family-First) across:
- Last 30 Sessions Hit Rate % & Count
- Last 60 Sessions Hit Rate % & Count
- Last 180 Sessions Hit Rate % & Count
- Live Expected Return % and Stance
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

console = Console()

FAMILIES = {
    "GBDT": ["LightGBM", "XGBoost"],
    "Regularized Linear": ["Ridge", "BayesianRidge"],
    "Robust M-Estimator": ["Huber"],
}

# Multi-Horizon Only Importance Weights: 10D > 5D > 3D > 15D > 30D
HORIZON_WEIGHTS = {
    "10d": 0.35,
    "5d": 0.25,
    "3d": 0.20,
    "15d": 0.12,
    "30d": 0.08,
}

LOOKBACKS = [("3m", 63), ("6m", 126), ("12m", 252), ("18m", 378), ("24m", 504)]
SWING_HORIZONS = ["3d", "5d", "10d", "15d", "30d"]
SYMBOLS = ["ASELS", "AKBNK", "GARAN", "ISCTR", "YKBNK"]


def get_hit_rate_slices(ledger: list[dict]) -> tuple[tuple[int, int, float], tuple[int, int, float], tuple[int, int, float]]:
    """Return (hits, total, pct) for last 30, last 60, and last 180."""
    total = len(ledger)
    # Last 30
    s30 = ledger[-30:] if total >= 30 else ledger
    h30 = sum(1 for r in s30 if r.get("is_hit"))
    n30 = len(s30)
    p30 = (h30 / n30 * 100.0) if n30 > 0 else 0.0

    # Last 60
    s60 = ledger[-60:] if total >= 60 else ledger
    h60 = sum(1 for r in s60 if r.get("is_hit"))
    n60 = len(s60)
    p60 = (h60 / n60 * 100.0) if n60 > 0 else 0.0

    # Last 180 (or all)
    s180 = ledger[-180:] if total >= 180 else ledger
    h180 = sum(1 for r in s180 if r.get("is_hit"))
    n180 = len(s180)
    p180 = (h180 / n180 * 100.0) if n180 > 0 else 0.0

    return (h30, n30, p30), (h60, n60, p60), (h180, n180, p180)


def evaluate_swings_for_symbol(df_feat: pd.DataFrame):
    """Evaluate all candidates across all swing horizons for 60 sessions."""
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
                    })

    return swing_evals


def run_symbol_experiment(db: PostgresManager, symbol: str):
    console.print(f"\n[bold cyan]Processing {symbol}...[/bold cyan]")
    df_raw = extract_3pillar_time_series(db, symbol, lookback_days=1100)
    df_feat, _ = engineer_multi_horizon_features(df_raw)

    # 1. Swings 60-session arena
    swing_evals = evaluate_swings_for_symbol(df_feat)

    # 2. Read Status Quo Configs
    with open("config/tertip_multi_horizon_crowned_models.yaml") as f:
        mh_cfg = yaml.safe_load(f)
        mh_sq = mh_cfg.get("symbols", {}).get(symbol, {})

    # =========================================================================
    # STAGE 1: CROWN MODEL FAMILY USING MULTI-HORIZON MODELS ONLY
    # 10D (0.35) > 5D (0.25) > 3D (0.20) > 15D (0.12) > 30D (0.08)
    # =========================================================================
    family_scores = {}
    for fam_name in FAMILIES.keys():
        fam_h_wins = {}
        for h in SWING_HORIZONS:
            fam_evals = [e for e in swing_evals[h] if e["family"] == fam_name]
            best = max(fam_evals, key=lambda x: (x["weighted_hit_rate_pct"], x["recent_20_hit_rate"]))
            fam_h_wins[h] = best["weighted_hit_rate_pct"]

        score = sum(HORIZON_WEIGHTS[h] * fam_h_wins[h] for h in SWING_HORIZONS)
        family_scores[fam_name] = {
            "score": score,
            "10d": fam_h_wins["10d"],
            "5d": fam_h_wins["5d"],
            "3d": fam_h_wins["3d"],
            "15d": fam_h_wins["15d"],
            "30d": fam_h_wins["30d"],
        }

    crowned_family = max(family_scores.keys(), key=lambda k: family_scores[k]["score"])

    # =========================================================================
    # STAGE 2: INTRA-FAMILY SPECIALISTS & DEEP 180-SESSION AUDIT
    # =========================================================================
    rows = []

    for h in SWING_HORIZONS:
        # SQ info
        sq_info = mh_sq.get(h, {})
        sq_model = sq_info.get("model", "N/A")
        sq_lb = sq_info.get("horizon", "N/A")
        sq_sessions = sq_info.get("training_lookback_sessions", 378)

        # Family-First Specialist
        fam_evals = [e for e in swing_evals[h] if e["family"] == crowned_family]
        best_spec = max(fam_evals, key=lambda x: (x["weighted_hit_rate_pct"], x["recent_20_hit_rate"], -x["mae_pct"]))
        new_model = best_spec["model"]
        new_lb = best_spec["lookback"]
        new_sessions = best_spec["sessions"]

        # Run 180-session walk-forward for SQ Model
        sq_wf = evaluate_horizon_walk_forward(
            df=df_feat,
            horizon_key=h,
            model_name=sq_model,
            train_lookback_sessions=sq_sessions,
            n_eval_sessions=180,
        )
        sq_ledger = sq_wf.get("ledger", [])
        sq_30, sq_60, sq_180 = get_hit_rate_slices(sq_ledger)

        # Run 180-session walk-forward for New Family-First Model
        if new_model == sq_model and new_sessions == sq_sessions:
            new_ledger = sq_ledger
            new_30, new_60, new_180 = sq_30, sq_60, sq_180
        else:
            new_wf = evaluate_horizon_walk_forward(
                df=df_feat,
                horizon_key=h,
                model_name=new_model,
                train_lookback_sessions=new_sessions,
                n_eval_sessions=180,
            )
            new_ledger = new_wf.get("ledger", [])
            new_30, new_60, new_180 = get_hit_rate_slices(new_ledger)

        # Live Signals
        sq_live = generate_live_horizon_forecast(df_feat, h, sq_model, sq_sessions)
        new_live = generate_live_horizon_forecast(df_feat, h, new_model, new_sessions)

        # Deltas
        d30 = new_30[2] - sq_30[2]
        d60 = new_60[2] - sq_60[2]
        d180 = new_180[2] - sq_180[2]

        rows.append({
            "horizon": h.upper(),
            "weight": f"{HORIZON_WEIGHTS[h]*100:.0f}%",
            "sq_model": f"{sq_model} ({sq_lb})",
            "new_model": f"{new_model} ({new_lb})",
            "is_changed": (sq_model != new_model or sq_lb != new_lb),
            # Slices
            "last_30_sq": f"{sq_30[0]}/{sq_30[1]} ({sq_30[2]:.1f}%)",
            "last_30_new": f"{new_30[0]}/{new_30[1]} ({new_30[2]:.1f}%)",
            "delta_30": f"{d30:+.1f}%",
            "last_60_sq": f"{sq_60[0]}/{sq_60[1]} ({sq_60[2]:.1f}%)",
            "last_60_new": f"{new_60[0]}/{new_60[1]} ({new_60[2]:.1f}%)",
            "delta_60": f"{d60:+.1f}%",
            "last_180_sq": f"{sq_180[0]}/{sq_180[1]} ({sq_180[2]:.1f}%)",
            "last_180_new": f"{new_180[0]}/{new_180[1]} ({new_180[2]:.1f}%)",
            "delta_180": f"{d180:+.1f}%",
            # Live Signal
            "sq_signal": f"{sq_live.get('expected_return_pct', 0.0):+.2f}% ({sq_live.get('stance', 'N/A')})",
            "new_signal": f"{new_live.get('expected_return_pct', 0.0):+.2f}% ({new_live.get('stance', 'N/A')})",
        })

    return crowned_family, family_scores, rows


def main():
    db = PostgresManager()
    console.print("\n[bold yellow]" + "=" * 120 + "[/bold yellow]")
    console.print("[bold cyan]MULTI-HORIZON ONLY FAMILY-FIRST EXPERIMENT (10D:35% > 5D:25% > 3D:20% > 15D:12% > 30D:8%)[/bold cyan]")
    console.print("[bold cyan]AUDITING HIT RATE IMPACT ACROSS LAST 30, LAST 60, AND LAST 180 SESSIONS[/bold cyan]")
    console.print("[bold yellow]" + "=" * 120 + "[/bold yellow]")

    for sym in SYMBOLS:
        crowned_fam, fam_scores, rows = run_symbol_experiment(db, sym)

        console.print(f"\n🏆 [bold yellow]{sym} Crowned Multi-Horizon Family: {crowned_fam.upper()}[/bold yellow]")
        for f_name, f_data in fam_scores.items():
            badge = "★ CROWNED" if f_name == crowned_fam else "  "
            console.print(
                f"   {badge} {f_name:<20}: Score = {f_data['score']:.1f}% "
                f"[10D:{f_data['10d']:.1f}% | 5D:{f_data['5d']:.1f}% | 3D:{f_data['3d']:.1f}% | 15D:{f_data['15d']:.1f}% | 30D:{f_data['30d']:.1f}%]"
            )

        # 1. Hit Rate Table: Last 30, Last 60, Last 180
        t_hits = Table(title=f"📊 {sym} — Hit Rate Comparison (Last 30, Last 60, Last 180 Sessions)", title_style="bold green")
        t_hits.add_column("Horizon", style="bold cyan")
        t_hits.add_column("Before Model", style="magenta")
        t_hits.add_column(f"New Model ({crowned_fam})", style="green")
        t_hits.add_column("Last 30 (Old -> New)", justify="center")
        t_hits.add_column("Δ 30D", justify="right", style="bold")
        t_hits.add_column("Last 60 (Old -> New)", justify="center")
        t_hits.add_column("Δ 60D", justify="right", style="bold")
        t_hits.add_column("Last 180 (Old -> New)", justify="center")
        t_hits.add_column("Δ 180D", justify="right", style="bold")

        for r in rows:
            # Color deltas
            def fmt_delta(d_str: str) -> str:
                v = float(d_str.replace("%", "").replace("+", ""))
                if v > 0:
                    return f"[green]{d_str}[/green]"
                elif v < 0:
                    return f"[red]{d_str}[/red]"
                return f"[white]{d_str}[/white]"

            chg_mark = "★ " if r["is_changed"] else ""
            t_hits.add_row(
                r["horizon"],
                r["sq_model"],
                chg_mark + r["new_model"],
                f"{r['last_30_sq']} -> {r['last_30_new']}",
                fmt_delta(r["delta_30"]),
                f"{r['last_60_sq']} -> {r['last_60_new']}",
                fmt_delta(r["delta_60"]),
                f"{r['last_180_sq']} -> {r['last_180_new']}",
                fmt_delta(r["delta_180"]),
            )

        console.print(t_hits)

        # 2. Live Signal & Term-Structure Consistency Table
        t_signals = Table(title=f"🧭 {sym} — Live Signal & Sign Confluence (Before vs After)", title_style="bold yellow")
        t_signals.add_column("Horizon", style="bold cyan")
        t_signals.add_column("Weight", justify="center", style="yellow")
        t_signals.add_column("Before Signal", justify="center", style="magenta")
        t_signals.add_column("New Signal", justify="center", style="green")
        t_signals.add_column("Impact Description", style="white")

        for r in rows:
            impact = "Model Retained" if not r["is_changed"] else f"Changed to {crowned_fam} specialist"
            t_signals.add_row(
                r["horizon"],
                r["weight"],
                r["sq_signal"],
                r["new_signal"],
                impact,
            )

        console.print(t_signals)


if __name__ == "__main__":
    main()
