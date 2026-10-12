"""Test Two-Stage Family Tournament with Option 1: Tiered 30-Day Bucket Recency.

Formula:
Recency_Weighted_Hit_Rate = 0.50 * HitRate(0-30d) + 0.30 * HitRate(30-60d) + 0.20 * HitRate(60-90d)

Horizons: Multi-Horizon ONLY (3D, 5D, 10D, 15D, 30D)
Horizon Weights for Family Selection: 10D (35%) > 5D (25%) > 3D (20%) > 15D (12%) > 30D (8%)

Equities: ASELS, AKBNK, GARAN, ISCTR, YKBNK
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

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


def compute_option1_recency_score(ledger: list[dict]) -> tuple[float, float, float, float]:
    """Compute Option 1 score: 0.50*b1(0-30) + 0.30*b2(30-60) + 0.20*b3(60-90)."""
    n = len(ledger)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Bucket 1: 0 to 30 sessions (most recent)
    b1 = ledger[-30:] if n >= 30 else ledger
    h1 = sum(1 for r in b1 if r.get("is_hit"))
    r1 = (h1 / len(b1) * 100.0) if b1 else 0.0

    # Bucket 2: 30 to 60 sessions
    b2 = ledger[-60:-30] if n >= 60 else (ledger[:-30] if n > 30 else [])
    h2 = sum(1 for r in b2 if r.get("is_hit"))
    r2 = (h2 / len(b2) * 100.0) if b2 else r1

    # Bucket 3: 60 to 90 sessions
    b3 = ledger[-90:-60] if n >= 90 else (ledger[:-60] if n > 60 else [])
    h3 = sum(1 for r in b3 if r.get("is_hit"))
    r3 = (h3 / len(b3) * 100.0) if b3 else r2

    weighted_score = 0.50 * r1 + 0.30 * r2 + 0.20 * r3
    return weighted_score, r1, r2, r3


def evaluate_swings_for_symbol(df_feat: pd.DataFrame):
    """Evaluate all candidates across all swing horizons over 90 sessions."""
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
                        n_eval_sessions=90,
                    )
                    ledger = res.get("ledger", [])
                    opt1_score, r1, r2, r3 = compute_option1_recency_score(ledger)

                    swing_evals[h].append({
                        "family": fam_name,
                        "model": m,
                        "lookback": lb_name,
                        "sessions": lb_s,
                        "opt1_score": opt1_score,
                        "r_0_30": r1,
                        "r_30_60": r2,
                        "r_60_90": r3,
                        "hit_rate_pct": res.get("hit_rate_pct", 0.0),
                        "mae_pct": res.get("mae_pct", 0.0),
                    })

    return swing_evals


def run_symbol_experiment(db: PostgresManager, symbol: str):
    console.print(f"\n[bold cyan]Processing {symbol}...[/bold cyan]")
    df_raw = extract_3pillar_time_series(db, symbol, lookback_days=1100)
    df_feat, _ = engineer_multi_horizon_features(df_raw)

    # 1. 90-session swing arena
    swing_evals = evaluate_swings_for_symbol(df_feat)

    # 2. Status Quo Configs
    with open("config/tertip_multi_horizon_crowned_models.yaml") as f:
        mh_cfg = yaml.safe_load(f)
        mh_sq = mh_cfg.get("symbols", {}).get(symbol, {})

    # =========================================================================
    # STAGE 1: CROWN MODEL FAMILY USING OPTION 1 RECENCY SCORES
    # Family Score = sum(horizon_weight * max_opt1_score_in_family)
    # =========================================================================
    family_scores = {}
    for fam_name in FAMILIES.keys():
        fam_h_scores = {}
        for h in SWING_HORIZONS:
            fam_evals = [e for e in swing_evals[h] if e["family"] == fam_name]
            best = max(fam_evals, key=lambda x: (x["opt1_score"], x["r_0_30"], -x["mae_pct"]))
            fam_h_scores[h] = best["opt1_score"]

        score = sum(HORIZON_WEIGHTS[h] * fam_h_scores[h] for h in SWING_HORIZONS)
        family_scores[fam_name] = {
            "score": score,
            "10d": fam_h_scores["10d"],
            "5d": fam_h_scores["5d"],
            "3d": fam_h_scores["3d"],
            "15d": fam_h_scores["15d"],
            "30d": fam_h_scores["30d"],
        }

    crowned_family = max(family_scores.keys(), key=lambda k: family_scores[k]["score"])

    # =========================================================================
    # STAGE 2: INTRA-FAMILY SPECIALISTS
    # =========================================================================
    rows = []
    for h in SWING_HORIZONS:
        sq_info = mh_sq.get(h, {})
        sq_model = sq_info.get("model", "N/A")
        sq_lb = sq_info.get("horizon", "N/A")
        sq_sessions = sq_info.get("training_lookback_sessions", 378)

        # Specialist within crowned family
        fam_evals = [e for e in swing_evals[h] if e["family"] == crowned_family]
        best_spec = max(fam_evals, key=lambda x: (x["opt1_score"], x["r_0_30"], -x["mae_pct"]))
        new_model = best_spec["model"]
        new_lb = best_spec["lookback"]
        new_sessions = best_spec["sessions"]

        # SQ evaluate 90 sessions for comparison
        sq_wf = evaluate_horizon_walk_forward(
            df=df_feat,
            horizon_key=h,
            model_name=sq_model,
            train_lookback_sessions=sq_sessions,
            n_eval_sessions=90,
        )
        sq_opt1, sq_r1, sq_r2, sq_r3 = compute_option1_recency_score(sq_wf.get("ledger", []))

        # Live Signals
        sq_live = generate_live_horizon_forecast(df_feat, h, sq_model, sq_sessions)
        new_live = generate_live_horizon_forecast(df_feat, h, new_model, new_sessions)

        rows.append({
            "horizon": h.upper(),
            "weight": f"{HORIZON_WEIGHTS[h]*100:.0f}%",
            "sq_model": f"{sq_model} ({sq_lb})",
            "new_model": f"{new_model} ({new_lb})",
            "is_changed": (sq_model != new_model or sq_lb != new_lb),
            "opt1_score_sq": f"{sq_opt1:.1f}%",
            "opt1_score_new": f"{best_spec['opt1_score']:.1f}%",
            "delta_score": f"{best_spec['opt1_score'] - sq_opt1:+.1f}%",
            "b1_0_30": f"{sq_r1:.1f}% -> {best_spec['r_0_30']:.1f}% ({best_spec['r_0_30'] - sq_r1:+.1f}%)",
            "b2_30_60": f"{sq_r2:.1f}% -> {best_spec['r_30_60']:.1f}% ({best_spec['r_30_60'] - sq_r2:+.1f}%)",
            "b3_60_90": f"{sq_r3:.1f}% -> {best_spec['r_60_90']:.1f}% ({best_spec['r_60_90'] - sq_r3:+.1f}%)",
            "sq_signal": f"{sq_live.get('expected_return_pct', 0.0):+.2f}% ({sq_live.get('stance', 'N/A')})",
            "new_signal": f"{new_live.get('expected_return_pct', 0.0):+.2f}% ({new_live.get('stance', 'N/A')})",
        })

    return crowned_family, family_scores, rows


def main():
    db = PostgresManager()
    console.print("\n[bold yellow]" + "=" * 125 + "[/bold yellow]")
    console.print("[bold cyan]OPTION 1 TIERED 30-DAY BUCKET RECENCY TOURNAMENT (0.50 * 0-30d + 0.30 * 30-60d + 0.20 * 60-90d)[/bold cyan]")
    console.print("[bold cyan]MULTI-HORIZON ONLY (10D: 35% > 5D: 25% > 3D: 20% > 15D: 12% > 30D: 8%)[/bold cyan]")
    console.print("[bold yellow]" + "=" * 125 + "[/bold yellow]")

    for sym in SYMBOLS:
        crowned_fam, fam_scores, rows = run_symbol_experiment(db, sym)

        console.print(f"\n🏆 [bold yellow]{sym} Crowned Multi-Horizon Family: {crowned_fam.upper()}[/bold yellow]")
        for f_name, f_data in fam_scores.items():
            badge = "★ CROWNED" if f_name == crowned_fam else "  "
            console.print(
                f"   {badge} {f_name:<20}: Option 1 Score = {f_data['score']:.1f}% "
                f"[10D:{f_data['10d']:.1f}% | 5D:{f_data['5d']:.1f}% | 3D:{f_data['3d']:.1f}% | 15D:{f_data['15d']:.1f}% | 30D:{f_data['30d']:.1f}%]"
            )

        table = Table(title=f"📊 {sym} — Option 1 Recency Breakdown & Hit Rates", title_style="bold green")
        table.add_column("Horizon", style="bold cyan")
        table.add_column("Before Model", style="magenta")
        table.add_column(f"New Model ({crowned_fam})", style="green")
        table.add_column("Option 1 Score (Old -> New)", justify="center")
        table.add_column("Δ Score", justify="right", style="bold")
        table.add_column("Bucket 1: 0-30d (50% Wt)", justify="center")
        table.add_column("Bucket 2: 30-60d (30% Wt)", justify="center")
        table.add_column("Bucket 3: 60-90d (20% Wt)", justify="center")
        table.add_column("New Signal", justify="center", style="yellow")

        for r in rows:
            def fmt_delta(d_str: str) -> str:
                v = float(d_str.replace("%", "").replace("+", ""))
                if v > 0:
                    return f"[green]{d_str}[/green]"
                elif v < 0:
                    return f"[red]{d_str}[/red]"
                return f"[white]{d_str}[/white]"

            chg_mark = "★ " if r["is_changed"] else ""
            table.add_row(
                r["horizon"],
                r["sq_model"],
                chg_mark + r["new_model"],
                f"{r['opt1_score_sq']} -> {r['opt1_score_new']}",
                fmt_delta(r["delta_score"]),
                r["b1_0_30"],
                r["b2_30_60"],
                r["b3_60_90"],
                r["new_signal"],
            )

        console.print(table)


if __name__ == "__main__":
    main()
