import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time

import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import evaluate_horizon_walk_forward

CANDIDATE_MODELS = ["LightGBM", "XGBoost", "BayesianRidge", "Ridge", "Huber"]
LOOKBACKS = [
    ("3m", 63),
    ("6m", 126),
    ("12m", 252),
    ("18m", 378),
    ("24m", 504),
]

db = PostgresManager()
df = extract_3pillar_time_series(db, 'AKBNK', lookback_days=1100)

print("Starting evaluation for AKBNK 10d with divergence features...", flush=True)
results = []
for m in CANDIDATE_MODELS:
    for lb_str, lb_s in LOOKBACKS:
        t0 = time.time()
        res = evaluate_horizon_walk_forward(
            df=df,
            horizon_key='10d',
            model_name=m,
            train_lookback_sessions=lb_s,
            n_eval_sessions=60,
        )
        elapsed = time.time() - t0
        hit_rate = res.get('hit_rate_pct', 0.0)
        hits = res.get('hits', 0)
        mae = res.get('mae_pct', 0.0)
        loss = res.get('loss', 999.0)
        print(f"[{m:<14} {lb_str:<4} ({lb_s}s)] Win: {hit_rate:5.1f}% ({hits}/60) | MAE: {mae:5.2f}% | Loss: {loss:6.2f} (took {elapsed:.1f}s)", flush=True)
        results.append({
            'model': m,
            'lookback': lb_str,
            'sessions': lb_s,
            'hit_rate': hit_rate,
            'hits': hits,
            'mae': mae,
            'loss': loss,
        })

print("\n=== TOP 5 CANDIDATES FOR AKBNK 10D ===", flush=True)

res_df = pd.DataFrame(results).sort_values(by='loss', ascending=True)
print(res_df.head(5).to_string(index=False), flush=True)
