"""Benchmark Sequential LSTM, Hybrid Residual Models, and Tabular Champions.

Compares:
1. Prophet Baseline (Zero-leakage univariate prior)
2. Raw Ridge Champion
3. Raw XGBoost Champion
4. Prophet + Ridge Residual
5. Prophet + XGBoost Residual
6. Prophet + LSTM Residual (Seq=5 sessions)
7. Prophet + LSTM Residual (Seq=10 sessions)

Evaluates on out-of-sample walk-forward sessions with:
- Calibrated ±0.25% (25 bps) deadband
- Piecewise volatility sample weighting (0.75x quiet, 2.5x >=1%, 3.0x >=2%)
- Composite tournament loss & computational latency
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    DEADBAND_PCT,
    FEATURE_COLS,
    _check_hit,
    attach_prophet_rolling_features,
    extract_3pillar_time_series,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. PyTorch LSTM Neural Network & Sklearn Wrapper
# ---------------------------------------------------------------------------
class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)
        last_hidden = self.layer_norm(out[:, -1, :])
        last_hidden = self.dropout(last_hidden)
        return self.head(last_hidden).squeeze(-1)


class PyTorchLSTMResidualForecaster:
    """Sequential LSTM forecaster trained on baseline residuals."""

    def __init__(
        self,
        seq_len: int = 5,
        hidden_dim: int = 16,
        dropout: float = 0.25,
        epochs: int = 35,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: _LSTMNet | None = None

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None, weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        N, D = X.shape
        L = self.seq_len
        if N < L:
            raise ValueError(f"Need at least {L} samples, got {N}")

        num_seq = N - L + 1
        X_seq = np.empty((num_seq, L, D), dtype=np.float32)
        for i in range(num_seq):
            X_seq[i] = X[i : i + L]

        if y is not None:
            # y aligns with the last element of the sequence
            y_seq = y[L - 1 :].astype(np.float32)
            w_seq = weights[L - 1 :].astype(np.float32) if weights is not None else np.ones(num_seq, dtype=np.float32)
            return X_seq, y_seq, w_seq
        return X_seq, None, None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> PyTorchLSTMResidualForecaster:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq, w_seq = self._create_sequences(X_scaled, y, sample_weight)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32, device=DEVICE)
        w_tensor = torch.tensor(w_seq, dtype=torch.float32, device=DEVICE)
        # Normalize weights
        w_tensor = w_tensor / w_tensor.mean()

        D = X.shape[1]
        self.model = _LSTMNet(input_dim=D, hidden_dim=self.hidden_dim, dropout=self.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.SmoothL1Loss(reduction="none")

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss_unweighted = criterion(preds, y_tensor)
            loss = (loss_unweighted * w_tensor).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

        return self

    def predict_residual(self, X_recent: np.ndarray) -> float:
        """X_recent has shape (>= seq_len, D). We take the trailing seq_len steps."""
        if self.model is None:
            return 0.0
        X_recent_scaled = self.scaler.transform(X_recent[-self.seq_len :])
        X_seq = np.expand_dims(X_recent_scaled, axis=0)  # [1, L, D]
        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)

        self.model.eval()
        with torch.no_grad():
            res = float(self.model(X_tensor).cpu().numpy()[0])
        # Clip residual to realistic bound [-5%, +5%]
        return float(np.clip(res, -5.0, 5.0))


# ---------------------------------------------------------------------------
# 2. Walk-Forward Evaluation Harness
# ---------------------------------------------------------------------------
def run_model_comparison(
    symbol: str = "ASELS",
    lookback_sessions: int = 126,
    n_eval: int = 30,
) -> pd.DataFrame:
    """Run walk-forward tournament comparing sequential vs tabular models."""
    print("=" * 110)
    print(f"WALK-FORWARD EXPERIMENT: {symbol} | Lookback: {lookback_sessions} sessions | Out-of-Sample: {n_eval} sessions")
    print(f"Acceleration Device: {DEVICE}")
    print("=" * 110)

    db = PostgresManager()
    df = extract_3pillar_time_series(db, symbol)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    df = attach_prophet_rolling_features(df, symbol, train_lookback_sessions=lookback_sessions)
    f_cols = [c for c in FEATURE_COLS if c in df.columns]

    T = len(df)
    models_to_test = [
        "Prophet Baseline",
        "Raw Ridge",
        "Raw XGBoost",
        "Prophet + Ridge Residual",
        "Prophet + XGBoost Residual",
        "Prophet + LSTM Residual (L=5)",
        "Prophet + LSTM Residual (L=10)",
    ]

    records = {m: [] for m in models_to_test}
    latencies = {m: [] for m in models_to_test}

    for i in range(n_eval):
        eval_idx = T - n_eval + i
        eval_row = df.iloc[eval_idx]
        actual_price = float(eval_row["close_price"])
        prev_row = df.iloc[eval_idx - 1]
        prev_price = float(prev_row["close_price"])
        actual_ret = (actual_price - prev_price) / prev_price * 100.0

        train_data = df.iloc[:eval_idx]
        tr_clean = train_data.iloc[-lookback_sessions:].dropna(subset=f_cols).copy()

        # Actual returns on training set
        y_tr_raw = (
            (tr_clean["close_price"].shift(-1) - tr_clean["close_price"]) / tr_clean["close_price"] * 100.0
        ).iloc[:-1]
        X_tr = tr_clean[f_cols].iloc[:-1].to_numpy(dtype=np.float32)

        # Baseline Prophet returns on training set & test session
        prophet_tr = tr_clean["feat_prophet_ret_today_pct"].iloc[:-1].to_numpy(dtype=np.float32)
        prophet_test = float(prev_row["feat_prophet_ret_today_pct"]) if "feat_prophet_ret_today_pct" in prev_row else 0.0

        # Residual target
        y_tr_residual = y_tr_raw.to_numpy(dtype=np.float32) - prophet_tr

        # Sample weights: piecewise 0.75x quiet, 2.5x >=1%, 3.0x >=2%
        abs_y = np.abs(y_tr_raw.to_numpy(dtype=np.float32))
        sample_weights = np.where(abs_y < 1.0, 0.75, np.where(abs_y < 2.0, 2.5, 3.0)).astype(np.float32)

        # Recent context for prediction (up to 15 sessions)
        X_recent = tr_clean[f_cols].iloc[-15:].to_numpy(dtype=np.float32)
        X_prev_single = X_recent[-1:]

        # -------------------------------------------------------------------
        # 1. Prophet Baseline
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        p_ret = float(np.clip(prophet_test, -10.0, 10.0))
        latencies["Prophet Baseline"].append((time.perf_counter() - t0) * 1000)
        records["Prophet Baseline"].append((p_ret, actual_ret))

        # -------------------------------------------------------------------
        # 2. Raw Ridge
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        ridge_raw = Ridge(alpha=10.0, random_state=42)
        ridge_raw.fit(X_tr, y_tr_raw, sample_weight=sample_weights)
        p_ret = float(np.clip(ridge_raw.predict(X_prev_single)[0], -10.0, 10.0))
        latencies["Raw Ridge"].append((time.perf_counter() - t0) * 1000)
        records["Raw Ridge"].append((p_ret, actual_ret))

        # -------------------------------------------------------------------
        # 3. Raw XGBoost
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        xgb_raw = XGBRegressor(
            n_estimators=45,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        xgb_raw.fit(X_tr, y_tr_raw, sample_weight=sample_weights)
        p_ret = float(np.clip(xgb_raw.predict(X_prev_single)[0], -10.0, 10.0))
        latencies["Raw XGBoost"].append((time.perf_counter() - t0) * 1000)
        records["Raw XGBoost"].append((p_ret, actual_ret))

        # -------------------------------------------------------------------
        # 4. Prophet + Ridge Residual
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        ridge_res = Ridge(alpha=10.0, random_state=42)
        ridge_res.fit(X_tr, y_tr_residual, sample_weight=sample_weights)
        res_pred = float(np.clip(ridge_res.predict(X_prev_single)[0], -5.0, 5.0))
        p_ret = float(np.clip(prophet_test + res_pred, -10.0, 10.0))
        latencies["Prophet + Ridge Residual"].append((time.perf_counter() - t0) * 1000)
        records["Prophet + Ridge Residual"].append((p_ret, actual_ret))

        # -------------------------------------------------------------------
        # 5. Prophet + XGBoost Residual
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        xgb_res = XGBRegressor(
            n_estimators=40,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        xgb_res.fit(X_tr, y_tr_residual, sample_weight=sample_weights)
        res_pred = float(np.clip(xgb_res.predict(X_prev_single)[0], -5.0, 5.0))
        p_ret = float(np.clip(prophet_test + res_pred, -10.0, 10.0))
        latencies["Prophet + XGBoost Residual"].append((time.perf_counter() - t0) * 1000)
        records["Prophet + XGBoost Residual"].append((p_ret, actual_ret))

        # -------------------------------------------------------------------
        # 6. Prophet + LSTM Residual (L=5)
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        lstm_5 = PyTorchLSTMResidualForecaster(seq_len=5, hidden_dim=16, epochs=35, lr=0.01)
        lstm_5.fit(X_tr, y_tr_residual, sample_weight=sample_weights)
        res_pred = lstm_5.predict_residual(X_recent)
        p_ret = float(np.clip(prophet_test + res_pred, -10.0, 10.0))
        latencies["Prophet + LSTM Residual (L=5)"].append((time.perf_counter() - t0) * 1000)
        records["Prophet + LSTM Residual (L=5)"].append((p_ret, actual_ret))

        # -------------------------------------------------------------------
        # 7. Prophet + LSTM Residual (L=10)
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        lstm_10 = PyTorchLSTMResidualForecaster(seq_len=10, hidden_dim=16, epochs=35, lr=0.01)
        lstm_10.fit(X_tr, y_tr_residual, sample_weight=sample_weights)
        res_pred = lstm_10.predict_residual(X_recent)
        p_ret = float(np.clip(prophet_test + res_pred, -10.0, 10.0))
        latencies["Prophet + LSTM Residual (L=10)"].append((time.perf_counter() - t0) * 1000)
        records["Prophet + LSTM Residual (L=10)"].append((p_ret, actual_ret))

    # -----------------------------------------------------------------------
    # Aggregate Metrics & Print Table
    # -----------------------------------------------------------------------
    summary = []
    for m in models_to_test:
        rec = records[m]
        df_m = pd.DataFrame(rec, columns=["pred", "actual"])
        hits = sum(_check_hit(p, a, DEADBAND_PCT) for p, a in rec)
        total = len(rec)
        hit_rate = (hits / total) * 100.0
        errors = np.abs(df_m["pred"] - df_m["actual"])
        mae = errors.mean()

        hit_mask = [_check_hit(p, a, DEADBAND_PCT) for p, a in rec]
        hit_mae = errors[hit_mask].mean() if any(hit_mask) else 0.0
        miss_mae = errors[~np.array(hit_mask)].mean() if not all(hit_mask) else 0.0

        # Tournament Loss: (100 - hit_rate) + 1.0 * hit_mae + 2.5 * miss_mae
        loss = (100.0 - hit_rate) + 1.0 * hit_mae + 2.5 * miss_mae

        # Big moves (>= 1.0%)
        sig_df = df_m[np.abs(df_m["actual"]) >= 1.0]
        sig_hits = sum(_check_hit(r["pred"], r["actual"], DEADBAND_PCT) for _, r in sig_df.iterrows())
        sig_rate = (sig_hits / len(sig_df) * 100.0) if len(sig_df) > 0 else 0.0

        avg_lat = np.mean(latencies[m])

        summary.append({
            "Model": m,
            "Hits": f"{hits}/{total}",
            "HitRate%": hit_rate,
            "Sig≥1%": f"{sig_hits}/{len(sig_df)} ({sig_rate:.0f}%)",
            "MAE%": mae,
            "HitMAE%": hit_mae,
            "MissMAE%": miss_mae,
            "Loss": loss,
            "Latency(ms)": avg_lat,
        })

    res_df = pd.DataFrame(summary).sort_values("Loss")

    print("\n" + "=" * 115)
    print(f"{'Model Paradigm':<32} | {'Hits':<7} | {'Hit%':<6} | {'Sig≥1%':<14} | {'MAE%':<6} | {'Loss':<6} | {'Latency':<8}")
    print("-" * 115)
    for _, row in res_df.iterrows():
        print(
            f"{row['Model']:<32} | {row['Hits']:<7} | {row['HitRate%']:>5.1f}% | {row['Sig≥1%']:<14} | {row['MAE%']:>5.2f}% | {row['Loss']:>6.1f} | {row['Latency(ms)']:>6.1f}ms"
        )
    print("=" * 115)
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sequential LSTM vs Tabular Models on Institutional Residuals")
    parser.add_argument("--symbol", type=str, default="ASELS", help="Stock ticker (e.g. ASELS, THYAO)")
    parser.add_argument("--lookback", type=int, default=126, help="Training lookback in sessions (63=3M, 126=6M, 252=12M)")
    parser.add_argument("--eval-sessions", type=int, default=30, help="Number of out-of-sample sessions")
    args = parser.parse_args()

    run_model_comparison(symbol=args.symbol, lookback_sessions=args.lookback, n_eval=args.eval_sessions)
