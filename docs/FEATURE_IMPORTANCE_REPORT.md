# Feature Importance Report — Crowned Models, Market Regime, Pruning

> **Superseded (2026-10-25):** computed on silver whose FIFO history had been corrupted by an incremental silver run
> (see RESEARCH_HANDOFF §0.8, data incident). FIFO-based feature numbers are invalid; see the tournament-variant
> results on the repaired data.

Focus six (ASELS, GARAN, AKBNK, YKBNK, ISCTR, TTKOM) + THYAO · horizons 1D, 3D, 5D, 10D, 15D, 30D · data through
2026-10-23 · experiment `scripts/experiments/feature_importance_experiment.py` (read-only, outputs under
`~/data/mdk_oracle/experiments/feature_importance/`). Every number is point-in-time; "beats both" means a higher hit
rate than both always-UP and always-DOWN on the same sessions with the same hit rule.

## Summary

1. **The models are mean-reversion machines.** The features they weight most — distance of price from its SMA50 /
   SMA20, BIG5 / MLB FIFO cost spreads, YTD PnL shares, institutional inventory shares — all carry *negative* IC with
   the forward return: the further a stock falls below its averages or below the desks' cost basis, the more bullish
   the forecast. That is exactly why THYAO 10D called +4% to +10% every session from 09-15 to 10-08 while the stock and the
   BIST 30 (-19.6% from 09-11 to 10-23) kept falling.
2. **BIST 30 regime features do not fix it.** Market momentum, distance from SMA20/50, drawdown, volatility and breadth
   also have negative IC (the market mean-reverts too), and their effect flips between periods: market-only models
   score 61% at 30D in the last 90 sessions but were 14 pp behind the better constant call at 30D (23 pp at 15D) in
   the discovery period.
   Adding them to the production features lowered hit rates at every horizon (table C1).
3. **Neither trend-following nor contrarian rules are stable.** Trend rules lose 15-25 pp to the better constant in the
   last 90 sessions (the market rallied into mid-September, so trend said UP into the crash); contrarian rules win at
   10D-30D now but lose in discovery (table C2).
4. **Pruning non-informative features helps fixed models** (+1 to +2 pp at 3D-15D in the last 90 sessions, features
   chosen on the earlier discovery window only), **but not the re-crowned tournament**: re-crowning on the pruned sets
   was worse than production at 10D-30D (table C3). Production stays as it is.
5. **Horizons do want different features**, and the data agrees with intuition: the only stable short-horizon signal is
   the daily BofA (MLB) flow family (IC +0.05 to +0.09 at 1D in both windows, fading to 0 by 10D); 10D-30D are driven by
   price-trend / price-level variables whose IC grows with the horizon.
6. **Nothing tested beats both constant calls on the last 90 sessions across all seven stocks.** Always-DOWN won 55-60%
   in this falling market. The crowned models still beat both at 15D / 30D on the focus stocks, but on 3-9 independent
   outcomes per stock.

## THYAO 10D — why it kept saying UP

| Window | Champion in force | Calls UP | Market fell (10D) | Hit rate |
|---|---|---|---|---|
| Last 90 graded sessions | BayesianRidge, 63-session window (Jul-Sep crowns), then BayesianRidge/378 (Oct) | 43% | 58% of sessions | 40% |

From 09-17 to 09-30 the forecast sat at +9% to +10% while realized 10D returns were -0.4% to -6.2%.
The live model's largest weights are `big5_cost_spread_pct`, `mlb_tertip_share`, `big5_tertip_share`,
`price_vs_sma50_pct`: after the 09-16 crash (BIST 30 -5.6% in a day) THYAO sat far below its SMA50 and below the BIG5
desks' cost, which these features translate into "rebound ahead". A 63-session training window also made the
September champions overfit the June-August rally. The October champion (BayesianRidge/378) is now BEARISH (-4.5%).

## 1. Crowned models (October 2026 champions)

*In-window hit* = the champion refitted point-in-time on the last 90 graded sessions. Those sessions largely overlap
the window it was crowned on, so this number is inflated by selection; it describes what the selected model leaned
on, not its skill. *Honest track hit* = the 90-session gold track, each session graded with the champion that was in
force that month. *Native share* = standardized |coefficient| (linear) or gain (trees) of the live fit. The number in
brackets is the **honest** permutation importance: drop in honest-track hit rate (pp) when the feature is shuffled
across that month's test sessions (positive = the feature helped out of sample, negative = it hurt).

#### ASELS

| Horizon | October champion | In-window hit | Honest track hit | Top 5 features of the live model (native share, honest permutation drop in pp) |
|---|---|---|---|---|
| 1D | Ridge/378/YTD_SUBSTITUTION | 59% | 54% | `kamu_ytd_pnl_share_20d_to` 13% (+0.9); `mlb_turnover_intensity` 11% (+2.3); `inst_ytd_pnl_divergence` 9% (+0.1); `bofa_aggression_ratio` 8% (+1.4); `kamu_tertip_share` 5% (-0.8) |
| 3D | BayesianRidge/252/FIFO_PLUS_YTD | 51% | 52% | `inst_ytd_pnl_divergence` 17% (-3.7); `big5_ytd_pnl_share_20d_to` 16% (+6.0); `kamu_tertip_share` 14% (-1.2); `mlb_ytd_pnl_share_20d_to` 11% (+3.6); `mlb_tertip_share` 6% (-0.1) |
| 5D | BayesianRidge/378/FIFO_PLUS_YTD | 69% | 44% | `inst_ytd_pnl_divergence` 19% (-1.1); `big5_ytd_pnl_share_20d_to` 18% (+2.5); `kamu_tertip_share` 13% (-0.8); `mlb_ytd_pnl_share_20d_to` 13% (+4.4); `big5_cost_spread_pct` 5% (+0.3) |
| 10D | BayesianRidge/504/FIFO_PLUS_YTD | 66% | 54% | `kamu_tertip_share` 21% (+3.4); `mlb_tertip_share` 13% (-7.7); `mlb_ytd_flow_share_20d_to` 11% (-0.1); `big5_ytd_pnl_share_20d_to` 9% (-1.7); `big5_cost_spread_pct` 8% (+1.8) |
| 15D | BayesianRidge/126/FIFO_PLUS_YTD | 76% | 66% | `mlb_ytd_flow_share_20d_to` 22% (+5.1); `big5_ytd_pnl_share_20d_to` 14% (+7.9); `mlb_cost_spread_pct` 6% (+2.9); `mlb_ytd_pnl_share_20d_to` 5% (+2.4); `big5_cost_spread_pct` 5% (+1.7) |
| 30D | BayesianRidge/126/FIFO_CORE | 68% | 67% | `mlb_cost_spread_pct` 24% (-0.2); `big5_cost_spread_pct` 18% (+0.2); `mlb_tertip_share` 9% (+1.7); `kamu_cost_spread_pct` 6% (-3.2); `kamu_tertip_share` 3% (+3.2) |

#### GARAN

| Horizon | October champion | In-window hit | Honest track hit | Top 5 features of the live model (native share, honest permutation drop in pp) |
|---|---|---|---|---|
| 1D | LightGBM/63/FIFO_CORE | 57% | 56% | `ret_yesterday_pct` 56% (+3.2); `big5_cost_spread_pct` 11% (+1.2); `mlb_tertip_share` 9% (+4.2); `big5_buy_share_today` 8% (+2.8); `mlb_buy_share_today` 6% (-1.1) |
| 3D | BayesianRidge/63/FIFO_PLUS_YTD | 57% | 57% | `kamu_tertip_share` 15% (-0.2); `big5_ytd_pnl_share_20d_to` 12% (+3.8); `mlb_profitable_tertip_share` 11% (-2.2); `big5_tertip_share` 9% (+1.5); `big5_pnl_share_today` 9% (+2.4) |
| 5D | Ridge/126/FIFO_PLUS_YTD | 53% | 51% | `inst_ytd_pnl_divergence` 10% (-8.3); `big5_tertip_share` 5% (-0.4); `price_vs_sma50_pct` 5% (-0.2); `mlb_ytd_pnl_share_20d_to` 5% (+5.2); `mlb_tertip_share` 5% (+1.6) |
| 10D | BayesianRidge/63/YTD_SUBSTITUTION | 72% | 70% | `big5_ytd_pnl_share_20d_to` 15% (+0.8); `mlb_tertip_share` 15% (+0.1); `kamu_tertip_share` 10% (-0.3); `inst_ytd_pnl_divergence` 7% (-0.3); `big5_tertip_share` 6% (+1.3) |
| 15D | Ridge/126/FIFO_PLUS_YTD | 73% | 72% | `big5_ytd_pnl_share_20d_to` 8% (+6.6); `kamu_ytd_pnl_share_20d_to` 7% (-0.7); `big5_cost_spread_pct` 6% (+0.0); `mlb_tertip_share` 6% (+6.8); `inst_ytd_pnl_divergence` 5% (+0.2) |
| 30D | Ridge/378/FIFO_CORE | 69% | 69% | `price_vs_sma50_pct` 22% (-0.3); `price_vs_sma20_pct` 17% (+5.4); `mlb_cost_spread_pct` 9% (+14.6); `sma20_vs_sma50_pct` 7% (+3.3); `kamu_cost_spread_pct` 6% (+5.2) |

#### AKBNK

| Horizon | October champion | In-window hit | Honest track hit | Top 5 features of the live model (native share, honest permutation drop in pp) |
|---|---|---|---|---|
| 1D | Ridge/126/YTD_SUBSTITUTION | 59% | 54% | `mlb_ytd_flow_share_20d_to` 15% (+0.6); `mlb_tertip_share` 10% (+1.4); `kamu_tertip_share` 6% (+0.8); `total_inst_net_share_today` 6% (+0.9); `kamu_ytd_pnl_share_20d_to` 5% (+1.2) |
| 3D | Huber/126/YTD_SUBSTITUTION | 61% | 50% | `mlb_ytd_flow_share_20d_to` 21% (+0.6); `mlb_tertip_share` 12% (+0.1); `kamu_tertip_share` 7% (+1.3); `stock_vol_20d` 4% (+1.7); `price_vs_sma50_pct` 4% (+2.6) |
| 5D | Huber/126/FIFO_PLUS_YTD | 63% | 46% | `mlb_ytd_flow_share_20d_to` 23% (-1.2); `mlb_tertip_share` 13% (-1.8); `kamu_tertip_share` 6% (-2.3); `inst_ytd_pnl_divergence` 5% (+0.9); `kamu_ytd_pnl_share_20d_to` 4% (+0.6) |
| 10D | Huber/126/YTD_SUBSTITUTION | 81% | 61% | `mlb_ytd_flow_share_20d_to` 11% (-0.6); `price_vs_sma50_pct` 10% (-0.1); `sma20_vs_sma50_pct` 7% (+2.8); `inst_ytd_pnl_divergence` 6% (+2.8); `kamu_tertip_share` 4% (+0.1) |
| 15D | Huber/126/FIFO_CORE | 67% | 66% | `mlb_cost_spread_pct` 9% (+nan); `mlb_tertip_share` 9% (+4.9); `sma20_vs_sma50_pct` 7% (+6.3); `price_vs_sma50_pct` 7% (-6.8); `big5_tertip_share` 5% (-1.6) |
| 30D | Huber/63/YTD_SUBSTITUTION | 82% | 67% | `close_ret_20d` 9% (-0.3); `price_vs_sma50_pct` 8% (-0.4); `close_ret_5d` 6% (-0.0); `sma10_vs_sma20_pct` 6% (-0.3); `inst_ytd_pnl_divergence` 5% (-0.0) |

#### YKBNK

| Horizon | October champion | In-window hit | Honest track hit | Top 5 features of the live model (native share, honest permutation drop in pp) |
|---|---|---|---|---|
| 1D | BayesianRidge/504/YTD_SUBSTITUTION | 63% | 54% | `big5_ytd_pnl_share_20d_to` 7% (+3.9); `bollinger_bandwidth_pct` 6% (+0.7); `days_since_neg_shock` 6% (+0.2); `kamu_buy_share_today` 6% (+3.9); `total_inst_net_share_today` 5% (-2.3) |
| 3D | BayesianRidge/252/FIFO_CORE | 61% | 52% | `big5_profitable_tertip_share` 14% (+4.7); `mlb_tertip_share` 13% (-1.1); `bollinger_bandwidth_pct` 11% (+1.5); `price_vs_sma50_pct` 6% (+2.0); `mlb_profitable_tertip_share` 5% (+0.0) |
| 5D | BayesianRidge/252/FIFO_CORE | 60% | 63% | `mlb_tertip_share` 12% (-3.1); `bollinger_bandwidth_pct` 8% (+0.0); `big5_profitable_tertip_share` 8% (+1.7); `price_vs_sma50_pct` 8% (+3.3); `kamu_tertip_share` 7% (+1.0) |
| 10D | BayesianRidge/252/FIFO_CORE | 74% | 69% | `mlb_tertip_share` 13% (+0.3); `big5_profitable_tertip_share` 8% (+1.7); `big5_cost_spread_pct` 6% (+0.9); `price_vs_sma50_pct` 6% (+1.4); `kamu_tertip_share` 5% (-0.0) |
| 15D | BayesianRidge/252/FIFO_PLUS_YTD | 83% | 63% | `big5_ytd_pnl_share_20d_to` 20% (+7.8); `mlb_tertip_share` 9% (+1.7); `mlb_ytd_pnl_share_20d_to` 8% (-0.3); `kamu_tertip_share` 6% (+0.0); `stock_vol_10d` 5% (-0.6) |
| 30D | BayesianRidge/63/YTD_SUBSTITUTION | 81% | 43% | `inst_ytd_pnl_divergence` 15% (+0.0); `big5_tertip_share` 11% (+0.1); `big5_ytd_pnl_share_20d_to` 8% (+0.0); `close_ret_20d` 7% (+1.1); `kamu_tertip_share` 7% (+0.5) |

#### ISCTR

| Horizon | October champion | In-window hit | Honest track hit | Top 5 features of the live model (native share, honest permutation drop in pp) |
|---|---|---|---|---|
| 1D | Ridge/63/YTD_SUBSTITUTION | 52% | 51% | `kamu_tertip_share` 12% (-0.8); `mlb_tertip_share` 11% (-1.0); `kamu_w5_share` 10% (+0.3); `inst_ytd_pnl_divergence` 8% (+2.1); `big5_turnover_intensity` 7% (+2.7) |
| 3D | LightGBM/63/YTD_SUBSTITUTION | 49% | 40% | `price_vs_sma50_pct` 29% (-2.9); `big5_tertip_share` 18% (-0.7); `inst_flow_divergence_5d_10d` 15% (-1.8); `days_since_pos_shock` 12% (-0.3); `kamu_buy_share_today` 8% (+0.6) |
| 5D | XGBoost/126/YTD_SUBSTITUTION | 62% | 41% | `mlb_ytd_pnl_share_20d_to` 6% (+2.4); `tertip_inventory_zscore` 5% (-0.9); `price_vs_sma50_pct` 5% (-8.9); `close_ret_20d` 5% (-0.2); `big5_tertip_share` 5% (+0.1) |
| 10D | LightGBM/252/FIFO_PLUS_YTD | 60% | 41% | `big5_cost_spread_pct` 22% (-0.6); `big5_tertip_share` 17% (-1.2); `sma20_vs_sma50_pct` 9% (-0.1); `stock_vol_20d` 8% (-2.0); `bollinger_bandwidth_pct` 7% (-1.8) |
| 15D | XGBoost/126/YTD_SUBSTITUTION | 76% | 33% | `kamu_ytd_pnl_share_20d_to` 35% (+0.6); `mlb_ytd_pnl_share_20d_to` 12% (+4.8); `big5_ytd_pnl_share_20d_to` 6% (+6.8); `mlb_tertip_share` 5% (-0.2); `mlb_ytd_flow_share_20d_to` 4% (+0.2) |
| 30D | LightGBM/252/YTD_SUBSTITUTION | 72% | 68% | `inst_ytd_pnl_divergence` 36% (+0.7); `big5_ytd_pnl_share_20d_to` 13% (+0.6); `kamu_ytd_pnl_share_20d_to` 12% (+0.0); `stock_vol_20d` 9% (-0.1); `sma20_vs_sma50_pct` 8% (+2.4) |

#### TTKOM

| Horizon | October champion | In-window hit | Honest track hit | Top 5 features of the live model (native share, honest permutation drop in pp) |
|---|---|---|---|---|
| 1D | LightGBM/126/YTD_SUBSTITUTION | 52% | 52% | `total_inst_net_share_today` 22% (+2.0); `kamu_buy_share_today` 17% (+0.7); `kamu_tertip_share` 12% (-0.5); `kamu_turnover_intensity` 12% (+1.4); `ret_3d_cum_pct` 7% (-0.2) |
| 3D | Ridge/378/FIFO_PLUS_YTD | 47% | 50% | `price_vs_sma50_pct` 22% (-3.0); `sma20_vs_sma50_pct` 14% (-3.1); `price_vs_sma20_pct` 9% (+1.6); `price_vs_sma10_pct` 8% (+0.1); `mlb_tertip_share` 6% (+2.3) |
| 5D | Ridge/252/FIFO_CORE | 46% | 41% | `price_vs_sma50_pct` 19% (-7.1); `price_vs_sma20_pct` 10% (-3.2); `sma20_vs_sma50_pct` 10% (-1.9); `mlb_tertip_share` 4% (-1.4); `kamu_cost_spread_pct` 4% (+3.9) |
| 10D | Ridge/378/FIFO_PLUS_YTD | 63% | 50% | `price_vs_sma50_pct` 18% (-0.6); `sma20_vs_sma50_pct` 11% (-0.1); `price_vs_sma10_pct` 9% (-1.1); `kamu_ytd_pnl_share_20d_to` 8% (-0.1); `mlb_tertip_share` 5% (-2.9) |
| 15D | Ridge/378/YTD_SUBSTITUTION | 74% | 42% | `price_vs_sma50_pct` 16% (-1.4); `sma20_vs_sma50_pct` 12% (+0.9); `price_vs_sma10_pct` 10% (+1.6); `mlb_ytd_flow_share_20d_to` 9% (-0.3); `mlb_tertip_share` 7% (+0.4) |
| 30D | Ridge/378/YTD_SUBSTITUTION | 56% | 33% | `price_vs_sma20_pct` 15% (+0.3); `price_vs_sma10_pct` 11% (+1.1); `mlb_ytd_flow_share_20d_to` 10% (-7.3); `sma10_vs_sma20_pct` 9% (-2.6); `mlb_tertip_share` 8% (+1.7) |

#### THYAO

| Horizon | October champion | In-window hit | Honest track hit | Top 5 features of the live model (native share, honest permutation drop in pp) |
|---|---|---|---|---|
| 1D | XGBoost/504/YTD_SUBSTITUTION | 57% | 48% | `big5_w5_share` 5% (-0.1); `total_inst_net_share_today` 5% (-3.3); `volatility_pinch_5d_20d` 4% (-0.8); `mlb_ytd_pnl_share_20d_to` 4% (+1.1); `ret_today_pct` 4% (-2.3) |
| 3D | BayesianRidge/504/YTD_SUBSTITUTION | 70% | 60% | `mlb_tertip_share` 20% (-0.2); `mlb_ytd_flow_share_20d_to` 15% (+3.4); `big5_tertip_share` 11% (+1.9); `inst_ytd_pnl_divergence` 7% (+0.5); `mlb_ytd_pnl_share_20d_to` 7% (+1.4) |
| 5D | BayesianRidge/378/FIFO_PLUS_YTD | 72% | 53% | `mlb_tertip_share` 23% (-3.4); `inst_ytd_pnl_divergence` 10% (-0.8); `mlb_ytd_flow_share_20d_to` 10% (-2.4); `kamu_profitable_tertip_share` 8% (-1.0); `mlb_profitable_tertip_share` 7% (+1.3) |
| 10D | BayesianRidge/378/FIFO_CORE | 62% | 40% | `big5_cost_spread_pct` 13% (-2.4); `mlb_tertip_share` 13% (-5.4); `big5_tertip_share` 10% (-4.6); `price_vs_sma50_pct` 7% (-3.8); `mlb_cost_spread_pct` 5% (-0.1) |
| 15D | Ridge/378/FIFO_CORE | 71% | 60% | `price_vs_sma50_pct` 17% (-5.4); `price_vs_sma10_pct` 13% (-2.9); `price_vs_sma20_pct` 11% (-1.3); `sma10_vs_sma20_pct` 9% (+0.9); `big5_cost_spread_pct` 5% (+0.3) |
| 30D | BayesianRidge/378/FIFO_CORE | 74% | 60% | `mlb_cost_spread_pct` 14% (+2.3); `big5_cost_spread_pct` 14% (+2.7); `price_vs_sma50_pct` 13% (+2.3); `mlb_tertip_share` 9% (-0.5); `sma20_vs_sma50_pct` 8% (+0.0) |

### Features that helped / hurt out of sample (honest permutation, 42 champions)

Helped most:

| Feature | Mean honest drop (pp) | Helps in % of models | Hurts in % of models |
|---|---|---|---|
| `big5_ytd_pnl_share_20d_to` | +1.35 | 44% | 22% |
| `sma20_vs_sma50_pct` | +1.17 | 57% | 14% |
| `price_vs_sma20_pct` | +1.14 | 49% | 34% |
| `big5_cost_spread_pct` | +1.01 | 46% | 15% |
| `mlb_ytd_pnl_share_20d_to` | +0.97 | 46% | 20% |
| `price_vs_sma10_pct` | +0.43 | 40% | 37% |
| `big5_profitable_tertip_share` | +0.41 | 31% | 15% |
| `bollinger_bandwidth_pct` | +0.37 | 38% | 17% |
| `kamu_ytd_pnl_share_20d_to` | +0.37 | 41% | 27% |
| `close_ret_20d` | +0.36 | 43% | 26% |

Hurt most (shuffling them *raised* the honest hit rate):

| Feature | Mean honest drop (pp) | Helps in % of models | Hurts in % of models |
|---|---|---|---|
| `price_vs_sma50_pct` | -2.21 | 26% | 54% |
| `stock_vol_20d` | -0.97 | 20% | 49% |
| `sma10_vs_sma20_pct` | -0.66 | 31% | 43% |
| `mlb_flow_divergence_3d_10d` | -0.47 | 11% | 49% |
| `total_inst_5d_share` | -0.46 | 17% | 46% |
| `inst_ytd_pnl_divergence` | -0.40 | 27% | 34% |
| `kamu_turnover_intensity` | -0.37 | 19% | 33% |
| `tertip_inventory_zscore` | -0.35 | 26% | 38% |
| `kamu_post_shock_defense` | -0.35 | 19% | 36% |
| `kamu_tertip_share` | -0.32 | 24% | 38% |

`price_vs_sma50_pct` is the single most harmful feature: heavily weighted, it hurt in 54% of the models that use it.

## 2. Feature x horizon information (rank IC with the forward N-day return, 7 stocks pooled)

Discovery = origins from 2025-01 whose outcome is realized before the evaluation window starts; evaluation = last 90
graded origins. Long horizons overlap heavily (a 30D label shares 29 days with the next one), so 15D / 30D ICs rest
on very few independent outcomes, and price-level features carry part of their within-window IC mechanically
(handoff §0.3).

Discovery:

| Group | Feature | 1D | 3D | 5D | 10D | 15D | 30D |
|---|---|---|---|---|---|---|---|
| Price level / mean reversion | `price_vs_sma50_pct` | -0.03 | -0.06 | -0.10 | -0.15 | -0.19 | -0.27 |
|  | `price_vs_sma20_pct` | -0.01 | -0.02 | -0.06 | -0.11 | -0.14 | -0.11 |
|  | `big5_cost_spread_pct` | -0.07 | -0.12 | -0.18 | -0.27 | -0.35 | -0.47 |
|  | `mlb_cost_spread_pct` | +0.01 | +0.03 | +0.03 | +0.00 | -0.01 | -0.09 |
|  | `big5_ytd_pnl_share_20d_to` | -0.10 | -0.15 | -0.20 | -0.28 | -0.33 | -0.39 |
| Stock momentum | `close_ret_20d` | -0.01 | -0.04 | -0.07 | -0.14 | -0.13 | -0.14 |
|  | `close_ret_10d` | -0.03 | -0.04 | -0.08 | -0.09 | -0.13 | -0.09 |
|  | `sma20_vs_sma50_pct` | -0.03 | -0.08 | -0.10 | -0.14 | -0.19 | -0.27 |
|  | `ret_rollover_5d_20d` | +0.01 | +0.04 | +0.07 | +0.11 | +0.12 | +0.12 |
| Institutional inventory | `mlb_tertip_share` | -0.05 | -0.09 | -0.10 | -0.14 | -0.13 | -0.24 |
|  | `kamu_tertip_share` | +0.07 | +0.11 | +0.14 | +0.16 | +0.17 | +0.28 |
|  | `inst_ytd_pnl_divergence` | +0.07 | +0.10 | +0.14 | +0.20 | +0.24 | +0.28 |
| Daily MLB flow | `mlb_turnover_intensity` | +0.06 | +0.03 | +0.03 | +0.02 | +0.00 | -0.00 |
|  | `bofa_aggression_ratio` | +0.05 | +0.02 | +0.02 | +0.01 | -0.00 | -0.00 |
|  | `mlb_ewma_flow_share_5d` | +0.04 | +0.02 | -0.00 | -0.02 | -0.02 | -0.02 |
|  | `mlb_buy_share_today` | +0.05 | +0.03 | +0.04 | +0.04 | +0.03 | -0.01 |
| Volatility | `stock_vol_20d` | -0.05 | -0.11 | -0.14 | -0.18 | -0.17 | -0.09 |
|  | `bollinger_bandwidth_pct` | -0.08 | -0.12 | -0.15 | -0.22 | -0.21 | -0.10 |
| Market regime (new) | `mkt:bist30_ret_20d` | -0.02 | -0.06 | -0.10 | -0.21 | -0.16 | -0.17 |
|  | `mkt:bist30_ret_60d` | -0.05 | -0.09 | -0.13 | -0.22 | -0.26 | -0.41 |
|  | `mkt:bist30_vs_sma50` | -0.05 | -0.10 | -0.14 | -0.23 | -0.21 | -0.25 |
|  | `mkt:bist30_drawdown_60d` | -0.02 | -0.04 | -0.08 | -0.19 | -0.20 | -0.26 |
|  | `mkt:bist30_vol_20d` | -0.06 | -0.10 | -0.15 | -0.25 | -0.23 | -0.13 |
|  | `mkt:breadth_above_sma20` | -0.01 | -0.05 | -0.09 | -0.18 | -0.15 | -0.13 |
|  | `mkt:mlb_flow_share_5d` | +0.04 | +0.04 | +0.05 | -0.00 | +0.04 | +0.05 |
|  | `mkt:rel_strength_20d` | +0.02 | +0.01 | +0.00 | -0.01 | -0.06 | -0.02 |
|  | `mkt:repo_change_60d` | +0.05 | +0.07 | +0.10 | +0.19 | +0.27 | +0.30 |

Evaluation (last 90 sessions):

| Group | Feature | 1D | 3D | 5D | 10D | 15D | 30D |
|---|---|---|---|---|---|---|---|
| Price level / mean reversion | `price_vs_sma50_pct` | -0.12 | -0.27 | -0.36 | -0.44 | -0.58 | -0.64 |
|  | `price_vs_sma20_pct` | -0.06 | -0.14 | -0.18 | -0.19 | -0.39 | -0.55 |
|  | `big5_cost_spread_pct` | -0.10 | -0.21 | -0.26 | -0.35 | -0.46 | -0.51 |
|  | `mlb_cost_spread_pct` | -0.05 | -0.11 | -0.17 | -0.25 | -0.28 | -0.36 |
|  | `big5_ytd_pnl_share_20d_to` | -0.10 | -0.19 | -0.24 | -0.30 | -0.39 | -0.32 |
| Stock momentum | `close_ret_20d` | -0.10 | -0.24 | -0.29 | -0.37 | -0.55 | -0.52 |
|  | `close_ret_10d` | -0.03 | -0.01 | +0.01 | -0.05 | -0.25 | -0.49 |
|  | `sma20_vs_sma50_pct` | -0.14 | -0.25 | -0.31 | -0.37 | -0.32 | -0.14 |
|  | `ret_rollover_5d_20d` | +0.08 | +0.15 | +0.16 | +0.34 | +0.43 | +0.32 |
| Institutional inventory | `mlb_tertip_share` | -0.04 | -0.12 | -0.16 | -0.18 | -0.15 | -0.15 |
|  | `kamu_tertip_share` | -0.06 | -0.05 | -0.06 | -0.11 | -0.16 | -0.03 |
|  | `inst_ytd_pnl_divergence` | +0.10 | +0.15 | +0.17 | +0.22 | +0.29 | +0.33 |
| Daily MLB flow | `mlb_turnover_intensity` | +0.09 | +0.04 | +0.10 | +0.01 | +0.04 | -0.01 |
|  | `bofa_aggression_ratio` | +0.08 | +0.04 | +0.09 | +0.01 | +0.04 | -0.01 |
|  | `mlb_ewma_flow_share_5d` | +0.07 | +0.08 | +0.10 | +0.04 | +0.03 | -0.09 |
|  | `mlb_buy_share_today` | +0.05 | -0.03 | -0.03 | -0.13 | -0.08 | -0.10 |
| Volatility | `stock_vol_20d` | +0.05 | +0.00 | -0.05 | +0.00 | -0.02 | -0.16 |
|  | `bollinger_bandwidth_pct` | +0.06 | +0.01 | -0.01 | -0.03 | -0.02 | -0.01 |
| Market regime (new) | `mkt:bist30_ret_20d` | -0.06 | -0.19 | -0.27 | -0.32 | -0.53 | -0.27 |
|  | `mkt:bist30_ret_60d` | -0.01 | -0.10 | -0.14 | -0.21 | -0.34 | -0.43 |
|  | `mkt:bist30_vs_sma50` | -0.04 | -0.13 | -0.21 | -0.28 | -0.45 | -0.25 |
|  | `mkt:bist30_drawdown_60d` | -0.06 | -0.17 | -0.26 | -0.36 | -0.56 | -0.31 |
|  | `mkt:bist30_vol_20d` | -0.01 | +0.03 | +0.04 | +0.02 | +0.01 | -0.21 |
|  | `mkt:breadth_above_sma20` | +0.01 | -0.11 | -0.15 | -0.20 | -0.36 | -0.35 |
|  | `mkt:mlb_flow_share_5d` | +0.13 | +0.16 | +0.07 | +0.05 | -0.03 | -0.09 |
|  | `mkt:rel_strength_20d` | -0.06 | -0.12 | -0.15 | -0.21 | -0.29 | -0.53 |
|  | `mkt:repo_change_60d` | — | — | — | — | — | — |

Reading the map:

- **1D**: only the daily MLB flow family (`mlb_turnover_intensity`, `bofa_aggression_ratio`, `mlb_ewma_flow_share_5d`)
  is positive in both windows; it fades to zero by 10D. This matches the earlier finding that BofA flow is mostly priced
  within one or two sessions.
- **3D-5D**: a mix — flow still contributes, price-level and momentum features start to dominate.
- **10D-30D**: price level / trend variables (SMA distances, 20D return, cost spreads, YTD PnL) dominate with negative
  IC (mean reversion), plus `inst_ytd_pnl_divergence` and `ret_rollover_5d_20d` with positive IC.
- **Unstable sign** (do not trust): `kamu_tertip_share`, volatility features (`stock_vol_20d`,
  `bollinger_bandwidth_pct`), `mkt:bist30_vol_20d`.

## 3. New market-regime features

Added (all backward-looking, BIST 30 / XU030): returns over 5 / 10 / 20 / 60 sessions, distance from SMA20 / SMA50,
drawdown from the 60-session high, 20-session realized volatility, 5-vs-20 momentum spread, breadth (share of BIST 30
above SMA20, 5-day advance share), market-wide BofA net flow share (1 / 5 / 10 sessions), stock-vs-index relative
strength (10 / 20), 60-session change of the TCMB repo rate.

Finding: their IC is negative (mean reversion) at 3D-30D in both windows, so a linear model reads a falling market as
a buying opportunity — the opposite of the intuition that a crushing market should push the forecast down. Their
contribution also flips between regimes (tables C1, C2). The repo-rate change has the most consistent positive IC
(+0.19 to +0.30 at 10D-30D in discovery) but did not move in the evaluation window (no rate change in the last 90
sessions), so it could not be tested out of sample.

## 4. Experiments

### C1. Feature-set variants, fixed models (Ridge/252, BayesianRidge/252, LightGBM/252 averaged; no selection)

Hit rate %, 7 stocks pooled, last 90 graded sessions:

| Variant | Features | 1D | 3D | 5D | 10D | 15D | 30D |
|---|---|---|---|---|---|---|---|
| base (production FIFO_CORE) | 39 / 63 | 51.2 | 49.7 | 52.0 | 54.0 | 57.6 | 57.5 |
| **pruned** (consistent discovery IC only) | 16-36 | 51.1 | **51.4** | **53.2** | **56.1** | **58.9** | 55.3 |
| base + market regime | 56 / 80 | 50.9 | 49.4 | 50.0 | 53.6 | 55.8 | 56.9 |
| pruned + market regime | 33-53 | 50.6 | 49.9 | 51.5 | 53.2 | 55.9 | 55.9 |
| horizon-specific top-12 (by discovery IC) | 12 | 52.6 | 50.1 | 52.3 | 51.6 | 57.5 | 57.5 |
| market regime only | 17 | 49.6 | 47.4 | 49.4 | 53.4 | 56.2 | 61.0 |
| Always UP | | 45.9 | 45.1 | 44.3 | 40.8 | 43.7 | 43.5 |
| **Always DOWN** | | **56.3** | **55.4** | **56.8** | **60.0** | 56.3 | 56.5 |

Pruned feature sets per horizon (chosen on discovery): 1D keeps 16 (daily MLB / BofA flow, institutional net share,
range, volatility pinch, shock distance, inventory shares); 3D 18; 5D 26; 10D 34; 15D 36; 30D 30 (price trend and
level variables, returns 3-20D, volatility, inventory shares, daily desk shares). Dropped everywhere: YTD features
(not in FIFO_CORE), most single-day BIG5 / KAMU PnL shares, `kamu_post_shock_defense`, `tertip_inventory_zscore`,
`mlb_flow_3d/5d/10d_share` and `mlb_ewma_flow_share_5d` (swing horizons; the 1D candidates use the T+1 feature
suite, which does not contain the multi-day flow features).

### C2. Simple market rules (no fitting): edge vs the better constant call, pp

| Rule | Window | 1D | 3D | 5D | 10D | 15D | 30D |
|---|---|---|---|---|---|---|---|
| trend: BIST 30 20D return | discovery | -1.4 | -4.2 | -4.6 | -11.1 | -11.8 | -9.0 |
| trend: BIST 30 20D return | last 90 | -4.9 | -8.9 | -11.3 | -20.2 | -22.9 | -19.4 |
| contrarian: BIST 30 vs SMA50 | discovery | +1.0 | -1.6 | -2.0 | -1.1 | -6.5 | -8.5 |
| contrarian: BIST 30 vs SMA50 | last 90 | -5.6 | -0.6 | +1.3 | +5.7 | +11.7 | +3.3 |
| trend: stock vs BIST 30 (20D) | discovery | +0.2 | -2.1 | -2.2 | -3.2 | -8.4 | -5.5 |
| trend: stock vs BIST 30 (20D) | last 90 | -3.8 | -5.9 | -6.0 | -13.3 | -12.5 | -23.8 |

No rule is positive in both windows at any horizon beyond noise.

### C3. Honest monthly re-crowning on the pruned sets vs production (same 90 sessions per stock)

| | 1D | 3D | 5D | 10D | 15D | 30D |
|---|---|---|---|---|---|---|
| Production re-crowned (75 candidates) | 52.9 | 51.6 | 48.6 | 55.1 | **57.5** | **58.1** |
| Pruned re-crowned (25 candidates) | 52.1 | 51.0 | 48.9 | 51.6 | 50.2 | 54.6 |
| Always DOWN | 56.3 | 55.4 | 56.8 | 60.0 | 56.3 | 56.5 |
| Stocks beating both — production | 0/7 | 3/7 | 1/7 | 3/7 | 4/7 | 4/7 |
| Stocks beating both — pruned | 1/7 | 1/7 | 1/7 | 2/7 | 3/7 | 4/7 |

Pruning wins for a fixed model, loses inside the tournament: the production pool includes the YTD feature modes,
which the honest permutation test rates among the most useful (`big5_ytd_pnl_share_20d_to`, `mlb_ytd_pnl_share_20d_to`).

## 5. Do the features make sense per horizon?

- **1D** — yes for the daily MLB flow family; most other 1D inputs (single-day BIG5 / KAMU shares, PnL shares, W5
  shares, shock distances) carry nothing. A lean 1D set (flow + range + inventory) is justified.
- **3D-5D** — weakest horizons: nothing beats either constant call; flow has faded and price-level effects are not yet
  strong. These are the horizons where new information (VİOP basis, order book) is most needed.
- **10D-30D** — the models are mean-reversion bets on price level vs averages and vs desks' cost. That works in range
  markets and fails in sustained trends (this autumn). To call trends they need inputs that *change sign* in a trend
  regime — e.g. a regime switch (trend strength / persistence of the index) interacting with the mean-reversion
  features, rather than more level variables.
- **Inventory share features** (`mlb_tertip_share`, `kamu_tertip_share`) are heavily weighted but unstable in sign;
  `price_vs_sma50_pct` is weighted most and hurts most out of sample.

## 6. Recommendations and next experiments

1. Keep production as is (honest re-crowning, 75 candidates); do not add the market features linearly.
2. Next experiment — **regime-conditional mean reversion**: interact the price-level features with an index
   trend-persistence flag (e.g. BIST 30 below SMA50 for N sessions and falling breadth), or train separate models per
   regime, so "below average" means "buy" only in range regimes.
3. **Drop `price_vs_sma50_pct`** from candidates in a controlled re-crowning test (most harmful feature out of sample),
   and test a 1D-specific lean set.
4. **Longer minimum training windows**: 63-session champions (THYAO 10D Jul-Sep) overfit the preceding rally; test
   excluding the 3M lookback from the tournament.
5. New information remains the main route to beating both constant calls: VİOP single-stock / index futures basis and
   open interest, event calendar (MPC, index rebalancing, dividends), order-book data.

## Data notes

- Stock silver has no sessions on 2026-10-01, 10-02, 10-05 (no raw files); the BIST 30 benchmark table has those days.
- The benchmark's `daily_return_pct` is NaN on 2026-10-09 and 10-16; features fall back to the turnover-weighted BIST 30
  basket return.
- TRALT (renamed from KOZAL) has too little history under the new ticker to be crowned before September 2026.

## Reproduce

```bash
.venv/bin/python scripts/experiments/feature_importance_experiment.py --part all   # frames, A, A2, B, D, C
```
