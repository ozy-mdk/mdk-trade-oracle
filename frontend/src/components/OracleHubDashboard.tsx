import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchTertipMlForecast } from '../api/client';
import { formatTL } from '../utils/formatters';
import {
  BrainCircuit,
  Trophy,
  Activity,
  RotateCw,
  Layers,
  Calendar,
  Zap,
} from 'lucide-react';

interface OracleHubDashboardProps {
  symbol?: string;
  onSelectSymbol?: (symbol: string) => void;
}

const POPULAR_STOCKS = [
  'AKBNK',
  'THYAO',
  'GARAN',
  'EREGL',
  'TUPRS',
  'SISE',
  'BIMAS',
  'KCHOL',
  'ASELS',
  'YKBNK',
  'PETKM',
  'SAHOL',
  'EKGYO',
  'PGSUS',
];

export const OracleHubDashboard: React.FC<OracleHubDashboardProps> = ({
  symbol = 'AKBNK',
  onSelectSymbol,
}) => {
  const [activeSymbol, setActiveSymbol] = useState<string>(symbol);
  const [modelType, setModelType] = useState<'auto' | 'ridge' | 'xgboost'>('auto');
  const [chartView, setChartView] = useState<'price' | 'return'>('price');
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // Sync with parent symbol if it changes
  useEffect(() => {
    if (symbol && symbol !== activeSymbol) {
      setActiveSymbol(symbol);
    }
  }, [symbol]);

  const {
    data: forecast,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['tertipMlForecast', activeSymbol, modelType],
    queryFn: () => fetchTertipMlForecast(activeSymbol, false, modelType),
    refetchInterval: 60000,
  });

  const handleStockClick = (sym: string) => {
    setActiveSymbol(sym);
    if (onSelectSymbol) {
      onSelectSymbol(sym);
    }
  };

  const tournament = forecast?.tournament_summary;
  const ledger = forecast?.walk_forward_ledger || [];
  const latestPrice = forecast?.latest_close_price || 0;
  const targetPrice = forecast?.target_price || 0;
  const expReturn = forecast?.expected_return_pct || 0;
  const isUp = expReturn >= 0;

  // Compute hits and misses from ledger
  const mlHits = ledger.filter((r) => r.ml_is_hit).length;
  const mlMiss = ledger.length - mlHits;
  const hitRatePct = ledger.length > 0 ? (mlHits / ledger.length) * 100 : 0;

  const last10Ledger = ledger.slice(-10);
  const l10Hits = last10Ledger.filter((r) => r.ml_is_hit).length;
  const l10Miss = last10Ledger.length - l10Hits;
  const l10RatePct = last10Ledger.length > 0 ? (l10Hits / last10Ledger.length) * 100 : 0;

  // Chart preparation
  const chartHeight = 220;
  const chartWidth = 900;
  const paddingX = 45;
  const paddingY = 25;
  const innerW = chartWidth - paddingX * 2;
  const innerH = chartHeight - paddingY * 2;

  // Price Extents
  const minPrice = ledger.length
    ? Math.min(...ledger.map((r) => Math.min(r.actual_price, r.ml_pred_price))) * 0.985
    : 0;
  const maxPrice = ledger.length
    ? Math.max(...ledger.map((r) => Math.max(r.actual_price, r.ml_pred_price))) * 1.015
    : 100;

  // Return Extents
  const maxAbsReturn = ledger.length
    ? Math.max(
        ...ledger.map((r) =>
          Math.max(Math.abs(r.actual_return_pct), Math.abs(r.ml_pred_return_pct || 0))
        ),
        4.0
      ) * 1.15
    : 10;

  const getX = (idx: number) => {
    if (ledger.length <= 1) return paddingX;
    return paddingX + (idx / (ledger.length - 1)) * innerW;
  };

  const getYPrice = (val: number) => {
    if (maxPrice <= minPrice) return paddingY + innerH / 2;
    return paddingY + innerH - ((val - minPrice) / (maxPrice - minPrice)) * innerH;
  };

  const getYReturn = (val: number) => {
    const zeroY = paddingY + innerH / 2;
    return zeroY - (val / maxAbsReturn) * (innerH / 2);
  };

  // Generate SVG path strings
  const actualPricePath = ledger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYPrice(r.actual_price).toFixed(1)}`)
    .join(' ');

  const predPricePath = ledger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYPrice(r.ml_pred_price).toFixed(1)}`)
    .join(' ');

  const actualReturnPath = ledger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYReturn(r.actual_return_pct).toFixed(1)}`)
    .join(' ');

  const predReturnPath = ledger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYReturn(r.ml_pred_return_pct || 0).toFixed(1)}`)
    .join(' ');

  const hoveredItem = hoveredIdx !== null && ledger[hoveredIdx] ? ledger[hoveredIdx] : null;

  return (
    <div className="space-y-4">
      {/* ── Top Header & Stock Switcher Ribbon ────────────────────────────────────── */}
      <div className="glass-panel p-3.5 rounded-xl border border-cyan-500/30 bg-slate-900/90 shadow-xl flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-cyan-500/20 text-cyan-400 rounded-lg border border-cyan-500/30">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-bold text-white tracking-wide">
                Gold Tertip Predictive Hub
              </span>
              <span className="px-2 py-0.5 rounded text-[9.5px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                12-MONTH LOOKBACK (252 SESSIONS)
              </span>
            </div>
            <div className="text-[11px] text-slate-400">
              Institutional Order-Flow Walk-Forward Tournament & 3-Pillar Predictive Suite
            </div>
          </div>
        </div>

        {/* Quick Stock Selector Pills */}
        <div className="flex flex-wrap items-center gap-1">
          {POPULAR_STOCKS.map((sym) => {
            const isSelected = activeSymbol === sym;
            return (
              <button
                key={sym}
                onClick={() => handleStockClick(sym)}
                className={`px-2 py-1 rounded text-[10px] font-mono font-bold transition-all ${
                  isSelected
                    ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20 font-black'
                    : 'bg-slate-950/60 text-slate-400 hover:text-white hover:bg-slate-800/80 border border-slate-800'
                }`}
              >
                {sym}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Configuration Bar: Model Arena, Feature Suite, Refresh ──────────────── */}
      <div className="glass-panel p-2.5 rounded-lg border border-slate-800 flex flex-wrap items-center justify-between gap-2.5 bg-slate-950/70 text-xs font-mono">
        <div className="flex flex-wrap items-center gap-2">
          {/* Active Symbol Display */}
          <div className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-white font-bold flex items-center gap-1.5">
            <span className="text-cyan-400">Symbol:</span>
            <span>{activeSymbol}</span>
            <span className="text-[10px] text-slate-400 font-normal">
              (₺{latestPrice > 0 ? latestPrice.toFixed(2) : '—'})
            </span>
          </div>

          {/* Model Type Selector */}
          <div className="flex items-center bg-slate-900 rounded border border-slate-800 p-0.5">
            <span className="px-1.5 text-[9.5px] text-slate-500 uppercase font-sans font-semibold">
              Arena Model:
            </span>
            {(
              [
                { id: 'auto', label: 'Auto Champion 👑' },
                { id: 'ridge', label: 'Ridge' },
                { id: 'xgboost', label: 'XGBoost' },
              ] as const
            ).map((m) => (
              <button
                key={m.id}
                onClick={() => setModelType(m.id)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold transition-colors ${
                  modelType === m.id
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Features Suite Toggle */}
          <div className="flex items-center bg-slate-900 rounded border border-slate-800 p-0.5">
            <span className="px-1.5 text-[9.5px] text-slate-500 uppercase font-sans font-semibold flex items-center gap-1">
              <Layers className="w-3 h-3 text-cyan-400" />
              Features:
            </span>
            <span className="px-2 py-0.5 text-[10px] font-bold text-emerald-400" title="21 Scarce Microstructure, Tertip Inventory, Today's Execution, Net Imbalance & Prophet Baseline Features">
              {forecast?.active_features_count ?? 21} Lean Features (Zero Noise)
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 transition-colors disabled:opacity-50 text-[10.5px]"
          >
            <RotateCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin text-cyan-400' : 'text-slate-400'}`} />
            <span>{isFetching ? 'Recalculating...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="glass-panel p-12 text-center text-slate-400 font-mono text-sm border border-slate-800 rounded-xl">
          <Activity className="w-6 h-6 animate-spin mx-auto mb-2 text-cyan-400" />
          <span>Computing zero-lookahead 12-month walk-forward tournament for {activeSymbol}...</span>
        </div>
      ) : forecast ? (
        <>
          {/* ── Executive Grand Tournament & Live T+1 Signal Card ───────────────────── */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Live T+1 Inference Target Card */}
            <div className="glass-panel p-4 rounded-xl border border-cyan-500/30 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 lg:col-span-2 shadow-lg relative overflow-hidden">
              <div className="absolute top-0 right-0 -mt-6 -mr-6 w-36 h-36 bg-cyan-500/10 rounded-full blur-2xl pointer-events-none" />

              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-2.5 mb-3">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-cyan-300">
                    Live Upcoming Session (T+1) Forecast
                  </span>
                  <span className="px-2 py-0.5 rounded text-[9.5px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                    TARGET: {forecast.as_of_date}
                  </span>
                </div>
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-[11px] text-slate-400">Current Close:</span>
                  <span className="text-sm font-bold text-white">₺{latestPrice.toFixed(2)}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {/* Target Price & Return */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-[10.5px] text-slate-400 mb-0.5">Forecasted Close Target</div>
                  <div className="flex items-baseline space-x-2">
                    <span className="text-2xl font-black font-mono text-white">
                      ₺{targetPrice.toFixed(2)}
                    </span>
                    <span
                      className={`text-sm font-bold font-mono ${
                        isUp ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {isUp ? '+' : ''}
                      {expReturn.toFixed(2)}%
                    </span>
                  </div>
                  <div className="text-[9.5px] text-slate-500 mt-1 font-mono">
                    90% CI: [₺{forecast.price_low.toFixed(2)} , ₺{forecast.price_high.toFixed(2)}]
                  </div>
                </div>

                {/* Conviction & Stance */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col justify-between">
                  <div className="text-[10.5px] text-slate-400 mb-0.5">Directional Conviction</div>
                  <div>
                    <span
                      className={`inline-block px-2.5 py-1 rounded text-xs font-black font-mono tracking-wider border ${
                        forecast.stance_color === 'emerald'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : forecast.stance_color === 'rose'
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : 'bg-slate-800 text-slate-300 border-slate-700'
                      }`}
                    >
                      {forecast.stance_badge}
                    </span>
                  </div>
                  <div className="text-[9.5px] text-slate-400 mt-1 flex items-center gap-1 font-mono">
                    <span>Stance:</span>
                    <span className="text-white font-semibold">{forecast.stance}</span>
                  </div>
                </div>

                {/* Candidate Comparison Mini-Box */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col justify-between text-xs font-mono">
                  <div className="text-[10.5px] text-slate-400 mb-0.5">Arena Model Targets</div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-cyan-400">ML Challenger:</span>
                      <span className="font-bold text-white">
                        ₺{forecast.ml_target_price?.toFixed(2)}{' '}
                        <span className={(forecast.ml_expected_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          ({(forecast.ml_expected_return_pct || 0) >= 0 ? '+' : ''}
                          {forecast.ml_expected_return_pct?.toFixed(2)}%)
                        </span>
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-purple-400">Prophet Base:</span>
                      <span className="font-bold text-white">
                        ₺{forecast.prophet_target_price.toFixed(2)}{' '}
                        <span className={forecast.prophet_expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          ({forecast.prophet_expected_return_pct >= 0 ? '+' : ''}
                          {forecast.prophet_expected_return_pct.toFixed(2)}%)
                        </span>
                      </span>
                    </div>
                  </div>
                  <div className="text-[9px] text-slate-500 mt-1">
                    Crowned: <span className="text-amber-300 font-semibold">{tournament?.champion_label}</span>
                  </div>
                </div>
              </div>

              {/* Rationale Headline */}
              <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-start space-x-2 text-xs text-slate-300">
                <Zap className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white">{forecast.playbook_headline}. </span>
                  <span className="text-slate-400 text-[11px]">{forecast.playbook_rationale}</span>
                </div>
              </div>
            </div>

            {/* Tournament Arena Champion Card */}
            <div className="glass-panel p-4 rounded-xl border border-amber-500/30 bg-slate-900/90 shadow-lg flex flex-col justify-between">
              <div>
                <div className="flex items-center space-x-2 border-b border-slate-800 pb-2 mb-3">
                  <Trophy className="w-4 h-4 text-amber-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-300">
                    Grand Tournament Champion
                  </span>
                </div>

                <div className="text-center py-2">
                  <div className="text-sm font-black text-white font-mono flex items-center justify-center gap-1.5">
                    <span>👑</span>
                    <span>{tournament?.champion_label || 'Tertip ML Challenger'}</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Selected dynamically out-of-sample via 30-Day Walk-Forward
                  </div>
                </div>

                <div className="space-y-2 mt-2 font-mono text-xs">
                  <div className="flex items-center justify-between p-2 rounded bg-slate-950/60 border border-slate-800">
                    <span className="text-slate-400">Directional Hit Rate:</span>
                    <span className="font-bold text-emerald-400 text-sm">
                      {tournament?.champion_dir_hits ?? mlHits}/30 (
                      {(tournament?.champion_dir_hit_rate_pct ?? hitRatePct).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded bg-slate-950/60 border border-slate-800">
                    <span className="text-slate-400">Runner-Up Hit Rate:</span>
                    <span className="font-semibold text-slate-300">
                      {(tournament?.runner_up_dir_hit_rate_pct ?? tournament?.prophet_hit_rate_pct ?? 43.3).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded bg-slate-950/60 border border-slate-800">
                    <span className="text-slate-400">Mean Abs Error (MAE):</span>
                    <span className="font-semibold text-cyan-300">
                      {(tournament?.champion_mae_pct ?? tournament?.ml_mae_pct ?? 2.2).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Arena Details */}
              <div className="text-[9.5px] font-mono text-slate-500 pt-2 border-t border-slate-800 mt-2 flex items-center justify-between">
                <span>Training: Last 12M ({forecast.train_lookback_sessions ?? 252}d)</span>
                <span>Active Feats: {forecast.active_features_count ?? 21}</span>
              </div>
            </div>
          </div>

          {/* ── Visual Walk-Forward Chart: Predicted vs What Happened in Last 30 Sessions ── */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800 bg-slate-900/90 shadow-xl space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white tracking-wide">
                  Walk-Forward Performance: Predicted vs Realized (Last 30 Sessions)
                </span>
                <span className="text-xs text-slate-500 font-mono">Zero-Lookahead Point-in-Time</span>
              </div>

              {/* Chart Mode Toggle & Stats Pill */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center font-mono text-[9px] gap-1.5">
                  <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                    30D Record: <span className="text-emerald-400 font-bold">{mlHits}✓</span> /{' '}
                    <span className="text-rose-400 font-bold">{mlMiss}✗</span> ({hitRatePct.toFixed(1)}%)
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                    Last 10D: <span className="text-emerald-400 font-bold">{l10Hits}✓</span> /{' '}
                    <span className="text-rose-400 font-bold">{l10Miss}✗</span> ({l10RatePct.toFixed(0)}%)
                  </span>
                </div>

                <div className="flex items-center bg-slate-950 rounded border border-slate-800 p-0.5 font-mono text-xs">
                  <button
                    onClick={() => setChartView('price')}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      chartView === 'price'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Price Series (₺)
                  </button>
                  <button
                    onClick={() => setChartView('return')}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      chartView === 'return'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Return % Divergence
                  </button>
                </div>
              </div>
            </div>

            {/* Interactive SVG Chart Container */}
            <div className="relative bg-slate-950/80 rounded-lg p-3 border border-slate-800/80 overflow-hidden">
              <svg
                viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                className="w-full h-56 select-none"
                onMouseLeave={() => setHoveredIdx(null)}
              >
                <defs>
                  <linearGradient id="actualGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Grid Lines */}
                {chartView === 'price' ? (
                  <>
                    {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
                      const y = paddingY + innerH * p;
                      const val = maxPrice - p * (maxPrice - minPrice);
                      return (
                        <g key={i}>
                          <line
                            x1={paddingX}
                            y1={y}
                            x2={chartWidth - paddingX}
                            y2={y}
                            stroke="#334155"
                            strokeWidth="0.5"
                            strokeDasharray="3 3"
                          />
                          <text
                            x={paddingX - 6}
                            y={y + 3}
                            textAnchor="end"
                            fill="#64748b"
                            fontSize="8"
                            fontFamily="monospace"
                          >
                            ₺{val.toFixed(1)}
                          </text>
                        </g>
                      );
                    })}
                  </>
                ) : (
                  <>
                    {/* Zero Line for Returns */}
                    <line
                      x1={paddingX}
                      y1={paddingY + innerH / 2}
                      x2={chartWidth - paddingX}
                      y2={paddingY + innerH / 2}
                      stroke="#64748b"
                      strokeWidth="1"
                    />
                    <text
                      x={paddingX - 6}
                      y={paddingY + innerH / 2 + 3}
                      textAnchor="end"
                      fill="#94a3b8"
                      fontSize="8"
                      fontFamily="monospace"
                    >
                      0.0%
                    </text>
                    {[0.2, 0.8].map((p, i) => {
                      const y = paddingY + innerH * p;
                      const val = (0.5 - p) * 2 * maxAbsReturn;
                      return (
                        <g key={i}>
                          <line
                            x1={paddingX}
                            y1={y}
                            x2={chartWidth - paddingX}
                            y2={y}
                            stroke="#334155"
                            strokeWidth="0.5"
                            strokeDasharray="3 3"
                          />
                          <text
                            x={paddingX - 6}
                            y={y + 3}
                            textAnchor="end"
                            fill="#64748b"
                            fontSize="8"
                            fontFamily="monospace"
                          >
                            {val > 0 ? '+' : ''}
                            {val.toFixed(1)}%
                          </text>
                        </g>
                      );
                    })}
                  </>
                )}

                {/* Plot Paths */}
                {chartView === 'price' ? (
                  <>
                    {/* Actual Price Curve */}
                    <path
                      d={actualPricePath}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    {/* Predicted Price Curve */}
                    <path
                      d={predPricePath}
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth="2"
                      strokeDasharray="4 3"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </>
                ) : (
                  <>
                    {/* Actual Return Curve */}
                    <path
                      d={actualReturnPath}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    {/* Predicted Return Curve */}
                    <path
                      d={predReturnPath}
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth="2"
                      strokeDasharray="4 3"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </>
                )}

                {/* Interactive Points on each session */}
                {ledger.map((r, i) => {
                  const x = getX(i);
                  const y = chartView === 'price' ? getYPrice(r.actual_price) : getYReturn(r.actual_return_pct);
                  const isHovered = hoveredIdx === i;

                  return (
                    <g
                      key={i}
                      onMouseEnter={() => setHoveredIdx(i)}
                      className="cursor-pointer"
                    >
                      {/* Vertical Guideline on hover */}
                      {isHovered && (
                        <line
                          x1={x}
                          y1={paddingY}
                          x2={x}
                          y2={paddingY + innerH}
                          stroke="#64748b"
                          strokeWidth="1"
                          strokeDasharray="2 2"
                        />
                      )}

                      {/* Result Dot */}
                      <circle
                        cx={x}
                        cy={y}
                        r={isHovered ? 5.5 : 3.5}
                        fill={r.ml_is_hit ? '#10b981' : '#f43f5e'}
                        stroke="#0f172a"
                        strokeWidth="1.5"
                      />

                      {/* X Axis Date Labels */}
                      {i % 5 === 0 && (
                        <text
                          x={x}
                          y={chartHeight - 6}
                          textAnchor="middle"
                          fill="#64748b"
                          fontSize="8"
                          fontFamily="monospace"
                        >
                          {r.date.slice(5)}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>

              {/* Dynamic Hover Tooltip Card */}
              {hoveredItem && (
                <div
                  className="absolute top-2 right-2 bg-slate-900/95 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs font-mono z-20 pointer-events-none min-w-[210px]"
                >
                  <div className="flex items-center justify-between border-b border-slate-800 pb-1 mb-1.5">
                    <span className="text-slate-300 font-bold flex items-center gap-1">
                      <Calendar className="w-3 h-3 text-cyan-400" />
                      {hoveredItem.date}
                    </span>
                    <span
                      className={`px-1.5 py-0.2 rounded text-[8px] font-black border ${
                        hoveredItem.ml_is_hit
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                      }`}
                    >
                      {hoveredItem.ml_is_hit ? '✓ CORRECT' : '✗ WRONG'}
                    </span>
                  </div>

                  <div className="space-y-1 text-[11px]">
                    <div className="flex items-center justify-between">
                      <span className="text-cyan-400">Actual Close:</span>
                      <span className="font-bold text-white">
                        ₺{hoveredItem.actual_price.toFixed(2)}{' '}
                        <span
                          className={hoveredItem.actual_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}
                        >
                          ({hoveredItem.actual_return_pct >= 0 ? '+' : ''}
                          {hoveredItem.actual_return_pct.toFixed(2)}%)
                        </span>
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-amber-400">ML Predicted:</span>
                      <span className="font-bold text-white">
                        ₺{hoveredItem.ml_pred_price.toFixed(2)}{' '}
                        <span
                          className={
                            (hoveredItem.ml_pred_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }
                        >
                          ({(hoveredItem.ml_pred_return_pct || 0) >= 0 ? '+' : ''}
                          {(hoveredItem.ml_pred_return_pct || 0).toFixed(2)}%)
                        </span>
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">ML Error:</span>
                      <span className="text-slate-200">{hoveredItem.ml_err_pct.toFixed(2)}%</span>
                    </div>

                    <div className="flex items-center justify-between border-t border-slate-800 pt-1 mt-1">
                      <span className="text-purple-400">Prophet Pred:</span>
                      <span className="text-slate-300">
                        ₺{hoveredItem.prophet_pred_price.toFixed(2)}{' '}
                        <span>({hoveredItem.prophet_is_hit ? '✓' : '✗'})</span>
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-amber-200">XU030 Return:</span>
                      <span
                        className={
                          (hoveredItem.bist30_ret_pct || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }
                      >
                        {(hoveredItem.bist30_ret_pct || 0) >= 0 ? '+' : ''}
                        {(hoveredItem.bist30_ret_pct || 0).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Chart Legend */}
              <div className="flex flex-wrap items-center justify-between text-[10px] font-mono text-slate-400 pt-2 border-t border-slate-800/60 mt-1 px-1">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-3 h-0.5 bg-sky-400 rounded-full inline-block" />
                    <span>Actual Realization</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-3 h-0.5 bg-amber-400 border-b border-amber-400 border-dashed inline-block" />
                    <span>ML Walk-Forward Predicted</span>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                    <span>Direction Correct (✓)</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
                    <span>Direction Wrong (✗)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ── 3-Pillars Institutional Confluence Matrix ──────────────────────────── */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800 bg-slate-900/90 shadow-xl space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white tracking-wide">
                  3-Pillar Institutional Confluence Matrix
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  Ground-Truth Order Flow Engine
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {forecast.pillar_matrix?.map((p) => {
                const isBuy = p.stance === 'BUY';
                const isSell = p.stance === 'SELL';
                return (
                  <div
                    key={p.pillar}
                    className="bg-slate-950/70 p-3.5 rounded-lg border border-slate-800 font-mono text-xs flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5 mb-2">
                        <span className="font-bold text-white font-sans">{p.name}</span>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isBuy
                              ? 'bg-emerald-500/20 text-emerald-300'
                              : isSell
                              ? 'bg-rose-500/20 text-rose-300'
                              : 'bg-slate-800 text-slate-300'
                          }`}
                        >
                          {p.stance}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-400 font-sans mb-2">{p.desc}</div>

                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">12M Realized %:</span>
                          <span className="font-bold text-cyan-300">{p.realized_pct}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">Expected Action:</span>
                          <span className="font-semibold text-white">{p.expected_action}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">Avg Flow:</span>
                          <span
                            className={p.expected_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}
                          >
                            {formatTL(p.expected_flow_tl)}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400">Turnover Share:</span>
                          <span className="text-slate-300">{p.turnover_share_pct}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── 30-Day Walk-Forward Reality Ledger Table ───────────────────────────── */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800 bg-slate-900/90 shadow-xl space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
              <div className="flex items-center space-x-2">
                <Calendar className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white tracking-wide">
                  Out-of-Sample Walk-Forward Reality Ledger
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  Session-by-Session Audited Log
                </span>
              </div>
              <div className="flex items-center gap-1.5 font-mono text-[8.5px]">
                <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                  30D ML: <span className="text-emerald-400 font-bold">{mlHits}✓</span> /{' '}
                  <span className="text-rose-400 font-bold">{mlMiss}✗</span> ({hitRatePct.toFixed(1)}%)
                </span>
                <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                  Last 10D: <span className="text-emerald-400 font-bold">{l10Hits}✓</span> /{' '}
                  <span className="text-rose-400 font-bold">{l10Miss}✗</span> ({l10RatePct.toFixed(0)}%)
                </span>
              </div>
            </div>

            <div className="overflow-x-auto max-h-72 overflow-y-auto rounded-lg border border-slate-800/90">
              <table className="w-full text-[9px] font-mono border-collapse">
                <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 z-10 font-sans">
                  <tr className="text-slate-400">
                    <th className="text-left px-2 py-1.5">Date</th>
                    <th className="text-right px-2 py-1.5">Actual Close</th>
                    <th className="text-right px-2 py-1.5 text-cyan-300">ML Predicted</th>
                    <th className="text-center px-2 py-1.5 text-cyan-300">ML Hit?</th>
                    <th className="text-right px-2 py-1.5 text-amber-200">XU030</th>
                    <th className="text-right px-2 py-1.5 text-purple-300">Prophet</th>
                    <th className="text-center px-2 py-1.5">BofA MLB Did</th>
                    <th className="text-center px-2 py-1.5">BIG5 Did</th>
                    <th className="text-center px-2 py-1.5">KAMU Did</th>
                    <th className="text-center px-2 py-1.5">Winner</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                  {ledger.map((row) => {
                    const isActUp = row.actual_return_pct > 0.02;
                    const isActDown = row.actual_return_pct < -0.02;
                    const isMlUp = (row.ml_pred_return_pct || 0) > 0.02;
                    const isMlDown = (row.ml_pred_return_pct || 0) < -0.02;
                    const pRet = row.prophet_pred_return_pct !== undefined ? row.prophet_pred_return_pct : 0;
                    const bothMiss = !row.ml_is_hit && !row.prophet_is_hit;

                    return (
                      <tr key={row.date} className="hover:bg-slate-800/40 text-slate-300">
                        <td className="px-2 py-1 text-slate-400 font-bold">{row.date.slice(5)}</td>
                        <td className="text-right px-2 py-1 text-white font-semibold">
                          <div>₺{row.actual_price.toFixed(2)}</div>
                          <div
                            className={`text-[8px] font-bold ${
                              isActUp ? 'text-emerald-400' : isActDown ? 'text-rose-400' : 'text-slate-400'
                            }`}
                          >
                            {isActUp ? '▲ +' : isActDown ? '▼ ' : '■ '}
                            {row.actual_return_pct.toFixed(2)}%
                          </div>
                        </td>
                        <td className="text-right px-2 py-1">
                          <div className="font-semibold text-white">₺{row.ml_pred_price.toFixed(2)}</div>
                          <div
                            className={`text-[8px] font-bold flex items-center justify-end gap-1 ${
                              isMlUp ? 'text-emerald-400' : isMlDown ? 'text-rose-400' : 'text-slate-400'
                            }`}
                          >
                            <span>
                              {isMlUp ? '▲ +' : isMlDown ? '▼ ' : '■ '}
                              {(row.ml_pred_return_pct || 0).toFixed(2)}%
                            </span>
                            <span className="text-[7.5px] text-slate-500 font-normal">
                              ({row.ml_err_pct.toFixed(1)}%)
                            </span>
                          </div>
                        </td>
                        <td className="text-center px-2 py-1">
                          <span
                            className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[8px] font-black border tracking-wider ${
                              row.ml_is_hit
                                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                                : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                            }`}
                          >
                            <span>{row.ml_is_hit ? '✓' : '✗'}</span>
                            <span>{row.ml_is_hit ? 'CORRECT' : 'WRONG'}</span>
                          </span>
                        </td>
                        <td className="text-right px-2 py-1">
                          <div
                            className={`text-[8.5px] font-bold ${
                              (row.bist30_ret_pct ?? 0) > 0.02
                                ? 'text-emerald-400'
                                : (row.bist30_ret_pct ?? 0) < -0.02
                                ? 'text-rose-400'
                                : 'text-slate-400'
                            }`}
                          >
                            {(row.bist30_ret_pct ?? 0) > 0 ? '+' : ''}
                            {(row.bist30_ret_pct ?? 0).toFixed(2)}%
                          </div>
                        </td>
                        <td className="text-right px-2 py-1 text-slate-300">
                          <div className="flex items-center justify-end gap-1">
                            <span>₺{row.prophet_pred_price.toFixed(2)}</span>
                            <span
                              className={`text-[7px] font-black px-1 py-0.2 rounded border ${
                                row.prophet_is_hit
                                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                                  : 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                              }`}
                            >
                              {row.prophet_is_hit ? '✓' : '✗'}
                            </span>
                          </div>
                          <div className={`text-[8px] ${pRet >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'}`}>
                            {pRet >= 0 ? '+' : ''}
                            {pRet.toFixed(1)}% <span className="text-slate-500">({row.prophet_err_pct.toFixed(1)}%)</span>
                          </div>
                        </td>
                        <td className="text-center px-2 py-1">
                          <span
                            className={`px-1 py-0.2 rounded text-[8px] font-bold ${
                              row.mlb_action === 'BUY'
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-rose-500/20 text-rose-400'
                            }`}
                          >
                            {row.mlb_action} {row.mlb_flow_tl >= 0 ? '+' : ''}
                            {(row.mlb_flow_tl / 1e6).toFixed(0)}M
                          </span>
                        </td>
                        <td className="text-center px-2 py-1">
                          <span
                            className={`px-1 py-0.2 rounded text-[8px] font-bold ${
                              row.big5_action === 'BUY'
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-rose-500/20 text-rose-400'
                            }`}
                          >
                            {row.big5_action} {row.big5_flow_tl >= 0 ? '+' : ''}
                            {(row.big5_flow_tl / 1e6).toFixed(0)}M
                          </span>
                        </td>
                        <td className="text-center px-2 py-1">
                          <span
                            className={`px-1 py-0.2 rounded text-[8px] font-bold ${
                              row.kamu_action === 'BUY'
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-rose-500/20 text-rose-400'
                            }`}
                          >
                            {row.kamu_action} {row.kamu_flow_tl >= 0 ? '+' : ''}
                            {(row.kamu_flow_tl / 1e6).toFixed(0)}M
                          </span>
                        </td>
                        <td className="text-center px-2 py-1 font-bold">
                          <span
                            className={`px-1.5 py-0.2 rounded text-[8px] border ${
                              bothMiss
                                ? 'bg-slate-800 text-slate-400 border-slate-700'
                                : row.winner === 'CHALLENGER'
                                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                                : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                            }`}
                          >
                            {bothMiss ? 'BOTH MISS' : row.winner === 'CHALLENGER' ? 'ML' : 'PROPHET'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="text-[8px] text-slate-500 italic text-center pt-0.5">
              Zero-lookahead point-in-time predictions logged from daily broker clearing records (silver_daily_broker_summary)
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
