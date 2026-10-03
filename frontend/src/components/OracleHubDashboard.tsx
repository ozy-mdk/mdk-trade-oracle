import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchTertipMlForecast } from '../api/client';
import { WalkForwardLedgerItem } from '../types/api';
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
  const [activeModel, setActiveModel] = useState<'champion' | 'ml' | 'prophet'>('champion');
  const [horizonRange, setHorizonRange] = useState<'30d' | '6m'>('30d');
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

  // Horizon slice: Last 30 sessions (focused recent view) vs Full hydrated multi-month history (126-180 sessions)
  const displayLedger = horizonRange === '30d' ? ledger.slice(-30) : ledger;


  // Dynamic targeting based on selected perspective
  let activeTargetPrice = forecast?.target_price || 0;
  let activeExpReturn = forecast?.expected_return_pct || 0;
  let activeStance = forecast?.stance || 'NEUTRAL';
  let activeStanceBadge = forecast?.stance_badge || 'NEUTRAL CONSOLIDATION';
  let activeStanceColor = forecast?.stance_color || 'slate';
  let activeHeadline = forecast?.playbook_headline || '';
  let activeRationale = forecast?.playbook_rationale || '';

  if (activeModel === 'ml') {
    activeTargetPrice = forecast?.ml_target_price ?? latestPrice;
    activeExpReturn = forecast?.ml_expected_return_pct ?? 0;
    activeStance = forecast?.ml_stance || 'NEUTRAL';
    activeStanceBadge = forecast?.ml_stance_badge || 'ML CHALLENGER';
    activeStanceColor = 'indigo';
    activeHeadline = `Projecting ML Challenger ${activeStanceBadge} towards ₺${activeTargetPrice.toFixed(2)} (${activeExpReturn >= 0 ? '+' : ''}${activeExpReturn.toFixed(2)}%)`;
    activeRationale = `Pure ${tournament?.ml_champion_type || 'Machine Learning'} model trained strictly on trailing 12 months (252 sessions) using 21 scarce microstructure features and point-in-time FIFO inventory tracking.`;
  } else if (activeModel === 'prophet') {
    activeTargetPrice = forecast?.prophet_target_price ?? latestPrice;
    activeExpReturn = forecast?.prophet_expected_return_pct ?? 0;
    activeStance = forecast?.prophet_stance || 'NEUTRAL';
    activeStanceBadge = forecast?.prophet_stance_badge || 'PROPHET BASELINE';
    activeStanceColor = forecast?.prophet_stance_color || 'purple';
    activeHeadline = `Projecting Prophet Baseline ${activeStanceBadge} towards ₺${activeTargetPrice.toFixed(2)} (${activeExpReturn >= 0 ? '+' : ''}${activeExpReturn.toFixed(2)}%)`;
    activeRationale = `Univariate Bayesian structural time-series baseline capturing non-linear calendar seasonality and trailing momentum without order-flow features.`;
  }

  const isUp = activeExpReturn >= 0;

  // Row value accessors reflecting the currently active model perspective
  const getRowPredPrice = (r: WalkForwardLedgerItem) => {
    if (activeModel === 'ml') return r.ml_pred_price;
    if (activeModel === 'prophet') return r.prophet_pred_price;
    return r.champion_pred_price !== undefined ? r.champion_pred_price : (r.confluence_pred_price !== undefined ? r.confluence_pred_price : r.ml_pred_price);
  };

  const getRowPredReturn = (r: WalkForwardLedgerItem) => {
    if (activeModel === 'ml') return r.ml_pred_return_pct || 0;
    if (activeModel === 'prophet') return r.prophet_pred_return_pct || 0;
    return r.champion_pred_return_pct !== undefined ? r.champion_pred_return_pct : (r.confluence_pred_return_pct !== undefined ? r.confluence_pred_return_pct : (r.ml_pred_return_pct || 0));
  };

  const getRowHit = (r: WalkForwardLedgerItem) => {
    if (activeModel === 'ml') return r.ml_is_hit;
    if (activeModel === 'prophet') return r.prophet_is_hit;
    return r.champion_is_hit !== undefined ? r.champion_is_hit : (r.confluence_is_hit !== undefined ? r.confluence_is_hit : r.ml_is_hit);
  };

  const getRowErrPct = (r: WalkForwardLedgerItem) => {
    if (activeModel === 'ml') return r.ml_err_pct;
    if (activeModel === 'prophet') return r.prophet_err_pct;
    return r.champion_err_pct !== undefined ? r.champion_err_pct : (r.confluence_err_pct !== undefined ? r.confluence_err_pct : r.ml_err_pct);
  };

  // Performance hits and hit rates evaluated over the visible horizon (displayLedger)
  const activeHits = displayLedger.filter((r) => getRowHit(r)).length;
  const activeMiss = displayLedger.length - activeHits;
  const activeHitRatePct = displayLedger.length > 0 ? (activeHits / displayLedger.length) * 100 : 0;

  // Big Move Opportunity Metrics:
  // Both directionally correct AND suggestion (prediction) was 2% or more
  const bigMoveRows = displayLedger.filter((r) => Math.abs(r.actual_return_pct) >= 2.0);
  const bigMoveTotal = bigMoveRows.length;
  const bigMoveHits = displayLedger.filter(
    (r) => Math.abs(getRowPredReturn(r)) >= 2.0 && Math.abs(r.actual_return_pct) >= 2.0 && getRowHit(r)
  ).length;
  const bigMoveSignalled = displayLedger.filter((r) => Math.abs(getRowPredReturn(r)) >= 2.0).length;
  const bigMoveMissed = displayLedger.filter(
    (r) => Math.abs(r.actual_return_pct) >= 2.0 && Math.abs(getRowPredReturn(r)) < 2.0
  ).length;
  const bigMoveFalseAlarms = displayLedger.filter(
    (r) => Math.abs(getRowPredReturn(r)) >= 2.0 && !getRowHit(r)
  ).length;
  const bigMoveHitRatePct = bigMoveTotal > 0 ? (bigMoveHits / bigMoveTotal) * 100 : 0;

  // 3 Selection Criteria + Big Move Opportunity Strike Loss
  const activeHitErrors = displayLedger.filter((r) => getRowHit(r)).map((r) => getRowErrPct(r));
  const activeMissErrors = displayLedger.filter((r) => !getRowHit(r)).map((r) => getRowErrPct(r));
  const activeHitMae = activeHitErrors.length > 0 ? activeHitErrors.reduce((a, b) => a + b, 0) / activeHitErrors.length : 0;
  const activeMissMae = activeMissErrors.length > 0 ? activeMissErrors.reduce((a, b) => a + b, 0) / activeMissErrors.length : 0;
  const bigMovePenalty = bigMoveTotal > 0 ? (100 - bigMoveHitRatePct) * 0.5 : 0;
  const falseAlarmPenalty = bigMoveFalseAlarms * 2.0;
  const active3CritLoss = (100 - activeHitRatePct) + bigMovePenalty + falseAlarmPenalty + 1.0 * activeHitMae + 2.5 * activeMissMae;

  const mlHits = displayLedger.filter((r) => r.ml_is_hit).length;
  const mlHitRatePct = displayLedger.length > 0 ? (mlHits / displayLedger.length) * 100 : 0;

  const prophetHits = displayLedger.filter((r) => r.prophet_is_hit).length;
  const prophetHitRatePct = displayLedger.length > 0 ? (prophetHits / displayLedger.length) * 100 : 0;

  const last10Ledger = displayLedger.slice(-10);
  const l10Hits = last10Ledger.filter((r) => getRowHit(r)).length;
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
  const minPrice = displayLedger.length
    ? Math.min(...displayLedger.map((r) => Math.min(r.actual_price, getRowPredPrice(r)))) * 0.985
    : 0;
  const maxPrice = displayLedger.length
    ? Math.max(...displayLedger.map((r) => Math.max(r.actual_price, getRowPredPrice(r)))) * 1.015
    : 100;

  // Return Extents
  const maxAbsReturn = displayLedger.length
    ? Math.max(
        ...displayLedger.map((r) =>
          Math.max(Math.abs(r.actual_return_pct), Math.abs(getRowPredReturn(r)))
        ),
        4.0
      ) * 1.15
    : 10;

  const getX = (idx: number) => {
    if (displayLedger.length <= 1) return paddingX;
    return paddingX + (idx / (displayLedger.length - 1)) * innerW;
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
  const actualPricePath = displayLedger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYPrice(r.actual_price).toFixed(1)}`)
    .join(' ');

  const predPricePath = displayLedger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYPrice(getRowPredPrice(r)).toFixed(1)}`)
    .join(' ');

  const actualReturnPath = displayLedger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYReturn(r.actual_return_pct).toFixed(1)}`)
    .join(' ');

  const predReturnPath = displayLedger
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYReturn(getRowPredReturn(r)).toFixed(1)}`)
    .join(' ');

  const hoveredItem = hoveredIdx !== null && displayLedger[hoveredIdx] ? displayLedger[hoveredIdx] : null;

  // Adaptive x-axis label spacing depending on horizon
  const dateInterval = displayLedger.length > 50 ? Math.ceil(displayLedger.length / 8) : 5;

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
              Institutional Order-Flow Walk-Forward Tournament & Multi-Horizon Confluence Engine
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

      {/* ── Interactive Model Perspective & Horizon Switcher Ribbon ────────────── */}
      <div className="glass-panel p-2.5 rounded-xl border border-cyan-500/40 bg-gradient-to-r from-slate-900 via-slate-900/95 to-slate-950 shadow-lg flex flex-wrap items-center justify-between gap-3">
        {/* Model Perspective Switcher */}
        <div className="flex items-center space-x-2">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 font-mono">
            Perspective:
          </span>
          <div className="flex items-center bg-slate-950 rounded-lg border border-slate-800 p-0.5 font-mono text-xs shadow-inner">
            <button
              onClick={() => setActiveModel('champion')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md font-bold transition-all ${
                activeModel === 'champion'
                  ? 'bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20 font-black'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <span>👑</span>
              <span>Crowned Champion (Default)</span>
              <span className="text-[9px] opacity-80 font-normal">
                ({tournament?.grand_champion_key || (tournament?.champion === 'PROPHET_BASE' ? 'Prophet' : tournament?.ml_champion_type || 'Champion')})
              </span>
            </button>

            <button
              onClick={() => setActiveModel('ml')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md font-bold transition-all ${
                activeModel === 'ml'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 font-black'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <span>⚡</span>
              <span>Pure ML Challenger</span>
              <span className="text-[9px] opacity-80 font-normal">
                ({tournament?.ml_champion_type || 'Ridge'})
              </span>
            </button>

            <button
              onClick={() => setActiveModel('prophet')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md font-bold transition-all ${
                activeModel === 'prophet'
                  ? 'bg-purple-500 text-slate-950 shadow-md shadow-purple-500/20 font-black'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <span>📈</span>
              <span>Prophet Baseline</span>
              <span className="text-[9px] opacity-80 font-normal">
                (Univariate Prior)
              </span>
            </button>
          </div>
        </div>

        {/* Horizon Range Switcher: 30D Focus vs 6-Month Full Arena */}
        <div className="flex items-center space-x-2">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 font-mono">
            Horizon:
          </span>
          <div className="flex items-center bg-slate-950 rounded-lg border border-slate-800 p-0.5 font-mono text-xs">
            <button
              onClick={() => setHorizonRange('30d')}
              className={`px-2.5 py-1 rounded text-[10.5px] font-bold transition-all ${
                horizonRange === '30d'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Last 30 Sessions (Focus)
            </button>
            <button
              onClick={() => setHorizonRange('6m')}
              className={`px-2.5 py-1 rounded text-[10.5px] font-bold transition-all ${
                horizonRange === '6m'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              6 Months ({ledger.length} Sessions)
            </button>
          </div>
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
                  <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase border ${
                    activeModel === 'champion'
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                      : activeModel === 'ml'
                      ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30'
                      : 'bg-purple-500/20 text-purple-300 border-purple-500/30'
                  }`}>
                    {activeModel === 'champion'
                      ? `👑 CROWNED CHAMPION (${tournament?.grand_champion_key || (tournament?.champion === 'PROPHET_BASE' ? 'PROPHET' : tournament?.ml_champion_type || 'CHAMPION')})`
                      : activeModel === 'ml'
                      ? '⚡ PURE ML'
                      : '📈 PROPHET BASELINE'}
                  </span>
                </div>
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-[11px] text-slate-400">Current Close:</span>
                  <span className="text-sm font-bold text-white">₺{latestPrice.toFixed(2)}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {/* Projected Percentage Movement & Converted Scale */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-[10.5px] text-slate-400 mb-0.5">Projected Movement (Δ%)</div>
                  <div className="flex items-baseline space-x-2">
                    <span
                      className={`text-2xl font-black font-mono ${
                        isUp ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {isUp ? '+' : ''}
                      {activeExpReturn.toFixed(2)}%
                    </span>
                    <span className="text-sm font-semibold font-mono text-slate-300">
                      → ₺{activeTargetPrice.toFixed(2)}
                    </span>
                  </div>
                  <div className="text-[9.5px] text-slate-500 mt-1 font-mono">
                    Converted Scale (90% CI: [₺{forecast.price_low.toFixed(2)} , ₺{forecast.price_high.toFixed(2)}])
                  </div>
                </div>

                {/* Conviction & Stance */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col justify-between">
                  <div className="text-[10.5px] text-slate-400 mb-0.5">Directional Conviction</div>
                  <div>
                    <span
                      className={`inline-block px-2.5 py-1 rounded text-xs font-black font-mono tracking-wider border ${
                        activeStanceColor === 'emerald'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : activeStanceColor === 'rose'
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : activeStanceColor === 'indigo'
                          ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
                          : activeStanceColor === 'purple'
                          ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                          : 'bg-slate-800 text-slate-300 border-slate-700'
                      }`}
                    >
                      {activeStanceBadge}
                    </span>
                  </div>
                  <div className="text-[9.5px] text-slate-400 mt-1 flex items-center gap-1 font-mono">
                    <span>Stance:</span>
                    <span className="text-white font-semibold">{activeStance}</span>
                  </div>
                </div>

                {/* Candidate Comparison Mini-Box */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col justify-between text-xs font-mono">
                  <div className="text-[10.5px] text-slate-400 mb-0.5">Model Movements Overview (Δ% → Scale)</div>
                  <div className="space-y-1">
                    <div className={`flex items-center justify-between text-[11px] ${activeModel === 'champion' ? 'font-bold' : ''}`}>
                      <span className="text-amber-400">Crowned Champion:</span>
                      <span className="text-white">
                        <span className={forecast.expected_return_pct >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                          {forecast.expected_return_pct >= 0 ? '+' : ''}{forecast.expected_return_pct.toFixed(2)}%
                        </span>{' '}
                        <span className="text-slate-400">→</span> ₺{forecast.target_price.toFixed(2)}
                      </span>
                    </div>
                    <div className={`flex items-center justify-between text-[11px] ${activeModel === 'ml' ? 'font-bold' : ''}`}>
                      <span className="text-indigo-400">ML Challenger:</span>
                      <span className="text-white">
                        <span className={(forecast.ml_expected_return_pct || 0) >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                          {(forecast.ml_expected_return_pct || 0) >= 0 ? '+' : ''}
                          {forecast.ml_expected_return_pct?.toFixed(2)}%
                        </span>{' '}
                        <span className="text-slate-400">→</span> ₺{forecast.ml_target_price?.toFixed(2)}
                      </span>
                    </div>
                    <div className={`flex items-center justify-between text-[11px] ${activeModel === 'prophet' ? 'font-bold' : ''}`}>
                      <span className="text-purple-400">Prophet Base:</span>
                      <span className="text-white">
                        <span className={forecast.prophet_expected_return_pct >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                          {forecast.prophet_expected_return_pct >= 0 ? '+' : ''}
                          {forecast.prophet_expected_return_pct.toFixed(2)}%
                        </span>{' '}
                        <span className="text-slate-400">→</span> ₺{forecast.prophet_target_price.toFixed(2)}
                      </span>
                    </div>
                  </div>
                  <div className="text-[9px] text-slate-400 mt-1">
                    Crowned: <span className="text-amber-300 font-semibold">{tournament?.champion_label || 'Tournament Champion'}</span>
                  </div>
                </div>
              </div>

              {/* Rationale Headline */}
              <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-start space-x-2 text-xs text-slate-300">
                <Zap className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white">{activeHeadline}. </span>
                  <span className="text-slate-400 text-[11px]">{activeRationale}</span>
                </div>
              </div>
            </div>

            {/* Tournament Arena Champion Card */}
            <div className="glass-panel p-4 rounded-xl border border-amber-500/30 bg-slate-900/90 shadow-lg flex flex-col justify-between">
              <div>
                <div className="flex items-center space-x-2 border-b border-slate-800 pb-2 mb-3">
                  <Trophy className="w-4 h-4 text-amber-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-300">
                    {activeModel === 'champion' ? 'Grand Tournament Champion' : activeModel === 'ml' ? 'ML Challenger Perspective' : 'Prophet Prior Perspective'}
                  </span>
                </div>

                <div className="text-center py-2">
                  <div className="text-sm font-black text-white font-mono flex items-center justify-center gap-1.5">
                    <span>{activeModel === 'champion' ? '👑' : activeModel === 'ml' ? '⚡' : '📈'}</span>
                    <span>
                      {activeModel === 'champion'
                        ? tournament?.champion_label || 'Crowned Champion'
                        : activeModel === 'ml'
                        ? `${tournament?.ml_champion_type || 'Pure ML'} Challenger`
                        : 'Prophet Baseline Prior'}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {horizonRange === '30d' ? '30-Day Walk-Forward Focus' : `6-Month Multi-Horizon (${displayLedger.length} Sessions)`}
                  </div>
                </div>

                <div className="space-y-1.5 mt-2 font-mono text-xs">
                  {/* Criterion 1: Directional Hit % */}
                  <div className="flex items-center justify-between p-1.5 px-2 rounded bg-slate-950/60 border border-slate-800">
                    <span className="text-slate-400 text-[11px]">1. Directional Hits:</span>
                    <span className="font-bold text-emerald-400 text-xs">
                      {activeHits}/{displayLedger.length} ({activeHitRatePct.toFixed(1)}%)
                    </span>
                  </div>

                  {/* Actionable Big Move Opportunity Strike Rate (≥ ±2% Moves) */}
                  <div className="p-1.5 px-2 rounded bg-amber-500/10 border border-amber-500/25 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-amber-300 text-[11px] font-semibold flex items-center gap-1">
                        <span>⚡</span> Actionable Big Move Strike:
                      </span>
                      <span className="font-bold text-amber-400 text-xs">
                        {bigMoveHits}/{bigMoveTotal} ({bigMoveHitRatePct.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[8.5px] text-slate-400 font-mono pt-0.5 border-t border-amber-500/20">
                      <span>Signalled: <strong className="text-amber-300">{bigMoveSignalled}</strong></span>
                      <span>Missed: <strong className="text-rose-400">{bigMoveMissed}</strong></span>
                      <span>False Alarms: <strong className="text-rose-400">{bigMoveFalseAlarms}</strong></span>
                    </div>
                  </div>

                  {/* Criterion 2: Hit Return Error (Hit MAE) */}
                  <div className="flex items-center justify-between p-1.5 px-2 rounded bg-slate-950/60 border border-slate-800">
                    <span className="text-slate-400 text-[11px]">2. Hit Return Error (MAE %):</span>
                    <span className="font-semibold text-cyan-300 text-xs">
                      {activeHitMae.toFixed(2)}%
                    </span>
                  </div>

                  {/* Criterion 3: Miss Return Error (Miss MAE) */}
                  <div className="flex items-center justify-between p-1.5 px-2 rounded bg-slate-950/60 border border-slate-800">
                    <span className="text-slate-400 text-[11px]">3. Miss Return Error (MAE %):</span>
                    <span className="font-semibold text-rose-300 text-xs">
                      {activeMissMae.toFixed(2)}%
                    </span>
                  </div>

                  {/* Composite Tournament Loss */}
                  <div className="flex items-center justify-between p-1.5 px-2 rounded bg-amber-500/10 border border-amber-500/30">
                    <span className="text-amber-300 font-bold text-[11px]">Tournament Composite Loss:</span>
                    <span className="font-bold text-amber-400 text-xs">
                      {active3CritLoss.toFixed(2)}
                    </span>
                  </div>

                  {/* Peer Arena Comparison */}
                  <div className="pt-1.5 border-t border-slate-800/80 grid grid-cols-2 gap-1 text-[10px]">
                    <div className="p-1 rounded bg-slate-950/40 border border-slate-800/60 flex justify-between">
                      <span className="text-slate-400">ML Alone:</span>
                      <span className="text-indigo-300 font-semibold">{mlHits}/{displayLedger.length} ({mlHitRatePct.toFixed(0)}%)</span>
                    </div>
                    <div className="p-1 rounded bg-slate-950/40 border border-slate-800/60 flex justify-between">
                      <span className="text-slate-400">Prophet:</span>
                      <span className="text-purple-300 font-semibold">{prophetHits}/{displayLedger.length} ({prophetHitRatePct.toFixed(0)}%)</span>
                    </div>
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

          {/* ── Visual Walk-Forward Chart: Predicted vs What Happened ────────────── */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800 bg-slate-900/90 shadow-xl space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white tracking-wide">
                  Walk-Forward Performance: {activeModel === 'champion' ? (tournament?.champion_label || 'Crowned Champion') : activeModel === 'ml' ? 'Pure ML' : 'Prophet'} vs Realized
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  ({horizonRange === '30d' ? 'Last 30 Sessions' : `${displayLedger.length} Sessions (6 Months)`})
                </span>
              </div>

              {/* Chart Mode Toggle & Stats Pill */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center font-mono text-[9px] gap-1.5">
                  <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                    {activeModel === 'champion' ? 'Champion' : activeModel === 'ml' ? 'Pure ML' : 'Prophet'}:{' '}
                    <span className="text-emerald-400 font-bold">{activeHits}✓</span> /{' '}
                    <span className="text-rose-400 font-bold">{activeMiss}✗</span> ({activeHitRatePct.toFixed(1)}%)
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                    Last 10D: <span className="text-emerald-400 font-bold">{l10Hits}✓</span> /{' '}
                    <span className="text-rose-400 font-bold">{l10Miss}✗</span> ({l10RatePct.toFixed(0)}%)
                  </span>
                  <span className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-semibold" title="Sessions where |actual return| >= 2.0%">
                    ⚡ Big Moves: <span className="text-amber-400 font-bold">{bigMoveHits}/{bigMoveTotal} ({bigMoveHitRatePct.toFixed(0)}%)</span>
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
                      stroke={activeModel === 'champion' ? '#f59e0b' : activeModel === 'ml' ? '#6366f1' : '#d946ef'}
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
                      stroke={activeModel === 'champion' ? '#f59e0b' : activeModel === 'ml' ? '#6366f1' : '#d946ef'}
                      strokeWidth="2"
                      strokeDasharray="4 3"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </>
                )}

                {/* Interactive Points on each session */}
                {displayLedger.map((r, i) => {
                  const x = getX(i);
                  const yActual = chartView === 'price' ? getYPrice(r.actual_price) : getYReturn(r.actual_return_pct);
                  const yPred = chartView === 'price' ? getYPrice(getRowPredPrice(r)) : getYReturn(getRowPredReturn(r));
                  const isHovered = hoveredIdx === i;
                  const isHit = getRowHit(r);
                  const predColor = activeModel === 'champion' ? '#f59e0b' : activeModel === 'ml' ? '#6366f1' : '#d946ef';
                  const actualColor = '#38bdf8';
                  const baseR = displayLedger.length > 60 ? 2.2 : 3.5;
                  const dotR = isHovered ? baseR + 2.0 : baseR;

                  return (
                    <g
                      key={i}
                      onMouseEnter={() => setHoveredIdx(i)}
                      className="cursor-pointer"
                    >
                      {/* Vertical Guideline & Inter-point Delta on hover */}
                      {isHovered && (
                        <>
                          <line
                            x1={x}
                            y1={paddingY}
                            x2={x}
                            y2={paddingY + innerH}
                            stroke="#64748b"
                            strokeWidth="1"
                            strokeDasharray="2 2"
                          />
                          <line
                            x1={x}
                            y1={yActual}
                            x2={x}
                            y2={yPred}
                            stroke={isHit ? '#10b981' : '#f43f5e'}
                            strokeWidth="2"
                            strokeDasharray="2 2"
                          />
                        </>
                      )}

                      {/* Actual Price Point (Solid Sky Blue Circle) */}
                      <circle
                        cx={x}
                        cy={yActual}
                        r={dotR}
                        fill={actualColor}
                        stroke="#0f172a"
                        strokeWidth="1.2"
                      />

                      {/* Predicted Price Point: Green for Hit (✓), Red for Miss (✗), with Model Border */}
                      <circle
                        cx={x}
                        cy={yPred}
                        r={dotR}
                        fill={isHit ? '#10b981' : '#f43f5e'}
                        stroke={predColor}
                        strokeWidth="1.8"
                      />

                      {/* Outer Glow Halo on Hover */}
                      {isHovered && (
                        <circle
                          cx={x}
                          cy={yPred}
                          r={dotR + 2.5}
                          fill="none"
                          stroke={isHit ? '#10b981' : '#f43f5e'}
                          strokeWidth="1.5"
                          opacity="0.85"
                        />
                      )}

                      {/* X Axis Date Labels */}
                      {(i % dateInterval === 0 || i === displayLedger.length - 1) && (
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
                        getRowHit(hoveredItem)
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                      }`}
                    >
                      {getRowHit(hoveredItem) ? '✓ CORRECT' : '✗ WRONG'}
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
                      <span className={activeModel === 'champion' ? 'text-amber-400' : activeModel === 'ml' ? 'text-indigo-400' : 'text-fuchsia-400'}>
                        {activeModel === 'champion' ? 'Champion Pred:' : activeModel === 'ml' ? 'ML Pred:' : 'Prophet Pred:'}
                      </span>
                      <span className="font-bold text-white">
                        ₺{getRowPredPrice(hoveredItem).toFixed(2)}{' '}
                        <span
                          className={
                            getRowPredReturn(hoveredItem) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }
                        >
                          ({getRowPredReturn(hoveredItem) >= 0 ? '+' : ''}
                          {getRowPredReturn(hoveredItem).toFixed(2)}%)
                        </span>
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Error:</span>
                      <span className="text-slate-200">{getRowErrPct(hoveredItem).toFixed(2)}%</span>
                    </div>

                    {/* Breakdown Details */}
                    <div className="flex items-center justify-between border-t border-slate-800 pt-1 mt-1 text-[10px]">
                      <span className="text-indigo-300">ML Alone:</span>
                      <span className="text-slate-300">
                        ₺{hoveredItem.ml_pred_price.toFixed(2)}{' '}
                        <span
                          className={
                            hoveredItem.ml_is_hit
                              ? 'text-emerald-400 font-bold'
                              : 'text-rose-400 font-bold'
                          }
                        >
                          ({hoveredItem.ml_is_hit ? '✓' : '✗'})
                        </span>
                      </span>
                    </div>

                    <div className="flex items-center justify-between border-t border-slate-800/60 pt-0.5 mt-0.5 text-[10px]">
                      <span className="text-purple-400">Prophet Pred:</span>
                      <span className="text-slate-300">
                        ₺{hoveredItem.prophet_pred_price.toFixed(2)}{' '}
                        <span
                          className={
                            hoveredItem.prophet_is_hit
                              ? 'text-emerald-400 font-bold'
                              : 'text-rose-400 font-bold'
                          }
                        >
                          ({hoveredItem.prophet_is_hit ? '✓' : '✗'})
                        </span>
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
                    <span className="w-2.5 h-2.5 rounded-full bg-sky-400 border border-slate-900 inline-block" />
                    <span className="w-3 h-0.5 bg-sky-400 rounded-full inline-block" />
                    <span>Actual Price Point & Realization</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className={`w-3 h-0.5 border-b border-dashed inline-block ${
                      activeModel === 'champion'
                        ? 'border-amber-400'
                        : activeModel === 'ml'
                        ? 'border-indigo-400'
                        : 'border-fuchsia-400'
                    }`} />
                    <span>{activeModel === 'champion' ? `Champion (${tournament?.grand_champion_key || 'Amber'})` : activeModel === 'ml' ? 'Pure ML (Indigo)' : 'Prophet (Fuchsia)'}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 border border-slate-900 inline-block" />
                    <span>Direction Hit (✓)</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500 border border-slate-900 inline-block" />
                    <span>Direction Miss (✗)</span>
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

          {/* ── Walk-Forward Reality Ledger Table ──────────────────────────────────── */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800 bg-slate-900/90 shadow-xl space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
              <div className="flex items-center space-x-2">
                <Calendar className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white tracking-wide">
                  Out-of-Sample Walk-Forward Reality Ledger ({displayLedger.length} Sessions)
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  Showing {activeModel === 'champion' ? (tournament?.champion_label || 'Crowned Champion') : activeModel === 'ml' ? 'Pure ML' : 'Prophet'} Perspective
                </span>
              </div>
              <div className="flex items-center gap-1.5 font-mono text-[8.5px]">
                <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                  {activeModel === 'champion' ? 'Champion' : activeModel === 'ml' ? 'ML' : 'Prophet'}:{' '}
                  <span className="text-emerald-400 font-bold">{activeHits}✓</span> /{' '}
                  <span className="text-rose-400 font-bold">{activeMiss}✗</span> ({activeHitRatePct.toFixed(1)}%)
                </span>
                <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                  Last 10D: <span className="text-emerald-400 font-bold">{l10Hits}✓</span> /{' '}
                  <span className="text-rose-400 font-bold">{l10Miss}✗</span> ({l10RatePct.toFixed(0)}%)
                </span>
                <span className="px-1.5 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-400">
                  ML Hit Rate: <span className="text-indigo-300 font-bold">{mlHitRatePct.toFixed(1)}%</span>
                </span>
                <span className="px-1.5 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-400">
                  Prophet Hit Rate: <span className="text-purple-300 font-bold">{prophetHitRatePct.toFixed(1)}%</span>
                </span>
                <span className="px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-semibold" title="Sessions where |actual return| >= 2.0%">
                  ⚡ Big Moves: <span className="text-amber-400 font-bold">{bigMoveHits}/{bigMoveTotal} ({bigMoveHitRatePct.toFixed(0)}%)</span>
                </span>
              </div>
            </div>

            <div className="overflow-x-auto max-h-80 overflow-y-auto rounded-lg border border-slate-800/90">
              <table className="w-full text-[9px] font-mono border-collapse">
                <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 z-10 font-sans">
                  <tr className="text-slate-400">
                    <th className="text-left px-2 py-1.5">Date</th>
                    <th className="text-right px-2 py-1.5">Actual Close</th>
                    <th className="text-right px-2 py-1.5 text-amber-300">
                      {activeModel === 'champion' ? 'Champion Pred' : activeModel === 'ml' ? 'Pure ML Pred' : 'Prophet Pred'}
                    </th>
                    <th className="text-center px-2 py-1.5 text-cyan-300">
                      {activeModel === 'champion' ? 'Champion Hit?' : activeModel === 'ml' ? 'ML Hit?' : 'Prophet Hit?'}
                    </th>
                    {activeModel !== 'ml' && (
                      <th className="text-right px-2 py-1.5 text-indigo-400">ML Alone</th>
                    )}
                    {activeModel !== 'prophet' && (
                      <th className="text-right px-2 py-1.5 text-purple-300">Prophet Base</th>
                    )}
                    <th className="text-right px-2 py-1.5 text-amber-200">XU030</th>
                    <th className="text-center px-2 py-1.5">BofA MLB Did</th>
                    <th className="text-center px-2 py-1.5">BIG5 Did</th>
                    <th className="text-center px-2 py-1.5">KAMU Did</th>
                    <th className="text-center px-2 py-1.5">Winner</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                  {displayLedger.map((row) => {
                    const isActUp = row.actual_return_pct > 0.02;
                    const isActDown = row.actual_return_pct < -0.02;
                    const isBigMove = Math.abs(row.actual_return_pct) >= 2.0;
                    const predRet = getRowPredReturn(row);
                    const isPredUp = predRet > 0.02;
                    const isPredDown = predRet < -0.02;
                    const isMlUp = (row.ml_pred_return_pct || 0) > 0.02;
                    const isMlDown = (row.ml_pred_return_pct || 0) < -0.02;
                    const pRet = row.prophet_pred_return_pct !== undefined ? row.prophet_pred_return_pct : 0;
                    const isRowHit = getRowHit(row);
                    const bothMiss = !row.ml_is_hit && !row.prophet_is_hit;

                    return (
                      <tr key={row.date} className="hover:bg-slate-800/40 text-slate-300">
                        <td className="px-2 py-1 text-slate-400 font-bold">
                          <div className="flex items-center gap-1">
                            <span>{row.date.slice(5)}</span>
                            {isBigMove && (
                              <span className="px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 text-[7px] font-black border border-amber-500/40" title="Big Move (≥ ±2%)">
                                ⚡2%
                              </span>
                            )}
                          </div>
                        </td>
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
                          <div className="font-semibold text-white">₺{getRowPredPrice(row).toFixed(2)}</div>
                          <div
                            className={`text-[8px] font-bold flex items-center justify-end gap-1 ${
                              isPredUp ? 'text-emerald-400' : isPredDown ? 'text-rose-400' : 'text-slate-400'
                            }`}
                          >
                            <span>
                              {isPredUp ? '▲ +' : isPredDown ? '▼ ' : '■ '}
                              {predRet.toFixed(2)}%
                            </span>
                            <span className="text-[7.5px] text-slate-500 font-normal">
                              ({getRowErrPct(row).toFixed(1)}%)
                            </span>
                          </div>
                        </td>
                        <td className="text-center px-2 py-1">
                          <span
                            className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[8px] font-black border tracking-wider ${
                              isRowHit
                                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                                : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                            }`}
                          >
                            <span>{isRowHit ? '✓' : '✗'}</span>
                            <span>{isRowHit ? 'CORRECT' : 'WRONG'}</span>
                          </span>
                        </td>
                        {activeModel !== 'ml' && (
                          <td className="text-right px-2 py-1 text-slate-300">
                            <div className="font-semibold">₺{row.ml_pred_price.toFixed(2)}</div>
                            <div className="text-[7.5px] flex items-center justify-end gap-1">
                              <span className={isMlUp ? 'text-emerald-400' : isMlDown ? 'text-rose-400' : 'text-slate-400'}>
                                {(row.ml_pred_return_pct || 0) > 0 ? '+' : ''}{(row.ml_pred_return_pct || 0).toFixed(2)}%
                              </span>
                              <span className={row.ml_is_hit ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                                {row.ml_is_hit ? '✓' : '✗'}
                              </span>
                            </div>
                          </td>
                        )}
                        {activeModel !== 'prophet' && (
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
                        )}
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

export default OracleHubDashboard;
