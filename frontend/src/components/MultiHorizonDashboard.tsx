import React, { useState, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  fetchMultiHorizonForecast,
  fetchMultiHorizonBacktest,
} from '../api/client';
import { formatPercent } from '../utils/formatters';
import {
  Layers,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Award,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  Clock,
  Sparkles,
  BarChart3,
  AlertTriangle,
  Info,
} from 'lucide-react';

interface MultiHorizonDashboardProps {
  symbol?: string;
  onSelectSymbol?: (symbol: string) => void;
  onNavigateTab?: (tab: string) => void;
}

const HORIZON_ORDER = ['3d', '5d', '10d', '15d', '30d'] as const;

export const MultiHorizonDashboard: React.FC<MultiHorizonDashboardProps> = ({
  symbol = 'THYAO',
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbol);
  const [activeHorizon, setActiveHorizon] = useState<string>('5d');
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // Sync with global symbol changes from App top navbar
  useEffect(() => {
    if (symbol && symbol !== selectedSymbol) {
      setSelectedSymbol(symbol);
    }
  }, [symbol]);

  // 1. Fetch Multi-Horizon Forecast Payload
  const {
    data: forecastData,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['multi-horizon-forecast', selectedSymbol],
    queryFn: () => fetchMultiHorizonForecast(selectedSymbol),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  // 2. Fetch Backtest Ledger for active inspected horizon
  const {
    data: backtestData,
    isLoading: isLoadingBt,
  } = useQuery({
    queryKey: ['multi-horizon-backtest', selectedSymbol, activeHorizon],
    queryFn: () => fetchMultiHorizonBacktest(selectedSymbol, activeHorizon, 30),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  const horizons = forecastData?.horizons || {};
  const activeHorizonData = horizons[activeHorizon];
  const activeMeta = activeHorizonData?.meta;

  // Ledger: prefer dedicated backtest endpoint data if available, fallback to forecastData
  const ledger = useMemo(() => {
    if (backtestData && backtestData.length > 0) {
      return backtestData;
    }
    return activeHorizonData?.backtest_ledger || [];
  }, [backtestData, activeHorizonData]);

  // Stance styling helper
  const getStanceBadge = (stance: string) => {
    switch (stance?.toUpperCase()) {
      case 'BULLISH':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';
      case 'BEARISH':
        return 'bg-rose-500/20 text-rose-400 border-rose-500/40';
      default:
        return 'bg-slate-700/40 text-slate-300 border-slate-600/40';
    }
  };

  const getConvictionBadge = (conviction: string) => {
    switch (conviction?.toUpperCase()) {
      case 'STRONG_BUY':
        return 'bg-emerald-500/30 text-emerald-300 border-emerald-500/60 shadow-sm shadow-emerald-500/20';
      case 'BUY':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';
      case 'WEAK_BUY':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'STRONG_SELL':
        return 'bg-rose-500/30 text-rose-300 border-rose-500/60 shadow-sm shadow-rose-500/20';
      case 'SELL':
        return 'bg-rose-500/20 text-rose-400 border-rose-500/40';
      case 'WEAK_SELL':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Top Header Deck */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800/80 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-wrap items-center justify-between gap-4 relative z-10">
          <div>
            <div className="flex items-center space-x-3">
              <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-600 shadow-lg shadow-indigo-500/20">
                <Layers className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center space-x-2.5">
                  <h1 className="text-xl font-bold tracking-tight text-white font-mono">
                    {selectedSymbol}
                  </h1>
                  <span className="text-xs px-2.5 py-0.5 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 font-semibold font-mono">
                    MULTI-HORIZON PREDICTIVE HUB
                  </span>
                  <span className="text-xs text-slate-400">
                    {forecastData?.company_name || 'Turkish Airlines'} • {forecastData?.sector || 'Aviation'}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  Zero-Lookahead Institutional Target Predictions Across 3D, 5D, 10D, 15D, & 30D Forward Market Windows
                </p>
              </div>
            </div>
          </div>

          {/* Price, As-of Date & Refresh Button */}
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-xs text-slate-400 font-mono">Current Session Close</div>
              <div className="text-lg font-bold text-white font-mono">
                ₺{forecastData?.current_price?.toFixed(2) || '---'}
              </div>
            </div>

            <div className="text-right border-l border-slate-800 pl-4 hidden sm:block">
              <div className="text-xs text-slate-400 font-mono">As of Date</div>
              <div className="text-xs font-semibold text-slate-200 font-mono">
                {forecastData?.as_of_date || '---'}
              </div>
            </div>

            <button
              onClick={() => refetch()}
              disabled={isFetching}
              className="p-2.5 rounded-xl bg-slate-900 border border-slate-700/80 hover:border-cyan-500/50 text-slate-300 hover:text-white transition-all shadow-inner disabled:opacity-50"
              title="Refresh Forecast Data"
            >
              <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {/* 5 Forward Prediction Cards (3D, 5D, 10D, 15D, 30D) */}
      <div>
        <div className="flex items-center justify-between mb-3 px-1">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-bold text-white tracking-wide">
              FORWARD MARKET PREDICTION HORIZONS
            </h2>
            <span className="text-[11px] text-slate-400">
              (Arithmetic average price change over the forward trading sessions)
            </span>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Click any card to inspect audited walk-forward performance
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3.5">
          {HORIZON_ORDER.map((hKey) => {
            const hData = horizons[hKey];
            const meta = hData?.meta;
            const fc = hData?.live_forecast;
            const isSelected = activeHorizon === hKey;
            const expRet = fc?.expected_return_pct ?? 0.0;
            const isBullish = expRet >= 0.25;
            const isBearish = expRet <= -0.25;

            return (
              <div
                key={hKey}
                onClick={() => setActiveHorizon(hKey)}
                className={`glass-panel p-4 rounded-xl border transition-all cursor-pointer relative overflow-hidden group ${
                  isSelected
                    ? 'border-cyan-500/80 bg-cyan-950/20 shadow-lg shadow-cyan-500/10'
                    : 'border-slate-800/80 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60'
                }`}
              >
                {/* Active Indicator Top Bar */}
                {isSelected && (
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-indigo-500" />
                )}

                {/* Horizon Label & Days Badge */}
                <div className="flex items-center justify-between mb-2.5">
                  <div className="flex items-center space-x-1.5">
                    <span className="font-mono font-black text-sm text-white tracking-wider">
                      {hKey.toUpperCase()}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      ({meta?.horizon_days || (hKey === '3d' ? 3 : hKey === '5d' ? 5 : hKey === '10d' ? 10 : hKey === '15d' ? 15 : 30)} Market Days)
                    </span>
                  </div>
                  <span
                    className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${getStanceBadge(
                      fc?.stance || 'NEUTRAL'
                    )}`}
                  >
                    {fc?.stance || 'NEUTRAL'}
                  </span>
                </div>

                {/* Expected Return % & Direction Gauge */}
                <div className="my-2">
                  <div className="flex items-baseline space-x-1.5">
                    <span
                      className={`text-2xl font-black font-mono tracking-tight ${
                        isBullish
                          ? 'text-emerald-400'
                          : isBearish
                          ? 'text-rose-400'
                          : 'text-slate-300'
                      }`}
                    >
                      {formatPercent(expRet)}
                    </span>
                    {isBullish ? (
                      <TrendingUp className="w-4 h-4 text-emerald-400 inline" />
                    ) : isBearish ? (
                      <TrendingDown className="w-4 h-4 text-rose-400 inline" />
                    ) : null}
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                    Expected Avg Price: <span className="text-white font-bold">₺{fc?.target_price?.toFixed(2) || '---'}</span>
                  </div>
                </div>

                {/* Credible Price Range Envelope */}
                <div className="bg-slate-900/80 rounded-lg p-2 border border-slate-800/80 my-2.5 text-[11px] font-mono">
                  <div className="text-[9px] text-slate-400 uppercase tracking-wider mb-0.5">
                    Vol-Calibrated Range
                  </div>
                  <div className="text-slate-200 flex justify-between">
                    <span>₺{fc?.price_low?.toFixed(2) || '---'}</span>
                    <span className="text-slate-500">──</span>
                    <span>₺{fc?.price_high?.toFixed(2) || '---'}</span>
                  </div>
                </div>

                {/* Crowned Model & 30D Hit Rate */}
                <div className="space-y-1.5 text-xs pt-2 border-t border-slate-800/60 font-mono">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-400 flex items-center gap-1">
                      <Award className="w-3 h-3 text-amber-400" /> Crown:
                    </span>
                    <span className="font-semibold text-amber-300">
                      {meta?.crowned_model || 'Auto'} ({meta?.crowned_horizon || '6m'})
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-400 flex items-center gap-1">
                      <ShieldCheck className="w-3 h-3 text-emerald-400" /> Win Rate:
                    </span>
                    <span className="font-bold text-emerald-400">
                      {meta?.hit_rate_pct != null ? `${meta.hit_rate_pct.toFixed(1)}%` : '---'}
                    </span>
                  </div>

                  {/* Actual Time Window Used */}
                  <div className="pt-1.5 border-t border-slate-800/40 text-[10px]">
                    <div className="text-slate-400 mb-0.5">Actual Training Window:</div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-200 font-semibold truncate max-w-[130px]" title={fc?.actual_window_desc}>
                        {fc?.actual_window_desc || `${meta?.training_lookback_sessions || 126} sessions`}
                      </span>
                      {fc?.data_sufficiency_status === 'PARTIAL_ADAPTED' ? (
                        <span className="px-1 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 text-[8px] font-bold">
                          PARTIAL
                        </span>
                      ) : fc?.data_sufficiency_status === 'INSUFFICIENT' ? (
                        <span className="px-1 py-0.2 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40 text-[8px] font-bold">
                          &lt; 3M STOP
                        </span>
                      ) : (
                        <span className="px-1 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[8px] font-bold">
                          FULL
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Conviction Badge & Playbook */}
                <div className="mt-3 pt-2 border-t border-slate-800/40 flex items-center justify-between">
                  <span
                    className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border ${getConvictionBadge(
                      fc?.conviction || 'NEUTRAL'
                    )}`}
                  >
                    {fc?.conviction?.replace('_', ' ') || 'NEUTRAL'}
                  </span>
                  <span className="text-[9px] text-slate-400 font-mono truncate max-w-[100px]" title={fc?.playbook}>
                    {fc?.playbook?.replace(/_/g, ' ') || 'Range'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Term Structure Trajectory Visualizer */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800/80 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white tracking-wide">
              MULTI-HORIZON EXPECTED RETURN TERM STRUCTURE
            </h3>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Forward Term Structure Curve (3D → 5D → 10D → 15D → 30D)
          </div>
        </div>

        {/* Visual Term Structure Bar & Slope Ribbon */}
        <div className="grid grid-cols-5 gap-3 pt-4 pb-2">
          {HORIZON_ORDER.map((hKey) => {
            const hData = horizons[hKey];
            const expRet = hData?.live_forecast?.expected_return_pct ?? 0.0;
            const isBullish = expRet >= 0.25;
            const isBearish = expRet <= -0.25;
            const isInspected = activeHorizon === hKey;

            // Normalize height for bar (bounded between -15% and +15%)
            const barHeightPct = Math.min(100, (Math.abs(expRet) / 15.0) * 100);

            return (
              <div
                key={hKey}
                onClick={() => setActiveHorizon(hKey)}
                className={`flex flex-col items-center cursor-pointer p-3 rounded-xl border transition-all ${
                  isInspected
                    ? 'border-cyan-500/60 bg-cyan-950/20'
                    : 'border-slate-800/60 bg-slate-900/30 hover:border-slate-700'
                }`}
              >
                <span className="text-xs font-mono font-bold text-white mb-2">
                  {hKey.toUpperCase()}
                </span>

                {/* Return Bar */}
                <div className="w-12 h-28 bg-slate-950/80 rounded-lg flex items-center justify-center p-1 relative border border-slate-800/80">
                  {/* Zero Line */}
                  <div className="absolute left-0 right-0 top-1/2 h-[1px] bg-slate-700/80 z-10" />

                  <div
                    className={`w-8 rounded-sm transition-all ${
                      isBullish
                        ? 'bg-gradient-to-t from-emerald-600 to-emerald-400 self-end'
                        : isBearish
                        ? 'bg-gradient-to-b from-rose-600 to-rose-400 self-start'
                        : 'bg-slate-600 self-center h-1'
                    }`}
                    style={{
                      height: `${Math.max(4, barHeightPct * 0.5)}%`,
                    }}
                  />
                </div>

                <span
                  className={`text-xs font-mono font-bold mt-2 ${
                    isBullish ? 'text-emerald-400' : isBearish ? 'text-rose-400' : 'text-slate-400'
                  }`}
                >
                  {formatPercent(expRet)}
                </span>
                <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                  ₺{hData?.live_forecast?.target_price?.toFixed(2) || '---'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Audited Zero-Lookahead Walk-Forward Backtest Performance Ledger */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800/80 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-sm font-bold text-white tracking-wide">
                  AUDITED WALK-FORWARD REALITY LEDGER ({activeHorizon.toUpperCase()})
                </h3>
                <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 font-mono font-semibold">
                  {activeMeta?.crowned_model || 'Algorithm'} ({activeMeta?.crowned_horizon || '6M'})
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Session-by-session zero-lookahead backtest tracking over trailing 30 evaluated sessions
              </p>
            </div>
          </div>

          {/* Horizon Selection Tabs */}
          <div className="flex items-center space-x-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
            {HORIZON_ORDER.map((hKey) => (
              <button
                key={hKey}
                onClick={() => setActiveHorizon(hKey)}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all ${
                  activeHorizon === hKey
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {hKey.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* 5 Scorecard KPI Metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400 font-mono uppercase">Directional Hit Rate</div>
            <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
              {activeMeta?.hit_rate_pct != null ? `${activeMeta.hit_rate_pct.toFixed(1)}%` : '---'}
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              {activeMeta?.hits || 0} hits / {activeMeta?.total_evals || 0} sessions
            </div>
          </div>

          <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400 font-mono uppercase">Mean Absolute Error</div>
            <div className="text-lg font-bold text-slate-200 font-mono mt-0.5">
              {activeMeta?.mae_pct != null ? `${activeMeta.mae_pct.toFixed(2)}%` : '---'}
            </div>
            <div className="text-[10px] text-slate-400 font-mono">Overall price spread</div>
          </div>

          <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400 font-mono uppercase">Hit MAE vs Miss MAE</div>
            <div className="text-lg font-bold text-cyan-400 font-mono mt-0.5">
              {activeMeta?.hit_mae_pct != null ? `${activeMeta.hit_mae_pct.toFixed(2)}%` : '---'}{' '}
              <span className="text-xs text-slate-500">/</span>{' '}
              <span className="text-rose-400">
                {activeMeta?.miss_mae_pct != null ? `${activeMeta.miss_mae_pct.toFixed(2)}%` : '---'}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 font-mono">Calibration & safety</div>
          </div>

          <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400 font-mono uppercase">3-Criteria Loss</div>
            <div className="text-lg font-bold text-amber-400 font-mono mt-0.5">
              {activeMeta?.loss != null ? activeMeta.loss.toFixed(2) : '---'}
            </div>
            <div className="text-[10px] text-slate-400 font-mono">Referee Tournament Score</div>
          </div>

          <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400 font-mono uppercase">Data Sufficiency</div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className={`text-lg font-bold font-mono ${
                (activeMeta?.sessions_skipped_insufficient ?? 0) > 0
                  ? 'text-rose-400'
                  : (activeMeta?.sessions_with_partial_history ?? 0) > 0
                  ? 'text-amber-400'
                  : 'text-emerald-400'
              }`}>
                {activeMeta?.data_sufficiency_pct != null ? `${activeMeta.data_sufficiency_pct.toFixed(0)}%` : '100%'}
              </span>
              <span className={`text-[8px] font-bold px-1 py-0.2 rounded border font-mono ${
                (activeMeta?.sessions_skipped_insufficient ?? 0) > 0
                  ? 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                  : (activeMeta?.sessions_with_partial_history ?? 0) > 0
                  ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                  : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              }`}>
                {(activeMeta?.sessions_skipped_insufficient ?? 0) > 0
                  ? `${activeMeta?.sessions_skipped_insufficient} STOPPED`
                  : (activeMeta?.sessions_with_partial_history ?? 0) > 0
                  ? `${activeMeta?.sessions_with_partial_history} PARTIAL`
                  : 'FULL'}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 font-mono truncate" title={`Target: ${activeMeta?.target_lookback_months || 6}M (${activeMeta?.target_lookback_sessions || 126}d) • Hard Stop: <3M (63d)`}>
              Target {activeMeta?.target_lookback_months || 6}M • Min 3M
            </div>
          </div>
        </div>

        {/* Data Sufficiency Alert / Info Banner */}
        {(activeMeta?.sessions_skipped_insufficient ?? 0) > 0 && (
          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-start space-x-2.5 text-xs text-rose-200">
            <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-bold text-rose-300">3-Month Training Hard Stop Active: </span>
              {activeMeta?.sessions_skipped_insufficient} historical sessions were skipped without predicting because the stock had fewer than 63 completed sessions (&lt; 3 months) at evaluation time.
              {activeMeta?.sessions_with_partial_history ? ` Additionally, ${activeMeta.sessions_with_partial_history} sessions used an adapted window between 3 months and the target lookback.` : ''}
            </div>
          </div>
        )}

        {(activeMeta?.sessions_skipped_insufficient ?? 0) === 0 && (activeMeta?.sessions_with_partial_history ?? 0) > 0 && (
          <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start space-x-2.5 text-xs text-amber-200">
            <Info className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-bold text-amber-300">Adapted Historical Lookback: </span>
              {activeMeta?.sessions_with_partial_history} of {activeMeta?.total_evals || 0} evaluated sessions utilized an expanding lookback window (≥ 63 sessions / 3 months) as history accumulated before reaching the target {activeMeta?.target_lookback_months || 12}M depth.
            </div>
          </div>
        )}

        {/* Audit Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-xs font-mono text-left">
            <thead className="bg-slate-900/90 text-slate-400 text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Decision Date</th>
                <th className="py-2.5 px-3">Target Horizon Dates</th>
                <th className="py-2.5 px-3 text-center">Training Window</th>
                <th className="py-2.5 px-3 text-right">Close Price</th>
                <th className="py-2.5 px-3 text-right">Predicted Avg</th>
                <th className="py-2.5 px-3 text-right">Realized Avg</th>
                <th className="py-2.5 px-3 text-right">Pred Return</th>
                <th className="py-2.5 px-3 text-right">Actual Return</th>
                <th className="py-2.5 px-3 text-right">Error %</th>
                <th className="py-2.5 px-3 text-center">Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
              {isLoadingBt || isLoading ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-slate-400 font-mono">
                    Loading audited backtest data...
                  </td>
                </tr>
              ) : ledger.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-slate-400 font-mono">
                    No audited sessions available for this horizon yet.
                  </td>
                </tr>
              ) : (
                ledger.map((item, idx) => {
                  const isHit = item.is_hit;
                  const isPending = item.is_pending;
                  const isHovered = hoveredIdx === idx;

                  return (
                    <tr
                      key={`${item.trade_date}-${idx}`}
                      onMouseEnter={() => setHoveredIdx(idx)}
                      onMouseLeave={() => setHoveredIdx(null)}
                      className={`transition-colors ${
                        isHovered ? 'bg-slate-800/40' : 'hover:bg-slate-900/40'
                      }`}
                    >
                      <td className="py-2 px-3 text-white font-medium">
                        {item.trade_date}
                      </td>
                      <td className="py-2 px-3 text-slate-400 text-[11px]">
                        {item.target_date_start} → {item.target_date_end}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <div className="flex items-center justify-center gap-1.5">
                          <span className="text-[11px] text-slate-300">
                            {item.actual_training_sessions ? `${item.actual_training_sessions}d (~${item.actual_training_months}M)` : '---'}
                          </span>
                          {item.data_sufficiency_status === 'PARTIAL_ADAPTED' ? (
                            <span className="text-[8px] font-bold px-1 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30">
                              PARTIAL
                            </span>
                          ) : item.data_sufficiency_status === 'FULL' ? (
                            <span className="text-[8px] font-bold px-1 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                              FULL
                            </span>
                          ) : null}
                        </div>
                      </td>
                      <td className="py-2 px-3 text-right text-slate-300">
                        ₺{item.current_price?.toFixed(2)}
                      </td>
                      <td className="py-2 px-3 text-right text-cyan-300 font-bold">
                        ₺{item.pred_avg_price?.toFixed(2)}
                      </td>
                      <td className="py-2 px-3 text-right text-white font-bold">
                        ₺{item.actual_avg_price?.toFixed(2)}
                      </td>
                      <td
                        className={`py-2 px-3 text-right font-bold ${
                          item.pred_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {formatPercent(item.pred_return_pct)}
                      </td>
                      <td
                        className={`py-2 px-3 text-right font-bold ${
                          item.actual_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {formatPercent(item.actual_return_pct)}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-300">
                        {item.err_pct?.toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-center">
                        {isPending ? (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
                            <Clock className="w-3 h-3" /> IN FLIGHT
                          </span>
                        ) : isHit ? (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                            <CheckCircle2 className="w-3 h-3" /> HIT
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30">
                            <XCircle className="w-3 h-3" /> MISS
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
