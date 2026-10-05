import React, { useState, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  fetchWeekStartOpportunities,
  fetchWeekStartForecast,
} from '../api/client';
import { formatPercent } from '../utils/formatters';
import {
  Calendar,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  BarChart3,
  Table as TableIcon,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

interface WeekStartDashboardProps {
  symbol?: string;
  onSelectSymbol?: (symbol: string) => void;
  onNavigateTab?: (tab: string) => void;
}

export const WeekStartDashboard: React.FC<WeekStartDashboardProps> = ({
  symbol = 'THYAO',
  onSelectSymbol,
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbol);
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // Sync with global symbol changes from App top navbar
  useEffect(() => {
    if (symbol && symbol !== selectedSymbol) {
      setSelectedSymbol(symbol);
    }
  }, [symbol]);

  // 1. Fetch Universe Opportunities (Top bullish & bearish setups)
  const {
    data: opportunitiesData,
    isLoading: isLoadingOpp,
    isFetching: isFetchingOpp,
    refetch: refetchOpp,
  } = useQuery({
    queryKey: ['week-start-opportunities'],
    queryFn: () => fetchWeekStartOpportunities(),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  // 2. Fetch Single Symbol Week Start Forecast & Backtest Ledger
  const {
    data: forecastData,
    refetch: refetchFc,
  } = useQuery({
    queryKey: ['week-start-forecast', selectedSymbol],
    queryFn: () => fetchWeekStartForecast(selectedSymbol),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  const handleSelectSymbol = (sym: string) => {
    setSelectedSymbol(sym);
    if (onSelectSymbol) {
      onSelectSymbol(sym);
    }
  };

  const refreshAll = () => {
    refetchOpp();
    refetchFc();
  };

  // Chronological ledger for charting (oldest to newest)
  const chronologicalLedger = useMemo(() => {
    if (!forecastData?.backtest_ledger) return [];
    return [...forecastData.backtest_ledger].reverse();
  }, [forecastData?.backtest_ledger]);

  // Chart dimensions & scaling
  const chartWidth = 900;
  const chartHeight = 240;
  const paddingX = 55;
  const paddingY = 25;
  const innerW = chartWidth - paddingX * 2;
  const innerH = chartHeight - paddingY * 2;

  const { minVal, maxVal, zeroY, pointsActual, pointsPred } = useMemo(() => {
    if (chronologicalLedger.length === 0) {
      return { minVal: -5, maxVal: 5, zeroY: paddingY + innerH / 2, pointsActual: [], pointsPred: [] };
    }

    const allVals = chronologicalLedger.flatMap((d) => [
      d.actual_return_pct,
      d.ml_pred_return_pct,
    ]);
    const maxA = Math.max(2.5, ...allVals.map((v) => Math.abs(v)));
    const minV = -maxA * 1.15;
    const maxV = maxA * 1.15;

    const scaleY = (val: number) => {
      const norm = (val - minV) / (maxV - minV);
      return paddingY + innerH * (1 - norm);
    };

    const stepX = innerW / Math.max(1, chronologicalLedger.length - 1);

    const actualPts = chronologicalLedger.map((d, i) => ({
      x: paddingX + i * stepX,
      y: scaleY(d.actual_return_pct),
      val: d.actual_return_pct,
      date: d.trade_date,
      isHit: d.ml_is_hit,
      err: d.ml_err_pct,
    }));

    const predPts = chronologicalLedger.map((d, i) => ({
      x: paddingX + i * stepX,
      y: scaleY(d.ml_pred_return_pct),
      val: d.ml_pred_return_pct,
      date: d.trade_date,
    }));

    return {
      minVal: minV,
      maxVal: maxV,
      zeroY: scaleY(0),
      pointsActual: actualPts,
      pointsPred: predPts,
    };
  }, [chronologicalLedger, innerW, innerH, paddingX, paddingY]);

  const pathActual = useMemo(() => {
    if (pointsActual.length === 0) return '';
    return pointsActual.reduce(
      (acc, pt, i) => (i === 0 ? `M ${pt.x},${pt.y}` : `${acc} L ${pt.x},${pt.y}`),
      ''
    );
  }, [pointsActual]);

  const pathPred = useMemo(() => {
    if (pointsPred.length === 0) return '';
    return pointsPred.reduce(
      (acc, pt, i) => (i === 0 ? `M ${pt.x},${pt.y}` : `${acc} L ${pt.x},${pt.y}`),
      ''
    );
  }, [pointsPred]);

  const getTierBadge = (tier: string) => {
    switch (tier) {
      case 'HIGH_CONVICTION_LONG':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm shadow-emerald-500/20">
            ★ HIGH CONVICTION LONG
          </span>
        );
      case 'HIGH_CONVICTION_SHORT':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 shadow-sm shadow-rose-500/20">
            ★ HIGH CONVICTION SHORT
          </span>
        );
      case 'MODERATE_LONG':
      case 'MILD_LONG':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
            BULLISH LONG
          </span>
        );
      case 'MODERATE_SHORT':
      case 'MILD_SHORT':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-rose-500/10 text-rose-300 border border-rose-500/30">
            BEARISH SHORT
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
            CONSOLIDATION
          </span>
        );
    }
  };

  const getPlaybookBadge = (playbook: string) => {
    let color = 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30';
    if (playbook.includes('ABSORPTION')) color = 'bg-indigo-500/10 text-indigo-300 border-indigo-500/30';
    if (playbook.includes('LIQUIDATION')) color = 'bg-rose-500/10 text-rose-300 border-rose-500/30';
    if (playbook.includes('CONTINUATION')) color = 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30';

    return (
      <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${color}`}>
        {playbook.replace(/_/g, ' ')}
      </span>
    );
  };

  // Popular BIST 30 tickers for quick selection pills
  const quickTickers = ['THYAO', 'AKBNK', 'ASELS', 'EREGL', 'FROTO', 'GARAN', 'ISCTR', 'KCHOL', 'KRDMD', 'MGROS', 'PGSUS', 'SISE', 'TUPRS', 'YKBNK'];

  return (
    <div className="space-y-4">
      {/* ── Top Header & Breadth Ribbon ── */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              <Calendar className="w-5 h-5" />
            </span>
            <div>
              <h1 className="text-base font-bold text-white flex items-center space-x-2">
                <span>Week Start Predictive Hub</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                  MONDAY DAY CLOSE FORECASTER
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Weekend Carry Friction (3-Day Repo) &amp; Friday W5 Institutional Closing Footprints
              </p>
            </div>
          </div>

          {/* Session Timing & Refresh */}
          <div className="flex items-center flex-wrap gap-2 text-xs">
            <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 flex items-center space-x-2 font-mono">
              <span className="text-slate-400">Friday As-Of:</span>
              <span className="text-cyan-300 font-semibold">{opportunitiesData?.as_of_date || 'Latest Close'}</span>
              <span className="text-slate-600">→</span>
              <span className="text-slate-400">Target Monday:</span>
              <span className="text-emerald-400 font-bold">{opportunitiesData?.target_date || 'Upcoming Monday'}</span>
            </div>

            <button
              onClick={refreshAll}
              disabled={isLoadingOpp || isFetchingOpp}
              className="flex items-center space-x-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition-colors text-xs font-semibold"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isFetchingOpp ? 'animate-spin text-cyan-400' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Aggregate Market Breadth Indicators */}
        {opportunitiesData && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-slate-800/80">
            <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">BIST 30 Monday Bias</div>
              <div className="flex items-center space-x-1.5 mt-0.5 text-xs font-mono font-semibold">
                <span className="text-emerald-400">{opportunitiesData.bullish_count} Bullish</span>
                <span className="text-slate-600">/</span>
                <span className="text-rose-400">{opportunitiesData.bearish_count} Bearish</span>
                <span className="text-slate-600">/</span>
                <span className="text-slate-400">{opportunitiesData.neutral_count} Neut</span>
              </div>
            </div>

            <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Avg Expected Monday Move</div>
              <div className={`text-sm font-mono font-bold mt-0.5 ${opportunitiesData.avg_expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {formatPercent(opportunitiesData.avg_expected_return_pct)}
              </div>
            </div>

            <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Significant Setups (|Δ| ≥ 1%)</div>
              <div className="text-sm font-mono font-bold text-cyan-400 mt-0.5">
                {opportunitiesData.significant_moves_count} Equities ({opportunitiesData.high_conviction_count} ≥ 2%)
              </div>
            </div>

            <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Weekend Financing Hurdle</div>
              <div className="text-sm font-mono font-bold text-amber-400 mt-0.5 flex items-center space-x-1">
                <span>{opportunitiesData.avg_weekend_carry_bps.toFixed(1)} bps</span>
                <span className="text-[10px] text-slate-500 font-normal">/ 3-day repo</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Dual Spotlight Decks (Praised Bullish vs Short Pressure Setups) ── */}
      {opportunitiesData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Top Bullish Setups */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  <TrendingUp className="w-4 h-4" />
                </span>
                <h2 className="text-xs font-bold text-white tracking-wide uppercase">
                  Top Bullish Week Openings (Upside Momentum)
                </h2>
              </div>
              <span className="text-[11px] text-slate-400 font-mono">Click to Inspect</span>
            </div>

            <div className="space-y-2">
              {opportunitiesData.top_longs.slice(0, 3).map((item) => {
                const isSelected = selectedSymbol === item.symbol;
                return (
                  <div
                    key={item.symbol}
                    onClick={() => handleSelectSymbol(item.symbol)}
                    className={`p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between ${
                      isSelected
                        ? 'bg-emerald-500/15 border-emerald-500/50 shadow-md shadow-emerald-500/10 ring-1 ring-emerald-500/40'
                        : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-mono font-bold text-white">{item.symbol}</span>
                      <div>
                        <div className="text-[11px] text-slate-400 truncate max-w-[140px] sm:max-w-[200px]">
                          {item.company_name}
                        </div>
                        <div className="flex items-center space-x-1.5 mt-0.5">
                          {getPlaybookBadge(item.playbook || 'TACTICAL_WEEK_OPEN_LONG')}
                          <span className="text-[10px] font-mono text-slate-400">{item.ml_champion_type}</span>
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-sm font-mono font-bold text-emerald-400">
                        +{item.expected_return_pct.toFixed(2)}%
                      </div>
                      <div className="text-[11px] font-mono text-slate-400">
                        Target: <span className="text-white font-medium">{item.target_price.toFixed(2)} TL</span>
                      </div>
                      {item.champion_dir_hit_rate_pct !== undefined && (
                        <div className="text-[10px] font-mono text-cyan-400">
                          Win: {item.champion_dir_hit_rate_pct.toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Top Short & Fade Setups */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className="p-1.5 rounded-md bg-rose-500/10 text-rose-400 border border-rose-500/30">
                  <TrendingDown className="w-4 h-4" />
                </span>
                <h2 className="text-xs font-bold text-white tracking-wide uppercase">
                  Top Short &amp; Fade Setups (Downside Pressure)
                </h2>
              </div>
              <span className="text-[11px] text-slate-400 font-mono">Click to Inspect</span>
            </div>

            <div className="space-y-2">
              {opportunitiesData.top_shorts.slice(0, 3).map((item) => {
                const isSelected = selectedSymbol === item.symbol;
                return (
                  <div
                    key={item.symbol}
                    onClick={() => handleSelectSymbol(item.symbol)}
                    className={`p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between ${
                      isSelected
                        ? 'bg-rose-500/15 border-rose-500/50 shadow-md shadow-rose-500/10 ring-1 ring-rose-500/40'
                        : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-mono font-bold text-white">{item.symbol}</span>
                      <div>
                        <div className="text-[11px] text-slate-400 truncate max-w-[140px] sm:max-w-[200px]">
                          {item.company_name}
                        </div>
                        <div className="flex items-center space-x-1.5 mt-0.5">
                          {getPlaybookBadge(item.playbook || 'TACTICAL_WEEK_OPEN_SHORT')}
                          <span className="text-[10px] font-mono text-slate-400">{item.ml_champion_type}</span>
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-sm font-mono font-bold text-rose-400">
                        {item.expected_return_pct.toFixed(2)}%
                      </div>
                      <div className="text-[11px] font-mono text-slate-400">
                        Target: <span className="text-white font-medium">{item.target_price.toFixed(2)} TL</span>
                      </div>
                      {item.champion_dir_hit_rate_pct !== undefined && (
                        <div className="text-[10px] font-mono text-cyan-400">
                          Win: {item.champion_dir_hit_rate_pct.toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── Quick Symbol Selector Pills ── */}
      <div className="flex items-center space-x-1.5 overflow-x-auto pb-1">
        <span className="text-[11px] font-mono text-slate-400 whitespace-nowrap mr-1">Active Constituent:</span>
        {quickTickers.map((t) => (
          <button
            key={t}
            onClick={() => handleSelectSymbol(t)}
            className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-all whitespace-nowrap ${
              selectedSymbol === t
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm shadow-cyan-500/20'
                : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* ── Single Stock Deep Dive: Live Forecast & Visual Reality Chart ── */}
      {forecastData && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl space-y-4">
          {/* Header & Badges */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-3 border-b border-slate-800">
            <div>
              <div className="flex items-center space-x-2.5 flex-wrap">
                <span className="text-xl font-mono font-bold text-white">{forecastData.symbol}</span>
                <span className="text-xs text-slate-400 font-medium">({forecastData.company_name})</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                  {forecastData.sector}
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-mono font-bold">
                  {forecastData.ml_champion_type} ({forecastData.crowned_horizon.toUpperCase()})
                </span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1 font-mono">
                As-Of: <span className="text-slate-200 font-semibold">{forecastData.as_of_date}</span> → Target Monday: <span className="text-emerald-400 font-bold">{forecastData.target_date}</span>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {getTierBadge(forecastData.conviction)}
              {getPlaybookBadge(forecastData.playbook)}
            </div>
          </div>

          {/* 4 Telemetry Metrics Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* Monday Target Price & Return */}
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Monday Target Expectation</div>
              <div className="flex items-baseline space-x-2 mt-1">
                <span className="text-xl font-mono font-bold text-white">
                  {forecastData.target_price.toFixed(2)} TL
                </span>
                <span className={`text-sm font-mono font-bold ${forecastData.expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {formatPercent(forecastData.expected_return_pct)}
                </span>
              </div>
              <div className="text-[11px] text-slate-500 font-mono mt-1">
                Base Close: {forecastData.current_price.toFixed(2)} TL
              </div>
            </div>

            {/* Target Price Volatility Bounds */}
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Volatility Target Bounds</div>
              <div className="text-sm font-mono font-bold text-cyan-300 mt-1">
                [{forecastData.price_low.toFixed(2)} – {forecastData.price_high.toFixed(2)}] TL
              </div>
              <div className="text-[11px] text-slate-500 font-mono mt-1">
                Credible Range Span: {(forecastData.price_high - forecastData.price_low).toFixed(2)} TL
              </div>
            </div>

            {/* Microstructure Flow & Carry Friction */}
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">3-Day Financing Carry Hurdle</div>
              <div className="text-sm font-mono font-bold text-amber-400 mt-1 flex items-center space-x-1.5">
                <span>{forecastData.weekend_carry_cost_bps.toFixed(1)} bps</span>
                <span className="text-[10px] text-slate-500 font-normal">carry cost</span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-1">
                Fri W5 MLB Flow: <span className={forecastData.fri_w5_mlb_share_pct >= 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                  {forecastData.fri_w5_mlb_share_pct.toFixed(2)}%
                </span>
              </div>
            </div>

            {/* 20-Monday Tournament Accuracy */}
            <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">20-Monday Champion Track</div>
              <div className="flex items-baseline space-x-2 mt-1">
                <span className="text-base font-mono font-bold text-emerald-400">
                  {forecastData.champion_dir_hits !== undefined ? `${forecastData.champion_dir_hits}/20` : '–'}
                </span>
                <span className="text-sm font-mono font-bold text-white">
                  ({forecastData.champion_dir_hit_rate_pct !== undefined ? `${forecastData.champion_dir_hit_rate_pct.toFixed(1)}%` : '–'})
                </span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-1">
                Calibration MAE: <span className="text-cyan-300 font-semibold">{forecastData.champion_mae_pct !== undefined ? `${forecastData.champion_mae_pct.toFixed(2)}%` : '–'}</span>
              </div>
            </div>
          </div>

          {/* ── Walk-Forward Reality Chart vs Table Toggle Header ── */}
          <div className="pt-2">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold text-white tracking-wide uppercase">
                  Walk-Forward Reality Performance (Last 20 Mondays)
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  ±0.25% Deadband
                </span>
              </div>

              <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setViewMode('chart')}
                  className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-semibold transition-all ${
                    viewMode === 'chart'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <BarChart3 className="w-3.5 h-3.5" />
                  <span>Chart View</span>
                </button>
                <button
                  onClick={() => setViewMode('table')}
                  className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-semibold transition-all ${
                    viewMode === 'table'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <TableIcon className="w-3.5 h-3.5" />
                  <span>Ledger Table</span>
                </button>
              </div>
            </div>

            {/* ── Interactive SVG Return Reality Chart ── */}
            {viewMode === 'chart' && (
              <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 space-y-3">
                {/* Legend & Hover Status */}
                <div className="flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1.5">
                      <span className="w-3 h-0.5 bg-cyan-400 inline-block" />
                      <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" />
                      <span className="text-slate-300 font-semibold">Realized Monday Return %</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <span className="w-3 h-0.5 bg-amber-400 border-t border-dashed inline-block" />
                      <span className="w-2 h-2 rotate-45 bg-amber-400 inline-block" />
                      <span className="text-slate-300 font-semibold">ML Forecasted Return %</span>
                    </div>
                  </div>

                  {hoveredIdx !== null && chronologicalLedger[hoveredIdx] && (
                    <div className="flex items-center space-x-2 text-[11px] bg-slate-900 px-2.5 py-1 rounded border border-slate-700">
                      <span className="text-slate-400">{chronologicalLedger[hoveredIdx].trade_date}:</span>
                      <span className="text-cyan-300 font-bold">
                        Realized {formatPercent(chronologicalLedger[hoveredIdx].actual_return_pct)}
                      </span>
                      <span className="text-slate-500">|</span>
                      <span className="text-amber-300 font-bold">
                        Pred {formatPercent(chronologicalLedger[hoveredIdx].ml_pred_return_pct)}
                      </span>
                      <span className="text-slate-500">|</span>
                      {chronologicalLedger[hoveredIdx].ml_is_hit ? (
                        <span className="text-emerald-400 font-bold flex items-center space-x-0.5">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>HIT</span>
                        </span>
                      ) : (
                        <span className="text-rose-400 font-bold flex items-center space-x-0.5">
                          <XCircle className="w-3 h-3" />
                          <span>MISS</span>
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* SVG Visual Canvas */}
                <div className="relative">
                  <svg
                    viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                    className="w-full h-60 select-none overflow-visible"
                    onMouseLeave={() => setHoveredIdx(null)}
                  >
                    {/* Grid Lines */}
                    {[maxVal, maxVal / 2, 0, minVal / 2, minVal].map((val, i) => {
                      const y = paddingY + innerH * (1 - (val - minVal) / (maxVal - minVal));
                      return (
                        <g key={i}>
                          <line
                            x1={paddingX}
                            y1={y}
                            x2={chartWidth - paddingX}
                            y2={y}
                            stroke={val === 0 ? '#475569' : '#334155'}
                            strokeWidth={val === 0 ? 1 : 0.5}
                            strokeDasharray={val === 0 ? undefined : '3 3'}
                          />
                          <text
                            x={paddingX - 8}
                            y={y + 3}
                            textAnchor="end"
                            fill="#64748b"
                            fontSize="9"
                            fontFamily="monospace"
                          >
                            {val.toFixed(1)}%
                          </text>
                        </g>
                      );
                    })}

                    {/* Shaded Neutral Deadband (±0.25%) */}
                    <rect
                      x={paddingX}
                      y={zeroY - (innerH * (0.25 / (maxVal - minVal)))}
                      width={innerW}
                      height={Math.max(2, (innerH * (0.50 / (maxVal - minVal))))}
                      fill="#38bdf8"
                      fillOpacity="0.07"
                    />

                    {/* ML Predicted Return Line (Dashed Amber) */}
                    <path
                      d={pathPred}
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth="2"
                      strokeDasharray="4 4"
                      strokeOpacity="0.85"
                    />

                    {/* Realized Monday Return Line (Solid Cyan) */}
                    <path
                      d={pathActual}
                      fill="none"
                      stroke="#22d3ee"
                      strokeWidth="2.5"
                    />

                    {/* Hover Target Line */}
                    {hoveredIdx !== null && pointsActual[hoveredIdx] && (
                      <line
                        x1={pointsActual[hoveredIdx].x}
                        y1={paddingY}
                        x2={pointsActual[hoveredIdx].x}
                        y2={chartHeight - paddingY}
                        stroke="#94a3b8"
                        strokeWidth="1"
                        strokeDasharray="2 2"
                      />
                    )}

                    {/* Prediction Diamond Markers */}
                    {pointsPred.map((pt, i) => (
                      <rect
                        key={`pred-${i}`}
                        x={pt.x - 3}
                        y={pt.y - 3}
                        width="6"
                        height="6"
                        transform={`rotate(45 ${pt.x} ${pt.y})`}
                        fill="#fbbf24"
                        stroke="#0f172a"
                        strokeWidth="1"
                        className="pointer-events-none"
                      />
                    ))}

                    {/* Realized Circular Data Points with Hit/Miss Glow */}
                    {pointsActual.map((pt, i) => {
                      const isHovered = hoveredIdx === i;
                      return (
                        <g
                          key={`act-${i}`}
                          className="cursor-pointer"
                          onMouseEnter={() => setHoveredIdx(i)}
                        >
                          <circle
                            cx={pt.x}
                            cy={pt.y}
                            r={isHovered ? 6 : 4}
                            fill={pt.isHit ? '#10b981' : '#f43f5e'}
                            stroke="#ffffff"
                            strokeWidth={isHovered ? 2 : 1}
                          />
                          {/* Invisible larger hit target for smooth mouseover */}
                          <circle
                            cx={pt.x}
                            cy={pt.y}
                            r={14}
                            fill="transparent"
                          />
                        </g>
                      );
                    })}

                    {/* X-Axis Date Labels (Every 2-3 sessions) */}
                    {pointsActual.map((pt, i) => {
                      if (i % 3 !== 0 && i !== pointsActual.length - 1) return null;
                      const dShort = pt.date.slice(5); // MM-DD
                      return (
                        <text
                          key={`lbl-${i}`}
                          x={pt.x}
                          y={chartHeight - 6}
                          textAnchor="middle"
                          fill="#64748b"
                          fontSize="9"
                          fontFamily="monospace"
                        >
                          {dShort}
                        </text>
                      );
                    })}
                  </svg>
                </div>
              </div>
            )}

            {/* ── Alternate Table Ledger View ── */}
            {viewMode === 'table' && (
              <div className="overflow-x-auto rounded-lg border border-slate-800">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-950 text-slate-400 text-[11px] uppercase border-b border-slate-800">
                    <tr>
                      <th className="px-3 py-2">Monday Date</th>
                      <th className="px-3 py-2">Prior Friday</th>
                      <th className="px-3 py-2">Realized Price</th>
                      <th className="px-3 py-2">Realized Return</th>
                      <th className="px-3 py-2">ML Pred Return</th>
                      <th className="px-3 py-2">Error %</th>
                      <th className="px-3 py-2">BIST30 Ret</th>
                      <th className="px-3 py-2">Carry Cost</th>
                      <th className="px-3 py-2 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-900/60">
                    {forecastData.backtest_ledger && forecastData.backtest_ledger.length > 0 ? (
                      forecastData.backtest_ledger.map((bt) => (
                        <tr key={bt.trade_date} className="hover:bg-slate-800/40 transition-colors">
                          <td className="px-3 py-2 font-bold text-white">{bt.trade_date}</td>
                          <td className="px-3 py-2 text-slate-400">{bt.prior_date}</td>
                          <td className="px-3 py-2 text-slate-200">{bt.actual_price.toFixed(2)} TL</td>
                          <td className={`px-3 py-2 font-bold ${bt.actual_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {formatPercent(bt.actual_return_pct)}
                          </td>
                          <td className={`px-3 py-2 ${bt.ml_pred_return_pct >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                            {formatPercent(bt.ml_pred_return_pct)}
                          </td>
                          <td className="px-3 py-2 text-slate-300">{bt.ml_err_pct.toFixed(2)}%</td>
                          <td className={`px-3 py-2 ${bt.bist30_ret_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {formatPercent(bt.bist30_ret_pct)}
                          </td>
                          <td className="px-3 py-2 text-amber-400">{bt.weekend_carry_cost_bps.toFixed(0)} bps</td>
                          <td className="px-3 py-2 text-center">
                            {bt.ml_is_hit ? (
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                HIT
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">
                                MISS
                              </span>
                            )}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={9} className="px-3 py-4 text-center text-slate-500">
                          No backtest records available.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
