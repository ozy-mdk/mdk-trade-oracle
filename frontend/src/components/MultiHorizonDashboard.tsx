import React, { useState, useMemo, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchMultiHorizonForecast,
  fetchMultiHorizonBacktest,
  fetchTertipMlForecast,
} from '../api/client';
import {
  MultiHorizonBacktestItem,
  MultiHorizonItem,
} from '../types/api';
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
  Activity,
  LineChart,
  Table as TableIcon,
  Calendar,
  Eye,
  EyeOff,
} from 'lucide-react';

interface MultiHorizonDashboardProps {
  symbol?: string;
  onSelectSymbol?: (symbol: string) => void;
  onNavigateTab?: (tab: string) => void;
  tradeDate?: string;
  activeHorizon?: string;
  onHorizonChange?: (horizon: string) => void;
}

const CORE_HORIZONS = ['1d', '3d', '5d', '10d', '15d'] as const;
const ALL_HORIZONS = ['1d', '3d', '5d', '10d', '15d', '30d'] as const;

export const MultiHorizonDashboard: React.FC<MultiHorizonDashboardProps> = ({
  symbol = 'THYAO',
  tradeDate,
  activeHorizon: externalHorizon,
  onHorizonChange,
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbol);
  const [activeHorizon, setActiveHorizon] = useState<string>(externalHorizon || '1d');
  const [show30D, setShow30D] = useState<boolean>(false);
  const [evalSessions, setEvalSessions] = useState<number>(60);
  const [displayMode, setDisplayMode] = useState<'both' | 'chart' | 'ledger'>('both');
  const [chartView, setChartView] = useState<'price' | 'return'>('return');
  const [hoveredTradeDate, setHoveredTradeDate] = useState<string | null>(null);

  // Sync with global symbol changes from App top navbar
  useEffect(() => {
    if (symbol && symbol !== selectedSymbol) {
      setSelectedSymbol(symbol);
    }
  }, [symbol]);

  // Sync with external horizon changes (e.g. from Opportunity Actions launch)
  useEffect(() => {
    if (externalHorizon && externalHorizon !== activeHorizon) {
      setActiveHorizon(externalHorizon);
      if (externalHorizon === '30d') setShow30D(true);
    }
  }, [externalHorizon]);

  const handleHorizonSelect = (hKey: string) => {
    setActiveHorizon(hKey);
    if (hKey === '30d') setShow30D(true);
    if (onHorizonChange) onHorizonChange(hKey);
  };

  const queryClient = useQueryClient();
  const [isRefreshing, setIsRefreshing] = useState(false);

  // 1. Fetch Multi-Horizon Forecast Payload (3d, 5d, 10d, 15d, 30d)
  const {
    data: forecastData,
    isLoading: isLoadingMulti,
    isFetching: isFetchingMulti,
    refetch: refetchMulti,
  } = useQuery({
    queryKey: ['multi-horizon-forecast', selectedSymbol, evalSessions, tradeDate],
    queryFn: () => fetchMultiHorizonForecast(selectedSymbol, false, evalSessions, tradeDate),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  // 2. Fetch Next-Day (1D) ML Forecast Payload
  const {
    data: nextDayData,
    isLoading: isLoading1d,
    isFetching: isFetching1d,
    refetch: refetch1d,
  } = useQuery({
    queryKey: ['tertipMlForecast', selectedSymbol, 'auto', tradeDate],
    queryFn: () => fetchTertipMlForecast(selectedSymbol, false, 'auto', tradeDate),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await Promise.all([
        fetchMultiHorizonForecast(selectedSymbol, true, evalSessions, tradeDate),
        fetchTertipMlForecast(selectedSymbol, true, 'auto', tradeDate),
      ]);
      await queryClient.invalidateQueries({ queryKey: ['multi-horizon-forecast', selectedSymbol] });
      await queryClient.invalidateQueries({ queryKey: ['multi-horizon-backtest', selectedSymbol] });
      await queryClient.invalidateQueries({ queryKey: ['tertipMlForecast', selectedSymbol] });
      await Promise.all([refetchMulti(), refetch1d()]);
    } finally {
      setIsRefreshing(false);
    }
  };

  // 3. Fetch Backtest Ledger for active inspected horizon (when not 1d)
  const {
    data: backtestData,
    isLoading: isLoadingBt,
  } = useQuery({
    queryKey: ['multi-horizon-backtest', selectedSymbol, activeHorizon, evalSessions, tradeDate],
    queryFn: () => fetchMultiHorizonBacktest(selectedSymbol, activeHorizon, evalSessions, tradeDate),
    enabled: activeHorizon !== '1d',
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  // 4. Synthesize 1D model into uniform MultiHorizonItem structure
  const oneDayHorizonItem: MultiHorizonItem | null = useMemo(() => {
    if (!nextDayData) return null;
    const tourney = nextDayData.tournament_summary;
    const rawLedger = nextDayData.walk_forward_ledger || [];
    const slicedRaw = rawLedger.slice(-evalSessions);
    const completedRaw = slicedRaw.filter((r) => r.actual_price != null && r.actual_price > 0);
    const hits = completedRaw.filter((r) => r.champion_is_hit ?? r.ml_is_hit).length;
    const total = completedRaw.length;
    const hitRate = total > 0 ? (hits / total) * 100 : (tourney?.champion_dir_hit_rate_pct ?? 0);

    const errs = completedRaw.map((r) => r.champion_err_pct ?? r.ml_err_pct ?? 0);
    const mae = errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : (tourney?.champion_mae_pct ?? 0);

    const hitErrs = completedRaw.filter((r) => r.champion_is_hit ?? r.ml_is_hit).map((r) => r.champion_err_pct ?? r.ml_err_pct ?? 0);
    const hitMae = hitErrs.length > 0 ? hitErrs.reduce((a, b) => a + b, 0) / hitErrs.length : (tourney?.champion_hit_mae_pct ?? 0);

    const missErrs = completedRaw.filter((r) => !(r.champion_is_hit ?? r.ml_is_hit)).map((r) => r.champion_err_pct ?? r.ml_err_pct ?? 0);
    const missMae = missErrs.length > 0 ? missErrs.reduce((a, b) => a + b, 0) / missErrs.length : (tourney?.champion_miss_mae_pct ?? 0);

    const loss = (100.0 - hitRate) + 1.0 * hitMae + 2.5 * missMae;

    const cleanModelName = (() => {
      if (tourney?.champion?.toLowerCase() === 'prophet' || tourney?.champion_label?.toLowerCase().includes('prophet')) {
        return 'Prophet';
      }
      if (tourney?.ml_champion_type) return tourney.ml_champion_type;
      if (tourney?.grand_champion_key) return tourney.grand_champion_key;
      if (tourney?.champion_label) {
        const match = tourney.champion_label.match(/\(([^)]+)\)/);
        if (match) return match[1];
        const stripped = tourney.champion_label.replace(/^(?:Crowned|Grand|ML)?\s*Champion\s*(?:\()?/i, '').replace(/\)$/, '').trim();
        if (stripped) return stripped;
      }
      if (tourney?.champion) return tourney.champion;
      return 'BayesianRidge';
    })();

    const backtestLedger: MultiHorizonBacktestItem[] = [...slicedRaw].reverse().map((r) => ({
      trade_date: r.date,
      target_date_start: r.date,
      target_date_end: r.date,
      current_price: r.actual_price,
      actual_avg_price: r.actual_price,
      actual_return_pct: r.actual_return_pct,
      pred_avg_price: r.champion_pred_price ?? r.ml_pred_price,
      pred_return_pct: r.champion_pred_return_pct ?? r.ml_pred_return_pct ?? 0,
      pred_direction: r.champion_direction ?? r.ml_direction ?? (r.champion_pred_return_pct && r.champion_pred_return_pct >= 0 ? 'UP' : 'DOWN'),
      actual_direction: r.actual_return_pct >= 0.25 ? 'UP' : r.actual_return_pct <= -0.25 ? 'DOWN' : 'FLAT',
      err_pct: r.champion_err_pct ?? r.ml_err_pct ?? 0,
      is_hit: r.champion_is_hit ?? r.ml_is_hit ?? false,
      is_pending: false,
      champion_model: cleanModelName,
      actual_training_sessions: nextDayData.actual_training_sessions ?? nextDayData.train_lookback_sessions ?? 63,
      data_sufficiency_status: nextDayData.data_sufficiency_status || 'FULL',
    }));

    const expRet = nextDayData.expected_return_pct ?? 0;
    const conviction =
      expRet >= 2.0
        ? 'STRONG_BUY'
        : expRet >= 0.5
        ? 'BUY'
        : expRet >= 0.2
        ? 'WEAK_BUY'
        : expRet <= -2.0
        ? 'STRONG_SELL'
        : expRet <= -0.5
        ? 'SELL'
        : expRet <= -0.2
        ? 'WEAK_SELL'
        : 'NEUTRAL';

    return {
      meta: {
        horizon: '1d',
        horizon_days: 1,
        crowned_model: cleanModelName,
        crowned_horizon: tourney?.crowned_horizon || '3m',
        training_lookback_sessions: nextDayData.actual_training_sessions ?? nextDayData.train_lookback_sessions ?? 63,
        hit_rate_pct: hitRate,
        hits,
        total_evals: total,
        mae_pct: mae,
        hit_mae_pct: hitMae,
        miss_mae_pct: missMae,
        loss,
        data_sufficiency_pct: 100.0,
        target_lookback_months: (nextDayData.train_lookback_sessions ?? 63) >= 200 ? 12 : (nextDayData.train_lookback_sessions ?? 63) >= 100 ? 6 : 3,
        target_lookback_sessions: nextDayData.train_lookback_sessions ?? 63,
        min_required_train_sessions: 63,
      },
      live_forecast: {
        horizon: '1d',
        horizon_days: 1,
        as_of_date: nextDayData.as_of_date,
        current_price: nextDayData.latest_close_price,
        target_price: nextDayData.target_price,
        expected_return_pct: nextDayData.expected_return_pct,
        price_low: nextDayData.price_low,
        price_high: nextDayData.price_high,
        stance: nextDayData.stance,
        conviction,
        playbook: nextDayData.playbook_headline || 'Tactical Momentum',
        champion_model: cleanModelName,
        training_lookback_sessions: nextDayData.actual_training_sessions ?? nextDayData.train_lookback_sessions ?? 63,
        actual_training_sessions: nextDayData.actual_training_sessions,
        actual_training_months: nextDayData.actual_training_months,
        target_training_months: nextDayData.target_training_months,
        actual_window_desc: nextDayData.actual_window_desc,
        data_sufficiency_status: nextDayData.data_sufficiency_status || 'FULL',
      },
      backtest_ledger: backtestLedger,
    };
  }, [nextDayData, evalSessions]);

  // 5. Unified Horizons Mapping
  const horizons = useMemo<Record<string, MultiHorizonItem>>(() => {
    const combined: Record<string, MultiHorizonItem> = { ...(forecastData?.horizons || {}) };
    if (oneDayHorizonItem) {
      combined['1d'] = oneDayHorizonItem;
    }
    return combined;
  }, [forecastData, oneDayHorizonItem]);

  // Visible horizons for cards & term structure
  const visibleHorizons = useMemo(() => {
    if (show30D || activeHorizon === '30d') {
      return ALL_HORIZONS;
    }
    return CORE_HORIZONS;
  }, [show30D, activeHorizon]);

  const activeHorizonData = horizons[activeHorizon];
  const activeMeta = activeHorizonData?.meta;

  // Unified Backtest Ledger (DESC - newest first for table)
  const ledger = useMemo<MultiHorizonBacktestItem[]>(() => {
    if (activeHorizon === '1d') {
      return oneDayHorizonItem?.backtest_ledger || [];
    }
    if (backtestData && backtestData.length > 0) {
      return [...backtestData.slice(0, evalSessions)].sort((a, b) => b.trade_date.localeCompare(a.trade_date));
    }
    const full = activeHorizonData?.backtest_ledger || [];
    return [...full.slice(-evalSessions)].sort((a, b) => b.trade_date.localeCompare(a.trade_date));
  }, [activeHorizon, oneDayHorizonItem, backtestData, activeHorizonData, evalSessions]);

  // Chronological data for left-to-right chart rendering (oldest first)
  const chronologicalChartData = useMemo<MultiHorizonBacktestItem[]>(() => {
    return [...ledger].sort((a, b) => a.trade_date.localeCompare(b.trade_date));
  }, [ledger]);

  // Dynamic hit rate KPIs over the selected evaluation window
  const completedLedger = useMemo(() => ledger.filter((r) => !r.is_pending), [ledger]);
  const dynamicHits = useMemo(() => completedLedger.filter((r) => r.is_hit).length, [completedLedger]);
  const dynamicTotal = useMemo(() => completedLedger.length, [completedLedger]);
  const dynamicHitRatePct = useMemo(() => {
    if (dynamicTotal > 0) return (dynamicHits / dynamicTotal) * 100.0;
    return activeMeta?.hit_rate_pct ?? 0.0;
  }, [dynamicHits, dynamicTotal, activeMeta]);
  const dynamicMaePct = useMemo(() => {
    if (completedLedger.length === 0) return activeMeta?.mae_pct ?? 0.0;
    const errs = completedLedger.map((r) => r.err_pct || 0);
    return errs.reduce((a, b) => a + b, 0) / errs.length;
  }, [completedLedger, activeMeta]);

  // Trailing 10-session stats
  const last10 = useMemo(() => chronologicalChartData.slice(-10), [chronologicalChartData]);
  const l10Hits = useMemo(() => last10.filter((r) => r.is_hit).length, [last10]);
  const l10Total = useMemo(() => last10.length, [last10]);
  const l10RatePct = useMemo(() => (l10Total > 0 ? (l10Hits / l10Total) * 100 : 0), [l10Hits, l10Total]);

  // Big moves (|actual return| >= 2.0%)
  const bigMoves = useMemo(
    () => chronologicalChartData.filter((r) => Math.abs(r.actual_return_pct) >= 2.0),
    [chronologicalChartData]
  );
  const bigMoveHits = useMemo(() => bigMoves.filter((r) => r.is_hit).length, [bigMoves]);
  const bigMoveTotal = useMemo(() => bigMoves.length, [bigMoves]);
  const bigMoveHitRatePct = useMemo(
    () => (bigMoveTotal > 0 ? (bigMoveHits / bigMoveTotal) * 100 : 0),
    [bigMoveHits, bigMoveTotal]
  );

  // SVG Chart Geometry Calculations
  const chartWidth = 900;
  const chartHeight = 220;
  const paddingX = 48;
  const paddingY = 24;
  const innerW = chartWidth - paddingX * 2;
  const innerH = chartHeight - paddingY * 2;

  const prices = useMemo(
    () =>
      chronologicalChartData
        .flatMap((r) => [r.actual_avg_price, r.pred_avg_price])
        .filter((p) => p != null && p > 0),
    [chronologicalChartData]
  );
  const minPrice = useMemo(() => (prices.length ? Math.min(...prices) * 0.985 : 0), [prices]);
  const maxPrice = useMemo(() => (prices.length ? Math.max(...prices) * 1.015 : 100), [prices]);

  const returns = useMemo(
    () =>
      chronologicalChartData
        .flatMap((r) => [Math.abs(r.actual_return_pct || 0), Math.abs(r.pred_return_pct || 0)])
        .filter((ret) => !isNaN(ret)),
    [chronologicalChartData]
  );
  const maxAbsReturn = useMemo(
    () => (returns.length ? Math.max(2.0, Math.max(...returns) * 1.15) : 5.0),
    [returns]
  );

  const getX = (idx: number) => {
    if (chronologicalChartData.length <= 1) return paddingX + innerW / 2;
    return paddingX + (idx / (chronologicalChartData.length - 1)) * innerW;
  };

  const getYPrice = (val: number) => {
    if (maxPrice <= minPrice) return paddingY + innerH / 2;
    return paddingY + innerH - ((val - minPrice) / (maxPrice - minPrice)) * innerH;
  };

  const getYReturn = (val: number) => {
    const zeroY = paddingY + innerH / 2;
    return zeroY - (val / maxAbsReturn) * (innerH / 2);
  };

  const actualPricePath = useMemo(() => {
    return chronologicalChartData
      .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYPrice(r.actual_avg_price).toFixed(1)}`)
      .join(' ');
  }, [chronologicalChartData, minPrice, maxPrice]);

  const predPricePath = useMemo(() => {
    return chronologicalChartData
      .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYPrice(r.pred_avg_price).toFixed(1)}`)
      .join(' ');
  }, [chronologicalChartData, minPrice, maxPrice]);

  const actualReturnPath = useMemo(() => {
    return chronologicalChartData
      .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYReturn(r.actual_return_pct).toFixed(1)}`)
      .join(' ');
  }, [chronologicalChartData, maxAbsReturn]);

  const predReturnPath = useMemo(() => {
    return chronologicalChartData
      .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getYReturn(r.pred_return_pct).toFixed(1)}`)
      .join(' ');
  }, [chronologicalChartData, maxAbsReturn]);

  const hoveredItem = useMemo(() => {
    if (!hoveredTradeDate) return null;
    return chronologicalChartData.find((r) => r.trade_date === hoveredTradeDate) || null;
  }, [hoveredTradeDate, chronologicalChartData]);

  const dateInterval = useMemo(() => {
    return chronologicalChartData.length > 50 ? Math.ceil(chronologicalChartData.length / 8) : 5;
  }, [chronologicalChartData.length]);

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

  const isPageLoading = (isLoadingMulti && isLoading1d) || isRefreshing;

  return (
    <div className="space-y-6 pb-12">
      {/* ── Top Header Deck ──────────────────────────────────────────────────────── */}
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
                  Zero-Lookahead Institutional Target Predictions Across 1D (Next-Day), 3D, 5D, 10D, 15D, & 30D Market Windows
                </p>
              </div>
            </div>
          </div>

          {/* Price, As-of Date & Refresh Button */}
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-xs text-slate-400 font-mono">Current Session Close</div>
              <div className="text-lg font-bold text-white font-mono">
                ₺{(forecastData?.current_price ?? nextDayData?.latest_close_price)?.toFixed(2) || '---'}
              </div>
            </div>

            <div className="text-right border-l border-slate-800 pl-4 hidden sm:block">
              <div className="text-xs text-slate-400 font-mono">As of Date</div>
              <div className="text-xs font-semibold text-slate-200 font-mono">
                {forecastData?.as_of_date || nextDayData?.as_of_date || '---'}
              </div>
            </div>

            {/* Hit Rate Horizon Window (30, 60, 90, 180) with 60 as default */}
            <div className="text-right border-l border-slate-800 pl-4 hidden md:block">
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Hit Rate Window</div>
              <div className="flex items-center space-x-1 bg-slate-950/80 p-0.5 rounded-lg border border-slate-800 mt-0.5 font-mono">
                {([30, 60, 90, 180] as const).map((sessions) => (
                  <button
                    key={sessions}
                    onClick={() => setEvalSessions(sessions)}
                    className={`px-2 py-0.5 rounded text-[11px] font-bold transition-all ${
                      evalSessions === sessions
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                    title={`Evaluate hit rates over trailing ${sessions} sessions`}
                  >
                    {sessions}D{sessions === 60 ? ' (Def)' : ''}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleRefresh}
              disabled={isFetchingMulti || isFetching1d || isRefreshing}
              className="p-2.5 rounded-xl bg-slate-900 border border-slate-700/80 hover:border-cyan-500/50 text-slate-300 hover:text-white transition-all shadow-inner disabled:opacity-50"
              title="Refresh Forecast Data (Force Fresh)"
            >
              <RefreshCw className={`w-4 h-4 ${(isFetchingMulti || isFetching1d || isRefreshing) ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {/* ── Forward Prediction Horizon Cards (1D, 3D, 5D, 10D, 15D, ±30D) ────────── */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3 px-1">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-bold text-white tracking-wide">
              FORWARD MARKET PREDICTION HORIZONS
            </h2>
            <span className="text-[11px] text-slate-400 hidden sm:inline">
              (Terminal price change at target session T+N vs entry session T)
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {/* 30D Macro Horizon Visibility Toggle */}
            <button
              onClick={() => setShow30D(!show30D)}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-[11px] font-mono font-bold transition-all border ${
                show30D || activeHorizon === '30d'
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-sm'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
              title="Toggle 30-Day Forward Macro Horizon visibility"
            >
              {show30D || activeHorizon === '30d' ? (
                <Eye className="w-3.5 h-3.5 text-amber-400" />
              ) : (
                <EyeOff className="w-3.5 h-3.5 text-slate-400" />
              )}
              <span>{show30D || activeHorizon === '30d' ? '30D Macro: Visible' : '+ 30D (Macro)'}</span>
            </button>

            <span className="text-xs text-slate-400 font-mono hidden md:inline">
              Click any card to inspect audited reality track
            </span>
          </div>
        </div>

        {/* Dynamic Responsive Grid: 5 columns for 1D-15D, 6 columns when 30D is enabled */}
        <div
          className={`grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 ${
            visibleHorizons.length > 5 ? 'lg:grid-cols-6' : 'lg:grid-cols-5'
          } gap-3`}
        >
          {visibleHorizons.map((hKey) => {
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
                onClick={() => handleHorizonSelect(hKey)}
                className={`glass-panel p-3.5 rounded-xl border transition-all cursor-pointer relative overflow-hidden group ${
                  isSelected
                    ? 'border-cyan-500/80 bg-cyan-950/20 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/40'
                    : 'border-slate-800/80 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60'
                }`}
              >
                {/* Active Indicator Top Bar */}
                {isSelected && (
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-indigo-500" />
                )}

                {/* Horizon Label & Days Badge */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="font-mono font-black text-sm text-white tracking-wider">
                      {hKey.toUpperCase()}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      ({hKey === '1d' ? 'Next-Day' : `${meta?.horizon_days || (hKey === '3d' ? 3 : hKey === '5d' ? 5 : hKey === '10d' ? 10 : hKey === '15d' ? 15 : 30)}d`})
                    </span>
                  </div>
                  <span
                    className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border ${getStanceBadge(
                      fc?.stance || 'NEUTRAL'
                    )}`}
                  >
                    {fc?.stance || 'NEUTRAL'}
                  </span>
                </div>

                {/* Expected Return % & Direction Gauge */}
                <div className="my-1.5">
                  <div className="flex items-baseline space-x-1.5">
                    <span
                      className={`text-xl font-black font-mono tracking-tight ${
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
                      <TrendingUp className="w-3.5 h-3.5 text-emerald-400 inline" />
                    ) : isBearish ? (
                      <TrendingDown className="w-3.5 h-3.5 text-rose-400 inline" />
                    ) : null}
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                    Target: <span className="text-white font-bold">₺{fc?.target_price?.toFixed(2) || '---'}</span>
                  </div>
                </div>

                {/* Credible Price Range Envelope */}
                <div className="bg-slate-900/80 rounded-lg p-1.5 border border-slate-800/80 my-2 text-[10px] font-mono">
                  <div className="text-[8.5px] text-slate-400 uppercase tracking-wider mb-0.5">
                    Vol-Calibrated Range
                  </div>
                  <div className="text-slate-200 flex justify-between">
                    <span>₺{fc?.price_low?.toFixed(2) || '---'}</span>
                    <span className="text-slate-500">──</span>
                    <span>₺{fc?.price_high?.toFixed(2) || '---'}</span>
                  </div>
                </div>

                {/* Crowned Model & Dynamic Win Rate */}
                <div className="space-y-1 text-xs pt-1.5 border-t border-slate-800/60 font-mono">
                  <div className="flex items-center justify-between text-[10.5px]">
                    <span className="text-slate-400 flex items-center gap-1 truncate max-w-[80px]">
                      <Award className="w-3 h-3 text-amber-400 shrink-0" /> Model:
                    </span>
                    <span className="font-semibold text-amber-300 truncate max-w-[110px]" title={meta?.crowned_model}>
                      {meta?.crowned_model || 'Auto'}
                      {meta?.feature_mode && meta.feature_mode !== 'FIFO_CORE' && (
                        <span className="text-[8.5px] text-cyan-400 font-mono block">
                          {meta.feature_mode === 'YTD_SUBSTITUTION' ? 'YTD Sub' : 'FIFO+YTD'}
                        </span>
                      )}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[10.5px]">
                    <span className="text-slate-400 flex items-center gap-1">
                      <ShieldCheck className="w-3 h-3 text-emerald-400 shrink-0" /> Win Rate:
                    </span>
                    <span className="font-bold text-emerald-400">
                      {meta?.hits != null && meta?.total_evals != null ? (
                        <span className="text-[9px] text-slate-400 mr-1 font-normal">{meta.hits}/{meta.total_evals}</span>
                      ) : null}
                      {meta?.hit_rate_pct != null ? `${meta.hit_rate_pct.toFixed(1)}%` : '---'}
                    </span>
                  </div>

                  {/* Actual Window Used */}
                  <div className="pt-1 border-t border-slate-800/40 text-[9.5px]">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 truncate max-w-[110px]" title={fc?.actual_window_desc}>
                        {fc?.actual_window_desc || `${meta?.training_lookback_sessions || 63} sessions`}
                      </span>
                      {fc?.data_sufficiency_status === 'PARTIAL_ADAPTED' ? (
                        <span className="px-1 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 text-[7.5px] font-bold">
                          PARTIAL
                        </span>
                      ) : fc?.data_sufficiency_status === 'INSUFFICIENT' ? (
                        <span className="px-1 py-0.2 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40 text-[7.5px] font-bold">
                          &lt; 3M STOP
                        </span>
                      ) : (
                        <span className="px-1 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[7.5px] font-bold">
                          FULL
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Conviction Badge & Playbook */}
                <div className="mt-2 pt-1.5 border-t border-slate-800/40 flex items-center justify-between">
                  <span
                    className={`text-[8.5px] font-mono font-bold px-1.5 py-0.5 rounded border ${getConvictionBadge(
                      fc?.conviction || 'NEUTRAL'
                    )}`}
                  >
                    {fc?.conviction?.replace('_', ' ') || 'NEUTRAL'}
                  </span>
                  <span className="text-[8.5px] text-slate-400 font-mono truncate max-w-[90px]" title={fc?.playbook}>
                    {fc?.playbook?.replace(/_/g, ' ') || 'Range'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Expected Return Term Structure Visualizer ────────────────────────────── */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800/80 shadow-xl">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white tracking-wide">
              MULTI-HORIZON EXPECTED RETURN TERM STRUCTURE
            </h3>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Forward Term Structure Curve ({visibleHorizons.map((h) => h.toUpperCase()).join(' → ')})
          </div>
        </div>

        <div
          className={`grid grid-cols-2 sm:grid-cols-3 ${
            visibleHorizons.length > 5 ? 'lg:grid-cols-6' : 'lg:grid-cols-5'
          } gap-2.5 pt-2 pb-1`}
        >
          {visibleHorizons.map((hKey) => {
            const hData = horizons[hKey];
            const expRet = hData?.live_forecast?.expected_return_pct ?? 0.0;
            const isBullish = expRet >= 0.25;
            const isBearish = expRet <= -0.25;
            const isInspected = activeHorizon === hKey;
            const barHeightPct = Math.min(100, (Math.abs(expRet) / 15.0) * 100);

            return (
              <div
                key={hKey}
                onClick={() => handleHorizonSelect(hKey)}
                className={`flex flex-col items-center cursor-pointer p-2.5 rounded-xl border transition-all ${
                  isInspected
                    ? 'border-cyan-500/60 bg-cyan-950/20 shadow-sm shadow-cyan-500/10'
                    : 'border-slate-800/60 bg-slate-900/30 hover:border-slate-700'
                }`}
              >
                <span className="text-xs font-mono font-bold text-white mb-1.5">
                  {hKey.toUpperCase()}
                </span>

                {/* Return Bar */}
                <div className="w-10 h-24 bg-slate-950/80 rounded-lg flex items-center justify-center p-1 relative border border-slate-800/80">
                  <div className="absolute left-0 right-0 top-1/2 h-[1px] bg-slate-700/80 z-10" />
                  <div
                    className={`w-6 rounded-sm transition-all ${
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
                  className={`text-xs font-mono font-bold mt-1.5 ${
                    isBullish ? 'text-emerald-400' : isBearish ? 'text-rose-400' : 'text-slate-400'
                  }`}
                >
                  {formatPercent(expRet)}
                </span>
                <span className="text-[9.5px] text-slate-400 font-mono mt-0.5">
                  ₺{hData?.live_forecast?.target_price?.toFixed(2) || '---'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Audited Walk-Forward Reality Track: Chart & Ledger Deck ───────────────── */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800/80 shadow-xl space-y-4">
        {/* Section Header Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-sm font-bold text-white tracking-wide">
                  AUDITED WALK-FORWARD REALITY TRACK ({activeHorizon.toUpperCase()})
                </h3>
                <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 font-mono font-semibold">
                  {activeMeta?.crowned_model || 'Algorithm'} ({activeMeta?.crowned_horizon || '3M/6M'})
                  {activeMeta?.feature_mode && (
                    <span className="ml-1 text-cyan-400 font-bold">
                      • {activeMeta.feature_mode}
                    </span>
                  )}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Session-by-session zero-lookahead backtest tracking over trailing {evalSessions} evaluated sessions
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            {/* View Mode Toggle: Visual Chart vs Audit Table vs Both */}
            <div className="flex items-center bg-slate-950 rounded-xl border border-slate-800 p-0.5 font-mono text-xs">
              <button
                onClick={() => setDisplayMode('chart')}
                className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                  displayMode === 'chart'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Display visual performance divergence chart"
              >
                <LineChart className="w-3.5 h-3.5" />
                <span>Chart</span>
              </button>
              <button
                onClick={() => setDisplayMode('ledger')}
                className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                  displayMode === 'ledger'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Display tabular audit ledger"
              >
                <TableIcon className="w-3.5 h-3.5" />
                <span>Table</span>
              </button>
              <button
                onClick={() => setDisplayMode('both')}
                className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                  displayMode === 'both'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Display both visual chart and audit ledger table"
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Both</span>
              </button>
            </div>

            {/* Hit Rate Window Selector Pills (30, 60, 90, 180) */}
            <div className="flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 font-mono text-xs">
              {([30, 60, 90, 180] as const).map((sessions) => (
                <button
                  key={sessions}
                  onClick={() => setEvalSessions(sessions)}
                  className={`px-2 py-0.5 rounded-lg text-xs font-bold transition-all ${
                    evalSessions === sessions
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {sessions}D
                </button>
              ))}
            </div>

            {/* Horizon Selection Tabs (1D, 3D, 5D, 10D, 15D, 30D) */}
            <div className="flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
              {ALL_HORIZONS.map((hKey) => (
                <button
                  key={hKey}
                  onClick={() => handleHorizonSelect(hKey)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold transition-all ${
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
        </div>

        {/* 5 Scorecard KPI Metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400 font-mono uppercase">Directional Hit Rate ({evalSessions}D)</div>
            <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
              {dynamicHitRatePct != null ? `${dynamicHitRatePct.toFixed(1)}%` : '---'}
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              {dynamicHits} hits / {dynamicTotal} sessions
            </div>
          </div>

          <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400 font-mono uppercase">Mean Absolute Error ({evalSessions}D)</div>
            <div className="text-lg font-bold text-slate-200 font-mono mt-0.5">
              {dynamicMaePct != null ? `${dynamicMaePct.toFixed(2)}%` : '---'}
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
              <span
                className={`text-lg font-bold font-mono ${
                  (activeMeta?.sessions_skipped_insufficient ?? 0) > 0
                    ? 'text-rose-400'
                    : (activeMeta?.sessions_with_partial_history ?? 0) > 0
                    ? 'text-amber-400'
                    : 'text-emerald-400'
                }`}
              >
                {activeMeta?.data_sufficiency_pct != null ? `${activeMeta.data_sufficiency_pct.toFixed(0)}%` : '100%'}
              </span>
              <span
                className={`text-[8px] font-bold px-1 py-0.2 rounded border font-mono ${
                  (activeMeta?.sessions_skipped_insufficient ?? 0) > 0
                    ? 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                    : (activeMeta?.sessions_with_partial_history ?? 0) > 0
                    ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                    : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                }`}
              >
                {(activeMeta?.sessions_skipped_insufficient ?? 0) > 0
                  ? `${activeMeta?.sessions_skipped_insufficient} STOPPED`
                  : (activeMeta?.sessions_with_partial_history ?? 0) > 0
                  ? `${activeMeta?.sessions_with_partial_history} PARTIAL`
                  : 'FULL'}
              </span>
            </div>
            <div
              className="text-[10px] text-slate-400 font-mono truncate"
              title={`Target: ${activeMeta?.target_lookback_months || 6}M (${activeMeta?.target_lookback_sessions || 126}d) • Hard Stop: <3M (63d)`}
            >
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
              {activeMeta?.sessions_with_partial_history
                ? ` Additionally, ${activeMeta.sessions_with_partial_history} sessions used an adapted window between 3 months and the target lookback.`
                : ''}
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

        {/* ── Visual Walk-Forward Performance Divergence Chart ─────────────────────── */}
        {(displayMode === 'chart' || displayMode === 'both') && (
          <div className="glass-panel p-4 rounded-xl border border-slate-800 bg-slate-900/90 shadow-xl space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white tracking-wide">
                  Walk-Forward Performance: {activeMeta?.crowned_model || 'Champion'} vs Realized Market Reality
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  ({chronologicalChartData.length} Sessions [{evalSessions}D Horizon])
                </span>
              </div>

              {/* Chart Mode Toggle & Stats Pill */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center font-mono text-[9px] gap-1.5">
                  <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                    Hits:{' '}
                    <span className="text-emerald-400 font-bold">{dynamicHits}✓</span> /{' '}
                    <span className="text-rose-400 font-bold">{dynamicTotal - dynamicHits}✗</span> ({dynamicHitRatePct.toFixed(1)}%)
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                    Last 10: <span className="text-emerald-400 font-bold">{l10Hits}✓</span> /{' '}
                    <span className="text-rose-400 font-bold">{l10Total - l10Hits}✗</span> ({l10RatePct.toFixed(0)}%)
                  </span>
                  <span
                    className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-semibold"
                    title="Sessions where |actual return| >= 2.0%"
                  >
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
              {chronologicalChartData.length === 0 ? (
                <div className="h-56 flex items-center justify-center text-slate-400 font-mono text-xs">
                  {isLoadingBt || isPageLoading ? 'Loading backtest walk-forward data...' : 'No historical data available for this horizon.'}
                </div>
              ) : (
                <svg
                  viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                  className="w-full h-56 select-none"
                  onMouseLeave={() => setHoveredTradeDate(null)}
                >
                  <defs>
                    <linearGradient id="actualGradMulti" x1="0%" y1="0%" x2="0%" y2="100%">
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
                  {chronologicalChartData.map((r, i) => {
                    const x = getX(i);
                    const yActual = chartView === 'price' ? getYPrice(r.actual_avg_price) : getYReturn(r.actual_return_pct);
                    const yPred = chartView === 'price' ? getYPrice(r.pred_avg_price) : getYReturn(r.pred_return_pct);
                    const isHovered = hoveredTradeDate === r.trade_date;
                    const isHit = r.is_hit;
                    const predColor = '#f59e0b';
                    const actualColor = '#38bdf8';
                    const baseR = chronologicalChartData.length > 60 ? 2.2 : 3.5;
                    const dotR = isHovered ? baseR + 2.0 : baseR;

                    return (
                      <g
                        key={`${r.trade_date}-${i}`}
                        onMouseEnter={() => setHoveredTradeDate(r.trade_date)}
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

                        {/* Predicted Price Point: Green for Hit (✓), Red for Miss (✗), with Amber Border */}
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
                        {(i % dateInterval === 0 || i === chronologicalChartData.length - 1) && (
                          <text
                            x={x}
                            y={chartHeight - 6}
                            textAnchor="middle"
                            fill="#64748b"
                            fontSize="8"
                            fontFamily="monospace"
                          >
                            {r.trade_date.slice(5)}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}

              {/* Dynamic Hover Tooltip Card */}
              {hoveredItem && (
                <div className="absolute top-2 right-2 bg-slate-900/95 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs font-mono z-20 pointer-events-none min-w-[220px]">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-1 mb-1.5">
                    <span className="text-slate-300 font-bold flex items-center gap-1">
                      <Calendar className="w-3 h-3 text-cyan-400" />
                      {hoveredItem.trade_date}
                    </span>
                    <span
                      className={`px-1.5 py-0.2 rounded text-[8px] font-black border ${
                        hoveredItem.is_hit
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                      }`}
                    >
                      {hoveredItem.is_hit ? '✓ HIT' : '✗ MISS'}
                    </span>
                  </div>

                  <div className="space-y-1 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Target Range:</span>
                      <span className="text-slate-200">
                        {hoveredItem.target_date_start} → {hoveredItem.target_date_end}
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-slate-400">Realized Close:</span>
                      <span className="text-sky-300 font-bold">
                        ₺{hoveredItem.actual_avg_price?.toFixed(2)} ({formatPercent(hoveredItem.actual_return_pct)})
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-slate-400">Predicted Target:</span>
                      <span className="text-amber-300 font-bold">
                        ₺{hoveredItem.pred_avg_price?.toFixed(2)} ({formatPercent(hoveredItem.pred_return_pct)})
                      </span>
                    </div>

                    <div className="flex justify-between pt-1 border-t border-slate-800 text-[10px]">
                      <span className="text-slate-400">Absolute Error:</span>
                      <span className="text-slate-200">{hoveredItem.err_pct?.toFixed(2)}%</span>
                    </div>

                    <div className="flex justify-between text-[9.5px]">
                      <span className="text-slate-500">Model:</span>
                      <span className="text-slate-400 truncate max-w-[120px]">
                        {hoveredItem.champion_model || activeMeta?.crowned_model || 'Algorithm'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Chart Legend Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1 px-1 font-mono text-[10px] text-slate-400">
              <div className="flex items-center space-x-4">
                <span className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-sky-400 inline-block rounded" />
                  <span className="text-sky-300 font-semibold">Realized Reality ({chartView === 'price' ? '₺' : '%'})</span>
                </span>
                <span className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-amber-400 border-b border-dashed border-amber-400 inline-block" />
                  <span className="text-amber-300 font-semibold">{activeMeta?.crowned_model || 'Champion'} Predicted</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                  <span className="text-emerald-400">Hit (✓)</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
                  <span className="text-rose-400">Miss (✗)</span>
                </span>
              </div>
              <div className="text-slate-500 text-[9.5px]">
                Hover over data points to inspect individual session call and error spread
              </div>
            </div>
          </div>
        )}

        {/* ── Tabular Audited Reality Ledger Table ─────────────────────────────────── */}
        {(displayMode === 'ledger' || displayMode === 'both') && (
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-xs font-mono text-left">
              <thead className="bg-slate-900/90 text-slate-400 text-[11px] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Decision Date</th>
                  <th className="py-2.5 px-3">Target Horizon Dates</th>
                  <th className="py-2.5 px-3 text-center">Training Window</th>
                  <th className="py-2.5 px-3 text-right">Close Price</th>
                  <th className="py-2.5 px-3 text-right">Target Close</th>
                  <th className="py-2.5 px-3 text-right">Realized Close</th>
                  <th className="py-2.5 px-3 text-right">Pred Return</th>
                  <th className="py-2.5 px-3 text-right">Actual Return</th>
                  <th className="py-2.5 px-3 text-right">Error %</th>
                  <th className="py-2.5 px-3 text-center">Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                {isLoadingBt || isPageLoading ? (
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
                    const isHovered = hoveredTradeDate === item.trade_date;

                    return (
                      <tr
                        key={`${item.trade_date}-${idx}`}
                        onMouseEnter={() => setHoveredTradeDate(item.trade_date)}
                        onMouseLeave={() => setHoveredTradeDate(null)}
                        className={`transition-colors ${
                          isHovered ? 'bg-slate-800/60 ring-1 ring-cyan-500/30' : 'hover:bg-slate-900/40'
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
                              {item.actual_training_sessions ? `${item.actual_training_sessions}d` : '---'}
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
        )}
      </div>
    </div>
  );
};
