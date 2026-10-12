import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchStockYtdPnl, fetchYtdOverview } from '../api/client';
import { StockYtdPnlResponse, YtdConstituentOverviewItem } from '../types/api';
import { formatTL } from '../utils/formatters';
import {
  RefreshCw,
  Coins,
  ArrowUpRight,
  ArrowDownRight,
  Search,
  BarChart2,
  Layers,
} from 'lucide-react';

interface YtdPnlDashboardProps {
  symbol?: string;
  tradeDate?: string;
  onSelectSymbol?: (symbol: string) => void;
}

const POPULAR_STOCKS = [
  'ASELS',
  'GARAN',
  'AKBNK',
  'ISCTR',
  'YKBNK',
  'TUPRS',
  'THYAO',
  'BIMAS',
  'KCHOL',
  'EKGYO',
  'EREGL',
  'SAHOL',
];

export const YtdPnlDashboard: React.FC<YtdPnlDashboardProps> = ({
  symbol = 'THYAO',
  tradeDate,
  onSelectSymbol,
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbol);
  const [metricMode, setMetricMode] = useState<'pnl' | 'flow'>('pnl');
  const [searchFilter, setSearchFilter] = useState<string>('');
  const [sectorFilter, setSectorFilter] = useState<string>('ALL');
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // Sync with prop changes
  React.useEffect(() => {
    if (symbol && symbol !== selectedSymbol) {
      setSelectedSymbol(symbol);
    }
  }, [symbol]);

  // 1. Fetch Single-Stock YTD PnL & Progression Timeline
  const {
    data: stockData,
    isFetching: isFetchingStock,
    refetch: refetchStock,
  } = useQuery<StockYtdPnlResponse>({
    queryKey: ['stockYtdPnl', selectedSymbol, tradeDate],
    queryFn: () => fetchStockYtdPnl(selectedSymbol, tradeDate),
    staleTime: 60 * 1000,
  });

  // 2. Fetch BIST 30 Overview Leaderboard
  const {
    data: overviewData,
    isFetching: isFetchingOverview,
    refetch: refetchOverview,
  } = useQuery<YtdConstituentOverviewItem[]>({
    queryKey: ['ytdOverview', tradeDate],
    queryFn: () => fetchYtdOverview(tradeDate),
    staleTime: 60 * 1000,
  });

  const handleSelect = (sym: string) => {
    setSelectedSymbol(sym);
    if (onSelectSymbol) {
      onSelectSymbol(sym);
    }
  };

  const pillars = stockData?.pillars;
  const timeline = stockData?.timeline || [];

  // Filtered leaderboard
  const filteredOverview = useMemo(() => {
    if (!overviewData) return [];
    return overviewData.filter((item) => {
      const matchSearch =
        item.symbol.toLowerCase().includes(searchFilter.toLowerCase()) ||
        item.name.toLowerCase().includes(searchFilter.toLowerCase());
      const matchSector = sectorFilter === 'ALL' || item.sector === sectorFilter;
      return matchSearch && matchSector;
    });
  }, [overviewData, searchFilter, sectorFilter]);

  const uniqueSectors = useMemo(() => {
    if (!overviewData) return [];
    const set = new Set(overviewData.map((d) => d.sector).filter(Boolean));
    return Array.from(set).sort();
  }, [overviewData]);

  // SVG Chart Dimensions & Computations
  const chartHeight = 220;
  const chartWidth = 900;
  const padding = { top: 20, right: 30, bottom: 35, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  const chartSeries = useMemo(() => {
    if (timeline.length < 2) return null;

    const values = timeline.flatMap((d) =>
      metricMode === 'pnl'
        ? [d.mlb_ytd_pnl_tl, d.big5_ytd_pnl_tl, d.kamu_ytd_pnl_tl]
        : [d.mlb_net_flow_tl, d.big5_net_flow_tl, d.kamu_net_flow_tl]
    );

    const minVal = Math.min(0, ...values);
    const maxVal = Math.max(0, ...values);
    const span = maxVal - minVal || 1;

    const scaleX = (idx: number) => padding.left + (idx / (timeline.length - 1)) * innerWidth;
    const scaleY = (val: number) => padding.top + innerHeight - ((val - minVal) / span) * innerHeight;
    const zeroY = scaleY(0);

    const makePath = (accessor: (d: typeof timeline[0]) => number) => {
      return timeline
        .map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(1)} ${scaleY(accessor(d)).toFixed(1)}`)
        .join(' ');
    };

    return {
      minVal,
      maxVal,
      zeroY,
      scaleX,
      scaleY,
      pathMlb: makePath((d) => (metricMode === 'pnl' ? d.mlb_ytd_pnl_tl : d.mlb_net_flow_tl)),
      pathBig5: makePath((d) => (metricMode === 'pnl' ? d.big5_ytd_pnl_tl : d.big5_net_flow_tl)),
      pathKamu: makePath((d) => (metricMode === 'pnl' ? d.kamu_ytd_pnl_tl : d.kamu_net_flow_tl)),
    };
  }, [timeline, metricMode, innerWidth, innerHeight]);

  const activePoint = hoveredIdx !== null && timeline[hoveredIdx] ? timeline[hoveredIdx] : null;

  return (
    <div className="space-y-6">
      {/* ── Top Header Banner & Stock Switcher ────────────────────────── */}
      <div className="glass-panel rounded-2xl border border-slate-800 p-5 bg-gradient-to-r from-slate-900/90 via-[#0c1427]/80 to-slate-900/90 shadow-2xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3.5">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-500/20 via-cyan-500/20 to-indigo-600/30 border border-amber-500/30 flex items-center justify-center shadow-lg shadow-amber-500/10">
              <Coins className="w-6 h-6 text-amber-400" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h1 className="text-xl font-extrabold text-white tracking-wide">
                  Institutional Year-to-Date (YTD) P&L
                </h1>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-300 border border-amber-500/30">
                  CALENDAR YEAR {stockData?.year || '2026'}
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  {stockData?.sessions_count || 0} SESSIONS
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Point-in-time cumulative economic mark-to-market performance for Bank of America (MLB), Big 5 domestic banks, and state conduits from year start.
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => {
                refetchStock();
                refetchOverview();
              }}
              disabled={isFetchingStock || isFetchingOverview}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors shadow-sm disabled:opacity-50"
              title="Refresh YTD data"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isFetchingStock || isFetchingOverview ? 'animate-spin text-cyan-400' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Quick Ticker Switcher */}
        <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center space-x-2 overflow-x-auto pb-1">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap mr-1">
            Focus Equities:
          </span>
          {POPULAR_STOCKS.map((sym) => {
            const isSelected = selectedSymbol === sym;
            return (
              <button
                key={sym}
                onClick={() => handleSelect(sym)}
                className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold transition-all whitespace-nowrap ${
                  isSelected
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-sm shadow-amber-500/20'
                    : 'bg-slate-800/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800'
                }`}
              >
                {sym}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Active Stock Header & 3-Pillar Scorecards ──────────────────── */}
      {stockData && pillars && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-2xl font-black text-white font-mono">{stockData.symbol}</span>
              <span className="text-sm font-medium text-slate-300">{stockData.symbol_name}</span>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/60">
                {stockData.sector}
              </span>
              <span className="text-xs text-slate-400">
                Close: <strong className="text-white font-mono">₺{stockData.latest_close_price.toFixed(2)}</strong>
              </span>
            </div>
            <div className="text-xs text-slate-400 font-mono">
              As of: <span className="text-slate-200">{stockData.trade_date}</span>
            </div>
          </div>

          {/* 3 Pillar Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Pillar 1: Bank of America (MLB) */}
            <div className="glass-panel rounded-xl border border-cyan-500/30 bg-gradient-to-br from-cyan-950/20 via-slate-900/90 to-slate-900 p-4 shadow-xl">
              <div className="flex items-center justify-between pb-2.5 border-b border-cyan-500/20">
                <div className="flex items-center space-x-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-cyan-300">
                    Bank of America (MLB)
                  </span>
                </div>
                <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  FOREIGN ALGO
                </span>
              </div>

              <div className="mt-3.5 space-y-3">
                <div>
                  <div className="text-[11px] text-slate-400 font-medium">YTD Total Economic P&L</div>
                  <div className="flex items-baseline space-x-2 mt-0.5">
                    <span
                      className={`text-2xl font-black font-mono ${
                        pillars.mlb.ytd_total_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(pillars.mlb.ytd_total_pnl_tl)}
                    </span>
                    {pillars.mlb.ytd_total_pnl_tl >= 0 ? (
                      <ArrowUpRight className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <ArrowDownRight className="w-5 h-5 text-rose-400" />
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-xs font-mono">
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Realized PnL:</span>
                    <span className={pillars.mlb.ytd_realized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.mlb.ytd_realized_pnl_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Open Unrealized:</span>
                    <span className={pillars.mlb.unrealized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.mlb.unrealized_pnl_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">YTD Net Flow:</span>
                    <span className={pillars.mlb.ytd_net_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.mlb.ytd_net_flow_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Buy / Sell Turn:</span>
                    <span className="text-slate-300">
                      {formatTL(pillars.mlb.ytd_buy_tl)} / {formatTL(pillars.mlb.ytd_sell_tl)}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Pillar 2: Domestic Big 5 (BIG5) */}
            <div className="glass-panel rounded-xl border border-indigo-500/30 bg-gradient-to-br from-indigo-950/20 via-slate-900/90 to-slate-900 p-4 shadow-xl">
              <div className="flex items-center justify-between pb-2.5 border-b border-indigo-500/20">
                <div className="flex items-center space-x-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-indigo-400 shadow-sm shadow-indigo-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-indigo-300">
                    Domestic Leaders (BIG5)
                  </span>
                </div>
                <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                  YKR, IYM, AKM, GRM, ZRY
                </span>
              </div>

              <div className="mt-3.5 space-y-3">
                <div>
                  <div className="text-[11px] text-slate-400 font-medium">YTD Total Economic P&L</div>
                  <div className="flex items-baseline space-x-2 mt-0.5">
                    <span
                      className={`text-2xl font-black font-mono ${
                        pillars.big5.ytd_total_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(pillars.big5.ytd_total_pnl_tl)}
                    </span>
                    {pillars.big5.ytd_total_pnl_tl >= 0 ? (
                      <ArrowUpRight className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <ArrowDownRight className="w-5 h-5 text-rose-400" />
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-xs font-mono">
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Realized PnL:</span>
                    <span className={pillars.big5.ytd_realized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.big5.ytd_realized_pnl_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Open Unrealized:</span>
                    <span className={pillars.big5.unrealized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.big5.unrealized_pnl_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">YTD Net Flow:</span>
                    <span className={pillars.big5.ytd_net_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.big5.ytd_net_flow_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Buy / Sell Turn:</span>
                    <span className="text-slate-300">
                      {formatTL(pillars.big5.ytd_buy_tl)} / {formatTL(pillars.big5.ytd_sell_tl)}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Pillar 3: State Conduits (KAMU) */}
            <div className="glass-panel rounded-xl border border-emerald-500/30 bg-gradient-to-br from-emerald-950/20 via-slate-900/90 to-slate-900 p-4 shadow-xl">
              <div className="flex items-center justify-between pb-2.5 border-b border-emerald-500/20">
                <div className="flex items-center space-x-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-emerald-300">
                    State Conduits (KAMU)
                  </span>
                </div>
                <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  ZRY, VKY, HLY, TRA
                </span>
              </div>

              <div className="mt-3.5 space-y-3">
                <div>
                  <div className="text-[11px] text-slate-400 font-medium">YTD Total Economic P&L</div>
                  <div className="flex items-baseline space-x-2 mt-0.5">
                    <span
                      className={`text-2xl font-black font-mono ${
                        pillars.kamu.ytd_total_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(pillars.kamu.ytd_total_pnl_tl)}
                    </span>
                    {pillars.kamu.ytd_total_pnl_tl >= 0 ? (
                      <ArrowUpRight className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <ArrowDownRight className="w-5 h-5 text-rose-400" />
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-xs font-mono">
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Realized PnL:</span>
                    <span className={pillars.kamu.ytd_realized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.kamu.ytd_realized_pnl_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Open Unrealized:</span>
                    <span className={pillars.kamu.unrealized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.kamu.unrealized_pnl_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">YTD Net Flow:</span>
                    <span className={pillars.kamu.ytd_net_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(pillars.kamu.ytd_net_flow_tl)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block font-sans">Buy / Sell Turn:</span>
                    <span className="text-slate-300">
                      {formatTL(pillars.kamu.ytd_buy_tl)} / {formatTL(pillars.kamu.ytd_sell_tl)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ── YTD Progression Timeline Interactive Chart ────────────────── */}
          <div className="glass-panel rounded-xl border border-slate-800 p-5 bg-[#0a0f1d]/90 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white tracking-wide">
                  Cumulative Year-to-Date Trajectory ({stockData.year})
                </h3>
              </div>

              {/* Metric Mode Toggle */}
              <div className="flex items-center space-x-1 bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs font-semibold">
                <button
                  onClick={() => setMetricMode('pnl')}
                  className={`px-3 py-1 rounded-md transition-all ${
                    metricMode === 'pnl'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Cumulative P&L (₺)
                </button>
                <button
                  onClick={() => setMetricMode('flow')}
                  className={`px-3 py-1 rounded-md transition-all ${
                    metricMode === 'flow'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Cumulative Net Flow (₺)
                </button>
              </div>
            </div>

            {/* Legend & Hovered Point Inspection */}
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs mb-3 font-mono bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1.5">
                  <div className="w-3 h-0.5 bg-cyan-400 rounded" />
                  <span className="text-cyan-300 font-semibold font-sans">BofA (MLB)</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <div className="w-3 h-0.5 bg-indigo-400 rounded" />
                  <span className="text-indigo-300 font-semibold font-sans">BIG 5</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <div className="w-3 h-0.5 bg-emerald-400 rounded" />
                  <span className="text-emerald-300 font-semibold font-sans">KAMU</span>
                </div>
              </div>

              {activePoint ? (
                <div className="flex items-center space-x-3 text-slate-300">
                  <span>
                    Date: <strong className="text-white">{activePoint.trade_date}</strong>
                  </span>
                  <span>
                    Close: <strong className="text-white">₺{activePoint.close_price.toFixed(2)}</strong>
                  </span>
                  <span>
                    MLB:{' '}
                    <strong className={activePoint.mlb_ytd_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(metricMode === 'pnl' ? activePoint.mlb_ytd_pnl_tl : activePoint.mlb_net_flow_tl)}
                    </strong>
                  </span>
                  <span>
                    BIG5:{' '}
                    <strong className={activePoint.big5_ytd_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(metricMode === 'pnl' ? activePoint.big5_ytd_pnl_tl : activePoint.big5_net_flow_tl)}
                    </strong>
                  </span>
                  <span>
                    KAMU:{' '}
                    <strong className={activePoint.kamu_ytd_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {formatTL(metricMode === 'pnl' ? activePoint.kamu_ytd_pnl_tl : activePoint.kamu_net_flow_tl)}
                    </strong>
                  </span>
                </div>
              ) : (
                <span className="text-slate-500 font-sans italic text-[11px]">
                  Hover over the chart to inspect session values
                </span>
              )}
            </div>

            {/* SVG Visual Canvas */}
            {chartSeries && timeline.length >= 2 ? (
              <div className="relative overflow-hidden w-full">
                <svg
                  viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                  className="w-full h-auto cursor-crosshair select-none"
                  onMouseLeave={() => setHoveredIdx(null)}
                >
                  <defs>
                    <linearGradient id="mlbGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>

                  {/* Horizontal Grid & Zero Reference Axis */}
                  <line
                    x1={padding.left}
                    y1={chartSeries.zeroY}
                    x2={chartWidth - padding.right}
                    y2={chartSeries.zeroY}
                    stroke="#475569"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                  />
                  <text
                    x={padding.left - 8}
                    y={chartSeries.zeroY + 3}
                    textAnchor="end"
                    fill="#94a3b8"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    ₺0
                  </text>

                  {/* Top Max & Bottom Min Axis Labels */}
                  <text
                    x={padding.left - 8}
                    y={padding.top + 8}
                    textAnchor="end"
                    fill="#64748b"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    {formatTL(chartSeries.maxVal)}
                  </text>
                  <text
                    x={padding.left - 8}
                    y={chartHeight - padding.bottom}
                    textAnchor="end"
                    fill="#64748b"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    {formatTL(chartSeries.minVal)}
                  </text>

                  {/* Trajectory Paths */}
                  <path d={chartSeries.pathMlb} fill="none" stroke="#22d3ee" strokeWidth="2.5" />
                  <path d={chartSeries.pathBig5} fill="none" stroke="#818cf8" strokeWidth="2" />
                  <path d={chartSeries.pathKamu} fill="none" stroke="#34d399" strokeWidth="2" />

                  {/* Interactive Cursor Hover Overlay */}
                  {timeline.map((d, i) => {
                    const cx = chartSeries.scaleX(i);
                    return (
                      <rect
                        key={d.trade_date}
                        x={cx - (innerWidth / timeline.length) / 2}
                        y={padding.top}
                        width={innerWidth / timeline.length}
                        height={innerHeight}
                        fill="transparent"
                        onMouseEnter={() => setHoveredIdx(i)}
                      />
                    );
                  })}

                  {/* Hover Indicator Crosshair */}
                  {hoveredIdx !== null && (
                    <g>
                      <line
                        x1={chartSeries.scaleX(hoveredIdx)}
                        y1={padding.top}
                        x2={chartSeries.scaleX(hoveredIdx)}
                        y2={chartHeight - padding.bottom}
                        stroke="#f8fafc"
                        strokeWidth="1.2"
                        strokeDasharray="3 3"
                      />
                      <circle
                        cx={chartSeries.scaleX(hoveredIdx)}
                        y={chartSeries.scaleY(
                          metricMode === 'pnl'
                            ? timeline[hoveredIdx].mlb_ytd_pnl_tl
                            : timeline[hoveredIdx].mlb_net_flow_tl
                        )}
                        r="4"
                        fill="#22d3ee"
                        stroke="#0f172a"
                        strokeWidth="1.5"
                      />
                    </g>
                  )}
                </svg>
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500 text-xs">
                Insufficient timeline points to render progression chart.
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── BIST 30 Leaderboard: Year-to-Date Institutional Overview ── */}
      <div className="glass-panel rounded-xl border border-slate-800 p-5 bg-[#080d19]/90 shadow-2xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2.5">
            <Layers className="w-5 h-5 text-amber-400" />
            <div>
              <h2 className="text-sm font-bold text-white tracking-wide">
                BIST 30 Constituent Year-to-Date Leaderboard
              </h2>
              <p className="text-[11px] text-slate-400">
                Click any stock to inspect its full institutional YTD timeline above.
              </p>
            </div>
          </div>

          {/* Search & Sector Filters */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 transform -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search symbol / company..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg pl-8 pr-3 py-1 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-500/60"
              />
            </div>

            <select
              value={sectorFilter}
              onChange={(e) => setSectorFilter(e.target.value)}
              aria-label="Filter by Sector"
              className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-amber-500/60"
            >
              <option value="ALL">All Sectors</option>
              {uniqueSectors.map((sec) => (
                <option key={sec} value={sec}>
                  {sec}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Leaderboard Table */}
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/90 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3">Sector</th>
                <th className="py-2.5 px-3 text-right">Close (₺)</th>
                <th className="py-2.5 px-3 text-right text-cyan-300">BofA YTD PnL</th>
                <th className="py-2.5 px-3 text-right text-cyan-300/80">BofA Net Flow</th>
                <th className="py-2.5 px-3 text-right text-indigo-300">BIG5 YTD PnL</th>
                <th className="py-2.5 px-3 text-right text-indigo-300/80">BIG5 Net Flow</th>
                <th className="py-2.5 px-3 text-right text-emerald-300">KAMU YTD PnL</th>
                <th className="py-2.5 px-3 text-right text-emerald-300/80">KAMU Net Flow</th>
                <th className="py-2.5 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
              {filteredOverview.map((row) => {
                const isSelected = row.symbol === selectedSymbol;
                return (
                  <tr
                    key={row.symbol}
                    onClick={() => handleSelect(row.symbol)}
                    className={`hover:bg-slate-800/50 cursor-pointer transition-colors ${
                      isSelected ? 'bg-amber-500/10 border-l-2 border-amber-400' : ''
                    }`}
                  >
                    <td className="py-2 px-3">
                      <div className="flex items-center space-x-1.5">
                        <strong className="text-white text-xs">{row.symbol}</strong>
                        <span className="text-[10px] text-slate-400 font-sans truncate max-w-[120px]">
                          {row.name}
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-3 text-slate-400 font-sans text-[11px]">{row.sector}</td>
                    <td className="py-2 px-3 text-right text-slate-200">₺{row.close_price.toFixed(2)}</td>

                    {/* BofA */}
                    <td
                      className={`py-2 px-3 text-right font-bold ${
                        row.mlb_ytd_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(row.mlb_ytd_pnl_tl)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">{formatTL(row.mlb_net_flow_tl)}</td>

                    {/* BIG5 */}
                    <td
                      className={`py-2 px-3 text-right font-bold ${
                        row.big5_ytd_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(row.big5_ytd_pnl_tl)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">{formatTL(row.big5_net_flow_tl)}</td>

                    {/* KAMU */}
                    <td
                      className={`py-2 px-3 text-right font-bold ${
                        row.kamu_ytd_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(row.kamu_ytd_pnl_tl)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">{formatTL(row.kamu_net_flow_tl)}</td>

                    <td className="py-2 px-3 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSelect(row.symbol);
                        }}
                        className="px-2 py-0.5 rounded text-[10px] font-sans font-semibold bg-slate-800 hover:bg-cyan-500/20 hover:text-cyan-300 text-slate-300 border border-slate-700 transition-colors"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
