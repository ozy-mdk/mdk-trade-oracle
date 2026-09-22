import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchMarketSummary, fetchTertipHorizons } from '../api/client';
import { TradingViewCandleChart } from './TradingViewCandleChart';
import { formatTL, formatPercent, getBiasBadgeStyle } from '../utils/formatters';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Layers,
  ShieldCheck,
  CheckSquare,
  Square,
  Compass,
} from 'lucide-react';

interface CandleDashboardProps {
  symbol: string;
  brokerId: string;
  tradeDate?: string;
}

export const CandleDashboard: React.FC<CandleDashboardProps> = ({
  symbol,
  brokerId,
  tradeDate,
}) => {
  const [interval, setInterval] = useState<'1m' | '5m' | '1d'>('1d');
  const [showCostLine, setShowCostLine] = useState<boolean>(true);

  const { data: summary } = useQuery({
    queryKey: ['marketSummary', symbol, brokerId, tradeDate],
    queryFn: () => fetchMarketSummary(symbol, brokerId, tradeDate),
    refetchInterval: 30000,
  });

  const { data: tertipData } = useQuery({
    queryKey: ['tertipHorizons', symbol, brokerId, tradeDate],
    queryFn: () => fetchTertipHorizons(symbol, brokerId, tradeDate),
  });

  const badgeStyle = getBiasBadgeStyle(summary?.bias_badge || 'NEUTRAL');
  const isNetPositive = (summary?.broker_net_flow_tl || 0) >= 0;

  // Diagnostic badge styling
  const diagBadge = tertipData?.diagnostic?.diagnostic_badge || 'STABLE_CORE_HOLD';
  const diagColor =
    diagBadge === 'PROFIT_HARVEST'
      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
      : diagBadge === 'DEFENSE_SUPPORT'
      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
      : diagBadge === 'MOMENTUM_EXPANSION'
      ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
      : diagBadge === 'DE_GROSSING_UNWIND'
      ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
      : diagBadge === 'CAPITULATION_DUMP'
      ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
      : 'bg-slate-800 text-slate-300 border-slate-700';

  const costBasis = summary?.fifo_avg_cost || tertipData?.fifo_avg_cost || 0;
  const brokerLabel = brokerId === 'BIG5' ? 'Big Five' : brokerId === 'KAMU' ? 'Kamu' : brokerId;

  return (
    <div className="space-y-4">
      {/* 5 Core Trader KPI Badges */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {/* KPI 1: Last Price & Return */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Close Price & Return</span>
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-xl font-bold font-mono text-white">
              {summary ? `₺${summary.close_price.toFixed(2)}` : '—'}
            </span>
            {summary && (
              <span
                className={`text-xs font-semibold font-mono flex items-center ${
                  summary.daily_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {summary.daily_return_pct >= 0 ? (
                  <TrendingUp className="w-3 h-3 mr-0.5 inline" />
                ) : (
                  <TrendingDown className="w-3 h-3 mr-0.5 inline" />
                )}
                {formatPercent(summary.daily_return_pct)}
              </span>
            )}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Session: {summary?.trade_date || tradeDate || 'Latest'}
          </div>
        </div>

        {/* KPI 2: Total Market Turnover */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Total Turnover</span>
            <DollarSign className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-xl font-bold font-mono text-white">
            {summary ? formatTL(summary.total_turnover_tl) : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            {brokerLabel} Share: {summary ? `${summary.broker_share_pct.toFixed(1)}%` : '0%'}
          </div>
        </div>

        {/* KPI 3: Institutional Net Flow */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>{brokerLabel} Net Order Flow</span>
            <span
              className={`w-2 h-2 rounded-full ${
                isNetPositive ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400 animate-pulse'
              }`}
            />
          </div>
          <div
            className={`text-xl font-bold font-mono ${
              isNetPositive ? 'text-emerald-400 glow-green' : 'text-rose-400 glow-red'
            }`}
          >
            {summary ? formatTL(summary.broker_net_flow_tl) : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1 flex justify-between">
            <span>B: {summary ? formatTL(summary.broker_buy_turnover_tl) : '0'}</span>
            <span>S: {summary ? formatTL(summary.broker_sell_turnover_tl) : '0'}</span>
          </div>
        </div>

        {/* KPI 4: Daily Realized PnL */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>FIFO Realized PnL</span>
            <Layers className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div
            className={`text-xl font-bold font-mono ${
              (summary?.daily_realized_pnl_tl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {summary ? formatTL(summary.daily_realized_pnl_tl) : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Carry FIFO: {summary ? formatTL(summary.carry_fifo_realized_pnl_tl) : '—'}
          </div>
        </div>

        {/* KPI 5: Institutional Bias Badge */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Institutional Bias</span>
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="my-auto">
            <span
              className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold border uppercase tracking-wider ${badgeStyle.bg} ${badgeStyle.text} ${badgeStyle.border}`}
            >
              {badgeStyle.label}
            </span>
          </div>
          <div className="text-[11px] text-slate-500">
            Algorithmic Conviction Filter
          </div>
        </div>
      </div>

      {/* Executive EWMA Tertip Forensics Strip */}
      {tertipData && tertipData.diagnostic && (
        <div className="glass-panel px-4 py-2.5 rounded-xl border border-slate-800/80 bg-slate-900/50 flex flex-wrap items-center justify-between gap-3 shadow-lg">
          {/* Action Diagnosis Hero Tag */}
          <div className="flex items-center space-x-2.5 min-w-[300px]">
            <Compass className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <div className="flex items-center space-x-2">
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-bold border uppercase font-mono tracking-wider ${diagColor}`}
              >
                {tertipData.diagnostic.diagnostic_badge}
              </span>
              <span className="text-xs text-slate-200 font-semibold truncate">
                {tertipData.diagnostic.headline}
              </span>
            </div>
          </div>

          {/* Multi-Horizon EWMA Flow Pills */}
          <div className="flex items-center space-x-1.5 overflow-x-auto py-0.5">
            {tertipData.horizons
              .filter((h) => h.code !== '1D')
              .map((h) => {
                const isPos = h.cum_net_flow_tl >= 0;
                return (
                  <div
                    key={h.code}
                    className="flex items-center space-x-1 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono whitespace-nowrap"
                    title={`${h.label}: Net Flow ${formatTL(h.cum_net_flow_tl)} | EWMA Qty: ${h.ewma_inventory_qty.toLocaleString()} | Stance: ${h.stance}`}
                  >
                    <span className="text-slate-400 font-sans font-bold">{h.code}:</span>
                    <span
                      className={`font-semibold ${isPos ? 'text-emerald-400' : 'text-rose-400'}`}
                    >
                      {formatTL(h.cum_net_flow_tl)}
                    </span>
                  </div>
                );
              })}
          </div>

          {/* EWMA Ribbon Trend & Cost Line Toggle */}
          <div className="flex items-center space-x-3 text-xs">
            <span className="text-[11px] font-mono text-cyan-300 hidden xl:inline bg-cyan-950/40 px-2.5 py-1 rounded border border-cyan-800/40">
              {tertipData.ribbon_status}
            </span>

            {costBasis > 0 && (
              <button
                onClick={() => setShowCostLine(!showCostLine)}
                className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-amber-500/40 text-amber-300 text-[11px] font-mono transition-colors"
                title="Toggle BofA FIFO Average Cost line on price chart"
              >
                {showCostLine ? (
                  <CheckSquare className="w-3.5 h-3.5 text-amber-400" />
                ) : (
                  <Square className="w-3.5 h-3.5 text-slate-500" />
                )}
                <span>{brokerLabel} Cost: ₺{costBasis.toFixed(2)}</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Timeframe Bar & Chart Controls */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center space-x-1.5 bg-slate-900/90 p-1 rounded-lg border border-slate-800">
          {(['1m', '5m', '1d'] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setInterval(tf)}
              className={`px-3 py-1 text-xs font-semibold rounded transition-colors ${
                interval === tf
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              {tf.toUpperCase()}
            </button>
          ))}
        </div>

        <div className="text-xs text-slate-400 flex items-center space-x-4">
          <span className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" />
            <span>Net Buyer Bar</span>
          </span>
          <span className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" />
            <span>Net Seller Bar</span>
          </span>
        </div>
      </div>

      {/* Dual-Pane Candlestick & Institutional Flow Chart */}
      <TradingViewCandleChart
        symbol={symbol}
        interval={interval}
        brokerId={brokerId}
        fifoAvgCost={costBasis}
        showCostLine={showCostLine}
        tertipData={tertipData}
      />
    </div>
  );
};
