import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  fetchTertipPortfolio,
  fetchTertipLots,
  fetchTertipHistory,
  fetchTertipHorizons,
  fetchInstruments,
} from '../api/client';
import { formatTL, formatPercent, formatVolume } from '../utils/formatters';
import { TertipEwmaChart } from './TertipEwmaChart';
import {
  Briefcase,
  TrendingUp,
  TrendingDown,
  Search,
  Layers,
  Shield,
  Compass,
  Activity,
  ChevronDown,
} from 'lucide-react';

interface TertipDashboardProps {
  brokerId: string;
  symbol?: string;
  tradeDate?: string;
  onSelectSymbol?: (symbol: string) => void;
}

export const TertipDashboard: React.FC<TertipDashboardProps> = ({
  brokerId,
  symbol = 'THYAO',
  tradeDate,
  onSelectSymbol,
}) => {
  const [subView, setSubView] = useState<'intelligence' | 'positions' | 'lots' | 'history'>('intelligence');
  const [activeSymbol, setActiveSymbol] = useState<string>(symbol);
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Keep activeSymbol synced if prop changes
  useEffect(() => {
    if (symbol) setActiveSymbol(symbol);
  }, [symbol]);

  // Fetch instruments list for symbol selector
  const { data: instruments } = useQuery({
    queryKey: ['metaInstruments'],
    queryFn: fetchInstruments,
  });

  // 1. Fetch Tertip Horizons & Action Forensics for activeSymbol
  const { data: tertipData } = useQuery({
    queryKey: ['tertipHorizons', activeSymbol, brokerId, tradeDate],
    queryFn: () => fetchTertipHorizons(activeSymbol, brokerId, tradeDate),
  });

  // 2. Fetch Portfolio Positions
  const { data: portfolio } = useQuery({
    queryKey: ['tertipPortfolio', brokerId, tradeDate],
    queryFn: () => fetchTertipPortfolio(brokerId, tradeDate),
  });

  // 3. Fetch Open Lots
  const { data: lots } = useQuery({
    queryKey: ['tertipLots', brokerId],
    queryFn: () => fetchTertipLots(brokerId),
    enabled: subView === 'lots',
  });

  // 4. Fetch History
  const { data: history } = useQuery({
    queryKey: ['tertipHistory', brokerId],
    queryFn: () => fetchTertipHistory(brokerId, undefined, 50),
    enabled: subView === 'history',
  });

  // Filter positions by search term
  const filteredPositions = (portfolio?.positions || []).filter(
    (p) =>
      p.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.symbol_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Filter lots by search term
  const filteredLots = (lots || []).filter((l) =>
    l.symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Diagnostic badge color
  const diagBadge = tertipData?.diagnostic?.diagnostic_badge || 'STABLE_CORE_HOLD';
  const diagBadgeStyle =
    diagBadge === 'PROFIT_HARVEST'
      ? { bg: 'bg-emerald-500/20', text: 'text-emerald-300', border: 'border-emerald-500/40' }
      : diagBadge === 'DEFENSE_SUPPORT'
      ? { bg: 'bg-cyan-500/20', text: 'text-cyan-300', border: 'border-cyan-500/40' }
      : diagBadge === 'MOMENTUM_EXPANSION'
      ? { bg: 'bg-indigo-500/20', text: 'text-indigo-300', border: 'border-indigo-500/40' }
      : diagBadge === 'DE_GROSSING_UNWIND'
      ? { bg: 'bg-amber-500/20', text: 'text-amber-300', border: 'border-amber-500/40' }
      : diagBadge === 'CAPITULATION_DUMP'
      ? { bg: 'bg-rose-500/20', text: 'text-rose-300', border: 'border-rose-500/40' }
      : { bg: 'bg-slate-800', text: 'text-slate-300', border: 'border-slate-700' };

  return (
    <div className="space-y-4">
      {/* Portfolio Executive Banner */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {/* Total MTM Valuation */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Portfolio MTM Valuation</span>
            <Briefcase className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">
            {portfolio ? formatTL(portfolio.total_market_value_tl) : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Institution: <span className="text-slate-300 font-semibold">{brokerId === 'BIG5' ? 'BIG FIVE (Bundle)' : brokerId === 'KAMU' ? 'KAMU (State Bundle)' : brokerId}</span> | Positions:{' '}
            {portfolio?.positions.length || 0}
          </div>
        </div>

        {/* Total Unrealized PnL */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Unrealized PnL</span>
            {(portfolio?.total_unrealized_pnl_tl || 0) >= 0 ? (
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-rose-400" />
            )}
          </div>
          <div
            className={`text-2xl font-bold font-mono ${
              (portfolio?.total_unrealized_pnl_tl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {portfolio ? formatTL(portfolio.total_unrealized_pnl_tl) : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Mark-To-Market against session close
          </div>
        </div>

        {/* Daily Realized PnL */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Daily Realized PnL</span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          <div
            className={`text-2xl font-bold font-mono ${
              (portfolio?.total_daily_realized_pnl_tl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {portfolio ? formatTL(portfolio.total_daily_realized_pnl_tl) : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Intraday matched + Carry FIFO realized
          </div>
        </div>

        {/* Cumulative Realized PnL */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Cumulative Realized PnL</span>
            <Shield className="w-4 h-4 text-indigo-400" />
          </div>
          <div
            className={`text-2xl font-bold font-mono ${
              (portfolio?.total_cumulative_realized_pnl_tl || 0) >= 0
                ? 'text-emerald-400'
                : 'text-rose-400'
            }`}
          >
            {portfolio ? formatTL(portfolio.total_cumulative_realized_pnl_tl) : '—'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            All-time audited FIFO track record
          </div>
        </div>
      </div>

      {/* Sub-View Navigation Bar & Stock Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/80 p-2 rounded-xl border border-slate-800">
        <div className="flex items-center space-x-1.5">
          <button
            onClick={() => setSubView('intelligence')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors flex items-center space-x-1.5 ${
              subView === 'intelligence'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass className="w-3.5 h-3.5 text-cyan-400" />
            <span>Tertip Intelligence & Action Forensics</span>
          </button>
          <button
            onClick={() => setSubView('positions')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
              subView === 'positions'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Active Stock Inventory ({portfolio?.positions.length || 0})
          </button>
          <button
            onClick={() => setSubView('lots')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
              subView === 'lots'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Audited Open Lots
          </button>
          <button
            onClick={() => setSubView('history')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
              subView === 'history'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            PnL Progression
          </button>
        </div>

        {/* Stock Selector for Intelligence View */}
        {subView === 'intelligence' ? (
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Target Symbol:</span>
            <div className="relative">
              <select
                value={activeSymbol}
                onChange={(e) => {
                  setActiveSymbol(e.target.value);
                  if (onSelectSymbol) onSelectSymbol(e.target.value);
                }}
                className="bg-slate-900 border border-slate-700/80 hover:border-cyan-500/50 rounded-lg px-3 py-1.5 text-xs font-mono font-bold text-white focus:outline-none appearance-none pr-8 cursor-pointer transition-colors"
              >
                {(instruments || []).map((inst) => (
                  <option key={inst.symbol} value={inst.symbol}>
                    {inst.symbol} — {inst.name.length > 20 ? inst.name.substring(0, 20) + '...' : inst.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 transform -translate-y-1/2 pointer-events-none" />
            </div>
          </div>
        ) : (
          subView !== 'history' && (
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 transform -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Filter by symbol..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-slate-900 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
              />
            </div>
          )
        )}
      </div>

      {/* ── Sub-View: Intelligence & Action Forensics (Primary) ─────────────── */}
      {subView === 'intelligence' && (
        <div className="space-y-4">
          {/* Loading Indicator */}
          {!tertipData && (
            <div className="glass-panel p-8 rounded-xl border border-slate-800 flex items-center justify-center space-x-3 text-cyan-400 font-mono text-xs">
              <Activity className="w-4 h-4 animate-spin text-cyan-400" />
              <span>Analyzing Tertip Intelligence & Multi-Horizon EWMA Ribbons for {activeSymbol}...</span>
            </div>
          )}

          {/* Zone 1: Action Diagnosis Hero Card */}
          {tertipData && tertipData.diagnostic && (
            <div className="glass-panel p-4 rounded-xl border border-slate-800 shadow-xl bg-gradient-to-r from-slate-900/90 via-slate-900/60 to-slate-950/90">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3 mb-3">
                <div className="flex items-center space-x-3">
                  <span
                    className={`px-3 py-1 rounded-md text-xs font-bold border uppercase font-mono tracking-wider shadow-sm ${diagBadgeStyle.bg} ${diagBadgeStyle.text} ${diagBadgeStyle.border}`}
                  >
                    {tertipData.diagnostic.diagnostic_badge}
                  </span>
                  <span className="text-sm font-semibold text-white">
                    {tertipData.diagnostic.headline}
                  </span>
                </div>
                <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
                  <span>Conviction:</span>
                  <span className="font-bold text-cyan-400">
                    {tertipData.diagnostic.conviction_pct}%
                  </span>
                  <span className="text-slate-600">•</span>
                  <span>Session:</span>
                  <span className="text-white font-semibold">{tertipData.trade_date}</span>
                </div>
              </div>

              {/* Rationale Text */}
              <p className="text-xs text-slate-300 leading-relaxed font-sans mb-3">
                {tertipData.diagnostic.rationale}
              </p>

              {/* Core Execution & Posture Stats */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 border-t border-slate-800/60 text-xs font-mono">
                <div>
                  <div className="text-[11px] text-slate-500">Day Net Flow:</div>
                  <div
                    className={`text-sm font-bold ${
                      tertipData.day_net_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {formatTL(tertipData.day_net_flow_tl)}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    B: {formatTL(tertipData.day_buy_turnover_tl)} | S: {formatTL(tertipData.day_sell_turnover_tl)}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] text-slate-500">Open Inventory:</div>
                  <div className="text-sm font-bold text-white">
                    {formatVolume(tertipData.open_stock_quantity)} sh
                  </div>
                  <div className="text-[10px] text-slate-400">
                    MTM: {formatTL(tertipData.market_value_tl)}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] text-slate-500">FIFO Unit Cost:</div>
                  <div className="text-sm font-bold text-white">
                    ₺{tertipData.fifo_avg_cost.toFixed(2)}
                  </div>
                  <div
                    className={`text-[10px] ${
                      tertipData.unrealized_pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    PnL: {formatPercent(tertipData.unrealized_pnl_pct)} ({formatTL(tertipData.unrealized_pnl_tl)})
                  </div>
                </div>

                <div>
                  <div className="text-[11px] text-slate-500">EWMA Posture & Saturation:</div>
                  <div className="text-sm font-bold text-cyan-300 truncate">
                    {tertipData.ribbon_status}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    12M Saturation: <span className="text-amber-300">{tertipData.global_saturation_pct}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Zone 2: Multi-Horizon EWMA Stance Matrix */}
          {tertipData && (
            <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-xl">
              <div className="px-4 py-2.5 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between text-xs">
                <span className="font-bold text-white uppercase tracking-wider flex items-center space-x-2">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Multi-Horizon EWMA Positioning Matrix ({activeSymbol})</span>
                </span>
                <span className="text-[11px] text-slate-500">
                  Older sessions penalized via exponential decay weights
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-900/60 text-slate-400 uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">Horizon</th>
                      <th className="py-2.5 px-3 text-right">Cum. Net Flow</th>
                      <th className="py-2.5 px-3 text-right">Net Shares</th>
                      <th className="py-2.5 px-3 text-right">EWMA Inventory</th>
                      <th className="py-2.5 px-3 text-right">EWMA Cost</th>
                      <th className="py-2.5 px-3 text-right">Spread vs Close</th>
                      <th className="py-2.5 px-3 text-right">Saturation</th>
                      <th className="py-2.5 px-3 text-center">Horizon Stance</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {tertipData.horizons.map((h) => {
                      const isFlowPos = h.cum_net_flow_tl >= 0;
                      const isSpreadPos = h.cost_spread_pct >= 0;
                      return (
                        <tr key={h.code} className="hover:bg-slate-800/40 transition-colors">
                          <td className="py-2 px-3 font-semibold text-white font-sans">
                            <div className="flex items-center space-x-1.5">
                              <span className="font-mono text-cyan-400">{h.code}</span>
                              <span className="text-slate-400 text-[11px] hidden sm:inline">
                                ({h.label})
                              </span>
                            </div>
                          </td>
                          <td
                            className={`py-2 px-3 text-right font-semibold ${
                              isFlowPos ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {formatTL(h.cum_net_flow_tl)}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-300">
                            {formatVolume(h.cum_net_shares)}
                          </td>
                          <td className="py-2 px-3 text-right text-white font-semibold">
                            {formatVolume(h.ewma_inventory_qty)}
                          </td>
                          <td className="py-2 px-3 text-right text-amber-300">
                            ₺{h.ewma_unit_cost.toFixed(2)}
                          </td>
                          <td
                            className={`py-2 px-3 text-right ${
                              isSpreadPos ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {formatPercent(h.cost_spread_pct)}
                          </td>
                          <td className="py-2 px-3 text-right">
                            <span className="text-slate-300">{h.saturation_pct}%</span>
                          </td>
                          <td className="py-2 px-3 text-center font-sans">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                h.stance === 'ACCUMULATION' || h.stance === 'EXPANDING' || h.stance === 'BUYING'
                                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                                  : h.stance === 'DISTRIBUTION' || h.stance === 'UNWINDING' || h.stance === 'SELLING'
                                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                                  : 'bg-slate-800 text-slate-300 border border-slate-700'
                              }`}
                            >
                              {h.stance}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Zone 3: Interactive Dual-Scale TradingView Chart */}
          <TertipEwmaChart symbol={activeSymbol} brokerId={brokerId} tertipData={tertipData} />
        </div>
      )}

      {/* ── Sub-View: Active Equity Positions Table ─────────────────────────── */}
      {subView === 'positions' && (
        <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-xl">
          <div className="overflow-x-auto max-h-[550px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider sticky top-0 border-b border-slate-800 z-10">
                <tr>
                  <th className="py-2.5 px-3">Symbol</th>
                  <th className="py-2.5 px-3">Side</th>
                  <th className="py-2.5 px-3 text-right">Open Quantity</th>
                  <th className="py-2.5 px-3 text-right">FIFO Avg Cost</th>
                  <th className="py-2.5 px-3 text-right">Market Price</th>
                  <th className="py-2.5 px-3 text-right">MTM Value</th>
                  <th className="py-2.5 px-3 text-right">Unrealized PnL</th>
                  <th className="py-2.5 px-3 text-right">Daily Realized</th>
                  <th className="py-2.5 px-3 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {filteredPositions.map((pos) => {
                  const isLong = pos.position_side === 'LONG';
                  const isPnlPositive = pos.unrealized_pnl_tl >= 0;
                  return (
                    <tr key={pos.symbol} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2 px-3 font-semibold text-white">
                        <div className="flex items-center space-x-1.5">
                          <span>{pos.symbol}</span>
                          <span className="text-[10px] text-slate-500 font-sans hidden sm:inline">
                            {pos.sector}
                          </span>
                        </div>
                      </td>
                      <td className="py-2 px-3 font-sans">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isLong
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                              : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                          }`}
                        >
                          {pos.position_side}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right text-slate-300">
                        {formatVolume(pos.open_stock_quantity)}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-300">
                        ₺{pos.fifo_avg_cost.toFixed(2)}
                      </td>
                      <td className="py-2 px-3 text-right text-white font-semibold">
                        ₺{pos.market_close_price.toFixed(2)}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-200">
                        {formatTL(pos.market_value_tl)}
                      </td>
                      <td
                        className={`py-2 px-3 text-right font-semibold ${
                          isPnlPositive ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {formatTL(pos.unrealized_pnl_tl)} ({formatPercent(pos.unrealized_pnl_pct)})
                      </td>
                      <td
                        className={`py-2 px-3 text-right ${
                          pos.daily_realized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {formatTL(pos.daily_realized_pnl_tl)}
                      </td>
                      <td className="py-2 px-3 text-center font-sans space-x-1">
                        <button
                          onClick={() => {
                            setActiveSymbol(pos.symbol);
                            setSubView('intelligence');
                          }}
                          className="px-2 py-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 rounded text-[11px] transition-colors"
                        >
                          Forensics
                        </button>
                        <button
                          onClick={() => onSelectSymbol && onSelectSymbol(pos.symbol)}
                          className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] transition-colors"
                        >
                          Chart
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Sub-View: Audited Open Lots ─────────────────────────────────────── */}
      {subView === 'lots' && (
        <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-xl">
          <div className="overflow-x-auto max-h-[550px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider sticky top-0 border-b border-slate-800 z-10">
                <tr>
                  <th className="py-2.5 px-3">Lot Identifier</th>
                  <th className="py-2.5 px-3">Symbol</th>
                  <th className="py-2.5 px-3">Direction</th>
                  <th className="py-2.5 px-3">Open Date</th>
                  <th className="py-2.5 px-3 text-right">Remaining Lots</th>
                  <th className="py-2.5 px-3 text-right">Unit Cost</th>
                  <th className="py-2.5 px-3 text-right">Remaining Value</th>
                  <th className="py-2.5 px-3 text-center">Days Held</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {filteredLots.map((lot) => (
                  <tr key={lot.lot_id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2 px-3 text-slate-400 text-[11px]">{lot.lot_id}</td>
                    <td className="py-2 px-3 font-semibold text-white">{lot.symbol}</td>
                    <td className="py-2 px-3 font-sans">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          lot.direction === 'LONG'
                            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                            : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                        }`}
                      >
                        {lot.direction}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-slate-300">{lot.open_date}</td>
                    <td className="py-2 px-3 text-right text-slate-200">
                      {formatVolume(lot.remaining_quantity)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">
                      ₺{lot.unit_cost.toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-right text-white font-semibold">
                      {formatTL(lot.remaining_value_tl)}
                    </td>
                    <td className="py-2 px-3 text-center">
                      <span className="px-2 py-0.5 bg-slate-800 rounded text-slate-300 text-[11px]">
                        {lot.days_held}d
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Sub-View: PnL Performance History ───────────────────────────────── */}
      {subView === 'history' && (
        <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-xl p-4">
          <div className="text-sm font-semibold text-white mb-3 flex items-center justify-between">
            <span>Historical Point-in-Time Realized PnL Progression</span>
            <span className="text-xs text-slate-400">Last 50 Sessions</span>
          </div>
          <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900 text-slate-400 uppercase tracking-wider sticky top-0 border-b border-slate-800">
                <tr>
                  <th className="py-2 px-3">Session Date</th>
                  <th className="py-2 px-3 text-right">Daily Realized PnL</th>
                  <th className="py-2 px-3 text-right">Intraday Matched PnL</th>
                  <th className="py-2 px-3 text-right">Carry FIFO Realized</th>
                  <th className="py-2 px-3 text-right">Day Net Flow</th>
                  <th className="py-2 px-3 text-right">MTM Valuation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {(history || []).slice().reverse().map((h) => (
                  <tr key={h.trade_date} className="hover:bg-slate-800/40">
                    <td className="py-2 px-3 text-slate-300 font-semibold">{h.trade_date}</td>
                    <td
                      className={`py-2 px-3 text-right font-bold ${
                        h.daily_realized_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(h.daily_realized_pnl_tl)}
                    </td>
                    <td
                      className={`py-2 px-3 text-right ${
                        h.intraday_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(h.intraday_pnl_tl)}
                    </td>
                    <td
                      className={`py-2 px-3 text-right ${
                        h.carry_pnl_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(h.carry_pnl_tl)}
                    </td>
                    <td
                      className={`py-2 px-3 text-right ${
                        h.net_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(h.net_flow_tl)}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-200">
                      {formatTL(h.mtm_valuation_tl)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
