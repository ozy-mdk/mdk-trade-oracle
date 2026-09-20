import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchTertipPortfolio, fetchTertipLots, fetchTertipHistory } from '../api/client';
import { formatTL, formatPercent, formatVolume } from '../utils/formatters';
import { Briefcase, TrendingUp, TrendingDown, Search, Layers, Shield } from 'lucide-react';

interface TertipDashboardProps {
  brokerId: string;
  tradeDate?: string;
  onSelectSymbol?: (symbol: string) => void;
}

export const TertipDashboard: React.FC<TertipDashboardProps> = ({
  brokerId,
  tradeDate,
  onSelectSymbol,
}) => {
  const [subView, setSubView] = useState<'positions' | 'lots' | 'history'>('positions');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // 1. Fetch Portfolio Positions
  const { data: portfolio } = useQuery({
    queryKey: ['tertipPortfolio', brokerId, tradeDate],
    queryFn: () => fetchTertipPortfolio(brokerId, tradeDate),
  });

  // 2. Fetch Open Lots
  const { data: lots } = useQuery({
    queryKey: ['tertipLots', brokerId],
    queryFn: () => fetchTertipLots(brokerId),
    enabled: subView === 'lots',
  });

  // 3. Fetch History
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
            Institution: <span className="text-slate-300 font-semibold">{brokerId}</span> | Positions:{' '}
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

      {/* Sub-View Navigation & Search Filter */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-1.5 bg-slate-900/90 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setSubView('positions')}
            className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${
              subView === 'positions'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Active Stock Inventory ({portfolio?.positions.length || 0})
          </button>
          <button
            onClick={() => setSubView('lots')}
            className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${
              subView === 'lots'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Audited Open Lots
          </button>
          <button
            onClick={() => setSubView('history')}
            className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${
              subView === 'history'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            PnL Performance Progression
          </button>
        </div>

        {subView !== 'history' && (
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
        )}
      </div>

      {/* Sub-View 1: Active Equity Positions Table */}
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
                      <td className="py-2 px-3 text-center font-sans">
                        <button
                          onClick={() => onSelectSymbol && onSelectSymbol(pos.symbol)}
                          className="px-2 py-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 rounded text-[11px] transition-colors"
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

      {/* Sub-View 2: Audited Open Lots */}
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

      {/* Sub-View 3: PnL Performance History */}
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
