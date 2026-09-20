import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchAllSignals } from '../api/client';
import { formatTL, formatPercent, getPlaybookStyle } from '../utils/formatters';
import { BrainCircuit, Compass, Target, ArrowUpRight, ArrowDownRight, Zap } from 'lucide-react';

interface OracleHubDashboardProps {
  onSelectSymbol?: (symbol: string) => void;
}

export const OracleHubDashboard: React.FC<OracleHubDashboardProps> = ({ onSelectSymbol }) => {
  const [sectorFilter, setSectorFilter] = useState<'all' | 'buy' | 'sell'>('all');
  const [reactionWindow, setReactionWindow] = useState<'w2' | 'w3' | 'w5'>('w2');

  const { data: signals } = useQuery({
    queryKey: ['allSignals'],
    queryFn: () => fetchAllSignals(),
    refetchInterval: 60000,
  });

  const macro = signals?.macro_day_start;
  const sectors = signals?.sector_allocations || [];
  const stocks = signals?.stock_reactions || [];

  const filteredSectors = sectors.filter((s) => {
    if (sectorFilter === 'buy') return s.predicted_open_net_flow_tl > 0;
    if (sectorFilter === 'sell') return s.predicted_open_net_flow_tl < 0;
    return true;
  });

  const macroPlaybookStyle = getPlaybookStyle(macro?.predicted_playbook || 'NEUTRAL_WAIT');
  const isMacroBuy = (macro?.predicted_open_net_flow_tl || 0) >= 0;

  return (
    <div className="space-y-5">
      {/* Live Upcoming Session Signal Card (T+1) */}
      <div className="glass-panel p-5 rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 -mt-4 -mr-4 w-40 h-40 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-wrap items-center justify-between gap-4 mb-4 border-b border-slate-800/80 pb-3">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-cyan-500/20 text-cyan-400 rounded-xl border border-cyan-500/30">
              <BrainCircuit className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-base font-bold text-white tracking-wide">
                  Model 1: Day-Start Macro Forecaster
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                  LIVE T+1 PREDICTION
                </span>
              </div>
              <div className="text-xs text-slate-400 mt-0.5">
                Target Session: <span className="text-white font-mono font-semibold">{signals?.forecast_date}</span> | Window: <span className="text-slate-300">09:55 - 10:30 TRT</span>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="text-right">
              <div className="text-[11px] text-slate-400">Directional Conviction</div>
              <div className="text-sm font-bold font-mono text-cyan-400">
                {macro ? `${(macro.direction_confidence * 100).toFixed(1)}%` : '—'}
              </div>
            </div>
            <div className={`px-3 py-1.5 rounded-lg border text-xs font-bold font-mono tracking-wider ${
              isMacroBuy
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 glow-green'
                : 'bg-rose-500/20 text-rose-300 border-rose-500/40 glow-red'
            }`}>
              {macro?.predicted_direction || 'NEUTRAL'}
            </div>
          </div>
        </div>

        {/* 3 Key Macro Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Net Flow TL */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
            <div className="text-xs text-slate-400 mb-1">Forecasted BofA Opening Net Flow</div>
            <div className={`text-2xl font-black font-mono ${
              isMacroBuy ? 'text-emerald-400 glow-green' : 'text-rose-400 glow-red'
            }`}>
              {macro ? formatTL(macro.predicted_open_net_flow_tl) : '—'}
            </div>
            <div className="text-[11px] text-slate-500 mt-2 font-mono">
              90% CI: [{macro ? formatTL(macro.predicted_open_flow_lower_90) : '—'} , {macro ? formatTL(macro.predicted_open_flow_upper_90) : '—'}]
            </div>
          </div>

          {/* Institutional Playbook */}
          <div className={`p-4 rounded-xl border ${macroPlaybookStyle.bg} ${macroPlaybookStyle.border}`}>
            <div className="text-xs text-slate-400 mb-1">Institutional Execution Playbook</div>
            <div className={`text-lg font-bold font-mono ${macroPlaybookStyle.text} flex items-center space-x-1.5`}>
              <Zap className="w-4 h-4" />
              <span>{macro?.predicted_playbook || 'NEUTRAL_WAIT'}</span>
            </div>
            <div className="text-[11px] text-slate-400 mt-2">
              Context blueprint for trader position sizing and risk management
            </div>
          </div>

          {/* Sector Guidance */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
            <div className="text-xs text-slate-400 mb-1">Top Sector Rotation Guidance</div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400 flex items-center">
                  <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400 mr-1" /> Top Buy:
                </span>
                <span className="font-semibold text-emerald-300 font-mono">
                  {macro?.top_predicted_buy_sector || '—'}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400 flex items-center">
                  <ArrowDownRight className="w-3.5 h-3.5 text-rose-400 mr-1" /> Top Sell:
                </span>
                <span className="font-semibold text-rose-300 font-mono">
                  {macro?.top_predicted_sell_sector || '—'}
                </span>
              </div>
            </div>
            <div className="text-[10px] text-slate-500 mt-1 font-mono">
              Model: {macro?.model_name || 'Tournament Champion'} ({macro?.model_version || 'v2'})
            </div>
          </div>
        </div>
      </div>

      {/* Two Column Layout: Model 2 Sector Allocation & Model 3 Stock Reaction */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Model 2: Sector Day-Start Allocation */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3 text-xs">
            <div className="flex items-center space-x-2">
              <Compass className="w-4 h-4 text-cyan-400" />
              <span className="font-bold text-white text-sm">Model 2: Sector Capital Allocation</span>
            </div>
            <div className="flex items-center space-x-1 bg-slate-900 p-0.5 rounded border border-slate-800">
              {(['all', 'buy', 'sell'] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setSectorFilter(f)}
                  className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                    sectorFilter === f ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400'
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider sticky top-0 border-b border-slate-800 font-sans">
                <tr>
                  <th className="py-2 px-2.5">Sector</th>
                  <th className="py-2 px-2.5 text-right">Predicted Flow</th>
                  <th className="py-2 px-2.5 text-center">Direction</th>
                  <th className="py-2 px-2.5 text-right">Confidence</th>
                  <th className="py-2 px-2.5 text-right font-sans">Playbook</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredSectors.map((s) => (
                  <tr key={s.sector} className="hover:bg-slate-800/40">
                    <td className="py-2 px-2.5 font-sans font-semibold text-white">{s.sector}</td>
                    <td
                      className={`py-2 px-2.5 text-right font-bold ${
                        s.predicted_open_net_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatTL(s.predicted_open_net_flow_tl)}
                    </td>
                    <td className="py-2 px-2.5 text-center">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          s.predicted_open_net_flow_tl >= 0
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-rose-500/20 text-rose-300'
                        }`}
                      >
                        {s.direction}
                      </span>
                    </td>
                    <td className="py-2 px-2.5 text-right text-slate-300">
                      {(s.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-2.5 text-right text-slate-400 text-[11px] font-sans">
                      {s.playbook}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Model 3: Stock Intraday Reaction Rankings */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3 text-xs">
            <div className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-emerald-400" />
              <span className="font-bold text-white text-sm">Model 3: Stock Reaction Forecaster</span>
            </div>
            <div className="flex items-center space-x-1 bg-slate-900 p-0.5 rounded border border-slate-800">
              {(
                [
                  { id: 'w2', label: 'W2 (10:30-11:30)' },
                  { id: 'w3', label: 'W3 (11:30-14:30)' },
                  { id: 'w5', label: 'W5 (16:00-18:15)' },
                ] as const
              ).map((w) => (
                <button
                  key={w.id}
                  onClick={() => setReactionWindow(w.id)}
                  className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                    reactionWindow === w.id ? 'bg-emerald-500/20 text-emerald-300' : 'text-slate-400'
                  }`}
                >
                  {w.label}
                </button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider sticky top-0 border-b border-slate-800 font-sans">
                <tr>
                  <th className="py-2 px-2.5">Symbol</th>
                  <th className="py-2 px-2.5 text-right">Predicted Return</th>
                  <th className="py-2 px-2.5 text-center">90% CI Range</th>
                  <th className="py-2 px-2.5 text-center">Direction</th>
                  <th className="py-2 px-2.5 text-right font-sans">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {stocks.map((st) => (
                  <tr key={st.symbol} className="hover:bg-slate-800/40">
                    <td className="py-2 px-2.5 font-bold text-white">{st.symbol}</td>
                    <td
                      className={`py-2 px-2.5 text-right font-bold ${
                        st.predicted_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatPercent(st.predicted_return_pct)}
                    </td>
                    <td className="py-2 px-2.5 text-center text-slate-400 text-[11px]">
                      [{formatPercent(st.predicted_return_lower_90)}, {formatPercent(st.predicted_return_upper_90)}]
                    </td>
                    <td className="py-2 px-2.5 text-center">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          st.predicted_return_pct >= 0
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-rose-500/20 text-rose-300'
                        }`}
                      >
                        {st.predicted_direction}
                      </span>
                    </td>
                    <td className="py-2 px-2.5 text-right font-sans">
                      <button
                        onClick={() => onSelectSymbol && onSelectSymbol(st.symbol)}
                        className="px-2 py-0.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 rounded text-[10px] transition-colors"
                      >
                        Chart
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
