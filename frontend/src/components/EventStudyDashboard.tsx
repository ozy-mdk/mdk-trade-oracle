import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchEventStudyScan } from '../api/client';
import { formatTL, formatPercent } from '../utils/formatters';
import { Search, Filter, Zap } from 'lucide-react';

interface EventStudyDashboardProps {
  initialSymbol?: string;
  brokerId: string;
  onSelectSymbol?: (symbol: string) => void;
}

export const EventStudyDashboard: React.FC<EventStudyDashboardProps> = ({
  initialSymbol,
  brokerId,
  onSelectSymbol,
}) => {
  const [symbol, setSymbol] = useState<string>(initialSymbol || '');
  const [flowPreset, setFlowPreset] = useState<'any' | 'buy50m' | 'buy100m' | 'sell50m'>('buy50m');
  const [d1Preset, setD1Preset] = useState<'all' | 'dip' | 'rally'>('all');

  // Compute query parameters based on presets
  const getQueryParams = () => {
    let minFlow: number | undefined;
    let maxFlow: number | undefined;
    if (flowPreset === 'buy50m') minFlow = 50_000_000;
    if (flowPreset === 'buy100m') minFlow = 100_000_000;
    if (flowPreset === 'sell50m') maxFlow = -50_000_000;

    let minD1: number | undefined;
    let maxD1: number | undefined;
    if (d1Preset === 'dip') maxD1 = -1.5; // Stocks that fell at least 1.5% the prior session
    if (d1Preset === 'rally') minD1 = 1.5; // Stocks that rose at least 1.5% the prior session

    return {
      brokerId,
      symbol: symbol.trim() ? symbol.toUpperCase() : undefined,
      minFlowTl: minFlow,
      maxFlowTl: maxFlow,
      minD1Return: minD1,
      maxD1Return: maxD1,
      limit: 100,
    };
  };

  const { data: events, refetch } = useQuery({
    queryKey: ['eventStudyScan', brokerId, symbol, flowPreset, d1Preset],
    queryFn: () => fetchEventStudyScan(getQueryParams()),
  });

  // Calculate Forward Return Statistics
  const validT1 = (events || []).filter((e) => e.return_t1 !== null && e.return_t1 !== undefined);
  const validT5 = (events || []).filter((e) => e.return_t5 !== null && e.return_t5 !== undefined);

  const t1WinRate = validT1.length > 0
    ? (validT1.filter((e) => (e.return_t1 || 0) > 0).length / validT1.length) * 100
    : 0;

  const t5WinRate = validT5.length > 0
    ? (validT5.filter((e) => (e.return_t5 || 0) > 0).length / validT5.length) * 100
    : 0;

  const avgT1Return = validT1.length > 0
    ? validT1.reduce((acc, e) => acc + (e.return_t1 || 0), 0) / validT1.length
    : 0;

  const avgT5Return = validT5.length > 0
    ? validT5.reduce((acc, e) => acc + (e.return_t5 || 0), 0) / validT5.length
    : 0;

  return (
    <div className="space-y-4">
      {/* Scanner Parameter Control Bar */}
      <div className="glass-panel p-4 rounded-xl border border-slate-800">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3 text-xs">
            {/* Symbol Filter */}
            <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5">
              <Search className="w-3.5 h-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Symbol (e.g. THYAO or empty for all)"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="bg-transparent text-slate-200 placeholder-slate-500 focus:outline-none w-48 font-mono text-xs"
              />
            </div>

            {/* Institutional Flow Threshold Preset */}
            <div className="flex items-center space-x-1 bg-slate-900 border border-slate-800 rounded-lg p-1">
              <span className="text-slate-400 px-2 flex items-center font-sans">
                <Zap className="w-3 h-3 mr-1 text-cyan-400" /> Flow:
              </span>
              {(
                [
                  { id: 'any', label: 'All Flows' },
                  { id: 'buy50m', label: 'Buy ≥ ₺50M' },
                  { id: 'buy100m', label: 'Buy ≥ ₺100M' },
                  { id: 'sell50m', label: 'Sell ≤ -₺50M' },
                ] as const
              ).map((f) => (
                <button
                  key={f.id}
                  onClick={() => setFlowPreset(f.id)}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
                    flowPreset === f.id
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>

            {/* D-1 Prior Return Conditioning Preset */}
            <div className="flex items-center space-x-1 bg-slate-900 border border-slate-800 rounded-lg p-1">
              <span className="text-slate-400 px-2 font-sans">Prior D-1:</span>
              {(
                [
                  { id: 'all', label: 'Any D-1' },
                  { id: 'dip', label: 'Buy The Dip (D-1 ≤ -1.5%)' },
                  { id: 'rally', label: 'Momentum (D-1 ≥ +1.5%)' },
                ] as const
              ).map((d) => (
                <button
                  key={d.id}
                  onClick={() => setD1Preset(d.id)}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
                    d1Preset === d.id
                      ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => refetch()}
            className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs rounded-lg transition-colors shadow-lg shadow-cyan-900/30 flex items-center space-x-1.5"
          >
            <Filter className="w-3.5 h-3.5" />
            <span>Scan Events</span>
          </button>
        </div>
      </div>

      {/* Aggregate Forward Return Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="text-slate-400 text-xs mb-1">Occurrences Scanned</div>
          <div className="text-2xl font-bold font-mono text-white">{events?.length || 0}</div>
          <div className="text-[11px] text-slate-500 mt-1">
            Institutional execution events matching criteria
          </div>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="text-slate-400 text-xs mb-1">T+1 Forward Win Rate</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 glow-green">
            {t1WinRate.toFixed(1)}%
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Avg T+1 Return: <span className={avgT1Return >= 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
              {formatPercent(avgT1Return)}
            </span>
          </div>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="text-slate-400 text-xs mb-1">T+5 Forward Win Rate</div>
          <div className="text-2xl font-bold font-mono text-teal-400">
            {t5WinRate.toFixed(1)}%
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Avg T+5 Return: <span className={avgT5Return >= 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
              {formatPercent(avgT5Return)}
            </span>
          </div>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="text-slate-400 text-xs mb-1">Institutional Edge Signal</div>
          <div className="text-xl font-bold text-cyan-300 font-mono">
            {t1WinRate >= 60 ? 'HIGH ALPHA' : t1WinRate >= 50 ? 'MODERATE EDGE' : 'NEUTRAL / CAUTION'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Empirical post-flow reaction probability
          </div>
        </div>
      </div>

      {/* Historical Event Study Scan Ledger */}
      <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-xl">
        <div className="overflow-x-auto max-h-[550px] overflow-y-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider sticky top-0 border-b border-slate-800 font-sans z-10">
              <tr>
                <th className="py-2.5 px-3">Date</th>
                <th className="py-2.5 px-3">Symbol</th>
                <th className="py-2.5 px-3 text-right">Close Price</th>
                <th className="py-2.5 px-3 text-right">Day Return</th>
                <th className="py-2.5 px-3 text-right">D-1 Return</th>
                <th className="py-2.5 px-3 text-right">{brokerId} Net Flow</th>
                <th className="py-2.5 px-3 text-center bg-cyan-950/20">T+1 Ret</th>
                <th className="py-2.5 px-3 text-center bg-cyan-950/20">T+2 Ret</th>
                <th className="py-2.5 px-3 text-center bg-cyan-950/20">T+3 Ret</th>
                <th className="py-2.5 px-3 text-center bg-cyan-950/20">T+5 Ret</th>
                <th className="py-2.5 px-3 text-center bg-cyan-950/20">T+10 Ret</th>
                <th className="py-2.5 px-3 text-center font-sans">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {(events || []).map((e, idx) => (
                <tr key={`${e.symbol}_${e.trade_date}_${idx}`} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-2 px-3 text-slate-300 font-semibold">{e.trade_date}</td>
                  <td className="py-2 px-3 text-white font-bold">{e.symbol}</td>
                  <td className="py-2 px-3 text-right text-slate-200">₺{e.close_price.toFixed(2)}</td>
                  <td
                    className={`py-2 px-3 text-right font-semibold ${
                      e.daily_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {formatPercent(e.daily_return_pct)}
                  </td>
                  <td
                    className={`py-2 px-3 text-right ${
                      (e.d1_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {e.d1_return_pct !== null && e.d1_return_pct !== undefined
                      ? formatPercent(e.d1_return_pct)
                      : '—'}
                  </td>
                  <td
                    className={`py-2 px-3 text-right font-bold ${
                      e.bofa_net_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {formatTL(e.bofa_net_flow_tl)}
                  </td>

                  {/* Forward Return Horizons */}
                  {[e.return_t1, e.return_t2, e.return_t3, e.return_t5, e.return_t10].map(
                    (ret, rIdx) => (
                      <td key={rIdx} className="py-2 px-3 text-center bg-cyan-950/10">
                        {ret !== null && ret !== undefined ? (
                          <span
                            className={`px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                              ret > 0
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                : ret < 0
                                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                : 'text-slate-400'
                            }`}
                          >
                            {formatPercent(ret)}
                          </span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                    )
                  )}

                  <td className="py-2 px-3 text-center font-sans">
                    <button
                      onClick={() => onSelectSymbol && onSelectSymbol(e.symbol)}
                      className="px-2 py-1 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 rounded text-[11px] transition-colors"
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
