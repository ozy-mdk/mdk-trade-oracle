import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchTimeWindowAnalysis } from '../api/client';
import { formatTL, formatVolume } from '../utils/formatters';
import { Sunrise, Sunset, BarChart3 } from 'lucide-react';

interface TimeWindowTerminalProps {
  symbol: string;
  brokerId: string;
  tradeDate?: string;
}

export const TimeWindowTerminal: React.FC<TimeWindowTerminalProps> = ({
  symbol,
  brokerId,
  tradeDate,
}) => {
  const { data } = useQuery({
    queryKey: ['timeWindowAnalysis', symbol, brokerId, tradeDate],
    queryFn: () => fetchTimeWindowAnalysis(symbol, brokerId, tradeDate),
  });

  const windows = data?.windows || [];

  // Calculate cumulative net flow curve
  let runningFlow = 0;
  const cumulativeFlows = windows.map((w) => {
    runningFlow += w.net_flow_tl;
    return {
      window_name: w.window_name,
      net_flow: w.net_flow_tl,
      cumulative_flow: runningFlow,
    };
  });

  const getWindowLabel = (name: string) => {
    switch (name) {
      case 'day_start':
        return { title: 'Window 1: Day Start', hours: '09:55 - 10:30 TRT', desc: 'Opening auction match & institutional positioning' };
      case 'first_reaction':
        return { title: 'Window 2: First Reaction', hours: '10:30 - 11:30 TRT', desc: 'Liquidity absorption & initial trend confirmation' };
      case 'midday_followup':
        return { title: 'Window 3: Midday Followup', hours: '11:30 - 14:30 TRT', desc: 'Slow institutional churn & VWAP pacing' };
      case 'afternoon_reaction':
        return { title: 'Window 4: Afternoon Reaction', hours: '14:30 - 16:00 TRT', desc: 'Pre-close repositioning & global market alignment' };
      case 'closing_session':
        return { title: 'Window 5: Closing Session', hours: '16:00 - 18:15 TRT', desc: 'Aggressive imbalance execution & closing auction' };
      default:
        return { title: name, hours: '', desc: '' };
    }
  };

  return (
    <div className="space-y-4">
      {/* Microstructure Auction Banners */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Opening Auction */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-amber-500/10 rounded-lg border border-amber-500/30 text-amber-400">
              <Sunrise className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs text-slate-400">Opening Auction (09:55 TRT) Flow</div>
              <div className="text-lg font-bold font-mono text-white">
                {data?.opening_auction_net_tl !== null && data?.opening_auction_net_tl !== undefined
                  ? formatTL(data.opening_auction_net_tl)
                  : '—'}
              </div>
              <div className="text-[11px] text-slate-500">Day-Start Opening Imbalance</div>
            </div>
          </div>
          <div className="text-right">
            <span
              className={`px-2.5 py-1 rounded text-xs font-semibold font-mono ${
                (data?.opening_auction_net_tl || 0) >= 0
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-rose-500/20 text-rose-300'
              }`}
            >
              {(data?.opening_auction_net_tl || 0) >= 0 ? 'NET BUYER' : 'NET SELLER'}
            </span>
          </div>
        </div>

        {/* Closing Auction */}
        <div className="glass-panel p-4 rounded-xl border border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-indigo-500/10 rounded-lg border border-indigo-500/30 text-indigo-400">
              <Sunset className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs text-slate-400">Closing Session (16:00 - 18:15 TRT) Flow</div>
              <div className="text-lg font-bold font-mono text-white">
                {data?.closing_auction_net_tl !== null && data?.closing_auction_net_tl !== undefined
                  ? formatTL(data.closing_auction_net_tl)
                  : '—'}
              </div>
              <div className="text-[11px] text-slate-500">Closing Imbalance & Dark Pool Volume</div>
            </div>
          </div>
          <div className="text-right">
            <span
              className={`px-2.5 py-1 rounded text-xs font-semibold font-mono ${
                (data?.closing_auction_net_tl || 0) >= 0
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-rose-500/20 text-rose-300'
              }`}
            >
              {(data?.closing_auction_net_tl || 0) >= 0 ? 'NET BUYER' : 'NET SELLER'}
            </span>
          </div>
        </div>
      </div>

      {/* Intraday Cumulative Net Flow Evolution Tracker */}
      <div className="glass-panel p-4 rounded-xl border border-slate-800">
        <div className="flex items-center justify-between mb-3 text-xs">
          <span className="font-semibold text-white flex items-center">
            <BarChart3 className="w-4 h-4 mr-1.5 text-cyan-400" />
            Cumulative Order Flow Trajectory Across Windows
          </span>
          <span className="text-slate-400 font-mono">
            Final Day Net: <span className={runningFlow >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
              {formatTL(runningFlow)}
            </span>
          </span>
        </div>

        {/* Step Progression Visualizer */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
          {cumulativeFlows.map((cf) => {
            const isPos = cf.cumulative_flow >= 0;
            const meta = getWindowLabel(cf.window_name);
            return (
              <div
                key={cf.window_name}
                className="bg-slate-900/90 border border-slate-800/90 rounded-lg p-3 relative overflow-hidden"
              >
                <div className="text-[11px] text-slate-400 font-medium">{meta.title.split(':')[0]}</div>
                <div className="text-xs text-slate-500 mb-2">{meta.hours.split(' ')[0]}</div>
                <div className="text-xs text-slate-400">Step Net:</div>
                <div className={`text-sm font-bold font-mono ${cf.net_flow >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {formatTL(cf.net_flow)}
                </div>
                <div className="mt-2 pt-2 border-t border-slate-800">
                  <div className="text-[10px] text-slate-500">Cumulative:</div>
                  <div className={`text-base font-bold font-mono ${isPos ? 'text-emerald-400 glow-green' : 'text-rose-400 glow-red'}`}>
                    {formatTL(cf.cumulative_flow)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Detailed 5-Window Execution Matrix */}
      <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-xl">
        <div className="p-3 bg-slate-900/80 border-b border-slate-800 flex items-center justify-between text-xs">
          <span className="font-semibold text-white">Institutional Intraday Breakdown ({symbol} - {brokerId})</span>
          <span className="text-slate-400">Timezone: Europe/Istanbul (TRT / UTC+3)</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider border-b border-slate-800 font-sans">
              <tr>
                <th className="py-2.5 px-3">Execution Window</th>
                <th className="py-2.5 px-3">Interval (TRT)</th>
                <th className="py-2.5 px-3 text-right">Buy Turnover</th>
                <th className="py-2.5 px-3 text-right">Sell Turnover</th>
                <th className="py-2.5 px-3 text-right">Net Flow (TL)</th>
                <th className="py-2.5 px-3 text-right">Total Turnover</th>
                <th className="py-2.5 px-3 text-right">Buy Lots</th>
                <th className="py-2.5 px-3 text-right">Sell Lots</th>
                <th className="py-2.5 px-3 text-right">Net Lots</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {windows.map((w) => {
                const meta = getWindowLabel(w.window_name);
                const isNetPos = w.net_flow_tl >= 0;
                return (
                  <tr key={w.window_name} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-white font-sans">
                      <div>{meta.title}</div>
                      <div className="text-[10px] text-slate-500 font-normal">{meta.desc}</div>
                    </td>
                    <td className="py-2.5 px-3 text-cyan-400 font-sans">{meta.hours}</td>
                    <td className="py-2.5 px-3 text-right text-slate-300">{formatTL(w.buy_turnover_tl)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-300">{formatTL(w.sell_turnover_tl)}</td>
                    <td
                      className={`py-2.5 px-3 text-right font-bold ${
                        isNetPos ? 'text-emerald-400 glow-green' : 'text-rose-400 glow-red'
                      }`}
                    >
                      {formatTL(w.net_flow_tl)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-white">{formatTL(w.total_turnover_tl)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-300">{formatVolume(w.buy_volume)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-300">{formatVolume(w.sell_volume)}</td>
                    <td
                      className={`py-2.5 px-3 text-right font-semibold ${
                        w.net_volume >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {formatVolume(w.net_volume)}
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
