import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, ExternalLink, Newspaper, ShieldAlert } from 'lucide-react';
import { fetchNewsRegime } from '../api/client';
import { formatPercent } from '../utils/formatters';

interface NewsRegimePanelProps {
  symbol: string;
}

const REGIME_LABELS: Record<string, { label: string; className: string }> = {
  FON_KRIZI_LIQUIDATION: { label: 'Fund crisis liquidation', className: 'bg-rose-500/10 text-rose-300 border-rose-500/30' },
  BOFA_DISTRIBUTION: { label: 'BofA distribution', className: 'bg-orange-500/10 text-orange-300 border-orange-500/30' },
  KAMU_FLOOR_DEFENSE: { label: 'State floor defense', className: 'bg-amber-500/10 text-amber-300 border-amber-500/30' },
  BULLISH_SOVEREIGN_OR_KAP_CATALYST: { label: 'Bullish catalyst', className: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' },
  NEUTRAL_FLOW: { label: 'Neutral flow', className: 'bg-slate-500/10 text-slate-300 border-slate-500/30' },
};

const GATE_MODE_LABELS: Record<string, string> = {
  off: 'Gate off',
  flow_only: 'Flow rules only',
  flow_news: 'Flow rules + keyword news',
};

const ForceBar: React.FC<{ label: string; value: number }> = ({ label, value }) => {
  const width = Math.min(Math.abs(value) * 5, 50);
  return (
    <div className="flex items-center gap-2 text-[11px] font-mono">
      <span className="w-10 text-slate-400">{label}</span>
      <div className="relative flex-1 h-2 rounded bg-slate-800">
        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600" />
        <div
          className={`absolute inset-y-0 rounded ${value >= 0 ? 'bg-emerald-500/70' : 'bg-rose-500/70'}`}
          style={value >= 0 ? { left: '50%', width: `${width}%` } : { right: '50%', width: `${width}%` }}
        />
      </div>
      <span className={`w-14 text-right ${value >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{formatPercent(value)}</span>
    </div>
  );
};

const HitCell: React.FC<{ value: number; beatsBoth?: boolean }> = ({ value, beatsBoth }) => (
  <td className={`px-2 py-1 text-right ${beatsBoth ? 'text-emerald-300 font-semibold' : 'text-slate-300'}`}>
    {value.toFixed(1)}%
  </td>
);

export const NewsRegimePanel: React.FC<NewsRegimePanelProps> = ({ symbol }) => {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['newsRegime', symbol],
    queryFn: () => fetchNewsRegime(symbol),
    staleTime: 5 * 60 * 1000,
  });

  if (isLoading) {
    return <div className="glass-panel p-4 rounded-xl border border-slate-800/80 text-xs text-slate-400">Loading news and flow state...</div>;
  }
  if (isError || !data) {
    return <div className="glass-panel p-4 rounded-xl border border-rose-500/30 text-xs text-rose-300">News and flow state is unavailable.</div>;
  }

  const regime = data.regime;
  const badge = regime ? REGIME_LABELS[regime.regime] ?? REGIME_LABELS.NEUTRAL_FLOW : null;

  return (
    <div className="glass-panel p-4 rounded-xl border border-slate-800/80 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Newspaper className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">Three Forces and News</h2>
          {badge && <span className={`text-[11px] px-2 py-0.5 rounded border font-mono ${badge.className}`}>{badge.label}</span>}
          {regime?.is_passive_absorption_trap && (
            <span className="text-[11px] px-2 py-0.5 rounded border font-mono bg-rose-500/10 text-rose-300 border-rose-500/30">Absorption trap</span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-slate-400">
          <span>Forecast gate: {GATE_MODE_LABELS[data.gate_mode] ?? data.gate_mode}</span>
          <span>As of {data.regime_latest_session ?? 'n/a'}</span>
        </div>
      </div>

      {data.is_stale && (
        <div className="flex items-start gap-2 text-xs rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-200 p-2.5">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>
            News and flow state is {data.sessions_behind ?? 'an unknown number of'} sessions behind the latest market session
            ({data.market_latest_session ?? 'n/a'}). Treat it as out of date.
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="space-y-4">
          {regime ? (
            <div className="space-y-1.5">
              <div className="text-[11px] text-slate-400 uppercase tracking-wider">5-day net flow, % of turnover</div>
              <ForceBar label="MLB" value={regime.mlb_5d_share_pct} />
              <ForceBar label="KAMU" value={regime.kamu_5d_share_pct} />
              <ForceBar label="BIG5" value={regime.big5_5d_share_pct} />
              <div className="text-[11px] text-slate-400 font-mono pt-1">
                Cost spread: MLB {formatPercent(regime.mlb_cost_spread_pct)}, BIG5 {formatPercent(regime.big5_cost_spread_pct)}
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-400">No flow state for {data.symbol}.</div>
          )}

          <div>
            <div className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">Gate track, last 90 sessions</div>
            {data.gate_horizons.length > 0 ? (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full text-[11px] font-mono">
                    <thead className="text-slate-400">
                      <tr>
                        <th className="px-2 py-1 text-left font-normal">Horizon</th>
                        <th className="px-2 py-1 text-right font-normal">No gate</th>
                        <th className="px-2 py-1 text-right font-normal">Flow</th>
                        <th className="px-2 py-1 text-right font-normal">Flow + news</th>
                        <th className="px-2 py-1 text-right font-normal">Always up</th>
                        <th className="px-2 py-1 text-right font-normal">Always down</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.gate_horizons.map((h) => (
                        <tr key={h.horizon} className="border-t border-slate-800/80">
                          <td className="px-2 py-1 text-slate-200 uppercase">{h.horizon}</td>
                          <HitCell value={h.hit_off_pct} />
                          <HitCell value={h.hit_flow_pct} beatsBoth={h.flow_beats_both} />
                          <HitCell value={h.hit_news_pct} beatsBoth={h.news_beats_both} />
                          <HitCell value={h.always_up_pct} />
                          <HitCell value={h.always_down_pct} />
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="flex items-start gap-1.5 text-[11px] text-slate-400 mt-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 shrink-0 mt-px" />
                  Not out-of-sample: the rules were written on this period. Green means the mode beat both constant calls.
                </p>
              </>
            ) : (
              <div className="text-xs text-slate-400">No gate track for {data.symbol}. Only the 6 focus stocks have one.</div>
            )}
          </div>

          {data.gate_changes.length > 0 && (
            <div>
              <div className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">Latest forecasts the gate changed</div>
              <ul className="space-y-1 text-[11px] font-mono text-slate-300 max-h-40 overflow-y-auto">
                {data.gate_changes.map((c) => (
                  <li key={`${c.trade_date}-${c.horizon}`} className="flex flex-wrap gap-x-2">
                    <span className="text-slate-400">{c.trade_date} {c.horizon.toUpperCase()}</span>
                    <span>
                      {formatPercent(c.pred_off_pct)} → flow {formatPercent(c.pred_flow_pct)} / news {formatPercent(c.pred_news_pct)}
                    </span>
                    <span className="text-slate-400">actual {formatPercent(c.actual_return_pct)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div>
          <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
            <div className="text-[11px] text-slate-400 uppercase tracking-wider">Headlines</div>
            <div className="text-[11px] text-slate-400 font-mono">
              {data.live_headline_count} live, {data.backfill_headline_count} backfill, {data.llm_labels} LLM shadow labels
            </div>
          </div>
          {data.headlines.length > 0 ? (
            <ul className="space-y-2 max-h-80 overflow-y-auto pr-1">
              {data.headlines.map((h) => (
                <li key={`${h.published_at_trt}-${h.title}`} className="text-xs text-slate-200">
                  {h.link ? (
                    <a href={h.link} target="_blank" rel="noopener noreferrer" className="hover:text-cyan-300">
                      {h.title} <ExternalLink className="inline w-3 h-3 text-slate-400" />
                    </a>
                  ) : (
                    h.title
                  )}
                  <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                    {h.published_at_trt} TRT · counts from {h.effective_trade_date} · {h.collection_mode.toLowerCase()}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-slate-400">No headlines mapped to a completed session yet.</div>
          )}
        </div>
      </div>
    </div>
  );
};
