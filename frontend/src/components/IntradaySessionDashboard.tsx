import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  fetchIntradayLedger,
  fetchIntradaySignal,
  fetchIntradayTournament,
  fetchIntradayUniverse,
} from '../api/client';
import { IntradaySignalResponse } from '../types/api';
import { IntradayWalkForwardTrack } from './IntradayWalkForwardTrack';
import { Award, Clock, Info, Trophy, TrendingDown, TrendingUp } from 'lucide-react';

/**
 * Intraday Session Forecaster: published after the previous close; the trader compares the 11:30 price with the day's
 * open and trades only when a level is crossed, exiting at the closing auction. Layout follows the Multi-Horizon Hub:
 * stock selector, today's signal, audited walk-forward reality track (chart + ledger), tournament standings.
 */

const FOCUS = ['ASELS', 'GARAN', 'AKBNK', 'YKBNK', 'ISCTR', 'TTKOM'];
const MONTHS: Record<number, string> = { 63: '3 months', 126: '6 months', 252: '12 months', 504: '24 months' };
const GRID_SHOW = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3];

const pct = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? '–' : `${x > 0 ? '+' : ''}${x.toFixed(d)}%`);
const tone = (x: number | null | undefined) => (x === null || x === undefined || x === 0 ? 'text-slate-400' : x > 0 ? 'text-emerald-400' : 'text-rose-400');

const tickSize = (p: number) => (p < 20 ? 0.01 : p < 50 ? 0.02 : p < 100 ? 0.05 : p < 250 ? 0.1 : p < 500 ? 0.25 : p < 1000 ? 0.5 : 1);
const toTick = (v: number, up: boolean) => {
  const t = tickSize(v);
  return (up ? Math.ceil(v / t) : Math.floor(v / t)) * t;
};

function gridAt(sig: IntradaySignalResponse, m: number): number {
  const xs = sig.grid_morning_pct;
  const ys = sig.grid_forecast_pct;
  if (!xs.length) return 0;
  const x = Math.min(Math.max(m, xs[0]), xs[xs.length - 1]);
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + w * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

const Stat: React.FC<{ label: string; value: React.ReactNode; sub?: React.ReactNode }> = ({ label, value, sub }) => (
  <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2">
    <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
    <div className="text-lg font-mono text-slate-100">{value}</div>
    {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
  </div>
);

export const IntradaySessionDashboard: React.FC<{ symbol?: string }> = ({ symbol }) => {
  const [selected, setSelected] = useState<string>(symbol && FOCUS.includes(symbol) ? symbol : 'GARAN');
  const [openPrice, setOpenPrice] = useState<string>('');
  const [price1130, setPrice1130] = useState<string>('');
  useEffect(() => {
    setOpenPrice('');
    setPrice1130('');
  }, [selected]);

  const universe = useQuery({ queryKey: ['intraday-universe'], queryFn: fetchIntradayUniverse, staleTime: 300_000 });
  const signal = useQuery({ queryKey: ['intraday-signal', selected], queryFn: () => fetchIntradaySignal(selected), staleTime: 60_000 });
  const ledger = useQuery({ queryKey: ['intraday-ledger', selected], queryFn: () => fetchIntradayLedger(selected), staleTime: 60_000 });
  const tourney = useQuery({ queryKey: ['intraday-tournament', selected], queryFn: () => fetchIntradayTournament(selected), staleTime: 300_000 });

  const others = (universe.data || []).filter((u) => !u.is_focus).map((u) => u.symbol);
  const sig = signal.data;
  const open = parseFloat(openPrice.replace(',', '.'));
  const hasOpen = Number.isFinite(open) && open > 0;
  const p1130 = parseFloat(price1130.replace(',', '.'));
  const hasP1130 = Number.isFinite(p1130) && p1130 > 0;
  const cost = 0.14;

  return (
    <div className="space-y-4">
      <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2"><Clock className="w-4 h-4 text-amber-400" />Intraday Session Forecaster</h2>
          <span className="text-xs text-slate-400">At 11:30 compare the price with today&apos;s open · trade only if a level is crossed · exit at the closing auction</span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {FOCUS.map((s) => (
            <button key={s} onClick={() => setSelected(s)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold border ${selected === s ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' : 'text-slate-300 border-slate-700 hover:bg-slate-800/60'}`}>
              {s}
            </button>
          ))}
          <label className="ml-2 text-xs text-slate-400 flex items-center gap-1.5" htmlFor="intraday-other">
            Other BIST30
            <select id="intraday-other" value={others.includes(selected) ? selected : ''} onChange={(e) => e.target.value && setSelected(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-slate-200 text-xs">
              <option value="">choose…</option>
              {others.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
        </div>
        {!FOCUS.includes(selected) && (
          <div className="text-xs text-amber-300/90 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-1.5 flex items-center gap-2">
            <Info className="w-3.5 h-3.5" />The research behind this signal was done on the six focus stocks. For {selected}, its own track record below is the evidence.
          </div>
        )}
      </section>

      {signal.isLoading && <div className="text-slate-400 text-sm">Loading signal…</div>}
      {signal.isError && <div className="text-rose-300 text-sm">No signal published for {selected} yet.</div>}
      {sig && (
        <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="xl:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div className="text-slate-100 font-semibold">{sig.symbol} · session {sig.session_date}</div>
              <div className="text-xs text-slate-400">built from the close of {sig.as_of_close}{sig.prev_close ? ` (${sig.prev_close.toFixed(2)} TL)` : ''} · {sig.source === 'LIVE' ? 'published live' : 'simulated'}</div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/30 p-3">
                <div className="text-[10px] uppercase tracking-wider text-emerald-300 flex items-center gap-1"><TrendingUp className="w-3 h-3" />Buy at 11:30 if</div>
                <div className="text-xl font-mono text-emerald-300">{sig.buy_level_pct === null ? 'no buy level' : `≥ ${pct(sig.buy_level_pct, 1)}`}</div>
                <div className="text-xs text-slate-400 font-mono">{hasOpen && sig.buy_level_pct !== null ? `price ≥ ${toTick(open * (1 + sig.buy_level_pct / 100), true).toFixed(2)} TL` : 'vs today\'s open'}</div>
              </div>
              <div className="rounded-lg bg-rose-500/10 border border-rose-500/30 p-3">
                <div className="text-[10px] uppercase tracking-wider text-rose-300 flex items-center gap-1"><TrendingDown className="w-3 h-3" />Sell at 11:30 if</div>
                <div className="text-xl font-mono text-rose-300">{sig.sell_level_pct === null ? 'no sell level' : `≤ ${pct(sig.sell_level_pct, 1)}`}</div>
                <div className="text-xs text-slate-400 font-mono">{hasOpen && sig.sell_level_pct !== null ? `price ≤ ${toTick(open * (1 + sig.sell_level_pct / 100), false).toFixed(2)} TL` : 'vs today\'s open'}</div>
              </div>
            </div>
            <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-3 space-y-2">
              <div className="text-xs font-semibold text-cyan-300">Your 11:30 check: enter today&apos;s prices to get the forecast for the close</div>
              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
                <label className="flex items-center gap-2" htmlFor="intraday-open">
                  Opening price (TL)
                  <input id="intraday-open" inputMode="decimal" value={openPrice} onChange={(e) => setOpenPrice(e.target.value)} placeholder={sig.prev_close ? sig.prev_close.toFixed(2) : ''}
                    className="w-28 bg-slate-950 border border-slate-700 rounded-md px-2 py-1 font-mono text-slate-100" />
                </label>
                <label className="flex items-center gap-2" htmlFor="intraday-1130">
                  Price at 11:30 (TL)
                  <input id="intraday-1130" inputMode="decimal" value={price1130} onChange={(e) => setPrice1130(e.target.value)} placeholder={openPrice || (sig.prev_close ? sig.prev_close.toFixed(2) : '')}
                    className="w-28 bg-slate-950 border border-slate-700 rounded-md px-2 py-1 font-mono text-slate-100" />
                </label>
              </div>
              {hasOpen && hasP1130 ? (() => {
                const m = (p1130 / open - 1) * 100;
                const f = gridAt(sig, m);
                const trade = sig.trade_threshold_pct === null ? f !== 0 : Math.abs(f) >= sig.trade_threshold_pct;
                const dir = trade ? Math.sign(f) : 0;
                const expClose = p1130 * (1 + f / 100);
                const right = sig.win_typ_pct === null ? null : p1130 * (1 + (dir * sig.win_typ_pct) / 100);
                const wrong = sig.loss_typ_pct === null ? null : p1130 * (1 - (dir * sig.loss_typ_pct) / 100);
                const outside = Math.abs(m) > 6;
                return (
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                    <Stat label="Morning move" value={<span className={tone(m)}>{pct(m)}</span>} sub="11:30 price vs today's open" />
                    <Stat label="Forecast to the close" value={<span className={tone(f)}>{pct(f)}</span>} sub={outside ? 'morning move beyond ±6%: read at the edge' : 'from the 11:30 price'} />
                    <Stat label="Expected close" value={`${expClose.toFixed(2)} TL`} sub={`from ${p1130.toFixed(2)} TL at 11:30`} />
                    <div className={`rounded-lg border px-3 py-2 ${dir > 0 ? 'bg-emerald-500/10 border-emerald-500/40' : dir < 0 ? 'bg-rose-500/10 border-rose-500/40' : 'bg-slate-800/40 border-slate-700'}`}>
                      <div className="text-[10px] uppercase tracking-wider text-slate-500">Suggestion</div>
                      <div className={`text-lg font-bold ${dir > 0 ? 'text-emerald-300' : dir < 0 ? 'text-rose-300' : 'text-slate-300'}`}>{dir > 0 ? 'BUY until the close' : dir < 0 ? 'SELL until the close' : 'NO TRADE'}</div>
                      <div className="text-[11px] text-slate-400">
                        {dir !== 0 && right !== null && wrong !== null
                          ? `if right, typically ≈ ${right.toFixed(2)} TL · if wrong, typically ≈ ${wrong.toFixed(2)} TL`
                          : 'expected move too small to cover costs and noise'}
                      </div>
                    </div>
                  </div>
                );
              })() : (
                <div className="text-[11px] text-slate-500">The forecast uses only last night&apos;s published signal and the two prices you enter; nothing is sent anywhere.</div>
              )}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <Stat label="Potential" value={<span className="text-emerald-400">{sig.win_typ_pct === null ? '–' : `+${sig.win_typ_pct.toFixed(1)}%`}</span>}
                sub={sig.win_q75_pct === null ? 'if it goes your way, typically' : `if it goes your way, typically · 1 in 4 above +${sig.win_q75_pct.toFixed(1)}%`} />
              <Stat label="Risk" value={<span className="text-rose-400">{sig.loss_typ_pct === null ? '–' : `−${sig.loss_typ_pct.toFixed(1)}%`}</span>} sub="if it goes against you, typically" />
              <Stat label="Record" value={sig.record_hit_rate_pct === null ? '–' : `${sig.record_hit_rate_pct.toFixed(0)}%`}
                sub={`right on the last ${sig.record_trades ?? 0} trades · ${pct(sig.record_avg_after_cost_pct)} per trade after costs`} />
            </div>
            {(() => {
              const isTrade = (f: number) => (sig.trade_threshold_pct === null ? f !== 0 : Math.abs(f) >= sig.trade_threshold_pct);
              const cells = GRID_SHOW.map((m) => ({ m, f: gridAt(sig, m) }));
              if (!cells.some((c) => isTrade(c.f))) {
                return (
                  <div className="text-xs text-slate-300 bg-slate-800/40 border border-slate-700 rounded-md px-3 py-2">
                    No trade for {sig.symbol} this session: no morning move between −3% and +3% gives a forecast strong enough to act on.
                  </div>
                );
              }
              return (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="text-slate-500">
                        <td className="py-1 pr-2 font-sans">If at 11:30 the price is this far from today&apos;s open…</td>
                        {cells.map((c) => <td key={c.m} className="text-right px-1.5">{pct(c.m, 1)}</td>)}
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="py-1 pr-2 font-sans text-slate-400">…our forecast from 11:30 to the close</td>
                        {cells.map((c) => (
                          <td key={c.m} className={`text-right px-1.5 ${isTrade(c.f) ? tone(c.f) : 'text-slate-600'}`}>
                            {isTrade(c.f) ? `${c.f > 0 ? 'BUY ' : 'SELL '}${pct(c.f)}` : '–'}
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                  <div className="text-[11px] text-slate-500 pt-1">– = no trade at that morning move.</div>
                </div>
              );
            })()}
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="text-slate-100 font-semibold flex items-center gap-2"><Award className="w-4 h-4 text-amber-400" />Champion</div>
            <div className="text-sm text-slate-200">{sig.model === 'RULE' ? 'Rule (BofA flow + morning move)' : sig.model}</div>
            <div className="text-xs text-slate-400">Features: {sig.feature_set} · trained on the last {sig.training_window} sessions ({MONTHS[sig.training_window ?? 0] ?? '—'})</div>
            <div className="text-xs text-slate-400">Crowned on {sig.crowned_on} from the previous {tourney.data?.crown_sessions ?? 120} sessions: {sig.crown_hit_rate_pct === null ? 'default (no candidate had enough trades)' : `${sig.crown_hit_rate_pct}% right on ${sig.crown_trades} trades`}</div>
            <div className="text-[11px] text-slate-500 pt-1">Re-crowned on the first session of each month. The rule stays champion unless a challenger beat it by 2 points of hit rate with a profit after costs over the previous {tourney.data?.crown_sessions ?? 120} sessions; regression models only trade on forecasts of 1% or more.</div>
          </div>
        </section>
      )}

      <IntradayWalkForwardTrack
        items={ledger.data?.items ?? []}
        summary={ledger.data?.summary ?? []}
        championLabel={sig ? (sig.model === 'RULE' ? 'Rule' : `${sig.model} (${MONTHS[sig.training_window ?? 0] ?? `${sig.training_window}d`})`) : 'Champion'}
        featureLabel={sig?.feature_set}
        isLoading={ledger.isLoading}
        costPct={cost}
      />

      {tourney.data && (
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-2">
          <div className="text-slate-100 font-semibold flex items-center gap-2"><Trophy className="w-4 h-4 text-amber-400" />Tournament standings, crowned {tourney.data.month_start}</div>
          <div className="text-xs text-slate-400">Each candidate&apos;s out-of-sample trades over the {tourney.data.crown_sessions} sessions before the crowning date (at least {tourney.data.min_trades} trades to qualify). Ranked by hit rate, then gain after costs.</div>
          <div className="overflow-x-auto max-h-[360px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-900 text-slate-500 uppercase tracking-wider text-[10px]">
                <tr><th className="text-left py-1.5 px-2">#</th><th className="text-left px-2">Model</th><th className="text-left px-2">Features</th><th className="text-right px-2">Window</th>
                  <th className="text-right px-2">Trades</th><th className="text-right px-2">Hit rate</th><th className="text-right px-2">Gain / trade</th></tr>
              </thead>
              <tbody className="font-mono">
                {tourney.data.standings.slice(0, 25).map((s, i) => (
                  <tr key={s.candidate} className={`border-t border-slate-800/70 ${s.is_champion ? 'bg-amber-500/10' : ''} ${s.eligible ? '' : 'opacity-50'}`}>
                    <td className="py-1 px-2 text-slate-500">{i + 1}</td>
                    <td className="px-2 font-sans text-slate-200">{s.model === 'RULE' ? 'Rule' : s.model}{s.is_champion && <span className="ml-1.5 text-amber-300 text-[10px]">champion</span>}</td>
                    <td className="px-2 font-sans text-slate-400">{s.feature_set}</td>
                    <td className="text-right px-2 text-slate-400">{s.training_window}</td>
                    <td className="text-right px-2 text-slate-300">{s.trades}</td>
                    <td className="text-right px-2 text-slate-200">{s.hit_rate_pct.toFixed(1)}%</td>
                    <td className={`text-right px-2 ${tone(s.avg_gain_after_cost_pct)}`}>{pct(s.avg_gain_after_cost_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};
