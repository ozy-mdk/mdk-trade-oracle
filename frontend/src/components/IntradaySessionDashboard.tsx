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
import { Award, Clock, Info, Star, Trophy } from 'lucide-react';

/**
 * Intraday Session Forecaster: every evening, for every stock, a forecast of the % change from the 11:30 price to the
 * close, as a function of the morning move (11:30 price vs today's open). HIGH conviction where the BofA-flow rule
 * fires, NORMAL elsewhere. Layout follows the Multi-Horizon Hub: stock selector, today's forecast with the 11:30
 * calculator, audited walk-forward reality track, tournament standings.
 */

const FOCUS = ['ASELS', 'GARAN', 'AKBNK', 'YKBNK', 'ISCTR', 'TTKOM'];
const MONTHS: Record<number, string> = { 63: '3 months', 126: '6 months', 252: '12 months', 504: '24 months' };
const GRID_SHOW = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3];
const COST = 0.14;

const pct = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? '–' : `${x > 0 ? '+' : ''}${x.toFixed(d)}%`);
const tone = (x: number | null | undefined) => (x === null || x === undefined || x === 0 ? 'text-slate-400' : x > 0 ? 'text-emerald-400' : 'text-rose-400');
const tickSize = (p: number) => (p < 20 ? 0.01 : p < 50 ? 0.02 : p < 100 ? 0.05 : p < 250 ? 0.1 : p < 500 ? 0.25 : p < 1000 ? 0.5 : 1);
const toTick = (v: number, up: boolean) => {
  const t = tickSize(v);
  return (up ? Math.ceil(v / t) : Math.floor(v / t)) * t;
};

function gridAt(sig: IntradaySignalResponse, m: number): { f: number; high: boolean } {
  const xs = sig.grid_morning_pct;
  const ys = sig.grid_forecast_pct;
  if (!xs.length) return { f: 0, high: false };
  const x = Math.min(Math.max(m, xs[0]), xs[xs.length - 1]);
  let f = ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      f = ys[i - 1] + w * (ys[i] - ys[i - 1]);
      break;
    }
  }
  let near = 0;
  xs.forEach((v, i) => {
    if (Math.abs(v - x) < Math.abs(xs[near] - x)) near = i;
  });
  return { f, high: Boolean(sig.grid_high_conviction[near]) };
}

const Stat: React.FC<{ label: string; value: React.ReactNode; sub?: React.ReactNode }> = ({ label, value, sub }) => (
  <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2">
    <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
    <div className="text-lg font-mono text-slate-100">{value}</div>
    {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
  </div>
);

const Conviction: React.FC<{ high: boolean }> = ({ high }) =>
  high ? (
    <span className="inline-flex items-center gap-1 text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/40"><Star className="w-3 h-3" />HIGH</span>
  ) : (
    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-slate-700/40 text-slate-300 border border-slate-600/40">NORMAL</span>
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
  const champLabel = sig ? `${sig.model} · ${sig.feature_set} · ${MONTHS[sig.training_window ?? 0] ?? `${sig.training_window} sessions`}` : 'Daily forecaster';

  return (
    <div className="space-y-4">
      <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2"><Clock className="w-4 h-4 text-amber-400" />Intraday Session Forecaster</h2>
          <span className="text-xs text-slate-400">A forecast every session: % change from the 11:30 price to the closing auction · ★ HIGH conviction when BofA flow and the morning move agree</span>
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
            <Info className="w-3.5 h-3.5" />The research behind this forecaster was done on the six focus stocks. For {selected}, its own track record below is the evidence.
          </div>
        )}
      </section>

      {signal.isLoading && <div className="text-slate-400 text-sm">Loading forecast…</div>}
      {signal.isError && <div className="text-rose-300 text-sm">No forecast published for {selected} yet.</div>}
      {sig && (
        <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="xl:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div className="text-slate-100 font-semibold">{sig.symbol} · forecast for {sig.session_date}</div>
              <div className="text-xs text-slate-400">built from the close of {sig.as_of_close}{sig.prev_close ? ` (${sig.prev_close.toFixed(2)} TL)` : ''} · {sig.source === 'LIVE' ? 'published live' : 'simulated'}</div>
            </div>

            <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-3 space-y-2">
              <div className="text-xs font-semibold text-cyan-300">11:30 check: enter today&apos;s opening price and the price at 11:30</div>
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
                const mm = (p1130 / open - 1) * 100;
                const { f, high } = gridAt(sig, mm);
                const dir = Math.sign(f);
                const expClose = p1130 * (1 + f / 100);
                const right = sig.win_typ_pct === null ? null : p1130 * (1 + (dir * sig.win_typ_pct) / 100);
                const wrong = sig.loss_typ_pct === null ? null : p1130 * (1 - (dir * sig.loss_typ_pct) / 100);
                return (
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                    <Stat label="Morning move" value={<span className={tone(mm)}>{pct(mm)}</span>} sub="11:30 price vs today's open" />
                    <Stat label="Forecast to the close" value={<span className={tone(f)}>{pct(f)}</span>} sub={<span className="flex items-center gap-1.5">conviction <Conviction high={high} /></span>} />
                    <Stat label="Expected close" value={`${expClose.toFixed(2)} TL`} sub={`from ${p1130.toFixed(2)} TL at 11:30`} />
                    <div className={`rounded-lg border px-3 py-2 ${dir > 0 ? 'bg-emerald-500/10 border-emerald-500/40' : dir < 0 ? 'bg-rose-500/10 border-rose-500/40' : 'bg-slate-800/40 border-slate-700'}`}>
                      <div className="text-[10px] uppercase tracking-wider text-slate-500">Suggestion</div>
                      <div className={`text-lg font-bold ${dir > 0 ? 'text-emerald-300' : dir < 0 ? 'text-rose-300' : 'text-slate-300'}`}>
                        {dir > 0 ? 'BUY until the close' : dir < 0 ? 'SELL until the close' : 'FLAT'}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        {dir !== 0 && Math.abs(f) < COST ? 'forecast smaller than trading costs · ' : ''}
                        {right !== null && wrong !== null && dir !== 0 ? `if right ≈ ${right.toFixed(2)} TL · if wrong ≈ ${wrong.toFixed(2)} TL` : ''}
                      </div>
                    </div>
                  </div>
                );
              })() : (
                <div className="text-[11px] text-slate-500">The forecast uses last night&apos;s published model and the two prices you enter; nothing is sent anywhere.</div>
              )}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-slate-500">
                    <td className="py-1 pr-2 font-sans">If at 11:30 the price is this far from today&apos;s open…</td>
                    {GRID_SHOW.map((mv) => <td key={mv} className="text-right px-1.5">{pct(mv, 1)}</td>)}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="py-1 pr-2 font-sans text-slate-400">…our forecast from 11:30 to the close</td>
                    {GRID_SHOW.map((mv) => {
                      const { f, high } = gridAt(sig, mv);
                      return (
                        <td key={mv} className={`text-right px-1.5 ${tone(f)} ${high ? 'font-bold' : ''}`}>
                          {high && <span className="text-amber-300">★</span>}{pct(f)}
                        </td>
                      );
                    })}
                  </tr>
                </tbody>
              </table>
              <div className="text-[11px] text-slate-500 pt-1">
                ★ = HIGH conviction
                {sig.buy_level_pct !== null && `: morning move ≥ ${pct(sig.buy_level_pct, 1)}${hasOpen ? ` (price ≥ ${toTick(open * (1 + sig.buy_level_pct / 100), true).toFixed(2)} TL)` : ''}`}
                {sig.sell_level_pct !== null && `${sig.buy_level_pct !== null ? ' or' : ':'} ≤ ${pct(sig.sell_level_pct, 1)}${hasOpen ? ` (price ≤ ${toTick(open * (1 + sig.sell_level_pct / 100), false).toFixed(2)} TL)` : ''}`}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <Stat label="Record, last 60 sessions" value={sig.record_hit_rate_pct === null ? '–' : `${sig.record_hit_rate_pct.toFixed(0)}% right`}
                sub={sig.record_high_hit_rate_pct === null ? 'a forecast every session' : `★ HIGH conviction: ${sig.record_high_hit_rate_pct.toFixed(0)}% right (${sig.record_high_sessions} sessions)`} />
              <Stat label="When right" value={<span className="text-emerald-400">{sig.win_typ_pct === null ? '–' : `${sig.win_typ_pct.toFixed(1)}%`}</span>}
                sub={sig.win_q75_pct === null ? 'typical move in the forecast direction' : `typical move in the forecast direction · 1 in 4 above ${sig.win_q75_pct.toFixed(1)}%`} />
              <Stat label="When wrong" value={<span className="text-rose-400">{sig.loss_typ_pct === null ? '–' : `${sig.loss_typ_pct.toFixed(1)}%`}</span>} sub="typical move against the forecast" />
            </div>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="text-slate-100 font-semibold flex items-center gap-2"><Award className="w-4 h-4 text-amber-400" />Model</div>
            <div className="text-sm text-slate-200">Daily forecaster: {sig.model}</div>
            <div className="text-xs text-slate-400">Features: {sig.feature_set} (yesterday&apos;s institutional flows) + today&apos;s morning move</div>
            <div className="text-xs text-slate-400">Trained every evening on the last {sig.training_window} sessions ({MONTHS[sig.training_window ?? 0] ?? '—'}) up to the latest close</div>
            <div className="text-xs text-slate-400">Chosen on {sig.crowned_on} as the best of 60 candidates over the previous {tourney.data?.crown_sessions ?? 120} sessions{sig.crown_hit_rate_pct === null ? '' : `: right ${sig.crown_hit_rate_pct}% of days`}</div>
            <div className="text-xs text-slate-400">★ HIGH conviction: where yesterday&apos;s BofA flow and the morning move point the same way strongly, that rule&apos;s forecast is used</div>
          </div>
        </section>
      )}

      <IntradayWalkForwardTrack
        items={ledger.data?.items ?? []}
        summary={ledger.data?.summary ?? []}
        championLabel={champLabel}
        isLoading={ledger.isLoading}
        costPct={COST}
      />

      {tourney.data && (
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-2">
          <div className="text-slate-100 font-semibold flex items-center gap-2"><Trophy className="w-4 h-4 text-amber-400" />Tournament standings, chosen {tourney.data.month_start}</div>
          <div className="text-xs text-slate-400">Every candidate&apos;s day-by-day forecasts over the {tourney.data.crown_sessions} sessions before the choice date (at least {tourney.data.min_sessions} sessions to qualify). Ranked by hit rate, then error.</div>
          <div className="overflow-x-auto max-h-[360px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-900 text-slate-500 uppercase tracking-wider text-[10px]">
                <tr><th className="text-left py-1.5 px-2">#</th><th className="text-left px-2">Model</th><th className="text-left px-2">Features</th><th className="text-right px-2">Window</th>
                  <th className="text-right px-2">Sessions</th><th className="text-right px-2">Hit rate</th><th className="text-right px-2">MAE</th></tr>
              </thead>
              <tbody className="font-mono">
                {tourney.data.standings.slice(0, 25).map((s, i) => (
                  <tr key={s.candidate} className={`border-t border-slate-800/70 ${s.is_champion ? 'bg-amber-500/10' : ''} ${s.eligible ? '' : 'opacity-50'}`}>
                    <td className="py-1 px-2 text-slate-500">{i + 1}</td>
                    <td className="px-2 font-sans text-slate-200">{s.model}{s.is_champion && <span className="ml-1.5 text-amber-300 text-[10px]">chosen</span>}</td>
                    <td className="px-2 font-sans text-slate-400">{s.feature_set}</td>
                    <td className="text-right px-2 text-slate-400">{MONTHS[s.training_window] ?? s.training_window}</td>
                    <td className="text-right px-2 text-slate-300">{s.sessions}</td>
                    <td className="text-right px-2 text-slate-200">{s.hit_rate_pct.toFixed(1)}%</td>
                    <td className="text-right px-2 text-slate-400">{s.mae_pp.toFixed(2)}</td>
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
