import React, { useMemo, useState } from 'react';
import {
  Activity,
  Calendar,
  CheckCircle2,
  Layers,
  LineChart,
  MinusCircle,
  ShieldCheck,
  Star,
  Table as TableIcon,
  XCircle,
} from 'lucide-react';
import { IntradayLedgerItem, IntradayLedgerSummary } from '../types/api';
import { formatPercent } from '../utils/formatters';

/**
 * Audited walk-forward reality track for the Intraday Session Forecaster, mirroring the Multi-Horizon Hub:
 * chart / table / both, 30-60-90-180 session windows, scorecards, predicted vs realized chart (price or return) with
 * hit / miss points and hover tooltip, and the audit ledger. Every session has a forecast: prediction = 11:30 price x
 * (1 + forecast); reality = the closing-auction price. HIGH conviction sessions are marked and can be filtered.
 */

interface Props {
  items: IntradayLedgerItem[];
  summary: IntradayLedgerSummary[];
  championLabel: string;
  featureLabel?: string | null;
  isLoading?: boolean;
  costPct?: number;
}

type Row = IntradayLedgerItem & { pred_close: number | null; outcome: 'HIT' | 'MISS' | 'FLAT' };

export const IntradayWalkForwardTrack: React.FC<Props> = ({ items, summary, championLabel, featureLabel, isLoading, costPct = 0.14 }) => {
  const [evalSessions, setEvalSessions] = useState<number>(60);
  const [displayMode, setDisplayMode] = useState<'both' | 'chart' | 'ledger'>('both');
  const [chartView, setChartView] = useState<'price' | 'return'>('return');
  const [source, setSource] = useState<'ALL' | 'LIVE' | 'SIMULATED'>('ALL');
  const [onlyHigh, setOnlyHigh] = useState<boolean>(false);
  const [hovered, setHovered] = useState<string | null>(null);

  // newest first (ledger order), limited to the selected window
  const ledger = useMemo<Row[]>(() => {
    return items
      .filter((r) => source === 'ALL' || r.source === source)
      .slice(0, evalSessions)
      .filter((r) => !onlyHigh || r.conviction === 'HIGH')
      .map((r) => ({
        ...r,
        pred_close: r.price_1130 == null ? null : r.price_1130 * (1 + (r.forecast_pct ?? 0) / 100),
        outcome: r.call === 0 ? 'FLAT' : r.hit ? 'HIT' : 'MISS',
      }));
  }, [items, source, evalSessions, onlyHigh]);
  const chrono = useMemo(() => [...ledger].sort((a, b) => a.session_date.localeCompare(b.session_date)), [ledger]);

  const trades = useMemo(() => ledger.filter((r) => r.call !== 0), [ledger]);
  const hits = trades.filter((r) => r.hit).length;
  const hitRate = trades.length ? (100 * hits) / trades.length : null;
  const highRows = trades.filter((r) => r.conviction === 'HIGH');
  const highHits = highRows.filter((r) => r.hit).length;
  const mae = ledger.length ? ledger.reduce((a, r) => a + (r.abs_error_pp ?? 0), 0) / ledger.length : null;
  const hitMae = hits ? trades.filter((r) => r.hit).reduce((a, r) => a + (r.abs_error_pp ?? 0), 0) / hits : null;
  const missN = trades.length - hits;
  const missMae = missN ? trades.filter((r) => !r.hit).reduce((a, r) => a + (r.abs_error_pp ?? 0), 0) / missN : null;
  const gain = trades.length ? trades.reduce((a, r) => a + r.call * (r.actual_pct ?? 0), 0) / trades.length - costPct : null;
  const last10 = chrono.filter((r) => r.call !== 0).slice(-10);
  const l10Hits = last10.filter((r) => r.hit).length;
  const big = trades.filter((r) => Math.abs(r.actual_pct ?? 0) >= 2.0);
  const bigHits = big.filter((r) => r.hit).length;
  const live = summary.find((s) => s.source === 'LIVE');

  // SVG geometry (same as the Multi-Horizon Hub)
  const chartWidth = 900, chartHeight = 220, paddingX = 48, paddingY = 24;
  const innerW = chartWidth - paddingX * 2, innerH = chartHeight - paddingY * 2;
  const prices = chrono.flatMap((r) => [r.close_price ?? 0, r.pred_close ?? 0]).filter((p) => p > 0);
  const minPrice = prices.length ? Math.min(...prices) * 0.985 : 0;
  const maxPrice = prices.length ? Math.max(...prices) * 1.015 : 100;
  const rets = chrono.flatMap((r) => [Math.abs(r.actual_pct ?? 0), Math.abs(r.forecast_pct ?? 0)]);
  const maxAbs = rets.length ? Math.max(2.0, Math.max(...rets) * 1.15) : 5.0;
  const getX = (i: number) => (chrono.length <= 1 ? paddingX + innerW / 2 : paddingX + (i / (chrono.length - 1)) * innerW);
  const getYPrice = (v: number) => (maxPrice <= minPrice ? paddingY + innerH / 2 : paddingY + innerH - ((v - minPrice) / (maxPrice - minPrice)) * innerH);
  const getYReturn = (v: number) => paddingY + innerH / 2 - (v / maxAbs) * (innerH / 2);
  const yAct = (r: Row) => (chartView === 'price' ? getYPrice(r.close_price ?? 0) : getYReturn(r.actual_pct ?? 0));
  const yPred = (r: Row) => (chartView === 'price' ? getYPrice(r.pred_close ?? 0) : getYReturn(r.forecast_pct ?? 0));
  const path = (f: (r: Row) => number) => chrono.map((r, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${f(r).toFixed(1)}`).join(' ');
  const dateInterval = chrono.length > 50 ? Math.ceil(chrono.length / 8) : 5;
  const hoveredItem = hovered ? chrono.find((r) => r.session_date === hovered) ?? null : null;
  const dotColor = (r: Row) => (r.outcome === 'FLAT' ? '#475569' : r.outcome === 'HIT' ? '#10b981' : '#f43f5e');

  const pill = (active: boolean) =>
    `px-2 py-0.5 rounded-lg text-xs font-bold transition-all ${active ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm' : 'text-slate-400 hover:text-white'}`;

  return (
    <div className="glass-panel p-5 rounded-2xl border border-slate-800/80 shadow-xl space-y-4">
      {/* Section header controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-bold text-white tracking-wide">AUDITED WALK-FORWARD REALITY TRACK (11:30 → CLOSE)</h3>
              <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 font-mono font-semibold">
                {championLabel}
                {featureLabel && <span className="ml-1 text-cyan-400 font-bold">• {featureLabel}</span>}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Each session graded from the signal published the evening before · trailing {evalSessions} sessions
              {live && live.trades > 0 ? ` · live since ${live.first_session}` : ' · live track starts with the first published session'}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center bg-slate-950 rounded-xl border border-slate-800 p-0.5 font-mono text-xs">
            {([['chart', LineChart, 'Chart'], ['ledger', TableIcon, 'Table'], ['both', Layers, 'Both']] as const).map(([mode, Icon, label]) => (
              <button key={mode} onClick={() => setDisplayMode(mode)} className={`flex items-center space-x-1.5 px-2.5 py-1 ${pill(displayMode === mode)}`}>
                <Icon className="w-3.5 h-3.5" />
                <span>{label}</span>
              </button>
            ))}
          </div>
          <div className="flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 font-mono text-xs">
            {([30, 60, 90, 180] as const).map((n) => (
              <button key={n} onClick={() => setEvalSessions(n)} className={pill(evalSessions === n)}>{n}D</button>
            ))}
          </div>
          <div className="flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 font-mono text-xs">
            {(['ALL', 'LIVE', 'SIMULATED'] as const).map((s) => (
              <button key={s} onClick={() => setSource(s)} className={pill(source === s)}>{s === 'ALL' ? 'All' : s === 'LIVE' ? 'Live' : 'Simulated'}</button>
            ))}
          </div>
          <button onClick={() => setOnlyHigh(!onlyHigh)} className={`flex items-center gap-1 font-mono text-xs border border-slate-800 rounded-xl px-2 py-1 ${onlyHigh ? 'bg-amber-500/15 text-amber-300 border-amber-500/40' : 'text-slate-400 bg-slate-900/90'}`}>
            <Star className="w-3 h-3" />High conviction only
          </button>
        </div>
      </div>

      {/* 5 scorecard KPI metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <div className="text-[10px] text-slate-400 font-mono uppercase">Directional Hit Rate ({evalSessions}D)</div>
          <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">{hitRate == null ? '---' : `${hitRate.toFixed(1)}%`}</div>
          <div className="text-[10px] text-slate-400 font-mono">{hits} hits / {trades.length} sessions</div>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <div className="text-[10px] text-slate-400 font-mono uppercase">Mean Absolute Error ({evalSessions}D)</div>
          <div className="text-lg font-bold text-slate-200 font-mono mt-0.5">{mae == null ? '---' : `${mae.toFixed(2)}%`}</div>
          <div className="text-[10px] text-slate-400 font-mono">Forecast vs actual, every session</div>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <div className="text-[10px] text-slate-400 font-mono uppercase">Hit MAE vs Miss MAE</div>
          <div className="text-lg font-bold text-cyan-400 font-mono mt-0.5">
            {hitMae == null ? '---' : `${hitMae.toFixed(2)}%`} <span className="text-xs text-slate-500">/</span>{' '}
            <span className="text-rose-400">{missMae == null ? '---' : `${missMae.toFixed(2)}%`}</span>
          </div>
          <div className="text-[10px] text-slate-400 font-mono">Calibration & safety</div>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <div className="text-[10px] text-slate-400 font-mono uppercase">★ High Conviction Hit Rate</div>
          <div className="text-lg font-bold text-amber-400 font-mono mt-0.5">{highRows.length ? `${((100 * highHits) / highRows.length).toFixed(1)}%` : '---'}</div>
          <div className="text-[10px] text-slate-400 font-mono">{highHits} hits / {highRows.length} sessions</div>
        </div>
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
          <div className="text-[10px] text-slate-400 font-mono uppercase">Big Moves (≥ 2%)</div>
          <div className="text-lg font-bold text-amber-400 font-mono mt-0.5">{big.length ? `${((100 * bigHits) / big.length).toFixed(1)}%` : '---'}</div>
          <div className="text-[10px] text-slate-400 font-mono">{bigHits} hits / {big.length} sessions{gain != null ? ` · ${formatPercent(gain)} avg after costs` : ''}</div>
        </div>
      </div>

      {/* Walk-forward performance chart */}
      {(displayMode === 'chart' || displayMode === 'both') && (
        <div className="glass-panel p-4 rounded-xl border border-slate-800 bg-slate-900/90 shadow-xl space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-2.5">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="text-sm font-bold text-white tracking-wide">Walk-Forward Performance: {championLabel} vs Realized Market Reality</span>
              <span className="text-xs text-slate-500 font-mono">({chrono.length} Sessions)</span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center font-mono text-[9px] gap-1.5">
                <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                  Hits: <span className="text-emerald-400 font-bold">{hits}✓</span> / <span className="text-rose-400 font-bold">{trades.length - hits}✗</span>
                  {hitRate != null && ` (${hitRate.toFixed(1)}%)`}
                </span>
                <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                  Last 10: <span className="text-emerald-400 font-bold">{l10Hits}✓</span> / <span className="text-rose-400 font-bold">{last10.length - l10Hits}✗</span>
                </span>
                <span className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-semibold" title="Trades where |actual move| >= 2.0%">
                  ⚡ Big Moves: <span className="text-amber-400 font-bold">{bigHits}/{big.length}{big.length ? ` (${((100 * bigHits) / big.length).toFixed(0)}%)` : ''}</span>
                </span>
              </div>
              <div className="flex items-center bg-slate-950 rounded border border-slate-800 p-0.5 font-mono text-xs">
                <button onClick={() => setChartView('price')} className={`px-2 py-0.5 rounded text-[10px] font-bold ${chartView === 'price' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400 hover:text-white'}`}>Price Series (₺)</button>
                <button onClick={() => setChartView('return')} className={`px-2 py-0.5 rounded text-[10px] font-bold ${chartView === 'return' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400 hover:text-white'}`}>Return % Divergence</button>
              </div>
            </div>
          </div>

          <div className="relative bg-slate-950/80 rounded-lg p-3 border border-slate-800/80 overflow-hidden">
            {chrono.length === 0 ? (
              <div className="h-56 flex items-center justify-center text-slate-400 font-mono text-xs">
                {isLoading ? 'Loading walk-forward data...' : 'No graded sessions for this selection yet.'}
              </div>
            ) : (
              <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-56 select-none" onMouseLeave={() => setHovered(null)}>
                {chartView === 'price' ? (
                  [0, 0.25, 0.5, 0.75, 1].map((p, i) => {
                    const y = paddingY + innerH * p;
                    const val = maxPrice - p * (maxPrice - minPrice);
                    return (
                      <g key={i}>
                        <line x1={paddingX} y1={y} x2={chartWidth - paddingX} y2={y} stroke="#334155" strokeWidth="0.5" strokeDasharray="3 3" />
                        <text x={paddingX - 6} y={y + 3} textAnchor="end" fill="#64748b" fontSize="8" fontFamily="monospace">₺{val.toFixed(1)}</text>
                      </g>
                    );
                  })
                ) : (
                  <>
                    <line x1={paddingX} y1={paddingY + innerH / 2} x2={chartWidth - paddingX} y2={paddingY + innerH / 2} stroke="#64748b" strokeWidth="1" />
                    <text x={paddingX - 6} y={paddingY + innerH / 2 + 3} textAnchor="end" fill="#94a3b8" fontSize="8" fontFamily="monospace">0.0%</text>
                    {[0.2, 0.8].map((p, i) => {
                      const y = paddingY + innerH * p;
                      const val = (0.5 - p) * 2 * maxAbs;
                      return (
                        <g key={i}>
                          <line x1={paddingX} y1={y} x2={chartWidth - paddingX} y2={y} stroke="#334155" strokeWidth="0.5" strokeDasharray="3 3" />
                          <text x={paddingX - 6} y={y + 3} textAnchor="end" fill="#64748b" fontSize="8" fontFamily="monospace">{val > 0 ? '+' : ''}{val.toFixed(1)}%</text>
                        </g>
                      );
                    })}
                  </>
                )}
                <path d={path(yAct)} fill="none" stroke="#38bdf8" strokeWidth={chartView === 'price' ? 2.5 : 2} strokeLinecap="round" strokeLinejoin="round" />
                <path d={path(yPred)} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4 3" strokeLinecap="round" strokeLinejoin="round" />
                {chrono.map((r, i) => {
                  const x = getX(i), ya = yAct(r), yp = yPred(r);
                  const isHov = hovered === r.session_date;
                  const baseR = chrono.length > 60 ? 2.2 : 3.5;
                  const dotR = isHov ? baseR + 2.0 : baseR;
                  return (
                    <g key={`${r.session_date}-${i}`} onMouseEnter={() => setHovered(r.session_date)} className="cursor-pointer">
                      {isHov && (
                        <>
                          <line x1={x} y1={paddingY} x2={x} y2={paddingY + innerH} stroke="#64748b" strokeWidth="1" strokeDasharray="2 2" />
                          <line x1={x} y1={ya} x2={x} y2={yp} stroke={dotColor(r)} strokeWidth="2" strokeDasharray="2 2" />
                        </>
                      )}
                      <circle cx={x} cy={ya} r={dotR} fill="#38bdf8" stroke="#0f172a" strokeWidth="1.2" />
                      <circle cx={x} cy={yp} r={dotR} fill={dotColor(r)} stroke="#f59e0b" strokeWidth={r.call === 0 ? 0.8 : 1.8} />
                      {isHov && <circle cx={x} cy={yp} r={dotR + 2.5} fill="none" stroke={dotColor(r)} strokeWidth="1.5" opacity="0.85" />}
                      {r.source === 'LIVE' && <circle cx={x} cy={paddingY - 6} r={1.8} fill="#34d399" />}
                      {(i % dateInterval === 0 || i === chrono.length - 1) && (
                        <text x={x} y={chartHeight - 6} textAnchor="middle" fill="#64748b" fontSize="8" fontFamily="monospace">{r.session_date.slice(5)}</text>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}

            {hoveredItem && (
              <div className="absolute top-2 right-2 bg-slate-900/95 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs font-mono z-20 pointer-events-none min-w-[230px]">
                <div className="flex items-center justify-between border-b border-slate-800 pb-1 mb-1.5">
                  <span className="text-slate-300 font-bold flex items-center gap-1"><Calendar className="w-3 h-3 text-cyan-400" />{hoveredItem.session_date}</span>
                  <span className={`px-1.5 rounded text-[8px] font-black border ${hoveredItem.outcome === 'HIT' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' : hoveredItem.outcome === 'MISS' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' : 'bg-slate-700/40 text-slate-300 border-slate-600/40'}`}>
                    {hoveredItem.outcome === 'HIT' ? '✓ HIT' : hoveredItem.outcome === 'MISS' ? '✗ MISS' : 'FLAT'}{hoveredItem.conviction === 'HIGH' ? ' ★' : ''}
                  </span>
                </div>
                <div className="space-y-1 text-[11px]">
                  <div className="flex justify-between"><span className="text-slate-400">Morning move:</span><span className="text-slate-200">{formatPercent(hoveredItem.morning_move_pct ?? 0)}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Call at 11:30:</span><span className={hoveredItem.call > 0 ? 'text-emerald-400 font-bold' : hoveredItem.call < 0 ? 'text-rose-400 font-bold' : 'text-slate-400'}>{hoveredItem.call > 0 ? 'UP' : hoveredItem.call < 0 ? 'DOWN' : 'FLAT'} @ ₺{hoveredItem.price_1130?.toFixed(2)} · {hoveredItem.conviction === 'HIGH' ? 'HIGH' : 'normal'}</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Realized Close:</span><span className="text-sky-300 font-bold">₺{hoveredItem.close_price?.toFixed(2)} ({formatPercent(hoveredItem.actual_pct ?? 0)})</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Predicted Target:</span><span className="text-amber-300 font-bold">₺{hoveredItem.pred_close?.toFixed(2)} ({formatPercent(hoveredItem.forecast_pct ?? 0)})</span></div>
                  <div className="flex justify-between pt-1 border-t border-slate-800 text-[10px]"><span className="text-slate-400">Absolute Error:</span><span className="text-slate-200">{hoveredItem.abs_error_pp?.toFixed(2)}%</span></div>
                  <div className="flex justify-between text-[9.5px]"><span className="text-slate-500">Model · source:</span><span className="text-slate-400 truncate max-w-[140px]">{hoveredItem.candidate} · {hoveredItem.source === 'LIVE' ? 'live' : 'sim'}</span></div>
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-1 px-1 font-mono text-[10px] text-slate-400">
            <div className="flex flex-wrap items-center gap-4">
              <span className="flex items-center space-x-1.5"><span className="w-3 h-0.5 bg-sky-400 inline-block rounded" /><span className="text-sky-300 font-semibold">Realized Reality ({chartView === 'price' ? '₺' : '%'})</span></span>
              <span className="flex items-center space-x-1.5"><span className="w-3 h-0.5 bg-amber-400 border-b border-dashed border-amber-400 inline-block" /><span className="text-amber-300 font-semibold">{championLabel} Predicted</span></span>
              <span className="flex items-center space-x-1"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /><span className="text-emerald-400">Hit (✓)</span></span>
              <span className="flex items-center space-x-1"><span className="w-2 h-2 rounded-full bg-rose-500 inline-block" /><span className="text-rose-400">Miss (✗)</span></span>
              
              <span className="flex items-center space-x-1"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" /><span className="text-emerald-300">Live session</span></span>
            </div>
            <div className="text-slate-500 text-[9.5px]">Hover over data points to inspect individual session call and error spread</div>
          </div>
        </div>
      )}

      {/* Audit ledger table */}
      {(displayMode === 'ledger' || displayMode === 'both') && (
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-xs font-mono text-left">
            <thead className="bg-slate-900/90 text-slate-400 text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Session</th>
                <th className="py-2.5 px-3 text-center">Source</th>
                <th className="py-2.5 px-3 text-right">Morning Move</th>
                <th className="py-2.5 px-3 text-center">Direction</th>
                <th className="py-2.5 px-3 text-center">Conviction</th>
                <th className="py-2.5 px-3 text-right">11:30 Price</th>
                <th className="py-2.5 px-3 text-right">Target Close</th>
                <th className="py-2.5 px-3 text-right">Realized Close</th>
                <th className="py-2.5 px-3 text-right">Pred Return</th>
                <th className="py-2.5 px-3 text-right">Actual Return</th>
                <th className="py-2.5 px-3 text-right">Error %</th>
                <th className="py-2.5 px-3 text-center">Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
              {isLoading ? (
                <tr><td colSpan={12} className="py-8 text-center text-slate-400 font-mono">Loading audited sessions...</td></tr>
              ) : ledger.length === 0 ? (
                <tr><td colSpan={12} className="py-8 text-center text-slate-400 font-mono">No audited sessions for this selection yet.</td></tr>
              ) : (
                ledger.map((r, idx) => {
                  const isHov = hovered === r.session_date;
                  const pred = r.forecast_pct ?? 0;
                  return (
                    <tr key={`${r.session_date}-${idx}`} onMouseEnter={() => setHovered(r.session_date)} onMouseLeave={() => setHovered(null)}
                      className={`transition-colors ${isHov ? 'bg-slate-800/60 ring-1 ring-cyan-500/30' : 'hover:bg-slate-900/40'}`}>
                      <td className="py-2 px-3 text-white font-medium">{r.session_date}</td>
                      <td className="py-2 px-3 text-center">
                        <span className={`text-[8px] font-bold px-1 rounded border ${r.source === 'LIVE' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-slate-700/30 text-slate-400 border-slate-600/40'}`}>{r.source === 'LIVE' ? 'LIVE' : 'SIM'}</span>
                      </td>
                      <td className={`py-2 px-3 text-right ${(r.morning_move_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{formatPercent(r.morning_move_pct ?? 0)}</td>
                      <td className="py-2 px-3 text-center font-bold">{r.call > 0 ? <span className="text-emerald-400">UP</span> : r.call < 0 ? <span className="text-rose-400">DOWN</span> : <span className="text-slate-500 font-normal">FLAT</span>}</td>
                      <td className="py-2 px-3 text-center">{r.conviction === 'HIGH' ? <span className="text-amber-300 font-bold">★ HIGH</span> : <span className="text-slate-500">normal</span>}</td>
                      <td className="py-2 px-3 text-right text-slate-300">₺{r.price_1130?.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right text-cyan-300 font-bold">₺{r.pred_close?.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right text-white font-bold">₺{r.close_price?.toFixed(2)}</td>
                      <td className={`py-2 px-3 text-right font-bold ${pred > 0 ? 'text-emerald-400' : pred < 0 ? 'text-rose-400' : 'text-slate-500'}`}>{formatPercent(pred)}</td>
                      <td className={`py-2 px-3 text-right font-bold ${(r.actual_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{formatPercent(r.actual_pct ?? 0)}</td>
                      <td className="py-2 px-3 text-right text-slate-300">{r.abs_error_pp?.toFixed(2)}%</td>
                      <td className="py-2 px-3 text-center">
                        {r.outcome === 'HIT' ? (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"><CheckCircle2 className="w-3 h-3" /> HIT</span>
                        ) : r.outcome === 'MISS' ? (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30"><XCircle className="w-3 h-3" /> MISS</span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-slate-700/30 text-slate-400 border border-slate-600/40"><MinusCircle className="w-3 h-3" /> FLAT</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
