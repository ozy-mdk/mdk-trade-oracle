import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchPredictedOpportunities } from '../api/client';
import { OpportunityItem } from '../types/api';
import { formatTL, formatPercent } from '../utils/formatters';
import {
  Compass,
  TrendingUp,
  TrendingDown,
  Sparkles,
  LineChart,
  Search,
  RefreshCw,
  Flame,
  Target,
  Activity,
  Layers,
  Award,
  BarChart3,
  Calendar,
  ShieldCheck,
  ChevronRight,
  ArrowUpRight,
  ArrowDownRight,
  Info,
} from 'lucide-react';

interface OpportunityActionsDashboardProps {
  onSelectSymbol: (symbol: string) => void;
  onNavigateTab?: (tab: 'candles' | 'tertip' | 'opportunities' | 'multihorizon') => void;
  currentTradeDate?: string;
}

const HORIZONS = [
  { key: '1d', label: '1D', desc: 'Tactical T+1' },
  { key: '3d', label: '3D', desc: '3-Day Swing' },
  { key: '5d', label: '5D', desc: '5-Day Swing' },
  { key: '10d', label: '10D', desc: '10-Day Trend' },
  { key: '15d', label: '15D', desc: '15-Day Trend' },
  { key: '30d', label: '30D', desc: 'Macro Swing' },
] as const;

const THRESHOLD_OPTIONS = [
  { val: 0, label: 'All Stocks' },
  { val: 65, label: '≥65% Win' },
  { val: 70, label: '★ ≥70% Win (Def)' },
  { val: 75, label: '≥75% Win' },
  { val: 80, label: '≥80% Win' },
] as const;

type FilterCategory = 'all' | 'high_rel' | 'big_moves' | 'sig_moves' | 'longs' | 'shorts' | 'bofa_buy' | 'bofa_sell';
type SortField = 'return_desc' | 'return_asc' | 'magnitude' | 'hit_rate_30d' | 'hit_rate_60d' | 'bofa_flow';

export const OpportunityActionsDashboard: React.FC<OpportunityActionsDashboardProps> = ({
  onSelectSymbol,
  onNavigateTab,
  currentTradeDate,
}) => {
  const [activeHorizon, setActiveHorizon] = useState<string>('1d');
  const [reliabilityThreshold, setReliabilityThreshold] = useState<number>(70);
  const [onlyBigMoves, setOnlyBigMoves] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [activeFilter, setActiveFilter] = useState<FilterCategory>('all');
  const [sortField, setSortField] = useState<SortField>('return_desc');
  const [selectedSector, setSelectedSector] = useState<string>('all');

  // Query opportunities for active horizon and session date
  const {
    data: opportunityData,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['predicted-opportunities', activeHorizon, currentTradeDate],
    queryFn: () => fetchPredictedOpportunities(activeHorizon, currentTradeDate),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  const opportunities: OpportunityItem[] = useMemo(() => {
    return opportunityData?.opportunities || [];
  }, [opportunityData]);

  // Extract unique sectors
  const availableSectors = useMemo(() => {
    const set = new Set<string>();
    opportunities.forEach((op) => {
      if (op.sector) set.add(op.sector);
    });
    return Array.from(set).sort();
  }, [opportunities]);

  // High-reliability filtered candidates for the top selective showcase
  const highlyReliableOpportunities = useMemo(() => {
    return opportunities.filter((op) => {
      const hitRate = op.hit_rate_30d_pct ?? op.champion_dir_hit_rate_pct ?? 0;
      const meetsThreshold = reliabilityThreshold === 0 || hitRate >= reliabilityThreshold;
      const meetsMagnitude = !onlyBigMoves || Math.abs(op.expected_return_pct) >= 2.0;
      return meetsThreshold && meetsMagnitude;
    });
  }, [opportunities, reliabilityThreshold, onlyBigMoves]);

  // Top hit rate constituents as runners-up when 0 meet strict threshold
  const topHitRateStocks = useMemo(() => {
    return [...opportunities]
      .sort((a, b) => (b.hit_rate_30d_pct ?? 0) - (a.hit_rate_30d_pct ?? 0))
      .slice(0, 5);
  }, [opportunities]);

  // Filtered and sorted opportunity list for the all-stocks matrix table
  const filteredOpportunities = useMemo(() => {
    return opportunities
      .filter((op) => {
        // Sector filter
        if (selectedSector !== 'all' && op.sector !== selectedSector) return false;

        // Search filter
        if (searchTerm) {
          const q = searchTerm.toLowerCase();
          const matchSym = op.symbol.toLowerCase().includes(q);
          const matchName = op.company_name?.toLowerCase().includes(q);
          const matchSec = op.sector?.toLowerCase().includes(q);
          const matchPlay = op.playbook?.toLowerCase().includes(q);
          if (!matchSym && !matchName && !matchSec && !matchPlay) return false;
        }

        // Category filter
        if (activeFilter === 'high_rel') {
          const hitRate = op.hit_rate_30d_pct ?? op.champion_dir_hit_rate_pct ?? 0;
          return hitRate >= (reliabilityThreshold || 70) && (!onlyBigMoves || Math.abs(op.expected_return_pct) >= 2.0);
        }
        if (activeFilter === 'big_moves') return Math.abs(op.expected_return_pct) >= 2.0;
        if (activeFilter === 'sig_moves') return Math.abs(op.expected_return_pct) >= 1.0;
        if (activeFilter === 'longs') return op.expected_return_pct > 0.25;
        if (activeFilter === 'shorts') return op.expected_return_pct < -0.25;
        if (activeFilter === 'bofa_buy') return op.mlb_net_flow_tl > 0;
        if (activeFilter === 'bofa_sell') return op.mlb_net_flow_tl < 0;

        return true;
      })
      .sort((a, b) => {
        switch (sortField) {
          case 'return_desc':
            return b.expected_return_pct - a.expected_return_pct;
          case 'return_asc':
            return a.expected_return_pct - b.expected_return_pct;
          case 'magnitude':
            return Math.abs(b.expected_return_pct) - Math.abs(a.expected_return_pct);
          case 'hit_rate_30d':
            return (b.hit_rate_30d_pct ?? b.champion_dir_hit_rate_pct ?? 0) - (a.hit_rate_30d_pct ?? a.champion_dir_hit_rate_pct ?? 0);
          case 'hit_rate_60d':
            return (b.hit_rate_60d_pct ?? 0) - (a.hit_rate_60d_pct ?? 0);
          case 'bofa_flow':
            return b.mlb_net_flow_tl - a.mlb_net_flow_tl;
          default:
            return b.expected_return_pct - a.expected_return_pct;
        }
      });
  }, [opportunities, selectedSector, searchTerm, activeFilter, sortField, reliabilityThreshold, onlyBigMoves]);

  const handleLaunchSymbol = (sym: string, targetTab?: 'candles' | 'multihorizon') => {
    onSelectSymbol(sym);
    if (targetTab && onNavigateTab) {
      onNavigateTab(targetTab);
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[450px] space-y-4">
        <div className="p-4 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 animate-pulse">
          <Compass className="w-10 h-10 animate-spin" />
        </div>
        <div className="text-center">
          <div className="text-sm font-semibold text-white">Scanning Constituent Opportunities...</div>
          <div className="text-xs text-slate-400 mt-1">Evaluating multi-horizon ML models & institutional order flow for {activeHorizon.toUpperCase()}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      {/* ── TOP HERO HEADER ──────────────────────────────────────────────────────── */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800/80 bg-gradient-to-r from-slate-900/90 via-slate-900/70 to-slate-950/90 shadow-xl relative overflow-hidden">
        <div className="absolute -top-24 -left-24 w-72 h-72 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -right-24 w-72 h-72 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 relative z-10">
          <div>
            <div className="flex items-center space-x-3 mb-1.5">
              <div className="p-2.5 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 shadow-sm shadow-cyan-500/20">
                <Compass className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h1 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
                    Predicted Opportunity Actions & Radar
                    <span className="text-[10px] uppercase font-mono font-semibold px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                      {activeHorizon.toUpperCase()} Specialist Radar
                    </span>
                  </h1>
                </div>
                <p className="text-xs text-slate-400">
                  Institutional Flow Confluence • Verified 30D/60D Model Precision • Bidirectional Opportunity Radar
                </p>
              </div>
            </div>
          </div>

          {/* Quick Session Date & Refresh Action */}
          <div className="flex items-center space-x-3">
            <div className="text-right font-mono">
              <div className="text-[11px] text-slate-400 flex items-center justify-end space-x-1">
                <Calendar className="w-3 h-3 text-cyan-400" />
                <span>As of Date: <strong className="text-slate-200">{opportunityData?.as_of_date || 'Latest'}</strong></span>
              </div>
              <div className="text-[11px] text-emerald-400 font-semibold">
                Horizon Target: {activeHorizon.toUpperCase()} ({activeHorizon === '1d' ? 'Next-Day' : `${activeHorizon.replace('d', '')} Market Sessions`})
              </div>
            </div>

            <button
              onClick={() => refetch()}
              disabled={isFetching}
              className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-300 hover:text-white transition-all shadow-sm"
              title="Refresh Forecast Opportunities"
            >
              <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>
        </div>

        {/* ── HORIZON MODEL SELECTOR RIBBON (1D, 3D, 5D, 10D, 15D, 30D) ──────────── */}
        <div className="mt-4 pt-3.5 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2.5">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-mono font-bold text-white uppercase tracking-wider">
              Prediction Horizon Model:
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
            {HORIZONS.map((h) => {
              const isSelected = activeHorizon === h.key;
              return (
                <button
                  key={h.key}
                  onClick={() => setActiveHorizon(h.key)}
                  className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all ${
                    isSelected
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm shadow-cyan-500/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
                  }`}
                  title={`Inspect ${h.label} (${h.desc}) opportunities`}
                >
                  <span>{h.label}</span>
                  <span className="text-[10px] text-slate-500 font-normal hidden sm:inline">
                    ({h.desc})
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Market Opportunity Breadth Gauge */}
        {opportunityData && (
          <div className="mt-4 pt-3 border-t border-slate-800/60 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
            <div className="bg-slate-950/60 border border-slate-800/70 p-2.5 rounded-xl">
              <div className="text-[9.5px] font-mono text-slate-400 uppercase tracking-wider mb-0.5">BIST 30 Tracked</div>
              <div className="text-base font-bold font-mono text-white flex items-center space-x-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
                <span>{opportunityData.total_constituents} Equities</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-emerald-900/30 p-2.5 rounded-xl">
              <div className="text-[9.5px] font-mono text-emerald-400 uppercase tracking-wider mb-0.5">Bullish Bias</div>
              <div className="text-base font-bold font-mono text-emerald-300 flex items-center space-x-1.5">
                <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
                <span>{opportunityData.bullish_count} ({opportunityData.total_constituents > 0 ? ((opportunityData.bullish_count / opportunityData.total_constituents) * 100).toFixed(0) : 0}%)</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-rose-900/30 p-2.5 rounded-xl">
              <div className="text-[9.5px] font-mono text-rose-400 uppercase tracking-wider mb-0.5">Bearish Bias</div>
              <div className="text-base font-bold font-mono text-rose-300 flex items-center space-x-1.5">
                <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
                <span>{opportunityData.bearish_count} ({opportunityData.total_constituents > 0 ? ((opportunityData.bearish_count / opportunityData.total_constituents) * 100).toFixed(0) : 0}%)</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-amber-900/30 p-2.5 rounded-xl">
              <div className="text-[9.5px] font-mono text-amber-400 uppercase tracking-wider mb-0.5">Big Moves (|Δ| ≥ 2.0%)</div>
              <div className="text-base font-bold font-mono text-amber-300 flex items-center space-x-1.5">
                <Flame className="w-3.5 h-3.5 text-amber-400" />
                <span>{opportunityData.high_conviction_count} Setups</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-indigo-900/30 p-2.5 rounded-xl">
              <div className="text-[9.5px] font-mono text-indigo-400 uppercase tracking-wider mb-0.5">Sig Moves (|Δ| ≥ 1.0%)</div>
              <div className="text-base font-bold font-mono text-indigo-300 flex items-center space-x-1.5">
                <Target className="w-3.5 h-3.5 text-indigo-400" />
                <span>{opportunityData.significant_moves_count} Setups</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-slate-800/70 p-2.5 rounded-xl">
              <div className="text-[9.5px] font-mono text-slate-400 uppercase tracking-wider mb-0.5">Market Mean Δ</div>
              <div className={`text-base font-bold font-mono flex items-center space-x-1.5 ${opportunityData.avg_expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                <Activity className="w-3.5 h-3.5" />
                <span>{formatPercent(opportunityData.avg_expected_return_pct)}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── SELECTIVE HIGH-RELIABILITY BAR & RADAR (LAST 30 DAYS WIN RATE) ────────── */}
      <div className="glass-card p-4 rounded-2xl border border-amber-500/30 bg-gradient-to-r from-amber-950/15 via-slate-900/90 to-slate-950/90 shadow-xl space-y-3.5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-amber-500/20 pb-3">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-xl bg-amber-500/20 border border-amber-500/40 text-amber-300 shadow-sm shadow-amber-500/20">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-sm font-bold text-white tracking-wide uppercase font-mono flex items-center gap-2">
                  <span>⚡ Selective High-Reliability Opportunities</span>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    {activeHorizon.toUpperCase()} Model Precision
                  </span>
                </h2>
              </div>
              <p className="text-xs text-slate-400">
                Filtered strictly by trailing 30-day directional win rate threshold & high return magnitude
              </p>
            </div>
          </div>

          {/* Selectable Win Rate Threshold Pills (65%, 70%, 75%, 80%) & Big Moves Check */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center space-x-1 bg-slate-950/90 p-1 rounded-xl border border-slate-800 font-mono text-xs">
              <span className="text-[10px] text-slate-400 uppercase px-1.5 font-bold">
                Win Rate Cutoff:
              </span>
              {THRESHOLD_OPTIONS.map((opt) => {
                const isSelected = reliabilityThreshold === opt.val;
                return (
                  <button
                    key={opt.val}
                    onClick={() => setReliabilityThreshold(opt.val)}
                    className={`px-2.5 py-0.5 rounded-lg text-xs font-bold transition-all ${
                      isSelected
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>

            {/* Toggle: Only |Δ| ≥ 2% Moves */}
            <button
              onClick={() => setOnlyBigMoves(!onlyBigMoves)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-xl text-xs font-mono font-bold transition-all border ${
                onlyBigMoves
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                  : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-white'
              }`}
              title="Toggle requirement for |expected return| >= 2.0%"
            >
              <Flame className="w-3.5 h-3.5" />
              <span>{onlyBigMoves ? '|Δ| ≥ 2.0% Required' : 'All Magnitudes'}</span>
            </button>
          </div>
        </div>

        {/* High-Reliability Cards or Informative Empty State */}
        {highlyReliableOpportunities.length === 0 ? (
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-amber-500/20 text-xs font-mono text-slate-300 space-y-1.5">
            <div className="flex items-center space-x-2 text-amber-300 font-bold">
              <Info className="w-4 h-4 text-amber-400 shrink-0" />
              <span>
                0 constituents currently meet the strict ≥{reliabilityThreshold}% win rate {onlyBigMoves ? '+ ≥2.0% move ' : ''}cutoff for {activeHorizon.toUpperCase()}.
              </span>
            </div>
            <div className="text-[11px] text-slate-400 flex flex-wrap items-center gap-2 pt-1">
              <span>Top performing models available in {activeHorizon.toUpperCase()}:</span>
              {topHitRateStocks.map((s) => (
                <button
                  key={s.symbol}
                  onClick={() => handleLaunchSymbol(s.symbol, 'multihorizon')}
                  className="px-2 py-0.5 rounded bg-slate-900 border border-slate-700 hover:border-cyan-500/50 text-slate-200 hover:text-cyan-300 transition-colors"
                >
                  <strong className="text-white">{s.symbol}</strong>: {s.hit_rate_30d_pct ?? 0}% ({s.hits_30d ?? 0}/{s.total_30d ?? 30}) • {s.expected_return_pct >= 0 ? '+' : ''}{s.expected_return_pct.toFixed(1)}%
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {highlyReliableOpportunities.map((op) => {
              const isBullish = op.expected_return_pct >= 0.25;
              const isBearish = op.expected_return_pct <= -0.25;
              const hitRate30 = op.hit_rate_30d_pct ?? op.champion_dir_hit_rate_pct ?? 0;
              const hitRate60 = op.hit_rate_60d_pct ?? 0;

              return (
                <div
                  key={op.symbol}
                  className={`p-3.5 rounded-xl border transition-all relative overflow-hidden group ${
                    isBullish
                      ? 'border-emerald-500/40 bg-emerald-950/20 hover:border-emerald-500/70'
                      : isBearish
                      ? 'border-rose-500/40 bg-rose-950/20 hover:border-rose-500/70'
                      : 'border-slate-800 bg-slate-900/60'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono font-bold text-base text-white group-hover:text-cyan-300 transition-colors">
                        {op.symbol}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-1.5 py-0.2 rounded border border-slate-800">
                        {op.sector}
                      </span>
                    </div>

                    {/* 30D Win Rate Badge */}
                    <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40">
                      ★ 30D: {hitRate30.toFixed(1)}% {op.hits_30d != null ? `(${op.hits_30d}/${op.total_30d})` : ''}
                    </span>
                  </div>

                  <div className="text-[11px] text-slate-400 truncate mb-2">
                    {op.company_name}
                  </div>

                  {/* Return % & Direction Gauge */}
                  <div className="flex items-baseline justify-between my-2 font-mono">
                    <div className="flex items-baseline space-x-1">
                      <span
                        className={`text-xl font-black ${
                          isBullish ? 'text-emerald-400' : isBearish ? 'text-rose-400' : 'text-slate-300'
                        }`}
                      >
                        {formatPercent(op.expected_return_pct)}
                      </span>
                      {isBullish ? (
                        <TrendingUp className="w-4 h-4 text-emerald-400 inline" />
                      ) : isBearish ? (
                        <TrendingDown className="w-4 h-4 text-rose-400 inline" />
                      ) : null}
                    </div>

                    <div className="text-right text-[10px] text-slate-400">
                      Target: <strong className="text-white">₺{op.target_price.toFixed(2)}</strong>
                    </div>
                  </div>

                  {/* Model & 60D Win Rate Info Row */}
                  <div className="pt-2 border-t border-slate-800/60 text-[10.5px] font-mono space-y-1">
                    <div className="flex justify-between items-center text-slate-400">
                      <span className="flex items-center gap-1">
                        <Award className="w-3 h-3 text-amber-400" /> Model:
                      </span>
                      <span className="text-amber-300 font-semibold truncate max-w-[120px]">
                        {op.ml_champion_type || 'Algorithm'}
                      </span>
                    </div>

                    {hitRate60 > 0 && (
                      <div className="flex justify-between items-center text-slate-400">
                        <span>60D Win Rate:</span>
                        <span className="text-slate-200">
                          {hitRate60.toFixed(1)}% {op.hits_60d != null ? `(${op.hits_60d}/${op.total_60d})` : ''}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Launch Actions */}
                  <div className="mt-2.5 pt-2 border-t border-slate-800/40 flex items-center justify-between">
                    <span className="text-[9px] font-mono text-slate-400 truncate max-w-[120px]" title={op.playbook}>
                      {op.playbook?.replace(/_/g, ' ') || 'Tactical'}
                    </span>

                    <button
                      onClick={() => handleLaunchSymbol(op.symbol, 'multihorizon')}
                      className="px-2 py-1 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-300 text-[10px] font-mono font-bold transition-all flex items-center gap-1"
                      title="Inspect in Multi-Horizon Hub"
                    >
                      <span>Hub</span>
                      <ChevronRight className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── HIGHEST POSITIVE & NEGATIVE OPPORTUNITIES SHOWCASE ─────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Top Bullish Opportunities (Highest Positive Expected Return) */}
        <div className="glass-card p-5 rounded-2xl border border-emerald-900/50 bg-gradient-to-br from-emerald-950/20 via-slate-900/80 to-slate-950/90 shadow-xl flex flex-col">
          <div className="flex items-center justify-between pb-3.5 mb-4 border-b border-emerald-900/40">
            <div className="flex items-center space-x-2.5">
              <div className="p-2 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 shadow-sm shadow-emerald-500/20">
                <TrendingUp className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  Top Bullish Opportunities ({activeHorizon.toUpperCase()})
                  <span className="text-[10px] font-mono uppercase bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full border border-emerald-500/30">
                    Highest Expected Upside
                  </span>
                </h2>
                <div className="text-[11px] text-slate-400">Institutional accumulation & squeeze setups ranked by forecast magnitude</div>
              </div>
            </div>
          </div>

          <div className="space-y-3 flex-1">
            {opportunityData?.top_longs?.slice(0, 4).map((op, idx) => (
              <div
                key={op.symbol}
                className="group p-3 rounded-xl bg-slate-900/80 hover:bg-slate-850 border border-emerald-500/30 hover:border-emerald-400/60 transition-all shadow-md hover:shadow-emerald-500/10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 relative overflow-hidden"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-7 h-7 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center font-mono font-bold text-emerald-400 text-xs">
                    #{idx + 1}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-base font-bold text-white font-mono group-hover:text-cyan-300 transition-colors">
                        {op.symbol}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded border border-slate-700/60">
                        {op.sector}
                      </span>
                      <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                        {op.ml_champion_type}
                      </span>
                      {(op.hit_rate_30d_pct != null || op.champion_dir_hit_rate_pct != null) && (
                        <span className="text-[9.5px] font-mono font-bold px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          ★ {(op.hit_rate_30d_pct ?? op.champion_dir_hit_rate_pct)?.toFixed(1)}% (30D)
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 truncate max-w-[240px]">
                      {op.company_name}
                    </div>
                  </div>
                </div>

                {/* Return & Target Stats */}
                <div className="flex items-center space-x-4 w-full sm:w-auto justify-between sm:justify-end">
                  <div className="text-right">
                    <div className="text-lg font-bold font-mono text-emerald-400 flex items-center justify-end space-x-1">
                      <span>+{op.expected_return_pct.toFixed(2)}%</span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-400">
                      {op.current_price.toFixed(2)} → <strong className="text-emerald-300">{op.target_price.toFixed(2)} TL</strong>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-1.5">
                    <button
                      onClick={() => handleLaunchSymbol(op.symbol, 'candles')}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-cyan-950/60 border border-slate-700 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 text-xs font-semibold transition-all flex items-center space-x-1"
                      title="Open in Candlestick & Flow"
                    >
                      <LineChart className="w-3.5 h-3.5" />
                      <span className="hidden md:inline">Chart</span>
                    </button>
                    <button
                      onClick={() => handleLaunchSymbol(op.symbol, 'multihorizon')}
                      className="px-2.5 py-1.5 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 text-xs font-semibold transition-all flex items-center space-x-1"
                      title="Open in Multi-Horizon Hub"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      <span className="hidden md:inline">Hub</span>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Bearish Opportunities (Highest Negative Expected Return) */}
        <div className="glass-card p-5 rounded-2xl border border-rose-900/50 bg-gradient-to-br from-rose-950/20 via-slate-900/80 to-slate-950/90 shadow-xl flex flex-col">
          <div className="flex items-center justify-between pb-3.5 mb-4 border-b border-rose-900/40">
            <div className="flex items-center space-x-2.5">
              <div className="p-2 rounded-lg bg-rose-500/20 border border-rose-500/40 text-rose-400 shadow-sm shadow-rose-500/20">
                <TrendingDown className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  Top Bearish Opportunities ({activeHorizon.toUpperCase()})
                  <span className="text-[10px] font-mono uppercase bg-rose-500/20 text-rose-300 px-2 py-0.5 rounded-full border border-rose-500/30">
                    Highest Expected Downside
                  </span>
                </h2>
                <div className="text-[11px] text-slate-400">Institutional distribution & tactical short setups ranked by downside magnitude</div>
              </div>
            </div>
          </div>

          <div className="space-y-3 flex-1">
            {opportunityData?.top_shorts?.slice(0, 4).map((op, idx) => (
              <div
                key={op.symbol}
                className="group p-3 rounded-xl bg-slate-900/80 hover:bg-slate-850 border border-rose-500/30 hover:border-rose-400/60 transition-all shadow-md hover:shadow-rose-500/10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 relative overflow-hidden"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-7 h-7 rounded-lg bg-rose-500/20 border border-rose-500/40 flex items-center justify-center font-mono font-bold text-rose-400 text-xs">
                    #{idx + 1}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-base font-bold text-white font-mono group-hover:text-cyan-300 transition-colors">
                        {op.symbol}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded border border-slate-700/60">
                        {op.sector}
                      </span>
                      <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                        {op.ml_champion_type}
                      </span>
                      {(op.hit_rate_30d_pct != null || op.champion_dir_hit_rate_pct != null) && (
                        <span className="text-[9.5px] font-mono font-bold px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          ★ {(op.hit_rate_30d_pct ?? op.champion_dir_hit_rate_pct)?.toFixed(1)}% (30D)
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 truncate max-w-[240px]">
                      {op.company_name}
                    </div>
                  </div>
                </div>

                {/* Return & Target Stats */}
                <div className="flex items-center space-x-4 w-full sm:w-auto justify-between sm:justify-end">
                  <div className="text-right">
                    <div className="text-lg font-bold font-mono text-rose-400 flex items-center justify-end space-x-1">
                      <span>{op.expected_return_pct.toFixed(2)}%</span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-400">
                      {op.current_price.toFixed(2)} → <strong className="text-rose-300">{op.target_price.toFixed(2)} TL</strong>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-1.5">
                    <button
                      onClick={() => handleLaunchSymbol(op.symbol, 'candles')}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-cyan-950/60 border border-slate-700 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 text-xs font-semibold transition-all flex items-center space-x-1"
                      title="Open in Candlestick & Flow"
                    >
                      <LineChart className="w-3.5 h-3.5" />
                      <span className="hidden md:inline">Chart</span>
                    </button>
                    <button
                      onClick={() => handleLaunchSymbol(op.symbol, 'multihorizon')}
                      className="px-2.5 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-300 text-xs font-semibold transition-all flex items-center space-x-1"
                      title="Open in Multi-Horizon Hub"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      <span className="hidden md:inline">Hub</span>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── ALL IMPORTANT STOCKS RADAR & MATRIX TABLE (ALL 30 CONSTITUENTS) ───────── */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800/80 shadow-xl space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
              <BarChart3 className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-wide">
                BIST 30 Opportunity Matrix & Radar ({activeHorizon.toUpperCase()})
              </h2>
              <div className="text-xs text-slate-400">
                Displaying {filteredOpportunities.length} of {opportunities.length} tracked constituent equities with model win rate hints
              </div>
            </div>
          </div>

          {/* Search Box */}
          <div className="relative w-full md:w-64">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Filter by symbol or sector..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/60 font-mono"
            />
          </div>
        </div>

        {/* Filter Pills Deck & Sorters */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <div className="flex flex-wrap items-center gap-1.5">
            {[
              { id: 'all', label: `All Stocks (${opportunities.length})` },
              { id: 'high_rel', label: `⚡ High Reliability (≥${reliabilityThreshold}%)` },
              { id: 'big_moves', label: 'Big Moves (|Δ| ≥ 2%)' },
              { id: 'sig_moves', label: 'Sig Moves (|Δ| ≥ 1%)' },
              { id: 'longs', label: 'Bullish Setups' },
              { id: 'shorts', label: 'Bearish Setups' },
              { id: 'bofa_buy', label: 'BofA Buying' },
              { id: 'bofa_sell', label: 'BofA Selling' },
            ].map((f) => {
              const isSelected = activeFilter === f.id;
              return (
                <button
                  key={f.id}
                  onClick={() => setActiveFilter(f.id as FilterCategory)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-all ${
                    isSelected
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                      : 'bg-slate-900/60 text-slate-400 hover:text-white border border-slate-800 hover:border-slate-700'
                  }`}
                >
                  {f.label}
                </button>
              );
            })}
          </div>

          {/* Sector Selector & Sort Dropdown */}
          <div className="flex items-center space-x-2">
            <select
              value={selectedSector}
              onChange={(e) => setSelectedSector(e.target.value)}
              className="bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500/60 font-mono"
            >
              <option value="all">All Sectors</option>
              {availableSectors.map((sec) => (
                <option key={sec} value={sec}>
                  {sec}
                </option>
              ))}
            </select>

            <select
              value={sortField}
              onChange={(e) => setSortField(e.target.value as SortField)}
              className="bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500/60 font-mono"
            >
              <option value="return_desc">Expected Return: High → Low</option>
              <option value="return_asc">Expected Return: Low → High</option>
              <option value="magnitude">Opportunity Magnitude: Max First</option>
              <option value="hit_rate_30d">30D Hit Rate: Highest First</option>
              <option value="hit_rate_60d">60D Hit Rate: Highest First</option>
              <option value="bofa_flow">BofA Net Flow: Highest First</option>
            </select>
          </div>
        </div>

        {/* The Matrix Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-950/50">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                <th className="py-3 px-3">#</th>
                <th className="py-3 px-3">Ticker & Company</th>
                <th className="py-3 px-3 text-right">Current Price</th>
                <th className="py-3 px-3 text-right">Target Range</th>
                <th className="py-3 px-4 text-center">Forecast Return & Gauge</th>
                <th className="py-3 px-3">Institutional Playbook</th>
                <th className="py-3 px-3 text-center">Model Specialist</th>
                <th className="py-3 px-3 text-right">30D Win Rate</th>
                <th className="py-3 px-3 text-right">60D Win Rate</th>
                <th className="py-3 px-3 text-right">BofA Flow TL</th>
                <th className="py-3 px-3 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredOpportunities.length === 0 ? (
                <tr>
                  <td colSpan={11} className="py-12 text-center text-slate-500 font-sans">
                    No constituent opportunities match the selected criteria for {activeHorizon.toUpperCase()}.
                  </td>
                </tr>
              ) : (
                filteredOpportunities.map((op, idx) => {
                  const isHighConviction = Math.abs(op.expected_return_pct) >= 2.0;
                  const isSigMove = Math.abs(op.expected_return_pct) >= 1.0;
                  const isPositive = op.expected_return_pct > 0.25;
                  const isNegative = op.expected_return_pct < -0.25;
                  const hitRate30 = op.hit_rate_30d_pct ?? op.champion_dir_hit_rate_pct ?? 0;
                  const hitRate60 = op.hit_rate_60d_pct ?? 0;

                  // Row highlight styling
                  const rowBg = isHighConviction
                    ? isPositive
                      ? 'bg-emerald-950/15 hover:bg-emerald-950/25 border-l-2 border-l-emerald-400'
                      : 'bg-rose-950/15 hover:bg-rose-950/25 border-l-2 border-l-rose-400'
                    : isSigMove
                    ? isPositive
                      ? 'hover:bg-emerald-950/10'
                      : 'hover:bg-rose-950/10'
                    : 'hover:bg-slate-900/50';

                  return (
                    <tr key={op.symbol} className={`transition-colors ${rowBg}`}>
                      {/* Rank Index */}
                      <td className="py-3 px-3 text-slate-500 text-[11px]">
                        {idx + 1}
                      </td>

                      {/* Ticker & Name */}
                      <td className="py-3 px-3">
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => handleLaunchSymbol(op.symbol, 'candles')}
                            className="font-bold text-white hover:text-cyan-300 transition-colors text-sm"
                          >
                            {op.symbol}
                          </button>
                          <span className="text-[10px] text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                            {op.sector}
                          </span>
                        </div>
                        <div className="text-[10px] text-slate-400 truncate max-w-[180px] font-sans">
                          {op.company_name}
                        </div>
                      </td>

                      {/* Current Price */}
                      <td className="py-3 px-3 text-right text-slate-300">
                        ₺{op.current_price.toFixed(2)}
                      </td>

                      {/* Target Price */}
                      <td className="py-3 px-3 text-right">
                        <div className={`font-bold ${isPositive ? 'text-emerald-300' : isNegative ? 'text-rose-300' : 'text-slate-300'}`}>
                          ₺{op.target_price.toFixed(2)}
                        </div>
                        <div className="text-[9.5px] text-slate-500">
                          [{op.price_low.toFixed(1)} - {op.price_high.toFixed(1)}]
                        </div>
                      </td>

                      {/* Expected Return & Bidirectional Gauge */}
                      <td className="py-3 px-4 text-center">
                        <div className="flex flex-col items-center justify-center">
                          <span
                            className={`font-black text-sm flex items-center gap-1 ${
                              isPositive ? 'text-emerald-400' : isNegative ? 'text-rose-400' : 'text-slate-400'
                            }`}
                          >
                            {isPositive ? <ArrowUpRight className="w-3.5 h-3.5" /> : isNegative ? <ArrowDownRight className="w-3.5 h-3.5" /> : null}
                            {formatPercent(op.expected_return_pct)}
                          </span>

                          {/* Bidirectional Mini Bar */}
                          <div className="w-24 h-1.5 bg-slate-900 rounded-full overflow-hidden flex my-1 border border-slate-800">
                            <div className="w-1/2 flex justify-end">
                              {isNegative && (
                                <div
                                  className="h-full bg-rose-500 rounded-l"
                                  style={{ width: `${Math.min(100, Math.abs(op.expected_return_pct) * 20)}%` }}
                                />
                              )}
                            </div>
                            <div className="w-[1px] bg-slate-600 h-full" />
                            <div className="w-1/2 flex justify-start">
                              {isPositive && (
                                <div
                                  className="h-full bg-emerald-500 rounded-r"
                                  style={{ width: `${Math.min(100, Math.abs(op.expected_return_pct) * 20)}%` }}
                                />
                              )}
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* Playbook */}
                      <td className="py-3 px-3">
                        <span className="text-[11px] text-slate-300 truncate max-w-[150px] block" title={op.playbook}>
                          {op.playbook?.replace(/_/g, ' ') || 'Range'}
                        </span>
                      </td>

                      {/* Model Specialist */}
                      <td className="py-3 px-3 text-center">
                        <span className="text-[11px] font-semibold text-amber-300">
                          {op.ml_champion_type || 'Auto'}
                        </span>
                      </td>

                      {/* 30D Hit Rate Hint */}
                      <td className="py-3 px-3 text-right">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                            hitRate30 >= 70
                              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                              : hitRate30 >= 60
                              ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                              : 'bg-slate-900 text-slate-400 border-slate-800'
                          }`}
                        >
                          {hitRate30.toFixed(1)}% {op.hits_30d != null ? `(${op.hits_30d}/${op.total_30d})` : ''}
                        </span>
                      </td>

                      {/* 60D Hit Rate Hint */}
                      <td className="py-3 px-3 text-right">
                        {hitRate60 > 0 ? (
                          <span
                            className={`px-1.5 py-0.2 rounded text-[9.5px] font-medium ${
                              hitRate60 >= 65 ? 'text-emerald-400' : 'text-slate-400'
                            }`}
                          >
                            {hitRate60.toFixed(1)}% {op.hits_60d != null ? `(${op.hits_60d}/${op.total_60d})` : ''}
                          </span>
                        ) : (
                          <span className="text-slate-600">---</span>
                        )}
                      </td>

                      {/* BofA Flow */}
                      <td className="py-3 px-3 text-right">
                        <span
                          className={`font-semibold ${
                            op.mlb_net_flow_tl > 0 ? 'text-emerald-400' : op.mlb_net_flow_tl < 0 ? 'text-rose-400' : 'text-slate-500'
                          }`}
                        >
                          {formatTL(op.mlb_net_flow_tl)}
                        </span>
                      </td>

                      {/* Launch Actions */}
                      <td className="py-3 px-3 text-center">
                        <div className="flex items-center justify-center space-x-1.5">
                          <button
                            onClick={() => handleLaunchSymbol(op.symbol, 'candles')}
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-cyan-950 border border-slate-800 hover:border-cyan-500/50 text-slate-400 hover:text-cyan-300 transition-colors"
                            title={`Inspect ${op.symbol} Candlesticks & Order Flow`}
                          >
                            <LineChart className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => handleLaunchSymbol(op.symbol, 'multihorizon')}
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-emerald-950 border border-slate-800 hover:border-emerald-500/50 text-slate-400 hover:text-emerald-300 transition-colors"
                            title={`Inspect ${op.symbol} in Multi-Horizon Hub`}
                          >
                            <Sparkles className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
