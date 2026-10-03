import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchPredictedOpportunities } from '../api/client';
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
} from 'lucide-react';

interface OpportunityActionsDashboardProps {
  onSelectSymbol: (symbol: string) => void;
  onNavigateTab?: (tab: 'candles' | 'tertip' | 'event' | 'timewindow' | 'oracle' | 'opportunities') => void;
  currentTradeDate?: string;
}

type FilterCategory = 'all' | 'big_moves' | 'sig_moves' | 'longs' | 'shorts' | 'bofa_buy' | 'bofa_sell';
type SortField = 'return_desc' | 'return_asc' | 'magnitude' | 'hit_rate' | 'bofa_flow';

export const OpportunityActionsDashboard: React.FC<OpportunityActionsDashboardProps> = ({
  onSelectSymbol,
  onNavigateTab,
  currentTradeDate,
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [activeFilter, setActiveFilter] = useState<FilterCategory>('all');
  const [sortField, setSortField] = useState<SortField>('return_desc');
  const [selectedSector, setSelectedSector] = useState<string>('all');

  const {
    data: opportunityData,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['predicted-opportunities', currentTradeDate],
    queryFn: () => fetchPredictedOpportunities(currentTradeDate),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });

  const opportunities = opportunityData?.opportunities || [];

  // Extract unique sectors
  const availableSectors = useMemo(() => {
    const set = new Set<string>();
    opportunities.forEach((op) => {
      if (op.sector) set.add(op.sector);
    });
    return Array.from(set).sort();
  }, [opportunities]);

  // Filtered and sorted opportunity list
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
          case 'hit_rate':
            return (b.champion_dir_hit_rate_pct || 0) - (a.champion_dir_hit_rate_pct || 0);
          case 'bofa_flow':
            return b.mlb_net_flow_tl - a.mlb_net_flow_tl;
          default:
            return b.expected_return_pct - a.expected_return_pct;
        }
      });
  }, [opportunities, selectedSector, searchTerm, activeFilter, sortField]);

  const handleLaunchSymbol = (sym: string, targetTab?: 'candles' | 'oracle') => {
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
          <div className="text-xs text-slate-400 mt-1">Evaluating multi-horizon ML models & institutional order flow</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      {/* ── TOP HERO HEADER & MARKET BREADTH RIBBON ───────────────────────────────── */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800/80 bg-gradient-to-r from-slate-900/90 via-slate-900/70 to-slate-950/90 shadow-xl relative overflow-hidden">
        {/* Glow Accents */}
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
                      T+1 Tactical Radar
                    </span>
                  </h1>
                </div>
                <p className="text-xs text-slate-400">
                  Constituent Multi-Horizon Forecast Ranking • Institutional Footprint Confluence • High-Conviction Long & Short Setups
                </p>
              </div>
            </div>
          </div>

          {/* Quick Session Date & Refresh Action */}
          <div className="flex items-center space-x-3">
            <div className="text-right font-mono">
              <div className="text-[11px] text-slate-400 flex items-center justify-end space-x-1">
                <Calendar className="w-3 h-3 text-cyan-400" />
                <span>As of Date: <strong className="text-slate-200">{opportunityData?.as_of_date || '2026-09-30'}</strong></span>
              </div>
              <div className="text-[11px] text-emerald-400 font-semibold">
                Target: Next Trading Session
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

        {/* Market Opportunity Breadth Gauge */}
        {opportunityData && (
          <div className="mt-5 pt-4 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-slate-950/60 border border-slate-800/70 p-3 rounded-xl">
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider mb-0.5">BIST 30 Constituents</div>
              <div className="text-lg font-bold font-mono text-white flex items-center space-x-1.5">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                <span>{opportunityData.total_constituents} Equities</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-emerald-900/30 p-3 rounded-xl">
              <div className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider mb-0.5">Bullish Bias</div>
              <div className="text-lg font-bold font-mono text-emerald-300 flex items-center space-x-1.5">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <span>{opportunityData.bullish_count} ({((opportunityData.bullish_count / opportunityData.total_constituents) * 100).toFixed(0)}%)</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-rose-900/30 p-3 rounded-xl">
              <div className="text-[10px] font-mono text-rose-400 uppercase tracking-wider mb-0.5">Bearish Bias</div>
              <div className="text-lg font-bold font-mono text-rose-300 flex items-center space-x-1.5">
                <TrendingDown className="w-4 h-4 text-rose-400" />
                <span>{opportunityData.bearish_count} ({((opportunityData.bearish_count / opportunityData.total_constituents) * 100).toFixed(0)}%)</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-amber-900/30 p-3 rounded-xl">
              <div className="text-[10px] font-mono text-amber-400 uppercase tracking-wider mb-0.5">Big Moves (|Δ| ≥ 2.0%)</div>
              <div className="text-lg font-bold font-mono text-amber-300 flex items-center space-x-1.5">
                <Flame className="w-4 h-4 text-amber-400" />
                <span>{opportunityData.high_conviction_count} High Conviction</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-indigo-900/30 p-3 rounded-xl">
              <div className="text-[10px] font-mono text-indigo-400 uppercase tracking-wider mb-0.5">Sig Moves (|Δ| ≥ 1.0%)</div>
              <div className="text-lg font-bold font-mono text-indigo-300 flex items-center space-x-1.5">
                <Target className="w-4 h-4 text-indigo-400" />
                <span>{opportunityData.significant_moves_count} Actionable</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-slate-800/70 p-3 rounded-xl">
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider mb-0.5">Market Mean Δ</div>
              <div className={`text-lg font-bold font-mono flex items-center space-x-1.5 ${opportunityData.avg_expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                <Activity className="w-4 h-4" />
                <span>{formatPercent(opportunityData.avg_expected_return_pct)}</span>
              </div>
            </div>
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
                  Top Bullish Opportunities
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
                className="group p-3.5 rounded-xl bg-slate-900/80 hover:bg-slate-850 border border-emerald-500/30 hover:border-emerald-400/60 transition-all shadow-md hover:shadow-emerald-500/10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 relative overflow-hidden"
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
                        {(op.crowned_horizon || '12m').toUpperCase()} • {op.ml_champion_type}
                      </span>
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

                  {/* Playbook & Hit Rate */}
                  <div className="text-right hidden sm:block">
                    <div className="text-[11px] font-semibold text-emerald-300 bg-emerald-950/60 border border-emerald-500/30 px-2 py-0.5 rounded">
                      {op.playbook?.replace(/_/g, ' ') || 'BUY ACCUMULATION'}
                    </div>
                    <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                      30D Win: <strong className="text-slate-200">{op.champion_dir_hit_rate_pct?.toFixed(0)}%</strong> ({op.champion_dir_hits}/30)
                    </div>
                  </div>

                  {/* Quick Action Navigation */}
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
                      onClick={() => handleLaunchSymbol(op.symbol, 'oracle')}
                      className="px-2.5 py-1.5 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 text-xs font-semibold transition-all flex items-center space-x-1"
                      title="Open in Gold Predictive Hub"
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
                  Top Short / Fade Opportunities
                  <span className="text-[10px] font-mono uppercase bg-rose-500/20 text-rose-300 px-2 py-0.5 rounded-full border border-rose-500/30">
                    Highest Expected Downside
                  </span>
                </h2>
                <div className="text-[11px] text-slate-400">Institutional liquidation pressure & profit-taking fade setups</div>
              </div>
            </div>
          </div>

          <div className="space-y-3 flex-1">
            {opportunityData?.top_shorts?.slice(0, 4).map((op, idx) => (
              <div
                key={op.symbol}
                className="group p-3.5 rounded-xl bg-slate-900/80 hover:bg-slate-850 border border-rose-500/30 hover:border-rose-400/60 transition-all shadow-md hover:shadow-rose-500/10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 relative overflow-hidden"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-7 h-7 rounded-lg bg-rose-500/20 border border-rose-500/40 flex items-center justify-center font-mono font-bold text-rose-400 text-xs">
                    #{idx + 1}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-base font-bold text-white font-mono group-hover:text-rose-300 transition-colors">
                        {op.symbol}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded border border-slate-700/60">
                        {op.sector}
                      </span>
                      <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                        {(op.crowned_horizon || '12m').toUpperCase()} • {op.ml_champion_type}
                      </span>
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

                  {/* Playbook & Hit Rate */}
                  <div className="text-right hidden sm:block">
                    <div className="text-[11px] font-semibold text-rose-300 bg-rose-950/60 border border-rose-500/30 px-2 py-0.5 rounded">
                      {op.playbook?.replace(/_/g, ' ') || 'SELL PRESSURE'}
                    </div>
                    <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                      30D Win: <strong className="text-slate-200">{op.champion_dir_hit_rate_pct?.toFixed(0)}%</strong> ({op.champion_dir_hits}/30)
                    </div>
                  </div>

                  {/* Quick Action Navigation */}
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
                      onClick={() => handleLaunchSymbol(op.symbol, 'oracle')}
                      className="px-2.5 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-300 text-xs font-semibold transition-all flex items-center space-x-1"
                      title="Open in Gold Predictive Hub"
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

      {/* ── UNIVERSE OPPORTUNITY MATRIX (ALL 30 EQUITIES HIGHLIGHTED) ─────────────── */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800/80 bg-slate-900/60 shadow-xl space-y-4">
        {/* Controls Toolbar: Search, Filters, Sector & Sort */}
        <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          {/* Search Box */}
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search ticker, company name, sector, or playbook..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-950/80 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/60 transition-colors font-mono"
            />
          </div>

          {/* Quick Filter Pills */}
          <div className="flex items-center space-x-1.5 overflow-x-auto pb-1 lg:pb-0">
            {[
              { id: 'all', label: `All Equities (${opportunities.length})`, icon: Layers },
              { id: 'big_moves', label: `Big Moves (|Δ| ≥ 2%)`, icon: Flame, color: 'text-amber-400' },
              { id: 'sig_moves', label: `Significant (|Δ| ≥ 1%)`, icon: Target, color: 'text-indigo-400' },
              { id: 'longs', label: 'Bullish Longs', icon: TrendingUp, color: 'text-emerald-400' },
              { id: 'shorts', label: 'Bearish Shorts', icon: TrendingDown, color: 'text-rose-400' },
              { id: 'bofa_buy', label: 'BofA Buying (MLB > 0)', icon: Activity, color: 'text-cyan-400' },
              { id: 'bofa_sell', label: 'BofA Selling (MLB < 0)', icon: Activity, color: 'text-rose-400' },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeFilter === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveFilter(tab.id as FilterCategory)}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                    isActive
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm'
                      : 'bg-slate-950/60 text-slate-400 hover:text-slate-200 border border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${tab.color || (isActive ? 'text-cyan-400' : 'text-slate-400')}`} />
                  <span>{tab.label}</span>
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
              <option value="hit_rate">30D Hit Rate: Highest First</option>
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
                <th className="py-3 px-3">Opportunity Tier</th>
                <th className="py-3 px-3 text-right">Current Price</th>
                <th className="py-3 px-3 text-right">Target Range</th>
                <th className="py-3 px-4 text-center">Forecast Return & Gauge</th>
                <th className="py-3 px-3">Institutional Playbook</th>
                <th className="py-3 px-3 text-center">Champion Model</th>
                <th className="py-3 px-3 text-right">30D Win Rate</th>
                <th className="py-3 px-3 text-right">BofA Net Flow TL</th>
                <th className="py-3 px-3 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredOpportunities.length === 0 ? (
                <tr>
                  <td colSpan={11} className="py-12 text-center text-slate-500 font-sans">
                    No constituent opportunities match the selected criteria.
                  </td>
                </tr>
              ) : (
                filteredOpportunities.map((op, idx) => {
                  const isHighConviction = Math.abs(op.expected_return_pct) >= 2.0;
                  const isSigMove = Math.abs(op.expected_return_pct) >= 1.0;
                  const isPositive = op.expected_return_pct > 0.25;
                  const isNegative = op.expected_return_pct < -0.25;

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

                      {/* Opportunity Tier Badge */}
                      <td className="py-3 px-3">
                        {isHighConviction ? (
                          <span
                            className={`inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border shadow-sm ${
                              isPositive
                                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-emerald-500/10'
                                : 'bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-rose-500/10'
                            }`}
                          >
                            <Flame className="w-3 h-3" />
                            <span>★ High Conviction</span>
                          </span>
                        ) : isSigMove ? (
                          <span
                            className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-semibold uppercase border ${
                              isPositive
                                ? 'bg-emerald-950/40 text-emerald-300 border-emerald-500/30'
                                : 'bg-rose-950/40 text-rose-300 border-rose-500/30'
                            }`}
                          >
                            <Target className="w-2.5 h-2.5" />
                            <span>Significant</span>
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-normal uppercase text-slate-500 bg-slate-900 border border-slate-800">
                            Consolidation
                          </span>
                        )}
                      </td>

                      {/* Current Price */}
                      <td className="py-3 px-3 text-right text-slate-200">
                        {op.current_price.toFixed(2)} TL
                      </td>

                      {/* Target Price [Low - High] */}
                      <td className="py-3 px-3 text-right">
                        <div className={`font-semibold ${isPositive ? 'text-emerald-300' : isNegative ? 'text-rose-300' : 'text-slate-300'}`}>
                          {op.target_price.toFixed(2)} TL
                        </div>
                        <div className="text-[10px] text-slate-500">
                          [{op.price_low.toFixed(1)} - {op.price_high.toFixed(1)}]
                        </div>
                      </td>

                      {/* Expected Return & Visual Gauge */}
                      <td className="py-3 px-4 text-center min-w-[150px]">
                        <div
                          className={`text-sm font-bold flex items-center justify-center space-x-1 ${
                            isPositive ? 'text-emerald-400' : isNegative ? 'text-rose-400' : 'text-slate-400'
                          }`}
                        >
                          {isPositive && <TrendingUp className="w-3.5 h-3.5" />}
                          {isNegative && <TrendingDown className="w-3.5 h-3.5" />}
                          <span>{formatPercent(op.expected_return_pct)}</span>
                        </div>

                        {/* Gauge bar centered at zero */}
                        <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden mt-1.5 flex relative">
                          <div className="w-1/2 flex justify-end">
                            {isNegative && (
                              <div
                                className="bg-rose-500 h-full rounded-l-full"
                                style={{
                                  width: `${Math.min(100, (Math.abs(op.expected_return_pct) / 7.0) * 100)}%`,
                                }}
                              />
                            )}
                          </div>
                          <div className="w-[1px] bg-slate-700 h-full z-10" />
                          <div className="w-1/2 flex justify-start">
                            {isPositive && (
                              <div
                                className="bg-emerald-500 h-full rounded-r-full"
                                style={{
                                  width: `${Math.min(100, (op.expected_return_pct / 7.0) * 100)}%`,
                                }}
                              />
                            )}
                          </div>
                        </div>
                      </td>

                      {/* Institutional Playbook */}
                      <td className="py-3 px-3">
                        <span
                          className={`inline-block text-[11px] font-sans font-semibold px-2 py-0.5 rounded border ${
                            op.playbook?.includes('BUY') || op.playbook?.includes('SQUEEZE')
                              ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/30'
                              : op.playbook?.includes('SELL') || op.playbook?.includes('LIQUIDITY')
                              ? 'bg-rose-950/60 text-rose-300 border-rose-500/30'
                              : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}
                        >
                          {op.playbook?.replace(/_/g, ' ') || 'NEUTRAL WAIT'}
                        </span>
                      </td>

                      {/* Champion Model & Horizon */}
                      <td className="py-3 px-3 text-center">
                        <span className="text-[11px] font-semibold text-cyan-300 bg-cyan-950/40 border border-cyan-500/30 px-2 py-0.5 rounded">
                          {op.ml_champion_type} ({(op.crowned_horizon || '12m').toUpperCase()})
                        </span>
                      </td>

                      {/* 30D Hit Rate */}
                      <td className="py-3 px-3 text-right">
                        <div
                          className={`font-semibold ${
                            (op.champion_dir_hit_rate_pct || 0) >= 70
                              ? 'text-emerald-400'
                              : (op.champion_dir_hit_rate_pct || 0) >= 60
                              ? 'text-teal-300'
                              : 'text-slate-300'
                          }`}
                        >
                          {op.champion_dir_hit_rate_pct ? `${op.champion_dir_hit_rate_pct.toFixed(0)}%` : '—'}
                        </div>
                        <div className="text-[10px] text-slate-500">
                          {op.champion_dir_hits ? `${op.champion_dir_hits}/30 hits` : ''}
                        </div>
                      </td>

                      {/* BofA Net Flow TL */}
                      <td className="py-3 px-3 text-right">
                        <div
                          className={`font-semibold ${
                            op.mlb_net_flow_tl > 0
                              ? 'text-emerald-400'
                              : op.mlb_net_flow_tl < 0
                              ? 'text-rose-400'
                              : 'text-slate-500'
                          }`}
                        >
                          {formatTL(op.mlb_net_flow_tl)}
                        </div>
                      </td>

                      {/* Quick Actions */}
                      <td className="py-3 px-3 text-center">
                        <div className="flex items-center justify-center space-x-1">
                          <button
                            onClick={() => handleLaunchSymbol(op.symbol, 'candles')}
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-cyan-950 border border-slate-800 hover:border-cyan-500/50 text-slate-400 hover:text-cyan-300 transition-colors"
                            title={`Inspect ${op.symbol} Candlesticks & Order Flow`}
                          >
                            <LineChart className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => handleLaunchSymbol(op.symbol, 'oracle')}
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-emerald-950 border border-slate-800 hover:border-emerald-500/50 text-slate-400 hover:text-emerald-300 transition-colors"
                            title={`Inspect ${op.symbol} in Gold Predictive Hub`}
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

        {/* Footer Note */}
        <div className="text-[11px] text-slate-500 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pt-2">
          <div className="flex items-center space-x-1.5">
            <Award className="w-3.5 h-3.5 text-cyan-400" />
            <span>
              Opportunities crowned by Walk-Forward Arena composite loss against calibrated 25 bps consolidation deadband.
            </span>
          </div>
          <div className="font-mono text-slate-400">
            Displaying <strong className="text-white">{filteredOpportunities.length}</strong> of{' '}
            <strong className="text-white">{opportunities.length}</strong> BIST 30 Equities
          </div>
        </div>
      </div>
    </div>
  );
};
