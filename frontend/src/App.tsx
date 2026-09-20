import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchInstruments, fetchBrokers, fetchDateRange } from './api/client';
import { CandleDashboard } from './components/CandleDashboard';
import { TertipDashboard } from './components/TertipDashboard';
import { EventStudyDashboard } from './components/EventStudyDashboard';
import { TimeWindowTerminal } from './components/TimeWindowTerminal';
import { OracleHubDashboard } from './components/OracleHubDashboard';
import {
  LineChart,
  Boxes,
  Binary,
  Clock,
  Sparkles,
  ChevronDown,
  Cpu,
  Database,
} from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'candles' | 'tertip' | 'event' | 'timewindow' | 'oracle'>('candles');
  const [selectedSymbol, setSelectedSymbol] = useState<string>('THYAO');
  const [selectedBroker, setSelectedBroker] = useState<string>('MLB');
  const [selectedDate, setSelectedDate] = useState<string>('2026-09-16');
  const [currentTimeTRT, setCurrentTimeTRT] = useState<string>('');

  // Fetch Instruments and Brokers metadata
  const { data: instruments } = useQuery({
    queryKey: ['metaInstruments'],
    queryFn: fetchInstruments,
  });

  const { data: brokers } = useQuery({
    queryKey: ['metaBrokers'],
    queryFn: fetchBrokers,
  });

  // Fetch dynamic available date range from lakehouse
  const { data: dateRange } = useQuery({
    queryKey: ['metaDateRange'],
    queryFn: () => fetchDateRange(),
  });

  useEffect(() => {
    if (dateRange?.latest_date && selectedDate === '2026-09-16') {
      setSelectedDate(dateRange.latest_date);
    }
  }, [dateRange]);

  // Live TRT Clock (Turkish Time / Europe/Istanbul / UTC+3)
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const trtStr = now.toLocaleTimeString('tr-TR', {
        timeZone: 'Europe/Istanbul',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
      setCurrentTimeTRT(trtStr);
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectSymbol = (sym: string) => {
    setSelectedSymbol(sym);
    setActiveTab('candles');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#070b14] text-slate-100 font-sans">
      {/* Top Institutional Workstation Header */}
      <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 px-4 py-2.5 shadow-xl">
        <div className="max-w-[1920px] mx-auto flex flex-wrap items-center justify-between gap-3">
          {/* Brand & Mission Badge */}
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-sm tracking-wider text-white">
                  MDK TRADING ORACLE
                </span>
                <span className="text-[10px] font-mono font-semibold px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  WORKSTATION
                </span>
              </div>
              <div className="text-[10px] text-slate-400 hidden sm:block">
                Institutional Order Flow & Algorithmic Footprints Tracker
              </div>
            </div>
          </div>

          {/* Central Controls: Symbol, Broker, Date */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Symbol Selector */}
            <div className="relative">
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                aria-label="Stock Symbol"
                className="bg-slate-900 border border-slate-700/80 hover:border-cyan-500/50 rounded-lg px-3 py-1.5 text-xs font-mono font-bold text-white focus:outline-none appearance-none pr-8 cursor-pointer transition-colors shadow-inner"
              >
                {(instruments || []).map((inst) => (
                  <option key={inst.symbol} value={inst.symbol}>
                    {inst.symbol} — {inst.name.length > 24 ? inst.name.substring(0, 24) + '...' : inst.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 transform -translate-y-1/2 pointer-events-none" />
            </div>

            {/* Broker Selector */}
            <div className="relative">
              <select
                value={selectedBroker}
                onChange={(e) => setSelectedBroker(e.target.value)}
                aria-label="Broker Institution"
                className="bg-slate-900 border border-slate-700/80 hover:border-cyan-500/50 rounded-lg px-3 py-1.5 text-xs font-mono font-semibold text-slate-200 focus:outline-none appearance-none pr-8 cursor-pointer transition-colors shadow-inner"
              >
                {(brokers || []).map((b) => (
                  <option key={b.broker_id} value={b.broker_id}>
                    {b.broker_id} — {b.broker_name}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 transform -translate-y-1/2 pointer-events-none" />
            </div>

            {/* Date Selector */}
            <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs">
              <span className="text-slate-400 text-[11px]">Session:</span>
              <input
                type="date"
                value={selectedDate}
                min={dateRange?.min_date}
                max={dateRange?.max_date}
                onChange={(e) => setSelectedDate(e.target.value)}
                aria-label="Trading Session Date"
                className="bg-transparent text-slate-200 text-xs font-mono focus:outline-none cursor-pointer"
              />
              {dateRange && selectedDate !== dateRange.latest_date ? (
                <button
                  onClick={() => setSelectedDate(dateRange.latest_date)}
                  className="ml-1 px-1.5 py-0.5 text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 rounded border border-cyan-500/40 transition-colors"
                  title={`Jump to latest session (${dateRange.latest_date})`}
                >
                  Latest
                </button>
              ) : (
                <span className="ml-1 px-1 py-0.2 text-[9px] font-mono font-bold bg-emerald-500/10 text-emerald-400 rounded border border-emerald-500/30">
                  LATEST
                </span>
              )}
            </div>
          </div>

          {/* Engine Status & Live TRT Clock */}
          <div className="flex items-center space-x-3 text-xs">
            <div className="hidden lg:flex items-center space-x-2 text-[11px] text-slate-400 bg-slate-900/60 px-2.5 py-1 rounded-md border border-slate-800">
              <Database className="w-3 h-3 text-emerald-400" />
              <span>TimescaleDB</span>
              <span className="text-slate-600">•</span>
              <Cpu className="w-3 h-3 text-cyan-400" />
              <span>M5 Mac Pro</span>
            </div>

            <div className="flex items-center space-x-1.5 bg-cyan-950/40 border border-cyan-500/30 px-3 py-1 rounded-lg text-cyan-300 font-mono font-semibold">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              <span>{currentTimeTRT || '09:55:00'} TRT</span>
            </div>
          </div>
        </div>

        {/* 5 Modular Workstation Navigation Tabs */}
        <div className="max-w-[1920px] mx-auto mt-2.5 flex items-center space-x-1 border-t border-slate-800/60 pt-2 overflow-x-auto">
          {[
            { id: 'candles', label: 'Candlestick & Order Flow', icon: LineChart },
            { id: 'tertip', label: 'FIFO Tertip Inventory', icon: Boxes },
            { id: 'event', label: 'Event Study Scanner', icon: Binary },
            { id: 'timewindow', label: 'Time Window Terminal', icon: Clock },
            { id: 'oracle', label: 'Gold Predictive Hub', icon: Sparkles },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Workstation Workspace Area */}
      <main className="flex-1 max-w-[1920px] w-full mx-auto p-4">
        {activeTab === 'candles' && (
          <CandleDashboard
            symbol={selectedSymbol}
            brokerId={selectedBroker}
            tradeDate={selectedDate}
          />
        )}

        {activeTab === 'tertip' && (
          <TertipDashboard
            brokerId={selectedBroker}
            symbol={selectedSymbol}
            tradeDate={selectedDate}
            onSelectSymbol={handleSelectSymbol}
          />
        )}

        {activeTab === 'event' && (
          <EventStudyDashboard
            initialSymbol={selectedSymbol}
            brokerId={selectedBroker}
            onSelectSymbol={handleSelectSymbol}
          />
        )}

        {activeTab === 'timewindow' && (
          <TimeWindowTerminal
            symbol={selectedSymbol}
            brokerId={selectedBroker}
            tradeDate={selectedDate}
          />
        )}

        {activeTab === 'oracle' && (
          <OracleHubDashboard onSelectSymbol={handleSelectSymbol} />
        )}
      </main>

      {/* Micro-Footer Status Ribbon */}
      <footer className="glass-panel border-t border-slate-800/80 px-4 py-2 text-[11px] text-slate-500 flex flex-wrap items-center justify-between">
        <div>
          MDK Trading Oracle • Zero-Lookahead Institutional Order Flow Analytics • PostgreSQL 16 + TimescaleDB + React 18
        </div>
        <div className="font-mono text-slate-400">
          Selected: <span className="text-white font-semibold">{selectedSymbol}</span> | Broker: <span className="text-white font-semibold">{selectedBroker}</span> | Date: <span className="text-white font-semibold">{selectedDate}</span>
        </div>
      </footer>
    </div>
  );
};
