import React from 'react';
import { BarChart3, Building2, Calendar, Clock } from 'lucide-react';

const TIMEFRAMES = ['1m', '5m', '15m', '30m', '60m', '120m', '240m', '8h'];

export default function FilterBar({
  instruments,
  brokers,
  dates,
  selectedSymbol,
  onSelectSymbol,
  selectedBroker,
  onSelectBroker,
  selectedTimeframe,
  onSelectTimeframe,
  selectedDate,
  onSelectDate,
}) {
  return (
    <section className="filter-bar glass-card">
      {/* Symbol / Hisse Selector */}
      <div className="filter-item">
        <label className="filter-label">
          <BarChart3 size={14} />
          <span>Hisse Senedi</span>
        </label>
        <select
          className="filter-select num-mono"
          value={selectedSymbol}
          onChange={(e) => onSelectSymbol(e.target.value)}
        >
          {instruments.map((inst) => (
            <option key={inst.symbol} value={inst.symbol}>
              {inst.symbol} — {inst.name}
            </option>
          ))}
        </select>
      </div>

      {/* Broker / Kurum Selector */}
      <div className="filter-item">
        <label className="filter-label">
          <Building2 size={14} />
          <span>Aracı Kurum</span>
        </label>
        <select
          className="filter-select"
          value={selectedBroker}
          onChange={(e) => onSelectBroker(e.target.value)}
        >
          {brokers.map((brk) => (
            <option key={brk.brokerId} value={brk.brokerId}>
              {brk.brokerId === 'MLB' ? '⭐ ' : ''}
              {brk.brokerId} — {brk.brokerName}
            </option>
          ))}
        </select>
      </div>

      {/* Date / Tarih Selector */}
      <div className="filter-item">
        <label className="filter-label">
          <Calendar size={14} />
          <span>İşlem Tarihi</span>
        </label>
        <select
          className="filter-select num-mono"
          value={selectedDate}
          onChange={(e) => onSelectDate(e.target.value)}
        >
          {dates.map((d) => (
            <option key={d} value={d}>
              {d} {d === '2026-09-14' ? '(En Son Oturum)' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Timeframe Button Group */}
      <div className="filter-item" style={{ minWidth: '280px' }}>
        <label className="filter-label">
          <Clock size={14} />
          <span>Mum Periyodu (Timeframe)</span>
        </label>
        <div className="tf-group">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              type="button"
              className={`tf-btn ${selectedTimeframe === tf ? 'active' : ''}`}
              onClick={() => onSelectTimeframe(tf)}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
