import React from 'react';
import { ArrowDownRight, ArrowUpRight, Coins, DollarSign, Repeat, TrendingDown, TrendingUp } from 'lucide-react';

function formatTL(num) {
  if (num === null || num === undefined || isNaN(num)) return '0,00 ₺';
  const abs = Math.abs(num);
  let formatted = '';
  if (abs >= 1_000_000_000) {
    formatted = (num / 1_000_000_000).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' Milyar ₺';
  } else if (abs >= 1_000_000) {
    formatted = (num / 1_000_000).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' Milyon ₺';
  } else {
    formatted = num.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' ₺';
  }
  return formatted;
}

function formatLots(num) {
  if (num === null || num === undefined || isNaN(num)) return '0 Lot';
  return Math.round(num).toLocaleString('tr-TR') + ' Lot';
}

export default function BrokerMetricCards({ summary, candleSummary, selectedBroker }) {
  // Use candle-level aggregated stats for the selected symbol, or fall back to macro summary
  const buyTL = candleSummary.buyTurnoverTl;
  const sellTL = candleSummary.sellTurnoverTl;
  const netFlowTL = candleSummary.netFlowTl;
  const matchedVol = candleSummary.matchedVolume;
  const realizedPnL = candleSummary.realizedPnlTl;

  const isProfitable = realizedPnL >= 0;
  const isNetBuyer = netFlowTL >= 0;

  return (
    <section className="stats-grid">
      {/* 1. Buy Turnover */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>Toplam Alış Tutarı</span>
          <ArrowUpRight size={16} className="text-bull" />
        </div>
        <div className="stat-value num-mono text-bull">{formatTL(buyTL)}</div>
        <div className="stat-subtext num-mono">Hacim: {formatLots(candleSummary.buyVolume)}</div>
      </div>

      {/* 2. Sell Turnover */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>Toplam Satış Tutarı</span>
          <ArrowDownRight size={16} className="text-bear" />
        </div>
        <div className="stat-value num-mono text-bear">{formatTL(sellTL)}</div>
        <div className="stat-subtext num-mono">Hacim: {formatLots(candleSummary.sellVolume)}</div>
      </div>

      {/* 3. Net Flow */}
      <div className={`stat-card glass-card ${isNetBuyer ? 'bull' : 'bear'}`}>
        <div className="stat-header">
          <span>Net Pozisyon (Akış)</span>
          <span className={`status-pill ${isNetBuyer ? 'bg-bull-badge' : 'bg-bear-badge'}`} style={{ padding: '2px 8px', fontSize: '0.7rem' }}>
            {isNetBuyer ? 'NET ALICI' : 'NET SATICI'}
          </span>
        </div>
        <div className={`stat-value num-mono ${isNetBuyer ? 'text-bull' : 'text-bear'}`}>
          {isNetBuyer ? '+' : ''}{formatTL(netFlowTL)}
        </div>
        <div className="stat-subtext num-mono">Net Lot: {formatLots(candleSummary.netVolume)}</div>
      </div>

      {/* 4. Matched Turnaround Volume */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>Eşleşen Gün İçi Hacim</span>
          <Repeat size={16} style={{ color: 'var(--brand-blue)' }} />
        </div>
        <div className="stat-value num-mono" style={{ color: '#93c5fd' }}>
          {formatLots(matchedVol)}
        </div>
        <div className="stat-subtext">Kurum İçi Al-Sat Eşleşmesi</div>
      </div>

      {/* 5. Realized PnL */}
      <div className={`stat-card glass-card ${isProfitable ? 'bull' : 'bear'}`}>
        <div className="stat-header">
          <span>Gerçekleşen Kâr/Zarar</span>
          {isProfitable ? (
            <TrendingUp size={16} className="text-bull" />
          ) : (
            <TrendingDown size={16} className="text-bear" />
          )}
        </div>
        <div className={`stat-value num-mono ${isProfitable ? 'text-bull' : 'text-bear'}`}>
          {isProfitable ? '+' : ''}{formatTL(realizedPnL)}
        </div>
        <div className="stat-subtext">
          {isProfitable ? 'Net Pozitif Kazanç (Roundtrip)' : 'Net Zarar Realizasyonu'}
        </div>
      </div>
    </section>
  );
}
