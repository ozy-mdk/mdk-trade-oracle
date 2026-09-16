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
  // Invariant True Session PnL from the institutional daily FIFO ledger
  // (Prevents discrepancy across different candle timeframes like 1m vs 60m)
  const trueDailyPnL = summary && summary.realizedPnlTl !== undefined
    ? summary.realizedPnlTl
    : candleSummary.realizedPnlTl;

  const intraPnL = summary && summary.intradayPnlTl !== undefined
    ? summary.intradayPnlTl
    : 0;

  const carryPnL = summary && summary.carryFifoPnlTl !== undefined
    ? summary.carryFifoPnlTl
    : 0;

  const buyTL = summary && summary.totalBuyTurnoverTl ? summary.totalBuyTurnoverTl : candleSummary.buyTurnoverTl;
  const sellTL = summary && summary.totalSellTurnoverTl ? summary.totalSellTurnoverTl : candleSummary.sellTurnoverTl;
  const netFlowTL = summary && summary.netFlowTl ? summary.netFlowTl : candleSummary.netFlowTl;
  const matchedVol = summary && summary.matchedVolume ? summary.matchedVolume : candleSummary.matchedVolume;

  const isProfitable = trueDailyPnL >= 0;
  const isNetBuyer = netFlowTL >= 0;

  return (
    <section className="stats-grid">
      {/* 1. Total Buy Turnover */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>Toplam Günlük Alış</span>
          <ArrowUpRight size={16} className="text-bull" />
        </div>
        <div className="stat-value num-mono text-bull">{formatTL(buyTL)}</div>
        <div className="stat-subtext num-mono">Hacim: {formatLots(summary?.totalBuyVolume || candleSummary.buyVolume)}</div>
      </div>

      {/* 2. Total Sell Turnover */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>Toplam Günlük Satış</span>
          <ArrowDownRight size={16} className="text-bear" />
        </div>
        <div className="stat-value num-mono text-bear">{formatTL(sellTL)}</div>
        <div className="stat-subtext num-mono">Hacim: {formatLots(summary?.totalSellVolume || candleSummary.sellVolume)}</div>
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
        <div className="stat-subtext num-mono">
          {selectedBroker} Piyasa Yönü
        </div>
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
        <div className="stat-subtext">Gün İçi Al-Sat Turnaround</div>
      </div>

      {/* 5. Invariant Realized PnL */}
      <div className={`stat-card glass-card ${isProfitable ? 'bull' : 'bear'}`}>
        <div className="stat-header">
          <span>Günlük Gerçekleşen Net Kâr/Zarar</span>
          {isProfitable ? (
            <TrendingUp size={16} className="text-bull" />
          ) : (
            <TrendingDown size={16} className="text-bear" />
          )}
        </div>
        <div className={`stat-value num-mono ${isProfitable ? 'text-bull' : 'text-bear'}`}>
          {isProfitable ? '+' : ''}{formatTL(trueDailyPnL)}
        </div>
        <div className="stat-subtext" style={{ display: 'flex', gap: '8px', fontSize: '0.72rem' }}>
          <span title="Gün içinde alınan ve aynı gün satılan lotlardan kâr">
            Gün İçi: <b className={intraPnL >= 0 ? 'text-bull' : 'text-bear'}>{formatTL(intraPnL)}</b>
          </span>
          <span title="Önceki günlerden taşınan lotların kapanış kârı">
            Carry FIFO: <b className={carryPnL >= 0 ? 'text-bull' : 'text-bear'}>{formatTL(carryPnL)}</b>
          </span>
        </div>
      </div>
    </section>
  );
}
