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
  const sign = num > 0 ? '+' : (num < 0 ? '-' : '');
  return `${sign}${Math.abs(Math.round(num)).toLocaleString('tr-TR')} Lot`;
}

export default function BrokerMetricCards({ summary, candleSummary, selectedBroker, selectedSymbol }) {
  const isXU030 = !selectedSymbol || selectedSymbol === 'XU030';

  // 1. Volumes and turnovers
  const buyTL = summary?.totalBuyTurnoverTl ?? candleSummary.buyTurnoverTl;
  const sellTL = summary?.totalSellTurnoverTl ?? candleSummary.sellTurnoverTl;
  const buyVol = summary?.totalBuyVolume ?? candleSummary.buyVolume;
  const sellVol = summary?.totalSellVolume ?? candleSummary.sellVolume;

  // 2. Net flow and net volume (kalan miktar)
  const netFlowTL = summary?.netFlowTl ?? candleSummary.netFlowTl;
  const netVolume = (summary?.netVolume !== null && summary?.netVolume !== undefined)
    ? summary.netVolume
    : candleSummary.netVolume;

  // 3. Matched turnaround volume
  const matchedVol = summary?.matchedVolume ?? candleSummary.matchedVolume;

  // 4. Intraday and realized PnL
  const intraPnL = summary?.intradayPnlTl !== undefined
    ? summary.intradayPnlTl
    : candleSummary.realizedPnlTl;

  const carryPnL = summary?.carryFifoPnlTl ?? 0;

  const trueDailyPnL = summary?.realizedPnlTl !== undefined
    ? summary.realizedPnlTl
    : candleSummary.realizedPnlTl;

  // Status flags
  const isNetBuyer = isXU030 ? (netFlowTL >= 0) : (netVolume >= 0);
  const isProfitable = isXU030 ? (trueDailyPnL >= 0) : (intraPnL >= 0);

  return (
    <section className="stats-grid">
      {/* 1. Total Buy Turnover */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>{isXU030 ? 'BIST 30 Toplam Alış' : `${selectedSymbol} Toplam Alış`}</span>
          <ArrowUpRight size={16} className="text-bull" />
        </div>
        <div className="stat-value num-mono text-bull">{formatTL(buyTL)}</div>
        <div className="stat-subtext num-mono">
          Hacim: {formatLots(buyVol).replace('+', '')}
        </div>
      </div>

      {/* 2. Total Sell Turnover */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>{isXU030 ? 'BIST 30 Toplam Satış' : `${selectedSymbol} Toplam Satış`}</span>
          <ArrowDownRight size={16} className="text-bear" />
        </div>
        <div className="stat-value num-mono text-bear">{formatTL(sellTL)}</div>
        <div className="stat-subtext num-mono">
          Hacim: {formatLots(sellVol).replace('+', '')}
        </div>
      </div>

      {/* 3. Net Position (Akış veya Kalan Miktar) */}
      <div className={`stat-card glass-card ${isNetBuyer ? 'bull' : 'bear'}`}>
        <div className="stat-header">
          <span>{isXU030 ? 'Net Pozisyon (Parasal Akış)' : `${selectedSymbol} Kalan Net Miktar`}</span>
          <span
            className={`status-pill ${isNetBuyer ? 'bg-bull-badge' : 'bg-bear-badge'}`}
            style={{ padding: '2px 8px', fontSize: '0.7rem' }}
          >
            {isNetBuyer ? 'NET ALICI' : 'NET SATICI'}
          </span>
        </div>
        <div className={`stat-value num-mono ${isNetBuyer ? 'text-bull' : 'text-bear'}`}>
          {isXU030 ? (
            // For XU030: Show monetary amount, e.g. "+3,00 Milyon ₺ Net Alıcı"
            `${netFlowTL >= 0 ? '+' : ''}${formatTL(netFlowTL)}`
          ) : (
            // For specific stock: Show remaining lot quantity, e.g. "+6.366.015 Lot"
            formatLots(netVolume)
          )}
        </div>
        <div className="stat-subtext num-mono">
          {isXU030 ? (
            `${selectedBroker} Tüm Hisseler: ${formatTL(Math.abs(netFlowTL))} Net ${isNetBuyer ? 'Alıcı' : 'Satıcı'}`
          ) : (
            `Net Tutar: ${formatTL(netFlowTL)} Net ${isNetBuyer ? 'Alıcı' : 'Satıcı'}`
          )}
        </div>
      </div>

      {/* 4. Matched Turnaround Volume */}
      <div className="stat-card glass-card">
        <div className="stat-header">
          <span>Eşleşen Gün İçi Hacim</span>
          <Repeat size={16} style={{ color: 'var(--brand-blue)' }} />
        </div>
        <div className="stat-value num-mono" style={{ color: '#93c5fd' }}>
          {formatLots(matchedVol).replace('+', '')}
        </div>
        <div className="stat-subtext">
          {isXU030 ? 'Tüm BIST 30 Al-Sat Turnaround' : `${selectedSymbol} Al-Sat Turnaround`}
        </div>
      </div>

      {/* 5. Intraday PnL & Realized PnL */}
      <div className={`stat-card glass-card ${isProfitable ? 'bull' : 'bear'}`}>
        <div className="stat-header">
          <span>{isXU030 ? 'BIST 30 Toplam Günlük Kâr/Zarar' : `${selectedSymbol} Gün İçi Kâr/Zarar`}</span>
          {isProfitable ? (
            <TrendingUp size={16} className="text-bull" />
          ) : (
            <TrendingDown size={16} className="text-bear" />
          )}
        </div>
        <div className={`stat-value num-mono ${isProfitable ? 'text-bull' : 'text-bear'}`}>
          {isXU030
            ? `${trueDailyPnL >= 0 ? '+' : ''}${formatTL(trueDailyPnL)}`
            : `${intraPnL >= 0 ? '+' : ''}${formatTL(intraPnL)}`}
        </div>
        <div className="stat-subtext" style={{ display: 'flex', gap: '8px', fontSize: '0.72rem' }}>
          {isXU030 ? (
            <>
              <span title="Tüm hisselerde gün içinde alınan ve aynı gün satılan lotlardan kâr">
                Gün İçi: <b className={intraPnL >= 0 ? 'text-bull' : 'text-bear'}>{formatTL(intraPnL)}</b>
              </span>
              <span title="Önceki günlerden taşınan lotların kapanış kârı">
                Carry: <b className={carryPnL >= 0 ? 'text-bull' : 'text-bear'}>{formatTL(carryPnL)}</b>
              </span>
            </>
          ) : (
            <>
              <span title="Hissedeki gün içi alım satımlardan doğan net kâr/zarar">
                Gün İçi: <b className={intraPnL >= 0 ? 'text-bull' : 'text-bear'}>{formatTL(intraPnL)}</b>
              </span>
              <span title="Seçilen hisseden gün sonu kalan net pozisyon">
                Kalan: <b style={{ color: isNetBuyer ? 'var(--bull-green)' : 'var(--bear-red)' }}>{formatLots(netVolume)}</b>
              </span>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
