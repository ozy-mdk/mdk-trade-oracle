import React from 'react';

function formatTL(num) {
  if (num === null || num === undefined) return '—';
  return num.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' ₺';
}

function formatVol(num) {
  if (num === null || num === undefined) return '—';
  return Math.round(num).toLocaleString('tr-TR');
}

export default function CandleDataTable({ candles, brokerId }) {
  if (!candles || candles.length === 0) {
    return null;
  }

  return (
    <div className="table-card glass-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: '600' }}>Mum &amp; Kurum Akış Detay Tablosu</h3>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Toplam: {candles.length} Mum</span>
      </div>

      <div className="table-responsive">
        <table className="data-table num-mono">
          <thead>
            <tr>
              <th>Saat</th>
              <th>Açılış</th>
              <th>Yüksek</th>
              <th>Düşük</th>
              <th>Kapanış</th>
              <th>Piyasa VWAP</th>
              <th>Piyasa Hacmi</th>
              <th>{brokerId} Alış (TL)</th>
              <th>{brokerId} Satış (TL)</th>
              <th>{brokerId} Net Akış</th>
              <th>{brokerId} Kâr/Zarar</th>
            </tr>
          </thead>
          <tbody>
            {candles.map((c, i) => {
              const isProfit = c.realizedPnlTl >= 0;
              const isNetBuy = c.netFlowTl >= 0;
              const timeOnly = c.bucketStart.includes(' ') ? c.bucketStart.split(' ')[1] : c.bucketStart;

              return (
                <tr key={i}>
                  <td style={{ color: 'var(--text-secondary)' }}>{timeOnly}</td>
                  <td>{c.open.toFixed(2)}</td>
                  <td className="text-bull">{c.high.toFixed(2)}</td>
                  <td className="text-bear">{c.low.toFixed(2)}</td>
                  <td style={{ fontWeight: '600' }}>{c.close.toFixed(2)}</td>
                  <td style={{ color: '#93c5fd' }}>{c.vwap.toFixed(2)}</td>
                  <td>{formatVol(c.volume)}</td>
                  <td className="text-bull">{formatTL(c.buyTurnoverTl)}</td>
                  <td className="text-bear">{formatTL(c.sellTurnoverTl)}</td>
                  <td className={isNetBuy ? 'text-bull' : 'text-bear'} style={{ fontWeight: '600' }}>
                    {isNetBuy ? '+' : ''}
                    {formatTL(c.netFlowTl)}
                  </td>
                  <td className={isProfit ? 'text-bull' : 'text-bear'} style={{ fontWeight: '700' }}>
                    {c.realizedPnlTl !== 0 ? (isProfit ? '+' : '') + formatTL(c.realizedPnlTl) : '0,00 ₺'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
