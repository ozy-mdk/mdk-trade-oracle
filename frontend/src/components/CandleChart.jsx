import React, { useState } from 'react';

export default function CandleChart({ candles, symbol, brokerId, timeframe }) {
  const [hoveredCandle, setHoveredCandle] = useState(null);

  if (!candles || candles.length === 0) {
    return (
      <div className="chart-card glass-card">
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
          Bu seçim için henüz mum verisi bulunmamaktadır.
        </div>
      </div>
    );
  }

  // Determine bounds
  const prices = candles.flatMap((c) => [c.high, c.low]);
  const minPrice = Math.min(...prices) * 0.998;
  const maxPrice = Math.max(...prices) * 1.002;
  const priceRange = maxPrice - minPrice || 1;

  const maxVolume = Math.max(...candles.map((c) => c.volume)) || 1;

  // Chart Dimensions
  const width = 1000;
  const height = 360;
  const chartPaddingTop = 20;
  const chartHeightPrice = 240;
  const chartHeightVolume = 80;
  const volumeTop = chartPaddingTop + chartHeightPrice + 15;

  const barWidth = Math.max(3, Math.min(24, (width - 60) / candles.length - 4));
  const stepX = (width - 60) / candles.length;

  const scaleY = (p) => chartPaddingTop + (1 - (p - minPrice) / priceRange) * chartHeightPrice;
  const scaleVol = (v) => (v / maxVolume) * chartHeightVolume;

  return (
    <div className="chart-card glass-card">
      <div className="chart-header">
        <div className="chart-title-group">
          <span className="symbol-badge">{symbol}</span>
          <span className="sector-badge">{timeframe.toUpperCase()}</span>
          <span className="sector-badge" style={{ color: 'var(--brand-blue)' }}>
            Kurum: {brokerId}
          </span>
        </div>

        <div className="legend-group">
          <div className="legend-item">
            <div className="legend-indicator" style={{ background: 'var(--bull-green)' }}></div>
            <span>Yükseliş (Bull)</span>
          </div>
          <div className="legend-item">
            <div className="legend-indicator" style={{ background: 'var(--bear-red)' }}></div>
            <span>Düşüş (Bear)</span>
          </div>
          <div className="legend-item">
            <div className="legend-indicator" style={{ background: 'rgba(59, 130, 246, 0.6)' }}></div>
            <span>{brokerId} Hacmi</span>
          </div>
        </div>
      </div>

      <div className="chart-viewport" onMouseLeave={() => setHoveredCandle(null)}>
        {/* Dynamic Tooltip */}
        {hoveredCandle && (
          <div className="chart-tooltip num-mono">
            <div>
              <div style={{ color: 'var(--text-muted)' }}>ZAMAN</div>
              <div>{hoveredCandle.bucketStart.split(' ')[1] || hoveredCandle.bucketStart}</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)' }}>A / Y / D / K</div>
              <div>
                {hoveredCandle.open.toFixed(2)} / {hoveredCandle.high.toFixed(2)} / {hoveredCandle.low.toFixed(2)} / {hoveredCandle.close.toFixed(2)}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)' }}>VWAP</div>
              <div>{hoveredCandle.vwap.toFixed(2)} ₺</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)' }}>{brokerId} NET AKIŞ</div>
              <div className={hoveredCandle.netFlowTl >= 0 ? 'text-bull' : 'text-bear'}>
                {hoveredCandle.netFlowTl >= 0 ? '+' : ''}
                {(hoveredCandle.netFlowTl / 1_000_000).toFixed(2)}M ₺
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-muted)' }}>{brokerId} KÂR/ZARAR</div>
              <div className={hoveredCandle.realizedPnlTl >= 0 ? 'text-bull' : 'text-bear'}>
                {hoveredCandle.realizedPnlTl >= 0 ? '+' : ''}
                {hoveredCandle.realizedPnlTl.toLocaleString('tr-TR', { maximumFractionDigits: 0 })} ₺
              </div>
            </div>
          </div>
        )}

        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '100%' }}>
          <defs>
            <linearGradient id="volGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(59, 130, 246, 0.4)" />
              <stop offset="100%" stopColor="rgba(59, 130, 246, 0.05)" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
            const y = chartPaddingTop + pct * chartHeightPrice;
            const priceVal = maxPrice - pct * priceRange;
            return (
              <g key={i}>
                <line x1="30" y1={y} x2={width - 20} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                <text x="5" y={y + 3} fill="#6b7280" fontSize="9" fontFamily="var(--font-mono)">
                  {priceVal.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Candlesticks and Volume Bars */}
          {candles.map((c, i) => {
            const x = 30 + i * stepX + stepX / 2;
            const isBull = c.close >= c.open;
            const color = isBull ? 'var(--bull-green)' : 'var(--bear-red)';
            const yOpen = scaleY(c.open);
            const yClose = scaleY(c.close);
            const yHigh = scaleY(c.high);
            const yLow = scaleY(c.low);

            const candleTop = Math.min(yOpen, yClose);
            const candleHeight = Math.max(2, Math.abs(yClose - yOpen));

            const mktVolH = scaleVol(c.volume);
            const brkVolH = scaleVol(c.buyVolume + c.sellVolume);
            const volY = height - 10 - mktVolH;

            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredCandle(c)}
                style={{ cursor: 'pointer' }}
              >
                {/* Wick */}
                <line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={color} strokeWidth="1.2" />

                {/* Body */}
                <rect
                  x={x - barWidth / 2}
                  y={candleTop}
                  width={barWidth}
                  height={candleHeight}
                  fill={color}
                  rx="1"
                />

                {/* Market Volume Bar */}
                <rect
                  x={x - barWidth / 2}
                  y={volY}
                  width={barWidth}
                  height={mktVolH}
                  fill="rgba(255, 255, 255, 0.1)"
                  rx="1"
                />

                {/* Broker Volume Overlay */}
                <rect
                  x={x - barWidth / 2}
                  y={height - 10 - brkVolH}
                  width={barWidth}
                  height={brkVolH}
                  fill={isBull ? 'rgba(16, 185, 129, 0.4)' : 'rgba(244, 63, 94, 0.4)'}
                  rx="1"
                />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
