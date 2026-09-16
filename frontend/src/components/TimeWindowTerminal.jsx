import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Clock,
  Calendar,
  Search,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Sliders,
  DollarSign,
  ShieldAlert,
  Layers,
  Sparkles,
  ChevronRight,
  RefreshCw,
} from 'lucide-react';

function formatTL(num) {
  if (num === null || num === undefined || isNaN(num)) return '—';
  const abs = Math.abs(num);
  if (abs >= 1_000_000_000) {
    return (num / 1_000_000_000).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' Milyar ₺';
  } else if (abs >= 1_000_000) {
    return (num / 1_000_000).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' Milyon ₺';
  } else {
    return num.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' ₺';
  }
}

function formatLot(num) {
  if (num === null || num === undefined || isNaN(num)) return '—';
  const abs = Math.abs(num);
  if (abs >= 1_000_000) {
    return (num / 1_000_000).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' M Lot';
  } else if (abs >= 1_000) {
    return (num / 1_000).toLocaleString('tr-TR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + ' K Lot';
  } else {
    return Math.round(num).toLocaleString('tr-TR') + ' Lot';
  }
}

function formatPrice(num) {
  if (num === null || num === undefined || isNaN(num)) return '—';
  return num.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' ₺';
}

function safeFixed(num, digits = 2, fallback = '—') {
  if (num === null || num === undefined) return fallback;
  const parsed = typeof num === 'number' ? num : parseFloat(num);
  if (isNaN(parsed)) return fallback;
  return parsed.toFixed(digits);
}

export default function TimeWindowTerminal({ instruments, brokers, dates, fetchGraphQL }) {
  // Filters
  const [selectedSymbol, setSelectedSymbol] = useState('AKBNK');
  const [selectedBroker, setSelectedBroker] = useState('MLB');
  const [timeframe, setTimeframe] = useState('5m');
  const [startDatetime, setStartDatetime] = useState('2026-09-14T09:55');
  const [endDatetime, setEndDatetime] = useState('2026-09-14T18:08');

  // Data & UI states
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hoveredCandle, setHoveredCandle] = useState(null);
  const [tableSearch, setTableSearch] = useState('');

  // Synchronize initial date if dates list arrives
  useEffect(() => {
    if (dates && dates.length > 0 && !dates.includes(startDatetime.slice(0, 10))) {
      const d = dates[0];
      setStartDatetime(`${d}T09:55`);
      setEndDatetime(`${d}T18:08`);
    }
  }, [dates]);

  // Load Window Analysis from GraphQL
  const loadWindowAnalysis = useCallback(async () => {
    if (!selectedSymbol) return;
    setLoading(true);
    setError(null);

    const query = `
      query GetTimeWindow(
        $symbol: String!
        $brokerId: String!
        $startDatetime: String!
        $endDatetime: String!
        $timeframe: String!
      ) {
        timeWindowAnalysis(
          symbol: $symbol
          brokerId: $brokerId
          startDatetime: $startDatetime
          endDatetime: $endDatetime
          timeframe: $timeframe
        ) {
          symbol
          symbolName
          brokerId
          brokerName
          startDatetime
          endDatetime
          timeframe
          windowOpenPrice
          windowClosePrice
          priceChangeTl
          priceChangePct
          windowHighPrice
          windowLowPrice
          priceRangePct
          totalVolume
          totalTurnoverTl
          totalTradesCount
          brokerBuyVolume
          brokerBuyTurnoverTl
          brokerBuyVwap
          brokerSellVolume
          brokerSellTurnoverTl
          brokerSellVwap
          brokerNetVolume
          brokerNetFlowTl
          brokerRealizedPnlTl
          brokerMarketSharePct
          openingAuction {
            auctionType
            matchTime
            matchPrice
            totalVolume
            totalTurnoverTl
            brokerBuyVolume
            brokerSellVolume
            brokerNetVolume
            brokerNetFlowTl
            brokerSharePct
          }
          closingAuction {
            auctionType
            matchTime
            matchPrice
            totalVolume
            totalTurnoverTl
            brokerBuyVolume
            brokerSellVolume
            brokerNetVolume
            brokerNetFlowTl
            brokerSharePct
          }
          candles {
            bucketStart
            bucketEnd
            open
            high
            low
            close
            volume
            turnoverTl
            vwap
            tradesCount
            brokerBuyVolume
            brokerSellVolume
            brokerNetVolume
            brokerNetFlowTl
            brokerRealizedPnlTl
            brokerSharePct
          }
        }
      }
    `;

    try {
      const res = await fetchGraphQL(query, {
        symbol: selectedSymbol,
        brokerId: selectedBroker,
        startDatetime: startDatetime.replace('T', ' ') + (startDatetime.length === 16 ? ':00' : ''),
        endDatetime: endDatetime.replace('T', ' ') + (endDatetime.length === 16 ? ':00' : ''),
        timeframe,
      });
      setData(res.timeWindowAnalysis || null);
    } catch (err) {
      console.error('Time window analysis fetch error:', err);
      setError(err.message || 'Veri alınırken hata oluştu.');
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol, selectedBroker, startDatetime, endDatetime, timeframe, fetchGraphQL]);

  // Initial fetch
  useEffect(() => {
    loadWindowAnalysis();
  }, [loadWindowAnalysis]);

  // Preset quick hours on active date
  const applyPresetTime = (startH, endH) => {
    const activeDate = startDatetime.slice(0, 10);
    setStartDatetime(`${activeDate}T${startH}`);
    setEndDatetime(`${activeDate}T${endH}`);
  };

  // Filtered candle table rows
  const filteredCandles = useMemo(() => {
    if (!data || !data.candles) return [];
    if (!tableSearch.trim()) return data.candles;
    const term = tableSearch.trim().toLowerCase();
    return data.candles.filter((c) => c.bucketStart.toLowerCase().includes(term));
  }, [data, tableSearch]);

  // SVG Chart Config
  const chartConfig = useMemo(() => {
    if (!data || !data.candles || data.candles.length === 0) return null;
    const candles = data.candles;
    const n = candles.length;

    let minPrice = Infinity;
    let maxPrice = -Infinity;
    let maxNetFlowAbs = 1;

    candles.forEach((c) => {
      if (c.low < minPrice) minPrice = c.low;
      if (c.high > maxPrice) maxPrice = c.high;
      const absFlow = Math.abs(c.brokerNetFlowTl);
      if (absFlow > maxNetFlowAbs) maxNetFlowAbs = absFlow;
    });

    if (minPrice === Infinity) minPrice = 1.0;
    if (maxPrice === -Infinity) maxPrice = 2.0;
    if (minPrice === maxPrice) {
      minPrice -= 1.0;
      maxPrice += 1.0;
    }

    const pricePadding = (maxPrice - minPrice) * 0.12 || 1.0;
    const domainMin = Math.max(0.01, minPrice - pricePadding);
    const domainMax = maxPrice + pricePadding;
    const domainSpan = domainMax - domainMin || 1.0;

    const width = 1000;
    const height = 380;
    const margin = { top: 25, right: 70, bottom: 90, left: 65 };
    const innerWidth = width - margin.left - margin.right;
    const priceHeight = height - margin.top - margin.bottom - 70; // top area for candles
    const flowHeight = 60; // bottom sub-chart for broker flow
    const flowTop = height - margin.bottom - flowHeight;

    const scaleX = (idx) => margin.left + (idx / Math.max(1, n - 1)) * innerWidth;
    const candleWidth = Math.max(2, Math.min(18, (innerWidth / n) * 0.7));

    const scalePriceY = (val) => margin.top + priceHeight - ((val - domainMin) / domainSpan) * priceHeight;
    const flowZeroY = flowTop + flowHeight / 2;
    const scaleFlowHeight = (flow) => (Math.abs(flow) / maxNetFlowAbs) * (flowHeight / 2 - 4);

    return {
      width,
      height,
      margin,
      scaleX,
      candleWidth,
      scalePriceY,
      domainMin,
      domainMax,
      flowTop,
      flowZeroY,
      scaleFlowHeight,
      candles,
    };
  }, [data]);

  const isPositiveChange = data && data.priceChangePct >= 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* 1. Control & Filter Panel */}
      <div className="glass-card" style={{ padding: '22px 26px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '16px',
            marginBottom: '18px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '10px',
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--brand-blue)',
              }}
            >
              <Clock size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#fff', margin: 0 }}>
                Özel Tarih &amp; Saat Aralığı Analiz Terminali
              </h2>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', margin: '2px 0 0 0' }}>
                İstediğiniz başlangıç ve bitiş saatini seçerek açılış seansı (09:55), seans içi kırılım ve kapanış eşleşmesi (18:05) hareketlerini inceleyin.
              </p>
            </div>
          </div>

          {/* Quick Preset Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: '600', marginRight: '4px' }}>
              Hızlı Seans:
            </span>
            <button
              type="button"
              onClick={() => applyPresetTime('09:55', '18:08')}
              className="glass-card"
              style={{ padding: '6px 12px', fontSize: '0.78rem', fontWeight: '600', cursor: 'pointer' }}
            >
              Tam Gün (09:55 - 18:08)
            </button>
            <button
              type="button"
              onClick={() => applyPresetTime('09:55', '10:30')}
              className="glass-card"
              style={{ padding: '6px 12px', fontSize: '0.78rem', fontWeight: '600', cursor: 'pointer' }}
            >
              Açılış Seansı (09:55 - 10:30)
            </button>
            <button
              type="button"
              onClick={() => applyPresetTime('11:30', '14:30')}
              className="glass-card"
              style={{ padding: '6px 12px', fontSize: '0.78rem', fontWeight: '600', cursor: 'pointer' }}
            >
              Öğle Seansı (11:30 - 14:30)
            </button>
            <button
              type="button"
              onClick={() => applyPresetTime('14:30', '18:00')}
              className="glass-card"
              style={{ padding: '6px 12px', fontSize: '0.78rem', fontWeight: '600', cursor: 'pointer' }}
            >
              Öğleden Sonra (14:30 - 18:00)
            </button>
            <button
              type="button"
              onClick={() => applyPresetTime('18:00', '18:08')}
              className="glass-card"
              style={{
                padding: '6px 12px',
                fontSize: '0.78rem',
                fontWeight: '600',
                cursor: 'pointer',
                borderColor: 'var(--accent-purple)',
                color: 'var(--accent-purple)',
              }}
            >
              Kapanış Eşleşmesi (18:00 - 18:08)
            </button>
          </div>
        </div>

        {/* Form Inputs Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '14px',
            alignItems: 'end',
          }}
        >
          {/* Symbol Select */}
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '6px' }}>
              Hisse / Endeks
            </label>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 14px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: '#fff',
                fontSize: '0.9rem',
                fontWeight: '600',
                outline: 'none',
              }}
            >
              <option value="XU030" style={{ backgroundColor: '#111827', color: '#60a5fa', fontWeight: '700' }}>
                XU030 — BIST 30 Endeksi (Konsolide)
              </option>
              {instruments
                .filter((i) => i.symbol !== 'XU030')
                .map((inst) => (
                  <option key={inst.symbol} value={inst.symbol} style={{ backgroundColor: '#111827', color: '#fff' }}>
                    {inst.symbol} — {inst.name}
                  </option>
                ))}
            </select>
          </div>

          {/* Broker Select */}
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '6px' }}>
              Aracı Kurum
            </label>
            <select
              value={selectedBroker}
              onChange={(e) => setSelectedBroker(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 14px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: '#fff',
                fontSize: '0.9rem',
                fontWeight: '600',
                outline: 'none',
              }}
            >
              {brokers.map((brk) => (
                <option key={brk.brokerId} value={brk.brokerId} style={{ backgroundColor: '#111827', color: '#fff' }}>
                  {brk.brokerId} — {brk.brokerName}
                </option>
              ))}
            </select>
          </div>

          {/* Start Datetime */}
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '6px' }}>
              Başlangıç (Tarih &amp; Saat)
            </label>
            <input
              type="datetime-local"
              step="60"
              value={startDatetime}
              onChange={(e) => setStartDatetime(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 12px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: '#fff',
                fontSize: '0.85rem',
                fontFamily: 'var(--font-mono)',
                outline: 'none',
              }}
            />
          </div>

          {/* End Datetime */}
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '6px' }}>
              Bitiş (Tarih &amp; Saat)
            </label>
            <input
              type="datetime-local"
              step="60"
              value={endDatetime}
              onChange={(e) => setEndDatetime(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 12px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: '#fff',
                fontSize: '0.85rem',
                fontFamily: 'var(--font-mono)',
                outline: 'none',
              }}
            />
          </div>

          {/* Timeframe Select */}
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '6px' }}>
              Mum Periyodu
            </label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 14px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: '#fff',
                fontSize: '0.9rem',
                fontWeight: '600',
                outline: 'none',
              }}
            >
              <option value="1m" style={{ backgroundColor: '#111827', color: '#fff' }}>1 Dakikalık</option>
              <option value="5m" style={{ backgroundColor: '#111827', color: '#fff' }}>5 Dakikalık</option>
              <option value="15m" style={{ backgroundColor: '#111827', color: '#fff' }}>15 Dakikalık</option>
              <option value="30m" style={{ backgroundColor: '#111827', color: '#fff' }}>30 Dakikalık</option>
              <option value="60m" style={{ backgroundColor: '#111827', color: '#fff' }}>60 Dakikalık (1 Saat)</option>
            </select>
          </div>

          {/* Submit Button */}
          <div>
            <button
              type="button"
              onClick={loadWindowAnalysis}
              disabled={loading}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '11px 18px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--brand-blue)',
                color: '#fff',
                fontWeight: '700',
                fontSize: '0.9rem',
                border: 'none',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.7 : 1,
                boxShadow: '0 4px 14px rgba(59, 130, 246, 0.35)',
                transition: 'all 0.15s ease',
              }}
            >
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
              <span>{loading ? 'Yükleniyor...' : 'Pencereyi Analiz Et'}</span>
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div
          className="glass-card"
          style={{
            padding: '14px 18px',
            borderColor: 'var(--bear-red)',
            color: 'var(--bear-red)',
            backgroundColor: 'var(--bear-red-bg)',
            fontSize: '0.88rem',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <ShieldAlert size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* 2. Executive KPI Cards: Price Movement & Auctions */}
      {data && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '16px',
          }}
        >
          {/* Card 1: Kaçtan Kaça Gitti */}
          <div
            className="glass-card"
            style={{
              padding: '18px 20px',
              borderLeft: `4px solid ${isPositiveChange ? 'var(--bull-green)' : 'var(--bear-red)'}`,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Kaçtan Kaça Gitti (Pencere Hareketi)
              </span>
              {isPositiveChange ? (
                <TrendingUp size={18} style={{ color: 'var(--bull-green)' }} />
              ) : (
                <TrendingDown size={18} style={{ color: 'var(--bear-red)' }} />
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
              <span className="num-mono" style={{ fontSize: '1.2rem', color: 'var(--text-muted)' }}>
                {formatPrice(data.windowOpenPrice)}
              </span>
              <ArrowRight size={16} style={{ color: 'var(--text-muted)' }} />
              <span className="num-mono" style={{ fontSize: '1.5rem', fontWeight: '800', color: '#fff' }}>
                {formatPrice(data.windowClosePrice)}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '6px' }}>
              <span
                className={`num-mono ${isPositiveChange ? 'text-bull' : 'text-bear'}`}
                style={{ fontSize: '1.05rem', fontWeight: '700' }}
              >
                {isPositiveChange ? '+' : ''}{safeFixed(data.priceChangeTl, 2)} ₺ ({isPositiveChange ? '+' : ''}{safeFixed(data.priceChangePct, 2)}%)
              </span>
            </div>
          </div>

          {/* Card 2: Dalgalanma Marjı & Hacim */}
          <div className="glass-card" style={{ padding: '18px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Pencere Oynaklığı (High / Low)
              </span>
              <Activity size={18} style={{ color: 'var(--brand-blue)' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
              <span className="num-mono text-bull" style={{ fontSize: '1.25rem', fontWeight: '700' }}>
                {formatPrice(data.windowHighPrice)}
              </span>
              <span style={{ color: 'var(--text-muted)' }}>/</span>
              <span className="num-mono text-bear" style={{ fontSize: '1.25rem', fontWeight: '700' }}>
                {formatPrice(data.windowLowPrice)}
              </span>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px' }}>
              Dalgalanma Marjı:{' '}
              <strong style={{ color: '#fff' }}>%{safeFixed(data.priceRangePct, 2)}</strong> | Hacim:{' '}
              <strong style={{ color: '#fff' }}>{formatTL(data.totalTurnoverTl)}</strong>
            </div>
          </div>

          {/* Card 3: Açılış Seansı Eşleşmesi (09:55) */}
          <div className="glass-card" style={{ padding: '18px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Açılış Seansı (09:55 Eşleşmesi)
              </span>
              <Sparkles size={18} style={{ color: 'var(--accent-purple)' }} />
            </div>
            {data.openingAuction ? (
              <div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                  <span className="num-mono" style={{ fontSize: '1.4rem', fontWeight: '800', color: '#fff' }}>
                    {formatPrice(data.openingAuction.matchPrice)}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    ({data.openingAuction.matchTime.slice(11)})
                  </span>
                </div>
                <div style={{ fontSize: '0.8rem', marginTop: '6px' }}>
                  {data.brokerId} Açılış Akışı:{' '}
                  <strong className={data.openingAuction.brokerNetFlowTl >= 0 ? 'text-bull' : 'text-bear'}>
                    {data.openingAuction.brokerNetFlowTl >= 0 ? '+' : ''}{formatTL(data.openingAuction.brokerNetFlowTl)}
                  </strong>
                </div>
                <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  Açılış Hacmi: {formatTL(data.openingAuction.totalTurnoverTl)} ({formatLot(data.openingAuction.totalVolume)})
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '12px' }}>
                Pencere açılış seansını (09:55) içermiyor.
              </div>
            )}
          </div>

          {/* Card 4: Kapanış Seansı Eşleşmesi (18:05) */}
          <div className="glass-card" style={{ padding: '18px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Kapanış Seansı (18:05 Eşleşmesi)
              </span>
              <Layers size={18} style={{ color: 'var(--accent-purple)' }} />
            </div>
            {data.closingAuction ? (
              <div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                  <span className="num-mono" style={{ fontSize: '1.4rem', fontWeight: '800', color: '#fff' }}>
                    {formatPrice(data.closingAuction.matchPrice)}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    ({data.closingAuction.matchTime.slice(11)})
                  </span>
                </div>
                <div style={{ fontSize: '0.8rem', marginTop: '6px' }}>
                  {data.brokerId} Kapanış Akışı:{' '}
                  <strong className={data.closingAuction.brokerNetFlowTl >= 0 ? 'text-bull' : 'text-bear'}>
                    {data.closingAuction.brokerNetFlowTl >= 0 ? '+' : ''}{formatTL(data.closingAuction.brokerNetFlowTl)}
                  </strong>
                </div>
                <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  Kapanış Hacmi: {formatTL(data.closingAuction.totalTurnoverTl)} ({formatLot(data.closingAuction.totalVolume)})
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '12px' }}>
                Pencere kapanış seansını (18:05) içermiyor.
              </div>
            )}
          </div>

          {/* Card 5: Kurum Pencere İçi Toplam Akış */}
          <div className="glass-card" style={{ padding: '18px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                {data.brokerId} Pencere Net Akışı
              </span>
              <DollarSign size={18} style={{ color: data.brokerNetFlowTl >= 0 ? 'var(--bull-green)' : 'var(--bear-red)' }} />
            </div>
            <div
              className={`num-mono ${data.brokerNetFlowTl >= 0 ? 'text-bull' : 'text-bear'}`}
              style={{ fontSize: '1.45rem', fontWeight: '800' }}
            >
              {data.brokerNetFlowTl >= 0 ? '+' : ''}{formatTL(data.brokerNetFlowTl)}
            </div>
            <div style={{ fontSize: '0.8rem', marginTop: '6px', color: 'var(--text-secondary)' }}>
              Net Pozisyon:{' '}
              <strong style={{ color: '#fff' }}>
                {data.brokerNetVolume >= 0 ? '+' : ''}{formatLot(data.brokerNetVolume)}
              </strong>
            </div>
            <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Piyasa Payı: %{safeFixed(data.brokerMarketSharePct, 2)} | Toplam İşlem: {data.totalTradesCount.toLocaleString('tr-TR')} adet
            </div>
          </div>
        </div>
      )}

      {/* 3. Interactive Candlestick + Broker Flow SVG Chart */}
      {chartConfig && (
        <div className="glass-card" style={{ padding: '24px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '16px',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: '700', color: '#fff', margin: 0 }}>
                {data.symbol} — {data.timeframe.toUpperCase()} Fiyat Hareketi &amp; {data.brokerId} Net Akışı
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '2px 0 0 0' }}>
                Üst panel: Fiyat mumları | Alt panel: {data.brokerId} net TL akışı (Yeşil: Net Alıcı, Kırmızı: Net Satıcı)
              </p>
            </div>

            {/* Hover Legend */}
            {hoveredCandle && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '14px',
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.8rem',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                <span>Saat: <strong style={{ color: '#fff' }}>{hoveredCandle.bucketStart.slice(11, 16)}</strong></span>
                <span>O: <strong>{formatPrice(hoveredCandle.open)}</strong></span>
                <span>H: <strong>{formatPrice(hoveredCandle.high)}</strong></span>
                <span>L: <strong>{formatPrice(hoveredCandle.low)}</strong></span>
                <span>C: <strong style={{ color: '#fff' }}>{formatPrice(hoveredCandle.close)}</strong></span>
                <span>
                  {data.brokerId} Akış:{' '}
                  <strong className={hoveredCandle.brokerNetFlowTl >= 0 ? 'text-bull' : 'text-bear'}>
                    {hoveredCandle.brokerNetFlowTl >= 0 ? '+' : ''}{formatTL(hoveredCandle.brokerNetFlowTl)}
                  </strong>
                </span>
              </div>
            )}
          </div>

          <div style={{ width: '100%', overflowX: 'auto' }}>
            <svg
              viewBox={`0 0 ${chartConfig.width} ${chartConfig.height}`}
              style={{ width: '100%', minWidth: '750px', height: 'auto', display: 'block' }}
            >
              {/* Grid lines */}
              {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
                const y = chartConfig.margin.top + ratio * (chartConfig.flowTop - chartConfig.margin.top - 20);
                const priceVal = chartConfig.domainMax - ratio * (chartConfig.domainMax - chartConfig.domainMin);
                return (
                  <g key={ratio}>
                    <line
                      x1={chartConfig.margin.left}
                      y1={y}
                      x2={chartConfig.width - chartConfig.margin.right}
                      y2={y}
                      stroke="rgba(255, 255, 255, 0.06)"
                      strokeDasharray="4 4"
                    />
                    <text
                      x={chartConfig.width - chartConfig.margin.right + 8}
                      y={y + 4}
                      fill="var(--text-muted)"
                      fontSize="11"
                      fontFamily="var(--font-mono)"
                    >
                      {formatPrice(priceVal)}
                    </text>
                  </g>
                );
              })}

              {/* Sub-chart separator */}
              <line
                x1={chartConfig.margin.left}
                y1={chartConfig.flowTop - 10}
                x2={chartConfig.width - chartConfig.margin.right}
                y2={chartConfig.flowTop - 10}
                stroke="rgba(255, 255, 255, 0.12)"
              />
              <text
                x={chartConfig.margin.left}
                y={chartConfig.flowTop - 14}
                fill="var(--text-muted)"
                fontSize="10"
                fontWeight="700"
                textTransform="uppercase"
              >
                {data.brokerId} Net TL Akış Histogramı
              </text>

              {/* Zero line for flow */}
              <line
                x1={chartConfig.margin.left}
                y1={chartConfig.flowZeroY}
                x2={chartConfig.width - chartConfig.margin.right}
                y2={chartConfig.flowZeroY}
                stroke="rgba(255, 255, 255, 0.2)"
              />

              {/* Render each candle and its flow bar */}
              {chartConfig.candles.map((c, idx) => {
                const x = chartConfig.scaleX(idx);
                const openY = chartConfig.scalePriceY(c.open);
                const closeY = chartConfig.scalePriceY(c.close);
                const highY = chartConfig.scalePriceY(c.high);
                const lowY = chartConfig.scalePriceY(c.low);

                const isBull = c.close >= c.open;
                const candleColor = isBull ? 'var(--bull-green)' : 'var(--bear-red)';

                const barY = Math.min(openY, closeY);
                const barH = Math.max(2, Math.abs(closeY - openY));

                // Flow bar
                const isNetBuy = c.brokerNetFlowTl >= 0;
                const flowH = chartConfig.scaleFlowHeight(c.brokerNetFlowTl);
                const flowBarY = isNetBuy ? chartConfig.flowZeroY - flowH : chartConfig.flowZeroY;

                return (
                  <g
                    key={c.bucketStart}
                    onMouseEnter={() => setHoveredCandle(c)}
                    onMouseLeave={() => setHoveredCandle(null)}
                    style={{ cursor: 'pointer' }}
                  >
                    {/* Candle wick */}
                    <line
                      x1={x}
                      y1={highY}
                      x2={x}
                      y2={lowY}
                      stroke={candleColor}
                      strokeWidth="1.5"
                    />

                    {/* Candle body */}
                    <rect
                      x={x - chartConfig.candleWidth / 2}
                      y={barY}
                      width={chartConfig.candleWidth}
                      height={barH}
                      fill={candleColor}
                      rx="1"
                    />

                    {/* Flow Histogram Bar */}
                    <rect
                      x={x - chartConfig.candleWidth / 2}
                      y={flowBarY}
                      width={chartConfig.candleWidth}
                      height={Math.max(1, flowH)}
                      fill={isNetBuy ? 'var(--bull-green)' : 'var(--bear-red)'}
                      opacity={hoveredCandle && hoveredCandle.bucketStart === c.bucketStart ? 1 : 0.75}
                      rx="1"
                    />

                    {/* X-axis time label for milestone candles */}
                    {(idx === 0 || idx === Math.floor(chartConfig.candles.length / 2) || idx === chartConfig.candles.length - 1) && (
                      <text
                        x={x}
                        y={chartConfig.height - 20}
                        textAnchor="middle"
                        fill="var(--text-muted)"
                        fontSize="11"
                        fontFamily="var(--font-mono)"
                      >
                        {c.bucketStart.slice(11, 16)}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      )}

      {/* 4. Detailed Candle & Execution Table */}
      {data && data.candles && data.candles.length > 0 && (
        <div className="glass-card" style={{ padding: '22px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '16px',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: '700', color: '#fff', margin: 0 }}>
                Zaman Kırılımı ve Kurum İşlemleri Tablosu ({data.candles.length} Dilim)
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '2px 0 0 0' }}>
                Seçilen aralıktaki her {data.timeframe} diliminin fiyat hareketleri, hacimleri ve {data.brokerId} kurumunun net alım/satımları
              </p>
            </div>

            <div style={{ position: 'relative', width: '220px' }}>
              <Search size={16} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Saat ara (örn: 10:00)..."
                value={tableSearch}
                onChange={(e) => setTableSearch(e.target.value)}
                style={{
                  width: '100%',
                  padding: '7px 10px 7px 32px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid var(--border-subtle)',
                  color: '#fff',
                  fontSize: '0.8rem',
                  outline: 'none',
                }}
              />
            </div>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  <th style={{ padding: '10px 12px' }}>Saat Dilimi</th>
                  <th style={{ padding: '10px 12px' }}>Açılış</th>
                  <th style={{ padding: '10px 12px' }}>Yüksek</th>
                  <th style={{ padding: '10px 12px' }}>Düşük</th>
                  <th style={{ padding: '10px 12px' }}>Kapanış</th>
                  <th style={{ padding: '10px 12px' }}>Değişim</th>
                  <th style={{ padding: '10px 12px' }}>Toplam Hacim</th>
                  <th style={{ padding: '10px 12px' }}>{data.brokerId} Alış</th>
                  <th style={{ padding: '10px 12px' }}>{data.brokerId} Satış</th>
                  <th style={{ padding: '10px 12px' }}>{data.brokerId} Net TL</th>
                  <th style={{ padding: '10px 12px' }}>Kurum Payı</th>
                </tr>
              </thead>
              <tbody>
                {filteredCandles.map((c) => {
                  const retPct = c.open > 0 ? ((c.close - c.open) / c.open) * 100.0 : 0.0;
                  const isPos = retPct >= 0;
                  const isNetBuy = c.brokerNetFlowTl >= 0;

                  return (
                    <tr
                      key={c.bucketStart}
                      style={{
                        borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
                        fontSize: '0.82rem',
                        transition: 'background 0.15s ease',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      <td style={{ padding: '8px 12px', fontWeight: '600' }} className="num-mono">
                        {c.bucketStart.slice(11, 16)} - {c.bucketEnd ? c.bucketEnd.slice(11, 16) : ''}
                      </td>
                      <td style={{ padding: '8px 12px' }} className="num-mono">{formatPrice(c.open)}</td>
                      <td style={{ padding: '8px 12px', color: 'var(--bull-green)' }} className="num-mono">{formatPrice(c.high)}</td>
                      <td style={{ padding: '8px 12px', color: 'var(--bear-red)' }} className="num-mono">{formatPrice(c.low)}</td>
                      <td style={{ padding: '8px 12px', fontWeight: '700', color: '#fff' }} className="num-mono">{formatPrice(c.close)}</td>
                      <td style={{ padding: '8px 12px' }} className="num-mono">
                        <span
                          style={{
                            padding: '2px 6px',
                            borderRadius: 'var(--radius-sm)',
                            fontWeight: '700',
                            fontSize: '0.75rem',
                            backgroundColor: isPos ? 'var(--bull-green-bg)' : 'var(--bear-red-bg)',
                            color: isPos ? 'var(--bull-green)' : 'var(--bear-red)',
                          }}
                        >
                          {isPos ? '+' : ''}{safeFixed(retPct, 2)}%
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px' }} className="num-mono">{formatTL(c.turnoverTl)}</td>
                      <td style={{ padding: '8px 12px' }} className="num-mono">{formatLot(c.brokerBuyVolume)}</td>
                      <td style={{ padding: '8px 12px' }} className="num-mono">{formatLot(c.brokerSellVolume)}</td>
                      <td style={{ padding: '8px 12px', fontWeight: '700' }} className="num-mono">
                        <span className={isNetBuy ? 'text-bull' : 'text-bear'}>
                          {isNetBuy ? '+' : ''}{formatTL(c.brokerNetFlowTl)}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }} className="num-mono">
                        %{safeFixed(c.brokerSharePct, 1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
