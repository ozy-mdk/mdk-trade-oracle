import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Calendar,
  Sliders,
  Sparkles,
  BarChart3,
  Search,
  CheckCircle2,
  XCircle,
  Target,
  Clock,
  ArrowRight,
  Layers,
  Repeat,
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

function formatPrice(num) {
  if (num === null || num === undefined || isNaN(num)) return '—';
  return num.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' ₺';
}

export default function EventStudyDashboard({ instruments, fetchGraphQL }) {
  // Filter state
  const [selectedSymbol, setSelectedSymbol] = useState('AKBNK');
  const [conditionType, setConditionType] = useState('DAILY_RETURN');
  const [directionMode, setDirectionMode] = useState('DOWN'); // 'DOWN', 'UP', 'RANGE'
  const [singleValue, setSingleValue] = useState('-9.0'); // e.g. -9.0 for 9% drop
  const [minValue, setMinValue] = useState('-10.0');
  const [maxValue, setMaxValue] = useState('-8.0');
  const [forwardDays, setForwardDays] = useState(5);
  const [startDate, setStartDate] = useState('2022-01-01');
  const [endDate, setEndDate] = useState('2026-09-14');
  const [limit, setLimit] = useState(100);

  // Return calculation mode: 'DAILY' (independent daily returns) vs 'CUMULATIVE' (from T close)
  const [returnMetricMode, setReturnMetricMode] = useState('DAILY');

  // Data state
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Table search & hover
  const [tableSearch, setTableSearch] = useState('');
  const [hoveredDay, setHoveredDay] = useState(null);

  // Synchronize threshold inputs when directionMode or conditionType changes
  const handleDirectionChange = (mode) => {
    setDirectionMode(mode);
    if (conditionType === 'DAILY_RETURN') {
      if (mode === 'DOWN') {
        setSingleValue('-9.0');
      } else if (mode === 'UP') {
        setSingleValue('3.0');
      } else {
        setMinValue('-10.0');
        setMaxValue('-8.0');
      }
    } else if (conditionType === 'BOFA_NET_FLOW') {
      if (mode === 'DOWN') {
        setSingleValue('-100000000');
      } else if (mode === 'UP') {
        setSingleValue('100000000');
      } else {
        setMinValue('50000000');
        setMaxValue('200000000');
      }
    }
  };

  // Condition type label & preset helpers
  const conditionMeta = useMemo(() => {
    switch (conditionType) {
      case 'DAILY_RETURN':
        return {
          label: 'Günlük Getiri (%)',
          unit: '%',
          placeholderSingle: directionMode === 'DOWN' ? 'Örn: -9.0 veya 9.0' : 'Örn: 3.0',
          presets:
            directionMode === 'DOWN'
              ? [
                  { label: '-3.0% ve Altı (Düşüş)', val: '-3.0' },
                  { label: '-5.0% ve Altı (Sert Düşüş)', val: '-5.0' },
                  { label: '-7.0% ve Altı (Derin Düşüş)', val: '-7.0' },
                  { label: '-9.0% ve Altı (Taban / Tabana Yakın)', val: '-9.0' },
                ]
              : directionMode === 'UP'
              ? [
                  { label: '+3.0% ve Üzeri', val: '3.0' },
                  { label: '+5.0% ve Üzeri (Ralli)', val: '5.0' },
                  { label: '+7.0% ve Üzeri (Güçlü Alış)', val: '7.0' },
                  { label: '+9.0% ve Üzeri (Tavan)', val: '9.0' },
                ]
              : [
                  { label: '-10% ile -8% Arası', min: '-10.0', max: '-8.0' },
                  { label: '+2% ile +4% Arası', min: '2.0', max: '4.0' },
                  { label: '+4% ile +7% Arası', min: '4.0', max: '7.0' },
                ],
        };
      case 'BOFA_NET_FLOW':
        return {
          label: 'BofA (MLB) Net Akışı (TL)',
          unit: '₺',
          placeholderSingle: directionMode === 'DOWN' ? 'Örn: -100000000' : 'Örn: 100000000',
          presets:
            directionMode === 'DOWN'
              ? [
                  { label: 'BofA -50M ₺ Altı Satış', val: '-50000000' },
                  { label: 'BofA -100M ₺ Altı Güçlü Çıkış', val: '-100000000' },
                  { label: 'BofA -250M ₺ Altı Agresif Çıkış', val: '-250000000' },
                ]
              : directionMode === 'UP'
              ? [
                  { label: 'BofA +50M ₺ Üzeri Giriş', val: '50000000' },
                  { label: 'BofA +100M ₺ Üzeri Güçlü Giriş', val: '100000000' },
                  { label: 'BofA +250M ₺ Üzeri Agresif Alış', val: '250000000' },
                ]
              : [
                  { label: '50M ile 150M ₺ Arası', min: '50000000', max: '150000000' },
                  { label: '150M ile 300M ₺ Arası', min: '150000000', max: '300000000' },
                ],
        };
      case 'PRICE_RANGE':
        return {
          label: 'Gün İçi Dalgalanma Marjı (%)',
          unit: '%',
          placeholderSingle: 'Örn: 4.0',
          presets: [
            { label: '%4.0 ve Üzeri Dalgalanma', val: '4.0' },
            { label: '%6.0 ve Üzeri Dalgalanma', val: '6.0' },
            { label: '%8.0 ve Üzeri Aşırı Oynaklık', val: '8.0' },
          ],
        };
      case 'VOLUME_SURGE':
        return {
          label: 'Toplam İşlem Hacmi (TL)',
          unit: '₺',
          placeholderSingle: 'Örn: 5000000000',
          presets: [
            { label: '5 Milyar ₺ Üzeri Hacim', val: '5000000000' },
            { label: '10 Milyar ₺ Üzeri Dev Hacim', val: '10000000000' },
          ],
        };
      default:
        return { label: 'Koşul', unit: '', presets: [] };
    }
  }, [conditionType, directionMode]);

  // Load Event Study data from GraphQL
  const loadEventStudy = useCallback(async () => {
    if (!selectedSymbol) return;
    setLoading(true);
    setError(null);

    let finalMin = null;
    let finalMax = null;
    let finalDir = directionMode;

    if (directionMode === 'DOWN') {
      const num = parseFloat(singleValue);
      if (!isNaN(num)) {
        // Automatically ensure negative threshold for drop
        finalMin = -Math.abs(num);
        finalMax = null;
      }
    } else if (directionMode === 'UP') {
      const num = parseFloat(singleValue);
      if (!isNaN(num)) {
        finalMin = Math.abs(num);
        finalMax = null;
      }
    } else {
      finalMin = minValue !== '' && !isNaN(parseFloat(minValue)) ? parseFloat(minValue) : null;
      finalMax = maxValue !== '' && !isNaN(parseFloat(maxValue)) ? parseFloat(maxValue) : null;
      finalDir = null;
    }

    try {
      const query = `
        query GetEventStudy(
          $symbol: String!
          $conditionType: String!
          $minValue: Float
          $maxValue: Float
          $direction: String
          $forwardDays: Int!
          $startDate: String
          $endDate: String
          $limit: Int
        ) {
          eventStudy(
            symbol: $symbol
            conditionType: $conditionType
            minValue: $minValue
            maxValue: $maxValue
            direction: $direction
            forwardDays: $forwardDays
            startDate: $startDate
            endDate: $endDate
            limit: $limit
          ) {
            symbol
            conditionType
            minValue
            maxValue
            direction
            forwardDays
            totalOccurrences
            horizonStats {
              dayOffset
              avgReturnPct
              winRatePct
              medianReturnPct
              maxGainPct
              maxLossPct
              cumulAvgReturnPct
              cumulWinRatePct
              sampleCount
            }
            occurrences {
              eventDate
              prevClosePrice
              openPrice
              closePrice
              priceChangeTl
              priceChangePct
              movementValue
              bofaNetFlowTl
              totalTurnoverTl
              forwardReturns {
                dayOffset
                date
                prevClosePrice
                closePrice
                returnPct
                dailyReturnPct
                cumulativeReturnPct
              }
            }
          }
        }
      `;

      const result = await fetchGraphQL(query, {
        symbol: selectedSymbol,
        conditionType,
        minValue: finalMin,
        maxValue: finalMax,
        direction: finalDir,
        forwardDays: parseInt(forwardDays, 10),
        startDate: startDate || null,
        endDate: endDate || null,
        limit: parseInt(limit, 10),
      });

      setData(result.eventStudy || null);
    } catch (err) {
      console.error('Event study fetch error:', err);
      setError(err.message || 'Veri yüklenemedi.');
    } finally {
      setLoading(false);
    }
  }, [
    selectedSymbol,
    conditionType,
    directionMode,
    singleValue,
    minValue,
    maxValue,
    forwardDays,
    startDate,
    endDate,
    limit,
    fetchGraphQL,
  ]);

  // Initial fetch on mount & whenever symbol/threshold changes
  useEffect(() => {
    loadEventStudy();
  }, [loadEventStudy]);

  // Target horizon stat (at Day N) based on selected return metric mode
  const targetStat = useMemo(() => {
    if (!data || !data.horizonStats || data.horizonStats.length === 0) return null;
    return data.horizonStats[data.horizonStats.length - 1];
  }, [data]);

  // Filtered occurrences for table
  const filteredOccurrences = useMemo(() => {
    if (!data || !data.occurrences) return [];
    if (!tableSearch.trim()) return data.occurrences;
    const term = tableSearch.trim().toLowerCase();
    return data.occurrences.filter((o) => o.eventDate.toLowerCase().includes(term));
  }, [data, tableSearch]);

  // SVG Chart Dimensions & Computations
  const chartConfig = useMemo(() => {
    if (!data || !data.horizonStats || data.horizonStats.length === 0) return null;

    const stats = data.horizonStats;
    const n = stats.length;
    const isDailyMode = returnMetricMode === 'DAILY';

    // Collect values based on active metric mode
    let minY = 0;
    let maxY = 0;

    stats.forEach((s) => {
      const val = isDailyMode ? s.avgReturnPct : (s.cumulAvgReturnPct || s.avgReturnPct);
      if (val < minY) minY = val;
      if (val > maxY) maxY = val;
      if (s.maxGainPct > maxY) maxY = s.maxGainPct;
      if (s.maxLossPct < minY) minY = s.maxLossPct;
    });

    const padding = Math.max(Math.abs(minY), Math.abs(maxY)) * 0.2 || 2.0;
    const domainMin = Math.min(-1.0, minY - padding);
    const domainMax = Math.max(1.0, maxY + padding);
    const domainSpan = domainMax - domainMin;

    const width = 960;
    const height = 340;
    const margin = { top: 30, right: 40, bottom: 45, left: 65 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const scaleX = (day) => margin.left + (day / n) * innerWidth;
    const scaleY = (val) => margin.top + innerHeight - ((val - domainMin) / domainSpan) * innerHeight;
    const zeroY = scaleY(0);

    // Build trajectory points
    const avgPoints = [{ day: 0, val: 0.0, x: scaleX(0), y: scaleY(0) }];
    stats.forEach((s) => {
      const activeVal = isDailyMode ? s.avgReturnPct : (s.cumulAvgReturnPct || s.avgReturnPct);
      const activeWin = isDailyMode ? s.winRatePct : (s.cumulWinRatePct || s.winRatePct);
      avgPoints.push({
        day: s.dayOffset,
        val: activeVal,
        dailyVal: s.avgReturnPct,
        cumulVal: s.cumulAvgReturnPct,
        winRate: activeWin,
        median: s.medianReturnPct,
        sampleCount: s.sampleCount,
        maxGain: s.maxGainPct,
        maxLoss: s.maxLossPct,
        x: scaleX(s.dayOffset),
        y: scaleY(activeVal),
      });
    });

    const avgPathD = avgPoints
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
      .join(' ');

    // Sample individual paths
    const sampleOccs = (data.occurrences || []).slice(0, 30);
    const samplePaths = sampleOccs.map((o) => {
      const pts = [{ x: scaleX(0), y: scaleY(0) }];
      o.forwardReturns.forEach((fr) => {
        const retVal = isDailyMode ? fr.dailyReturnPct : fr.cumulativeReturnPct;
        if (retVal !== null && retVal !== undefined) {
          pts.push({ x: scaleX(fr.dayOffset), y: scaleY(retVal) });
        }
      });
      return {
        date: o.eventDate,
        pathD: pts.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' '),
      };
    });

    return {
      width,
      height,
      margin,
      scaleX,
      scaleY,
      zeroY,
      domainMin,
      domainMax,
      avgPoints,
      avgPathD,
      samplePaths,
      stats,
    };
  }, [data, returnMetricMode]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* 1. Header Banner */}
      <div
        className="glass-card"
        style={{
          padding: '20px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
          borderLeft: '4px solid var(--bull-green)',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Sparkles size={22} style={{ color: 'var(--bull-green)' }} />
            <h2 style={{ fontSize: '1.25rem', fontWeight: '700', letterSpacing: '-0.01em' }}>
              Hisse Hareket &amp; İleri Getiri Analizi (Event Study Terminal)
            </h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            BIST hisselerinde belirli bir fiyat değişimi veya kurumsal akış gerçekleştiğinde, sonraki seanslarda hissenin{' '}
            <strong style={{ color: '#fff' }}>kaçtan kaça gittiği ve bir önceki günün kapanışına göre bağımsız günlük getirileri</strong>.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            type="button"
            onClick={loadEventStudy}
            disabled={loading}
            className="glass-card"
            style={{
              padding: '10px 20px',
              backgroundColor: 'var(--brand-blue)',
              borderColor: 'var(--brand-blue)',
              color: '#fff',
              fontWeight: '600',
              fontSize: '0.9rem',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              boxShadow: 'var(--shadow-glow)',
            }}
          >
            <Activity size={18} className={loading ? 'spin' : ''} />
            <span>{loading ? 'Hesaplanıyor...' : 'Analizi Çalıştır'}</span>
          </button>
        </div>
      </div>

      {/* 2. Advanced Interactive Filter Bar */}
      <div
        className="glass-card"
        style={{
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
        }}
      >
        {/* Direction Mode Selector */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: '600' }}>
              HAREKET YÖNÜ:
            </span>
            <div style={{ display: 'inline-flex', backgroundColor: 'var(--bg-surface)', padding: '3px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <button
                type="button"
                onClick={() => handleDirectionChange('DOWN')}
                style={{
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.82rem',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: directionMode === 'DOWN' ? 'var(--bear-red-bg)' : 'transparent',
                  color: directionMode === 'DOWN' ? 'var(--bear-red)' : 'var(--text-secondary)',
                  border: directionMode === 'DOWN' ? '1px solid rgba(244, 63, 94, 0.4)' : '1px solid transparent',
                  cursor: 'pointer',
                }}
              >
                <TrendingDown size={16} />
                <span>Düşüş (≤ Eşik)</span>
              </button>

              <button
                type="button"
                onClick={() => handleDirectionChange('UP')}
                style={{
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.82rem',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: directionMode === 'UP' ? 'var(--bull-green-bg)' : 'transparent',
                  color: directionMode === 'UP' ? 'var(--bull-green)' : 'var(--text-secondary)',
                  border: directionMode === 'UP' ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid transparent',
                  cursor: 'pointer',
                }}
              >
                <TrendingUp size={16} />
                <span>Yükseliş (≥ Eşik)</span>
              </button>

              <button
                type="button"
                onClick={() => handleDirectionChange('RANGE')}
                style={{
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.82rem',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: directionMode === 'RANGE' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                  color: directionMode === 'RANGE' ? 'var(--brand-blue)' : 'var(--text-secondary)',
                  border: directionMode === 'RANGE' ? '1px solid var(--brand-blue)' : '1px solid transparent',
                  cursor: 'pointer',
                }}
              >
                <Sliders size={16} />
                <span>Aralık [Min, Max]</span>
              </button>
            </div>
          </div>

          {/* Metric Calculation Mode Selector: Independent Daily vs Cumulative */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: '600' }}>
              GETİRİ HESABI:
            </span>
            <div style={{ display: 'inline-flex', backgroundColor: 'var(--bg-surface)', padding: '3px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <button
                type="button"
                onClick={() => setReturnMetricMode('DAILY')}
                style={{
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: returnMetricMode === 'DAILY' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                  color: returnMetricMode === 'DAILY' ? '#fff' : 'var(--text-secondary)',
                  border: returnMetricMode === 'DAILY' ? '1px solid var(--brand-blue)' : '1px solid transparent',
                  cursor: 'pointer',
                }}
                title="Her günün getirisi, ondan bir önceki günün kapanışına göre bağımsız olarak hesaplanır."
              >
                <Repeat size={14} style={{ color: 'var(--brand-blue)' }} />
                <span>Günlük Bağımsız Getiri (T-1'e Göre)</span>
              </button>

              <button
                type="button"
                onClick={() => setReturnMetricMode('CUMULATIVE')}
                style={{
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: returnMetricMode === 'CUMULATIVE' ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                  color: returnMetricMode === 'CUMULATIVE' ? '#fff' : 'var(--text-secondary)',
                  border: returnMetricMode === 'CUMULATIVE' ? '1px solid var(--accent-purple)' : '1px solid transparent',
                  cursor: 'pointer',
                }}
                title="Her günün getirisi, olay günü T kapanışına göre kümülatif olarak hesaplanır."
              >
                <Layers size={14} style={{ color: 'var(--accent-purple)' }} />
                <span>Kümülatif Getiri (T Kapanışına Göre)</span>
              </button>
            </div>
          </div>
        </div>

        {/* Row 1: Symbol, Condition Type, Threshold Input, Horizon Slider */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
            alignItems: 'flex-end',
          }}
        >
          {/* Stock Selector */}
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: '600' }}>
              HİSSE SENEDİ
            </label>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'var(--bg-surface)',
                color: '#fff',
                border: '1px solid var(--border-subtle)',
                fontSize: '0.9rem',
                fontWeight: '600',
              }}
            >
              {instruments.map((inst) => (
                <option key={inst.symbol} value={inst.symbol}>
                  {inst.symbol} - {inst.name}
                </option>
              ))}
            </select>
          </div>

          {/* Condition Type Selector */}
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: '600' }}>
              KOŞUL TÜRÜ
            </label>
            <select
              value={conditionType}
              onChange={(e) => {
                setConditionType(e.target.value);
                if (e.target.value === 'DAILY_RETURN') {
                  setSingleValue(directionMode === 'DOWN' ? '-9.0' : '3.0');
                } else if (e.target.value === 'BOFA_NET_FLOW') {
                  setSingleValue(directionMode === 'DOWN' ? '-100000000' : '100000000');
                }
              }}
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'var(--bg-surface)',
                color: '#fff',
                border: '1px solid var(--border-subtle)',
                fontSize: '0.9rem',
                fontWeight: '600',
              }}
            >
              <option value="DAILY_RETURN">Günlük Getiri (%)</option>
              <option value="BOFA_NET_FLOW">BofA (MLB) Net Para Akışı (₺)</option>
              <option value="PRICE_RANGE">Gün İçi Dalgalanma Marjı (%)</option>
              <option value="VOLUME_SURGE">Toplam İşlem Hacmi (₺)</option>
            </select>
          </div>

          {/* Single Threshold or Range Inputs */}
          {directionMode !== 'RANGE' ? (
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: '600' }}>
                {directionMode === 'DOWN' ? 'DÜŞÜŞ EŞİĞİ (≤)' : 'YÜKSELİŞ EŞİĞİ (≥)'} {conditionMeta.unit && `(${conditionMeta.unit})`}
              </label>
              <input
                type="text"
                value={singleValue}
                onChange={(e) => setSingleValue(e.target.value)}
                placeholder={conditionMeta.placeholderSingle}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-surface)',
                  color: directionMode === 'DOWN' ? 'var(--bear-red)' : 'var(--bull-green)',
                  border: `1px solid ${directionMode === 'DOWN' ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
                  fontSize: '0.95rem',
                  fontWeight: '700',
                  fontFamily: 'var(--font-mono)',
                }}
              />
            </div>
          ) : (
            <>
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: '600' }}>
                  MİNİMUM EŞİK {conditionMeta.unit && `(${conditionMeta.unit})`}
                </label>
                <input
                  type="text"
                  value={minValue}
                  onChange={(e) => setMinValue(e.target.value)}
                  placeholder="Örn: -10.0"
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'var(--bg-surface)',
                    color: '#fff',
                    border: '1px solid var(--border-subtle)',
                    fontSize: '0.9rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: '600' }}>
                  MAKSİMUM EŞİK {conditionMeta.unit && `(${conditionMeta.unit})`}
                </label>
                <input
                  type="text"
                  value={maxValue}
                  onChange={(e) => setMaxValue(e.target.value)}
                  placeholder="Örn: -8.0"
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'var(--bg-surface)',
                    color: '#fff',
                    border: '1px solid var(--border-subtle)',
                    fontSize: '0.9rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>
            </>
          )}

          {/* Forward Days Horizon Slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: '600' }}>
                İLERİ GÜN SAYISI (T+N)
              </label>
              <span className="num-mono" style={{ fontSize: '0.85rem', fontWeight: '700', color: 'var(--brand-blue)' }}>
                T+{forwardDays} Gün
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="20"
              value={forwardDays}
              onChange={(e) => setForwardDays(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: 'var(--brand-blue)', cursor: 'pointer' }}
            />
          </div>
        </div>

        {/* Row 2: Quick Presets & Horizon Quick Chips */}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
            paddingTop: '12px',
            borderTop: '1px solid var(--border-subtle)',
          }}
        >
          {/* Presets */}
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: '600' }}>
              Hazır Şablonlar:
            </span>
            {conditionMeta.presets.map((p, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  if (p.val !== undefined) {
                    setSingleValue(p.val);
                  } else {
                    setMinValue(p.min);
                    setMaxValue(p.max);
                  }
                }}
                style={{
                  padding: '5px 10px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.78rem',
                  fontWeight: '500',
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 0.15)';
                  e.currentTarget.style.color = '#fff';
                  e.currentTarget.style.borderColor = 'var(--brand-blue)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.color = 'var(--text-secondary)';
                  e.currentTarget.style.borderColor = 'var(--border-subtle)';
                }}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Horizon Quick Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: '600' }}>
              Vade:
            </span>
            {[1, 2, 3, 5, 10, 15, 20].map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setForwardDays(d)}
                style={{
                  padding: '4px 8px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: forwardDays === d ? '700' : '500',
                  backgroundColor: forwardDays === d ? 'var(--brand-blue)' : 'rgba(255, 255, 255, 0.05)',
                  color: forwardDays === d ? '#fff' : 'var(--text-muted)',
                  border: '1px solid',
                  borderColor: forwardDays === d ? 'var(--brand-blue)' : 'var(--border-subtle)',
                  cursor: 'pointer',
                }}
              >
                T+{d}
              </button>
            ))}
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
            background: 'var(--bear-red-bg)',
            fontSize: '0.88rem',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <XCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* 3. Executive KPI Cards */}
      {data && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
            gap: '14px',
          }}
        >
          {/* Card 1: Total Events */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Eşleşen Olay Sayısı
              </span>
              <Target size={18} style={{ color: 'var(--brand-blue)' }} />
            </div>
            <div className="num-mono" style={{ fontSize: '1.6rem', fontWeight: '700', color: '#fff' }}>
              {data.totalOccurrences}{' '}
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: '400' }}>
                Olay
              </span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {selectedSymbol} için 2022 - 2026 aralığında
            </div>
          </div>

          {/* Card 2: Win Rate % at T+N */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                T+{forwardDays} Kazanma Oranı ({returnMetricMode === 'DAILY' ? 'Günlük' : 'Kümülatif'})
              </span>
              {targetStat && (returnMetricMode === 'DAILY' ? targetStat.winRatePct : (targetStat.cumulWinRatePct || targetStat.winRatePct)) >= 50 ? (
                <CheckCircle2 size={18} style={{ color: 'var(--bull-green)' }} />
              ) : (
                <XCircle size={18} style={{ color: 'var(--bear-red)' }} />
              )}
            </div>
            <div
              className="num-mono"
              style={{
                fontSize: '1.6rem',
                fontWeight: '700',
                color:
                  targetStat && (returnMetricMode === 'DAILY' ? targetStat.winRatePct : (targetStat.cumulWinRatePct || targetStat.winRatePct)) >= 50
                    ? 'var(--bull-green)'
                    : 'var(--bear-red)',
              }}
            >
              %{targetStat ? (returnMetricMode === 'DAILY' ? targetStat.winRatePct.toFixed(1) : (targetStat.cumulWinRatePct || targetStat.winRatePct).toFixed(1)) : '—'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {targetStat ? `${targetStat.sampleCount} örnekten artı kapananlar` : '—'}
            </div>
          </div>

          {/* Card 3: Avg Return % at T+N */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                T+{forwardDays} Ortalama Getiri ({returnMetricMode === 'DAILY' ? 'Bağımsız Günlük' : 'Kümülatif'})
              </span>
              {targetStat && (returnMetricMode === 'DAILY' ? targetStat.avgReturnPct : (targetStat.cumulAvgReturnPct || targetStat.avgReturnPct)) >= 0 ? (
                <TrendingUp size={18} style={{ color: 'var(--bull-green)' }} />
              ) : (
                <TrendingDown size={18} style={{ color: 'var(--bear-red)' }} />
              )}
            </div>
            <div
              className="num-mono"
              style={{
                fontSize: '1.6rem',
                fontWeight: '700',
                color:
                  targetStat && (returnMetricMode === 'DAILY' ? targetStat.avgReturnPct : (targetStat.cumulAvgReturnPct || targetStat.avgReturnPct)) >= 0
                    ? 'var(--bull-green)'
                    : 'var(--bear-red)',
              }}
            >
              {targetStat
                ? (() => {
                    const v = returnMetricMode === 'DAILY' ? targetStat.avgReturnPct : (targetStat.cumulAvgReturnPct || targetStat.avgReturnPct);
                    return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
                  })()
                : '—'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {returnMetricMode === 'DAILY' ? 'T+(N-1) kapanışına göre o gün' : 'T kapanışına göre toplam'}
            </div>
          </div>

          {/* Card 4: Median Return % at T+N */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                T+{forwardDays} Medyan Getiri
              </span>
              <BarChart3 size={18} style={{ color: 'var(--accent-purple)' }} />
            </div>
            <div
              className="num-mono"
              style={{
                fontSize: '1.6rem',
                fontWeight: '700',
                color: targetStat && targetStat.medianReturnPct >= 0 ? 'var(--bull-green)' : 'var(--bear-red)',
              }}
            >
              {targetStat ? (targetStat.medianReturnPct >= 0 ? '+' : '') + targetStat.medianReturnPct.toFixed(2) + '%' : '—'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Aykırı hareketlerden arındırılmış
            </div>
          </div>

          {/* Card 5: Best / Worst Scenario */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Uç Değerler (Maks / Min)
              </span>
              <Clock size={18} style={{ color: 'var(--text-muted)' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
              <span className="num-mono text-bull" style={{ fontSize: '1.25rem', fontWeight: '700' }}>
                {targetStat ? `+${targetStat.maxGainPct.toFixed(1)}%` : '—'}
              </span>
              <span style={{ color: 'var(--text-muted)' }}>/</span>
              <span className="num-mono text-bear" style={{ fontSize: '1.25rem', fontWeight: '700' }}>
                {targetStat ? `${targetStat.maxLossPct.toFixed(1)}%` : '—'}
              </span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              En yüksek getiri / en derin kayıp
            </div>
          </div>
        </div>
      )}

      {/* 4. Interactive Forward Return Trajectory Chart */}
      {chartConfig && (
        <div
          className="glass-card"
          style={{
            padding: '24px',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <TrendingUp size={18} style={{ color: 'var(--brand-blue)' }} />
                İleri Getiri Patikası (
                {returnMetricMode === 'DAILY'
                  ? 'Günlük Bağımsız Getiri: T-1 Kapanışına Göre'
                  : 'Kümülatif Getiri: T Kapanışına Göre'}
                )
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                Kalın çizgi ortalama hareketi, silik çizgiler geçmiş olayların gerçekleşen patikalarını gösterir.
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.78rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '12px', height: '3px', backgroundColor: 'var(--bull-green)', borderRadius: '2px' }}></span>
                <span style={{ color: 'var(--text-secondary)' }}>Ortalama Patika</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '12px', height: '1px', backgroundColor: 'rgba(255,255,255,0.2)' }}></span>
                <span style={{ color: 'var(--text-muted)' }}>Tarihsel Seans İzleri</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '12px', height: '1px', borderTop: '1px dashed #6b7280' }}></span>
                <span style={{ color: 'var(--text-muted)' }}>Sıfır Çizgisi (0%)</span>
              </div>
            </div>
          </div>

          {/* SVG Canvas */}
          <div style={{ width: '100%', overflowX: 'auto' }}>
            <svg
              viewBox={`0 0 ${chartConfig.width} ${chartConfig.height}`}
              style={{ width: '100%', height: 'auto', minWidth: '700px', display: 'block' }}
            >
              <defs>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="rgba(59, 130, 246, 0.4)" />
                </filter>
              </defs>

              {/* Horizontal Grid Lines */}
              {[-15, -10, -5, -2, 0, 2, 5, 10, 15, 20].map((level) => {
                if (level < chartConfig.domainMin || level > chartConfig.domainMax) return null;
                const y = chartConfig.scaleY(level);
                const isZero = level === 0;
                return (
                  <g key={level}>
                    <line
                      x1={chartConfig.margin.left}
                      y1={y}
                      x2={chartConfig.width - chartConfig.margin.right}
                      y2={y}
                      stroke={isZero ? 'rgba(255, 255, 255, 0.35)' : 'rgba(255, 255, 255, 0.06)'}
                      strokeWidth={isZero ? 1.5 : 1}
                      strokeDasharray={isZero ? '4 4' : undefined}
                    />
                    <text
                      x={chartConfig.margin.left - 10}
                      y={y + 4}
                      textAnchor="end"
                      fill={isZero ? '#fff' : 'var(--text-muted)'}
                      fontSize="11"
                      fontFamily="var(--font-mono)"
                    >
                      {level > 0 ? `+${level}%` : `${level}%`}
                    </text>
                  </g>
                );
              })}

              {/* Vertical Grid Lines for Each Day */}
              {chartConfig.avgPoints.map((p) => (
                <g key={p.day}>
                  <line
                    x1={p.x}
                    y1={chartConfig.margin.top}
                    x2={p.x}
                    y2={chartConfig.height - chartConfig.margin.bottom}
                    stroke="rgba(255, 255, 255, 0.06)"
                    strokeWidth="1"
                  />
                  <text
                    x={p.x}
                    y={chartConfig.height - chartConfig.margin.bottom + 20}
                    textAnchor="middle"
                    fill={p.day === 0 ? 'var(--text-muted)' : '#fff'}
                    fontSize="12"
                    fontWeight={p.day === 0 ? '400' : '600'}
                    fontFamily="var(--font-mono)"
                  >
                    {p.day === 0 ? 'T (Olay)' : `T+${p.day}`}
                  </text>
                </g>
              ))}

              {/* Sample individual historical event paths */}
              {chartConfig.samplePaths.map((sp, idx) => (
                <path
                  key={idx}
                  d={sp.pathD}
                  fill="none"
                  stroke="rgba(255, 255, 255, 0.12)"
                  strokeWidth="1"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))}

              {/* Average Trajectory Line */}
              <path
                d={chartConfig.avgPathD}
                fill="none"
                stroke={targetStat && targetStat.avgReturnPct >= 0 ? 'var(--bull-green)' : 'var(--bear-red)'}
                strokeWidth="3.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#glow)"
              />

              {/* Points on Average Line */}
              {chartConfig.avgPoints.map((p) => (
                <g
                  key={p.day}
                  onMouseEnter={() => setHoveredDay(p)}
                  onMouseLeave={() => setHoveredDay(null)}
                  style={{ cursor: 'pointer' }}
                >
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={hoveredDay && hoveredDay.day === p.day ? 7 : 5}
                    fill={p.val >= 0 ? 'var(--bull-green)' : 'var(--bear-red)'}
                    stroke="#fff"
                    strokeWidth="2"
                    transition="all 0.15s ease"
                  />
                  {p.day > 0 && (
                    <text
                      x={p.x}
                      y={p.y - 12}
                      textAnchor="middle"
                      fill={p.val >= 0 ? 'var(--bull-green)' : 'var(--bear-red)'}
                      fontSize="12"
                      fontWeight="700"
                      fontFamily="var(--font-mono)"
                    >
                      {p.val >= 0 ? `+${p.val.toFixed(2)}%` : `${p.val.toFixed(2)}%`}
                    </text>
                  )}
                </g>
              ))}
            </svg>
          </div>

          {/* Hover Day Detail Badge */}
          {hoveredDay && hoveredDay.day > 0 && (
            <div
              className="glass-card"
              style={{
                padding: '12px 18px',
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                borderColor: 'var(--brand-blue)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '12px',
                fontSize: '0.85rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="num-mono" style={{ fontWeight: '700', color: '#fff' }}>
                  T+{hoveredDay.day} Günü İstatistiği:
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                <span>
                  Günlük Bağımsız Getiri:{' '}
                  <strong className={hoveredDay.dailyVal >= 0 ? 'text-bull' : 'text-bear'}>
                    {hoveredDay.dailyVal >= 0 ? `+${hoveredDay.dailyVal.toFixed(2)}%` : `${hoveredDay.dailyVal.toFixed(2)}%`}
                  </strong>
                </span>
                {hoveredDay.cumulVal !== undefined && (
                  <span>
                    Kümülatif Getiri:{' '}
                    <strong className={hoveredDay.cumulVal >= 0 ? 'text-bull' : 'text-bear'}>
                      {hoveredDay.cumulVal >= 0 ? `+${hoveredDay.cumulVal.toFixed(2)}%` : `${hoveredDay.cumulVal.toFixed(2)}%`}
                    </strong>
                  </span>
                )}
                <span>
                  Kazanma Oranı:{' '}
                  <strong style={{ color: '#fff' }}>%{hoveredDay.winRate.toFixed(1)}</strong>
                </span>
                <span>
                  Medyan:{' '}
                  <strong style={{ color: '#fff' }}>
                    {hoveredDay.median >= 0 ? `+${hoveredDay.median.toFixed(2)}%` : `${hoveredDay.median.toFixed(2)}%`}
                  </strong>
                </span>
                <span>
                  Maks Kazanç:{' '}
                  <strong className="text-bull">+{hoveredDay.maxGain.toFixed(1)}%</strong>
                </span>
                <span>
                  Maks Kayıp:{' '}
                  <strong className="text-bear">{hoveredDay.maxLoss.toFixed(1)}%</strong>
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 5. Horizon Step-by-Step Breakdown Table */}
      {data && data.horizonStats && data.horizonStats.length > 0 && (
        <div
          className="glass-card"
          style={{
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Clock size={18} style={{ color: 'var(--brand-blue)' }} />
              Gün Gün Dağılım Tablosu (T+1 &rarr; T+{forwardDays})
            </h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {returnMetricMode === 'DAILY'
                ? 'Her günün bir önceki günün kapanışına göre bağımsız getirisi'
                : 'Olay günü T kapanışına göre kümülatif getiri'}
            </span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  <th style={{ padding: '10px 12px' }}>VADE</th>
                  <th style={{ padding: '10px 12px' }}>GÜNLÜK BAĞIMSIZ GETİRİ (T-1'E GÖRE)</th>
                  <th style={{ padding: '10px 12px' }}>KÜMÜLATİF GETİRİ (T'YE GÖRE)</th>
                  <th style={{ padding: '10px 12px' }}>KAZANMA ORANI (GÜNLÜK)</th>
                  <th style={{ padding: '10px 12px' }}>MEDYAN GETİRİ</th>
                  <th style={{ padding: '10px 12px' }}>EN YÜKSEK KAZANÇ</th>
                  <th style={{ padding: '10px 12px' }}>EN DERİN KAYIP</th>
                  <th style={{ padding: '10px 12px' }}>ÖRNEKLEM</th>
                </tr>
              </thead>
              <tbody>
                {data.horizonStats.map((stat) => {
                  const isDailyPos = stat.avgReturnPct >= 0;
                  const isCumulPos = (stat.cumulAvgReturnPct || stat.avgReturnPct) >= 0;
                  return (
                    <tr
                      key={stat.dayOffset}
                      style={{
                        borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                        fontSize: '0.85rem',
                        transition: 'background 0.15s ease',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      <td style={{ padding: '10px 12px', fontWeight: '700' }} className="num-mono">
                        <span
                          style={{
                            padding: '3px 8px',
                            borderRadius: 'var(--radius-sm)',
                            backgroundColor: 'rgba(59, 130, 246, 0.15)',
                            color: 'var(--brand-blue)',
                          }}
                        >
                          T+{stat.dayOffset}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: '700' }} className="num-mono">
                        <span className={isDailyPos ? 'text-bull' : 'text-bear'}>
                          {isDailyPos ? `+${stat.avgReturnPct.toFixed(2)}%` : `${stat.avgReturnPct.toFixed(2)}%`}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: '600' }} className="num-mono">
                        <span className={isCumulPos ? 'text-bull' : 'text-bear'}>
                          {stat.cumulAvgReturnPct !== null && stat.cumulAvgReturnPct !== undefined
                            ? `${isCumulPos ? '+' : ''}${stat.cumulAvgReturnPct.toFixed(2)}%`
                            : '—'}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div
                            style={{
                              width: '70px',
                              height: '6px',
                              backgroundColor: 'rgba(255, 255, 255, 0.1)',
                              borderRadius: '3px',
                              overflow: 'hidden',
                            }}
                          >
                            <div
                              style={{
                                width: `${Math.min(100, Math.max(0, stat.winRatePct))}%`,
                                height: '100%',
                                backgroundColor: stat.winRatePct >= 50 ? 'var(--bull-green)' : 'var(--bear-red)',
                                borderRadius: '3px',
                              }}
                            />
                          </div>
                          <span
                            className="num-mono"
                            style={{
                              fontWeight: '600',
                              color: stat.winRatePct >= 50 ? 'var(--bull-green)' : 'var(--bear-red)',
                            }}
                          >
                            %{stat.winRatePct.toFixed(1)}
                          </span>
                        </div>
                      </td>
                      <td style={{ padding: '10px 12px' }} className="num-mono">
                        <span className={stat.medianReturnPct >= 0 ? 'text-bull' : 'text-bear'}>
                          {stat.medianReturnPct >= 0 ? `+${stat.medianReturnPct.toFixed(2)}%` : `${stat.medianReturnPct.toFixed(2)}%`}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px' }} className="num-mono text-bull">
                        +{stat.maxGainPct.toFixed(2)}%
                      </td>
                      <td style={{ padding: '10px 12px' }} className="num-mono text-bear">
                        {stat.maxLossPct.toFixed(2)}%
                      </td>
                      <td style={{ padding: '10px 12px', color: 'var(--text-muted)' }} className="num-mono">
                        {stat.sampleCount} Seans
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 6. Granular Historical Event Occurrences Table */}
      {data && (
        <div
          className="glass-card"
          style={{
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Calendar size={18} style={{ color: 'var(--brand-blue)' }} />
                Tarihsel Olaylar, Fiyat Hareketi ve Sonraki Getiriler ({filteredOccurrences.length} Kayıt)
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                Her seans için: Önceki Kapanış, Açılış, Kapanış, Kaçtan Kaça Gittiği ve T+1...T+{forwardDays} gün getirileri
              </p>
            </div>

            {/* Table Search Input */}
            <div style={{ position: 'relative', width: '220px' }}>
              <Search
                size={16}
                style={{
                  position: 'absolute',
                  left: '10px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'var(--text-muted)',
                }}
              />
              <input
                type="text"
                placeholder="Tarih ara (YYYY-MM)..."
                value={tableSearch}
                onChange={(e) => setTableSearch(e.target.value)}
                style={{
                  width: '100%',
                  padding: '7px 10px 7px 32px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-surface)',
                  color: '#fff',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.82rem',
                }}
              />
            </div>
          </div>

          <div style={{ overflowX: 'auto', maxHeight: '550px' }}>
            <table className="data-table" style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead style={{ position: 'sticky', top: 0, backgroundColor: 'var(--bg-surface)', zIndex: 2 }}>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  <th style={{ padding: '10px 12px' }}>OLAY TARİHİ (T)</th>
                  <th style={{ padding: '10px 12px' }}>T-1 KAPANIŞ (₺)</th>
                  <th style={{ padding: '10px 12px' }}>T AÇILIŞ (₺)</th>
                  <th style={{ padding: '10px 12px' }}>T KAPANIŞ (₺)</th>
                  <th style={{ padding: '10px 12px' }}>FİYAT HAREKETİ (T)</th>
                  <th style={{ padding: '10px 12px' }}>BOFA NET AKIŞ</th>
                  {Array.from({ length: forwardDays }, (_, i) => (
                    <th key={i + 1} style={{ padding: '10px 12px', textAlign: 'center' }}>
                      T+{i + 1} ({returnMetricMode === 'DAILY' ? 'GÜNLÜK' : 'KÜMÜLATİF'})
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredOccurrences.length === 0 ? (
                  <tr>
                    <td colSpan={6 + forwardDays} style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      Belirtilen filtre kriterlerine uygun tarihsel olay bulunamadı.
                    </td>
                  </tr>
                ) : (
                  filteredOccurrences.map((occ) => {
                    const isMovePos = (occ.priceChangePct || 0) >= 0;
                    return (
                      <tr
                        key={occ.eventDate}
                        style={{
                          borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                          fontSize: '0.82rem',
                          transition: 'background 0.15s ease',
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                      >
                        {/* Event Date */}
                        <td style={{ padding: '9px 12px', fontWeight: '600' }} className="num-mono">
                          {occ.eventDate}
                        </td>

                        {/* T-1 Previous Close */}
                        <td style={{ padding: '9px 12px', color: 'var(--text-secondary)' }} className="num-mono">
                          {formatPrice(occ.prevClosePrice)}
                        </td>

                        {/* T Open */}
                        <td style={{ padding: '9px 12px', color: 'var(--text-secondary)' }} className="num-mono">
                          {formatPrice(occ.openPrice)}
                        </td>

                        {/* T Close */}
                        <td style={{ padding: '9px 12px', fontWeight: '700', color: '#fff' }} className="num-mono">
                          {formatPrice(occ.closePrice)}
                        </td>

                        {/* Price Journey: e.g. 69.10 ➔ 62.20 ₺ [-9.99%] */}
                        <td style={{ padding: '9px 12px' }} className="num-mono">
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                              {occ.prevClosePrice ? `${occ.prevClosePrice.toFixed(2)} ➔` : ''}{' '}
                              <strong style={{ color: '#fff' }}>{occ.closePrice.toFixed(2)} ₺</strong>
                            </span>
                            <span
                              style={{
                                padding: '2px 6px',
                                borderRadius: 'var(--radius-sm)',
                                fontWeight: '700',
                                fontSize: '0.75rem',
                                backgroundColor: isMovePos ? 'var(--bull-green-bg)' : 'var(--bear-red-bg)',
                                color: isMovePos ? 'var(--bull-green)' : 'var(--bear-red)',
                                border: `1px solid ${isMovePos ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
                              }}
                            >
                              {occ.priceChangePct !== null && occ.priceChangePct !== undefined
                                ? `${isMovePos ? '+' : ''}${occ.priceChangePct.toFixed(2)}%`
                                : `${occ.movementValue.toFixed(2)}%`}
                            </span>
                          </div>
                        </td>

                        {/* BofA Net Flow */}
                        <td style={{ padding: '9px 12px' }} className="num-mono">
                          <span className={occ.bofaNetFlowTl >= 0 ? 'text-bull' : 'text-bear'}>
                            {formatTL(occ.bofaNetFlowTl)}
                          </span>
                        </td>

                        {/* Day 1 to Day N return columns */}
                        {occ.forwardReturns.map((fr) => {
                          const activeRet = returnMetricMode === 'DAILY' ? fr.dailyReturnPct : fr.cumulativeReturnPct;
                          const hasRet = activeRet !== null && activeRet !== undefined;
                          const isPos = hasRet && activeRet > 0;
                          const isNeg = hasRet && activeRet < 0;

                          return (
                            <td key={fr.dayOffset} style={{ padding: '9px 12px', textAlign: 'center' }} className="num-mono">
                              {hasRet ? (
                                <div style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
                                  <span
                                    style={{
                                      display: 'inline-block',
                                      padding: '2px 8px',
                                      borderRadius: 'var(--radius-sm)',
                                      fontWeight: '600',
                                      fontSize: '0.78rem',
                                      backgroundColor: isPos
                                        ? 'var(--bull-green-bg)'
                                        : isNeg
                                        ? 'var(--bear-red-bg)'
                                        : 'rgba(255, 255, 255, 0.05)',
                                      color: isPos ? 'var(--bull-green)' : isNeg ? 'var(--bear-red)' : 'var(--text-muted)',
                                      border: `1px solid ${
                                        isPos
                                          ? 'rgba(16, 185, 129, 0.25)'
                                          : isNeg
                                          ? 'rgba(244, 63, 94, 0.25)'
                                          : 'rgba(255, 255, 255, 0.08)'
                                      }`,
                                    }}
                                  >
                                    {isPos ? `+${activeRet.toFixed(2)}%` : `${activeRet.toFixed(2)}%`}
                                  </span>

                                  {/* Price journey subtext: e.g. 62.20 ➔ 63.60 */}
                                  {fr.closePrice && fr.prevClosePrice && (
                                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                      {returnMetricMode === 'DAILY'
                                        ? `${fr.prevClosePrice.toFixed(2)} ➔ ${fr.closePrice.toFixed(2)} ₺`
                                        : `${occ.closePrice.toFixed(2)} ➔ ${fr.closePrice.toFixed(2)} ₺`}
                                    </span>
                                  )}
                                </div>
                              ) : (
                                <span style={{ color: 'var(--text-muted)' }}>—</span>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
