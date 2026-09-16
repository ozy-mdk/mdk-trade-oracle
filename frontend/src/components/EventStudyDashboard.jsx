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
  HelpCircle,
  Target,
  Clock,
  ArrowRight,
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

export default function EventStudyDashboard({ instruments, fetchGraphQL }) {
  // Filter state
  const [selectedSymbol, setSelectedSymbol] = useState('THYAO');
  const [conditionType, setConditionType] = useState('DAILY_RETURN');
  const [minValue, setMinValue] = useState('3.0');
  const [maxValue, setMaxValue] = useState('');
  const [forwardDays, setForwardDays] = useState(5);
  const [startDate, setStartDate] = useState('2022-01-01');
  const [endDate, setEndDate] = useState('2026-09-14');
  const [limit, setLimit] = useState(100);

  // Data state
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Table search & sort
  const [tableSearch, setTableSearch] = useState('');
  const [hoveredDay, setHoveredDay] = useState(null);

  // Condition type label & placeholder helpers
  const conditionMeta = useMemo(() => {
    switch (conditionType) {
      case 'DAILY_RETURN':
        return {
          label: 'Günlük Getiri (%)',
          unit: '%',
          placeholderMin: 'Örn: 3.0',
          placeholderMax: 'Örn: 10.0',
          presets: [
            { label: '+3.0% ve Üzeri', min: '3.0', max: '' },
            { label: '+5.0% ve Üzeri (Ralli)', min: '5.0', max: '' },
            { label: '-3.0% ve Altı (Düşüş)', min: '', max: '-3.0' },
            { label: '-5.0% ve Altı (Panik)', min: '', max: '-5.0' },
            { label: '+2.0% ile +4.0% Arası', min: '2.0', max: '4.0' },
          ],
        };
      case 'BOFA_NET_FLOW':
        return {
          label: 'BofA (MLB) Net Akışı (TL)',
          unit: '₺',
          placeholderMin: 'Örn: 100000000',
          placeholderMax: 'İsteğe bağlı',
          presets: [
            { label: 'BofA +100M ₺ Üzeri', min: '100000000', max: '' },
            { label: 'BofA +250M ₺ Üzeri', min: '250000000', max: '' },
            { label: 'BofA -100M ₺ Altı', min: '', max: '-100000000' },
            { label: 'BofA -250M ₺ Altı', min: '', max: '-250000000' },
          ],
        };
      case 'PRICE_RANGE':
        return {
          label: 'Gün İçi Dalgalanma Marjı (%)',
          unit: '%',
          placeholderMin: 'Örn: 4.0',
          placeholderMax: 'İsteğe bağlı',
          presets: [
            { label: '%4.0 ve Üzeri Dalgalanma', min: '4.0', max: '' },
            { label: '%6.0 ve Üzeri Dalgalanma', min: '6.0', max: '' },
            { label: '%8.0 ve Üzeri Aşırı Oynaklık', min: '8.0', max: '' },
          ],
        };
      case 'VOLUME_SURGE':
        return {
          label: 'Toplam İşlem Hacmi (TL)',
          unit: '₺',
          placeholderMin: 'Örn: 5000000000',
          placeholderMax: 'İsteğe bağlı',
          presets: [
            { label: '5 Milyar ₺ Üzeri Hacim', min: '5000000000', max: '' },
            { label: '10 Milyar ₺ Üzeri Dev Hacim', min: '10000000000', max: '' },
          ],
        };
      default:
        return { label: 'Koşul', unit: '', presets: [] };
    }
  }, [conditionType]);

  // Load Event Study data from GraphQL
  const loadEventStudy = useCallback(async () => {
    if (!selectedSymbol) return;
    setLoading(true);
    setError(null);

    const minValNum = minValue !== '' ? parseFloat(minValue) : null;
    const maxValNum = maxValue !== '' ? parseFloat(maxValue) : null;

    try {
      const query = `
        query GetEventStudy(
          $symbol: String!
          $conditionType: String!
          $minValue: Float
          $maxValue: Float
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
            forwardDays: $forwardDays
            startDate: $startDate
            endDate: $endDate
            limit: $limit
          ) {
            symbol
            conditionType
            minValue
            maxValue
            forwardDays
            totalOccurrences
            horizonStats {
              dayOffset
              avgReturnPct
              winRatePct
              medianReturnPct
              maxGainPct
              maxLossPct
              sampleCount
            }
            occurrences {
              eventDate
              closePrice
              movementValue
              bofaNetFlowTl
              totalTurnoverTl
              forwardReturns {
                dayOffset
                date
                closePrice
                returnPct
              }
            }
          }
        }
      `;

      const result = await fetchGraphQL(query, {
        symbol: selectedSymbol,
        conditionType,
        minValue: minValNum,
        maxValue: maxValNum,
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
  }, [selectedSymbol, conditionType, minValue, maxValue, forwardDays, startDate, endDate, limit, fetchGraphQL]);

  // Initial fetch on mount & symbol change
  useEffect(() => {
    loadEventStudy();
  }, [loadEventStudy]);

  // Target horizon stat (at Day N)
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

    // Collect all returns to determine min/max Y
    let minY = 0;
    let maxY = 0;

    stats.forEach((s) => {
      if (s.avgReturnPct < minY) minY = s.avgReturnPct;
      if (s.avgReturnPct > maxY) maxY = s.avgReturnPct;
      if (s.maxGainPct > maxY) maxY = s.maxGainPct;
      if (s.maxLossPct < minY) minY = s.maxLossPct;
    });

    // Also factor sample trajectories
    const sampleOccs = (data.occurrences || []).slice(0, 30);
    sampleOccs.forEach((o) => {
      o.forwardReturns.forEach((fr) => {
        if (fr.returnPct !== null && fr.returnPct !== undefined) {
          if (fr.returnPct < minY) minY = fr.returnPct;
          if (fr.returnPct > maxY) maxY = fr.returnPct;
        }
      });
    });

    // Add 10% breathing room padding
    const padding = Math.max(Math.abs(minY), Math.abs(maxY)) * 0.15 || 2.0;
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

    // Average trajectory points: start at T+0 (0.00%)
    const avgPoints = [{ day: 0, val: 0.0, x: scaleX(0), y: scaleY(0) }];
    stats.forEach((s) => {
      avgPoints.push({
        day: s.dayOffset,
        val: s.avgReturnPct,
        winRate: s.winRatePct,
        median: s.medianReturnPct,
        sampleCount: s.sampleCount,
        maxGain: s.maxGainPct,
        maxLoss: s.maxLossPct,
        x: scaleX(s.dayOffset),
        y: scaleY(s.avgReturnPct),
      });
    });

    // Build SVG path
    const avgPathD = avgPoints.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

    // Sample individual paths
    const samplePaths = sampleOccs.map((o) => {
      const pts = [{ x: scaleX(0), y: scaleY(0) }];
      o.forwardReturns.forEach((fr) => {
        if (fr.returnPct !== null && fr.returnPct !== undefined) {
          pts.push({ x: scaleX(fr.dayOffset), y: scaleY(fr.returnPct) });
        }
      });
      return {
        date: o.eventDate,
        pathD: pts.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' '),
        finalReturn: pts.length > 1 ? pts[pts.length - 1] : 0,
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
  }, [data]);

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
            BIST hisselerinde belirli bir fiyat veya kurumsal hacim hareketi gerçekleştikten sonraki{' '}
            <strong style={{ color: '#fff' }}>T+1 ... T+{forwardDays}</strong> gün boyunca oluşan ileri getirilerin tarihsel dağılımı.
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
        {/* Row 1: Symbol, Condition Type, Thresholds, Horizon */}
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
              HAREKET / KOŞUL TÜRÜ
            </label>
            <select
              value={conditionType}
              onChange={(e) => {
                setConditionType(e.target.value);
                // Reset defaults according to type
                if (e.target.value === 'DAILY_RETURN') {
                  setMinValue('3.0');
                  setMaxValue('');
                } else if (e.target.value === 'BOFA_NET_FLOW') {
                  setMinValue('100000000');
                  setMaxValue('');
                } else if (e.target.value === 'PRICE_RANGE') {
                  setMinValue('4.0');
                  setMaxValue('');
                } else {
                  setMinValue('5000000000');
                  setMaxValue('');
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

          {/* Min Value Input */}
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: '600' }}>
              MİNİMUM EŞİK {conditionMeta.unit && `(${conditionMeta.unit})`}
            </label>
            <input
              type="text"
              value={minValue}
              onChange={(e) => setMinValue(e.target.value)}
              placeholder={conditionMeta.placeholderMin}
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

          {/* Max Value Input (Optional) */}
          <div>
            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: '600' }}>
              MAKSİMUM EŞİK (İsteğe Bağlı)
            </label>
            <input
              type="text"
              value={maxValue}
              onChange={(e) => setMaxValue(e.target.value)}
              placeholder={conditionMeta.placeholderMax}
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
              Hazır Filtreler:
            </span>
            {conditionMeta.presets.map((p, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setMinValue(p.min);
                  setMaxValue(p.max);
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
              2022 - 2026 tarih aralığında
            </div>
          </div>

          {/* Card 2: Win Rate % at T+N */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                T+{forwardDays} Kazanma Oranı
              </span>
              {targetStat && targetStat.winRatePct >= 50 ? (
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
                color: targetStat && targetStat.winRatePct >= 50 ? 'var(--bull-green)' : 'var(--bear-red)',
              }}
            >
              %{targetStat ? targetStat.winRatePct.toFixed(1) : '—'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {targetStat ? `${targetStat.sampleCount} örnekten pozitif kapananlar` : '—'}
            </div>
          </div>

          {/* Card 3: Avg Return % at T+N */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                T+{forwardDays} Ortalama Getiri
              </span>
              {targetStat && targetStat.avgReturnPct >= 0 ? (
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
                color: targetStat && targetStat.avgReturnPct >= 0 ? 'var(--bull-green)' : 'var(--bear-red)',
              }}
            >
              {targetStat ? (targetStat.avgReturnPct >= 0 ? '+' : '') + targetStat.avgReturnPct.toFixed(2) + '%' : '—'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Kümülatif ortalama getiri
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
              Aykırı değerlerden arındırılmış
            </div>
          </div>

          {/* Card 5: Best / Worst Scenario */}
          <div className="glass-card" style={{ padding: '16px 20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                En İyi / En Kötü Senaryo
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
              Uç getiri / maksimum kayıp bandı
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
                İleri Getiri Patikası (Forward Return Trajectory: T+0 &rarr; T+{forwardDays})
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                Kalın çizgi kümülatif ortalama getiri patikasını, silik çizgiler geçmiş olayların gerçekleşen patikalarını gösterir.
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.78rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '12px', height: '3px', backgroundColor: 'var(--bull-green)', borderRadius: '2px' }}></span>
                <span style={{ color: 'var(--text-secondary)' }}>Kümülatif Ortalama</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '12px', height: '1px', backgroundColor: 'rgba(255,255,255,0.2)' }}></span>
                <span style={{ color: 'var(--text-muted)' }}>Tarihsel Örnek Olaylar</span>
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
                {/* Glow filter */}
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="rgba(59, 130, 246, 0.4)" />
                </filter>
                <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(16, 185, 129, 0.25)" />
                  <stop offset="100%" stopColor="rgba(16, 185, 129, 0.0)" />
                </linearGradient>
              </defs>

              {/* Horizontal Grid Lines */}
              {[-10, -5, -2, 0, 2, 5, 10, 15, 20].map((level) => {
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
                padding: '10px 16px',
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
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <span>
                  Ortalama Getiri:{' '}
                  <strong className={hoveredDay.val >= 0 ? 'text-bull' : 'text-bear'}>
                    {hoveredDay.val >= 0 ? `+${hoveredDay.val.toFixed(2)}%` : `${hoveredDay.val.toFixed(2)}%`}
                  </strong>
                </span>
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
              Hedef güne kadar her seansın kümülatif performans profili
            </span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  <th style={{ padding: '10px 12px' }}>VADE</th>
                  <th style={{ padding: '10px 12px' }}>ORTALAMA GETİRİ</th>
                  <th style={{ padding: '10px 12px' }}>KAZANMA ORANI (WIN RATE)</th>
                  <th style={{ padding: '10px 12px' }}>MEDYAN GETİRİ</th>
                  <th style={{ padding: '10px 12px' }}>EN YÜKSEK KAZANÇ</th>
                  <th style={{ padding: '10px 12px' }}>EN DERİN KAYIP</th>
                  <th style={{ padding: '10px 12px' }}>ÖRNEK SAYISI</th>
                </tr>
              </thead>
              <tbody>
                {data.horizonStats.map((stat) => {
                  const isPositive = stat.avgReturnPct >= 0;
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
                        <span className={isPositive ? 'text-bull' : 'text-bear'}>
                          {isPositive ? `+${stat.avgReturnPct.toFixed(2)}%` : `${stat.avgReturnPct.toFixed(2)}%`}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div
                            style={{
                              width: '80px',
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
                Tarihsel Olaylar ve Sonraki Getiriler ({filteredOccurrences.length} Kayıt)
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                Koşulu sağlayan her bir tarih için kapanış fiyatı ve T+1...T+{forwardDays} gün getiri oranları
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

          <div style={{ overflowX: 'auto', maxHeight: '500px' }}>
            <table className="data-table" style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead style={{ position: 'sticky', top: 0, backgroundColor: 'var(--bg-surface)', zIndex: 2 }}>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  <th style={{ padding: '10px 12px' }}>OLAY TARİHİ (T)</th>
                  <th style={{ padding: '10px 12px' }}>KAPANIŞ (₺)</th>
                  <th style={{ padding: '10px 12px' }}>HAREKET DEĞERİ</th>
                  <th style={{ padding: '10px 12px' }}>BOFA NET AKIŞ (T)</th>
                  {Array.from({ length: forwardDays }, (_, i) => (
                    <th key={i + 1} style={{ padding: '10px 12px', textAlign: 'center' }}>
                      T+{i + 1} GETİRİ
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredOccurrences.length === 0 ? (
                  <tr>
                    <td colSpan={4 + forwardDays} style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      Belirtilen filtre kriterlerine uygun tarihsel olay bulunamadı.
                    </td>
                  </tr>
                ) : (
                  filteredOccurrences.map((occ) => (
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
                      <td style={{ padding: '9px 12px', fontWeight: '600' }} className="num-mono">
                        {occ.eventDate}
                      </td>
                      <td style={{ padding: '9px 12px' }} className="num-mono">
                        {occ.closePrice.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ₺
                      </td>
                      <td style={{ padding: '9px 12px', fontWeight: '600' }} className="num-mono">
                        {conditionType === 'DAILY_RETURN' ? (
                          <span className={occ.movementValue >= 0 ? 'text-bull' : 'text-bear'}>
                            {occ.movementValue >= 0 ? `+${occ.movementValue.toFixed(2)}%` : `${occ.movementValue.toFixed(2)}%`}
                          </span>
                        ) : conditionType === 'BOFA_NET_FLOW' ? (
                          <span className={occ.movementValue >= 0 ? 'text-bull' : 'text-bear'}>
                            {formatTL(occ.movementValue)}
                          </span>
                        ) : (
                          <span>{occ.movementValue.toFixed(2)}%</span>
                        )}
                      </td>
                      <td style={{ padding: '9px 12px' }} className="num-mono">
                        <span className={occ.bofaNetFlowTl >= 0 ? 'text-bull' : 'text-bear'}>
                          {formatTL(occ.bofaNetFlowTl)}
                        </span>
                      </td>

                      {/* Day 1 to Day N return columns */}
                      {occ.forwardReturns.map((fr) => {
                        const hasRet = fr.returnPct !== null && fr.returnPct !== undefined;
                        const isPos = hasRet && fr.returnPct > 0;
                        const isNeg = hasRet && fr.returnPct < 0;
                        return (
                          <td key={fr.dayOffset} style={{ padding: '9px 12px', textAlign: 'center' }} className="num-mono">
                            {hasRet ? (
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
                                title={fr.date ? `${fr.date}: ${fr.closePrice ? fr.closePrice.toFixed(2) + ' ₺' : ''}` : ''}
                              >
                                {isPos ? `+${fr.returnPct.toFixed(2)}%` : `${fr.returnPct.toFixed(2)}%`}
                              </span>
                            ) : (
                              <span style={{ color: 'var(--text-muted)' }}>—</span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
