import React, { useState, useEffect } from 'react';
import { Archive, BarChart2, CheckCircle2, Clock, DollarSign, Layers, TrendingDown, TrendingUp } from 'lucide-react';

function formatTL(num) {
  if (num === null || num === undefined || isNaN(num)) return '—';
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
  if (num === null || num === undefined || isNaN(num)) return '—';
  return Math.round(num).toLocaleString('tr-TR') + ' Lot';
}

export default function TertipDashboard({
  brokers,
  instruments,
  dates,
  fetchGraphQL,
}) {
  const [selectedBroker, setSelectedBroker] = useState('MLB');
  const [selectedSymbol, setSelectedSymbol] = useState('ALL');
  const [selectedDate, setSelectedDate] = useState('2026-09-14');
  const [selectedStatus, setSelectedStatus] = useState('ALL');
  const [activeSubTab, setActiveSubTab] = useState('daily'); // 'daily' or 'lots'

  const [summary, setSummary] = useState(null);
  const [dailyRecords, setDailyRecords] = useState([]);
  const [lots, setLots] = useState([]);
  const [loading, setLoading] = useState(false);

  // Fetch Tertip Data
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const query = `
          query GetTertipData($brokerId: String!, $date: String!, $symbol: String, $status: String) {
            tertipSummary(brokerId: $brokerId, date: $date) {
              brokerId
              tradeDate
              cumulativeRealizedPnlTl
              dailyRealizedPnlTl
              intradayRealizedPnlTl
              carryFifoRealizedPnlTl
              openInventoryCostTl
              openInventoryValueTl
              unrealizedPnlTl
              totalDailyPnlTl
              totalOpenLotsCount
            }
            dailyFifo(brokerId: $brokerId, symbol: $symbol, limit: 100) {
              tradeDate
              symbol
              symbolName
              sector
              positionSide
              buyVwap
              sellVwap
              intradayRealizedPnlTl
              carryFifoRealizedPnlTl
              dailyRealizedPnlTl
              openStockQuantity
              fifoAvgCost
              unrealizedPnlTl
              cumulativeRealizedPnlTl
            }
            tertipLots(brokerId: $brokerId, symbol: $symbol, status: $status, limit: 100) {
              lotId
              symbol
              direction
              openDate
              openedQuantity
              openedUnitCost
              status
              closedDate
              totalQuantityClosed
              totalRealizedPnlTl
              remainingQuantity
            }
          }
        `;
        const data = await fetchGraphQL(query, {
          brokerId: selectedBroker,
          date: selectedDate,
          symbol: selectedSymbol === 'ALL' ? null : selectedSymbol,
          status: selectedStatus === 'ALL' ? null : selectedStatus,
        });
        setSummary(data.tertipSummary);
        setDailyRecords(data.dailyFifo || []);
        setLots(data.tertipLots || []);
      } catch (err) {
        console.error('Tertip load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [selectedBroker, selectedSymbol, selectedDate, selectedStatus]);

  const isProfitableDaily = (summary?.dailyRealizedPnlTl || 0) >= 0;
  const isProfitableCum = (summary?.cumulativeRealizedPnlTl || 0) >= 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Filter Bar */}
      <section className="filter-bar glass-card">
        {/* Broker Selector */}
        <div className="filter-item">
          <label className="filter-label">
            <Layers size={14} />
            <span>Kurum Seçimi</span>
          </label>
          <select
            className="filter-select"
            value={selectedBroker}
            onChange={(e) => setSelectedBroker(e.target.value)}
          >
            {brokers.map((b) => (
              <option key={b.brokerId} value={b.brokerId}>
                {b.brokerId === 'MLB' ? '⭐ ' : ''}
                {b.brokerId} — {b.brokerName}
              </option>
            ))}
          </select>
        </div>

        {/* Symbol Selector */}
        <div className="filter-item">
          <label className="filter-label">
            <BarChart2 size={14} />
            <span>Hisse Filtresi</span>
          </label>
          <select
            className="filter-select num-mono"
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
          >
            <option value="ALL">TÜMÜ (45 Likit BIST Hissesi)</option>
            {instruments.map((inst) => (
              <option key={inst.symbol} value={inst.symbol}>
                {inst.symbol} — {inst.name}
              </option>
            ))}
          </select>
        </div>

        {/* Date Selector */}
        <div className="filter-item">
          <label className="filter-label">
            <Clock size={14} />
            <span>Özet Oturum Tarihi</span>
          </label>
          <select
            className="filter-select num-mono"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
          >
            {dates.map((d) => (
              <option key={d} value={d}>
                {d} {d === '2026-09-14' ? '(En Son Oturum)' : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Status Selector */}
        <div className="filter-item">
          <label className="filter-label">
            <CheckCircle2 size={14} />
            <span>Tertip Durumu</span>
          </label>
          <select
            className="filter-select"
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
          >
            <option value="ALL">TÜMÜ (Açık ve Kapalı Tertipler)</option>
            <option value="OPEN">Sadece AÇIK Tertipler</option>
            <option value="CLOSED">Sadece KAPANMIŞ Tertipler</option>
          </select>
        </div>
      </section>

      {/* Executive Summary Cards */}
      <section className="stats-grid">
        {/* 1. Cumulative Realized PnL */}
        <div className={`stat-card glass-card ${isProfitableCum ? 'bull' : 'bear'}`}>
          <div className="stat-header">
            <span>2022'den Bugüne Kümülatif Kâr/Zarar</span>
            {isProfitableCum ? <TrendingUp size={16} className="text-bull" /> : <TrendingDown size={16} className="text-bear" />}
          </div>
          <div className={`stat-value num-mono ${isProfitableCum ? 'text-bull' : 'text-bear'}`}>
            {isProfitableCum ? '+' : ''}{formatTL(summary?.cumulativeRealizedPnlTl)}
          </div>
          <div className="stat-subtext">2022–2026 Denetlenmiş Tertip Kârı</div>
        </div>

        {/* 2. Daily Realized PnL */}
        <div className={`stat-card glass-card ${isProfitableDaily ? 'bull' : 'bear'}`}>
          <div className="stat-header">
            <span>Günün Gerçekleşen Net Kârı</span>
            <span className={`status-pill ${isProfitableDaily ? 'bg-bull-badge' : 'bg-bear-badge'}`} style={{ padding: '2px 8px', fontSize: '0.7rem' }}>
              {isProfitableDaily ? 'NET KÂR' : 'NET ZARAR'}
            </span>
          </div>
          <div className={`stat-value num-mono ${isProfitableDaily ? 'text-bull' : 'text-bear'}`}>
            {isProfitableDaily ? '+' : ''}{formatTL(summary?.dailyRealizedPnlTl)}
          </div>
          <div className="stat-subtext num-mono" style={{ display: 'flex', gap: '8px', fontSize: '0.72rem' }}>
            <span>Gün İçi: <b>{formatTL(summary?.intradayRealizedPnlTl)}</b></span>
            <span>Carry FIFO: <b>{formatTL(summary?.carryFifoRealizedPnlTl)}</b></span>
          </div>
        </div>

        {/* 3. Open Inventory Cost */}
        <div className="stat-card glass-card">
          <div className="stat-header">
            <span>Açık Pozisyon Portföy Maliyeti</span>
            <Archive size={16} style={{ color: 'var(--brand-blue)' }} />
          </div>
          <div className="stat-value num-mono" style={{ color: '#93c5fd' }}>
            {formatTL(summary?.openInventoryCostTl)}
          </div>
          <div className="stat-subtext">Açık FIFO Tertipleri Maliyeti</div>
        </div>

        {/* 4. Unrealized MTM PnL */}
        <div className={`stat-card glass-card ${(summary?.unrealizedPnlTl || 0) >= 0 ? 'bull' : 'bear'}`}>
          <div className="stat-header">
            <span>Taşınan Pozisyon Değerlemesi (MTM)</span>
            <DollarSign size={16} />
          </div>
          <div className={`stat-value num-mono ${(summary?.unrealizedPnlTl || 0) >= 0 ? 'text-bull' : 'text-bear'}`}>
            {(summary?.unrealizedPnlTl || 0) >= 0 ? '+' : ''}{formatTL(summary?.unrealizedPnlTl)}
          </div>
          <div className="stat-subtext">Piyasa Kapanış Değerlemesi (MTM)</div>
        </div>

        {/* 5. Total Active Open Lots */}
        <div className="stat-card glass-card">
          <div className="stat-header">
            <span>Aktif Açık Tertip Sayısı</span>
            <Layers size={16} style={{ color: 'var(--accent-purple)' }} />
          </div>
          <div className="stat-value num-mono" style={{ color: '#c4b5fd' }}>
            {summary?.totalOpenLotsCount?.toLocaleString('tr-TR') || '—'} Adet
          </div>
          <div className="stat-subtext">Kapanış Bekleyen Açık Lotlar</div>
        </div>
      </section>

      {/* Sub-Tab Navigation */}
      <div style={{ display: 'flex', gap: '10px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
        <button
          type="button"
          className={`tf-btn ${activeSubTab === 'daily' ? 'active' : ''}`}
          style={{ padding: '8px 16px', fontSize: '0.9rem' }}
          onClick={() => setActiveSubTab('daily')}
        >
          1. Günlük Tertip &amp; FIFO Defteri (silver_broker_fifo_daily)
        </button>
        <button
          type="button"
          className={`tf-btn ${activeSubTab === 'lots' ? 'active' : ''}`}
          style={{ padding: '8px 16px', fontSize: '0.9rem' }}
          onClick={() => setActiveSubTab('lots')}
        >
          2. Bireysel Tertip (Lot) Yaşam Döngüsü (silver_broker_fifo_lot_lifecycle)
        </button>
      </div>

      {/* View 1: Daily FIFO Ledger */}
      {activeSubTab === 'daily' && (
        <div className="table-card glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: '600' }}>
              Günlük Kurum Tertip ve Realized Kâr/Zarar Defteri (2022 - 2026)
            </h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Son {dailyRecords.length} Günlük Kayıt
            </span>
          </div>

          <div className="table-responsive">
            <table className="data-table num-mono">
              <thead>
                <tr>
                  <th>İşlem Tarihi</th>
                  <th>Hisse</th>
                  <th>Sektör</th>
                  <th>Yön</th>
                  <th>Alış VWAP</th>
                  <th>Satış VWAP</th>
                  <th>Gün İçi Kârı</th>
                  <th>Carry FIFO Kârı</th>
                  <th>Günlük Realized Kâr</th>
                  <th>Açık Lot</th>
                  <th>Ort. Maliyet</th>
                  <th>Kümülatif Realized Kâr</th>
                </tr>
              </thead>
              <tbody>
                {dailyRecords.map((r, i) => {
                  const isDailyProfit = r.dailyRealizedPnlTl >= 0;
                  const isCumProfit = r.cumulativeRealizedPnlTl >= 0;
                  return (
                    <tr key={i}>
                      <td style={{ color: 'var(--text-secondary)' }}>{r.tradeDate}</td>
                      <td style={{ fontWeight: '700', color: '#fff' }}>{r.symbol}</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{r.sector}</td>
                      <td>
                        <span className={`status-pill ${r.positionSide === 'LONG' ? 'bg-bull-badge' : (r.positionSide === 'SHORT' ? 'bg-bear-badge' : '')}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>
                          {r.positionSide}
                        </span>
                      </td>
                      <td>{r.buyVwap ? r.buyVwap.toFixed(2) + ' ₺' : '—'}</td>
                      <td>{r.sellVwap ? r.sellVwap.toFixed(2) + ' ₺' : '—'}</td>
                      <td className={r.intradayRealizedPnlTl >= 0 ? 'text-bull' : 'text-bear'}>
                        {formatTL(r.intradayRealizedPnlTl)}
                      </td>
                      <td className={r.carryFifoRealizedPnlTl >= 0 ? 'text-bull' : 'text-bear'}>
                        {formatTL(r.carryFifoRealizedPnlTl)}
                      </td>
                      <td className={isDailyProfit ? 'text-bull' : 'text-bear'} style={{ fontWeight: '700' }}>
                        {isDailyProfit ? '+' : ''}{formatTL(r.dailyRealizedPnlTl)}
                      </td>
                      <td>{formatLots(r.openStockQuantity)}</td>
                      <td>{r.fifoAvgCost ? r.fifoAvgCost.toFixed(2) + ' ₺' : '—'}</td>
                      <td className={isCumProfit ? 'text-bull' : 'text-bear'} style={{ fontWeight: '800' }}>
                        {isCumProfit ? '+' : ''}{formatTL(r.cumulativeRealizedPnlTl)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* View 2: Tertip Lot Lifecycles */}
      {activeSubTab === 'lots' && (
        <div className="table-card glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: '600' }}>
              Bireysel Tertip (Lot) Açılış, Kapanış ve Kâr/Zarar Takibi
            </h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Son {lots.length} Tertip
            </span>
          </div>

          <div className="table-responsive">
            <table className="data-table num-mono">
              <thead>
                <tr>
                  <th>Tertip No (Lot ID)</th>
                  <th>Hisse</th>
                  <th>Yön</th>
                  <th>Açılış Tarihi</th>
                  <th>Açılan Lot</th>
                  <th>Birim Maliyet</th>
                  <th>Durum</th>
                  <th>Kapanış Tarihi</th>
                  <th>Kapanan Lot</th>
                  <th>Tertip Realized Kâr/Zarar</th>
                  <th>Kalan Lot</th>
                </tr>
              </thead>
              <tbody>
                {lots.map((l, i) => {
                  const isProfit = l.totalRealizedPnlTl >= 0;
                  const isOpen = l.status === 'OPEN';
                  return (
                    <tr key={i}>
                      <td style={{ color: 'var(--text-secondary)', fontSize: '0.72rem' }}>{l.lotId}</td>
                      <td style={{ fontWeight: '700', color: '#fff' }}>{l.symbol}</td>
                      <td>
                        <span className={`status-pill ${l.direction === 'LONG' ? 'bg-bull-badge' : 'bg-bear-badge'}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>
                          {l.direction}
                        </span>
                      </td>
                      <td>{l.openDate}</td>
                      <td>{formatLots(l.openedQuantity)}</td>
                      <td>{l.openedUnitCost.toFixed(2)} ₺</td>
                      <td>
                        <span className={`status-pill ${isOpen ? 'bg-bull-badge' : ''}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>
                          {isOpen ? 'AÇIK' : 'KAPANDI'}
                        </span>
                      </td>
                      <td>{l.closedDate || '—'}</td>
                      <td>{formatLots(l.totalQuantityClosed)}</td>
                      <td className={isProfit ? 'text-bull' : 'text-bear'} style={{ fontWeight: '700' }}>
                        {l.totalRealizedPnlTl !== 0 ? (isProfit ? '+' : '') + formatTL(l.totalRealizedPnlTl) : '—'}
                      </td>
                      <td style={{ fontWeight: '600', color: isOpen ? '#93c5fd' : 'var(--text-muted)' }}>
                        {formatLots(l.remainingQuantity)}
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
