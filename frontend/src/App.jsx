import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import FilterBar from './components/FilterBar';
import BrokerMetricCards from './components/BrokerMetricCards';
import CandleChart from './components/CandleChart';
import CandleDataTable from './components/CandleDataTable';
import TertipDashboard from './components/TertipDashboard';
import EventStudyDashboard from './components/EventStudyDashboard';
import TimeWindowTerminal from './components/TimeWindowTerminal';
import { CandlestickChart, Layers, TrendingUp, Clock } from 'lucide-react';

const GRAPHQL_ENDPOINT = 'http://127.0.0.1:8000/graphql';

export async function fetchGraphQL(query, variables = {}) {
  const response = await fetch(GRAPHQL_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, variables }),
  });
  const json = await response.json();
  if (json.errors && json.errors.length > 0) {
    throw new Error(json.errors[0].message);
  }
  return json.data;
}

export default function App() {
  const [activeTab, setActiveTab] = useState('candles'); // 'candles' or 'tertip'

  const [instruments, setInstruments] = useState([]);
  const [brokers, setBrokers] = useState([]);
  const [dates, setDates] = useState([]);

  const [selectedSymbol, setSelectedSymbol] = useState('THYAO');
  const [selectedBroker, setSelectedBroker] = useState('MLB');
  const [selectedTimeframe, setSelectedTimeframe] = useState('60m');
  const [selectedDate, setSelectedDate] = useState('2026-09-14');

  const [candles, setCandles] = useState([]);
  const [brokerSummary, setBrokerSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 1. Fetch initial metadata
  useEffect(() => {
    async function loadMetadata() {
      try {
        const data = await fetchGraphQL(`
          query {
            instruments { symbol name sector }
            brokers { brokerId brokerName }
            availableDates
          }
        `);
        setInstruments(data.instruments || []);
        setBrokers(data.brokers || []);
        const availDates = data.availableDates || [];
        setDates(availDates);
        if (availDates.length > 0 && !availDates.includes(selectedDate)) {
          setSelectedDate(availDates[0]);
        }
      } catch (err) {
        console.error('Metadata fetch error:', err);
        setError('Sunucu bağlantısı kurulamadı. Lütfen backend sunucusunu kontrol edin.');
      }
    }
    loadMetadata();
  }, []);

  // 2. Fetch candles and broker summary
  const loadCandleData = useCallback(async () => {
    if (!selectedSymbol || !selectedDate) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchGraphQL(
        `
        query GetCandles($symbol: String!, $timeframe: String!, $date: String!, $brokerId: String!) {
          candles(symbol: $symbol, timeframe: $timeframe, date: $date, brokerId: $brokerId) {
            timeframe
            bucketStart
            bucketEnd
            symbol
            open
            high
            low
            close
            volume
            turnoverTl
            vwap
            tradeCount
            buyVolume
            buyTurnoverTl
            sellVolume
            sellTurnoverTl
            netVolume
            netFlowTl
            matchedVolume
            realizedPnlTl
            brokerSharePct
          }
          brokerSummary(date: $date, brokerId: $brokerId, symbol: $symbol) {
            brokerId
            tradeDate
            symbol
            totalBuyVolume
            totalBuyTurnoverTl
            totalSellVolume
            totalSellTurnoverTl
            netFlowTl
            netVolume
            openStockQuantity
            matchedVolume
            intradayPnlTl
            carryFifoPnlTl
            realizedPnlTl
            positionBias
          }
        }
      `,
        {
          symbol: selectedSymbol,
          timeframe: selectedTimeframe,
          date: selectedDate,
          brokerId: selectedBroker,
        }
      );
      setCandles(data.candles || []);
      setBrokerSummary(data.brokerSummary || null);
    } catch (err) {
      console.error('Candles fetch error:', err);
      setError(`Veri yüklenirken hata oluştu: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol, selectedTimeframe, selectedDate, selectedBroker]);

  useEffect(() => {
    if (activeTab === 'candles') {
      loadCandleData();
    }
  }, [loadCandleData, activeTab]);

  // Aggregate candle-specific stats for this symbol
  const candleSummary = {
    buyTurnoverTl: candles.reduce((acc, c) => acc + (c.buyTurnoverTl || 0), 0),
    sellTurnoverTl: candles.reduce((acc, c) => acc + (c.sellTurnoverTl || 0), 0),
    buyVolume: candles.reduce((acc, c) => acc + (c.buyVolume || 0), 0),
    sellVolume: candles.reduce((acc, c) => acc + (c.sellVolume || 0), 0),
    netFlowTl: candles.reduce((acc, c) => acc + (c.netFlowTl || 0), 0),
    netVolume: candles.reduce((acc, c) => acc + (c.netVolume || 0), 0),
    matchedVolume: candles.reduce((acc, c) => acc + (c.matchedVolume || 0), 0),
    realizedPnlTl: candles.reduce((acc, c) => acc + (c.realizedPnlTl || 0), 0),
  };

  return (
    <div className="app-container">
      <Header onRefresh={loadCandleData} loading={loading} />

      {/* Main Tab Navigation */}
      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          type="button"
          onClick={() => setActiveTab('candles')}
          className="glass-card"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '14px 20px',
            cursor: 'pointer',
            fontSize: '1rem',
            fontWeight: '600',
            color: activeTab === 'candles' ? '#fff' : 'var(--text-secondary)',
            borderColor: activeTab === 'candles' ? 'var(--brand-blue)' : 'var(--border-subtle)',
            background: activeTab === 'candles' ? 'rgba(59, 130, 246, 0.15)' : 'var(--bg-card)',
            boxShadow: activeTab === 'candles' ? 'var(--shadow-glow)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          <CandlestickChart size={20} className={activeTab === 'candles' ? 'text-bull' : ''} />
          <span>1. Mum Grafiği &amp; Gün İçi Akış (Candles &amp; Flow)</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('tertip')}
          className="glass-card"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '14px 20px',
            cursor: 'pointer',
            fontSize: '1rem',
            fontWeight: '600',
            color: activeTab === 'tertip' ? '#fff' : 'var(--text-secondary)',
            borderColor: activeTab === 'tertip' ? 'var(--accent-purple)' : 'var(--border-subtle)',
            background: activeTab === 'tertip' ? 'rgba(139, 92, 246, 0.15)' : 'var(--bg-card)',
            boxShadow: activeTab === 'tertip' ? '0 0 25px -5px rgba(139, 92, 246, 0.3)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          <Layers size={20} style={{ color: activeTab === 'tertip' ? 'var(--accent-purple)' : '' }} />
          <span>2. Kurum Tertip &amp; FIFO Takibi (2022 - 2026 Defteri)</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('eventStudy')}
          className="glass-card"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '14px 20px',
            cursor: 'pointer',
            fontSize: '1rem',
            fontWeight: '600',
            color: activeTab === 'eventStudy' ? '#fff' : 'var(--text-secondary)',
            borderColor: activeTab === 'eventStudy' ? 'var(--bull-green)' : 'var(--border-subtle)',
            background: activeTab === 'eventStudy' ? 'rgba(16, 185, 129, 0.15)' : 'var(--bg-card)',
            boxShadow: activeTab === 'eventStudy' ? '0 0 25px -5px rgba(16, 185, 129, 0.3)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          <TrendingUp size={20} style={{ color: activeTab === 'eventStudy' ? 'var(--bull-green)' : '' }} />
          <span>3. Hisse Hareket &amp; İleri Getiri Analizi (Event Study)</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('timeWindow')}
          className="glass-card"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '14px 20px',
            cursor: 'pointer',
            fontSize: '1rem',
            fontWeight: '600',
            color: activeTab === 'timeWindow' ? '#fff' : 'var(--text-secondary)',
            borderColor: activeTab === 'timeWindow' ? 'var(--brand-blue)' : 'var(--border-subtle)',
            background: activeTab === 'timeWindow' ? 'rgba(59, 130, 246, 0.15)' : 'var(--bg-card)',
            boxShadow: activeTab === 'timeWindow' ? '0 0 25px -5px rgba(59, 130, 246, 0.3)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          <Clock size={20} style={{ color: activeTab === 'timeWindow' ? 'var(--brand-blue)' : '' }} />
          <span>4. Özel Tarih &amp; Saat Aralığı (Açılış/Kapanış Analizi)</span>
        </button>
      </div>

      {error && (
        <div
          className="glass-card"
          style={{
            padding: '12px 16px',
            borderColor: 'var(--bear-red)',
            color: 'var(--bear-red)',
            background: 'var(--bear-red-bg)',
            fontSize: '0.85rem',
          }}
        >
          {error}
        </div>
      )}

      {/* Tab 1: Candles & Intraday Flow */}
      {activeTab === 'candles' && (
        <>
          <FilterBar
            instruments={instruments}
            brokers={brokers}
            dates={dates}
            selectedSymbol={selectedSymbol}
            onSelectSymbol={setSelectedSymbol}
            selectedBroker={selectedBroker}
            onSelectBroker={setSelectedBroker}
            selectedTimeframe={selectedTimeframe}
            onSelectTimeframe={setSelectedTimeframe}
            selectedDate={selectedDate}
            onSelectDate={setSelectedDate}
          />

          <BrokerMetricCards
            summary={brokerSummary}
            candleSummary={candleSummary}
            selectedBroker={selectedBroker}
            selectedSymbol={selectedSymbol}
          />

          <CandleChart
            candles={candles}
            symbol={selectedSymbol}
            brokerId={selectedBroker}
            timeframe={selectedTimeframe}
          />

          <CandleDataTable candles={candles} brokerId={selectedBroker} />
        </>
      )}

      {/* Tab 2: Tertip & FIFO Lifecycle Dashboard */}
      {activeTab === 'tertip' && (
        <TertipDashboard
          brokers={brokers}
          instruments={instruments}
          dates={dates}
          fetchGraphQL={fetchGraphQL}
        />
      )}

      {/* Tab 3: Event Study & Forward Return Dashboard */}
      {activeTab === 'eventStudy' && (
        <EventStudyDashboard
          instruments={instruments}
          fetchGraphQL={fetchGraphQL}
        />
      )}

      {/* Tab 4: Custom Date & Time Range Terminal */}
      {activeTab === 'timeWindow' && (
        <TimeWindowTerminal
          instruments={instruments}
          brokers={brokers}
          dates={dates}
          fetchGraphQL={fetchGraphQL}
        />
      )}
    </div>
  );
}

