import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import FilterBar from './components/FilterBar';
import BrokerMetricCards from './components/BrokerMetricCards';
import CandleChart from './components/CandleChart';
import CandleDataTable from './components/CandleDataTable';

const GRAPHQL_ENDPOINT = 'http://127.0.0.1:8000/graphql';

async function fetchGraphQL(query, variables = {}) {
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
          brokerSummary(date: $date, brokerId: $brokerId) {
            brokerId
            tradeDate
            totalBuyVolume
            totalBuyTurnoverTl
            totalSellVolume
            totalSellTurnoverTl
            netFlowTl
            matchedVolume
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
    loadCandleData();
  }, [loadCandleData]);

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
      />

      <CandleChart
        candles={candles}
        symbol={selectedSymbol}
        brokerId={selectedBroker}
        timeframe={selectedTimeframe}
      />

      <CandleDataTable candles={candles} brokerId={selectedBroker} />
    </div>
  );
}
