import {
  InstrumentItem,
  BrokerItem,
  DateRangeResponse,
  CandleBar,
  MarketSummaryResponse,
  TertipPortfolioResponse,
  TertipLotItem,
  TertipHistoryPoint,
  EventStudyScanItem,
  TimeWindowAnalysisResponse,
  AllSignalsResponse,
  TertipHorizonsResponse,
  TertipTimeseriesPoint,
  ShockDayItem,
} from '../types/api';

const BASE_URL = ''; // Relative URL handled by Vite proxy to localhost:8000

export async function fetchInstruments(): Promise<InstrumentItem[]> {
  const res = await fetch(`${BASE_URL}/api/v1/meta/instruments`);
  if (!res.ok) throw new Error('Failed to fetch instruments');
  return res.json();
}

export async function fetchBrokers(): Promise<BrokerItem[]> {
  const res = await fetch(`${BASE_URL}/api/v1/meta/brokers`);
  if (!res.ok) throw new Error('Failed to fetch brokers');
  return res.json();
}

export async function fetchDateRange(symbol?: string): Promise<DateRangeResponse> {
  let url = `${BASE_URL}/api/v1/meta/date-range`;
  if (symbol) url += `?symbol=${encodeURIComponent(symbol)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch date range');
  return res.json();
}

export async function fetchCandles(
  symbol: string,
  interval: string = '5m',
  limit: number = 1500,
  brokerId: string = 'MLB'
): Promise<CandleBar[]> {
  const res = await fetch(
    `${BASE_URL}/api/v1/market/candles?symbol=${encodeURIComponent(symbol)}&interval=${interval}&broker_id=${encodeURIComponent(brokerId)}&limit=${limit}`
  );
  if (!res.ok) throw new Error('Failed to fetch candlesticks');
  return res.json();
}

export async function fetchMarketSummary(
  symbol: string,
  brokerId: string = 'MLB',
  tradeDate?: string
): Promise<MarketSummaryResponse> {
  let url = `${BASE_URL}/api/v1/market/summary?symbol=${encodeURIComponent(symbol)}&broker_id=${encodeURIComponent(brokerId)}`;
  if (tradeDate) url += `&trade_date=${tradeDate}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch market summary');
  return res.json();
}

export async function fetchTertipPortfolio(
  brokerId: string = 'MLB',
  tradeDate?: string
): Promise<TertipPortfolioResponse> {
  let url = `${BASE_URL}/api/v1/tertip/portfolio?broker_id=${encodeURIComponent(brokerId)}`;
  if (tradeDate) url += `&trade_date=${tradeDate}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch tertip portfolio');
  return res.json();
}

export async function fetchTertipLots(
  brokerId: string = 'MLB',
  symbol?: string
): Promise<TertipLotItem[]> {
  let url = `${BASE_URL}/api/v1/tertip/lots?broker_id=${encodeURIComponent(brokerId)}`;
  if (symbol) url += `&symbol=${encodeURIComponent(symbol)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch tertip lots');
  return res.json();
}

export async function fetchTertipHistory(
  brokerId: string = 'MLB',
  symbol?: string,
  limit?: number
): Promise<TertipHistoryPoint[]> {
  let url = `${BASE_URL}/api/v1/tertip/history?broker_id=${encodeURIComponent(brokerId)}`;
  if (symbol) url += `&symbol=${encodeURIComponent(symbol)}`;
  if (limit) url += `&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch tertip history');
  return res.json();
}

export async function fetchTertipHorizons(
  symbol: string,
  brokerId: string = 'MLB',
  tradeDate?: string
): Promise<TertipHorizonsResponse> {
  let url = `${BASE_URL}/api/v1/tertip/horizons?symbol=${encodeURIComponent(symbol)}&broker_id=${encodeURIComponent(brokerId)}`;
  if (tradeDate) url += `&trade_date=${tradeDate}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch tertip horizons');
  return res.json();
}

export async function fetchTertipTimeseries(
  symbol: string,
  brokerId: string = 'MLB',
  limitDays: number = 1500
): Promise<TertipTimeseriesPoint[]> {
  const url = `${BASE_URL}/api/v1/tertip/timeseries?symbol=${encodeURIComponent(symbol)}&broker_id=${encodeURIComponent(brokerId)}&limit_days=${limitDays}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch tertip timeseries');
  return res.json();
}

export async function fetchEventStudyScan(params: {
  brokerId?: string;
  symbol?: string;
  minFlowTl?: number;
  maxFlowTl?: number;
  minD1Return?: number;
  maxD1Return?: number;
  limit?: number;
}): Promise<EventStudyScanItem[]> {
  const query = new URLSearchParams();
  if (params.brokerId) query.set('broker_id', params.brokerId);
  if (params.symbol) query.set('symbol', params.symbol);
  if (params.minFlowTl !== undefined && params.minFlowTl !== null)
    query.set('min_flow_tl', params.minFlowTl.toString());
  if (params.maxFlowTl !== undefined && params.maxFlowTl !== null)
    query.set('max_flow_tl', params.maxFlowTl.toString());
  if (params.minD1Return !== undefined && params.minD1Return !== null)
    query.set('min_d1_return', params.minD1Return.toString());
  if (params.maxD1Return !== undefined && params.maxD1Return !== null)
    query.set('max_d1_return', params.maxD1Return.toString());
  if (params.limit) query.set('limit', params.limit.toString());

  const res = await fetch(`${BASE_URL}/api/v1/event-study/scan?${query.toString()}`);
  if (!res.ok) throw new Error('Failed to scan event study');
  return res.json();
}

export async function fetchTimeWindowAnalysis(
  symbol: string,
  brokerId: string = 'MLB',
  tradeDate?: string
): Promise<TimeWindowAnalysisResponse> {
  let url = `${BASE_URL}/api/v1/time-window/analyze?symbol=${encodeURIComponent(symbol)}&broker_id=${encodeURIComponent(brokerId)}`;
  if (tradeDate) url += `&trade_date=${tradeDate}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to analyze time windows');
  return res.json();
}

export async function fetchAllSignals(): Promise<AllSignalsResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/signals/all`);
  if (!res.ok) throw new Error('Failed to fetch predictive signals');
  return res.json();
}

export async function fetchShockDays(
  thresholdPct: number = 0.03,
  fromDate?: string,
  toDate?: string
): Promise<ShockDayItem[]> {
  let url = `${BASE_URL}/api/v1/market/shock-days?threshold_pct=${thresholdPct}`;
  if (fromDate) url += `&from_date=${encodeURIComponent(fromDate)}`;
  if (toDate) url += `&to_date=${encodeURIComponent(toDate)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch shock days');
  return res.json();
}

