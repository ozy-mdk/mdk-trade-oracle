import {
  InstrumentItem,
  BrokerItem,
  DateRangeResponse,
  CandleBar,
  MarketSummaryResponse,
  TertipPortfolioResponse,
  TertipLotItem,
  TertipHistoryPoint,
  TertipHorizonsResponse,
  TertipTimeseriesPoint,
  ShockDayItem,
  TertipMlForecastResponse,
  OpportunityActionsResponse,
  MultiHorizonForecastResponse,
  MultiHorizonBacktestItem,
  MultiHorizonOpportunitiesResponse,
  StockYtdPnlResponse,
  YtdConstituentOverviewItem,
  IntradayUniverseItem,
  IntradaySignalResponse,
  IntradayLedgerResponse,
  IntradayTournamentResponse,
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

export async function fetchTertipMlForecast(
  symbol: string,
  fresh: boolean = false,
  modelType: string = 'auto',
  targetDate?: string,
): Promise<TertipMlForecastResponse> {
  let url = `${BASE_URL}/api/v1/tertip/ml-forecast?symbol=${encodeURIComponent(symbol)}&model_type=${encodeURIComponent(modelType)}`;
  if (fresh) {
    url += `&fresh=true&_t=${Date.now()}`;
  }
  if (targetDate) {
    url += `&target_date=${encodeURIComponent(targetDate)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch tertip ML forecast');
  return res.json();
}

export async function fetchPredictedOpportunities(
  horizon: string = '1d',
  targetDate?: string
): Promise<OpportunityActionsResponse> {
  let url = `${BASE_URL}/api/v1/tertip/opportunities?horizon=${encodeURIComponent(horizon)}`;
  if (targetDate) {
    url += `&target_date=${encodeURIComponent(targetDate)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch predicted opportunities');
  return res.json();
}


export async function fetchMultiHorizonForecast(
  symbol: string,
  fresh: boolean = false,
  evalSessions: number = 60,
  targetDate?: string,
): Promise<MultiHorizonForecastResponse> {
  let url = `${BASE_URL}/api/v1/tertip/multi-horizon/forecast?symbol=${encodeURIComponent(symbol)}&fresh=${fresh}&eval_sessions=${evalSessions}`;
  if (targetDate) {
    url += `&target_date=${encodeURIComponent(targetDate)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch multi-horizon forecast');
  return res.json();
}

export async function fetchMultiHorizonBacktest(
  symbol: string,
  horizon: string = '3d',
  limit: number = 60,
  targetDate?: string,
): Promise<MultiHorizonBacktestItem[]> {
  let url = `${BASE_URL}/api/v1/tertip/multi-horizon/backtest?symbol=${encodeURIComponent(symbol)}&horizon=${encodeURIComponent(horizon)}&limit=${limit}`;
  if (targetDate) {
    url += `&target_date=${encodeURIComponent(targetDate)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch multi-horizon backtest');
  return res.json();
}

export async function fetchMultiHorizonOpportunities(
  horizon: string = '5d'
): Promise<MultiHorizonOpportunitiesResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/tertip/multi-horizon/opportunities?horizon=${encodeURIComponent(horizon)}`);
  if (!res.ok) throw new Error('Failed to fetch multi-horizon opportunities');
  return res.json();
}

export async function fetchStockYtdPnl(
  symbol: string,
  tradeDate?: string
): Promise<StockYtdPnlResponse> {
  let url = `${BASE_URL}/api/v1/tertip/ytd-pnl?symbol=${encodeURIComponent(symbol)}`;
  if (tradeDate) {
    url += `&trade_date=${encodeURIComponent(tradeDate)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch YTD PnL for ${symbol}`);
  return res.json();
}

export async function fetchYtdOverview(
  tradeDate?: string
): Promise<YtdConstituentOverviewItem[]> {
  let url = `${BASE_URL}/api/v1/tertip/ytd-pnl/overview`;
  if (tradeDate) {
    url += `?trade_date=${encodeURIComponent(tradeDate)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch YTD PnL overview');
  return res.json();
}



// Intraday Session Forecaster (11:30 -> close)

export async function fetchIntradayUniverse(): Promise<IntradayUniverseItem[]> {
  const res = await fetch(`${BASE_URL}/api/v1/intraday/universe`);
  if (!res.ok) throw new Error('Failed to fetch intraday universe');
  return res.json();
}

export async function fetchIntradaySignal(symbol: string): Promise<IntradaySignalResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/intraday/signal?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error(`No intraday signal for ${symbol}`);
  return res.json();
}

export async function fetchIntradayLedger(symbol: string, limit: number = 2000): Promise<IntradayLedgerResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/intraday/ledger?symbol=${encodeURIComponent(symbol)}&limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch intraday ledger for ${symbol}`);
  return res.json();
}

export async function fetchIntradayTournament(symbol: string): Promise<IntradayTournamentResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/intraday/tournament?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error(`Failed to fetch intraday tournament for ${symbol}`);
  return res.json();
}
