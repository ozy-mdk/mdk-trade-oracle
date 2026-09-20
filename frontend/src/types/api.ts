/** Type definitions matching MDK Trading Oracle FastAPI backend schemas */

export interface InstrumentItem {
  symbol: string;
  name: string;
  sector: string;
  index_name: string;
}

export interface BrokerItem {
  broker_id: string;
  broker_name: string;
  category: string;
  is_primary_target: boolean;
}

export interface CandleBar {
  time: number; // Unix timestamp in seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  turnover_tl: number;
  trade_count: number;
  bofa_net_flow_tl: number;
}

export interface MarketSummaryResponse {
  symbol: string;
  broker_id: string;
  trade_date: string;
  close_price: number;
  daily_return_pct: number;
  total_turnover_tl: number;
  broker_buy_turnover_tl: number;
  broker_sell_turnover_tl: number;
  broker_net_flow_tl: number;
  broker_share_pct: number;
  matched_volume: number;
  intraday_realized_pnl_tl: number;
  carry_fifo_realized_pnl_tl: number;
  daily_realized_pnl_tl: number;
  cumulative_realized_pnl_tl: number;
  open_stock_quantity: number;
  fifo_avg_cost: number;
  market_value_tl: number;
  unrealized_pnl_tl: number;
  bias_badge: string;
}

export interface TertipPosition {
  symbol: string;
  symbol_name: string;
  sector: string;
  open_stock_quantity: number;
  fifo_avg_cost: number;
  market_close_price: number;
  market_value_tl: number;
  unrealized_pnl_tl: number;
  unrealized_pnl_pct: number;
  daily_realized_pnl_tl: number;
  cumulative_realized_pnl_tl: number;
  position_side: string;
}

export interface TertipPortfolioResponse {
  broker_id: string;
  trade_date: string;
  total_market_value_tl: number;
  total_unrealized_pnl_tl: number;
  total_daily_realized_pnl_tl: number;
  total_cumulative_realized_pnl_tl: number;
  positions: TertipPosition[];
}

export interface TertipLotItem {
  lot_id: string;
  symbol: string;
  direction: string;
  open_date: string;
  remaining_quantity: number;
  unit_cost: number;
  remaining_value_tl: number;
  days_held: number;
}

export interface TertipHistoryPoint {
  trade_date: string;
  daily_realized_pnl_tl: number;
  intraday_pnl_tl: number;
  carry_pnl_tl: number;
  net_flow_tl: number;
  mtm_valuation_tl: number;
  cumulative_realized_pnl_tl: number;
}

export interface EventStudyScanItem {
  trade_date: string;
  symbol: string;
  close_price: number;
  daily_return_pct: number;
  d1_return_pct?: number | null;
  bofa_net_flow_tl: number;
  total_turnover_tl: number;
  return_t1?: number | null;
  return_t2?: number | null;
  return_t3?: number | null;
  return_t5?: number | null;
  return_t10?: number | null;
}

export interface TimeWindowItem {
  window_name: string;
  window_order: number;
  start_time: string;
  end_time: string;
  buy_turnover_tl: number;
  sell_turnover_tl: number;
  net_flow_tl: number;
  total_turnover_tl: number;
  buy_volume: number;
  sell_volume: number;
  net_volume: number;
}

export interface TimeWindowAnalysisResponse {
  symbol: string;
  broker_id: string;
  trade_date: string;
  windows: TimeWindowItem[];
  opening_auction_net_tl?: number | null;
  closing_auction_net_tl?: number | null;
}

export interface MacroSignalResponse {
  forecast_date: string;
  predicted_open_net_flow_tl: number;
  predicted_open_flow_lower_90: number;
  predicted_open_flow_upper_90: number;
  predicted_direction: string;
  direction_confidence: number;
  predicted_playbook: string;
  top_predicted_buy_sector?: string | null;
  top_predicted_sell_sector?: string | null;
  model_name: string;
  model_version: string;
}

export interface SectorAllocationItem {
  forecast_date: string;
  sector: string;
  predicted_open_net_flow_tl: number;
  predicted_lower_90: number;
  predicted_upper_90: number;
  direction: string;
  confidence: number;
  playbook: string;
}

export interface StockReactionForecastItem {
  forecast_date: string;
  symbol: string;
  window_name: string;
  predicted_return_pct: number;
  predicted_return_lower_90: number;
  predicted_return_upper_90: number;
  predicted_direction: string;
  direction_confidence: number;
  predicted_playbook: string;
}

export interface AllSignalsResponse {
  forecast_date: string;
  macro_day_start?: MacroSignalResponse | null;
  sector_allocations: SectorAllocationItem[];
  stock_reactions: StockReactionForecastItem[];
}
