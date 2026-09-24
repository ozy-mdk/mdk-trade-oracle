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

export interface DateRangeResponse {
  min_date: string;
  max_date: string;
  latest_date: string;
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

export interface TertipDiagnostic {
  diagnostic_badge: string;
  badge_color: string;
  conviction_pct: number;
  headline: string;
  rationale: string;
}

export interface TertipHorizonItem {
  code: string;
  label: string;
  lookback_days: number;
  cum_net_flow_tl: number;
  cum_net_shares: number;
  ewma_inventory_qty: number;
  ewma_unit_cost: number;
  cost_spread_pct: number;
  saturation_pct: number;
  stance: string;
  description: string;
}

export interface TertipHorizonsResponse {
  symbol: string;
  broker_id: string;
  trade_date: string;
  market_close_price: number;
  day_net_flow_tl: number;
  day_buy_turnover_tl: number;
  day_sell_turnover_tl: number;
  open_stock_quantity: number;
  market_value_tl: number;
  fifo_avg_cost: number;
  unrealized_pnl_tl: number;
  unrealized_pnl_pct: number;
  matched_volume_pct: number;
  global_saturation_pct: number;
  ribbon_status: string;
  diagnostic: TertipDiagnostic;
  horizons: TertipHorizonItem[];
}

export interface TertipTimeseriesPoint {
  time: number;
  trade_date: string;
  close_price: number;
  fifo_avg_cost: number;
  ewma_cost_5d?: number;
  ewma_cost_10d?: number;
  ewma_cost_21d?: number;
  ewma_cost_63d: number;
  ewma_cost_126d: number;
  ewma_cost_252d?: number;
  open_quantity: number;
  ewma_qty_5d: number;
  ewma_qty_10d: number;
  ewma_qty_21d: number;
  ewma_qty_63d: number;
  ewma_qty_126d: number;
  ewma_qty_252d: number;
  net_flow_tl: number;
  unrealized_pnl_tl: number;
}

export type ForwardSignalSeverity = 'NEUTRAL' | 'MODERATE' | 'STRONG';
export type ForwardSignalDirection = 'BUY' | 'SELL' | 'NEUTRAL';
export type ForwardHorizonCode = '1W' | '2W' | '1M' | '3M' | '6M' | 'FIFO';

export interface ForwardOpportunityOutlook {
  horizonCode: ForwardHorizonCode;
  horizonLabel: string;
  closePrice: number;
  targetCost: number;
  spreadPct: number;
  potentialReturnPct: number;
  direction: ForwardSignalDirection;
  severity: ForwardSignalSeverity;
  badgeLabel: string;
  badgeColor: string;
  playbook: string;
  rationale: string;
}

export interface EwmaHorizonSummaryRow {
  code: ForwardHorizonCode;
  label: string;
  ewmaCost: number;
  spreadPct: number;
  potentialReturnPct: number;
  direction: ForwardSignalDirection;
  severity: ForwardSignalSeverity;
  netFlowTl: number;
  stance: string;
  inventoryQty: number;
}

export interface EwmaConfluenceSummary {
  buyCount: number;
  sellCount: number;
  neutralCount: number;
  totalHorizons: number;
  overallDirection: ForwardSignalDirection;
  confluenceLabel: string;
  ribbonStatus: string;
  rows: EwmaHorizonSummaryRow[];
}

export interface ShockDayItem {
  trade_date: string;
  time: number; // Unix timestamp in seconds
  close_price: number;
  daily_return_pct: number;
  total_turnover_tl: number;
  bofa_net_flow_tl: number;
  turnover_change_pct: number;
  shock_type: 'POSITIVE_SHOCK' | 'NEGATIVE_SHOCK';
  is_positive_shock: boolean;
  is_negative_shock: boolean;
  shock_magnitude_pct: number;
}



