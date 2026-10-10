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


export interface TertipDiagnostic {
  diagnostic_badge: string;
  badge_color: string;
  conviction_pct: number;
  headline: string;
  rationale: string;
}

export interface HorizonRealization12M {
  active_stance: string;
  total_occurrences: number;
  realized_count: number;
  opposite_count: number;
  neutral_count: number;
  realized_pct: number;
  opposite_pct: number;
  neutral_pct: number;
  next_day_buy_count: number;
  next_day_sell_count: number;
  next_day_buy_pct: number;
  next_day_sell_pct: number;
  avg_next_day_flow_tl: number;
  price_up_count: number;
  price_down_count: number;
  price_up_pct: number;
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
  realization_12m?: HorizonRealization12M | null;
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
  fifo_realization_12m?: HorizonRealization12M | null;
  confluence_realization_12m?: HorizonRealization12M | null;
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
  realization_12m?: HorizonRealization12M | null;
}

export interface EwmaConfluenceSummary {
  buyCount: number;
  sellCount: number;
  neutralCount: number;
  totalHorizons: number;
  overallDirection: ForwardSignalDirection;
  confluenceLabel: string;
  ribbonStatus: string;
  confluenceRealization?: HorizonRealization12M | null;
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

export interface PillarMatrixItem {
  pillar: string;
  name: string;
  desc: string;
  stance: string;
  realized_pct: number;
  opposite_pct: number;
  expected_action: string;
  expected_flow_tl: number;
  turnover_share_pct: number;
  price_up_pct: number;
}

export interface WalkForwardLedgerItem {
  date: string;
  actual_price: number;
  actual_return_pct: number;
  bist30_ret_pct?: number;
  ml_pred_price: number;
  ml_pred_return_pct?: number;
  ml_direction?: 'UP' | 'DOWN' | 'FLAT' | string;
  ml_err_pct: number;
  ml_is_hit: boolean;
  confluence_pred_price?: number;
  confluence_pred_return_pct?: number;
  confluence_direction?: 'UP' | 'DOWN' | 'FLAT' | string;
  confluence_err_pct?: number;
  confluence_is_hit?: boolean;
  champion_pred_price?: number;
  champion_pred_return_pct?: number;
  champion_direction?: 'UP' | 'DOWN' | 'FLAT' | string;
  champion_err_pct?: number;
  champion_is_hit?: boolean;
  convex_weight_ml?: number;
  convex_weight_prophet?: number;
  prophet_pred_price: number;
  prophet_pred_return_pct?: number;
  prophet_direction?: 'UP' | 'DOWN' | 'FLAT' | string;
  prophet_err_pct: number;
  prophet_is_hit: boolean;
  winner: 'CHALLENGER' | 'BASE';
  mlb_action: string;
  mlb_flow_tl: number;
  mlb_buy_tl?: number;
  mlb_sell_tl?: number;
  mlb_pnl_tl?: number;
  big5_action: string;
  big5_flow_tl: number;
  big5_buy_tl?: number;
  big5_sell_tl?: number;
  big5_pnl_tl?: number;
  kamu_action: string;
  kamu_flow_tl: number;
  kamu_buy_tl?: number;
  kamu_sell_tl?: number;
  kamu_pnl_tl?: number;
  actual_training_sessions?: number;
  actual_training_months?: number;
  data_sufficiency_status?: 'FULL' | 'PARTIAL_HISTORY' | 'PARTIAL_ADAPTED' | string;
}

export interface TournamentSummary {
  champion: string;
  champion_label: string;
  grand_champion_key?: string;
  ml_champion_type?: string;
  crowned_horizon?: string;
  training_lookback_sessions?: number;
  selection_window_sessions?: number;
  sessions_skipped_insufficient?: number;
  sessions_with_partial_history?: number;
  sessions_with_full_history?: number;
  data_sufficiency_pct?: number;
  actual_training_sessions?: number;
  actual_training_months?: number;
  data_sufficiency_status?: string;
  convex_weight_ml?: number;
  convex_weight_prophet?: number;
  champion_dir_hits?: number;
  champion_dir_hit_rate_pct?: number;
  runner_up_dir_hit_rate_pct?: number;
  champion_mae_pct?: number;
  champion_hit_mae_pct?: number;
  champion_miss_mae_pct?: number;
  champion_tournament_loss?: number;
  champion_30d_hits?: number;
  champion_30d_hit_rate_pct?: number;
  champion_30d_mae_pct?: number;
  champion_30d_hit_mae_pct?: number;
  champion_30d_miss_mae_pct?: number;
  champion_30d_big_move_hits?: number;
  champion_30d_big_move_total?: number;
  champion_30d_big_move_hit_rate_pct?: number;
  champion_30d_big_move_signalled?: number;
  champion_30d_big_move_missed?: number;
  champion_30d_big_move_false_alarms?: number;
  champion_30d_sig_move_hits?: number;
  champion_30d_sig_move_total?: number;
  champion_30d_sig_move_hit_rate_pct?: number;
  champion_30d_quiet_false_alarms?: number;
  champion_30d_penalty_loss?: number;
  champion_penalty_loss?: number;
  champion_full_hits?: number;
  champion_full_hit_rate_pct?: number;
  champion_full_mae_pct?: number;
  champion_full_hit_mae_pct?: number;
  champion_full_miss_mae_pct?: number;
  champion_full_big_move_hits?: number;
  champion_full_big_move_total?: number;
  champion_full_big_move_hit_rate_pct?: number;
  champion_full_penalty_loss?: number;
  confluence_30d_hits?: number;
  confluence_30d_hit_rate_pct?: number;
  confluence_30d_mae_pct?: number;
  ml_dir_hits?: number;
  ml_dir_hit_rate_pct?: number;
  ml_hit_rate_pct: number;
  ml_hit_mae_pct?: number;
  ml_miss_mae_pct?: number;
  ml_30d_sig_move_hits?: number;
  ml_30d_sig_move_total?: number;
  ml_30d_sig_move_hit_rate_pct?: number;
  ml_30d_quiet_false_alarms?: number;
  ml_30d_big_move_hits?: number;
  ml_30d_big_move_total?: number;
  ml_30d_big_move_hit_rate_pct?: number;
  ml_penalty_loss_30d?: number;
  ml_tournament_loss_30d?: number;
  ml_full_hits?: number;
  ml_full_hit_rate_pct?: number;
  ml_full_big_move_hits?: number;
  ml_full_big_move_total?: number;
  ml_full_big_move_hit_rate_pct?: number;
  prophet_dir_hits?: number;
  prophet_dir_hit_rate_pct?: number;
  prophet_hit_rate_pct: number;
  prophet_hit_mae_pct?: number;
  prophet_miss_mae_pct?: number;
  prophet_30d_sig_move_hits?: number;
  prophet_30d_sig_move_total?: number;
  prophet_30d_sig_move_hit_rate_pct?: number;
  prophet_30d_quiet_false_alarms?: number;
  prophet_30d_big_move_hits?: number;
  prophet_30d_big_move_total?: number;
  prophet_30d_big_move_hit_rate_pct?: number;
  prophet_penalty_loss_30d?: number;
  prophet_tournament_loss_30d?: number;
  prophet_full_hits?: number;
  prophet_full_hit_rate_pct?: number;
  prophet_full_big_move_hits?: number;
  prophet_full_big_move_total?: number;
  prophet_full_big_move_hit_rate_pct?: number;
  ridge_30d_hit_rate_pct?: number;
  ridge_dir_hits?: number;
  ridge_dir_hit_rate_pct?: number;
  ridge_hit_rate_pct?: number;
  ridge_mae_pct?: number;
  ridge_hit_mae_pct?: number;
  ridge_miss_mae_pct?: number;
  ridge_30d_sig_move_hits?: number;
  ridge_30d_sig_move_total?: number;
  ridge_30d_sig_move_hit_rate_pct?: number;
  ridge_30d_quiet_false_alarms?: number;
  ridge_30d_big_move_hits?: number;
  ridge_30d_big_move_total?: number;
  ridge_30d_big_move_hit_rate_pct?: number;
  ridge_penalty_loss_30d?: number;
  ridge_tournament_loss_30d?: number;
  xgboost_30d_hit_rate_pct?: number;
  xgboost_dir_hits?: number;
  xgboost_dir_hit_rate_pct?: number;
  xgboost_hit_rate_pct?: number;
  xgboost_mae_pct?: number;
  xgboost_hit_mae_pct?: number;
  xgboost_miss_mae_pct?: number;
  xgboost_30d_sig_move_hits?: number;
  xgboost_30d_sig_move_total?: number;
  xgboost_30d_sig_move_hit_rate_pct?: number;
  xgboost_30d_quiet_false_alarms?: number;
  xgboost_30d_big_move_hits?: number;
  xgboost_30d_big_move_total?: number;
  xgboost_30d_big_move_hit_rate_pct?: number;
  xgboost_penalty_loss_30d?: number;
  xgboost_tournament_loss_30d?: number;
  lightgbm_30d_hit_rate_pct?: number;
  lightgbm_dir_hits?: number;
  lightgbm_dir_hit_rate_pct?: number;
  lightgbm_hit_rate_pct?: number;
  lightgbm_mae_pct?: number;
  lightgbm_hit_mae_pct?: number;
  lightgbm_miss_mae_pct?: number;
  lightgbm_30d_sig_move_hits?: number;
  lightgbm_30d_sig_move_total?: number;
  lightgbm_30d_sig_move_hit_rate_pct?: number;
  lightgbm_30d_quiet_false_alarms?: number;
  lightgbm_30d_big_move_hits?: number;
  lightgbm_30d_big_move_total?: number;
  lightgbm_30d_big_move_hit_rate_pct?: number;
  lightgbm_penalty_loss_30d?: number;
  lightgbm_tournament_loss_30d?: number;
  huber_30d_hit_rate_pct?: number;
  huber_dir_hits?: number;
  huber_dir_hit_rate_pct?: number;
  huber_hit_rate_pct?: number;
  huber_mae_pct?: number;
  huber_hit_mae_pct?: number;
  huber_miss_mae_pct?: number;
  huber_30d_sig_move_hits?: number;
  huber_30d_sig_move_total?: number;
  huber_30d_sig_move_hit_rate_pct?: number;
  huber_30d_quiet_false_alarms?: number;
  huber_30d_big_move_hits?: number;
  huber_30d_big_move_total?: number;
  huber_30d_big_move_hit_rate_pct?: number;
  huber_penalty_loss_30d?: number;
  huber_tournament_loss_30d?: number;
  bayesian_ridge_30d_hit_rate_pct?: number;
  bayesian_ridge_dir_hits?: number;
  bayesian_ridge_dir_hit_rate_pct?: number;
  bayesian_ridge_hit_rate_pct?: number;
  bayesian_ridge_mae_pct?: number;
  bayesian_ridge_hit_mae_pct?: number;
  bayesian_ridge_miss_mae_pct?: number;
  bayesian_ridge_30d_sig_move_hits?: number;
  bayesian_ridge_30d_sig_move_total?: number;
  bayesian_ridge_30d_sig_move_hit_rate_pct?: number;
  bayesian_ridge_30d_quiet_false_alarms?: number;
  bayesian_ridge_30d_big_move_hits?: number;
  bayesian_ridge_30d_big_move_total?: number;
  bayesian_ridge_30d_big_move_hit_rate_pct?: number;
  bayesian_ridge_penalty_loss_30d?: number;
  bayesian_ridge_tournament_loss_30d?: number;
  ml_mae_pct: number;
  prophet_mae_pct: number;
  ml_error_wins?: number;
  prophet_error_wins?: number;
  ml_wins: number;
  prophet_wins: number;
  total_sessions: number;
}

export interface PillarExecutionDetail {
  buy_tl: number;
  sell_tl: number;
  net_flow_tl: number;
  daily_pnl_tl: number;
  unrealized_pnl_tl: number;
}

export interface PillarExecutionPillar {
  today: PillarExecutionDetail;
  yesterday: PillarExecutionDetail;
}

export interface PillarExecutionTodayYesterday {
  today_date: string;
  yesterday_date: string;
  mlb: PillarExecutionPillar;
  big5: PillarExecutionPillar;
  kamu: PillarExecutionPillar;
}

export interface Bist30Trend {
  today_pct: number;
  yesterday_pct: number;
  day_before_pct: number;
}

export interface TertipMlForecastResponse {
  symbol: string;
  as_of_date: string;
  latest_close_price: number;
  target_price: number;
  expected_return_pct: number;
  confluence_target_price?: number;
  confluence_expected_return_pct?: number;
  convex_weight_ml?: number;
  convex_weight_prophet?: number;
  ml_target_price?: number;
  ml_expected_return_pct?: number;
  ml_stance?: string;
  ml_stance_badge?: string;
  ml_stance_color?: string;
  price_low: number;
  price_high: number;
  stance: string;
  stance_badge: string;
  stance_color: string;
  prophet_target_price: number;
  prophet_expected_return_pct: number;
  prophet_stance?: string;
  prophet_stance_badge?: string;
  prophet_stance_color?: string;
  days_since_last_positive_shock?: number;
  days_since_last_negative_shock?: number;
  bist30_trend?: Bist30Trend;
  playbook_headline: string;
  playbook_rationale: string;
  tournament_summary: TournamentSummary;
  pillar_matrix: PillarMatrixItem[];
  pillar_execution?: PillarExecutionTodayYesterday | null;
  walk_forward_ledger: WalkForwardLedgerItem[];
  features_mode?: string;
  active_features_count?: number;
  train_lookback_sessions?: number;
  actual_training_sessions?: number;
  actual_training_months?: number;
  target_training_months?: number;
  actual_window_desc?: string;
  data_sufficiency_status?: string;
  active_features?: string[];
  excluded_features?: string[];
  calculated_at: string;
}

export interface OpportunityItem {
  symbol: string;
  company_name: string;
  sector: string;
  as_of_date: string;
  current_price: number;
  target_price: number;
  expected_return_pct: number;
  price_low: number;
  price_high: number;
  stance: string;
  conviction?: string;
  playbook?: string;
  ml_champion_type?: string;
  champion_dir_hits?: number;
  champion_dir_hit_rate_pct?: number;
  champion_mae_pct?: number;
  crowned_horizon?: string;
  training_lookback_sessions?: number;
  actual_training_sessions?: number;
  actual_window_desc?: string;
  data_sufficiency_status?: string;
  mlb_net_flow_tl: number;
  mlb_turnover_tl: number;
  tier: 'HIGH_CONVICTION_LONG' | 'HIGH_CONVICTION_SHORT' | 'MODERATE_LONG' | 'MODERATE_SHORT' | 'MILD_LONG' | 'MILD_SHORT' | 'CONSOLIDATION' | string;
  action_type: 'LONG' | 'SHORT' | 'NEUTRAL' | string;
}

export interface OpportunityActionsResponse {
  as_of_date: string;
  total_constituents: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  avg_expected_return_pct: number;
  high_conviction_count: number;
  significant_moves_count: number;
  top_longs: OpportunityItem[];
  top_shorts: OpportunityItem[];
  opportunities: OpportunityItem[];
}

export interface WeekStartBacktestItem {
  trade_date: string;
  prior_date: string;
  actual_price: number;
  actual_return_pct: number;
  bist30_ret_pct: number;
  ml_pred_price: number;
  ml_pred_return_pct: number;
  ml_direction: string;
  ml_err_pct: number;
  ml_is_hit: boolean;
  weekend_carry_cost_bps: number;
  fri_w5_mlb_share_pct: number;
  wtd_mlb_net_flow_tl: number;
  ml_champion_type: string;
  actual_training_weeks?: number;
  actual_training_months?: number;
  data_sufficiency_status?: string;
}

export interface WeekStartForecastResponse {
  symbol: string;
  company_name: string;
  sector: string;
  as_of_date: string;
  target_date: string;
  current_price: number;
  target_price: number;
  expected_return_pct: number;
  price_low: number;
  price_high: number;
  stance: string;
  conviction: string;
  playbook: string;
  ml_champion_type: string;
  champion_dir_hits?: number;
  champion_dir_hit_rate_pct?: number;
  champion_mae_pct?: number;
  weekend_carry_cost_bps: number;
  fri_w5_mlb_share_pct: number;
  wtd_mlb_net_flow_tl: number;
  training_lookback_weeks: number;
  crowned_horizon: string;
  actual_training_weeks?: number;
  actual_training_months?: number;
  target_training_months?: number;
  actual_window_desc?: string;
  data_sufficiency_status?: string;
  calculated_at: string;
  backtest_ledger: WeekStartBacktestItem[];
}

export interface WeekStartOpportunityItem {
  symbol: string;
  company_name: string;
  sector: string;
  as_of_date: string;
  target_date: string;
  current_price: number;
  target_price: number;
  expected_return_pct: number;
  price_low: number;
  price_high: number;
  stance: string;
  conviction?: string;
  playbook?: string;
  ml_champion_type?: string;
  champion_dir_hits?: number;
  champion_dir_hit_rate_pct?: number;
  champion_mae_pct?: number;
  crowned_horizon?: string;
  training_lookback_weeks?: number;
  actual_training_weeks?: number;
  actual_training_months?: number;
  actual_window_desc?: string;
  data_sufficiency_status?: string;
  weekend_carry_cost_bps: number;
  fri_w5_mlb_share_pct: number;
  wtd_mlb_net_flow_tl: number;
  tier: string;
  action_type: string;
}

export interface WeekStartOpportunitiesResponse {
  as_of_date: string;
  target_date: string;
  total_constituents: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  avg_expected_return_pct: number;
  high_conviction_count: number;
  significant_moves_count: number;
  avg_weekend_carry_bps: number;
  top_longs: WeekStartOpportunityItem[];
  top_shorts: WeekStartOpportunityItem[];
  opportunities: WeekStartOpportunityItem[];
}

export interface MultiHorizonBacktestItem {
  trade_date: string;
  target_date_start: string;
  target_date_end: string;
  current_price: number;
  actual_avg_price: number;
  actual_return_pct: number;
  pred_avg_price: number;
  pred_return_pct: number;
  pred_direction: string;
  actual_direction: string;
  err_pct: number;
  is_hit: boolean;
  is_pending?: boolean;
  champion_model?: string;
  actual_training_sessions?: number;
  target_lookback_sessions?: number;
  actual_training_months?: number;
  data_sufficiency_status?: string;
}

export interface MultiHorizonMeta {
  horizon: string;
  horizon_days: number;
  crowned_model: string;
  crowned_horizon: string;
  training_lookback_sessions: number;
  target_lookback_months?: number;
  target_lookback_sessions?: number;
  min_required_train_sessions?: number;
  hit_rate_pct: number;
  hits: number;
  total_evals: number;
  eval_sessions_requested?: number;
  sessions_skipped_insufficient?: number;
  sessions_with_partial_history?: number;
  sessions_with_full_history?: number;
  min_training_sessions_used?: number;
  max_training_sessions_used?: number;
  data_sufficiency_pct?: number;
  mae_pct: number;
  hit_mae_pct: number;
  miss_mae_pct: number;
  loss: number;
}

export interface MultiHorizonLiveForecast {
  horizon: string;
  horizon_days: number;
  as_of_date: string;
  current_price: number;
  target_price: number;
  expected_return_pct: number;
  price_low: number;
  price_high: number;
  stance: string;
  conviction: string;
  playbook: string;
  champion_model: string;
  training_lookback_sessions: number;
  actual_training_sessions?: number;
  actual_training_months?: number;
  target_training_months?: number;
  actual_window_desc?: string;
  data_sufficiency_status?: string;
}

export interface MultiHorizonItem {
  meta: MultiHorizonMeta;
  live_forecast: MultiHorizonLiveForecast;
  backtest_ledger: MultiHorizonBacktestItem[];
}

export interface MultiHorizonForecastResponse {
  status: string;
  symbol: string;
  company_name?: string;
  sector?: string;
  as_of_date: string;
  current_price: number;
  calculated_at: string;
  horizons: Record<string, MultiHorizonItem>;
}

export interface MultiHorizonOpportunitiesResponse {
  horizon: string;
  total_constituents: number;
  opportunities: Array<{
    symbol: string;
    company_name: string;
    sector: string;
    as_of_date: string;
    current_price: number;
    target_price: number;
    expected_return_pct: number;
    price_low: number;
    price_high: number;
    stance: string;
    conviction: string;
    playbook: string;
    ml_champion_type: string;
    champion_dir_hits: number;
    champion_total_evals: number;
    champion_dir_hit_rate_pct: number;
    champion_mae_pct: number;
    crowned_horizon: string;
  }>;
}

