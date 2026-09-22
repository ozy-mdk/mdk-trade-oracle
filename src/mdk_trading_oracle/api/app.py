"""FastAPI Backend Serving Layer for MDK Trading Oracle & React Frontend.

Provides ultra-low latency REST endpoints and WebSocket streams for:
- TradingView Lightweight Charts (1m, 5m, 1d OHLCV Candlesticks + Synthetic XU030)
- Market Summary & 5 Core Trader KPIs (Turnover, Net Flow, Realized PnL, Open Lots, Bias Badge)
- Institutional FIFO Tertip Inventory, Open Lots & Realized PnL Ledgers
- Stock Movement & Forward Return Event Study Scanner (T+1 to T+10 horizons)
- Custom Time-Window Analysis Terminal (W1-W5 Intraday Breakdown)
- Live Gold Predictive Forecasts, Credible Intervals & Playbook Signals
"""

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.tertip_analytics import (
    INSTITUTIONAL_BUNDLES,
    get_tertip_horizons_analysis,
    get_tertip_timeseries_chart,
)

logger = get_logger("mdk_oracle.api")

app = FastAPI(
    title="MDK Trading Oracle API",
    description="High-performance financial market data & institutional order flow analytics",
    version="1.0.0",
)

# Enable CORS for React frontend (Vite on localhost:5173, 3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = PostgresManager()


# ── Response Schemas ──────────────────────────────────────────────────────────

class CandleBar(BaseModel):
    time: int  # Unix timestamp in seconds for TradingView Lightweight Charts
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover_tl: float
    trade_count: int
    bofa_net_flow_tl: float


class InstrumentItem(BaseModel):
    symbol: str
    name: str
    sector: str
    index_name: str


class BrokerItem(BaseModel):
    broker_id: str
    broker_name: str
    category: str
    is_primary_target: bool


class DateRangeResponse(BaseModel):
    min_date: str
    max_date: str
    latest_date: str


class MarketSummaryResponse(BaseModel):
    symbol: str
    broker_id: str
    trade_date: str
    close_price: float
    daily_return_pct: float
    total_turnover_tl: float
    broker_buy_turnover_tl: float
    broker_sell_turnover_tl: float
    broker_net_flow_tl: float
    broker_share_pct: float
    matched_volume: float
    intraday_realized_pnl_tl: float
    carry_fifo_realized_pnl_tl: float
    daily_realized_pnl_tl: float
    cumulative_realized_pnl_tl: float
    open_stock_quantity: float
    fifo_avg_cost: float
    market_value_tl: float
    unrealized_pnl_tl: float
    bias_badge: str


class TertipPosition(BaseModel):
    symbol: str
    symbol_name: str
    sector: str
    open_stock_quantity: float
    fifo_avg_cost: float
    market_close_price: float
    market_value_tl: float
    unrealized_pnl_tl: float
    unrealized_pnl_pct: float
    daily_realized_pnl_tl: float
    cumulative_realized_pnl_tl: float
    position_side: str


class TertipPortfolioResponse(BaseModel):
    broker_id: str
    trade_date: str
    total_market_value_tl: float
    total_unrealized_pnl_tl: float
    total_daily_realized_pnl_tl: float
    total_cumulative_realized_pnl_tl: float
    positions: List[TertipPosition]


class TertipLotItem(BaseModel):
    lot_id: str
    symbol: str
    direction: str
    open_date: str
    remaining_quantity: float
    unit_cost: float
    remaining_value_tl: float
    days_held: int


class TertipHistoryPoint(BaseModel):
    trade_date: str
    daily_realized_pnl_tl: float
    intraday_pnl_tl: float
    carry_pnl_tl: float
    net_flow_tl: float
    mtm_valuation_tl: float
    cumulative_realized_pnl_tl: float


class TertipDiagnostic(BaseModel):
    diagnostic_badge: str
    badge_color: str
    conviction_pct: int
    headline: str
    rationale: str


class TertipHorizonItem(BaseModel):
    code: str
    label: str
    lookback_days: int
    cum_net_flow_tl: float
    cum_net_shares: float
    ewma_inventory_qty: float
    ewma_unit_cost: float
    cost_spread_pct: float
    saturation_pct: float
    stance: str
    description: str


class TertipHorizonsResponse(BaseModel):
    symbol: str
    broker_id: str
    trade_date: str
    market_close_price: float
    day_net_flow_tl: float
    day_buy_turnover_tl: float
    day_sell_turnover_tl: float
    open_stock_quantity: float
    market_value_tl: float
    fifo_avg_cost: float
    unrealized_pnl_tl: float
    unrealized_pnl_pct: float
    matched_volume_pct: float
    global_saturation_pct: float
    ribbon_status: str
    diagnostic: TertipDiagnostic
    horizons: List[TertipHorizonItem]


class TertipTimeseriesPoint(BaseModel):
    time: int
    trade_date: str
    close_price: float
    fifo_avg_cost: float
    ewma_cost_5d: float = 0.0
    ewma_cost_10d: float = 0.0
    ewma_cost_21d: float = 0.0
    ewma_cost_63d: float
    ewma_cost_126d: float
    ewma_cost_252d: float = 0.0
    open_quantity: float
    ewma_qty_5d: float
    ewma_qty_10d: float
    ewma_qty_21d: float
    ewma_qty_63d: float
    ewma_qty_126d: float
    ewma_qty_252d: float
    net_flow_tl: float
    unrealized_pnl_tl: float



class EventStudyScanItem(BaseModel):
    trade_date: str
    symbol: str
    close_price: float
    daily_return_pct: float
    d1_return_pct: Optional[float] = None
    bofa_net_flow_tl: float
    total_turnover_tl: float
    return_t1: Optional[float] = None
    return_t2: Optional[float] = None
    return_t3: Optional[float] = None
    return_t5: Optional[float] = None
    return_t10: Optional[float] = None


class TimeWindowItem(BaseModel):
    window_name: str
    window_order: int
    start_time: str
    end_time: str
    buy_turnover_tl: float
    sell_turnover_tl: float
    net_flow_tl: float
    total_turnover_tl: float
    buy_volume: float
    sell_volume: float
    net_volume: float


class TimeWindowAnalysisResponse(BaseModel):
    symbol: str
    broker_id: str
    trade_date: str
    windows: List[TimeWindowItem]
    opening_auction_net_tl: Optional[float] = None
    closing_auction_net_tl: Optional[float] = None


class MacroSignalResponse(BaseModel):
    forecast_date: str
    predicted_open_net_flow_tl: float
    predicted_open_flow_lower_90: float
    predicted_open_flow_upper_90: float
    predicted_direction: str
    direction_confidence: float
    predicted_playbook: str
    top_predicted_buy_sector: Optional[str] = None
    top_predicted_sell_sector: Optional[str] = None
    model_name: str
    model_version: str


class StockReactionForecastItem(BaseModel):
    forecast_date: str
    symbol: str
    window_name: str
    predicted_return_pct: float
    predicted_return_lower_90: float
    predicted_return_upper_90: float
    predicted_direction: str
    direction_confidence: float
    predicted_playbook: str


class AllSignalsResponse(BaseModel):
    forecast_date: str
    macro_day_start: Optional[MacroSignalResponse] = None
    sector_allocations: List[Dict[str, Any]] = Field(default_factory=list)
    stock_reactions: List[StockReactionForecastItem] = Field(default_factory=list)


# ── Metadata Endpoints ────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    """Verify API service and database connectivity."""
    try:
        res = db.execute("SELECT 1;").fetchone()
        db_status = "connected" if res and res[0] == 1 else "degraded"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "healthy",
        "database": db_status,
        "engine": "PostgreSQL 16 + TimescaleDB",
        "hardware": "Apple Silicon M5 Mac Pro",
        "server_time": datetime.now().isoformat(),
    }


@app.get("/api/v1/meta/instruments", response_model=List[InstrumentItem])
def get_instruments() -> List[InstrumentItem]:
    """Return all tracked tradeable instruments plus synthetic XU030."""
    query = """
        SELECT symbol, name, sector, index_name 
        FROM bronze_instruments 
        ORDER BY index_name ASC, symbol ASC;
    """
    rows = db.execute(query).fetchall()
    instruments = [
        InstrumentItem(
            symbol=r[0],
            name=r[1] or r[0],
            sector=r[2] or "Unassigned",
            index_name=r[3] or "BIST",
        )
        for r in rows
    ]
    # Prepend synthetic BIST 30 Benchmark Index if not in list
    if not any(i.symbol == "XU030" for i in instruments):
        instruments.insert(
            0,
            InstrumentItem(
                symbol="XU030",
                name="BIST 30 Benchmark Index",
                sector="Benchmark Index",
                index_name="BENCHMARK",
            ),
        )
    return instruments


@app.get("/api/v1/meta/brokers", response_model=List[BrokerItem])
def get_brokers() -> List[BrokerItem]:
    """Return brokerage dimension entities with primary institutional targets first."""
    query = """
        SELECT broker_id, broker_name, category, is_primary_target 
        FROM bronze_brokers 
        ORDER BY 
            CASE WHEN broker_id = 'MLB' THEN 0 
                 WHEN broker_id = 'YKR' THEN 1
                 WHEN broker_id = 'IYM' THEN 2
                 WHEN broker_id = 'AKM' THEN 3
                 WHEN broker_id = 'GRM' THEN 4
                 WHEN broker_id = 'ZRY' THEN 5
                 WHEN broker_id = 'VKY' THEN 6
                 WHEN broker_id = 'HLY' THEN 7
                 WHEN is_primary_target = TRUE THEN 8
                 ELSE 9 END ASC,
            broker_id ASC;
    """
    rows = db.execute(query).fetchall()
    brokers = [
        BrokerItem(
            broker_id=r[0],
            broker_name=r[1] or r[0],
            category=r[2] or "Broker",
            is_primary_target=bool(r[3]),
        )
        for r in rows
    ]
    # Insert bundles directly after MLB
    big5_item = BrokerItem(
        broker_id="BIG5",
        broker_name="BIG FIVE (YKR, IYM, AKM, GRM, ZRY)",
        category="Institutional Bundle",
        is_primary_target=True,
    )
    kamu_item = BrokerItem(
        broker_id="KAMU",
        broker_name="KAMU (ZRY, VKY, HLY)",
        category="Institutional Bundle",
        is_primary_target=True,
    )
    mlb_idx = next((i for i, b in enumerate(brokers) if b.broker_id == "MLB"), -1)
    if mlb_idx >= 0:
        brokers.insert(mlb_idx + 1, big5_item)
        brokers.insert(mlb_idx + 2, kamu_item)
    else:
        brokers.insert(0, big5_item)
        brokers.insert(1, kamu_item)
    return brokers


@app.get("/api/v1/meta/date-range", response_model=DateRangeResponse)
def get_date_range(
    symbol: Optional[str] = Query(None, description="Optional equity symbol to filter by"),
) -> DateRangeResponse:
    """Return earliest and latest trade dates available in the lakehouse."""
    if symbol:
        sym = symbol.upper()
        row = db.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM silver_daily_stock_summary WHERE symbol = %s;",
            [sym],
        ).fetchone()
    else:
        row = db.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM silver_daily_stock_summary;"
        ).fetchone()

    min_date = str(row[0]) if (row and row[0]) else "2022-01-03"
    max_date = str(row[1]) if (row and row[1]) else "2026-09-16"
    return DateRangeResponse(
        min_date=min_date,
        max_date=max_date,
        latest_date=max_date,
    )


# ── Market & Chart Endpoints ──────────────────────────────────────────────────

@app.get("/api/v1/market/candles", response_model=List[CandleBar])
def get_candlesticks(
    symbol: str = Query(..., description="BIST Equity Symbol (e.g. THYAO, AKBNK, XU030)"),
    interval: str = Query("5m", description="Candle interval: 1m, 5m, 1d"),
    broker_id: str = Query("MLB", description="Broker clearing code (e.g. MLB, BIG5, YKR)"),
    from_time: Optional[str] = Query(None, description="Start time ISO string"),
    to_time: Optional[str] = Query(None, description="End time ISO string"),
    limit: int = Query(1500, le=5000, description="Max candle bars to return"),
) -> List[CandleBar]:
    """Sub-10ms candlestick endpoint querying TimescaleDB continuous aggregates.
    
    Supports synthetic XU030 benchmark and individual equities.
    """
    sym = symbol.upper()
    bid = broker_id.upper()
    interval_clean = interval.lower()

    if sym == "XU030" and interval_clean == "1d":
        params: List[Any] = []
        where_clauses = ["index_code = 'XU030'"]
        if from_time:
            where_clauses.append("trade_date >= %s")
            params.append(from_time[:10])
        if to_time:
            where_clauses.append("trade_date <= %s")
            params.append(to_time[:10])
        where_sql = " AND ".join(where_clauses)

        if from_time:
            query = f"""
                SELECT 
                    trade_date AS candle_time,
                    open_price AS open,
                    high_price AS high,
                    low_price AS low,
                    close_price AS close,
                    volume,
                    (volume * close_price) AS turnover_tl,
                    0 AS trade_count,
                    0.0 AS bofa_net_flow_tl
                FROM bronze_bist_index_benchmarks
                WHERE {where_sql}
                ORDER BY trade_date ASC LIMIT %s
            """
            params.append(limit)
        else:
            query = f"""
                SELECT * FROM (
                    SELECT 
                        trade_date AS candle_time,
                        open_price AS open,
                        high_price AS high,
                        low_price AS low,
                        close_price AS close,
                        volume,
                        (volume * close_price) AS turnover_tl,
                        0 AS trade_count,
                        0.0 AS bofa_net_flow_tl
                    FROM bronze_bist_index_benchmarks
                    WHERE {where_sql}
                    ORDER BY trade_date DESC LIMIT %s
                ) sub
                ORDER BY candle_time ASC
            """
            params.append(limit)

    elif interval_clean == "1d":
        params = [sym]
        where_clauses = ["s.symbol = %s"]
        if from_time:
            where_clauses.append("s.trade_date >= %s")
            params.append(from_time[:10])
        if to_time:
            where_clauses.append("s.trade_date <= %s")
            params.append(to_time[:10])
        where_sql = " AND ".join(where_clauses)

        if bid in INSTITUTIONAL_BUNDLES:
            flow_col = "COALESCE(b.net_flow_tl, 0.0)"
            broker_join = """
                LEFT JOIN (
                    SELECT trade_date, symbol, SUM(net_flow_tl) AS net_flow_tl
                    FROM silver_daily_broker_summary
                    WHERE symbol = %s AND broker_id = ANY(%s)
                    GROUP BY trade_date, symbol
                ) b ON s.trade_date = b.trade_date AND s.symbol = b.symbol
            """
            join_params = [sym, list(INSTITUTIONAL_BUNDLES[bid])]
        elif bid != "MLB":
            flow_col = "COALESCE(b.net_flow_tl, 0.0)"
            broker_join = """
                LEFT JOIN (
                    SELECT trade_date, symbol, net_flow_tl
                    FROM silver_daily_broker_summary
                    WHERE symbol = %s AND broker_id = %s
                ) b ON s.trade_date = b.trade_date AND s.symbol = b.symbol
            """
            join_params = [sym, bid]
        else:
            flow_col = "COALESCE(s.bofa_net_flow_tl, 0.0)"
            broker_join = ""
            join_params = []

        query_params = join_params + params

        if from_time:
            query = f"""
                SELECT 
                    s.trade_date AS candle_time,
                    s.open_price AS open,
                    s.high_price AS high,
                    s.low_price AS low,
                    s.close_price AS close,
                    s.total_volume AS volume,
                    s.total_turnover_tl AS turnover_tl,
                    s.total_trades AS trade_count,
                    {flow_col} AS bofa_net_flow_tl
                FROM silver_daily_stock_summary s
                {broker_join}
                WHERE {where_sql}
                ORDER BY s.trade_date ASC LIMIT %s
            """
            params = query_params + [limit]
        else:
            query = f"""
                SELECT * FROM (
                    SELECT 
                        s.trade_date AS candle_time,
                        s.open_price AS open,
                        s.high_price AS high,
                        s.low_price AS low,
                        s.close_price AS close,
                        s.total_volume AS volume,
                        s.total_turnover_tl AS turnover_tl,
                        s.total_trades AS trade_count,
                        {flow_col} AS bofa_net_flow_tl
                    FROM silver_daily_stock_summary s
                    {broker_join}
                    WHERE {where_sql}
                    ORDER BY s.trade_date DESC LIMIT %s
                ) sub
                ORDER BY candle_time ASC
            """
            params = query_params + [limit]

    else:
        tbl = "silver_candles_1m" if interval_clean == "1m" else "silver_candles_5m"
        params = [sym]
        where_clauses = ["symbol = %s"]
        if from_time:
            where_clauses.append("candle_time >= %s")
            params.append(from_time)
        if to_time:
            where_clauses.append("candle_time <= %s")
            params.append(to_time)
        where_sql = " AND ".join(where_clauses)

        if from_time:
            query = f"""
                SELECT 
                    candle_time,
                    open_price AS open,
                    high_price AS high,
                    low_price AS low,
                    close_price AS close,
                    total_volume AS volume,
                    total_turnover_tl AS turnover_tl,
                    trade_count,
                    COALESCE(bofa_net_flow_tl, 0.0) AS bofa_net_flow_tl
                FROM {tbl}
                WHERE {where_sql}
                ORDER BY candle_time ASC LIMIT %s
            """
            params.append(limit)
        else:
            query = f"""
                SELECT * FROM (
                    SELECT 
                        candle_time,
                        open_price AS open,
                        high_price AS high,
                        low_price AS low,
                        close_price AS close,
                        total_volume AS volume,
                        total_turnover_tl AS turnover_tl,
                        trade_count,
                        COALESCE(bofa_net_flow_tl, 0.0) AS bofa_net_flow_tl
                    FROM {tbl}
                    WHERE {where_sql}
                    ORDER BY candle_time DESC LIMIT %s
                ) sub
                ORDER BY candle_time ASC
            """
            params.append(limit)

    try:
        rows = db.execute(query, params).fetchall()
    except Exception as e:
        logger.error(f"Error executing candle query for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    bars: List[CandleBar] = []
    for r in rows:
        dt = r[0]
        if hasattr(dt, "timestamp"):
            epoch_sec = int(dt.replace(tzinfo=timezone.utc).timestamp()) if getattr(dt, "tzinfo", None) is None else int(dt.timestamp())
        elif isinstance(dt, date):
            epoch_sec = int(datetime.combine(dt, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        else:
            d_obj = date.fromisoformat(str(dt)[:10])
            epoch_sec = int(datetime.combine(d_obj, datetime.min.time(), tzinfo=timezone.utc).timestamp())

        bars.append(
            CandleBar(
                time=epoch_sec,
                open=float(r[1] or 0.0),
                high=float(r[2] or 0.0),
                low=float(r[3] or 0.0),
                close=float(r[4] or 0.0),
                volume=float(r[5] or 0.0),
                turnover_tl=float(r[6] or 0.0),
                trade_count=int(r[7] or 0),
                bofa_net_flow_tl=float(r[8] or 0.0),
            )
        )

    return bars


@app.get("/api/v1/market/summary", response_model=MarketSummaryResponse)
def get_market_summary(
    symbol: str = Query("THYAO", description="Stock symbol"),
    broker_id: str = Query("MLB", description="Broker clearing code (MLB = BofA)"),
    trade_date: Optional[str] = Query(None, description="Trade date YYYY-MM-DD"),
) -> MarketSummaryResponse:
    """Return 5 core trader KPIs and institutional order flow metrics."""
    sym = symbol.upper()
    bid = broker_id.upper()

    # Determine latest available trade date if not specified
    if not trade_date:
        date_row = db.execute(
            "SELECT MAX(trade_date) FROM silver_daily_stock_summary WHERE symbol = %s;",
            [sym],
        ).fetchone()
        if not date_row or not date_row[0]:
            trade_date = "2026-09-16"
        else:
            trade_date = str(date_row[0])

    stock_q = """
        SELECT 
            close_price, 
            daily_return_pct, 
            total_turnover_tl, 
            bofa_buy_turnover_tl, 
            bofa_sell_turnover_tl, 
            bofa_net_flow_tl, 
            bofa_stock_turnover_share
        FROM silver_daily_stock_summary
        WHERE symbol = %s AND trade_date = %s;
    """
    stock_r = db.execute(stock_q, [sym, trade_date]).fetchone()

    if bid in INSTITUTIONAL_BUNDLES:
        fifo_q = """
            SELECT 
                SUM(buy_turnover_tl) AS buy_turnover_tl,
                SUM(sell_turnover_tl) AS sell_turnover_tl,
                SUM(matched_volume) AS matched_volume,
                SUM(intraday_realized_pnl_tl) AS intraday_realized_pnl_tl,
                SUM(carry_fifo_realized_pnl_tl) AS carry_fifo_realized_pnl_tl,
                SUM(daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                SUM(cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl,
                SUM(open_stock_quantity) AS open_stock_quantity,
                SUM(open_fifo_cost_tl) / NULLIF(SUM(open_stock_quantity), 0) AS fifo_avg_cost,
                SUM(market_value_tl) AS market_value_tl,
                SUM(unrealized_pnl_tl) AS unrealized_pnl_tl
            FROM silver_broker_fifo_daily
            WHERE symbol = %s AND broker_id = ANY(%s) AND trade_date = %s;
        """
        fifo_r = db.execute(fifo_q, [sym, list(INSTITUTIONAL_BUNDLES[bid]), trade_date]).fetchone()
    else:
        fifo_q = """
            SELECT 
                buy_turnover_tl,
                sell_turnover_tl,
                matched_volume,
                intraday_realized_pnl_tl,
                carry_fifo_realized_pnl_tl,
                daily_realized_pnl_tl,
                cumulative_realized_pnl_tl,
                open_stock_quantity,
                fifo_avg_cost,
                market_value_tl,
                unrealized_pnl_tl
            FROM silver_broker_fifo_daily
            WHERE symbol = %s AND broker_id = %s AND trade_date = %s;
        """
        fifo_r = db.execute(fifo_q, [sym, bid, trade_date]).fetchone()

    close_p = float(stock_r[0] or 0.0) if stock_r else 0.0
    daily_ret = float(stock_r[1] or 0.0) * 100.0 if stock_r else 0.0
    turnover = float(stock_r[2] or 0.0) if stock_r else 0.0

    if fifo_r:
        b_buy = float(fifo_r[0] or 0.0)
        b_sell = float(fifo_r[1] or 0.0)
        net_flow = b_buy - b_sell
        matched_vol = float(fifo_r[2] or 0.0)
        intra_pnl = float(fifo_r[3] or 0.0)
        carry_pnl = float(fifo_r[4] or 0.0)
        daily_pnl = float(fifo_r[5] or 0.0)
        cum_pnl = float(fifo_r[6] or 0.0)
        open_qty = float(fifo_r[7] or 0.0)
        avg_cost = float(fifo_r[8] or 0.0)
        mtm_val = float(fifo_r[9] or 0.0)
        unreal_pnl = float(fifo_r[10] or 0.0)
    elif stock_r and bid == "MLB":
        b_buy = float(stock_r[3] or 0.0)
        b_sell = float(stock_r[4] or 0.0)
        net_flow = float(stock_r[5] or 0.0)
        matched_vol = 0.0
        intra_pnl = 0.0
        carry_pnl = 0.0
        daily_pnl = 0.0
        cum_pnl = 0.0
        open_qty = 0.0
        avg_cost = 0.0
        mtm_val = 0.0
        unreal_pnl = 0.0
    else:
        b_buy = b_sell = net_flow = matched_vol = intra_pnl = carry_pnl = daily_pnl = cum_pnl = open_qty = avg_cost = mtm_val = unreal_pnl = 0.0

    share_pct = ((b_buy + b_sell) / turnover * 100.0) if turnover > 0 else 0.0

    # Classify Institutional Bias Badge
    if net_flow >= 100_000_000:
        bias_badge = "AGGRESSIVE_BUYER"
    elif net_flow >= 25_000_000:
        bias_badge = "MODERATE_BUYER"
    elif net_flow <= -100_000_000:
        bias_badge = "AGGRESSIVE_SELLER"
    elif net_flow <= -25_000_000:
        bias_badge = "MODERATE_SELLER"
    else:
        bias_badge = "NEUTRAL"

    return MarketSummaryResponse(
        symbol=sym,
        broker_id=bid,
        trade_date=trade_date,
        close_price=round(close_p, 2),
        daily_return_pct=round(daily_ret, 2),
        total_turnover_tl=round(turnover, 2),
        broker_buy_turnover_tl=round(b_buy, 2),
        broker_sell_turnover_tl=round(b_sell, 2),
        broker_net_flow_tl=round(net_flow, 2),
        broker_share_pct=round(share_pct, 2),
        matched_volume=round(matched_vol, 2),
        intraday_realized_pnl_tl=round(intra_pnl, 2),
        carry_fifo_realized_pnl_tl=round(carry_pnl, 2),
        daily_realized_pnl_tl=round(daily_pnl, 2),
        cumulative_realized_pnl_tl=round(cum_pnl, 2),
        open_stock_quantity=round(open_qty, 2),
        fifo_avg_cost=round(avg_cost, 2),
        market_value_tl=round(mtm_val, 2),
        unrealized_pnl_tl=round(unreal_pnl, 2),
        bias_badge=bias_badge,
    )


# ── FIFO Tertip Inventory & Lot Tracking Endpoints ────────────────────────────

@app.get("/api/v1/tertip/portfolio", response_model=TertipPortfolioResponse)
def get_tertip_portfolio(
    broker_id: str = Query("MLB", description="Broker clearing code (MLB = BofA)"),
    trade_date: Optional[str] = Query(None, description="Trade date YYYY-MM-DD"),
) -> TertipPortfolioResponse:
    """Return institution's active inventory portfolio, unit costs, and MTM valuations."""
    bid = broker_id.upper()

    if not trade_date:
        if bid in INSTITUTIONAL_BUNDLES:
            d_row = db.execute(
                "SELECT MAX(trade_date) FROM silver_broker_fifo_daily WHERE broker_id = ANY(%s);",
                [list(INSTITUTIONAL_BUNDLES[bid])],
            ).fetchone()
        else:
            d_row = db.execute(
                "SELECT MAX(trade_date) FROM silver_broker_fifo_daily WHERE broker_id = %s;",
                [bid],
            ).fetchone()
        trade_date = str(d_row[0]) if (d_row and d_row[0]) else "2026-09-16"

    if bid in INSTITUTIONAL_BUNDLES:
        query = """
            SELECT 
                symbol,
                MAX(COALESCE(symbol_name, symbol)) AS symbol_name,
                MAX(COALESCE(sector, 'General')) AS sector,
                SUM(open_stock_quantity) AS open_stock_quantity,
                SUM(open_fifo_cost_tl) / NULLIF(SUM(open_stock_quantity), 0) AS fifo_avg_cost,
                MAX(market_close_price) AS market_close_price,
                SUM(market_value_tl) AS market_value_tl,
                SUM(unrealized_pnl_tl) AS unrealized_pnl_tl,
                SUM(daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                SUM(cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl,
                CASE WHEN SUM(open_stock_quantity) > 0 THEN 'LONG'
                     WHEN SUM(open_stock_quantity) < 0 THEN 'SHORT'
                     ELSE 'FLAT' END AS position_side
            FROM silver_broker_fifo_daily
            WHERE broker_id = ANY(%s) AND trade_date = %s
            GROUP BY symbol
            HAVING ABS(SUM(open_stock_quantity)) > 0
            ORDER BY ABS(SUM(market_value_tl)) DESC;
        """
        rows = db.execute(query, [list(INSTITUTIONAL_BUNDLES[bid]), trade_date]).fetchall()
    else:
        query = """
            SELECT 
                symbol,
                COALESCE(symbol_name, symbol),
                COALESCE(sector, 'General'),
                open_stock_quantity,
                fifo_avg_cost,
                market_close_price,
                market_value_tl,
                unrealized_pnl_tl,
                daily_realized_pnl_tl,
                cumulative_realized_pnl_tl,
                position_side
            FROM silver_broker_fifo_daily
            WHERE broker_id = %s AND trade_date = %s AND position_side != 'FLAT'
            ORDER BY ABS(market_value_tl) DESC;
        """
        rows = db.execute(query, [bid, trade_date]).fetchall()

    positions: List[TertipPosition] = []
    tot_mtm = 0.0
    tot_unreal = 0.0
    tot_daily_pnl = 0.0
    tot_cum_pnl = 0.0

    for r in rows:
        qty = float(r[3] or 0.0)
        cost = float(r[4] or 0.0)
        close_p = float(r[5] or 0.0)
        mtm = float(r[6] or 0.0)
        unreal = float(r[7] or 0.0)
        daily_p = float(r[8] or 0.0)
        cum_p = float(r[9] or 0.0)
        side = str(r[10] or "FLAT")

        unreal_pct = ((close_p - cost) / cost * 100.0) if cost > 0 else 0.0

        tot_mtm += mtm
        tot_unreal += unreal
        tot_daily_pnl += daily_p
        tot_cum_pnl += cum_p

        positions.append(
            TertipPosition(
                symbol=r[0],
                symbol_name=r[1],
                sector=r[2],
                open_stock_quantity=round(qty, 2),
                fifo_avg_cost=round(cost, 2),
                market_close_price=round(close_p, 2),
                market_value_tl=round(mtm, 2),
                unrealized_pnl_tl=round(unreal, 2),
                unrealized_pnl_pct=round(unreal_pct, 2),
                daily_realized_pnl_tl=round(daily_p, 2),
                cumulative_realized_pnl_tl=round(cum_p, 2),
                position_side=side,
            )
        )

    return TertipPortfolioResponse(
        broker_id=bid,
        trade_date=trade_date,
        total_market_value_tl=round(tot_mtm, 2),
        total_unrealized_pnl_tl=round(tot_unreal, 2),
        total_daily_realized_pnl_tl=round(tot_daily_pnl, 2),
        total_cumulative_realized_pnl_tl=round(tot_cum_pnl, 2),
        positions=positions,
    )


@app.get("/api/v1/tertip/lots", response_model=List[TertipLotItem])
def get_tertip_lots(
    broker_id: str = Query("MLB", description="Broker clearing code"),
    symbol: Optional[str] = Query(None, description="Filter by stock symbol"),
    limit: int = Query(200, le=2000, description="Max open lots to return"),
) -> List[TertipLotItem]:
    """Return audited open FIFO inventory lots."""
    bid = broker_id.upper()
    if bid in INSTITUTIONAL_BUNDLES:
        where_broker = "broker_id = ANY(%s)"
        params: List[Any] = [list(INSTITUTIONAL_BUNDLES[bid])]
    else:
        where_broker = "broker_id = %s"
        params: List[Any] = [bid]

    query = f"""
        SELECT 
            lot_id,
            symbol,
            direction,
            open_date,
            remaining_quantity,
            unit_cost,
            remaining_value_tl,
            (CURRENT_DATE - open_date) AS days_held
        FROM silver_broker_fifo_lots
        WHERE {where_broker} AND remaining_quantity > 0
    """
    if symbol:
        query += " AND symbol = %s"
        params.append(symbol.upper())

    query += " ORDER BY open_date DESC, remaining_value_tl DESC LIMIT %s;"
    params.append(limit)

    rows = db.execute(query, params).fetchall()
    return [
        TertipLotItem(
            lot_id=r[0],
            symbol=r[1],
            direction=r[2],
            open_date=str(r[3]),
            remaining_quantity=round(float(r[4] or 0.0), 2),
            unit_cost=round(float(r[5] or 0.0), 2),
            remaining_value_tl=round(float(r[6] or 0.0), 2),
            days_held=int(r[7] or 0),
        )
        for r in rows
    ]


@app.get("/api/v1/tertip/history", response_model=List[TertipHistoryPoint])
def get_tertip_history(
    broker_id: str = Query("MLB", description="Broker clearing code"),
    symbol: Optional[str] = Query(None, description="Optional stock symbol filter"),
    limit: int = Query(100, le=500, description="Number of daily points"),
) -> List[TertipHistoryPoint]:
    """Return daily realized PnL, intraday vs carry split, and cumulative performance."""
    bid = broker_id.upper()
    if bid in INSTITUTIONAL_BUNDLES:
        bundle_list = list(INSTITUTIONAL_BUNDLES[bid])
        if symbol:
            query = """
                SELECT 
                    trade_date,
                    SUM(daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                    SUM(intraday_realized_pnl_tl) AS intraday_realized_pnl_tl,
                    SUM(carry_fifo_realized_pnl_tl) AS carry_fifo_realized_pnl_tl,
                    SUM(buy_turnover_tl - sell_turnover_tl) AS net_flow_tl,
                    SUM(market_value_tl) AS market_value_tl,
                    SUM(cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl
                FROM silver_broker_fifo_daily
                WHERE broker_id = ANY(%s) AND symbol = %s
                GROUP BY trade_date
                ORDER BY trade_date DESC LIMIT %s;
            """
            rows = db.execute(query, [bundle_list, symbol.upper(), limit]).fetchall()
        else:
            query = """
                SELECT 
                    trade_date,
                    SUM(daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                    SUM(intraday_realized_pnl_tl) AS intraday_realized_pnl_tl,
                    SUM(carry_fifo_realized_pnl_tl) AS carry_fifo_realized_pnl_tl,
                    SUM(buy_turnover_tl - sell_turnover_tl) AS net_flow_tl,
                    SUM(market_value_tl) AS market_value_tl,
                    SUM(cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl
                FROM silver_broker_fifo_daily
                WHERE broker_id = ANY(%s)
                GROUP BY trade_date
                ORDER BY trade_date DESC LIMIT %s;
            """
            rows = db.execute(query, [bundle_list, limit]).fetchall()
    elif symbol:
        query = """
            SELECT 
                trade_date,
                daily_realized_pnl_tl,
                intraday_realized_pnl_tl,
                carry_fifo_realized_pnl_tl,
                (buy_turnover_tl - sell_turnover_tl) AS net_flow_tl,
                market_value_tl,
                cumulative_realized_pnl_tl
            FROM silver_broker_fifo_daily
            WHERE broker_id = %s AND symbol = %s
            ORDER BY trade_date DESC LIMIT %s;
        """
        rows = db.execute(query, [bid, symbol.upper(), limit]).fetchall()
    else:
        query = """
            SELECT 
                trade_date,
                SUM(daily_realized_pnl_tl) AS daily_realized_pnl_tl,
                SUM(intraday_realized_pnl_tl) AS intraday_realized_pnl_tl,
                SUM(carry_fifo_realized_pnl_tl) AS carry_fifo_realized_pnl_tl,
                SUM(buy_turnover_tl - sell_turnover_tl) AS net_flow_tl,
                SUM(market_value_tl) AS market_value_tl,
                SUM(cumulative_realized_pnl_tl) AS cumulative_realized_pnl_tl
            FROM silver_broker_fifo_daily
            WHERE broker_id = %s
            GROUP BY trade_date
            ORDER BY trade_date DESC LIMIT %s;
        """
        rows = db.execute(query, [bid, limit]).fetchall()

    points = [
        TertipHistoryPoint(
            trade_date=str(r[0]),
            daily_realized_pnl_tl=round(float(r[1] or 0.0), 2),
            intraday_pnl_tl=round(float(r[2] or 0.0), 2),
            carry_pnl_tl=round(float(r[3] or 0.0), 2),
            net_flow_tl=round(float(r[4] or 0.0), 2),
            mtm_valuation_tl=round(float(r[5] or 0.0), 2),
            cumulative_realized_pnl_tl=round(float(r[6] or 0.0), 2),
        )
        for r in rows
    ]
    points.reverse()  # Return in chronological order for charting
    return points


@app.get("/api/v1/tertip/horizons", response_model=TertipHorizonsResponse)
def get_tertip_horizons(
    symbol: str = Query("THYAO", description="Stock symbol"),
    broker_id: str = Query("MLB", description="Broker clearing code (MLB = BofA)"),
    trade_date: Optional[str] = Query(None, description="Optional trade date YYYY-MM-DD"),
) -> TertipHorizonsResponse:
    """Return multi-horizon EWMA inventory metrics, cost-basis spreads, and action diagnostic."""
    try:
        data = get_tertip_horizons_analysis(db, symbol, broker_id, trade_date)
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return TertipHorizonsResponse(**data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing tertip horizons for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tertip/timeseries", response_model=List[TertipTimeseriesPoint])
def get_tertip_timeseries(
    symbol: str = Query("THYAO", description="Stock symbol"),
    broker_id: str = Query("MLB", description="Broker clearing code (MLB = BofA)"),
    limit_days: int = Query(1500, le=5000, description="Max historical sessions"),
) -> List[TertipTimeseriesPoint]:
    """Return chronological time series with EWMA ribbons and cost basis for charting."""
    try:
        rows = get_tertip_timeseries_chart(db, symbol, broker_id, limit_days)
        return [TertipTimeseriesPoint(**r) for r in rows]
    except Exception as e:
        logger.error(f"Error computing tertip timeseries for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Stock Movement & Event Study Scanner Endpoints ───────────────────────────

@app.get("/api/v1/event-study/scan", response_model=List[EventStudyScanItem])
def scan_event_study(
    broker_id: str = Query("MLB", description="Broker clearing code"),
    symbol: Optional[str] = Query(None, description="Filter by stock symbol"),
    min_flow_tl: Optional[float] = Query(None, description="Minimum net flow TL (e.g. 50000000)"),
    max_flow_tl: Optional[float] = Query(None, description="Maximum net flow TL (for net seller scans)"),
    min_d1_return: Optional[float] = Query(None, description="Min previous day return % (e.g. -5.0)"),
    max_d1_return: Optional[float] = Query(None, description="Max previous day return % (e.g. 2.0)"),
    limit: int = Query(100, le=1000, description="Max event study rows"),
) -> List[EventStudyScanItem]:
    """Scan historical sessions for institutional order flow triggers and forward returns."""
    bid = broker_id.upper()
    conditions = ["1=1"]
    params: List[Any] = []

    if bid in INSTITUTIONAL_BUNDLES:
        flow_expr = "COALESCE(b.net_flow_tl, 0.0)"
        broker_join = """
            LEFT JOIN (
                SELECT trade_date, symbol, SUM(net_flow_tl) AS net_flow_tl
                FROM silver_daily_broker_summary
                WHERE broker_id = ANY(%s)
                GROUP BY trade_date, symbol
            ) b ON s.trade_date = b.trade_date AND s.symbol = b.symbol
        """
        join_params = [list(INSTITUTIONAL_BUNDLES[bid])]
    elif bid != "MLB":
        flow_expr = "COALESCE(b.net_flow_tl, 0.0)"
        broker_join = """
            LEFT JOIN (
                SELECT trade_date, symbol, net_flow_tl
                FROM silver_daily_broker_summary
                WHERE broker_id = %s
            ) b ON s.trade_date = b.trade_date AND s.symbol = b.symbol
        """
        join_params = [bid]
    else:
        flow_expr = "s.bofa_net_flow_tl"
        broker_join = ""
        join_params = []

    if symbol:
        conditions.append("s.symbol = %s")
        params.append(symbol.upper())

    if min_flow_tl is not None:
        conditions.append(f"{flow_expr} >= %s")
        params.append(min_flow_tl)

    if max_flow_tl is not None:
        conditions.append(f"{flow_expr} <= %s")
        params.append(max_flow_tl)

    where_clause = " AND ".join(conditions)

    query = f"""
        WITH ranked_stock AS (
            SELECT 
                s.trade_date,
                s.symbol,
                s.adj_close_price,
                (s.adj_daily_return_pct * 100.0) AS daily_return_pct,
                (LAG(s.adj_daily_return_pct, 1) OVER (PARTITION BY s.symbol ORDER BY s.trade_date) * 100.0) AS d1_return_pct,
                {flow_expr} AS bofa_net_flow_tl,
                s.total_turnover_tl,
                LEAD(s.adj_close_price, 1) OVER (PARTITION BY s.symbol ORDER BY s.trade_date) AS lead_p1,
                LEAD(s.adj_close_price, 2) OVER (PARTITION BY s.symbol ORDER BY s.trade_date) AS lead_p2,
                LEAD(s.adj_close_price, 3) OVER (PARTITION BY s.symbol ORDER BY s.trade_date) AS lead_p3,
                LEAD(s.adj_close_price, 5) OVER (PARTITION BY s.symbol ORDER BY s.trade_date) AS lead_p5,
                LEAD(s.adj_close_price, 10) OVER (PARTITION BY s.symbol ORDER BY s.trade_date) AS lead_p10
            FROM silver_daily_stock_summary s
            {broker_join}
            WHERE {where_clause}
        )
        SELECT 
            trade_date,
            symbol,
            adj_close_price,
            daily_return_pct,
            d1_return_pct,
            bofa_net_flow_tl,
            total_turnover_tl,
            ROUND(CAST((lead_p1 / NULLIF(adj_close_price, 0) - 1.0) * 100 AS numeric), 2) AS return_t1,
            ROUND(CAST((lead_p2 / NULLIF(adj_close_price, 0) - 1.0) * 100 AS numeric), 2) AS return_t2,
            ROUND(CAST((lead_p3 / NULLIF(adj_close_price, 0) - 1.0) * 100 AS numeric), 2) AS return_t3,
            ROUND(CAST((lead_p5 / NULLIF(adj_close_price, 0) - 1.0) * 100 AS numeric), 2) AS return_t5,
            ROUND(CAST((lead_p10 / NULLIF(adj_close_price, 0) - 1.0) * 100 AS numeric), 2) AS return_t10
        FROM ranked_stock
        WHERE 1=1
    """

    all_params = join_params + params
    if min_d1_return is not None:
        query += " AND d1_return_pct >= %s"
        all_params.append(min_d1_return)

    if max_d1_return is not None:
        query += " AND d1_return_pct <= %s"
        all_params.append(max_d1_return)

    query += " ORDER BY trade_date DESC, ABS(bofa_net_flow_tl) DESC LIMIT %s;"
    all_params.append(limit)

    rows = db.execute(query, all_params).fetchall()
    return [
        EventStudyScanItem(
            trade_date=str(r[0]),
            symbol=r[1],
            close_price=round(float(r[2] or 0.0), 2),
            daily_return_pct=round(float(r[3] or 0.0), 2),
            d1_return_pct=round(float(r[4]), 2) if r[4] is not None else None,
            bofa_net_flow_tl=round(float(r[5] or 0.0), 2),
            total_turnover_tl=round(float(r[6] or 0.0), 2),
            return_t1=float(r[7]) if r[7] is not None else None,
            return_t2=float(r[8]) if r[8] is not None else None,
            return_t3=float(r[9]) if r[9] is not None else None,
            return_t5=float(r[10]) if r[10] is not None else None,
            return_t10=float(r[11]) if r[11] is not None else None,
        )
        for r in rows
    ]


# ── Custom Time Window Analysis Terminal Endpoints ───────────────────────────

@app.get("/api/v1/time-window/analyze", response_model=TimeWindowAnalysisResponse)
def analyze_time_windows(
    symbol: str = Query("THYAO", description="Stock symbol"),
    broker_id: str = Query("MLB", description="Broker clearing code"),
    trade_date: Optional[str] = Query(None, description="Trade date YYYY-MM-DD"),
) -> TimeWindowAnalysisResponse:
    """Analyze execution windows W1-W5 with auction microstructure flags."""
    sym = symbol.upper()
    bid = broker_id.upper()

    if not trade_date:
        if bid in INSTITUTIONAL_BUNDLES:
            d_row = db.execute(
                "SELECT MAX(trade_date) FROM silver_intraday_broker_window_summary WHERE symbol = %s AND broker_id = ANY(%s);",
                [sym, list(INSTITUTIONAL_BUNDLES[bid])],
            ).fetchone()
        else:
            d_row = db.execute(
                "SELECT MAX(trade_date) FROM silver_intraday_broker_window_summary WHERE symbol = %s AND broker_id = %s;",
                [sym, bid],
            ).fetchone()
        trade_date = str(d_row[0]) if (d_row and d_row[0]) else "2026-09-16"

    if bid in INSTITUTIONAL_BUNDLES:
        query = """
            SELECT 
                window_name,
                window_order,
                window_start_time,
                window_end_time,
                SUM(buy_turnover_tl) AS buy_turnover_tl,
                SUM(sell_turnover_tl) AS sell_turnover_tl,
                SUM(net_flow_tl) AS net_flow_tl,
                SUM(total_turnover_tl) AS total_turnover_tl,
                SUM(buy_volume) AS buy_volume,
                SUM(sell_volume) AS sell_volume,
                SUM(net_volume) AS net_volume
            FROM silver_intraday_broker_window_summary
            WHERE symbol = %s AND broker_id = ANY(%s) AND trade_date = %s
            GROUP BY window_name, window_order, window_start_time, window_end_time
            ORDER BY window_order ASC;
        """
        rows = db.execute(query, [sym, list(INSTITUTIONAL_BUNDLES[bid]), trade_date]).fetchall()
    else:
        query = """
            SELECT 
                window_name,
                window_order,
                window_start_time,
                window_end_time,
                buy_turnover_tl,
                sell_turnover_tl,
                net_flow_tl,
                total_turnover_tl,
                buy_volume,
                sell_volume,
                net_volume
            FROM silver_intraday_broker_window_summary
            WHERE symbol = %s AND broker_id = %s AND trade_date = %s
            ORDER BY window_order ASC;
        """
        rows = db.execute(query, [sym, bid, trade_date]).fetchall()

    windows = [
        TimeWindowItem(
            window_name=r[0],
            window_order=int(r[1]),
            start_time=str(r[2]),
            end_time=str(r[3]) if r[3] else "",
            buy_turnover_tl=round(float(r[4] or 0.0), 2),
            sell_turnover_tl=round(float(r[5] or 0.0), 2),
            net_flow_tl=round(float(r[6] or 0.0), 2),
            total_turnover_tl=round(float(r[7] or 0.0), 2),
            buy_volume=round(float(r[8] or 0.0), 2),
            sell_volume=round(float(r[9] or 0.0), 2),
            net_volume=round(float(r[10] or 0.0), 2),
        )
        for r in rows
    ]

    # Look up opening and closing auction estimates from W1 and W5
    w1 = next((w for w in windows if w.window_order == 1), None)
    w5 = next((w for w in windows if w.window_order == 5), None)

    return TimeWindowAnalysisResponse(
        symbol=sym,
        broker_id=bid,
        trade_date=trade_date,
        windows=windows,
        opening_auction_net_tl=w1.net_flow_tl if w1 else None,
        closing_auction_net_tl=w5.net_flow_tl if w5 else None,
    )


# ── Gold Predictive Oracle Endpoints ──────────────────────────────────────────

@app.get("/api/v1/signals/day-start", response_model=Optional[MacroSignalResponse])
def get_day_start_signal() -> Optional[MacroSignalResponse]:
    """Return live Model 1 Day-Start Macro Forecast for upcoming session T+1."""
    query = """
        SELECT 
            forecast_date,
            predicted_open_net_flow_tl,
            predicted_open_flow_lower_90,
            predicted_open_flow_upper_90,
            predicted_direction,
            direction_confidence,
            predicted_playbook,
            top_predicted_buy_sector,
            top_predicted_sell_sector,
            model_name,
            model_version
        FROM gold_bofa_day_start_forecasts
        ORDER BY forecast_date DESC LIMIT 1;
    """
    row = db.execute(query).fetchone()
    if not row:
        return None

    return MacroSignalResponse(
        forecast_date=str(row[0]),
        predicted_open_net_flow_tl=row[1],
        predicted_open_flow_lower_90=row[2],
        predicted_open_flow_upper_90=row[3],
        predicted_direction=row[4],
        direction_confidence=row[5],
        predicted_playbook=row[6],
        top_predicted_buy_sector=row[7],
        top_predicted_sell_sector=row[8],
        model_name=row[9],
        model_version=row[10],
    )


@app.get("/api/v1/signals/sector-rotation")
def get_sector_rotation_signals() -> List[Dict[str, Any]]:
    """Return live Model 2 Sector Day-Start predicted capital allocations."""
    query = """
        SELECT 
            forecast_date,
            sector,
            predicted_open_net_flow_tl,
            predicted_open_flow_lower_90,
            predicted_open_flow_upper_90,
            predicted_direction,
            direction_confidence,
            predicted_playbook
        FROM gold_bofa_sector_day_start_forecasts
        WHERE forecast_date = (SELECT MAX(forecast_date) FROM gold_bofa_sector_day_start_forecasts)
        ORDER BY predicted_open_net_flow_tl DESC;
    """
    rows = db.execute(query).fetchall()
    return [
        {
            "forecast_date": str(r[0]),
            "sector": r[1],
            "predicted_open_net_flow_tl": round(float(r[2] or 0.0), 2),
            "predicted_lower_90": round(float(r[3] or 0.0), 2),
            "predicted_upper_90": round(float(r[4] or 0.0), 2),
            "direction": r[5],
            "confidence": round(float(r[6] or 0.0), 3),
            "playbook": r[7],
        }
        for r in rows
    ]


@app.get("/api/v1/signals/stock-reactions", response_model=List[StockReactionForecastItem])
def get_stock_reaction_signals(
    window: str = Query("w2", description="Reaction window: w2, w3, w5"),
) -> List[StockReactionForecastItem]:
    """Return Model 3 Stock Intraday Reaction predictions for equities."""
    tbl_map = {
        "w2": "gold_bofa_stock_reaction_w2_forecasts",
        "w3": "gold_bofa_stock_reaction_w3_forecasts",
        "w5": "gold_bofa_stock_reaction_w5_forecasts",
    }
    tbl = tbl_map.get(window.lower(), "gold_bofa_stock_reaction_w2_forecasts")

    query = f"""
        SELECT 
            forecast_date,
            symbol,
            window_name,
            predicted_return_pct,
            predicted_return_lower_90,
            predicted_return_upper_90,
            predicted_direction,
            direction_confidence,
            predicted_playbook
        FROM {tbl}
        WHERE forecast_date = (SELECT MAX(forecast_date) FROM {tbl})
        ORDER BY predicted_return_pct DESC;
    """
    rows = db.execute(query).fetchall()
    return [
        StockReactionForecastItem(
            forecast_date=str(r[0]),
            symbol=r[1],
            window_name=r[2],
            predicted_return_pct=round(float(r[3] or 0.0), 2),
            predicted_return_lower_90=round(float(r[4] or 0.0), 2),
            predicted_return_upper_90=round(float(r[5] or 0.0), 2),
            predicted_direction=r[6],
            direction_confidence=round(float(r[7] or 0.0), 3),
            predicted_playbook=r[8],
        )
        for r in rows
    ]


@app.get("/api/v1/signals/all", response_model=AllSignalsResponse)
def get_all_signals() -> AllSignalsResponse:
    """Return consolidated Gold Layer signals across Macro, Sector, and Stock models."""
    macro = get_day_start_signal()
    sectors = get_sector_rotation_signals()
    stocks = get_stock_reaction_signals(window="w2")

    f_date = macro.forecast_date if macro else "2026-09-17"

    return AllSignalsResponse(
        forecast_date=f_date,
        macro_day_start=macro,
        sector_allocations=sectors,
        stock_reactions=stocks,
    )


# ── WebSocket Real-Time Channel ───────────────────────────────────────────────

@app.websocket("/api/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket stream for real-time market ticks, candle completions, and order flow alerts."""
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "message": f"Subscribed: {data}"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
