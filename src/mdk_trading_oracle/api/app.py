"""FastAPI Backend Serving Layer for MDK Trading Oracle & React Frontend.

Provides ultra-low latency REST endpoints and WebSocket streams for:
- TradingView Lightweight Charts (1m, 5m, 1d OHLCV Candlesticks)
- Institutional BofA Order Flow & Intraday Execution Windows (W1-W5)
- Live Gold Predictive Forecasts, Credible Intervals & Playbook Signals
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.api")

app = FastAPI(
    title="MDK Trading Oracle API",
    description="High-performance financial market data & institutional order flow analytics",
    version="1.0.0",
)

# Enable CORS for React frontend (Vite / Next.js on localhost:3000, 5173, etc.)
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

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


@app.get("/api/v1/market/candles", response_model=List[CandleBar])
def get_candlesticks(
    symbol: str = Query(..., description="BIST Equity Symbol (e.g. THYAO, AKBNK)"),
    interval: str = Query("5m", description="Candle interval: 1m, 5m, 1d"),
    from_time: Optional[str] = Query(None, description="Start time ISO string (e.g. '2026-03-01T09:55:00')"),
    to_time: Optional[str] = Query(None, description="End time ISO string"),
    limit: int = Query(500, le=5000, description="Max candle bars to return"),
) -> List[CandleBar]:
    """Sub-10ms candlestick endpoint querying TimescaleDB continuous aggregates.
    
    Returns data formatted specifically for TradingView Lightweight Charts.
    """
    table_map = {
        "1m": "silver_candles_1m",
        "5m": "silver_candles_5m",
        "1d": "silver_daily_stock_summary",
    }

    tbl = table_map.get(interval.lower(), "silver_candles_5m")

    if interval.lower() == "1d":
        query = f"""
            SELECT 
                trade_date AS candle_time,
                open_price AS open,
                high_price AS high,
                low_price AS low,
                close_price AS close,
                total_volume AS volume,
                total_turnover_tl AS turnover_tl,
                total_trades AS trade_count,
                bofa_net_flow_tl
            FROM {tbl}
            WHERE symbol = %s
        """
    else:
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
                bofa_net_flow_tl
            FROM {tbl}
            WHERE symbol = %s
        """

    params: List[Any] = [symbol.upper()]
    if from_time:
        query += " AND candle_time >= %s"
        params.append(from_time)
    if to_time:
        query += " AND candle_time <= %s"
        params.append(to_time)

    query += " ORDER BY candle_time ASC LIMIT %s"
    params.append(limit)

    try:
        rows = db.execute(query, params).fetchall()
    except Exception as e:
        logger.error(f"Error executing candle query for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    bars = []
    for r in rows:
        dt = r[0]
        # Convert date or datetime to UNIX epoch seconds
        if hasattr(dt, "timestamp"):
            epoch_sec = int(dt.timestamp())
        elif isinstance(dt, date):
            epoch_sec = int(datetime.combine(dt, datetime.min.time()).timestamp())
        else:
            epoch_sec = int(datetime.fromisoformat(str(dt)).timestamp())

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


@app.get("/api/v1/market/order-flow")
def get_order_flow_intraday(
    symbol: str = Query(..., description="Stock symbol"),
    trade_date: Optional[str] = Query(None, description="Trading date YYYY-MM-DD"),
) -> List[Dict[str, Any]]:
    """Return 5-window intraday institutional execution breakdown (W1-W5)."""
    query = """
        SELECT 
            trade_date,
            window_name,
            window_order,
            window_start_time,
            window_end_time,
            buy_turnover_tl,
            sell_turnover_tl,
            net_flow_tl,
            total_turnover_tl
        FROM silver_intraday_broker_window_summary
        WHERE symbol = %s AND broker_id = 'MLB'
    """
    params: List[Any] = [symbol.upper()]
    if trade_date:
        query += " AND trade_date = %s"
        params.append(trade_date)

    query += " ORDER BY trade_date DESC, window_order ASC LIMIT 25;"

    rows = db.execute(query, params).fetchall()
    return [
        {
            "trade_date": str(r[0]),
            "window_name": r[1],
            "window_order": r[2],
            "start_time": r[3],
            "end_time": r[4],
            "buy_turnover_tl": r[5],
            "sell_turnover_tl": r[6],
            "net_flow_tl": r[7],
            "total_turnover_tl": r[8],
        }
        for r in rows
    ]


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
            "predicted_open_net_flow_tl": r[2],
            "predicted_lower_90": r[3],
            "predicted_upper_90": r[4],
            "direction": r[5],
            "confidence": r[6],
            "playbook": r[7],
        }
        for r in rows
    ]


# ── WebSocket Real-Time Channel ───────────────────────────────────────────────

@app.websocket("/api/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket stream for real-time market ticks, candle completions, and order flow alerts."""
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            # Echo or process incoming subscription requests (e.g. subscribe:THYAO)
            await websocket.send_json({"type": "ack", "message": f"Subscribed: {data}"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
