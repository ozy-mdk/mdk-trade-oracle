# MDK Trading Oracle — React Frontend Architecture

A modern, high-performance financial analytics frontend powered by **React 18 + Vite + TypeScript + TradingView Lightweight Charts**.

---

## 1. Architectural Blueprint

```
frontend/
├── src/
│   ├── components/
│   │   ├── charts/
│   │   │   ├── TradingViewCandleChart.tsx    # Dual-pane Candlesticks + BofA Net Flow
│   │   │   ├── TimeframeSelector.tsx         # 1m, 5m, 1d switcher
│   │   │   └── IntradayWindowFootprint.tsx   # W1-W5 institutional execution bars
│   │   ├── signals/
│   │   │   ├── DayStartSignalCard.tsx        # Live T+1 forecast card & playbook badge
│   │   │   ├── SectorAllocationHeatmap.tsx   # Cross-sectional sector rotation
│   │   │   └── StockReactionLeaderboard.tsx  # BIST 30 rally/decline rankings
│   │   └── layout/
│   │       ├── Header.tsx                    # Symbol search, market status, server time
│   │       └── Sidebar.tsx                   # Watchlist & institutional filters
│   ├── hooks/
│   │   ├── useCandles.ts                     # TanStack Query hook fetching /api/v1/market/candles
│   │   ├── useSignals.ts                     # Fetches active T+1 predictive signals
│   │   └── useLiveStream.ts                  # WebSocket hook for live bar updates
│   ├── types/
│   │   └── market.ts                         # TypeScript interfaces (CandleBar, SignalResponse)
│   ├── App.tsx                               # Main dashboard container
│   └── main.tsx                              # React DOM mount point
├── package.json
└── vite.config.ts
```

---

## 2. Key Frontend Capabilities

1. **Dual-Pane Synchronized Charting**:
   - **Primary Pane**: Candlestick chart rendering Open, High, Low, Close with color-coded bull/bear bodies.
   - **Overlay**: 20-period Moving Average and Session VWAP.
   - **Secondary Sub-Pane**: **BofA Net Order Flow (TL)** histogram and cumulative curve, highlighting institutional aggressive buy and sell phases.
   - **Crosshair Synchronization**: Moving the mouse over the price chart automatically snaps the order flow indicator to the exact same bar timestamp.
2. **Sub-15ms Latency**:
   - Backed by TimescaleDB Continuous Aggregates (`silver_candles_5m`) and FastAPI `asyncpg` queries.
3. **TradingView Lightweight Charts (v4.2+)**:
   - Canvas/WebGL rendering engine capable of displaying 100,000+ data points smoothly without DOM lag.
   - Fully open-source and free (Apache 2.0 license).
