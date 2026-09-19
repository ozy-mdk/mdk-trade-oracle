import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  UTCTimestamp,
  ColorType,
} from 'lightweight-charts';

export interface CandleRecord {
  time: number; // UNIX epoch seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  turnover_tl: number;
  trade_count: number;
  bofa_net_flow_tl: number;
}

interface TradingViewChartProps {
  symbol: string;
  interval?: '1m' | '5m' | '1d';
  apiBaseUrl?: string;
}

export const TradingViewCandleChart: React.FC<TradingViewChartProps> = ({
  symbol,
  interval = '5m',
  apiBaseUrl = 'http://localhost:8000',
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const bofaSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // 1. Initialize Chart Canvas
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 550,
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' }, // Slate-900
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        mode: 1, // Magnet mode
      },
      rightPriceScale: {
        borderColor: '#334155',
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // 2. Add Candlestick Series (Upper Pane)
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981', // Emerald-500
      downColor: '#ef4444', // Red-500
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });
    candleSeriesRef.current = candleSeries;

    // 3. Add BofA Net Order Flow Histogram (Sub-Pane at bottom)
    const bofaSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'bofa_flow', // Separate sub-scale
    });

    chart.priceScale('bofa_flow').applyOptions({
      scaleMargins: {
        top: 0.75, // Keeps BofA flow pinned to bottom 25% of the chart
        bottom: 0,
      },
    });
    bofaSeriesRef.current = bofaSeries;

    // 4. Resize listener
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  // Fetch candle data from FastAPI backend whenever symbol or interval changes
  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    const fetchCandles = async () => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/api/v1/market/candles?symbol=${symbol}&interval=${interval}&limit=1000`
        );
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }
        const data: CandleRecord[] = await response.json();

        if (!isMounted) return;

        // Transform for TradingView Candlestick
        const candleData: CandlestickData[] = data.map((d) => ({
          time: d.time as UTCTimestamp,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));

        // Transform for BofA Net Flow (Green if net positive, Red if net negative)
        const bofaData: HistogramData[] = data.map((d) => ({
          time: d.time as UTCTimestamp,
          value: d.bofa_net_flow_tl,
          color: d.bofa_net_flow_tl >= 0 ? '#059669' : '#dc2626',
        }));

        if (candleSeriesRef.current && bofaSeriesRef.current && chartRef.current) {
          candleSeriesRef.current.setData(candleData);
          bofaSeriesRef.current.setData(bofaData);
          chartRef.current.timeScale().fitContent();
        }

        setLoading(false);
      } catch (err: any) {
        if (!isMounted) return;
        setError(err.message || 'Failed to fetch candlestick data');
        setLoading(false);
      }
    };

    fetchCandles();

    return () => {
      isMounted = false;
    };
  }, [symbol, interval, apiBaseUrl]);

  return (
    <div className="relative w-full bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-800/80 border-b border-slate-700">
        <div className="flex items-center space-x-3">
          <span className="text-xl font-bold text-white tracking-wider">{symbol}</span>
          <span className="px-2 py-0.5 text-xs font-semibold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            {interval.toUpperCase()}
          </span>
          <span className="text-xs text-slate-400">
            Sub-Pane: BofA (MLB) Net Flow (TL)
          </span>
        </div>
        {loading && <div className="text-xs text-cyan-400 animate-pulse">Streaming data...</div>}
        {error && <div className="text-xs text-red-400">{error}</div>}
      </div>

      {/* Chart Canvas */}
      <div ref={chartContainerRef} className="w-full h-[550px]" />
    </div>
  );
};
