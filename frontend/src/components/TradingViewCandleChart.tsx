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
import { CandleBar } from '../types/api';

interface TradingViewChartProps {
  symbol: string;
  interval?: '1m' | '5m' | '1d';
  brokerId?: string;
  fifoAvgCost?: number | null;
  showCostLine?: boolean;
  onHoverData?: (data: CandleBar | null) => void;
}

export const TradingViewCandleChart: React.FC<TradingViewChartProps> = ({
  symbol,
  interval = '5m',
  brokerId = 'MLB',
  fifoAvgCost,
  showCostLine = false,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const bofaSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const costLineRef = useRef<any>(null);
  const totalBarsRef = useRef<number>(0);
  const latestCandleRef = useRef<CandleBar | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCandle, setActiveCandle] = useState<CandleBar | null>(null);

  // Initialize TradingView Chart Canvas
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 560,
      layout: {
        background: { type: ColorType.Solid, color: '#090d16' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: 'rgba(30, 41, 59, 0.4)' },
        horzLines: { color: 'rgba(30, 41, 59, 0.4)' },
      },
      crosshair: {
        mode: 1, // Magnet mode
        vertLine: {
          color: '#06b6d4',
          width: 1,
          style: 3,
        },
        horzLine: {
          color: '#06b6d4',
          width: 1,
          style: 3,
        },
      },
      rightPriceScale: {
        borderColor: '#1e293b',
        scaleMargins: {
          top: 0.1,
          bottom: 0.28, // Leave room for lower sub-pane
        },
      },
      timeScale: {
        borderColor: '#1e293b',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 15, // 15 bars of whitespace on right: ensures latest date & candle are never blocked by scale
        barSpacing: 9,
        minBarSpacing: 2,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: true,
      },
    });

    chartRef.current = chart;

    // 1. Candlestick Series (Upper Pane)
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981', // Emerald
      downColor: '#f43f5e', // Rose
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#f43f5e',
    });
    candleSeriesRef.current = candleSeries;

    // 2. Broker Institutional Net Flow Histogram (Lower Sub-Pane)
    const bofaSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'bofa_flow',
    });

    chart.priceScale('bofa_flow').applyOptions({
      scaleMargins: {
        top: 0.76, // Pinned to bottom 24%
        bottom: 0.02,
      },
    });
    bofaSeriesRef.current = bofaSeries;

    // Crosshair move handler
    chart.subscribeCrosshairMove((param: any) => {
      if (!param || !param.time || !param.seriesData) {
        setActiveCandle(latestCandleRef.current);
        return;
      }
      const candlePrice = param.seriesData.get(candleSeries) as any;
      const bofaData = param.seriesData.get(bofaSeries) as any;
      if (candlePrice) {
        let epochSec = 0;
        if (typeof param.time === 'number') {
          epochSec = Number(param.time);
        } else if (typeof param.time === 'string') {
          epochSec = Math.floor(new Date(`${param.time}T00:00:00Z`).getTime() / 1000);
        } else if (typeof param.time === 'object' && param.time !== null) {
          epochSec = Math.floor(Date.UTC(param.time.year, param.time.month - 1, param.time.day) / 1000);
        }
        setActiveCandle({
          time: epochSec,
          open: candlePrice.open,
          high: candlePrice.high,
          low: candlePrice.low,
          close: candlePrice.close,
          volume: 0,
          turnover_tl: 0,
          trade_count: 0,
          bofa_net_flow_tl: bofaData?.value || 0,
        });
      }
    });

    // Handle Window Resize
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

  // Fetch candle data whenever symbol or interval changes
  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    const fetchCandleData = async () => {
      try {
        const response = await fetch(
          `/api/v1/market/candles?symbol=${encodeURIComponent(symbol)}&interval=${interval}&limit=1000`
        );
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }
        const data: CandleBar[] = await response.json();

        if (!isMounted) return;

        if (data.length === 0) {
          setError(`No candlestick records found for ${symbol} (${interval})`);
          setLoading(false);
          return;
        }

        const getTimeVal = (t: number) => {
          if (interval === '1d') {
            return new Date(t * 1000).toISOString().split('T')[0] as any;
          }
          return t as UTCTimestamp;
        };

        // Map to lightweight-charts CandlestickData
        const candleData: CandlestickData[] = data.map((d) => ({
          time: getTimeVal(d.time),
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));

        // Map to lightweight-charts HistogramData
        const bofaData: HistogramData[] = data.map((d) => ({
          time: getTimeVal(d.time),
          value: d.bofa_net_flow_tl,
          color: d.bofa_net_flow_tl >= 0 ? '#10b981' : '#f43f5e',
        }));

        if (candleSeriesRef.current && bofaSeriesRef.current && chartRef.current) {
          chartRef.current.timeScale().applyOptions({
            timeVisible: interval !== '1d',
          });
          candleSeriesRef.current.setData(candleData);
          bofaSeriesRef.current.setData(bofaData);

          const totalBars = candleData.length;
          totalBarsRef.current = totalBars;
          if (totalBars > 0) {
            const lastBar = data[data.length - 1];
            latestCandleRef.current = lastBar;
            setActiveCandle(lastBar);

            // Focus view on recent bars so the latest date and action are immediately visible without being squished
            const visibleCount = Math.min(totalBars, interval === '1d' ? 140 : 180);
            chartRef.current.timeScale().setVisibleLogicalRange({
              from: Math.max(0, totalBars - visibleCount),
              to: totalBars + 15,
            });
          }
        }

        setLoading(false);
      } catch (err: any) {
        if (!isMounted) return;
        setError(err.message || 'Failed to fetch candle data');
        setLoading(false);
      }
    };

    fetchCandleData();

    return () => {
      isMounted = false;
    };
  }, [symbol, interval]);

  // Update BofA FIFO Cost Price Line
  useEffect(() => {
    if (!candleSeriesRef.current) return;
    if (costLineRef.current) {
      try {
        candleSeriesRef.current.removePriceLine(costLineRef.current);
      } catch (e) {
        // Ignore removal error
      }
      costLineRef.current = null;
    }
    if (showCostLine && fifoAvgCost && fifoAvgCost > 0) {
      costLineRef.current = candleSeriesRef.current.createPriceLine({
        price: fifoAvgCost,
        color: '#eab308',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: `${brokerId} Cost: ₺${fifoAvgCost.toFixed(2)}`,
      });
    }
  }, [fifoAvgCost, showCostLine, brokerId]);

  return (
    <div className="relative w-full glass-panel rounded-xl overflow-hidden shadow-2xl border border-slate-800">
      {/* Sub-Header & Live Metric Inspector */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-900/90 border-b border-slate-800/80 text-xs">
        <div className="flex items-center space-x-3">
          <span className="font-semibold text-white tracking-wide text-sm">{symbol}</span>
          <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 font-mono">
            {interval.toUpperCase()}
          </span>
          <span className="text-slate-400">
            Sub-Pane: <span className="text-slate-200 font-medium">{brokerId} Net Flow (TL)</span>
          </span>
        </div>

        {/* Dynamic Crosshair Inspector */}
        {activeCandle ? (
          <div className="flex items-center space-x-3 font-mono text-slate-300">
            {activeCandle.time && (
              <span className="flex items-center space-x-1">
                <span className="text-slate-400">Date:</span>
                <span className="text-cyan-300 font-semibold">
                  {new Date(activeCandle.time * 1000).toISOString().split('T')[0]}
                </span>
                {latestCandleRef.current && activeCandle.time === latestCandleRef.current.time && (
                  <span className="ml-1 px-1 py-0.2 rounded text-[9px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                    LATEST
                  </span>
                )}
              </span>
            )}
            <span>O: <span className="text-white font-semibold">{activeCandle.open.toFixed(2)}</span></span>
            <span>H: <span className="text-emerald-400 font-semibold">{activeCandle.high.toFixed(2)}</span></span>
            <span>L: <span className="text-rose-400 font-semibold">{activeCandle.low.toFixed(2)}</span></span>
            <span>C: <span className="text-white font-semibold">{activeCandle.close.toFixed(2)}</span></span>
            <span>
              Net: <span className={activeCandle.bofa_net_flow_tl >= 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                ₺{(activeCandle.bofa_net_flow_tl / 1_000_000).toFixed(2)}M
              </span>
            </span>
          </div>
        ) : (
          <div className="text-slate-500 font-mono text-[11px]">Hover over candle to inspect micro-metrics</div>
        )}

        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 bg-slate-950/60 px-1.5 py-0.5 rounded border border-slate-800">
            <button
              onClick={() => {
                if (!chartRef.current) return;
                const totalBars = totalBarsRef.current;
                if (totalBars > 0) {
                  const visibleCount = Math.min(totalBars, interval === '1d' ? 140 : 180);
                  chartRef.current.timeScale().setVisibleLogicalRange({
                    from: Math.max(0, totalBars - visibleCount),
                    to: totalBars + 15,
                  });
                }
              }}
              className="px-1.5 py-0.5 rounded text-[10px] font-mono text-slate-300 hover:text-cyan-300 hover:bg-slate-800 transition-colors"
              title="Focus view on latest date with right breathing room"
            >
              Latest
            </button>
            <span className="text-slate-700">|</span>
            <button
              onClick={() => {
                if (!chartRef.current) return;
                chartRef.current.timeScale().fitContent();
              }}
              className="px-1.5 py-0.5 rounded text-[10px] font-mono text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Fit all historical records onto canvas"
            >
              Fit All
            </button>
          </div>
          {loading && <span className="text-cyan-400 animate-pulse font-mono">Streaming...</span>}
          {error && <span className="text-rose-400 font-mono">{error}</span>}
        </div>
      </div>

      {/* Chart Canvas */}
      <div ref={chartContainerRef} className="w-full h-[560px]" />
    </div>
  );
};
