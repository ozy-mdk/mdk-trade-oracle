import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineData,
  UTCTimestamp,
  ColorType,
} from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { fetchTertipTimeseries } from '../api/client';
import { TertipTimeseriesPoint } from '../types/api';
import { formatVolume } from '../utils/formatters';

interface TertipEwmaChartProps {
  symbol: string;
  brokerId?: string;
}

export const TertipEwmaChart: React.FC<TertipEwmaChartProps> = ({
  symbol,
  brokerId = 'MLB',
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // Upper Pane Series (Price & Cost)
  const priceSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const costSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const cost63SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // Lower Sub-Pane Series (EWMA Ribbons)
  const qtySeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma5SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma21SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma63SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma126SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma252SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // Visibility Toggles
  const [showCost, setShowCost] = useState(true);
  const [showCost63, setShowCost63] = useState(true);
  const [showEwma5, setShowEwma5] = useState(true);
  const [showEwma21, setShowEwma21] = useState(true);
  const [showEwma63, setShowEwma63] = useState(true);
  const [showEwma126, setShowEwma126] = useState(true);
  const [showEwma252, setShowEwma252] = useState(true);

  // Hover inspector state
  const [hoveredPoint, setHoveredPoint] = useState<TertipTimeseriesPoint | null>(null);

  // Fetch Time Series Data
  const { data: timeseries, isLoading, error } = useQuery({
    queryKey: ['tertipTimeseries', symbol, brokerId],
    queryFn: () => fetchTertipTimeseries(symbol, brokerId, 500),
  });

  // Initialize TradingView Canvas
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const initialWidth = chartContainerRef.current.clientWidth > 0 ? chartContainerRef.current.clientWidth : 800;
    const chart = createChart(chartContainerRef.current, {
      width: initialWidth,
      height: 520,
      layout: {
        background: { type: ColorType.Solid, color: '#090d16' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: 'rgba(30, 41, 59, 0.4)' },
        horzLines: { color: 'rgba(30, 41, 59, 0.4)' },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: '#06b6d4', width: 1, style: 3 },
        horzLine: { color: '#06b6d4', width: 1, style: 3 },
      },
      rightPriceScale: {
        borderColor: '#1e293b',
        scaleMargins: {
          top: 0.05,
          bottom: 0.45, // Upper pane occupies top 55%
        },
      },
      timeScale: {
        borderColor: '#1e293b',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // ── Upper Pane: Price & Costs (Right Price Scale) ──────────────────────────
    // 1. Stock Close Price
    const priceSeries = chart.addLineSeries({
      color: '#38bdf8', // Sky blue
      lineWidth: 2,
      priceScaleId: 'right',
      title: 'Close Price',
    });
    priceSeriesRef.current = priceSeries;

    // 2. FIFO Average Unit Cost
    const costSeries = chart.addLineSeries({
      color: '#eab308', // Amber
      lineWidth: 2,
      lineStyle: 2, // Dashed
      priceScaleId: 'right',
      title: 'FIFO Avg Cost',
    });
    costSeriesRef.current = costSeries;

    // 3. 3M EWMA Unit Cost
    const cost63Series = chart.addLineSeries({
      color: '#a855f7', // Purple
      lineWidth: 1,
      lineStyle: 3, // Dotted
      priceScaleId: 'right',
      title: '3M EWMA Cost',
    });
    cost63SeriesRef.current = cost63Series;

    // ── Lower Pane: EWMA Inventory Ribbons (Dedicated Scale) ───────────────────
    // 4. Actual Open Inventory Quantity (Registers 'inventory_scale')
    const qtySeries = chart.addLineSeries({
      color: '#64748b', // Slate
      lineWidth: 1,
      priceScaleId: 'inventory_scale',
      title: 'Open Qty',
    });
    qtySeriesRef.current = qtySeries;

    // Apply scale margins AFTER creating the series with that priceScaleId
    try {
      chart.priceScale('inventory_scale').applyOptions({
        scaleMargins: {
          top: 0.60, // Pinned to bottom 40%
          bottom: 0.02,
        },
      });
    } catch (err) {
      console.warn('Could not apply inventory_scale options:', err);
    }

    // 5. 1W EWMA (5d)
    const ewma5Series = chart.addLineSeries({
      color: '#f59e0b', // Amber
      lineWidth: 2,
      priceScaleId: 'inventory_scale',
      title: '1W EWMA',
    });
    ewma5SeriesRef.current = ewma5Series;

    // 6. 1M EWMA (21d)
    const ewma21Series = chart.addLineSeries({
      color: '#06b6d4', // Cyan
      lineWidth: 2,
      priceScaleId: 'inventory_scale',
      title: '1M EWMA',
    });
    ewma21SeriesRef.current = ewma21Series;

    // 7. 3M EWMA (63d)
    const ewma63Series = chart.addLineSeries({
      color: '#8b5cf6', // Violet
      lineWidth: 2,
      priceScaleId: 'inventory_scale',
      title: '3M EWMA',
    });
    ewma63SeriesRef.current = ewma63Series;

    // 8. 6M EWMA (126d)
    const ewma126Series = chart.addLineSeries({
      color: '#10b981', // Emerald
      lineWidth: 2,
      priceScaleId: 'inventory_scale',
      title: '6M EWMA',
    });
    ewma126SeriesRef.current = ewma126Series;

    // 9. 12M EWMA (252d)
    const ewma252Series = chart.addLineSeries({
      color: '#f43f5e', // Rose
      lineWidth: 2,
      priceScaleId: 'inventory_scale',
      title: '12M EWMA',
    });
    ewma252SeriesRef.current = ewma252Series;

    // Window Resize & Container Observer Handler
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0 && chart) {
          chart.applyOptions({ width: entry.contentRect.width });
        }
      }
    });
    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  // Update Data when timeseries loads
  useEffect(() => {
    if (!timeseries || timeseries.length === 0 || !chartRef.current) return;

    const priceData: LineData[] = [];
    const costData: LineData[] = [];
    const cost63Data: LineData[] = [];
    const qtyData: LineData[] = [];
    const ewma5Data: LineData[] = [];
    const ewma21Data: LineData[] = [];
    const ewma63Data: LineData[] = [];
    const ewma126Data: LineData[] = [];
    const ewma252Data: LineData[] = [];

    for (const p of timeseries) {
      const t = p.time as UTCTimestamp;
      priceData.push({ time: t, value: p.close_price });
      costData.push({ time: t, value: p.fifo_avg_cost });
      cost63Data.push({ time: t, value: p.ewma_cost_63d });
      qtyData.push({ time: t, value: p.open_quantity });
      ewma5Data.push({ time: t, value: p.ewma_qty_5d });
      ewma21Data.push({ time: t, value: p.ewma_qty_21d });
      ewma63Data.push({ time: t, value: p.ewma_qty_63d });
      ewma126Data.push({ time: t, value: p.ewma_qty_126d });
      ewma252Data.push({ time: t, value: p.ewma_qty_252d });
    }

    priceSeriesRef.current?.setData(priceData);
    costSeriesRef.current?.setData(costData);
    cost63SeriesRef.current?.setData(cost63Data);
    qtySeriesRef.current?.setData(qtyData);
    ewma5SeriesRef.current?.setData(ewma5Data);
    ewma21SeriesRef.current?.setData(ewma21Data);
    ewma63SeriesRef.current?.setData(ewma63Data);
    ewma126SeriesRef.current?.setData(ewma126Data);
    ewma252SeriesRef.current?.setData(ewma252Data);

    chartRef.current.timeScale().fitContent();

    // Set latest point for inspector
    setHoveredPoint(timeseries[timeseries.length - 1]);
  }, [timeseries]);

  // Handle visibility toggles
  useEffect(() => {
    costSeriesRef.current?.applyOptions({ visible: showCost });
    cost63SeriesRef.current?.applyOptions({ visible: showCost63 });
    ewma5SeriesRef.current?.applyOptions({ visible: showEwma5 });
    ewma21SeriesRef.current?.applyOptions({ visible: showEwma21 });
    ewma63SeriesRef.current?.applyOptions({ visible: showEwma63 });
    ewma126SeriesRef.current?.applyOptions({ visible: showEwma126 });
    ewma252SeriesRef.current?.applyOptions({ visible: showEwma252 });
  }, [showCost, showCost63, showEwma5, showEwma21, showEwma63, showEwma126, showEwma252]);

  return (
    <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-2xl">
      {/* Chart Header & Ribbon Legend Controls */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-900/90 border-b border-slate-800 text-xs gap-2">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-white tracking-wide">{symbol}</span>
          <span className="text-slate-400">•</span>
          <span className="text-slate-300 font-mono text-[11px]">
            Price vs. {brokerId} Tertip EWMA Ribbons
          </span>
        </div>

        {/* Interactive Legend Toggles */}
        <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono">
          <button
            onClick={() => setShowCost(!showCost)}
            className={`flex items-center space-x-1 px-2 py-0.5 rounded border transition-colors ${
              showCost
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                : 'bg-slate-900 text-slate-500 border-slate-800'
            }`}
          >
            <span className="w-2 h-0.5 bg-amber-400 inline-block" />
            <span>FIFO Cost</span>
          </button>

          <button
            onClick={() => setShowCost63(!showCost63)}
            className={`flex items-center space-x-1 px-2 py-0.5 rounded border transition-colors ${
              showCost63
                ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                : 'bg-slate-900 text-slate-500 border-slate-800'
            }`}
          >
            <span className="w-2 h-0.5 bg-purple-400 inline-block" />
            <span>3M Cost</span>
          </button>

          <span className="text-slate-700">|</span>

          <button
            onClick={() => setShowEwma5(!showEwma5)}
            className={`flex items-center space-x-1 px-2 py-0.5 rounded border transition-colors ${
              showEwma5
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                : 'bg-slate-900 text-slate-500 border-slate-800'
            }`}
          >
            <span className="w-2 h-0.5 bg-amber-400 inline-block" />
            <span>1W EWMA</span>
          </button>

          <button
            onClick={() => setShowEwma21(!showEwma21)}
            className={`flex items-center space-x-1 px-2 py-0.5 rounded border transition-colors ${
              showEwma21
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                : 'bg-slate-900 text-slate-500 border-slate-800'
            }`}
          >
            <span className="w-2 h-0.5 bg-cyan-400 inline-block" />
            <span>1M EWMA</span>
          </button>

          <button
            onClick={() => setShowEwma63(!showEwma63)}
            className={`flex items-center space-x-1 px-2 py-0.5 rounded border transition-colors ${
              showEwma63
                ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                : 'bg-slate-900 text-slate-500 border-slate-800'
            }`}
          >
            <span className="w-2 h-0.5 bg-purple-400 inline-block" />
            <span>3M EWMA</span>
          </button>

          <button
            onClick={() => setShowEwma126(!showEwma126)}
            className={`flex items-center space-x-1 px-2 py-0.5 rounded border transition-colors ${
              showEwma126
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                : 'bg-slate-900 text-slate-500 border-slate-800'
            }`}
          >
            <span className="w-2 h-0.5 bg-emerald-400 inline-block" />
            <span>6M EWMA</span>
          </button>

          <button
            onClick={() => setShowEwma252(!showEwma252)}
            className={`flex items-center space-x-1 px-2 py-0.5 rounded border transition-colors ${
              showEwma252
                ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                : 'bg-slate-900 text-slate-500 border-slate-800'
            }`}
          >
            <span className="w-2 h-0.5 bg-rose-400 inline-block" />
            <span>12M EWMA</span>
          </button>
        </div>
      </div>

      {/* Point Inspector Strip */}
      {hoveredPoint && (
        <div className="px-4 py-1.5 bg-slate-950/80 border-b border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400">
          <div className="flex items-center space-x-3">
            <span>Date: <span className="text-white font-semibold">{hoveredPoint.trade_date}</span></span>
            <span>Close: <span className="text-sky-300 font-semibold">₺{hoveredPoint.close_price.toFixed(2)}</span></span>
            <span>FIFO Cost: <span className="text-amber-300 font-semibold">₺{hoveredPoint.fifo_avg_cost.toFixed(2)}</span></span>
            <span>3M Cost: <span className="text-purple-300 font-semibold">₺{hoveredPoint.ewma_cost_63d.toFixed(2)}</span></span>
          </div>

          <div className="flex items-center space-x-3">
            <span>Qty: <span className="text-slate-200 font-semibold">{formatVolume(hoveredPoint.open_quantity)}</span></span>
            <span>1W: <span className="text-amber-300">{formatVolume(hoveredPoint.ewma_qty_5d)}</span></span>
            <span>1M: <span className="text-cyan-300">{formatVolume(hoveredPoint.ewma_qty_21d)}</span></span>
            <span>3M: <span className="text-purple-300">{formatVolume(hoveredPoint.ewma_qty_63d)}</span></span>
            <span>6M: <span className="text-emerald-300">{formatVolume(hoveredPoint.ewma_qty_126d)}</span></span>
            <span>12M: <span className="text-rose-300">{formatVolume(hoveredPoint.ewma_qty_252d)}</span></span>
          </div>
        </div>
      )}

      {/* Chart Canvas */}
      <div ref={chartContainerRef} className="w-full relative min-h-[520px]">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/60 z-10 text-cyan-400 text-xs font-mono">
            Loading Tertip EWMA time series...
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/60 z-10 text-rose-400 text-xs font-mono">
            Failed to load time series data
          </div>
        )}
      </div>

      {/* Footer Guidance */}
      <div className="px-4 py-1.5 bg-slate-900/60 border-t border-slate-800 text-[10px] text-slate-500 flex items-center justify-between">
        <span>Upper Pane: Price & FIFO Costs • Lower Pane: BofA Open Inventory & Multi-Horizon EWMA Ribbons</span>
        <span>Scroll to Zoom • Drag to Pan</span>
      </div>
    </div>
  );
};
