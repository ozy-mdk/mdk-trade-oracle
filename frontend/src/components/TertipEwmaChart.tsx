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
import { fetchTertipTimeseries, fetchTertipHorizons } from '../api/client';
import {
  TertipTimeseriesPoint,
  TertipHorizonsResponse,
  ForwardHorizonCode,
} from '../types/api';
import { formatVolume } from '../utils/formatters';
import {
  calculateForwardOpportunity,
  calculateEwmaConfluence,
} from '../utils/forwardOpportunity';
import {
  Compass,
  Target,
  TrendingUp,
  TrendingDown,
  Eye,
  EyeOff,
  Table,
} from 'lucide-react';

interface TertipEwmaChartProps {
  symbol: string;
  brokerId?: string;
  tertipData?: TertipHorizonsResponse | null;
}

type CostHorizonId = 'none' | '5d' | '10d' | '21d' | '63d' | '126d' | '252d';

interface CostHorizonConfig {
  id: CostHorizonId;
  label: string;
  shortLabel: string;
  color: string;
}

const COST_HORIZONS: CostHorizonConfig[] = [
  { id: '5d', label: '1W EWMA Cost', shortLabel: '1W', color: '#f59e0b' },
  { id: '10d', label: '2W EWMA Cost', shortLabel: '2W', color: '#fb923c' },
  { id: '21d', label: '1M EWMA Cost', shortLabel: '1M', color: '#06b6d4' },
  { id: '63d', label: '3M EWMA Cost', shortLabel: '3M', color: '#a855f7' },
  { id: '126d', label: '6M EWMA Cost', shortLabel: '6M', color: '#10b981' },
  { id: '252d', label: '12M EWMA Cost', shortLabel: '12M', color: '#f43f5e' },
];

export const TertipEwmaChart: React.FC<TertipEwmaChartProps> = ({
  symbol,
  brokerId = 'MLB',
  tertipData: tertipDataProp,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // Upper Pane Series (Price & Cost)
  const priceSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const costSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const costEwmaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const forwardAreaSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const outlookTargetLineRef = useRef<any>(null);

  // Lower Sub-Pane Series (EWMA Ribbons)
  const qtySeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma5SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma21SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma63SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma126SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ewma252SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // Visibility Toggles
  const [showCost, setShowCost] = useState(true);
  const [costHorizon, setCostHorizon] = useState<CostHorizonId>('63d');
  const [showEwma5, setShowEwma5] = useState(true);
  const [showEwma21, setShowEwma21] = useState(true);
  const [showEwma63, setShowEwma63] = useState(true);
  const [showEwma126, setShowEwma126] = useState(true);
  const [showEwma252, setShowEwma252] = useState(true);

  // Action Zone State
  const [showOutlookZone, setShowOutlookZone] = useState<boolean>(true);
  const [actionZoneTab, setActionZoneTab] = useState<'table' | 'outlook'>('table');
  const [selectedHorizon, setSelectedHorizon] = useState<ForwardHorizonCode>('3M');

  // Hover inspector state
  const [hoveredPoint, setHoveredPoint] = useState<TertipTimeseriesPoint | null>(null);
  const timeseriesRef = useRef<TertipTimeseriesPoint[]>([]);
  const timeMapRef = useRef<Map<string | number, TertipTimeseriesPoint>>(new Map());

  // Fetch Time Series Data (all available history up to latest date)
  const { data: timeseries, isLoading, error } = useQuery({
    queryKey: ['tertipTimeseries', symbol, brokerId],
    queryFn: () => fetchTertipTimeseries(symbol, brokerId, 1500),
  });

  // Fetch Horizons if not provided via props
  const { data: fetchedTertipData } = useQuery({
    queryKey: ['tertipHorizons', symbol, brokerId],
    queryFn: () => fetchTertipHorizons(symbol, brokerId),
    enabled: !tertipDataProp,
  });
  const tertipData = tertipDataProp || fetchedTertipData;

  const currentPrice =
    hoveredPoint?.close_price ||
    (timeseries && timeseries.length > 0 ? timeseries[timeseries.length - 1].close_price : 0) ||
    tertipData?.market_close_price ||
    0;

  const outlook = calculateForwardOpportunity(currentPrice, tertipData, selectedHorizon);
  const confluence = calculateEwmaConfluence(currentPrice, tertipData);

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
        timeVisible: false,
        secondsVisible: false,
        rightOffset: 15, // 15 bars margin on right: ensures latest date & points are never blocked by axis
        barSpacing: 8,
        minBarSpacing: 2,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: true,
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

    // 3. EWMA Unit Cost (Dynamic Horizon)
    const costEwmaSeries = chart.addLineSeries({
      color: '#a855f7', // Purple default
      lineWidth: 2,
      lineStyle: 3, // Dotted
      priceScaleId: 'right',
      title: '3M EWMA Cost',
    });
    costEwmaSeriesRef.current = costEwmaSeries;

    // 3b. Forward Opportunity Corridor (Upper Pane)
    const forwardAreaSeries = chart.addAreaSeries({
      priceScaleId: 'right',
      topColor: 'rgba(16, 185, 129, 0.18)',
      bottomColor: 'rgba(16, 185, 129, 0.01)',
      lineColor: '#10b981',
      lineWidth: 1,
      lineStyle: 2,
    });
    forwardAreaSeriesRef.current = forwardAreaSeries;

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

    // Crosshair move handler: inspect hovered date or snap to latest date on exit
    chart.subscribeCrosshairMove((param: any) => {
      if (!param || !param.time) {
        if (timeseriesRef.current && timeseriesRef.current.length > 0) {
          setHoveredPoint(timeseriesRef.current[timeseriesRef.current.length - 1]);
        }
        return;
      }
      const raw = param.time;
      let dateKey: string | number = raw;
      if (typeof raw === 'object' && raw !== null) {
        dateKey = `${raw.year}-${String(raw.month).padStart(2, '0')}-${String(raw.day).padStart(2, '0')}`;
      }
      const pt = timeMapRef.current.get(dateKey) || timeMapRef.current.get(String(raw)) || timeMapRef.current.get(Number(raw));
      if (pt) {
        setHoveredPoint(pt);
      }
    });

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
      forwardAreaSeriesRef.current = null;
      outlookTargetLineRef.current = null;
    };
  }, []);

  function getPointCost(p: TertipTimeseriesPoint, horizon: CostHorizonId): number {
    if (horizon === '5d') return p.ewma_cost_5d ?? p.fifo_avg_cost;
    if (horizon === '10d') return p.ewma_cost_10d ?? p.fifo_avg_cost;
    if (horizon === '21d') return p.ewma_cost_21d ?? p.fifo_avg_cost;
    if (horizon === '63d') return p.ewma_cost_63d ?? p.fifo_avg_cost;
    if (horizon === '126d') return p.ewma_cost_126d ?? p.fifo_avg_cost;
    if (horizon === '252d') return p.ewma_cost_252d ?? p.fifo_avg_cost;
    return p.fifo_avg_cost;
  }

  // Update Data when timeseries loads
  useEffect(() => {
    if (!timeseries || timeseries.length === 0 || !chartRef.current) return;

    const priceData: LineData[] = [];
    const costData: LineData[] = [];
    const qtyData: LineData[] = [];
    const ewma5Data: LineData[] = [];
    const ewma21Data: LineData[] = [];
    const ewma63Data: LineData[] = [];
    const ewma126Data: LineData[] = [];
    const ewma252Data: LineData[] = [];

    for (const p of timeseries) {
      // Use trade_date string directly so dates are 100% calendar-aligned with zero timezone distortion
      const t = p.trade_date as any;
      priceData.push({ time: t, value: p.close_price });
      costData.push({ time: t, value: p.fifo_avg_cost });
      qtyData.push({ time: t, value: p.open_quantity });
      ewma5Data.push({ time: t, value: p.ewma_qty_5d });
      ewma21Data.push({ time: t, value: p.ewma_qty_21d });
      ewma63Data.push({ time: t, value: p.ewma_qty_63d });
      ewma126Data.push({ time: t, value: p.ewma_qty_126d });
      ewma252Data.push({ time: t, value: p.ewma_qty_252d });
    }

    priceSeriesRef.current?.setData(priceData);
    costSeriesRef.current?.setData(costData);
    qtySeriesRef.current?.setData(qtyData);
    ewma5SeriesRef.current?.setData(ewma5Data);
    ewma21SeriesRef.current?.setData(ewma21Data);
    ewma63SeriesRef.current?.setData(ewma63Data);
    ewma126SeriesRef.current?.setData(ewma126Data);
    ewma252SeriesRef.current?.setData(ewma252Data);

    // Populate EWMA Cost Series
    if (costEwmaSeriesRef.current) {
      if (costHorizon === 'none') {
        costEwmaSeriesRef.current.applyOptions({ visible: false });
      } else {
        const cfg = COST_HORIZONS.find((c) => c.id === costHorizon);
        const ewmaCostData: LineData[] = timeseries.map((p) => ({
          time: p.trade_date as any,
          value: getPointCost(p, costHorizon),
        }));
        costEwmaSeriesRef.current.setData(ewmaCostData);
        costEwmaSeriesRef.current.applyOptions({
          visible: true,
          color: cfg?.color || '#a855f7',
          title: cfg?.label || 'EWMA Cost',
        });
      }
    }

    const totalPoints = timeseries.length;
    timeseriesRef.current = timeseries;
    const timeMap = new Map<string | number, TertipTimeseriesPoint>();
    for (const p of timeseries) {
      timeMap.set(p.trade_date, p);
      timeMap.set(p.time, p);
    }
    timeMapRef.current = timeMap;

    if (totalPoints > 0) {
      // Focus on recent 140 sessions with 15-bar margin on right so latest date is clearly visible and unblocked
      const visibleCount = Math.min(totalPoints, 140);
      chartRef.current.timeScale().setVisibleLogicalRange({
        from: Math.max(0, totalPoints - visibleCount),
        to: totalPoints + 15,
      });
    }

    // Set latest point for inspector
    setHoveredPoint(timeseries[timeseries.length - 1]);
  }, [timeseries]);

  // Handle EWMA Cost Horizon switch
  useEffect(() => {
    if (!timeseries || timeseries.length === 0 || !costEwmaSeriesRef.current) return;

    if (costHorizon === 'none') {
      costEwmaSeriesRef.current.applyOptions({ visible: false });
      return;
    }

    const cfg = COST_HORIZONS.find((c) => c.id === costHorizon);
    const ewmaCostData: LineData[] = timeseries.map((p) => ({
      time: p.time as UTCTimestamp,
      value: getPointCost(p, costHorizon),
    }));
    costEwmaSeriesRef.current.setData(ewmaCostData);
    costEwmaSeriesRef.current.applyOptions({
      visible: true,
      color: cfg?.color || '#a855f7',
      title: cfg?.label || 'EWMA Cost',
    });
  }, [costHorizon, timeseries]);

  // Handle visibility toggles
  useEffect(() => {
    costSeriesRef.current?.applyOptions({ visible: showCost });
    ewma5SeriesRef.current?.applyOptions({ visible: showEwma5 });
    ewma21SeriesRef.current?.applyOptions({ visible: showEwma21 });
    ewma63SeriesRef.current?.applyOptions({ visible: showEwma63 });
    ewma126SeriesRef.current?.applyOptions({ visible: showEwma126 });
    ewma252SeriesRef.current?.applyOptions({ visible: showEwma252 });
  }, [showCost, showEwma5, showEwma21, showEwma63, showEwma126, showEwma252]);

  const activeCostCfg = COST_HORIZONS.find((c) => c.id === costHorizon);
  const targetCostForOpp = hoveredPoint
    ? (costHorizon !== 'none' ? getPointCost(hoveredPoint, costHorizon) : hoveredPoint.fifo_avg_cost)
    : 0;
  const costSpreadPct = hoveredPoint && targetCostForOpp > 0
    ? ((hoveredPoint.close_price - targetCostForOpp) / targetCostForOpp) * 100
    : 0;
  const potentialReturnPct = hoveredPoint && targetCostForOpp > 0
    ? ((targetCostForOpp - hoveredPoint.close_price) / hoveredPoint.close_price) * 100
    : 0;

  const handleSelectHorizon = (code: ForwardHorizonCode) => {
    setSelectedHorizon(code);
    if (code === '1W') setCostHorizon('5d');
    else if (code === '2W') setCostHorizon('10d');
    else if (code === '1M') setCostHorizon('21d');
    else if (code === '3M') setCostHorizon('63d');
    else if (code === '6M') setCostHorizon('126d');
    else if (code === 'FIFO') setCostHorizon('none');
  };

  // Forward Opportunity Projection (Target Line & Forward Shaded Ribbon on Upper Pane)
  useEffect(() => {
    if (!priceSeriesRef.current || !chartRef.current) return;

    if (outlookTargetLineRef.current) {
      try {
        priceSeriesRef.current.removePriceLine(outlookTargetLineRef.current);
      } catch (e) {
        // Ignore removal error
      }
      outlookTargetLineRef.current = null;
    }

    if (!showOutlookZone || !outlook || !forwardAreaSeriesRef.current || !timeseries || timeseries.length === 0) {
      forwardAreaSeriesRef.current?.setData([]);
      return;
    }

    // 1. Create Target Price Line extending into the right scale
    const lineColor =
      outlook.direction === 'BUY'
        ? '#10b981'
        : outlook.direction === 'SELL'
        ? '#f43f5e'
        : '#64748b';
    const returnPrefix = outlook.potentialReturnPct >= 0 ? '+' : '';

    outlookTargetLineRef.current = priceSeriesRef.current.createPriceLine({
      price: outlook.targetCost,
      color: lineColor,
      lineWidth: 2,
      lineStyle: 2, // Dashed
      axisLabelVisible: true,
      title: `${brokerId} ${outlook.horizonCode} Target: ₺${outlook.targetCost.toFixed(2)} (${returnPrefix}${outlook.potentialReturnPct.toFixed(1)}%)`,
    });

    // 2. Populate Forward Area Ribbon across the 15-bar margin
    const lastPoint = timeseries[timeseries.length - 1];
    const lastClose = lastPoint.close_price;
    const targetCost = outlook.targetCost;
    const startDateStr = lastPoint.trade_date;
    const curDate = new Date(`${startDateStr}T00:00:00Z`);

    const topCol =
      outlook.direction === 'BUY'
        ? 'rgba(16, 185, 129, 0.18)'
        : outlook.direction === 'SELL'
        ? 'rgba(244, 63, 94, 0.18)'
        : 'rgba(100, 116, 139, 0.12)';
    const botCol =
      outlook.direction === 'BUY'
        ? 'rgba(16, 185, 129, 0.01)'
        : outlook.direction === 'SELL'
        ? 'rgba(244, 63, 94, 0.01)'
        : 'rgba(100, 116, 139, 0.01)';

    forwardAreaSeriesRef.current.applyOptions({
      topColor: topCol,
      bottomColor: botCol,
      lineColor: lineColor,
      lineWidth: 1,
      lineStyle: 2,
    });

    const forwardPoints: { time: any; value: number }[] = [];
    forwardPoints.push({
      time: startDateStr,
      value: lastClose,
    });

    const forwardDays = 8;
    for (let i = 1; i <= forwardDays; i++) {
      curDate.setUTCDate(curDate.getUTCDate() + 1);
      while (curDate.getUTCDay() === 0 || curDate.getUTCDay() === 6) {
        curDate.setUTCDate(curDate.getUTCDate() + 1);
      }
      const fDateStr = curDate.toISOString().split('T')[0];
      const alpha = i / forwardDays;
      const rampValue = Number((lastClose + (targetCost - lastClose) * alpha).toFixed(2));
      forwardPoints.push({
        time: fDateStr,
        value: rampValue,
      });
    }

    forwardAreaSeriesRef.current.setData(forwardPoints as any);
  }, [outlook, showOutlookZone, timeseries, brokerId]);

  return (
    <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden shadow-2xl">
      {/* Chart Header & Ribbon Legend Controls */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-900/90 border-b border-slate-800 text-xs gap-2">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-white tracking-wide">{symbol}</span>
          <span className="text-slate-400">•</span>
          <span className="text-slate-300 font-mono text-[11px]">
            Price & Costs (Top) vs. {brokerId} Inventory Ribbons (Bottom)
          </span>
          {timeseries && timeseries.length > 0 && (
            <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[10px] font-mono font-semibold">
              Latest: {timeseries[timeseries.length - 1].trade_date}
            </span>
          )}
        </div>

        {/* Interactive Legend Toggles */}
        <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
          {/* Upper Pane Cost Controls */}
          <div className="flex items-center space-x-1 bg-slate-950/70 px-2 py-1 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 mr-1 font-sans">Cost Line:</span>
            <button
              onClick={() => setShowCost(!showCost)}
              className={`flex items-center space-x-1 px-1.5 py-0.5 rounded border transition-colors ${
                showCost
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="Toggle FIFO Actual Unit Cost"
            >
              <span className="w-2 h-0.5 bg-amber-400 inline-block" />
              <span>FIFO</span>
            </button>

            {COST_HORIZONS.map((h) => {
              const isActive = costHorizon === h.id;
              return (
                <button
                  key={h.id}
                  onClick={() => setCostHorizon(isActive ? 'none' : h.id)}
                  className={`px-1.5 py-0.5 rounded border transition-all ${
                    isActive
                      ? 'border-transparent font-bold text-slate-950 shadow-sm'
                      : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-white'
                  }`}
                  style={{
                    backgroundColor: isActive ? h.color : undefined,
                  }}
                  title={`Show ${h.label} on upper price pane`}
                >
                  {h.shortLabel}
                </button>
              );
            })}
          </div>

          <span className="text-slate-700 hidden sm:inline">|</span>

          {/* Lower Pane Ribbons Controls */}
          <div className="flex items-center space-x-1 bg-slate-950/70 px-2 py-1 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 mr-1 font-sans">Ribbons:</span>
            <button
              onClick={() => setShowEwma5(!showEwma5)}
              className={`flex items-center space-x-1 px-1.5 py-0.5 rounded border transition-colors ${
                showEwma5
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="1-Week EWMA Quantity"
            >
              <span className="w-2 h-0.5 bg-amber-400 inline-block" />
              <span>1W</span>
            </button>

            <button
              onClick={() => setShowEwma21(!showEwma21)}
              className={`flex items-center space-x-1 px-1.5 py-0.5 rounded border transition-colors ${
                showEwma21
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="1-Month EWMA Quantity"
            >
              <span className="w-2 h-0.5 bg-cyan-400 inline-block" />
              <span>1M</span>
            </button>

            <button
              onClick={() => setShowEwma63(!showEwma63)}
              className={`flex items-center space-x-1 px-1.5 py-0.5 rounded border transition-colors ${
                showEwma63
                  ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="3-Month EWMA Quantity"
            >
              <span className="w-2 h-0.5 bg-purple-400 inline-block" />
              <span>3M</span>
            </button>

            <button
              onClick={() => setShowEwma126(!showEwma126)}
              className={`flex items-center space-x-1 px-1.5 py-0.5 rounded border transition-colors ${
                showEwma126
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="6-Month EWMA Quantity"
            >
              <span className="w-2 h-0.5 bg-emerald-400 inline-block" />
              <span>6M</span>
            </button>

            <button
              onClick={() => setShowEwma252(!showEwma252)}
              className={`flex items-center space-x-1 px-1.5 py-0.5 rounded border transition-colors ${
                showEwma252
                  ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                  : 'bg-slate-900 text-slate-500 border-slate-800'
              }`}
              title="12-Month EWMA Quantity"
            >
              <span className="w-2 h-0.5 bg-rose-400 inline-block" />
              <span>12M</span>
            </button>
          </div>

          <span className="text-slate-700 hidden sm:inline">|</span>

          {/* View Range Controls */}
          <div className="flex items-center space-x-1 bg-slate-950/70 px-1.5 py-1 rounded-lg border border-slate-800">
            <button
              onClick={() => {
                if (!chartRef.current || !timeseries) return;
                const totalPoints = timeseries.length;
                chartRef.current.timeScale().setVisibleLogicalRange({
                  from: Math.max(0, totalPoints - 140),
                  to: totalPoints + 15,
                });
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

          {/* Action Zone Toggle Button */}
          {tertipData && (
            <button
              onClick={() => setShowOutlookZone(!showOutlookZone)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono flex items-center space-x-1 border transition-colors ${
                showOutlookZone
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
              title="Toggle Institutional Forward Action Zone in whitespace"
            >
              {showOutlookZone ? (
                <Eye className="w-3 h-3 text-cyan-400" />
              ) : (
                <EyeOff className="w-3 h-3 text-slate-500" />
              )}
              <span>Action Zone</span>
            </button>
          )}
        </div>
      </div>

      {/* Point Inspector Strip */}
      {hoveredPoint && (
        <div className="px-4 py-1.5 bg-slate-950/80 border-b border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400 gap-2">
          <div className="flex items-center space-x-3">
            <span className="flex items-center space-x-1">
              <span className="text-slate-400">Date:</span>
              <span className="text-white font-semibold">{hoveredPoint.trade_date}</span>
              {timeseries && hoveredPoint.trade_date === timeseries[timeseries.length - 1]?.trade_date && (
                <span className="ml-1 px-1 py-0.2 rounded text-[9px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                  LATEST
                </span>
              )}
            </span>
            <span>Close: <span className="text-sky-300 font-semibold">₺{hoveredPoint.close_price.toFixed(2)}</span></span>
            <span>FIFO Cost: <span className="text-amber-300 font-semibold">₺{hoveredPoint.fifo_avg_cost.toFixed(2)}</span></span>
            {costHorizon !== 'none' && (
              <span>
                {activeCostCfg?.shortLabel} Cost:{' '}
                <span
                  className="font-semibold"
                  style={{ color: activeCostCfg?.color || '#a855f7' }}
                >
                  ₺{getPointCost(hoveredPoint, costHorizon).toFixed(2)}
                </span>
              </span>
            )}
            {targetCostForOpp > 0 && Math.abs(costSpreadPct) >= 3.0 && (
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-bold border tracking-wider uppercase ${
                  costSpreadPct <= -5.0
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                    : costSpreadPct <= -3.0
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                    : costSpreadPct >= 5.0
                    ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                    : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                }`}
              >
                {costSpreadPct <= -5.0
                  ? `STRONG BUY (+${potentialReturnPct.toFixed(1)}% to Cost)`
                  : costSpreadPct <= -3.0
                  ? `MODERATE BUY (+${potentialReturnPct.toFixed(1)}% to Cost)`
                  : costSpreadPct >= 5.0
                  ? `STRONG SELL (${potentialReturnPct.toFixed(1)}% vs Cost)`
                  : `MODERATE SELL (${potentialReturnPct.toFixed(1)}% vs Cost)`}
              </span>
            )}
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

      {/* Chart Canvas & Action Zone Overlay */}
      <div className="w-full relative min-h-[520px]">
        {/* Floating Institutional Forward Action Card (Positioned in 15-Bar Forward Whitespace) */}
        {showOutlookZone && outlook && (
          <div className="absolute top-4 right-14 z-20 w-[390px] glass-panel bg-slate-950/92 backdrop-blur-md border border-slate-700/80 p-3.5 rounded-xl shadow-2xl transition-all pointer-events-auto">
            {/* Card Header & View Tabs */}
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-2.5">
              <div className="flex items-center space-x-1.5">
                <Compass className="w-4 h-4 text-cyan-400" />
                <span className="font-bold text-xs text-white tracking-wide">
                  {brokerId} Action Zone (T+1)
                </span>
              </div>

              {/* View Mode Toggle: EWMA Table vs. Playbook Outlook */}
              <div className="flex items-center space-x-1 bg-slate-900 px-1 py-0.5 rounded border border-slate-800 text-[10px] font-mono">
                <button
                  onClick={() => setActionZoneTab('table')}
                  className={`flex items-center space-x-1 px-2 py-0.5 rounded transition-colors ${
                    actionZoneTab === 'table'
                      ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-500/50 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                  title="View multi-horizon EWMA inventory and cost table"
                >
                  <Table className="w-3 h-3" />
                  <span>EWMA Table</span>
                </button>
                <button
                  onClick={() => setActionZoneTab('outlook')}
                  className={`flex items-center space-x-1 px-2 py-0.5 rounded transition-colors ${
                    actionZoneTab === 'outlook'
                      ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-500/50 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                  title="View executive action narrative and institutional playbook"
                >
                  <Target className="w-3 h-3" />
                  <span>Playbook</span>
                </button>
              </div>
            </div>

            {/* TAB 1: EWMA Multi-Horizon Matrix Table */}
            {actionZoneTab === 'table' && (
              <div className="space-y-2">
                {/* Confluence & Ribbon Status Bar */}
                {confluence && (
                  <div className="flex items-center justify-between text-[10px] font-mono">
                    <span
                      className={`px-2 py-0.5 rounded font-bold border ${
                        confluence.overallDirection === 'BUY'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : confluence.overallDirection === 'SELL'
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : 'bg-slate-800 text-slate-300 border-slate-700'
                      }`}
                    >
                      {confluence.confluenceLabel}
                    </span>
                    <span className="text-cyan-300 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/40 font-semibold truncate max-w-[150px]">
                      {confluence.ribbonStatus}
                    </span>
                  </div>
                )}

                {/* Table of all Horizons */}
                {confluence && (
                  <div className="overflow-x-auto rounded-lg border border-slate-800/90">
                    <table className="w-full text-[10px] font-mono border-collapse">
                      <thead>
                        <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                          <th className="text-left px-2 py-1 font-sans">Horizon</th>
                          <th className="text-right px-1.5 py-1 font-sans">EWMA Cost</th>
                          <th className="text-right px-1.5 py-1 font-sans">Spread</th>
                          <th className="text-right px-1.5 py-1 font-sans">Target</th>
                          <th className="text-right px-2 py-1 font-sans">Stance</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                        {confluence.rows.map((row) => {
                          const isSelected = selectedHorizon === row.code;
                          return (
                            <tr
                              key={row.code}
                              onClick={() => handleSelectHorizon(row.code)}
                              className={`cursor-pointer transition-colors ${
                                isSelected
                                  ? 'bg-cyan-500/20 text-white font-semibold'
                                  : 'hover:bg-slate-800/50 text-slate-300'
                              }`}
                              title={`Click to set ${row.label} as active projection target and upper pane cost`}
                            >
                              <td className="px-2 py-1.5 flex items-center space-x-1.5">
                                <span
                                  className={`w-1.5 h-1.5 rounded-full ${
                                    isSelected ? 'bg-cyan-400 animate-pulse' : 'bg-slate-600'
                                  }`}
                                />
                                <span className={isSelected ? 'text-cyan-300 font-bold' : 'text-slate-300'}>
                                  {row.code}
                                </span>
                              </td>
                              <td className="text-right px-1.5 py-1.5 text-slate-200">
                                ₺{row.ewmaCost.toFixed(2)}
                              </td>
                              <td
                                className={`text-right px-1.5 py-1.5 font-semibold ${
                                  row.spreadPct <= -3
                                    ? 'text-cyan-300'
                                    : row.spreadPct >= 3
                                    ? 'text-rose-400'
                                    : 'text-slate-400'
                                }`}
                              >
                                {row.spreadPct >= 0 ? '+' : ''}{row.spreadPct.toFixed(1)}%
                              </td>
                              <td
                                className={`text-right px-1.5 py-1.5 font-semibold ${
                                  row.potentialReturnPct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                                }`}
                              >
                                {row.potentialReturnPct >= 0 ? '+' : ''}
                                {row.potentialReturnPct.toFixed(1)}%
                              </td>
                              <td className="text-right px-2 py-1.5">
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[9px] font-bold border ${
                                    row.direction === 'BUY'
                                      ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                                      : row.direction === 'SELL'
                                      ? 'bg-rose-500/20 text-rose-400 border-rose-500/40'
                                      : 'bg-slate-800 text-slate-400 border-slate-700'
                                  }`}
                                >
                                  {row.direction}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Active Target Projection Banner */}
                <div className="flex items-center justify-between text-[10px] font-mono bg-slate-900/80 px-2.5 py-1.5 rounded border border-slate-800/80">
                  <span className="text-slate-400">
                    Target: <span className="text-cyan-300 font-bold">{selectedHorizon}</span> (₺{outlook.targetCost.toFixed(2)})
                  </span>
                  <span
                    className={`font-semibold ${
                      outlook.potentialReturnPct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    Return: {outlook.potentialReturnPct >= 0 ? '+' : ''}{outlook.potentialReturnPct.toFixed(1)}%
                  </span>
                </div>
                <div className="text-[9px] text-slate-500 italic text-center">
                  Click any row to project that horizon's target line & upper pane cost
                </div>
              </div>
            )}

            {/* TAB 2: Executive Playbook Outlook */}
            {actionZoneTab === 'outlook' && (
              <div className="space-y-2">
                {/* Horizon Selector Pills */}
                <div className="flex items-center justify-between bg-slate-900/80 px-2 py-1 rounded border border-slate-800">
                  <span className="text-[10px] text-slate-400 font-mono">Horizon:</span>
                  <div className="flex items-center space-x-1">
                    {(['1W', '2W', '1M', '3M', '6M', 'FIFO'] as const).map((hz) => (
                      <button
                        key={hz}
                        onClick={() => handleSelectHorizon(hz)}
                        className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold transition-colors ${
                          selectedHorizon === hz
                            ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-500/50 shadow-sm'
                            : 'text-slate-400 hover:text-white'
                        }`}
                        title={`Target ${hz} Cost Horizon`}
                      >
                        {hz}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Direction & Severity Badge */}
                <div>
                  <div
                    className={`px-2.5 py-1 rounded-lg border text-xs font-bold font-mono tracking-wider flex items-center justify-between ${
                      outlook.direction === 'BUY'
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 glow-green'
                        : outlook.direction === 'SELL'
                        ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 glow-red'
                        : 'bg-slate-800 text-slate-300 border-slate-700'
                    }`}
                  >
                    <span className="flex items-center space-x-1">
                      {outlook.direction === 'BUY' ? (
                        <TrendingUp className="w-3.5 h-3.5 mr-1" />
                      ) : outlook.direction === 'SELL' ? (
                        <TrendingDown className="w-3.5 h-3.5 mr-1" />
                      ) : (
                        <Target className="w-3.5 h-3.5 mr-1 text-slate-400" />
                      )}
                      <span>{outlook.badgeLabel}</span>
                    </span>
                    <span className="text-[10px] font-mono px-1 rounded bg-slate-950/60">
                      {outlook.playbook}
                    </span>
                  </div>
                </div>

                {/* 2x2 Quantitative Metrics Grid */}
                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono bg-slate-900/60 p-2 rounded-lg border border-slate-800/80">
                  <div>
                    <div className="text-slate-400 text-[10px]">Close vs Cost</div>
                    <div className="text-slate-200 font-semibold">
                      ₺{outlook.closePrice.toFixed(2)} / ₺{outlook.targetCost.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-400 text-[10px]">Cost Spread</div>
                    <div
                      className={`font-semibold ${
                        outlook.spreadPct <= -3
                          ? 'text-cyan-400'
                          : outlook.spreadPct >= 3
                          ? 'text-rose-400'
                          : 'text-slate-300'
                      }`}
                    >
                      {outlook.spreadPct >= 0 ? '+' : ''}{outlook.spreadPct.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-400 text-[10px]">Reversion Target</div>
                    <div
                      className={`font-semibold ${
                        outlook.potentialReturnPct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {outlook.potentialReturnPct >= 0 ? '+' : ''}
                      {outlook.potentialReturnPct.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-400 text-[10px]">Active Horizon</div>
                    <div className="text-cyan-300 font-semibold">
                      {outlook.horizonCode} ({outlook.horizonLabel.split(' ')[0]})
                    </div>
                  </div>
                </div>

                {/* Microstructure Rationale */}
                <div className="text-[10px] text-slate-400 leading-tight border-t border-slate-800/60 pt-1.5">
                  {outlook.rationale}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Chart Canvas */}
        <div ref={chartContainerRef} className="w-full min-h-[520px]" />
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
        <span>Upper Pane: Price & Selectable EWMA Costs (₺/sh) • Lower Pane: BofA Open Inventory & EWMA Ribbons (Shares)</span>
        <span>Scroll to Zoom • Drag to Pan</span>
      </div>
    </div>
  );
};
