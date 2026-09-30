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
import { fetchTertipTimeseries, fetchTertipHorizons, fetchShockDays, fetchTertipMlForecast } from '../api/client';
import {
  TertipTimeseriesPoint,
  TertipHorizonsResponse,
  ForwardHorizonCode,
  ShockDayItem,
  TertipMlForecastResponse,
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
  Zap,
  BarChart3,
  History,
  Award,
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
  const [showShockDays, setShowShockDays] = useState<boolean>(true);
  const [shockTypeFilter, setShockTypeFilter] = useState<'ALL' | 'POSITIVE' | 'NEGATIVE'>('ALL');
  const [selectedShockDate, setSelectedShockDate] = useState<string | null>(null);

  // Overlay Canvas Ref for BIST 30 Shock Day Dashed Lines
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Action Zone State
  const [showOutlookZone, setShowOutlookZone] = useState<boolean>(true);
  const [actionZoneTab, setActionZoneTab] = useState<'table' | 'outlook' | 'realization' | 'forecast' | 'track30d'>('forecast');
  const [selectedHorizon, setSelectedHorizon] = useState<ForwardHorizonCode>('3M');

  // Hover inspector state
  const [hoveredPoint, setHoveredPoint] = useState<TertipTimeseriesPoint | null>(null);
  const timeseriesRef = useRef<TertipTimeseriesPoint[]>([]);
  const timeMapRef = useRef<Map<string | number, TertipTimeseriesPoint>>(new Map());

  // Fetch Tertip 3-Pillar ML Forecast & 30-Day Walk-Forward Track
  const { data: mlForecast } = useQuery<TertipMlForecastResponse>({
    queryKey: ['tertipMlForecast', symbol],
    queryFn: () => fetchTertipMlForecast(symbol),
    staleTime: 1000 * 60 * 5,
  });

  // Fetch Time Series Data (all available history up to latest date)
  const { data: timeseries, isLoading, error } = useQuery({
    queryKey: ['tertipTimeseries', symbol, brokerId],
    queryFn: () => fetchTertipTimeseries(symbol, brokerId, 1500),
  });

  // Fetch BIST 30 Shock Days (>= 3% drops and jumps)
  const { data: shockDays } = useQuery({
    queryKey: ['bist30ShockDays'],
    queryFn: () => fetchShockDays(0.03),
    staleTime: 1000 * 60 * 30, // 30 minutes
  });

  // Filtered shock days based on selectable filter (All / Positive only / Negative only)
  const filteredShockDays = React.useMemo(() => {
    if (!shockDays) return [];
    if (shockTypeFilter === 'POSITIVE') {
      return shockDays.filter((s) => s.is_positive_shock);
    }
    if (shockTypeFilter === 'NEGATIVE') {
      return shockDays.filter((s) => s.is_negative_shock);
    }
    return shockDays;
  }, [shockDays, shockTypeFilter]);

  const positiveShocksCount = React.useMemo(
    () => (shockDays ? shockDays.filter((s) => s.is_positive_shock).length : 0),
    [shockDays]
  );
  const negativeShocksCount = React.useMemo(
    () => (shockDays ? shockDays.filter((s) => s.is_negative_shock).length : 0),
    [shockDays]
  );

  // Handle selecting a specific shock day to focus chart
  const handleSelectShockDate = (dateStr: string) => {
    setSelectedShockDate(dateStr || null);
    if (!dateStr || !chartRef.current || !timeseries) return;
    const idx = timeseries.findIndex((p) => p.trade_date === dateStr);
    if (idx >= 0) {
      setHoveredPoint(timeseries[idx]);
      chartRef.current.timeScale().setVisibleLogicalRange({
        from: Math.max(0, idx - 15),
        to: idx + 35,
      });
    }
  };

  // O(1) Shock Map indexed by trade_date
  const shockMap = React.useMemo(() => {
    const map = new Map<string, ShockDayItem>();
    if (shockDays) {
      for (const s of shockDays) {
        map.set(s.trade_date, s);
      }
    }
    return map;
  }, [shockDays]);

  const hoveredShock = hoveredPoint ? shockMap.get(hoveredPoint.trade_date) : null;

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

  // Render BIST 30 Shock Day Vertical Dashes on Overlay Canvas
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const chart = chartRef.current;
    if (!canvas || !chart) return;

    const redrawOverlay = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const width = parent.clientWidth;
      const height = parent.clientHeight;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      if (!showShockDays || !filteredShockDays || filteredShockDays.length === 0) return;

      const timeScale = chart.timeScale();

      for (const shock of filteredShockDays) {
        const isSelected = selectedShockDate === shock.trade_date;
        const x = timeScale.timeToCoordinate(shock.time as any);
        if (x === null || x < 0 || x > width - 55) continue; // leave margin for price scale axis

        ctx.save();
        if (isSelected) {
          // Highlight ambient background glow for selected shock day
          ctx.fillStyle = shock.is_positive_shock ? 'rgba(16, 185, 129, 0.16)' : 'rgba(244, 63, 94, 0.16)';
          ctx.fillRect(x - 14, 0, 28, height);
        }

        ctx.beginPath();
        ctx.setLineDash(isSelected ? [] : [4, 4]); // Solid if selected
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        if (shock.is_positive_shock) {
          ctx.strokeStyle = isSelected ? '#34d399' : 'rgba(16, 185, 129, 0.75)'; // Emerald green
        } else {
          ctx.strokeStyle = isSelected ? '#fb7171' : 'rgba(244, 63, 94, 0.75)'; // Rose red
        }
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Header pill badge
        const retSign = shock.daily_return_pct >= 0 ? '+' : '';
        const text = `BIST30 ${retSign}${(shock.daily_return_pct * 100).toFixed(1)}%`;
        ctx.font = isSelected ? 'bold 10px monospace' : 'bold 9px monospace';
        const textWidth = ctx.measureText(text).width;
        const pillWidth = textWidth + (isSelected ? 10 : 8);
        const pillHeight = isSelected ? 17 : 15;
        const pillY = 6;
        const pillX = x - pillWidth / 2;

        ctx.fillStyle = shock.is_positive_shock
          ? isSelected ? 'rgba(6, 78, 59, 1.0)' : 'rgba(6, 78, 59, 0.92)'
          : isSelected ? 'rgba(136, 19, 55, 1.0)' : 'rgba(136, 19, 55, 0.92)';
        ctx.strokeStyle = shock.is_positive_shock
          ? isSelected ? '#34d399' : 'rgba(52, 211, 153, 0.9)'
          : isSelected ? '#fb7171' : 'rgba(251, 113, 133, 0.9)';
        ctx.lineWidth = isSelected ? 1.5 : 1;
        ctx.setLineDash([]);
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(pillX, pillY, pillWidth, pillHeight, 3);
        } else {
          ctx.rect(pillX, pillY, pillWidth, pillHeight);
        }
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = shock.is_positive_shock ? '#a7f3d0' : '#fecdd3';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x, pillY + pillHeight / 2);
        ctx.restore();
      }
    };

    redrawOverlay();

    const timeScale = chart.timeScale();
    timeScale.subscribeVisibleLogicalRangeChange(redrawOverlay);
    window.addEventListener('resize', redrawOverlay);

    return () => {
      timeScale.unsubscribeVisibleLogicalRangeChange(redrawOverlay);
      window.removeEventListener('resize', redrawOverlay);
    };
  }, [showShockDays, filteredShockDays, selectedShockDate, timeseries]);

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

          {/* BIST 30 Shock Days Controls */}
          <div className="flex items-center space-x-1.5 bg-slate-950/70 px-2 py-1 rounded-lg border border-slate-800">
            {/* Eye / EyeOff Toggle matching Action Zone */}
            <button
              onClick={() => setShowShockDays(!showShockDays)}
              className={`flex items-center space-x-1.5 px-2 py-0.5 rounded text-[10px] font-mono border transition-colors ${
                showShockDays
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-sm'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
              title="Toggle BIST 30 Shock Days (±3%) vertical markers"
            >
              {showShockDays ? (
                <Eye className="w-3 h-3 text-amber-400" />
              ) : (
                <EyeOff className="w-3 h-3 text-slate-500" />
              )}
              <Zap className={`w-3 h-3 ${showShockDays ? 'text-amber-400 fill-amber-400/30' : 'text-slate-500'}`} />
              <span>Shock Days</span>
              {shockDays && (
                <span className="text-[9px] px-1 py-0.2 rounded bg-slate-900 border border-slate-700/60 font-mono text-amber-300">
                  {shockDays.length}
                </span>
              )}
            </button>

            {/* Selectable Direction Filters & Session Jump Dropdown */}
            {showShockDays && (
              <>
                <div className="flex items-center space-x-1 border-l border-slate-800 pl-1.5 text-[9px] font-mono">
                  <button
                    onClick={() => setShockTypeFilter('ALL')}
                    className={`px-1.5 py-0.5 rounded transition-colors ${
                      shockTypeFilter === 'ALL'
                        ? 'bg-amber-500/30 text-amber-200 font-bold border border-amber-500/50'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    }`}
                    title="Show all BIST 30 shock days"
                  >
                    All ({shockDays?.length || 0})
                  </button>
                  <button
                    onClick={() => setShockTypeFilter('POSITIVE')}
                    className={`px-1.5 py-0.5 rounded transition-colors ${
                      shockTypeFilter === 'POSITIVE'
                        ? 'bg-emerald-500/30 text-emerald-200 font-bold border border-emerald-500/50'
                        : 'text-emerald-400/70 hover:text-emerald-300 hover:bg-slate-800'
                    }`}
                    title="Filter to Positive Shock Days (≥ +3.0%)"
                  >
                    ≥+3% ({positiveShocksCount})
                  </button>
                  <button
                    onClick={() => setShockTypeFilter('NEGATIVE')}
                    className={`px-1.5 py-0.5 rounded transition-colors ${
                      shockTypeFilter === 'NEGATIVE'
                        ? 'bg-rose-500/30 text-rose-200 font-bold border border-rose-500/50'
                        : 'text-rose-400/70 hover:text-rose-300 hover:bg-slate-800'
                    }`}
                    title="Filter to Negative Shock Days (≤ -3.0%)"
                  >
                    ≤-3% ({negativeShocksCount})
                  </button>
                </div>

                <div className="flex items-center space-x-1 border-l border-slate-800 pl-1.5">
                  <select
                    value={selectedShockDate || ''}
                    onChange={(e) => handleSelectShockDate(e.target.value)}
                    className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-1.5 py-0.5 text-[10px] font-mono focus:outline-none focus:border-amber-400 max-w-[170px] truncate"
                    title="Select a shock day to center chart and inspect trader follow-through"
                  >
                    <option value="">Jump to Shock ({filteredShockDays.length})...</option>
                    {filteredShockDays.map((s) => (
                      <option key={s.trade_date} value={s.trade_date}>
                        {s.trade_date} : BIST 30 {s.daily_return_pct >= 0 ? '+' : ''}
                        {(s.daily_return_pct * 100).toFixed(1)}% {s.is_positive_shock ? '▲' : '▼'}{' '}
                        {s.total_turnover_tl > 0 ? `(₺${(s.total_turnover_tl / 1e9).toFixed(0)}B)` : ''}
                      </option>
                    ))}
                  </select>

                  {selectedShockDate && (
                    <button
                      onClick={() => handleSelectShockDate('')}
                      className="px-1.5 py-0.5 text-[9px] font-mono text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded border border-slate-600 transition-colors"
                      title="Clear selected shock date"
                    >
                      ✕
                    </button>
                  )}
                </div>
              </>
            )}
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
              {hoveredShock && (
                <span
                  className={`ml-1.5 px-1.5 py-0.5 rounded text-[10px] font-bold border tracking-wider font-mono ${
                    hoveredShock.is_positive_shock
                      ? 'bg-emerald-500/25 text-emerald-300 border-emerald-500/50 shadow-sm'
                      : 'bg-rose-500/25 text-rose-300 border-rose-500/50 shadow-sm'
                  }`}
                  title={`Entire BIST 30 Basket Value Move: ${(hoveredShock.daily_return_pct * 100).toFixed(2)}% | Total Trading Value: ₺${(hoveredShock.total_turnover_tl / 1e9).toFixed(2)}B | BofA Flow: ₺${(hoveredShock.bofa_net_flow_tl / 1e6).toFixed(1)}M`}
                >
                  ⚡ BIST 30 Basket: {hoveredShock.daily_return_pct >= 0 ? '+' : ''}${(hoveredShock.daily_return_pct * 100).toFixed(2)}% ({hoveredShock.is_positive_shock ? 'POSITIVE' : 'NEGATIVE'} SHOCK){hoveredShock.total_turnover_tl > 0 ? ` | Turnover: ₺${(hoveredShock.total_turnover_tl / 1e9).toFixed(1)}B` : ''}{hoveredShock.bofa_net_flow_tl !== 0 ? ` | BofA: ${hoveredShock.bofa_net_flow_tl >= 0 ? '+' : ''}₺${(hoveredShock.bofa_net_flow_tl / 1e6).toFixed(0)}M` : ''}
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
          <div className="absolute top-4 right-14 z-20 w-[440px] glass-panel bg-slate-950/92 backdrop-blur-md border border-slate-700/80 p-3.5 rounded-xl shadow-2xl transition-all pointer-events-auto">
            {/* Card Header & View Tabs */}
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-2.5">
              <div className="flex items-center space-x-1.5">
                <Compass className="w-4 h-4 text-cyan-400" />
                <span className="font-bold text-xs text-white tracking-wide">
                  {brokerId} Action Zone (T+1)
                </span>
              </div>

              {/* View Mode Toggle: EWMA Table vs. Playbook Outlook vs. 12M Realizations */}
              <div className="flex items-center space-x-1 bg-slate-900 px-1 py-0.5 rounded border border-slate-800 text-[10px] font-mono">
                <button
                  onClick={() => setActionZoneTab('table')}
                  className={`flex items-center space-x-1 px-1.5 py-0.5 rounded transition-colors ${
                    actionZoneTab === 'table'
                      ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-500/50 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                  title="View multi-horizon EWMA inventory and cost table"
                >
                  <Table className="w-3 h-3" />
                  <span>EWMA</span>
                </button>
                <button
                  onClick={() => setActionZoneTab('outlook')}
                  className={`flex items-center space-x-1 px-1.5 py-0.5 rounded transition-colors ${
                    actionZoneTab === 'outlook'
                      ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-500/50 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                  title="View executive action narrative and institutional playbook"
                >
                  <Target className="w-3 h-3" />
                  <span>Playbook</span>
                </button>
                <button
                  onClick={() => setActionZoneTab('realization')}
                  className={`flex items-center space-x-1 px-1.5 py-0.5 rounded transition-colors ${
                    actionZoneTab === 'realization'
                      ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-500/50 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                  title="View 12-month historical execution realizations and follow-through track record"
                >
                  <BarChart3 className="w-3 h-3" />
                  <span>12M Track</span>
                </button>
                <button
                  onClick={() => setActionZoneTab('forecast')}
                  className={`flex items-center space-x-1 px-1.5 py-0.5 rounded transition-colors ${
                    actionZoneTab === 'forecast'
                      ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-500/50 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                  title="View live tomorrow (T+1) forecast and 3-Pillar institutional matrix"
                >
                  <Zap className="w-3 h-3 text-amber-400" />
                  <span>Forecast</span>
                </button>
                <button
                  onClick={() => setActionZoneTab('track30d')}
                  className={`flex items-center space-x-1 px-1.5 py-0.5 rounded transition-colors ${
                    actionZoneTab === 'track30d'
                      ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-500/50 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                  title="View 30-day zero-lookahead walk-forward reality ledger and realized actions"
                >
                  <Award className="w-3 h-3 text-cyan-400" />
                  <span>30D Track</span>
                </button>
              </div>
            </div>

            {/* TAB 1: EWMA Multi-Horizon Matrix Table */}
            {actionZoneTab === 'table' && (
              <div className="space-y-2">
                {/* Confluence & Ribbon Status Bar with 12M Realization Summary */}
                {confluence && (
                  <div className="space-y-1.5">
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

                    {confluence.confluenceRealization && confluence.confluenceRealization.total_occurrences > 0 && (
                      <div className="flex items-center justify-between text-[9.5px] font-mono bg-slate-900/90 px-2 py-1 rounded border border-slate-800">
                        <span className="text-slate-400 flex items-center space-x-1">
                          <History className="w-3 h-3 text-cyan-400" />
                          <span>12M Realization ({confluence.confluenceRealization.total_occurrences}d):</span>
                        </span>
                        <span className="font-semibold text-white space-x-1.5">
                          <span className="text-emerald-400" title="Sessions broker followed through next day in indicated direction">
                            {confluence.confluenceRealization.realized_pct.toFixed(0)}% Realized ({confluence.confluenceRealization.realized_count}d)
                          </span>
                          <span className="text-slate-600">|</span>
                          <span className="text-rose-400" title="Sessions broker executed in exact opposite direction">
                            {confluence.confluenceRealization.opposite_pct.toFixed(0)}% Opp ({confluence.confluenceRealization.opposite_count}d)
                          </span>
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Table of all Horizons with 12M Realization Columns */}
                {confluence && (
                  <div className="overflow-x-auto rounded-lg border border-slate-800/90">
                    <table className="w-full text-[9.5px] font-mono border-collapse">
                      <thead>
                        <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                          <th className="text-left px-2 py-1 font-sans">Horizon</th>
                          <th className="text-right px-1.5 py-1 font-sans">Cost</th>
                          <th className="text-right px-1.5 py-1 font-sans">Spread</th>
                          <th className="text-right px-1.5 py-1 font-sans">Target</th>
                          <th className="text-center px-1.5 py-1 font-sans">Stance</th>
                          <th className="text-right px-1.5 py-1 font-sans text-emerald-400" title="Next-day execution in indicated direction (Follow-Through)">Realized</th>
                          <th className="text-right px-2 py-1 font-sans text-rose-400" title="Next-day execution in exact opposite direction (Adverse Fade)">Opposite</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                        {confluence.rows.map((row) => {
                          const isSelected = selectedHorizon === row.code;
                          const r = row.realization_12m;
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
                              <td className="text-center px-1.5 py-1.5">
                                <span
                                  className={`px-1 py-0.2 rounded text-[8.5px] font-bold border ${
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
                              <td className="text-right px-1.5 py-1.5 font-semibold text-emerald-400">
                                {r && r.total_occurrences > 0 ? (
                                  <span title={`Follow-through: ${r.realized_count} of ${r.total_occurrences} sessions (${r.total_occurrences}d total)`}>
                                    {r.realized_pct.toFixed(0)}%
                                  </span>
                                ) : (
                                  '—'
                                )}
                              </td>
                              <td className="text-right px-2 py-1.5 font-semibold text-rose-400">
                                {r && r.total_occurrences > 0 ? (
                                  <span title={`Opposite fade: ${r.opposite_count} of ${r.total_occurrences} sessions (${r.total_occurrences}d total)`}>
                                    {r.opposite_pct.toFixed(0)}%
                                  </span>
                                ) : (
                                  '—'
                                )}
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

            {/* TAB 3: 12-Month Historical Realizations & Track Record */}
            {actionZoneTab === 'realization' && confluence && (
              <div className="space-y-2.5">
                {/* Confluence 12M Follow-Through Header Card */}
                <div className="bg-slate-900/90 rounded-lg p-2.5 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-1.5">
                      <History className="w-3.5 h-3.5 text-cyan-400" />
                      <span className="text-[11px] font-bold text-white tracking-wide">
                        12-Month Confluence Realization
                      </span>
                    </div>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-bold border ${
                        confluence.overallDirection === 'BUY'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : confluence.overallDirection === 'SELL'
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : 'bg-slate-800 text-slate-300 border-slate-700'
                      }`}
                    >
                      {confluence.overallDirection} STANCE
                    </span>
                  </div>

                  {confluence.confluenceRealization && confluence.confluenceRealization.total_occurrences > 0 ? (
                    <>
                      {/* Key Metrics 4-Grid */}
                      <div className="grid grid-cols-2 gap-2 text-mono text-[10px]">
                        <div className="bg-slate-950/70 p-2 rounded border border-slate-800/80">
                          <div className="text-slate-400 text-[9px]">12M Occurrences</div>
                          <div className="text-sm font-bold text-white mt-0.5">
                            {confluence.confluenceRealization.total_occurrences} <span className="text-[10px] text-slate-400 font-normal">sessions</span>
                          </div>
                          <div className="text-[8.5px] text-slate-500 mt-0.5">Prior ~252 trading days</div>
                        </div>

                        <div className="bg-slate-950/70 p-2 rounded border border-slate-800/80">
                          <div className="text-slate-400 text-[9px]">Realized Follow-Through</div>
                          <div className="text-sm font-bold text-emerald-400 mt-0.5">
                            {confluence.confluenceRealization.realized_pct.toFixed(1)}%
                          </div>
                          <div className="text-[8.5px] text-emerald-500/80 mt-0.5">
                            {confluence.confluenceRealization.realized_count} of {confluence.confluenceRealization.total_occurrences} next-day follow-through
                          </div>
                        </div>

                        <div className="bg-slate-950/70 p-2 rounded border border-slate-800/80">
                          <div className="text-slate-400 text-[9px]">Adverse Opposite (Fade)</div>
                          <div className="text-sm font-bold text-rose-400 mt-0.5">
                            {confluence.confluenceRealization.opposite_pct.toFixed(1)}%
                          </div>
                          <div className="text-[8.5px] text-rose-500/80 mt-0.5">
                            {confluence.confluenceRealization.opposite_count} of {confluence.confluenceRealization.total_occurrences} executed exact opposite
                          </div>
                        </div>

                        <div className="bg-slate-950/70 p-2 rounded border border-slate-800/80">
                          <div className="text-slate-400 text-[9px]">Avg Next-Day Flow</div>
                          <div className={`text-sm font-bold mt-0.5 ${
                            confluence.confluenceRealization.avg_next_day_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }`}>
                            {confluence.confluenceRealization.avg_next_day_flow_tl >= 0 ? '+' : ''}
                            ₺{(confluence.confluenceRealization.avg_next_day_flow_tl / 1e6).toFixed(1)}M
                          </div>
                          <div className="text-[8.5px] text-slate-500 mt-0.5">
                            Stock Price Up: {confluence.confluenceRealization.price_up_pct.toFixed(0)}%
                          </div>
                        </div>
                      </div>

                      {/* Visual Follow-Through Ratio Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-[9px] font-mono">
                          <span className="text-emerald-400 font-semibold">
                            Follow-Through: {confluence.confluenceRealization.realized_pct.toFixed(1)}%
                          </span>
                          <span className="text-rose-400 font-semibold">
                            Opposite: {confluence.confluenceRealization.opposite_pct.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden flex">
                          <div
                            className="bg-emerald-500 h-full transition-all duration-300"
                            style={{ width: `${confluence.confluenceRealization.realized_pct}%` }}
                          />
                          <div
                            className="bg-rose-500 h-full transition-all duration-300"
                            style={{ width: `${confluence.confluenceRealization.opposite_pct}%` }}
                          />
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-[10px] text-slate-400 py-2 text-center italic">
                      No historical occurrences matching this exact confluence in the past 12 months.
                    </div>
                  )}
                </div>

                {/* Horizon Breakdown Table */}
                <div className="space-y-1">
                  <div className="text-[10px] font-bold text-slate-300 tracking-wide flex items-center justify-between">
                    <span>Horizon Realization Matrix</span>
                    <span className="text-[9px] text-slate-500 font-mono">Prior ~252 Sessions</span>
                  </div>
                  <div className="overflow-x-auto rounded-lg border border-slate-800/90">
                    <table className="w-full text-[9.5px] font-mono border-collapse">
                      <thead>
                        <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                          <th className="text-left px-2 py-1 font-sans">Horizon</th>
                          <th className="text-center px-1.5 py-1 font-sans">Stance</th>
                          <th className="text-right px-1.5 py-1 font-sans">12M Days</th>
                          <th className="text-right px-1.5 py-1 font-sans text-emerald-400">Realized</th>
                          <th className="text-right px-1.5 py-1 font-sans text-rose-400">Opposite</th>
                          <th className="text-right px-2 py-1 font-sans">Avg Flow</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                        {confluence.rows.map((row) => {
                          const r = row.realization_12m;
                          return (
                            <tr key={row.code} className="hover:bg-slate-800/40 text-slate-300">
                              <td className="px-2 py-1.5 font-bold text-white">{row.code}</td>
                              <td className="text-center px-1.5 py-1.5">
                                <span
                                  className={`px-1 py-0.2 rounded text-[8.5px] font-bold border ${
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
                              <td className="text-right px-1.5 py-1.5 text-slate-300">
                                {r ? `${r.total_occurrences}d` : '—'}
                              </td>
                              <td className="text-right px-1.5 py-1.5 font-semibold text-emerald-400">
                                {r && r.total_occurrences > 0 ? (
                                  <span title={`${r.realized_count}/${r.total_occurrences} sessions`}>
                                    {r.realized_pct.toFixed(0)}%
                                  </span>
                                ) : (
                                  '—'
                                )}
                              </td>
                              <td className="text-right px-1.5 py-1.5 font-semibold text-rose-400">
                                {r && r.total_occurrences > 0 ? (
                                  <span title={`${r.opposite_count}/${r.total_occurrences} sessions`}>
                                    {r.opposite_pct.toFixed(0)}%
                                  </span>
                                ) : (
                                  '—'
                                )}
                              </td>
                              <td className={`text-right px-2 py-1.5 ${
                                (r?.avg_next_day_flow_tl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                              }`}>
                                {r && r.total_occurrences > 0 ? (
                                  `${((r.avg_next_day_flow_tl) / 1e6).toFixed(0)}M`
                                ) : (
                                  '—'
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="text-[8.5px] text-slate-500 italic text-center pt-0.5">
                    Realized = broker executed in expected direction on T+1 | Opposite = executed adverse flow
                  </div>
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

            {/* TAB 4: Live Tomorrow (T+1) ML Forecaster & 3-Pillar Synthesis */}
            {actionZoneTab === 'forecast' && (
              <div className="space-y-2.5">
                {mlForecast ? (
                  <>
                    {/* Champion & Stance Header Banner */}
                    <div className="flex flex-wrap items-center justify-between gap-1 bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                      <div className="flex items-center space-x-1.5">
                        <span className="flex items-center space-x-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          <Award className="w-3 h-3 text-amber-400" />
                          <span>CHAMPION: {mlForecast.tournament_summary.champion === 'TERTIP_ML_CHALLENGER' ? 'TERTIP ML' : 'PROPHET BASE'}</span>
                        </span>
                        <span className="text-[9px] text-slate-400 font-mono">
                          ({mlForecast.tournament_summary.ml_wins}/{mlForecast.tournament_summary.total_sessions}d won)
                        </span>
                      </div>

                      <span
                        className={`px-2 py-0.5 rounded text-[9.5px] font-bold border tracking-wider font-mono ${
                          mlForecast.stance.includes('BUY')
                            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                            : mlForecast.stance.includes('SELL')
                            ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                            : 'bg-slate-800 text-slate-300 border-slate-700'
                        }`}
                      >
                        {mlForecast.stance_badge}
                      </span>
                    </div>

                    {/* Price Projection 3-Card Summary */}
                    <div className="grid grid-cols-3 gap-2 text-center font-mono">
                      <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800">
                        <div className="text-[9px] text-slate-400">ML Target (T+1)</div>
                        <div className="text-sm font-bold text-white mt-0.5">
                          ₺{mlForecast.target_price.toFixed(2)}
                        </div>
                        <div className={`text-[9px] font-semibold ${
                          mlForecast.expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}>
                          {mlForecast.expected_return_pct >= 0 ? '+' : ''}{mlForecast.expected_return_pct.toFixed(2)}%
                        </div>
                      </div>

                      <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800">
                        <div className="text-[9px] text-slate-400">90% Range Envelope</div>
                        <div className="text-[11px] font-semibold text-slate-200 mt-1">
                          ₺{mlForecast.price_low.toFixed(1)} – ₺{mlForecast.price_high.toFixed(1)}
                        </div>
                        <div className="text-[8.5px] text-slate-500 mt-0.5">
                          As of {mlForecast.as_of_date}
                        </div>
                      </div>

                      <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800">
                        <div className="text-[9px] text-slate-400">Prophet Baseline</div>
                        <div className="text-[11.5px] font-bold text-slate-300 mt-0.5">
                          ₺{mlForecast.prophet_target_price.toFixed(2)}
                        </div>
                        <div className={`text-[9px] ${
                          mlForecast.prophet_expected_return_pct >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'
                        }`}>
                          {mlForecast.prophet_expected_return_pct >= 0 ? '+' : ''}{mlForecast.prophet_expected_return_pct.toFixed(2)}%
                        </div>
                      </div>
                    </div>

                    {/* 3-Pillar Institutional Balance Matrix */}
                    <div className="space-y-1">
                      <div className="text-[10px] font-bold text-slate-300 tracking-wide flex items-center justify-between">
                        <span className="flex items-center space-x-1">
                          <Zap className="w-3 h-3 text-cyan-400" />
                          <span>3-Pillar Institutional Market Forces</span>
                        </span>
                        <span className="text-[9px] text-slate-500 font-mono">MLB + BIG5 + KAMU</span>
                      </div>

                      <div className="overflow-x-auto rounded-lg border border-slate-800/90">
                        <table className="w-full text-[9.5px] font-mono border-collapse">
                          <thead>
                            <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                              <th className="text-left px-2 py-1 font-sans">Pillar</th>
                              <th className="text-center px-1.5 py-1 font-sans">Potential</th>
                              <th className="text-right px-1.5 py-1 font-sans text-emerald-400" title="Historical follow-through rate on T+1">Realized</th>
                              <th className="text-right px-1.5 py-1 font-sans text-rose-400" title="Historical opposite adverse execution rate on T+1">Opposite</th>
                              <th className="text-left px-2 py-1 font-sans">Expected Action</th>
                              <th className="text-right px-2 py-1 font-sans" title="Market share of total stock turnover">Volume %</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                            {mlForecast.pillar_matrix.map((p) => (
                              <tr key={p.pillar} className="hover:bg-slate-800/40 text-slate-300">
                                <td className="px-2 py-1.5 font-bold text-white">
                                  <div>{p.pillar}</div>
                                  <div className="text-[8px] text-slate-500 font-sans font-normal truncate max-w-[80px]">
                                    {p.desc}
                                  </div>
                                </td>
                                <td className="text-center px-1.5 py-1.5">
                                  <span
                                    className={`px-1 py-0.2 rounded text-[8.5px] font-bold border ${
                                      p.stance === 'BUY'
                                        ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                                        : p.stance === 'SELL'
                                        ? 'bg-rose-500/20 text-rose-400 border-rose-500/40'
                                        : 'bg-slate-800 text-slate-400 border-slate-700'
                                    }`}
                                  >
                                    {p.stance}
                                  </span>
                                </td>
                                <td className="text-right px-1.5 py-1.5 text-emerald-400 font-semibold">
                                  {p.realized_pct.toFixed(0)}%
                                </td>
                                <td className="text-right px-1.5 py-1.5 text-rose-400 font-semibold">
                                  {p.opposite_pct.toFixed(0)}%
                                </td>
                                <td className="px-2 py-1.5 font-semibold text-slate-200">
                                  <div>{p.expected_action}</div>
                                  <div className={`text-[8.5px] ${
                                    p.expected_flow_tl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                                  }`}>
                                    {p.expected_flow_tl >= 0 ? '+' : ''}₺{(p.expected_flow_tl / 1e6).toFixed(0)}M
                                  </div>
                                </td>
                                <td className="text-right px-2 py-1.5 text-cyan-300 font-semibold">
                                  {p.turnover_share_pct.toFixed(1)}%
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Playbook Rationale */}
                    <div className="text-[10px] text-slate-400 leading-tight bg-slate-900/60 p-2 rounded border border-slate-800/80">
                      <span className="font-semibold text-slate-300 font-sans">Tactical Assessment: </span>
                      {mlForecast.playbook_rationale}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-6 text-slate-400 text-xs font-mono">
                    Loading 3-Pillar Machine Learning forecast...
                  </div>
                )}
              </div>
            )}

            {/* TAB 5: 30-Day Walk-Forward Reality Ledger & Realized Pillar Actions */}
            {actionZoneTab === 'track30d' && (
              <div className="space-y-2.5">
                {mlForecast ? (
                  <>
                    {/* 30-Day Scorecard */}
                    <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between text-[10px] font-bold text-slate-300">
                        <span>30-Day Zero-Lookahead Arena</span>
                        <span className="text-cyan-400 font-mono">Prior 30 Trading Sessions</span>
                      </div>

                      <div className="grid grid-cols-4 gap-1.5 text-center font-mono">
                        <div className="bg-slate-950/70 p-1.5 rounded border border-slate-800/80">
                          <div className="text-[8.5px] text-slate-400">ML Hit Rate</div>
                          <div className="text-xs font-bold text-emerald-400 mt-0.5">
                            {mlForecast.tournament_summary.ml_hit_rate_pct.toFixed(1)}%
                          </div>
                          <div className="text-[8px] text-slate-500">
                            {mlForecast.tournament_summary.ml_wins} days won
                          </div>
                        </div>

                        <div className="bg-slate-950/70 p-1.5 rounded border border-slate-800/80">
                          <div className="text-[8.5px] text-slate-400">Prophet Hit Rate</div>
                          <div className="text-xs font-bold text-slate-300 mt-0.5">
                            {mlForecast.tournament_summary.prophet_hit_rate_pct.toFixed(1)}%
                          </div>
                          <div className="text-[8px] text-slate-500">
                            {mlForecast.tournament_summary.prophet_wins} days won
                          </div>
                        </div>

                        <div className="bg-slate-950/70 p-1.5 rounded border border-slate-800/80">
                          <div className="text-[8.5px] text-slate-400">ML MAE</div>
                          <div className="text-xs font-bold text-emerald-400 mt-0.5">
                            {mlForecast.tournament_summary.ml_mae_pct.toFixed(2)}%
                          </div>
                          <div className="text-[8px] text-emerald-500/80">Lower Error</div>
                        </div>

                        <div className="bg-slate-950/70 p-1.5 rounded border border-slate-800/80">
                          <div className="text-[8.5px] text-slate-400">Prophet MAE</div>
                          <div className="text-xs font-bold text-rose-400 mt-0.5">
                            {mlForecast.tournament_summary.prophet_mae_pct.toFixed(2)}%
                          </div>
                          <div className="text-[8px] text-rose-500/80">Higher Error</div>
                        </div>
                      </div>

                      {/* Win Ratio Visual Bar */}
                      <div className="space-y-0.5">
                        <div className="flex justify-between text-[8.5px] font-mono">
                          <span className="text-emerald-400 font-semibold">
                            ML Wins: {mlForecast.tournament_summary.ml_wins}d
                          </span>
                          <span className="text-cyan-400 font-semibold">
                            Prophet Wins: {mlForecast.tournament_summary.prophet_wins}d
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden flex">
                          <div
                            className="bg-emerald-500 h-full transition-all duration-300"
                            style={{
                              width: `${(mlForecast.tournament_summary.ml_wins / mlForecast.tournament_summary.total_sessions) * 100}%`,
                            }}
                          />
                          <div
                            className="bg-cyan-500 h-full transition-all duration-300"
                            style={{
                              width: `${(mlForecast.tournament_summary.prophet_wins / mlForecast.tournament_summary.total_sessions) * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* 30-Day Walk-Forward Reality Ledger Table */}
                    <div className="space-y-1">
                      <div className="text-[10px] font-bold text-slate-300 tracking-wide flex items-center justify-between">
                        <span>Out-of-Sample Walk-Forward Reality Ledger</span>
                        <span className="text-[8.5px] text-slate-500 font-mono">Ground-Truth Pillar Actions</span>
                      </div>

                      <div className="overflow-x-auto max-h-56 overflow-y-auto rounded-lg border border-slate-800/90">
                        <table className="w-full text-[9px] font-mono border-collapse">
                          <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 z-10">
                            <tr className="text-slate-400">
                              <th className="text-left px-2 py-1 font-sans">Date</th>
                              <th className="text-right px-1.5 py-1 font-sans">Actual</th>
                              <th className="text-right px-1.5 py-1 font-sans text-cyan-300">ML Pred</th>
                              <th className="text-center px-1.5 py-1 font-sans text-amber-300">Dir Correct?</th>
                              <th className="text-right px-1.5 py-1 font-sans">Prophet</th>
                              <th className="text-center px-1.5 py-1 font-sans">MLB Did</th>
                              <th className="text-center px-1.5 py-1 font-sans">BIG5 Did</th>
                              <th className="text-center px-1.5 py-1 font-sans">KAMU Did</th>
                              <th className="text-center px-1.5 py-1 font-sans">Winner</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                            {mlForecast.walk_forward_ledger.map((row) => {
                              const mlRet = row.ml_pred_return_pct !== undefined 
                                ? row.ml_pred_return_pct 
                                : ((row.ml_pred_price - (row.actual_price / (1 + (row.actual_return_pct || 0) / 100))) / (row.actual_price / (1 + (row.actual_return_pct || 0) / 100)) * 100);
                              const isMlUp = row.ml_direction ? row.ml_direction === 'UP' : mlRet > 0.02;
                              const isMlDown = row.ml_direction ? row.ml_direction === 'DOWN' : mlRet < -0.02;
                              const isActUp = row.actual_return_pct > 0.02;
                              const isActDown = row.actual_return_pct < -0.02;
                              const pRet = row.prophet_pred_return_pct !== undefined ? row.prophet_pred_return_pct : 0;

                              return (
                                <tr key={row.date} className="hover:bg-slate-800/40 text-slate-300">
                                  <td className="px-2 py-1 text-slate-400 font-bold">
                                    {row.date.slice(5)}
                                  </td>
                                  <td className="text-right px-1.5 py-1 text-white font-semibold">
                                    <div>₺{row.actual_price.toFixed(2)}</div>
                                    <div className={`text-[8px] font-bold ${
                                      isActUp ? 'text-emerald-400' : (isActDown ? 'text-rose-400' : 'text-slate-400')
                                    }`}>
                                      {isActUp ? '▲ UP +' : (isActDown ? '▼ DOWN ' : '■ FLAT ')}{row.actual_return_pct.toFixed(2)}%
                                    </div>
                                  </td>
                                  <td className="text-right px-1.5 py-1">
                                    <div className="font-semibold text-white">₺{row.ml_pred_price.toFixed(2)}</div>
                                    <div className={`text-[8.5px] font-bold flex items-center justify-end gap-1 ${
                                      isMlUp ? 'text-emerald-400' : (isMlDown ? 'text-rose-400' : 'text-slate-400')
                                    }`}>
                                      <span>{isMlUp ? '▲ UP +' : (isMlDown ? '▼ DOWN ' : '■ FLAT ')}{mlRet.toFixed(2)}%</span>
                                      <span className="text-[7.5px] text-slate-500 font-normal">({row.ml_err_pct.toFixed(1)}%)</span>
                                    </div>
                                  </td>
                                  <td className="text-center px-1.5 py-1">
                                    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[8px] font-black border tracking-wider ${
                                      row.ml_is_hit
                                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-sm shadow-emerald-950/40'
                                        : 'bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-sm shadow-rose-950/40'
                                    }`}>
                                      <span>{row.ml_is_hit ? '✓' : '✗'}</span>
                                      <span>{row.ml_is_hit ? 'CORRECT' : 'WRONG'}</span>
                                    </span>
                                  </td>
                                  <td className="text-right px-1.5 py-1 text-slate-300">
                                    <div>₺{row.prophet_pred_price.toFixed(2)}</div>
                                    <div className={`text-[8px] ${pRet >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'}`}>
                                      {pRet >= 0 ? '+' : ''}{pRet.toFixed(1)}% <span className="text-slate-500">({row.prophet_err_pct.toFixed(1)}%)</span>
                                    </div>
                                  </td>
                                  <td className="text-center px-1.5 py-1">
                                    <span className={`px-1 py-0.2 rounded text-[8px] font-bold ${
                                      row.mlb_action === 'BUY'
                                        ? 'bg-emerald-500/20 text-emerald-400'
                                        : 'bg-rose-500/20 text-rose-400'
                                    }`}>
                                      {row.mlb_action} {row.mlb_flow_tl >= 0 ? '+' : ''}{(row.mlb_flow_tl / 1e6).toFixed(0)}M
                                    </span>
                                  </td>
                                  <td className="text-center px-1.5 py-1">
                                    <span className={`px-1 py-0.2 rounded text-[8px] font-bold ${
                                      row.big5_action === 'BUY'
                                        ? 'bg-emerald-500/20 text-emerald-400'
                                        : 'bg-rose-500/20 text-rose-400'
                                    }`}>
                                      {row.big5_action} {row.big5_flow_tl >= 0 ? '+' : ''}{(row.big5_flow_tl / 1e6).toFixed(0)}M
                                    </span>
                                  </td>
                                  <td className="text-center px-1.5 py-1">
                                    <span className={`px-1 py-0.2 rounded text-[8px] font-bold ${
                                      row.kamu_action === 'BUY'
                                        ? 'bg-emerald-500/20 text-emerald-400'
                                        : 'bg-rose-500/20 text-rose-400'
                                    }`}>
                                      {row.kamu_action} {row.kamu_flow_tl >= 0 ? '+' : ''}{(row.kamu_flow_tl / 1e6).toFixed(0)}M
                                    </span>
                                  </td>
                                  <td className="text-center px-1.5 py-1 font-bold">
                                    <span className={`px-1.5 py-0.2 rounded text-[8px] border ${
                                      row.winner === 'CHALLENGER'
                                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                                        : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                                    }`}>
                                      {row.winner === 'CHALLENGER' ? 'ML' : 'PROPHET'}
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      <div className="text-[8px] text-slate-500 italic text-center pt-0.5">
                        Ground truth actions logged from daily broker clearing records (silver_daily_broker_summary)
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-6 text-slate-400 text-xs font-mono">
                    Loading 30-day walk-forward ledger...
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Chart Canvas */}
        <div ref={chartContainerRef} className="w-full min-h-[520px]" />
        {/* BIST 30 Shock Days Dashed Line Overlay Canvas */}
        <canvas ref={overlayCanvasRef} className="absolute inset-0 pointer-events-none z-10" />
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
