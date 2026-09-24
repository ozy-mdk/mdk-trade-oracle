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
import { useQuery } from '@tanstack/react-query';
import { CandleBar, TertipHorizonsResponse, ForwardHorizonCode, ShockDayItem } from '../types/api';
import { fetchShockDays } from '../api/client';
import { calculateForwardOpportunity, calculateEwmaConfluence } from '../utils/forwardOpportunity';
import { Compass, Target, TrendingUp, TrendingDown, Eye, EyeOff, Table, Zap } from 'lucide-react';

interface TradingViewChartProps {
  symbol: string;
  interval?: '1m' | '5m' | '1d';
  brokerId?: string;
  fifoAvgCost?: number | null;
  showCostLine?: boolean;
  tertipData?: TertipHorizonsResponse | null;
  onHoverData?: (data: CandleBar | null) => void;
}

export const TradingViewCandleChart: React.FC<TradingViewChartProps> = ({
  symbol,
  interval = '5m',
  brokerId = 'MLB',
  fifoAvgCost,
  showCostLine = false,
  tertipData,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const bofaSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const forwardAreaSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const costLineRef = useRef<any>(null);
  const outlookTargetLineRef = useRef<any>(null);
  const totalBarsRef = useRef<number>(0);
  const latestCandleRef = useRef<CandleBar | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCandle, setActiveCandle] = useState<CandleBar | null>(null);
  const [selectedHorizon, setSelectedHorizon] = useState<ForwardHorizonCode>('1M');
  const [showOutlookZone, setShowOutlookZone] = useState<boolean>(true);
  const [actionZoneTab, setActionZoneTab] = useState<'table' | 'outlook'>('table');
  const [showShockDays, setShowShockDays] = useState<boolean>(true);
  const [shockTypeFilter, setShockTypeFilter] = useState<'ALL' | 'POSITIVE' | 'NEGATIVE'>('ALL');
  const [selectedShockDate, setSelectedShockDate] = useState<string | null>(null);
  const candleBarsRef = useRef<CandleBar[]>([]);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Fetch BIST 30 Shock Days
  const { data: shockDays } = useQuery({
    queryKey: ['bist30ShockDays'],
    queryFn: () => fetchShockDays(0.03),
    staleTime: 1000 * 60 * 30,
  });

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

  const handleSelectShockDate = (dateStr: string) => {
    setSelectedShockDate(dateStr || null);
    if (!dateStr || !chartRef.current) return;
    const candles = candleBarsRef.current;
    if (!candles || candles.length === 0) return;

    const idx = candles.findIndex(
      (c) => new Date(c.time * 1000).toISOString().split('T')[0] === dateStr
    );
    if (idx >= 0) {
      setActiveCandle(candles[idx]);
      chartRef.current.timeScale().setVisibleLogicalRange({
        from: Math.max(0, idx - 15),
        to: idx + (interval === '1d' ? 35 : 80),
      });
    }
  };

  const shockMap = React.useMemo(() => {
    const map = new Map<string, ShockDayItem>();
    if (shockDays) {
      for (const s of shockDays) {
        map.set(s.trade_date, s);
      }
    }
    return map;
  }, [shockDays]);

  const activeDateStr = activeCandle?.time
    ? new Date(activeCandle.time * 1000).toISOString().split('T')[0]
    : null;
  const hoveredShock = activeDateStr ? shockMap.get(activeDateStr) : null;

  const currentPrice = activeCandle?.close || latestCandleRef.current?.close || 0;
  const outlook = calculateForwardOpportunity(currentPrice, tertipData, selectedHorizon);
  const confluence = calculateEwmaConfluence(currentPrice, tertipData);

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

    // 3. Forward Opportunity Ribbon (Upper Pane overlay in forward whitespace)
    const forwardAreaSeries = chart.addAreaSeries({
      priceScaleId: 'right',
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });
    forwardAreaSeriesRef.current = forwardAreaSeries;

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
          `/api/v1/market/candles?symbol=${encodeURIComponent(symbol)}&interval=${interval}&broker_id=${encodeURIComponent(brokerId)}&limit=1000`
        );
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }
        const data: CandleBar[] = await response.json();
        candleBarsRef.current = data;

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
        setError(err.message || 'Failed to load candlesticks');
        setLoading(false);
      }
    };

    fetchCandleData();

    return () => {
      isMounted = false;
    };
  }, [symbol, interval, brokerId]);

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

  // Update Forward Opportunity Projection (Target Line & Forward Shaded Ribbon)
  useEffect(() => {
    if (!candleSeriesRef.current || !chartRef.current) return;

    if (outlookTargetLineRef.current) {
      try {
        candleSeriesRef.current.removePriceLine(outlookTargetLineRef.current);
      } catch (e) {
        // Ignore removal error
      }
      outlookTargetLineRef.current = null;
    }

    if (!showOutlookZone || !outlook || !forwardAreaSeriesRef.current) {
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

    outlookTargetLineRef.current = candleSeriesRef.current.createPriceLine({
      price: outlook.targetCost,
      color: lineColor,
      lineWidth: 2,
      lineStyle: 2, // Dashed
      axisLabelVisible: true,
      title: `${brokerId} ${outlook.horizonCode} Target: ₺${outlook.targetCost.toFixed(2)} (${returnPrefix}${outlook.potentialReturnPct.toFixed(1)}%)`,
    });

    // 2. Populate Forward Area Ribbon across the 15-bar margin
    if (latestCandleRef.current) {
      const lastEpoch = latestCandleRef.current.time;
      const lastClose = latestCandleRef.current.close;
      const targetCost = outlook.targetCost;

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
      const startDateStr = new Date(lastEpoch * 1000).toISOString().split('T')[0];
      const curDate = new Date(`${startDateStr}T00:00:00Z`);

      // Anchor point on the latest candle
      forwardPoints.push({
        time: interval === '1d' ? startDateStr : (lastEpoch as any),
        value: lastClose,
      });

      const forwardDays = 8;
      for (let i = 1; i <= forwardDays; i++) {
        curDate.setUTCDate(curDate.getUTCDate() + 1);
        while (curDate.getUTCDay() === 0 || curDate.getUTCDay() === 6) {
          curDate.setUTCDate(curDate.getUTCDate() + 1);
        }
        const fDateStr = curDate.toISOString().split('T')[0];
        const fEpoch = Math.floor(curDate.getTime() / 1000);
        const alpha = i / forwardDays;
        const rampValue = Number((lastClose + (targetCost - lastClose) * alpha).toFixed(2));

        forwardPoints.push({
          time: interval === '1d' ? fDateStr : (fEpoch as any),
          value: rampValue,
        });
      }

      forwardAreaSeriesRef.current.setData(forwardPoints as any);
    }
  }, [outlook, showOutlookZone, interval, brokerId]);

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

        let targetCoord: number | null = null;
        if (interval === '1d') {
          targetCoord = timeScale.timeToCoordinate(shock.trade_date as any);
        } else {
          const firstBar = candleBarsRef.current?.find(
            (c) => new Date(c.time * 1000).toISOString().split('T')[0] === shock.trade_date
          );
          if (firstBar) {
            targetCoord = timeScale.timeToCoordinate(firstBar.time as any);
          }
        }

        if (targetCoord === null || targetCoord < 0 || targetCoord > width - 55) continue;
        const x = targetCoord;

        ctx.save();
        if (isSelected) {
          ctx.fillStyle = shock.is_positive_shock ? 'rgba(16, 185, 129, 0.16)' : 'rgba(244, 63, 94, 0.16)';
          ctx.fillRect(x - 14, 0, 28, height);
        }

        ctx.beginPath();
        ctx.setLineDash(isSelected ? [] : [4, 4]);
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        if (shock.is_positive_shock) {
          ctx.strokeStyle = isSelected ? '#34d399' : 'rgba(16, 185, 129, 0.75)'; // Emerald
        } else {
          ctx.strokeStyle = isSelected ? '#fb7171' : 'rgba(244, 63, 94, 0.75)'; // Rose red
        }
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Header pill badge
        const retSign = shock.daily_return_pct >= 0 ? '+' : '';
        const text = `${retSign}${(shock.daily_return_pct * 100).toFixed(1)}%`;
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
  }, [showShockDays, filteredShockDays, selectedShockDate, interval]);

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
                {hoveredShock && (
                  <span
                    className={`ml-1.5 px-1.5 py-0.2 rounded text-[10px] font-bold border tracking-wider font-mono ${
                      hoveredShock.is_positive_shock
                        ? 'bg-emerald-500/25 text-emerald-300 border-emerald-500/50 shadow-sm'
                        : 'bg-rose-500/25 text-rose-300 border-rose-500/50 shadow-sm'
                    }`}
                    title={`BIST 30 Official Benchmark Move: ${(hoveredShock.daily_return_pct * 100).toFixed(2)}%`}
                  >
                    ⚡ BIST 30: {hoveredShock.daily_return_pct >= 0 ? '+' : ''}${(hoveredShock.daily_return_pct * 100).toFixed(2)}% ({hoveredShock.is_positive_shock ? 'POSITIVE' : 'NEGATIVE'} SHOCK)
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
          {/* BIST 30 Shock Days Controls */}
          <div className="flex items-center space-x-1.5 bg-slate-950/70 px-2 py-0.5 rounded border border-slate-800">
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
                <div className="hidden lg:flex items-center space-x-1 border-l border-slate-800 pl-1.5 text-[9px] font-mono">
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
                    className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-1.5 py-0.5 text-[10px] font-mono focus:outline-none focus:border-amber-400 max-w-[160px] truncate"
                    title="Select a shock day to center chart and inspect trader follow-through"
                  >
                    <option value="">Jump to Shock ({filteredShockDays.length})...</option>
                    {filteredShockDays.map((s) => (
                      <option key={s.trade_date} value={s.trade_date}>
                        {s.trade_date} : {s.daily_return_pct >= 0 ? '+' : ''}
                        {(s.daily_return_pct * 100).toFixed(1)}% {s.is_positive_shock ? '▲' : '▼'}
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

          {/* Action Zone Toggle */}
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

          {loading && <span className="text-cyan-400 animate-pulse font-mono">Streaming...</span>}
          {error && <span className="text-rose-400 font-mono">{error}</span>}
        </div>
      </div>

      {/* Floating Institutional Forward Action Card (Positioned in 15-Bar Forward Whitespace) */}
      {showOutlookZone && outlook && (
        <div className="absolute top-14 right-14 z-20 w-[390px] glass-panel bg-slate-950/92 backdrop-blur-md border border-slate-700/80 p-3.5 rounded-xl shadow-2xl transition-all pointer-events-auto">
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
                            onClick={() => setSelectedHorizon(row.code)}
                            className={`cursor-pointer transition-colors ${
                              isSelected
                                ? 'bg-cyan-500/20 text-white font-semibold'
                                : 'hover:bg-slate-800/50 text-slate-300'
                            }`}
                            title={`Click to set ${row.label} as active projection target`}
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
                Click any row to project that horizon's target line onto the chart
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
                      onClick={() => setSelectedHorizon(hz)}
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
      <div ref={chartContainerRef} className="w-full h-[560px]" />
      {/* BIST 30 Shock Days Dashed Line Overlay Canvas */}
      <canvas ref={overlayCanvasRef} className="absolute inset-0 pointer-events-none z-10" />
    </div>
  );
};
