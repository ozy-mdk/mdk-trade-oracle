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
import { CandleBar, TertipHorizonsResponse, ForwardHorizonCode, ShockDayItem, TertipMlForecastResponse } from '../types/api';
import { fetchShockDays, fetchTertipMlForecast } from '../api/client';
import { calculateForwardOpportunity, calculateEwmaConfluence } from '../utils/forwardOpportunity';
import { Compass, Target, TrendingUp, TrendingDown, Eye, EyeOff, Table, Zap, BarChart3, History, Award, RefreshCw, Activity } from 'lucide-react';

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
  const [actionZoneTab, setActionZoneTab] = useState<'table' | 'outlook' | 'realization' | 'forecast' | 'track30d'>('forecast');
  const [showShockDays, setShowShockDays] = useState<boolean>(true);
  const [shockTypeFilter, setShockTypeFilter] = useState<'ALL' | 'POSITIVE' | 'NEGATIVE'>('ALL');
  const [selectedShockDate, setSelectedShockDate] = useState<string | null>(null);
  const candleBarsRef = useRef<CandleBar[]>([]);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Fetch Tertip 3-Pillar ML Forecast & 30-Day Walk-Forward Track
  const { data: mlForecast, refetch: refetchMlForecast, isFetching: isFetchingMl } = useQuery<TertipMlForecastResponse>({
    queryKey: ['tertipMlForecast', symbol],
    queryFn: () => fetchTertipMlForecast(symbol),
    staleTime: 1000 * 15,
  });

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
                    title={`Entire BIST 30 Basket Value Move: ${(hoveredShock.daily_return_pct * 100).toFixed(2)}% | Total Trading Value: ₺${(hoveredShock.total_turnover_tl / 1e9).toFixed(2)}B | BofA Flow: ₺${(hoveredShock.bofa_net_flow_tl / 1e6).toFixed(1)}M`}
                  >
                    ⚡ BIST 30 Basket: {hoveredShock.daily_return_pct >= 0 ? '+' : ''}${(hoveredShock.daily_return_pct * 100).toFixed(2)}% ({hoveredShock.is_positive_shock ? 'POSITIVE' : 'NEGATIVE'} SHOCK){hoveredShock.total_turnover_tl > 0 ? ` | Turnover: ₺${(hoveredShock.total_turnover_tl / 1e9).toFixed(1)}B` : ''}{hoveredShock.bofa_net_flow_tl !== 0 ? ` | BofA: ${hoveredShock.bofa_net_flow_tl >= 0 ? '+' : ''}₺${(hoveredShock.bofa_net_flow_tl / 1e6).toFixed(0)}M` : ''}
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
        <div className="absolute top-14 right-14 z-20 w-[440px] glass-panel bg-slate-950/92 backdrop-blur-md border border-slate-700/80 p-3.5 rounded-xl shadow-2xl transition-all pointer-events-auto">
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
                Click any row to project that horizon's target line onto the chart
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
                        <span>CHAMPION: {mlForecast.tournament_summary.champion === 'TERTIP_ML_CHALLENGER' ? `TERTIP ML (${mlForecast.tournament_summary.ml_champion_type || 'Auto'})` : 'PROPHET BASE'}</span>
                      </span>
                      <span className="text-[9px] text-slate-400 font-mono">
                        ({mlForecast.tournament_summary.champion_dir_hits ?? (mlForecast.tournament_summary.champion === 'TERTIP_ML_CHALLENGER' ? (mlForecast.tournament_summary.ml_dir_hits ?? 17) : (mlForecast.tournament_summary.prophet_dir_hits ?? 13))}/{mlForecast.tournament_summary.total_sessions}d direction correct • MAE {mlForecast.tournament_summary.champion_mae_pct?.toFixed(2) ?? (mlForecast.tournament_summary.champion === 'TERTIP_ML_CHALLENGER' ? mlForecast.tournament_summary.ml_mae_pct.toFixed(2) : mlForecast.tournament_summary.prophet_mae_pct.toFixed(2))}%)
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

                  {/* BIST 30 (XU030) 3-Day Momentum & Real-time Refresh Banner */}
                  <div className="flex flex-wrap items-center justify-between gap-1 text-[9px] font-mono bg-slate-950/70 px-2.5 py-1.5 rounded-lg border border-slate-800/80">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="text-amber-400/90 font-sans font-semibold flex items-center gap-1">
                        <Activity className="w-3 h-3 text-amber-400" />
                        <span>BIST 30 (XU030) 3-Day Return:</span>
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="text-slate-400 font-medium">Today (T):</span>
                        <span className={`font-bold ${
                          (mlForecast.bist30_trend?.today_pct ?? 0) > 0.02
                            ? 'text-emerald-400'
                            : (mlForecast.bist30_trend?.today_pct ?? 0) < -0.02 ? 'text-rose-400' : 'text-slate-200'
                        }`}>
                          {(mlForecast.bist30_trend?.today_pct ?? 0) > 0 ? '+' : ''}{(mlForecast.bist30_trend?.today_pct ?? 0).toFixed(2)}%
                        </span>
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="text-slate-400 font-medium">Yesterday (T-1):</span>
                        <span className={`font-bold ${
                          (mlForecast.bist30_trend?.yesterday_pct ?? 0) > 0.02
                            ? 'text-emerald-400'
                            : (mlForecast.bist30_trend?.yesterday_pct ?? 0) < -0.02 ? 'text-rose-400' : 'text-slate-200'
                        }`}>
                          {(mlForecast.bist30_trend?.yesterday_pct ?? 0) > 0 ? '+' : ''}{(mlForecast.bist30_trend?.yesterday_pct ?? 0).toFixed(2)}%
                        </span>
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="text-slate-400 font-medium">Day Before (T-2):</span>
                        <span className={`font-bold ${
                          (mlForecast.bist30_trend?.day_before_pct ?? 0) > 0.02
                            ? 'text-emerald-400'
                            : (mlForecast.bist30_trend?.day_before_pct ?? 0) < -0.02 ? 'text-rose-400' : 'text-slate-200'
                        }`}>
                          {(mlForecast.bist30_trend?.day_before_pct ?? 0) > 0 ? '+' : ''}{(mlForecast.bist30_trend?.day_before_pct ?? 0).toFixed(2)}%
                        </span>
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-[8px] text-slate-500">
                        {mlForecast.calculated_at ? new Date(mlForecast.calculated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : ''}
                      </span>
                      <button
                        onClick={() => {
                          fetchTertipMlForecast(symbol, true).then(() => {
                            refetchMlForecast();
                          });
                        }}
                        className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-[8.5px] font-sans transition-colors border border-slate-700/60"
                        title="Recalculate forecast live bypassing cache"
                      >
                        <RefreshCw className={`w-2.5 h-2.5 ${isFetchingMl ? 'animate-spin text-cyan-400' : 'text-slate-400'}`} />
                        <span>Refresh</span>
                      </button>
                    </div>
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

                  {/* 3-Pillar Execution & PnL Breakdown (Today vs Yesterday) */}
                  {mlForecast.pillar_execution && (
                    <div className="space-y-1.5 bg-slate-900/40 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] font-bold text-slate-300 tracking-wide flex items-center justify-between">
                        <span className="flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                          3-Pillar Execution & PnL (Today vs Yesterday)
                        </span>
                        <span className="text-[9px] text-slate-400 font-mono">
                          Today ({mlForecast.pillar_execution.today_date}) vs Yesterday ({mlForecast.pillar_execution.yesterday_date})
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
                        {[
                          { key: 'mlb', name: 'Bank of America (MLB)', color: 'border-cyan-500/40 bg-cyan-950/20' },
                          { key: 'big5', name: 'Big 5 Domestic Desks', color: 'border-indigo-500/40 bg-indigo-950/20' },
                          { key: 'kamu', name: 'Kamu Conduits (incl. TRA)', color: 'border-emerald-500/40 bg-emerald-950/20' },
                        ].map((p) => {
                          const pData = (mlForecast.pillar_execution as any)?.[p.key];
                          if (!pData) return null;
                          const today = pData.today;
                          const yest = pData.yesterday;

                          return (
                            <div key={p.key} className={`p-2 rounded border ${p.color} text-[9px] font-mono flex flex-col justify-between`}>
                              <div className="font-bold text-slate-200 border-b border-slate-800/80 pb-1 mb-1.5 flex justify-between items-center">
                                <span>{p.name}</span>
                              </div>

                              <div className="space-y-1.5">
                                {/* Today */}
                                <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/50">
                                  <div className="text-[8px] font-sans text-slate-400 font-semibold mb-0.5 flex justify-between">
                                    <span>TODAY</span>
                                    <span className={today.net_flow_tl >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                                      Net: {today.net_flow_tl >= 0 ? '+' : ''}₺{(today.net_flow_tl / 1e6).toFixed(1)}M
                                    </span>
                                  </div>
                                  <div className="grid grid-cols-3 gap-1 text-[8px] text-slate-300">
                                    <div><span className="text-slate-500">Buy: </span>₺{(today.buy_tl / 1e6).toFixed(1)}M</div>
                                    <div><span className="text-slate-500">Sell: </span>₺{(today.sell_tl / 1e6).toFixed(1)}M</div>
                                    <div>
                                      <span className="text-slate-500">PnL: </span>
                                      <span className={today.daily_pnl_tl >= 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                                        {today.daily_pnl_tl >= 0 ? '+' : ''}₺{(today.daily_pnl_tl / 1e6).toFixed(2)}M
                                      </span>
                                    </div>
                                  </div>
                                </div>

                                {/* Yesterday */}
                                <div className="bg-slate-950/40 p-1.5 rounded border border-slate-800/30">
                                  <div className="text-[8px] font-sans text-slate-500 font-semibold mb-0.5 flex justify-between">
                                    <span>YESTERDAY</span>
                                    <span className={yest.net_flow_tl >= 0 ? 'text-emerald-400/80 font-bold' : 'text-rose-400/80 font-bold'}>
                                      Net: {yest.net_flow_tl >= 0 ? '+' : ''}₺{(yest.net_flow_tl / 1e6).toFixed(1)}M
                                    </span>
                                  </div>
                                  <div className="grid grid-cols-3 gap-1 text-[8px] text-slate-400">
                                    <div><span className="text-slate-500">Buy: </span>₺{(yest.buy_tl / 1e6).toFixed(1)}M</div>
                                    <div><span className="text-slate-500">Sell: </span>₺{(yest.sell_tl / 1e6).toFixed(1)}M</div>
                                    <div>
                                      <span className="text-slate-500">PnL: </span>
                                      <span className={yest.daily_pnl_tl >= 0 ? 'text-emerald-400/80 font-semibold' : 'text-rose-400/80 font-semibold'}>
                                        {yest.daily_pnl_tl >= 0 ? '+' : ''}₺{(yest.daily_pnl_tl / 1e6).toFixed(2)}M
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
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
                      <div className={`p-1.5 rounded border ${mlForecast.tournament_summary.champion === 'TERTIP_ML_CHALLENGER' ? 'bg-emerald-950/40 border-emerald-500/40' : 'bg-slate-950/70 border-slate-800/80'}`}>
                        <div className="text-[8.5px] text-slate-400">ML Directional Hit ({mlForecast.tournament_summary.ml_champion_type || 'ML'})</div>
                        <div className="text-xs font-bold text-emerald-400 mt-0.5">
                          {(mlForecast.tournament_summary.ml_dir_hit_rate_pct ?? mlForecast.tournament_summary.ml_hit_rate_pct).toFixed(1)}%
                        </div>
                        <div className="text-[8px] text-slate-400">
                          {mlForecast.tournament_summary.ml_dir_hits ?? Math.round((mlForecast.tournament_summary.ml_hit_rate_pct / 100) * mlForecast.tournament_summary.total_sessions)}/30 days correct
                        </div>
                      </div>

                      <div className={`p-1.5 rounded border ${mlForecast.tournament_summary.champion === 'PROPHET_BASE' ? 'bg-cyan-950/40 border-cyan-500/40' : 'bg-slate-950/70 border-slate-800/80'}`}>
                        <div className="text-[8.5px] text-slate-400">Prophet Directional Hit</div>
                        <div className="text-xs font-bold text-cyan-400 mt-0.5">
                          {(mlForecast.tournament_summary.prophet_dir_hit_rate_pct ?? mlForecast.tournament_summary.prophet_hit_rate_pct).toFixed(1)}%
                        </div>
                        <div className="text-[8px] text-slate-400">
                          {mlForecast.tournament_summary.prophet_dir_hits ?? Math.round((mlForecast.tournament_summary.prophet_hit_rate_pct / 100) * mlForecast.tournament_summary.total_sessions)}/30 days correct
                        </div>
                      </div>

                      <div className="bg-slate-950/70 p-1.5 rounded border border-slate-800/80">
                        <div className="text-[8.5px] text-slate-400">ML Error (MAE)</div>
                        <div className="text-xs font-bold text-emerald-400 mt-0.5">
                          {mlForecast.tournament_summary.ml_mae_pct.toFixed(2)}%
                        </div>
                        <div className="text-[8px] text-slate-500">
                          {mlForecast.tournament_summary.ml_error_wins ?? mlForecast.tournament_summary.ml_wins}d lower error
                        </div>
                      </div>

                      <div className="bg-slate-950/70 p-1.5 rounded border border-slate-800/80">
                        <div className="text-[8.5px] text-slate-400">Prophet Error (MAE)</div>
                        <div className="text-xs font-bold text-slate-300 mt-0.5">
                          {mlForecast.tournament_summary.prophet_mae_pct.toFixed(2)}%
                        </div>
                        <div className="text-[8px] text-slate-500">
                          {mlForecast.tournament_summary.prophet_error_wins ?? mlForecast.tournament_summary.prophet_wins}d lower error
                        </div>
                      </div>
                    </div>

                    {/* Directional Win Visual Bar */}
                    <div className="space-y-0.5">
                      <div className="flex justify-between text-[8.5px] font-mono">
                        <span className="text-emerald-400 font-semibold">
                          ML Direction: {mlForecast.tournament_summary.ml_dir_hits ?? Math.round((mlForecast.tournament_summary.ml_hit_rate_pct / 100) * mlForecast.tournament_summary.total_sessions)}/30d correct
                        </span>
                        <span className="text-cyan-400 font-semibold">
                          Prophet Direction: {mlForecast.tournament_summary.prophet_dir_hits ?? Math.round((mlForecast.tournament_summary.prophet_hit_rate_pct / 100) * mlForecast.tournament_summary.total_sessions)}/30d correct
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden flex">
                        <div
                          className="bg-emerald-500 h-full transition-all duration-300"
                          style={{
                            width: `${((mlForecast.tournament_summary.ml_dir_hits ?? Math.round((mlForecast.tournament_summary.ml_hit_rate_pct / 100) * mlForecast.tournament_summary.total_sessions)) / mlForecast.tournament_summary.total_sessions) * 100}%`,
                          }}
                        />
                        <div
                          className="bg-cyan-500 h-full transition-all duration-300"
                          style={{
                            width: `${((mlForecast.tournament_summary.prophet_dir_hits ?? Math.round((mlForecast.tournament_summary.prophet_hit_rate_pct / 100) * mlForecast.tournament_summary.total_sessions)) / mlForecast.tournament_summary.total_sessions) * 100}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Dynamic Arena Selection Details */}
                    {mlForecast.tournament_summary && (
                      <div className="flex flex-wrap items-center justify-between text-[8px] font-mono px-2 py-1 rounded bg-slate-950/60 border border-slate-800/60 text-slate-400 gap-1">
                        <span className="text-slate-400 font-semibold flex items-center gap-1">
                          Arena 30D Candidates:
                          <span className="text-[7.5px] px-1 py-0.2 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800/40">
                            {mlForecast.tournament_summary.crowned_horizon?.toUpperCase() || '12M'} ({mlForecast.train_lookback_sessions || 252}d)
                          </span>
                        </span>
                        <div className="flex flex-wrap items-center gap-2">
                          {[
                            { name: 'XGBoost', hits: mlForecast.tournament_summary.xgboost_dir_hits, rate: mlForecast.tournament_summary.xgboost_30d_hit_rate_pct ?? mlForecast.tournament_summary.xgboost_dir_hit_rate_pct ?? mlForecast.tournament_summary.xgboost_hit_rate_pct, mae: mlForecast.tournament_summary.xgboost_mae_pct, loss: mlForecast.tournament_summary.xgboost_penalty_loss_30d },
                            { name: 'LightGBM', hits: mlForecast.tournament_summary.lightgbm_dir_hits, rate: mlForecast.tournament_summary.lightgbm_30d_hit_rate_pct ?? mlForecast.tournament_summary.lightgbm_dir_hit_rate_pct ?? mlForecast.tournament_summary.lightgbm_hit_rate_pct, mae: mlForecast.tournament_summary.lightgbm_mae_pct, loss: mlForecast.tournament_summary.lightgbm_penalty_loss_30d },
                            { name: 'Huber', hits: mlForecast.tournament_summary.huber_dir_hits, rate: mlForecast.tournament_summary.huber_30d_hit_rate_pct ?? mlForecast.tournament_summary.huber_dir_hit_rate_pct ?? mlForecast.tournament_summary.huber_hit_rate_pct, mae: mlForecast.tournament_summary.huber_mae_pct, loss: mlForecast.tournament_summary.huber_penalty_loss_30d },
                            { name: 'BayesianRidge', hits: mlForecast.tournament_summary.bayesian_ridge_dir_hits, rate: mlForecast.tournament_summary.bayesian_ridge_30d_hit_rate_pct ?? mlForecast.tournament_summary.bayesian_ridge_dir_hit_rate_pct ?? mlForecast.tournament_summary.bayesian_ridge_hit_rate_pct, mae: mlForecast.tournament_summary.bayesian_ridge_mae_pct, loss: mlForecast.tournament_summary.bayesian_ridge_penalty_loss_30d },
                            { name: 'Ridge', hits: mlForecast.tournament_summary.ridge_dir_hits, rate: mlForecast.tournament_summary.ridge_30d_hit_rate_pct ?? mlForecast.tournament_summary.ridge_dir_hit_rate_pct ?? mlForecast.tournament_summary.ridge_hit_rate_pct, mae: mlForecast.tournament_summary.ridge_mae_pct, loss: mlForecast.tournament_summary.ridge_penalty_loss_30d },
                          ].filter(m => m.rate !== undefined).map((m, idx, arr) => {
                            const isChamp = mlForecast.tournament_summary.ml_champion_type === m.name;
                            return (
                              <React.Fragment key={m.name}>
                                <span
                                  className={isChamp ? 'text-amber-300 font-bold' : 'text-slate-400'}
                                  title={m.loss !== undefined ? `Option 2 Penalty Loss: ${m.loss.toFixed(2)}% | MAE: ${m.mae?.toFixed(2)}%` : undefined}
                                >
                                  {m.name}: {m.hits !== undefined ? `${m.hits}/30d` : ''} ({m.rate?.toFixed(1)}%{m.loss !== undefined ? `, L: ${m.loss.toFixed(1)}%` : (m.mae !== undefined ? `, ${m.mae.toFixed(1)}%` : '')})
                                  {isChamp && ' 👑'}
                                </span>
                                {idx < arr.length - 1 && <span className="text-slate-600">|</span>}
                              </React.Fragment>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 30-Day Walk-Forward Reality Ledger Table */}
                  <div className="space-y-1">
                    <div className="text-[10px] font-bold text-slate-300 tracking-wide flex flex-wrap items-center justify-between gap-1">
                      <span>Out-of-Sample Walk-Forward Reality Ledger</span>
                      <div className="flex items-center gap-1.5 font-mono text-[8.5px]">
                        {(() => {
                          const ledger = mlForecast.walk_forward_ledger || [];
                          const tournament = mlForecast.tournament_summary;
                          const isConfluence = tournament?.champion === 'TERTIP_CONFLUENCE';
                          const getRowHit = (r: any) =>
                            isConfluence && r.confluence_is_hit !== undefined ? r.confluence_is_hit : r.ml_is_hit;

                          const champHits = ledger.filter((r) => getRowHit(r)).length;
                          const champMiss = ledger.length - champHits;
                          const mlHits = ledger.filter((r) => r.ml_is_hit).length;
                          const l10 = ledger.slice(-10);
                          const l10Hits = l10.filter((r) => getRowHit(r)).length;
                          const l10Miss = l10.length - l10Hits;
                          return (
                            <>
                              <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                                30D Champion: <span className="text-emerald-400 font-bold">{champHits}✓</span> / <span className="text-rose-400 font-bold">{champMiss}✗</span> ({((champHits / (ledger.length || 1)) * 100).toFixed(1)}%)
                              </span>
                              {isConfluence && (
                                <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                                  ML Alone: <span className="text-cyan-300 font-bold">{mlHits}✓</span>
                                </span>
                              )}
                              <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                                Last 10D: <span className="text-emerald-400 font-bold">{l10Hits}✓</span> / <span className="text-rose-400 font-bold">{l10Miss}✗</span> ({((l10Hits / (l10.length || 1)) * 100).toFixed(0)}%)
                              </span>
                            </>
                          );
                        })()}
                      </div>
                    </div>

                    <div className="overflow-x-auto max-h-56 overflow-y-auto rounded-lg border border-slate-800/90">
                      <table className="w-full text-[9px] font-mono border-collapse">
                        <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 z-10">
                          <tr className="text-slate-400">
                            <th className="text-left px-2 py-1 font-sans">Date</th>
                            <th className="text-right px-1.5 py-1 font-sans">Actual</th>
                            <th className="text-right px-1.5 py-1 font-sans text-cyan-300">
                              {mlForecast.tournament_summary?.champion === 'TERTIP_CONFLUENCE' ? 'Confluence Pred' : 'ML Pred'}
                            </th>
                            <th className="text-center px-1.5 py-1 font-sans text-cyan-300">
                              {mlForecast.tournament_summary?.champion === 'TERTIP_CONFLUENCE' ? 'Champion Hit?' : 'ML Hit?'}
                            </th>
                            {mlForecast.tournament_summary?.champion === 'TERTIP_CONFLUENCE' && (
                              <th className="text-right px-1.5 py-1 font-sans text-sky-400">ML Alone</th>
                            )}
                            <th className="text-right px-1.5 py-1 font-sans text-amber-200">XU030</th>
                            <th className="text-right px-1.5 py-1 font-sans text-purple-300">Prophet</th>
                            <th className="text-center px-1.5 py-1 font-sans">MLB Did</th>
                            <th className="text-center px-1.5 py-1 font-sans">BIG5 Did</th>
                            <th className="text-center px-1.5 py-1 font-sans">KAMU Did</th>
                            <th className="text-center px-1.5 py-1 font-sans">Winner</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 bg-slate-950/60">
                          {mlForecast.walk_forward_ledger.map((row) => {
                            const isConfluence = mlForecast.tournament_summary?.champion === 'TERTIP_CONFLUENCE';
                            const rowHit = isConfluence && row.confluence_is_hit !== undefined ? row.confluence_is_hit : row.ml_is_hit;
                            const predPrice = isConfluence && row.confluence_pred_price !== undefined ? row.confluence_pred_price : row.ml_pred_price;
                            const predRet = isConfluence && row.confluence_pred_return_pct !== undefined ? row.confluence_pred_return_pct : (row.ml_pred_return_pct || 0);
                            const errPct = isConfluence && row.confluence_err_pct !== undefined ? row.confluence_err_pct : row.ml_err_pct;
                            const isPredUp = predRet > 0.02;
                            const isPredDown = predRet < -0.02;
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
                                  <div className="font-semibold text-white">₺{predPrice.toFixed(2)}</div>
                                  <div className={`text-[8.5px] font-bold flex items-center justify-end gap-1 ${
                                    isPredUp ? 'text-emerald-400' : (isPredDown ? 'text-rose-400' : 'text-slate-400')
                                  }`}>
                                    <span>{isPredUp ? '▲ UP +' : (isPredDown ? '▼ DOWN ' : '■ FLAT ')}{predRet.toFixed(2)}%</span>
                                    <span className="text-[7.5px] text-slate-500 font-normal">({errPct.toFixed(1)}%)</span>
                                  </div>
                                </td>
                                <td className="text-center px-1.5 py-1">
                                  <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[8px] font-black border tracking-wider ${
                                    rowHit
                                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-sm shadow-emerald-950/40'
                                      : 'bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-sm shadow-rose-950/40'
                                  }`}>
                                    <span>{rowHit ? '✓' : '✗'}</span>
                                    <span>{rowHit ? 'CORRECT' : 'WRONG'}</span>
                                  </span>
                                </td>
                                {isConfluence && (
                                  <td className="text-right px-1.5 py-1 text-slate-300">
                                    <div className="font-semibold">₺{row.ml_pred_price.toFixed(2)}</div>
                                    <div className="text-[7.5px] flex items-center justify-end gap-1">
                                      <span className={(row.ml_pred_return_pct || 0) > 0 ? 'text-emerald-400' : 'text-rose-400'}>
                                        {(row.ml_pred_return_pct || 0) > 0 ? '+' : ''}{(row.ml_pred_return_pct || 0).toFixed(2)}%
                                      </span>
                                      <span className={row.ml_is_hit ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                                        {row.ml_is_hit ? '✓' : '✗'}
                                      </span>
                                    </div>
                                  </td>
                                )}
                                <td className="text-right px-1.5 py-1">
                                  <div className={`text-[8.5px] font-bold ${
                                    (row.bist30_ret_pct ?? 0) > 0.02 
                                      ? 'text-emerald-400' 
                                      : ((row.bist30_ret_pct ?? 0) < -0.02 ? 'text-rose-400' : 'text-slate-400')
                                  }`}>
                                    {(row.bist30_ret_pct ?? 0) > 0 ? '+' : ''}{(row.bist30_ret_pct ?? 0).toFixed(2)}%
                                  </div>
                                </td>
                                <td className="text-right px-1.5 py-1 text-slate-300">
                                  <div className="flex items-center justify-end gap-1">
                                    <span>₺{row.prophet_pred_price.toFixed(2)}</span>
                                    <span className={`text-[7px] font-black px-1 py-0.2 rounded border ${
                                      row.prophet_is_hit
                                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                                        : 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                                    }`}>
                                      {row.prophet_is_hit ? '✓' : '✗'}
                                    </span>
                                  </div>
                                  <div className={`text-[8px] ${pRet >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'}`}>
                                    {pRet >= 0 ? '+' : ''}{pRet.toFixed(1)}% <span className="text-slate-500">({row.prophet_err_pct.toFixed(1)}%)</span>
                                  </div>
                                </td>
                                <td className="text-center px-1.5 py-1">
                                  <span
                                    className={`px-1 py-0.2 rounded text-[8px] font-bold cursor-help ${
                                      row.mlb_action === 'BUY'
                                        ? 'bg-emerald-500/20 text-emerald-400'
                                        : 'bg-rose-500/20 text-rose-400'
                                    }`}
                                    title={`BofA MLB: Bought ₺${((row.mlb_buy_tl || 0) / 1e6).toFixed(1)}M | Sold ₺${((row.mlb_sell_tl || 0) / 1e6).toFixed(1)}M | PnL ₺${((row.mlb_pnl_tl || 0) / 1e6).toFixed(2)}M`}
                                  >
                                    {row.mlb_action} {row.mlb_flow_tl >= 0 ? '+' : ''}{(row.mlb_flow_tl / 1e6).toFixed(0)}M
                                  </span>
                                </td>
                                <td className="text-center px-1.5 py-1">
                                  <span
                                    className={`px-1 py-0.2 rounded text-[8px] font-bold cursor-help ${
                                      row.big5_action === 'BUY'
                                        ? 'bg-emerald-500/20 text-emerald-400'
                                        : 'bg-rose-500/20 text-rose-400'
                                    }`}
                                    title={`BIG 5: Bought ₺${((row.big5_buy_tl || 0) / 1e6).toFixed(1)}M | Sold ₺${((row.big5_sell_tl || 0) / 1e6).toFixed(1)}M | PnL ₺${((row.big5_pnl_tl || 0) / 1e6).toFixed(2)}M`}
                                  >
                                    {row.big5_action} {row.big5_flow_tl >= 0 ? '+' : ''}{(row.big5_flow_tl / 1e6).toFixed(0)}M
                                  </span>
                                </td>
                                <td className="text-center px-1.5 py-1">
                                  <span
                                    className={`px-1 py-0.2 rounded text-[8px] font-bold cursor-help ${
                                      row.kamu_action === 'BUY'
                                        ? 'bg-emerald-500/20 text-emerald-400'
                                        : 'bg-rose-500/20 text-rose-400'
                                    }`}
                                    title={`KAMU (w/ TRA): Bought ₺${((row.kamu_buy_tl || 0) / 1e6).toFixed(1)}M | Sold ₺${((row.kamu_sell_tl || 0) / 1e6).toFixed(1)}M | PnL ₺${((row.kamu_pnl_tl || 0) / 1e6).toFixed(2)}M`}
                                  >
                                    {row.kamu_action} {row.kamu_flow_tl >= 0 ? '+' : ''}{(row.kamu_flow_tl / 1e6).toFixed(0)}M
                                  </span>
                                </td>
                                <td className="text-center px-1.5 py-1 font-bold">
                                  <span className={`px-1.5 py-0.2 rounded text-[8px] border ${
                                    bothMiss
                                      ? 'bg-slate-800 text-slate-400 border-slate-700'
                                      : (row.winner === 'CHALLENGER'
                                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                                          : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40')
                                  }`}>
                                    {bothMiss ? 'BOTH MISS' : (row.winner === 'CHALLENGER' ? 'ML' : 'PROPHET')}
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
      <div ref={chartContainerRef} className="w-full h-[560px]" />
      {/* BIST 30 Shock Days Dashed Line Overlay Canvas */}
      <canvas ref={overlayCanvasRef} className="absolute inset-0 pointer-events-none z-10" />
    </div>
  );
};
