import {
  TertipHorizonsResponse,
  ForwardOpportunityOutlook,
  ForwardHorizonCode,
  EwmaConfluenceSummary,
  EwmaHorizonSummaryRow,
} from '../types/api';

/**
 * Compute institutional cost dislocation and forward opportunity outlook.
 *
 * Microstructure hypothesis:
 * Dominant institutional participants (e.g. BofA / MLB) manage multi-billion TL positions.
 * When price is dislocated below their EWMA entry cost by >= 3% or >= 5%, selling at market
 * causes extreme self-slippage; algorithms are incentivized to defend bids and squeeze price back to cost.
 * When price is >= 3% or >= 5% above cost, the desk systematically harvests liquidity.
 */
export function calculateForwardOpportunity(
  closePrice: number,
  tertipData: TertipHorizonsResponse | null | undefined,
  selectedHorizon: ForwardHorizonCode = '1M'
): ForwardOpportunityOutlook | null {
  if (!tertipData || !closePrice || closePrice <= 0) return null;

  let targetCost = 0;
  let horizonLabel = '';

  if (selectedHorizon === 'FIFO') {
    targetCost = tertipData.fifo_avg_cost;
    horizonLabel = 'Core FIFO Average Cost';
  } else {
    const h = (tertipData.horizons || []).find((item) => item.code === selectedHorizon);
    if (h && h.ewma_unit_cost > 0) {
      targetCost = h.ewma_unit_cost;
      horizonLabel = `${h.label} EWMA Cost`;
    } else {
      targetCost = tertipData.fifo_avg_cost;
      horizonLabel = 'FIFO Average Cost';
    }
  }

  if (targetCost <= 0) return null;

  // Spread: how far price is from institutional cost basis
  const spreadPct = ((closePrice - targetCost) / targetCost) * 100;
  // Potential return: return required from current price to reach institutional cost
  const potentialReturnPct = ((targetCost - closePrice) / closePrice) * 100;

  if (spreadPct <= -5.0) {
    return {
      horizonCode: selectedHorizon,
      horizonLabel,
      closePrice,
      targetCost,
      spreadPct,
      potentialReturnPct,
      direction: 'BUY',
      severity: 'STRONG',
      badgeLabel: 'STRONG BUY (DEFENSE REBOUND)',
      badgeColor: 'emerald',
      playbook: 'DEFENSE_SUPPORT',
      rationale: `Desk is deeply underwater by ${Math.abs(spreadPct).toFixed(1)}%. Book liquidity limits prevent dumping; algorithms are incentivized to cushion bids and trigger a mean-reversion squeeze toward cost.`,
    };
  }

  if (spreadPct <= -3.0) {
    return {
      horizonCode: selectedHorizon,
      horizonLabel,
      closePrice,
      targetCost,
      spreadPct,
      potentialReturnPct,
      direction: 'BUY',
      severity: 'MODERATE',
      badgeLabel: 'MODERATE BUY (TACTICAL BID)',
      badgeColor: 'cyan',
      playbook: 'TACTICAL_ACCUMULATION',
      rationale: `Price is tactically oversold vs ${selectedHorizon} cost by ${Math.abs(spreadPct).toFixed(1)}%. Expect algorithmic absorption at key support levels.`,
    };
  }

  if (spreadPct >= 5.0) {
    return {
      horizonCode: selectedHorizon,
      horizonLabel,
      closePrice,
      targetCost,
      spreadPct,
      potentialReturnPct,
      direction: 'SELL',
      severity: 'STRONG',
      badgeLabel: 'STRONG SELL (PROFIT HARVEST)',
      badgeColor: 'rose',
      playbook: 'PROFIT_HARVEST',
      rationale: `Price is extended +${spreadPct.toFixed(1)}% above institutional cost basis. Desk has substantial mark-to-market gains and will distribute into market rallies.`,
    };
  }

  if (spreadPct >= 3.0) {
    return {
      horizonCode: selectedHorizon,
      horizonLabel,
      closePrice,
      targetCost,
      spreadPct,
      potentialReturnPct,
      direction: 'SELL',
      severity: 'MODERATE',
      badgeLabel: 'MODERATE SELL (TRIMMING)',
      badgeColor: 'amber',
      playbook: 'TACTICAL_DISTRIBUTION',
      rationale: `Institution is sitting on tactical gains (+${spreadPct.toFixed(1)}%). Passive profit-taking into retail liquidity bids expected.`,
    };
  }

  return {
    horizonCode: selectedHorizon,
    horizonLabel,
    closePrice,
    targetCost,
    spreadPct,
    potentialReturnPct,
    direction: 'NEUTRAL',
    severity: 'NEUTRAL',
    badgeLabel: 'NEUTRAL (BALANCED PACING)',
    badgeColor: 'slate',
    playbook: 'NEUTRAL_PACING',
    rationale: `Price is closely balanced with institutional cost (spread ${spreadPct >= 0 ? '+' : ''}${spreadPct.toFixed(1)}%). Normal algorithmic flow with zero urgency.`,
  };
}

/**
 * Calculate multi-horizon EWMA confluence across all horizons:
 * 1W, 2W, 1M, 3M, 6M, and Core FIFO.
 *
 * Provides a holistic institutional check: assesses whether the desk
 * is underwater/in profit across all time horizons simultaneously.
 */
export function calculateEwmaConfluence(
  closePrice: number,
  tertipData: TertipHorizonsResponse | null | undefined
): EwmaConfluenceSummary | null {
  if (!tertipData || !closePrice || closePrice <= 0) return null;

  const targetCodes: { code: ForwardHorizonCode; label: string }[] = [
    { code: '1W', label: '1-Week (5d)' },
    { code: '2W', label: '2-Week (10d)' },
    { code: '1M', label: '1-Month (21d)' },
    { code: '3M', label: '3-Month (63d)' },
    { code: '6M', label: '6-Month (126d)' },
    { code: 'FIFO', label: 'Core FIFO Cost' },
  ];

  const rows: EwmaHorizonSummaryRow[] = [];

  for (const item of targetCodes) {
    let ewmaCost = 0;
    let netFlowTl = 0;
    let stance = 'STABLE_CORE_HOLD';
    let inventoryQty = 0;

    if (item.code === 'FIFO') {
      ewmaCost = tertipData.fifo_avg_cost;
      netFlowTl = tertipData.day_net_flow_tl;
      stance = tertipData.diagnostic.diagnostic_badge;
      inventoryQty = tertipData.open_stock_quantity;
    } else {
      const h = (tertipData.horizons || []).find((hItem) => hItem.code === item.code);
      if (h) {
        ewmaCost = h.ewma_unit_cost > 0 ? h.ewma_unit_cost : tertipData.fifo_avg_cost;
        netFlowTl = h.cum_net_flow_tl;
        stance = h.stance;
        inventoryQty = h.ewma_inventory_qty;
      } else {
        ewmaCost = tertipData.fifo_avg_cost;
        netFlowTl = 0;
        stance = 'CORE_HOLD';
        inventoryQty = tertipData.open_stock_quantity;
      }
    }

    if (ewmaCost <= 0) continue;

    const spreadPct = ((closePrice - ewmaCost) / ewmaCost) * 100;
    const potentialReturnPct = ((ewmaCost - closePrice) / closePrice) * 100;

    let direction: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
    let severity: 'STRONG' | 'MODERATE' | 'NEUTRAL' = 'NEUTRAL';

    if (spreadPct <= -5.0) {
      direction = 'BUY';
      severity = 'STRONG';
    } else if (spreadPct <= -3.0) {
      direction = 'BUY';
      severity = 'MODERATE';
    } else if (spreadPct >= 5.0) {
      direction = 'SELL';
      severity = 'STRONG';
    } else if (spreadPct >= 3.0) {
      direction = 'SELL';
      severity = 'MODERATE';
    }

    rows.push({
      code: item.code,
      label: item.label,
      ewmaCost,
      spreadPct,
      potentialReturnPct,
      direction,
      severity,
      netFlowTl,
      stance,
      inventoryQty,
    });
  }

  const buyCount = rows.filter((r) => r.direction === 'BUY').length;
  const sellCount = rows.filter((r) => r.direction === 'SELL').length;
  const neutralCount = rows.filter((r) => r.direction === 'NEUTRAL').length;
  const totalHorizons = rows.length;

  let overallDirection: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confluenceLabel = 'Divergent / Mixed Alignment';

  if (buyCount === totalHorizons) {
    overallDirection = 'BUY';
    confluenceLabel = `Maximum Buy Confluence (${buyCount}/${totalHorizons} Horizons)`;
  } else if (buyCount >= 4) {
    overallDirection = 'BUY';
    confluenceLabel = `Strong Buy Confluence (${buyCount}/${totalHorizons} Horizons)`;
  } else if (buyCount >= 3 && buyCount > sellCount) {
    overallDirection = 'BUY';
    confluenceLabel = `Moderate Buy Lean (${buyCount}/${totalHorizons} Horizons)`;
  } else if (sellCount === totalHorizons) {
    overallDirection = 'SELL';
    confluenceLabel = `Maximum Sell Confluence (${sellCount}/${totalHorizons} Horizons)`;
  } else if (sellCount >= 4) {
    overallDirection = 'SELL';
    confluenceLabel = `Strong Sell Confluence (${sellCount}/${totalHorizons} Horizons)`;
  } else if (sellCount >= 3 && sellCount > buyCount) {
    overallDirection = 'SELL';
    confluenceLabel = `Moderate Sell Lean (${sellCount}/${totalHorizons} Horizons)`;
  }

  return {
    buyCount,
    sellCount,
    neutralCount,
    totalHorizons,
    overallDirection,
    confluenceLabel,
    ribbonStatus: tertipData.ribbon_status || 'STABLE_CORE_HOLD',
    rows,
  };
}

