/** Formatting helpers for financial metrics, Turkish Lira, percentages, and badges */

export function formatTL(amount: number, compact: boolean = true): string {
  if (amount === undefined || amount === null || isNaN(amount)) return '₺0';

  if (!compact) {
    return '₺' + amount.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  const abs = Math.abs(amount);
  const sign = amount < 0 ? '-' : '';

  if (abs >= 1_000_000_000) {
    return `${sign}₺${(abs / 1_000_000_000).toFixed(2)}B`;
  }
  if (abs >= 1_000_000) {
    return `${sign}₺${(abs / 1_000_000).toFixed(2)}M`;
  }
  if (abs >= 1_000) {
    return `${sign}₺${(abs / 1_000).toFixed(1)}K`;
  }
  return `${sign}₺${abs.toFixed(2)}`;
}

export function formatVolume(vol: number): string {
  if (!vol || isNaN(vol)) return '0';
  const abs = Math.abs(vol);
  const sign = vol < 0 ? '-' : '';
  if (abs >= 1_000_000) return `${sign}${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}${(abs / 1_000).toFixed(1)}K`;
  return `${sign}${abs.toFixed(0)}`;
}

export function formatPercent(val: number, decimals: number = 2): string {
  if (val === undefined || val === null || isNaN(val)) return '0.00%';
  const prefix = val > 0 ? '+' : '';
  return `${prefix}${val.toFixed(decimals)}%`;
}

export function getBiasBadgeStyle(badge: string): { bg: string; text: string; border: string; label: string } {
  switch (badge) {
    case 'AGGRESSIVE_BUYER':
      return {
        bg: 'bg-emerald-500/20',
        text: 'text-emerald-400',
        border: 'border-emerald-500/40',
        label: 'Aggressive Buyer',
      };
    case 'MODERATE_BUYER':
      return {
        bg: 'bg-teal-500/20',
        text: 'text-teal-400',
        border: 'border-teal-500/40',
        label: 'Moderate Buyer',
      };
    case 'AGGRESSIVE_SELLER':
      return {
        bg: 'bg-rose-500/20',
        text: 'text-rose-400',
        border: 'border-rose-500/40',
        label: 'Aggressive Seller',
      };
    case 'MODERATE_SELLER':
      return {
        bg: 'bg-amber-500/20',
        text: 'text-amber-400',
        border: 'border-amber-500/40',
        label: 'Moderate Seller',
      };
    case 'NEUTRAL':
    default:
      return {
        bg: 'bg-slate-700/40',
        text: 'text-slate-300',
        border: 'border-slate-600',
        label: 'Neutral Order Flow',
      };
  }
}

export function getPlaybookStyle(playbook: string): { bg: string; text: string; border: string } {
  switch (playbook) {
    case 'MOMENTUM_EXPANSION':
    case 'SQUEEZE_LONG':
      return { bg: 'bg-emerald-950/60', text: 'text-emerald-300', border: 'border-emerald-700/60' };
    case 'LIQUIDITY_FADE':
    case 'DEFENSE_SUPPORT':
      return { bg: 'bg-cyan-950/60', text: 'text-cyan-300', border: 'border-cyan-700/60' };
    case 'SECTOR_ROTATION':
      return { bg: 'bg-purple-950/60', text: 'text-purple-300', border: 'border-purple-700/60' };
    case 'NEUTRAL_WAIT':
    default:
      return { bg: 'bg-slate-800/60', text: 'text-slate-400', border: 'border-slate-700' };
  }
}
