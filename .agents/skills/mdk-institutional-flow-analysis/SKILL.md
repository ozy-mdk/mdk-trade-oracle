---
name: mdk-institutional-flow-analysis
description: >-
  Domain knowledge, signal definitions, and analytical workflows for tracking BIST institutional
  order flows (specifically Bank of America / Merrill Lynch `MLB`). Use when building predictive models,
  backtesting institutional momentum strategies, or designing trading signals.
---

# MDK Institutional Flow Analysis & Signal Modeling Skill

This skill documents domain-specific metrics, institutional broker classifications, and quantitative flow signals for Borsa Istanbul (BIST).

---

## 1. Key Institutional Entities

| Broker Code | Broker Name | Tier / Category | Strategic Role |
| :--- | :--- | :--- | :--- |
| **`MLB`** | **Bank of America / Merrill Lynch** | **Foreign Institutional (Primary)** | Fast algorithmic execution, cross-border institutional flows, primary market maker |
| **`IYM`** | İş Yatırım Menkul Değerler | Domestic Major Bank | Largest domestic institutional & retail flow |
| **`YKR`** | Yapı Kredi Yatırım | Domestic Major Bank | Top tier institutional / active prop desk |
| **`AKM`** | Ak Yatırım Menkul Değerler | Domestic Major Bank | High volume domestic market participant |
| **`GRM`** | Garanti BBVA Yatırım | Domestic Major Bank | Major institutional liquidity provider |

---

## 2. Institutional Flow Metric Definitions

### A. Net Broker Flow (TL)
$$\text{Net Flow (TL)} = \sum (\text{Buy Volume} \times \text{Price}) - \sum (\text{Sell Volume} \times \text{Price})$$

### B. Volume VWAP
$$\text{Buy VWAP} = \frac{\sum (\text{Buy Price} \times \text{Buy Volume})}{\sum \text{Buy Volume}}$$

### C. Bank of America Market Volume Share
$$\text{BofA Volume Share} = \frac{\text{BofA Buy Volume} + \text{BofA Sell Volume}}{2 \times \text{Market Total Volume}}$$

### D. Rolling Institutional Flow Z-Score (20-Day)
$$Z_{20d} = \frac{\text{Net Flow}_{\text{today}} - \mu_{20d}(\text{Net Flow})}{\sigma_{20d}(\text{Net Flow})}$$
- **$Z > +2.0$**: Extreme institutional accumulation (High momentum breakout candidate)
- **$Z < -2.0$**: Extreme institutional distribution (Sell-off / liquidation pressure)

---

## 3. Querying Signals via Python / Polars

```python
from mdk_trading_oracle.core.db import DuckDBManager

db = DuckDBManager(read_only=True)
df = db.query_pl("""
    SELECT 
        trade_date,
        symbol,
        close_price,
        bofa_net_flow_tl / 1e6 AS bofa_net_flow_million_tl,
        bofa_accum_5d_tl / 1e6 AS bofa_accum_5d_million_tl,
        bofa_flow_zscore_20d
    FROM gold_institutional_daily_signals
    WHERE symbol = 'THYAO'
    ORDER BY trade_date DESC;
""")
print(df)
```
