"""StockReactionOrchestrator: Loops all BIST30 stocks x windows and coordinates Gold persistence.

Symbol resolution priority (highest to lowest):
  1. symbols=[...] constructor argument  (direct API call)
  2. --symbols CLI flag                  (passed in from run_pipeline.py)
  3. config/default.yaml stock_reaction.symbols list
  4. All BIST30 stocks from bronze_instruments  (default fallback)

Usage:
  # All BIST30 (default)
  orchestrator = StockReactionOrchestrator(db=db)
  orchestrator.run_all_windows()

  # Focused session — two stocks only
  orchestrator = StockReactionOrchestrator(db=db, symbols=["AKBNK", "GARAN"])
  orchestrator.run_all_windows()

  # Single stock, single window (direct)
  orchestrator.run_symbol_window("AKBNK", "w2")
"""

from typing import Any, Dict, List, Optional

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.registry import ModelRegistry
from mdk_trading_oracle.models.stock_reaction.forecaster import (
    WINDOW_MAP,
    StockReactionForecaster,
)

logger = get_logger("mdk_oracle.models.stock_reaction.orchestrator")

# All 3 windows Model 3 operates on
ALL_WINDOWS = ["w2", "w3", "w5"]


@ModelRegistry.register("stock_reaction_orchestrator")
class StockReactionOrchestrator:
    """Orchestrates Model 3 execution across the full BIST30 symbol universe (or a configured subset).

    Runs three operations per (symbol, window) in dependency order:
      1. reconcile_performance_ledger()  — fill actuals for any outstanding forecasts
      2. forecast_next_window()          — generate T+1 live prediction
      3. backtest_all_history()          — only when explicitly requested (slow)
    """

    def __init__(
        self,
        db: Optional[DuckDBManager] = None,
        symbols: Optional[List[str]] = None,
        windows: Optional[List[str]] = None,
        lookback_months: Optional[int] = None,
        model_type: str = "auto",
        include_pymc: bool = False,
        filter_weak_regimes: Optional[bool] = None,
    ):
        self.db = db or DuckDBManager()
        self.settings = get_settings()
        self.lookback_months = lookback_months
        self.model_type = model_type
        self.include_pymc = include_pymc
        self.filter_weak_regimes = filter_weak_regimes

        # Resolve symbol universe (priority chain)
        self.symbols = self._resolve_symbols(symbols)

        # Resolve windows
        cfg = self.settings.get_model_config("stock_reaction") or {}
        self.windows = windows or cfg.get("windows", ALL_WINDOWS)
        # Normalize window keys (e.g. 'first_reaction' -> 'w2')
        self.windows = [WINDOW_MAP.get(w, (w, w))[0] for w in self.windows]

        logger.info(
            f"StockReactionOrchestrator initialized: "
            f"{len(self.symbols)} symbols × {len(self.windows)} windows "
            f"({len(self.symbols) * len(self.windows)} total forecaster runs)"
        )

    def _resolve_symbols(self, symbols_arg: Optional[List[str]]) -> List[str]:
        """Resolve symbol list following the priority chain."""
        # Priority 1: direct argument
        if symbols_arg:
            resolved = [s.upper() for s in symbols_arg]
            logger.info(f"Symbol list from direct argument: {resolved}")
            return resolved

        # Priority 2/3: config file
        cfg = self.settings.get_model_config("stock_reaction") or {}
        config_symbols = cfg.get("symbols", None)
        if config_symbols:
            resolved = [s.upper() for s in config_symbols]
            logger.info(f"Symbol list from config/default.yaml: {resolved}")
            return resolved

        # Priority 4: Dynamic BIST 30 symbols from Bronze layer (bronze_bist30_membership / bronze_instruments)
        try:
            from mdk_trading_oracle.data.bronze.ingestor import BronzeIngestor

            ingestor = BronzeIngestor(self.db)
            resolved = ingestor.get_bist30_symbols(active_only=True)
            if resolved:
                logger.info(f"Symbol list resolved from Bronze layer: {len(resolved)} stocks (active BIST30)")
                return resolved
        except Exception as exc:
            logger.warning(f"Could not load symbols from Bronze layer: {exc}")

        # Hardcoded BIST30 fallback (30 latest active constituents)
        fallback = [
            "AEFES", "AKBNK", "ASELS", "ASTOR", "BIMAS",
            "DSTKF", "EKGYO", "ENKAI", "EREGL", "FROTO",
            "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD",
            "MGROS", "PETKM", "PGSUS", "SAHOL", "SASA",
            "SISE", "TAVHL", "TCELL", "THYAO", "TOASO",
            "TRALT", "TTKOM", "TUPRS", "VAKBN", "YKBNK",
        ]
        logger.info(f"Symbol list from hardcoded BIST30 fallback: {len(fallback)} stocks")
        return fallback

    def run_symbol_window(
        self,
        symbol: str,
        window: str,
        run_backtest: bool = False,
    ) -> Dict[str, Any]:
        """Run the full forecast + reconcile pipeline for a single (symbol, window)."""
        forecaster = StockReactionForecaster(
            symbol=symbol,
            window=window,
            db=self.db,
            lookback_months=self.lookback_months,
            model_type=self.model_type,
            include_pymc=self.include_pymc,
            filter_weak_regimes=self.filter_weak_regimes,
        )
        result: Dict[str, Any] = {
            "symbol": symbol,
            "window": window,
            "reconciled": 0,
            "forecasted": False,
            "backtest_rows": 0,
            "status": "ok",
        }
        try:
            result["reconciled"] = forecaster.reconcile_performance_ledger()
            forecast_result = forecaster.forecast_next_window()
            result["forecasted"] = forecast_result is not None
            if forecast_result:
                result["predicted_direction"] = forecast_result.predicted_direction
                result["predicted_return_pct"] = forecast_result.predicted_return_pct
                result["predicted_playbook"] = forecast_result.predicted_playbook
            if run_backtest:
                result["backtest_rows"] = forecaster.backtest_all_history()
        except Exception as exc:
            logger.error(f"[{symbol}/{window}] Orchestrator error: {exc}")
            result["status"] = "error"
            result["error"] = str(exc)
        return result

    def run_all_windows(
        self,
        run_backtest: bool = False,
        symbols: Optional[List[str]] = None,
        windows: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run all (symbol, window) pairs in sequence.

        Args:
            run_backtest:  Also run full walk-forward backtests (slow, typically run once).
            symbols:       Override symbol list for this specific call.
            windows:       Override window list for this specific call.

        Returns:
            Nested dict: symbol -> window -> result metrics.
        """
        target_symbols = [s.upper() for s in symbols] if symbols else self.symbols
        target_windows = windows or self.windows

        total = len(target_symbols) * len(target_windows)
        logger.info(
            f"Starting Stock Reaction batch: "
            f"{len(target_symbols)} symbols × {len(target_windows)} windows = {total} runs"
        )

        all_results: Dict[str, Any] = {}
        success_count = 0
        error_count = 0

        for symbol in target_symbols:
            all_results[symbol] = {}
            for window in target_windows:
                result = self.run_symbol_window(symbol, window, run_backtest=run_backtest)
                all_results[symbol][window] = result
                if result["status"] == "ok":
                    success_count += 1
                else:
                    error_count += 1

        logger.info(
            f"Stock Reaction batch complete: "
            f"{success_count} OK | {error_count} errors | {total} total"
        )
        return {
            "results": all_results,
            "symbols_run": len(target_symbols),
            "windows_run": len(target_windows),
            "total_runs": total,
            "success_count": success_count,
            "error_count": error_count,
        }

    def run_backtest_all(
        self,
        symbols: Optional[List[str]] = None,
        windows: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Convenience method to run full walk-forward backtests for all (symbol, window) pairs."""
        return self.run_all_windows(run_backtest=True, symbols=symbols, windows=windows)

    def get_latest_forecasts(self, window: str = "w2") -> Any:
        """Fetch the current live forecast snapshot from the Gold forecasts table."""
        table_key = WINDOW_MAP.get(window, ("w2", "first_reaction"))[0]
        conn = self.db.get_connection()
        try:
            return conn.execute(f"""
                SELECT forecast_date, symbol, window_name,
                       predicted_return_pct, predicted_return_lower_90, predicted_return_upper_90,
                       predicted_direction, direction_confidence, predicted_playbook,
                       bofa_w1_direction, bofa_w1_net_flow_tl, bofa_w1_volume_share,
                       model_name, generated_at
                FROM gold_bofa_stock_reaction_{table_key}_forecasts
                ORDER BY ABS(predicted_return_pct) DESC;
            """).pl()
        except Exception as exc:
            logger.error(f"Could not fetch forecasts for {window}: {exc}")
            return None

    def get_performance_summary(self, window: str = "w2", symbol: Optional[str] = None) -> Any:
        """Fetch performance ledger summary (hit rate, MAE) for a given window."""
        table_key = WINDOW_MAP.get(window, ("w2", "first_reaction"))[0]
        conn = self.db.get_connection()
        sym_filter = f"AND symbol = '{symbol.upper()}'" if symbol else ""
        try:
            return conn.execute(f"""
                SELECT
                    symbol,
                    COUNT(*) AS total_forecasts,
                    SUM(CASE WHEN is_direction_hit THEN 1 ELSE 0 END) AS direction_hits,
                    ROUND(AVG(CAST(is_direction_hit AS INTEGER)) * 100, 1) AS hit_rate_pct,
                    ROUND(AVG(absolute_error_pct), 3) AS mae_pct,
                    SUM(CASE WHEN is_inside_90_ci THEN 1 ELSE 0 END) AS inside_ci_count,
                    ROUND(AVG(CAST(is_inside_90_ci AS INTEGER)) * 100, 1) AS picp_90_pct
                FROM gold_bofa_stock_reaction_{table_key}_performance
                WHERE actual_return_pct IS NOT NULL {sym_filter}
                GROUP BY symbol
                ORDER BY hit_rate_pct DESC;
            """).pl()
        except Exception as exc:
            logger.error(f"Could not fetch performance for {window}: {exc}")
            return None
