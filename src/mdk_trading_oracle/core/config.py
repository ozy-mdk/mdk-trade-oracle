"""Dynamic configuration and environment settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamic determination of repository root (3 levels up from src/mdk_trading_oracle/core)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = Path.home() / "data" / "mdk_oracle"


class Settings(BaseSettings):
    """Application settings with environment variable fallbacks."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App meta
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    timezone: str = Field(default="Europe/Istanbul", alias="TIMEZONE")
    default_market: str = Field(default="BIST", alias="DEFAULT_MARKET")
    primary_institution: str = Field(default="MLB", alias="PRIMARY_INSTITUTION")

    # Project Directories (Inside repository)
    project_root: Path = PROJECT_ROOT
    config_dir: Path = PROJECT_ROOT / "config"
    notebooks_dir: Path = PROJECT_ROOT / "notebooks"

    # External Data Storage (Outside repository: default ~/data/mdk_oracle)
    data_dir: Path = Field(default=DEFAULT_DATA_DIR, alias="DATA_DIR")

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, v: Any) -> Path:
        """Expand user path and resolve to absolute Path."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        elif isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @property
    def raw_data_dir(self) -> Path:
        """Raw data landing zone (CSV, MySQL dumps)."""
        return self.data_dir / "00_raw_data"

    @property
    def database_dir(self) -> Path:
        """Directory for DuckDB database files."""
        return self.data_dir / "database"

    @property
    def database_path(self) -> Path:
        """Full path to DuckDB database file."""
        return self.database_dir / "mdk_oracle.duckdb"

    @property
    def duckdb_path(self) -> Path:
        """Alias for database_path."""
        return self.database_path

    def ensure_directories(self) -> None:
        """Ensure all data storage directories exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.database_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def load_yaml(self, file_name: str) -> Dict[str, Any]:
        """Load a YAML configuration file from the config directory."""
        yaml_path = self.config_dir / file_name
        if not yaml_path.exists():
            return {}
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_brokers(self) -> List[Dict[str, Any]]:
        """Get broker metadata list."""
        data = self.load_yaml("brokers.yaml")
        return data.get("brokers", [])

    def get_instruments(self) -> List[Dict[str, Any]]:
        """Get instrument metadata list."""
        data = self.load_yaml("instruments.yaml")
        return data.get("instruments", [])

    def get_default_config(self) -> Dict[str, Any]:
        """Get default application configuration."""
        return self.load_yaml("default.yaml")

    def get_intraday_windows(self) -> List[Dict[str, Any]]:
        """Get parameterized intraday time windows list."""
        data = self.get_default_config()
        return data.get(
            "intraday_windows",
            [
                {
                    "name": "day_start",
                    "label": "Day Start (09:55 - 10:30)",
                    "start_time": "09:55:00",
                    "end_time": "10:30:00",
                    "order": 1,
                },
                {
                    "name": "first_reaction",
                    "label": "First Reaction (10:30 - 11:30)",
                    "start_time": "10:30:00",
                    "end_time": "11:30:00",
                    "order": 2,
                },
                {
                    "name": "midday_followup",
                    "label": "Midday Follow-up (11:30 - 14:30)",
                    "start_time": "11:30:00",
                    "end_time": "14:30:00",
                    "order": 3,
                },
                {
                    "name": "afternoon_reaction",
                    "label": "Afternoon Reaction (14:30 - 16:00)",
                    "start_time": "14:30:00",
                    "end_time": "16:00:00",
                    "order": 4,
                },
                {
                    "name": "closing_session",
                    "label": "Closing & Auction (16:00 - 18:15)",
                    "start_time": "16:00:00",
                    "end_time": "18:15:00",
                    "order": 5,
                },
            ],
        )

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get predictive model configuration merged with defaults."""
        data = self.get_default_config()
        models_cfg = data.get("models", {})
        default_lookback = models_cfg.get("default_lookback_months", 12)
        default_eval_window = models_cfg.get("default_eval_window_days", 20)
        default_burn_in = models_cfg.get("default_burn_in_days", 5)

        model_specific = models_cfg.get(model_name, {})
        return {
            "lookback_months": model_specific.get("lookback_months", default_lookback),
            "eval_window_days": model_specific.get("eval_window_days", default_eval_window),
            "min_burn_in_days": model_specific.get("min_burn_in_days", default_burn_in),
            "model_type": model_specific.get("model_type", "auto"),
            "include_pymc_arena": model_specific.get("include_pymc_arena", False),
        }

    def get_backfill_config(self) -> Dict[str, Any]:
        """Get historical performance backfill configuration merged with defaults."""
        data = self.get_default_config()
        backfill_cfg = data.get("backfill", {})
        return {
            "default_lookback_months": backfill_cfg.get("default_lookback_months", 2),
            "default_lookback_days": backfill_cfg.get("default_lookback_days", None),
        }

    def get_features_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get feature catalog and selection configuration from features.yaml."""
        data = self.load_yaml("features.yaml")
        if model_name:
            return data.get(model_name, {})
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
