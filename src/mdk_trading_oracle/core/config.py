"""Dynamic configuration and environment settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

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
        return data.get("intraday_windows", [
            {"name": "day_start", "label": "Day Start", "start_time": "07:55:00", "end_time": "08:30:00", "order": 1},
            {"name": "morning_to_lunch", "label": "Morning to Lunch", "start_time": "08:30:00", "end_time": "11:00:00", "order": 2},
            {"name": "lunch_to_15", "label": "Lunch to 15:00", "start_time": "11:00:00", "end_time": "13:00:00", "order": 3},
            {"name": "closing_session", "label": "15:00 to Day Close", "start_time": "13:00:00", "end_time": "16:15:00", "order": 4},
        ])



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
