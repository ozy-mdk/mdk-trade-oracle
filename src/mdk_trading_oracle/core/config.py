"""Dynamic configuration and environment settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Dynamic determination of repository root (3 levels up from src/mdk_trading_oracle/core)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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

    # Directory Paths (Relative to project root by default)
    project_root: Path = PROJECT_ROOT
    config_dir: Path = PROJECT_ROOT / "config"
    data_dir: Path = PROJECT_ROOT / "data"
    bronze_dir: Path = PROJECT_ROOT / "data" / "01_bronze"
    silver_dir: Path = PROJECT_ROOT / "data" / "02_silver"
    gold_dir: Path = PROJECT_ROOT / "data" / "03_gold"
    database_dir: Path = PROJECT_ROOT / "data" / "database"
    database_path: Path = PROJECT_ROOT / "data" / "database" / "mdk_oracle.duckdb"

    def ensure_directories(self) -> None:
        """Ensure all data storage directories exist."""
        for path in [
            self.data_dir,
            self.bronze_dir,
            self.silver_dir,
            self.gold_dir,
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
