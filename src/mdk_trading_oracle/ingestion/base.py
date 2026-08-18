"""Base abstract interface for data ingestion."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from mdk_trading_oracle.core.db import DuckDBManager


class BaseIngestor(ABC):
    """Abstract base class for all data ingestors."""

    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager

    @abstractmethod
    def ingest(self, source_path: Any, **kwargs) -> Dict[str, Any]:
        """Ingest data into Bronze layer and return metadata summary."""
