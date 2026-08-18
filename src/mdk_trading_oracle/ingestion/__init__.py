"""Data ingestion interfaces and file loaders."""

from mdk_trading_oracle.ingestion.base import BaseIngestor
from mdk_trading_oracle.ingestion.file_loader import FileIngestor

__all__ = ["BaseIngestor", "FileIngestor"]
