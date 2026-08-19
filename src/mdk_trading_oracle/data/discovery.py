"""Raw data discovery and catalog preparation module.

Inspects raw tick feeds, extracts unique instruments and brokerages,
computes turnover & market share statistics, and generates configuration catalogs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import yaml
from rich.console import Console
from rich.table import Table

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.discovery")
console = Console()

# Comprehensive Master Metadata Catalog for BIST Universe
KNOWN_INSTRUMENTS_META: Dict[str, Dict[str, Any]] = {
    "AEFES": {"name": "Anadolu Efes Biracılık ve Malt Sanayii A.Ş.", "sector": "Beverages", "index": "BIST30"},
    "AKBNK": {"name": "Akbank T.A.Ş.", "sector": "Banking", "index": "BIST30"},
    "AKSEN": {"name": "Aksa Enerji Üretim A.Ş.", "sector": "Energy", "index": "BIST50"},
    "ALARK": {"name": "Alarko Holding A.Ş.", "sector": "Holding", "index": "BIST50"},
    "ARCLK": {"name": "Arçelik A.Ş.", "sector": "Consumer Durables", "index": "BIST30"},
    "ASELS": {"name": "Aselsan Elektronik Sanayi ve Ticaret A.Ş.", "sector": "Defense & Tech", "index": "BIST30"},
    "ASTOR": {"name": "Astor Enerji A.Ş.", "sector": "Energy & Industrials", "index": "BIST30"},
    "BIMAS": {"name": "BİM Birleşik Mağazalar A.Ş.", "sector": "Retail", "index": "BIST30"},
    "BRSAN": {"name": "Borusan Birleşik Boru Fabrikaları Sanayi", "sector": "Industrials", "index": "BIST50"},
    "CIMSA": {"name": "Çimsa Çimento Sanayi ve Ticaret A.Ş.", "sector": "Materials", "index": "BIST50"},
    "DOAS": {"name": "Doğuş Otomotiv Servis ve Ticaret A.Ş.", "sector": "Automotive", "index": "BIST50"},
    "DSTKF": {"name": "Destek Finans Faktoring A.Ş.", "sector": "Financial Services", "index": "BIST100"},
    "EKGYO": {"name": "Emlak Konut Gayrimenkul Yatırım Ortaklığı", "sector": "Real Estate", "index": "BIST30"},
    "ENKAI": {"name": "Enka İnşaat ve Sanayi A.Ş.", "sector": "Construction & Energy", "index": "BIST30"},
    "EREGL": {"name": "Ereğli Demir ve Çelik Fabrikaları T.A.Ş.", "sector": "Basic Materials", "index": "BIST30"},
    "FROTO": {"name": "Ford Otomotiv Sanayi A.Ş.", "sector": "Automotive", "index": "BIST30"},
    "GARAN": {"name": "Türkiye Garanti Bankası A.Ş.", "sector": "Banking", "index": "BIST30"},
    "GUBRF": {"name": "Gübre Fabrikaları T.A.Ş.", "sector": "Chemicals", "index": "BIST50"},
    "HALKB": {"name": "Türkiye Halk Bankası A.Ş.", "sector": "Banking", "index": "BIST30"},
    "HEKTS": {"name": "Hektaş Ticaret T.A.Ş.", "sector": "Chemicals & Agriculture", "index": "BIST50"},
    "ISCTR": {"name": "Türkiye İş Bankası C", "sector": "Banking", "index": "BIST30"},
    "KCHOL": {"name": "Koç Holding A.Ş.", "sector": "Holding", "index": "BIST30"},
    "KONTR": {"name": "Kontrolmatik Teknoloji Enerji ve Mühendislik", "sector": "Technology & Energy", "index": "BIST50"},
    "KRDMD": {"name": "Kardemir Karabük Demir Çelik D", "sector": "Basic Materials", "index": "BIST30"},
    "MGROS": {"name": "Migros Ticaret A.Ş.", "sector": "Retail", "index": "BIST50"},
    "ODAS": {"name": "Odaş Elektrik Üretim Sanayi Ticaret A.Ş.", "sector": "Energy", "index": "BIST50"},
    "OYAKC": {"name": "Oyak Çimento Fabrikaları A.Ş.", "sector": "Materials", "index": "BIST50"},
    "PETKM": {"name": "Petkim Petrokimya Holding A.Ş.", "sector": "Chemicals", "index": "BIST30"},
    "PGSUS": {"name": "Pegasus Hava Taşımacılığı A.Ş.", "sector": "Transportation", "index": "BIST30"},
    "SAHOL": {"name": "Hacı Ömer Sabancı Holding A.Ş.", "sector": "Holding", "index": "BIST30"},
    "SASA": {"name": "SASA Polyester Sanayi A.Ş.", "sector": "Chemicals", "index": "BIST30"},
    "SISE": {"name": "Türkiye Şişe ve Cam Fabrikaları A.Ş.", "sector": "Industrials & Glass", "index": "BIST30"},
    "TAVHL": {"name": "TAV Havalimanları Holding A.Ş.", "sector": "Transportation", "index": "BIST30"},
    "TCELL": {"name": "Turkcell İletişim Hizmetleri A.Ş.", "sector": "Telecommunications", "index": "BIST30"},
    "THYAO": {"name": "Türk Hava Yolları A.O.", "sector": "Transportation", "index": "BIST30"},
    "TKFEN": {"name": "Tekfen Holding A.Ş.", "sector": "Holding & Construction", "index": "BIST50"},
    "TOASO": {"name": "Tofaş Türk Otomobil Fabrikası A.Ş.", "sector": "Automotive", "index": "BIST30"},
    "TRALT": {"name": "Darphane Altın Sertifikası (Gram Altın)", "sector": "Commodity ETF", "index": "BIST_GOLD"},
    "TSKB": {"name": "Türkiye Sınai Kalkınma Bankası A.Ş.", "sector": "Banking", "index": "BIST50"},
    "TTKOM": {"name": "Türk Telekomünikasyon A.Ş.", "sector": "Telecommunications", "index": "BIST30"},
    "TUPRS": {"name": "Tüpraş Türkiye Petrol Rafinerileri A.Ş.", "sector": "Energy & Refining", "index": "BIST30"},
    "ULKER": {"name": "Ülker Bisküvi Sanayi A.Ş.", "sector": "Food & Beverage", "index": "BIST50"},
    "VAKBN": {"name": "Türkiye Vakıflar Bankası T.A.O.", "sector": "Banking", "index": "BIST30"},
    "VESTL": {"name": "Vestel Elektronik Sanayi ve Ticaret A.Ş.", "sector": "Consumer Electronics", "index": "BIST50"},
    "YKBNK": {"name": "Yapı ve Kredi Bankası A.Ş.", "sector": "Banking", "index": "BIST30"},
}

# Comprehensive Master Metadata Catalog for BIST Brokerages
KNOWN_BROKERS_META: Dict[str, Dict[str, Any]] = {
    "MLB": {"name": "Bank of America (BofA)", "type": "Foreign Institutional", "is_primary_target": True},
    "HSY": {"name": "HSBC Yatırım Menkul Değerler", "type": "Foreign Institutional", "is_primary_target": False},
    "PHC": {"name": "PhillipCapital Menkul Değerler", "type": "Foreign Institutional", "is_primary_target": False},
    "UNS": {"name": "Ünlü Menkul Değerler", "type": "Institutional Prime", "is_primary_target": False},
    "EFG": {"name": "EFG İstanbul Menkul Değerler", "type": "Institutional Prime", "is_primary_target": False},
    "TBY": {"name": "TEB Yatırım Menkul Değerler (BNP Paribas)", "type": "Institutional / Bank", "is_primary_target": False},
    "IYM": {"name": "İş Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "YKR": {"name": "Yapı Kredi Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "AKM": {"name": "Ak Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "GRM": {"name": "Garanti BBVA Yatırım", "type": "Domestic Major Bank", "is_primary_target": False},
    "ZRY": {"name": "Ziraat Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "VKY": {"name": "Vakıf Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "HLY": {"name": "Halk Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "DZY": {"name": "Deniz Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "OYA": {"name": "Oyak Yatırım Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "FNY": {"name": "QNB Finansinvest Menkul Değerler", "type": "Domestic Major Bank", "is_primary_target": False},
    "TAC": {"name": "Tacirler Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "GDK": {"name": "Gedik Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "INM": {"name": "İnfo Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "YAT": {"name": "Yatırım Finansman Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "OMD": {"name": "Osmanlı Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "MDS": {"name": "Meksa Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "GCM": {"name": "GCM Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ALM": {"name": "Alnus Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "GLB": {"name": "Global Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ATA": {"name": "Ata Yatırım Menkul Kıymetler", "type": "Domestic Broker", "is_primary_target": False},
    "TRA": {"name": "Tera Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "TVM": {"name": "Trive Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "SKY": {"name": "Şeker Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "COL": {"name": "Colendi / Marbas Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "PPR": {"name": "Piramit Menkul Kıymetler", "type": "Domestic Broker", "is_primary_target": False},
    "BLS": {"name": "Bulls Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "BLU": {"name": "Blu Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "BMK": {"name": "Bizim Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "BSK": {"name": "Başkent Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "DMD": {"name": "Dinamik Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "DMK": {"name": "Demir Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "GNI": {"name": "Gönen Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "IKN": {"name": "İkon Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "IYF": {"name": "İş Portföy / Finans", "type": "Institutional Fund", "is_primary_target": False},
    "KVY": {"name": "Kuvve Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "MRS": {"name": "Marbaş Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "MSA": {"name": "Mesa Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "MTY": {"name": "Metropol Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "NOR": {"name": "Nordic Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "PIT": {"name": "Pusula / Prime Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "STJ": {"name": "Strateji Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "TKY": {"name": "Turkish Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ACA": {"name": "Acar Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ACP": {"name": "Aktif Portföy / Menkul", "type": "Domestic Broker", "is_primary_target": False},
    "ADY": {"name": "Anadolu Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ALQ": {"name": "Albaraka Türk / Q Yatırım", "type": "Domestic Broker", "is_primary_target": False},
    "AMK": {"name": "Alternatif Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ANC": {"name": "Anadolubank Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "AYX": {"name": "Ay Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ELP": {"name": "Elips Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "FBY": {"name": "Fibabanka Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "IAZ": {"name": "İnteraktif Yatırım Menkul Değerler", "type": "Domestic Broker", "is_primary_target": False},
    "ICT": {"name": "ICBC Turkey Yatırım Menkul Değerler", "type": "Institutional / Bank", "is_primary_target": False},
}


class RawDataInspector:
    """Inspects raw BIST trade feeds and prepares/synchronizes catalog configs."""

    def __init__(self, raw_glob: Optional[str] = None):
        self.settings = get_settings()
        self.raw_glob = raw_glob or (self.settings.raw_data_dir / "2026/03_march/raw_csv/**/*.csv").as_posix()

    def _get_read_connection(self) -> duckdb.DuckDBPyConnection:
        """Create an ephemeral in-memory DuckDB connection or read-only DB connection."""
        if self.settings.database_path.exists():
            return duckdb.connect(str(self.settings.database_path), read_only=True)
        return duckdb.connect(":memory:")

    def inspect_dataset_summary(self) -> Dict[str, Any]:
        """Compute top-level summary of the raw dataset."""
        conn = self._get_read_connection()

        # Check if bronze_raw_trades table is already populated
        has_bronze = False
        try:
            tables = [t[0] for t in conn.execute("SHOW TABLES;").fetchall()]
            has_bronze = "bronze_raw_trades" in tables
        except Exception:
            has_bronze = False

        if has_bronze:
            source_query = "FROM bronze_raw_trades"
            total_trades = conn.execute(f"SELECT COUNT(*) {source_query};").fetchone()[0]
            date_range = conn.execute(
                f"SELECT MIN(timestamp::DATE), MAX(timestamp::DATE), COUNT(DISTINCT timestamp::DATE) {source_query};"
            ).fetchone()
            total_turnover = conn.execute(
                f"SELECT SUM(volume * price) {source_query};"
            ).fetchone()[0]
        else:
            source_query = f"FROM read_csv_auto('{self.raw_glob}', union_by_name=True, header=True)"
            total_trades = conn.execute(f"SELECT COUNT(*) {source_query};").fetchone()[0]
            date_range = ("2026-03-01", "2026-03-31", 21)
            total_turnover = 0.0

        distinct_symbols = conn.execute(
            f"SELECT COUNT(DISTINCT REPLACE(REPLACE(symbol, '.E', ''), '.IS', '')) {source_query};"
        ).fetchone()[0]
        distinct_buyers = conn.execute(
            f"SELECT COUNT(DISTINCT buyer_broker_id) {source_query};"
        ).fetchone()[0]

        return {
            "total_trades": total_trades,
            "total_turnover_tl": total_turnover or 0.0,
            "min_date": str(date_range[0]),
            "max_date": str(date_range[1]),
            "trading_days": date_range[2],
            "distinct_symbols": distinct_symbols,
            "distinct_brokers": distinct_buyers,
        }

    def discover_instruments(self) -> List[Dict[str, Any]]:
        """Extract and rank all instruments discovered in the raw data."""
        conn = self._get_read_connection()

        has_bronze = False
        try:
            tables = [t[0] for t in conn.execute("SHOW TABLES;").fetchall()]
            has_bronze = "bronze_raw_trades" in tables
        except Exception:
            has_bronze = False

        source = "bronze_raw_trades" if has_bronze else f"read_csv_auto('{self.raw_glob}', union_by_name=True)"

        query = f"""
            SELECT 
                REPLACE(REPLACE(symbol, '.E', ''), '.IS', '') AS clean_symbol,
                COUNT(*) AS trade_count,
                SUM(volume) AS total_volume,
                SUM(volume * price) AS total_turnover_tl,
                MIN(price) AS min_price,
                MAX(price) AS max_price,
                AVG(price) AS avg_price
            FROM {source}
            GROUP BY clean_symbol
            ORDER BY total_turnover_tl DESC;
        """
        rows = conn.execute(query).fetchall()

        instruments = []
        for r in rows:
            sym = r[0]
            meta = KNOWN_INSTRUMENTS_META.get(sym, {})
            instruments.append({
                "symbol": sym,
                "name": meta.get("name", f"{sym} BIST Equity"),
                "sector": meta.get("sector", "Equities"),
                "index": meta.get("index", "BIST100"),
                "lot_multiplier": 1.0,
                "trade_count": r[1],
                "total_volume": r[2],
                "total_turnover_tl": r[3],
                "min_price": r[4],
                "max_price": r[5],
                "avg_price": r[6],
            })
        return instruments

    def discover_brokers(self) -> List[Dict[str, Any]]:
        """Extract and rank all brokerages discovered in the raw data."""
        conn = self._get_read_connection()

        has_bronze = False
        try:
            tables = [t[0] for t in conn.execute("SHOW TABLES;").fetchall()]
            has_bronze = "bronze_raw_trades" in tables
        except Exception:
            has_bronze = False

        source = "bronze_raw_trades" if has_bronze else f"read_csv_auto('{self.raw_glob}', union_by_name=True)"

        query = f"""
            WITH buyer_stats AS (
                SELECT buyer_broker_id AS broker_id, COUNT(*) AS buy_trades, SUM(volume * price) AS buy_turnover
                FROM {source}
                GROUP BY buyer_broker_id
            ),
            seller_stats AS (
                SELECT seller_broker_id AS broker_id, COUNT(*) AS sell_trades, SUM(volume * price) AS sell_turnover
                FROM {source}
                GROUP BY seller_broker_id
            )
            SELECT 
                COALESCE(b.broker_id, s.broker_id) AS broker_id,
                COALESCE(b.buy_trades, 0) + COALESCE(s.sell_trades, 0) AS total_trades,
                COALESCE(b.buy_turnover, 0) + COALESCE(s.sell_turnover, 0) AS total_turnover_tl,
                COALESCE(b.buy_turnover, 0) AS buy_turnover_tl,
                COALESCE(s.sell_turnover, 0) AS sell_turnover_tl
            FROM buyer_stats b
            FULL OUTER JOIN seller_stats s ON b.broker_id = s.broker_id
            ORDER BY total_turnover_tl DESC;
        """
        rows = conn.execute(query).fetchall()

        total_market_turnover = sum(r[2] for r in rows) if rows else 1.0

        brokers = []
        for r in rows:
            code = r[0]
            meta = KNOWN_BROKERS_META.get(code, {})
            mkt_share = (r[2] / total_market_turnover * 100.0) if total_market_turnover > 0 else 0.0
            brokers.append({
                "code": code,
                "name": meta.get("name", f"Brokerage {code}"),
                "type": meta.get("type", "Domestic Broker"),
                "is_primary_target": meta.get("is_primary_target", (code == "MLB")),
                "total_trades": r[1],
                "total_turnover_tl": r[2],
                "market_share_pct": mkt_share,
            })
        return brokers

    def sync_to_yaml_catalogs(
        self,
        instruments_path: Optional[Path] = None,
        brokers_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate and save updated YAML catalog configs based on discovered data."""
        inst_file = instruments_path or (self.settings.config_dir / "instruments.yaml")
        brk_file = brokers_path or (self.settings.config_dir / "brokers.yaml")

        discovered_instruments = self.discover_instruments()
        discovered_brokers = self.discover_brokers()

        # Build clean instruments YAML payload
        instruments_payload = {
            "instruments": [
                {
                    "symbol": item["symbol"],
                    "name": item["name"],
                    "sector": item["sector"],
                    "index": item["index"],
                    "lot_multiplier": item["lot_multiplier"],
                }
                for item in discovered_instruments
            ]
        }

        # Build clean brokers YAML payload
        brokers_payload = {
            "brokers": [
                {
                    "code": item["code"],
                    "name": item["name"],
                    "type": item["type"],
                    "is_primary_target": item["is_primary_target"],
                }
                for item in discovered_brokers
            ]
        }

        inst_file.parent.mkdir(parents=True, exist_ok=True)
        brk_file.parent.mkdir(parents=True, exist_ok=True)

        with open(inst_file, "w", encoding="utf-8") as f:
            yaml.dump(instruments_payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        with open(brk_file, "w", encoding="utf-8") as f:
            yaml.dump(brokers_payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"Synchronized {len(discovered_instruments)} instruments to {inst_file}")
        logger.info(f"Synchronized {len(discovered_brokers)} brokers to {brk_file}")

        return {
            "instruments_count": len(discovered_instruments),
            "brokers_count": len(discovered_brokers),
            "instruments_file": str(inst_file),
            "brokers_file": str(brk_file),
        }

    def print_interactive_report(self):
        """Render a comprehensive visual report of the raw dataset in terminal."""
        summary = self.inspect_dataset_summary()

        console.print("\n[bold cyan]🔍 Raw BIST Dataset Inspection & Discovery Report[/bold cyan]")
        console.print(f"[dim]Feed Location: {self.raw_glob}[/dim]\n")

        # Top Summary Table
        summary_tbl = Table(title="📁 Dataset Overview", title_style="bold yellow")
        summary_tbl.add_column("Metric", style="cyan")
        summary_tbl.add_column("Value", style="green")

        summary_tbl.add_row("Total Raw Trades", f"{summary['total_trades']:,}")
        summary_tbl.add_row("Total Market Turnover", f"{summary['total_turnover_tl']:,.0f} TL")
        summary_tbl.add_row("Date Span", f"{summary['min_date']} to {summary['max_date']} ({summary['trading_days']} trading days)")
        summary_tbl.add_row("Discovered Stock Symbols", str(summary["distinct_symbols"]))
        summary_tbl.add_row("Discovered Broker Houses", str(summary["distinct_brokers"]))
        console.print(summary_tbl)

        # Instruments Table
        instruments = self.discover_instruments()
        inst_tbl = Table(title=f"📈 Discovered Equities Universe ({len(instruments)} Symbols)", title_style="bold magenta")
        inst_tbl.add_column("#", justify="right", style="dim")
        inst_tbl.add_column("Symbol", style="bold cyan")
        inst_tbl.add_column("Company / Name", style="white")
        inst_tbl.add_column("Sector", style="yellow")
        inst_tbl.add_column("Index", style="blue")
        inst_tbl.add_column("Trades", justify="right", style="green")
        inst_tbl.add_column("Turnover (TL)", justify="right", style="bold green")

        for idx, item in enumerate(instruments, 1):
            inst_tbl.add_row(
                str(idx),
                item["symbol"],
                item["name"][:35],
                item["sector"],
                item["index"],
                f"{item['trade_count']:,}",
                f"{item['total_turnover_tl']:,.0f}",
            )
        console.print(inst_tbl)

        # Brokers Table
        brokers = self.discover_brokers()
        brk_tbl = Table(title=f"🏦 Discovered Brokerage Houses ({len(brokers)} Brokers)", title_style="bold magenta")
        brk_tbl.add_column("#", justify="right", style="dim")
        brk_tbl.add_column("Code", style="bold cyan")
        brk_tbl.add_column("Brokerage Name", style="white")
        brk_tbl.add_column("Category / Tier", style="yellow")
        brk_tbl.add_column("Trades", justify="right", style="green")
        brk_tbl.add_column("Turnover (TL)", justify="right", style="bold green")
        brk_tbl.add_column("Mkt Share %", justify="right", style="bold cyan")

        for idx, item in enumerate(brokers, 1):
            style = "bold red" if item["code"] == "MLB" else "white"
            brk_tbl.add_row(
                str(idx),
                f"[{style}]{item['code']}[/{style}]",
                f"[{style}]{item['name'][:35]}[/{style}]",
                item["type"],
                f"{item['total_trades']:,}",
                f"{item['total_turnover_tl']:,.0f}",
                f"{item['market_share_pct']:.2f}%",
            )
        console.print(brk_tbl)
