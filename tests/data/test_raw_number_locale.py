"""Raw trade feeds arrive in Turkish and English number locales; both must parse to the same prices."""

from mdk_trading_oracle.data.bronze.ingestor import BronzeIngestor

TURKISH = "09:55:07;AKBNK;68,25;642;43.817;ELP;TAC;128\n09:55:08;AKBNK;1.068,50;2;2.137;ELP;TAC;128\n"
ENGLISH = "09:55:07;AKBNK;68.25;642;43,817;ELP;TAC;128\n09:55:08;AKBNK;1,068.50;2;2,137;ELP;TAC;128\n"


def _parse(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    meta = {"file_name": name, "trade_date": "2026-10-14"}
    return BronzeIngestor._parse_csv(None, path, meta).sort("timestamp")


def test_turkish_and_english_locales_parse_identically(tmp_path):
    tr = _parse(tmp_path, "20261013", TURKISH)
    en = _parse(tmp_path, "20261014", ENGLISH)
    assert tr["price"].to_list() == [68.25, 1068.5]
    assert en["price"].to_list() == [68.25, 1068.5]
    assert tr["volume"].to_list() == en["volume"].to_list() == [642.0, 2.0]
