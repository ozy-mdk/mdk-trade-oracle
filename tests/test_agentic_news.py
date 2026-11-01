"""Guards for the agentic news layer: gate modes, point-in-time mapping, append-only storage, provider behaviour."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mdk_trading_oracle.agents.news_classifier import LABEL_SCHEMA, build_user_prompt, validate_labels
from mdk_trading_oracle.agents.providers import CliProvider, ProviderError, parse_json_text
from mdk_trading_oracle.data.bronze import news
from mdk_trading_oracle.models.three_forces_gate import apply_three_forces_gate, gate_direction

_spec = importlib.util.spec_from_file_location(
    "scorer", Path(__file__).parents[1] / "scripts" / "agents" / "score_agentic_news_regimes.py"
)
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

BEARISH_FLOW = {
    "is_passive_absorption_trap": True,
    "mlb_5d_share_pct": -4.0,
    "mlb_10d_share_pct": -3.0,
    "big5_cost_spread_pct": 6.0,
    "three_forces_clash_score": -1.0,
}
BULLISH_FLOW = {
    "is_passive_absorption_trap": False,
    "mlb_5d_share_pct": 2.0,
    "kamu_5d_share_pct": 1.0,
    "big5_5d_share_pct": 1.0,
    "big5_cost_spread_pct": 0.0,
    "three_forces_clash_score": 0.5,
}
DATES = ["2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21"]


def test_flow_only_ignores_news_inputs() -> None:
    assert gate_direction({**BEARISH_FLOW, "fon_krizi_ewm5": 0.9}, "flow_only") is None
    assert gate_direction({**BEARISH_FLOW, "mlb_5d_share_pct": -5.5}, "flow_only") == "BEARISH"
    assert gate_direction({**BULLISH_FLOW, "fon_krizi_ewm5": 0.9}, "flow_only") == "BULLISH"


def test_flow_news_uses_news_when_available() -> None:
    assert gate_direction({**BEARISH_FLOW, "fon_krizi_ewm5": 0.2, "news_available": True}, "flow_news") == "BEARISH"
    assert gate_direction({**BULLISH_FLOW, "fon_krizi_ewm5": 0.2, "news_available": True}, "flow_news") is None


def test_missing_news_does_not_read_as_no_crisis() -> None:
    assert gate_direction({**BULLISH_FLOW, "news_available": False}, "flow_news") is None
    # Stored crisis values are ignored when news is flagged unavailable, and the flow-only bearish path still works.
    assert gate_direction({**BEARISH_FLOW, "fon_krizi_ewm5": 0.9, "news_available": False}, "flow_news") is None
    assert gate_direction({**BEARISH_FLOW, "mlb_5d_share_pct": -5.5, "news_available": False}, "flow_news") == "BEARISH"


def test_gate_only_touches_swing_horizons_and_off_mode_is_identity() -> None:
    row = {**BEARISH_FLOW, "mlb_5d_share_pct": -6.0}
    for h in ("1d", "3d", "30d"):
        assert apply_three_forces_gate(1.2, row, h, "flow_only") == 1.2
    assert apply_three_forces_gate(1.2, row, "10d", "off") == 1.2
    assert float(np.asarray(apply_three_forces_gate(1.2, row, "10d", "flow_only"))) <= -0.55


def test_exact_timestamp_before_close_maps_to_same_session() -> None:
    assert news.map_pubdate_to_trading_session(datetime(2026, 9, 17, 11, 47, 34), DATES, False) == "2026-09-17"


def test_after_close_maps_to_next_session() -> None:
    assert news.map_pubdate_to_trading_session(datetime(2026, 9, 17, 18, 10, 1), DATES, False) == "2026-09-18"


def test_date_only_always_shifts_to_next_session() -> None:
    assert news.map_pubdate_to_trading_session(datetime(2026, 9, 17, 10, 0, 0), DATES, True) == "2026-09-18"


def test_weekend_maps_to_monday_and_pending_returns_none() -> None:
    assert news.map_pubdate_to_trading_session(datetime(2026, 9, 19, 12, 0, 0), DATES, False) == "2026-09-21"
    assert news.map_pubdate_to_trading_session(datetime(2026, 9, 21, 19, 0, 0), DATES, False) is None


def test_articles_before_calendar_do_not_collapse_onto_first_session() -> None:
    assert news.map_pubdate_to_trading_session(datetime(2020, 5, 15, 12, 0, 0), DATES, True) is None


def test_placeholder_detection() -> None:
    assert news.is_placeholder_timestamp(datetime(2026, 9, 17, 10, 0, 0), "Thu, 17 Sep 2026 07:00:00 GMT")
    assert not news.is_placeholder_timestamp(datetime(2026, 9, 17, 11, 47, 34), "Thu, 17 Sep 2026 08:47:34 GMT")


def test_collection_mode_marks_late_retrieval_as_backfill() -> None:
    published = datetime(2026, 9, 17, 12, 0, 0)
    assert news.collection_mode(published, datetime(2026, 9, 17, 20, 0, 0), False) == "LIVE"
    assert news.collection_mode(published, datetime(2026, 10, 30, 20, 0, 0), False) == "BACKFILL"
    assert news.collection_mode(published, datetime(2026, 9, 17, 20, 0, 0), True) == "BACKFILL"


def test_headline_id_ignores_whitespace_and_case() -> None:
    assert news.headline_id("MACRO", "SPK  fon Kararı") == news.headline_id("MACRO", "spk fon kararı")
    assert news.headline_id("MACRO", "x") != news.headline_id("ASELS", "x")


def test_kap_requires_whole_word() -> None:
    assert scorer.score_single_headline("BİST 5 hisseyi tedbir kapsamına aldı", "STOCK_NEWS")["bullish_catalyst_intensity"] == 0.0
    assert scorer.score_single_headline("Aselsan KAP açıklaması yaptı", "STOCK_NEWS")["bullish_catalyst_intensity"] > 0.0


def test_geo_query_alone_does_not_score_escalation() -> None:
    assert scorer.score_single_headline("Borsa İstanbul haftayı tamamladı", "GEO_HORMUZ_ENERGY")["geo_escalation_intensity"] == 0.0
    assert scorer.score_single_headline("Hürmüz Boğazı'nda gerilim", "GEO_HORMUZ_ENERGY")["geo_escalation_intensity"] > 0.0


def test_verify_zero_leakage_rejects_future_articles() -> None:
    import pandas as pd

    bad = pd.DataFrame([{"published_at_trt": "2026-09-17 19:00:00", "effective_trade_date": "2026-09-17", "title": "x"}])
    with pytest.raises(ValueError):
        scorer.verify_zero_leakage(bad)


def test_prompt_sends_only_headline_text() -> None:
    prompt = build_user_prompt(["Aselsan\n  sözleşme imzaladı"])
    assert prompt == "Headlines:\n0: Aselsan sözleşme imzaladı"


def _label(i: int) -> dict:
    return {
        "id": i, "fon_krizi": 0.0, "bofa_sell": 0.0, "kamu_floor": 0.0, "bullish_catalyst": 0.5,
        "geo_escalation": 0.0, "geo_peace": 0.0, "is_market_wide": False, "stocks": ["ASELS"],
    }


def test_labels_require_one_valid_result_per_headline() -> None:
    assert len(validate_labels({"items": [_label(0), _label(1)]}, 2)) == 2
    with pytest.raises(ProviderError):
        validate_labels({"items": [_label(0)]}, 2)
    with pytest.raises(ProviderError):
        validate_labels({"items": [{**_label(0), "fon_krizi": 1.5}]}, 1)


SPEC = {
    "binary": "tool", "binary_fallback_globs": [], "forbid_env": ["ANTHROPIC_API_KEY"], "drop_env": [],
    "args": ["--model", "{model}", "--schema", "{schema_json}", "--system", "{system_prompt}"],
    "structured_output_path": "structured", "text_output_path": "result", "error_flag_path": "is_error",
    "usage_paths": {"output_tokens": "usage.out"}, "models": {"fast": "m1"},
}


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def _patch_run(monkeypatch: pytest.MonkeyPatch, proc: _Proc, captured: dict | None = None) -> None:
    monkeypatch.setattr("mdk_trading_oracle.agents.providers.resolve_binary", lambda spec: "/bin/tool")

    def fake_run(command, **kwargs):
        if captured is not None:
            captured.update(command=command, **kwargs)
        return proc

    monkeypatch.setattr("mdk_trading_oracle.agents.providers.subprocess.run", fake_run)


def test_provider_refuses_when_billing_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        CliProvider("p", SPEC).complete_json("s", "u", LABEL_SCHEMA, "m1", 5)


def test_provider_parses_structured_output_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured: dict = {}
    _patch_run(monkeypatch, _Proc(json.dumps({"structured": {"items": []}, "usage": {"out": 7}})), captured)
    res = CliProvider("p", SPEC).complete_json("sys", "user", {"a": 1}, "m1", 5)
    assert res.data == {"items": []} and res.usage == {"output_tokens": 7}
    assert captured["command"][:3] == ["/bin/tool", "--model", "m1"]
    assert '{"a":1}' in captured["command"] and captured["input"] == "user"
    assert "shell" not in captured


def test_provider_falls_back_to_text_for_cli_without_schema_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_run(monkeypatch, _Proc(json.dumps({"result": '```json\n{"items": []}\n```'})))
    assert CliProvider("p", SPEC).complete_json("s", "u", {}, "m1", 5).data == {"items": []}


def test_provider_surfaces_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_run(monkeypatch, _Proc("", 1, "not logged in"))
    with pytest.raises(ProviderError, match="not logged in"):
        CliProvider("p", SPEC).complete_json("s", "u", {}, "m1", 5)
    _patch_run(monkeypatch, _Proc(json.dumps({"is_error": True, "result": "limit reached"})))
    with pytest.raises(ProviderError, match="limit reached"):
        CliProvider("p", SPEC).complete_json("s", "u", {}, "m1", 5)


def test_parse_json_text_rejects_non_objects() -> None:
    with pytest.raises(ProviderError):
        parse_json_text("[1, 2]")
    with pytest.raises(ProviderError):
        parse_json_text("not json")


def test_news_regime_endpoint_contract() -> None:
    from fastapi.testclient import TestClient

    from mdk_trading_oracle.api.app import app

    res = TestClient(app).get("/api/v1/tertip/news-regime?symbol=ASELS&limit=3")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "ASELS" and data["gate_mode"] in ("off", "flow_only", "flow_news")
    assert len(data["headlines"]) <= 3
    for h in data["headlines"]:
        assert h["link"] is None or h["link"].startswith("https://")
    for g in data["gate_horizons"]:
        assert {"always_up_pct", "always_down_pct", "hit_flow_pct"} <= set(g)
