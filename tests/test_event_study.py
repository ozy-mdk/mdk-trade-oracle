"""Unit tests for Event Study and Forward Return Analysis."""


from mdk_trading_oracle.api.resolvers import get_event_study
from mdk_trading_oracle.api.schema import schema


def test_get_event_study_daily_return():
    """Test Event Study calculation for daily return threshold >= +3.0%."""
    result = get_event_study(
        symbol="THYAO",
        condition_type="DAILY_RETURN",
        min_value=3.0,
        forward_days=5,
        limit=20,
    )
    assert result.symbol == "THYAO"
    assert result.condition_type == "DAILY_RETURN"
    assert result.forward_days == 5
    assert result.total_occurrences > 0
    assert len(result.horizon_stats) == 5

    # Check horizon stats consistency
    for stat in result.horizon_stats:
        assert 1 <= stat.day_offset <= 5
        assert 0.0 <= stat.win_rate_pct <= 100.0
        assert stat.max_gain_pct >= stat.max_loss_pct
        assert stat.sample_count > 0

    # Check occurrences
    assert len(result.occurrences) > 0
    assert len(result.occurrences) <= 20
    first_occ = result.occurrences[0]
    assert first_occ.close_price > 0
    assert first_occ.movement_value >= 3.0
    assert len(first_occ.forward_returns) == 5


def test_get_event_study_bofa_flow():
    """Test Event Study calculation for BofA Net Flow threshold >= 100M TL."""
    result = get_event_study(
        symbol="THYAO",
        condition_type="BOFA_NET_FLOW",
        min_value=100_000_000,
        forward_days=3,
        limit=10,
    )
    assert result.symbol == "THYAO"
    assert result.condition_type == "BOFA_NET_FLOW"
    assert result.forward_days == 3
    assert result.total_occurrences > 0
    assert len(result.horizon_stats) == 3


def test_graphql_event_study_query():
    """Test GraphQL eventStudy query execution."""
    query = """
        query {
            eventStudy(symbol: "THYAO", conditionType: "DAILY_RETURN", minValue: 3.0, forwardDays: 3, limit: 5) {
                symbol
                totalOccurrences
                forwardDays
                horizonStats {
                    dayOffset
                    avgReturnPct
                    winRatePct
                }
                occurrences {
                    eventDate
                    closePrice
                    forwardReturns {
                        dayOffset
                        returnPct
                    }
                }
            }
        }
    """
    res = schema.execute_sync(query)
    assert res.errors is None
    assert res.data is not None
    event_data = res.data["eventStudy"]
    assert event_data["symbol"] == "THYAO"
    assert event_data["forwardDays"] == 3
    assert event_data["totalOccurrences"] > 0
    assert len(event_data["horizonStats"]) == 3
    assert len(event_data["occurrences"]) <= 5


def test_get_event_study_akbnk_negative_drop():
    """Test Event Study calculation for AKBNK with <= -9.0% drop."""
    result = get_event_study(
        symbol="AKBNK",
        condition_type="DAILY_RETURN",
        min_value=-9.0,
        direction="DOWN",
        forward_days=3,
        limit=10,
    )
    assert result.symbol == "AKBNK"
    assert result.total_occurrences == 13  # 13 occurrences in AKBNK with <= -9% drop from prev close
    assert len(result.occurrences) == 10  # Clamped to limit=10

    # Verify 2026-05-21 drop event details
    may_event = next(o for o in result.occurrences if o.event_date == "2026-05-21")
    assert may_event.prev_close_price == 69.10
    assert may_event.open_price == 69.10
    assert may_event.close_price == 62.20
    assert may_event.price_change_pct == -9.99

    # Check forward returns for 2026-05-21
    # T+1: 62.20 -> 63.60 (+2.25% daily)
    t1 = may_event.forward_returns[0]
    assert t1.prev_close_price == 62.20
    assert t1.close_price == 63.60
    assert t1.daily_return_pct == 2.25

    # T+2: 63.60 -> 65.40 (+2.83% independent daily return relative to T+1!)
    t2 = may_event.forward_returns[1]
    assert t2.prev_close_price == 63.60
    assert t2.close_price == 65.40
    assert t2.daily_return_pct == 2.83
    assert t2.cumulative_return_pct == 5.14


def test_get_event_study_upward_return_3_percent():
    """Test Event Study calculation for AKBNK with >= +3.0% rally (direction='UP')."""
    result = get_event_study(
        symbol="AKBNK",
        condition_type="DAILY_RETURN",
        min_value=3.0,
        direction="UP",
        forward_days=5,
        limit=20,
    )
    assert result.symbol == "AKBNK"
    assert result.total_occurrences > 0
    assert len(result.occurrences) <= 20
    assert len(result.horizon_stats) == 5

    # Verify that each occurrence has priceChangePct >= 3.0
    for occ in result.occurrences:
        assert occ.price_change_pct is not None
        assert occ.price_change_pct >= 2.99
        assert occ.close_price > 0
        assert occ.bofa_net_flow_tl is not None


