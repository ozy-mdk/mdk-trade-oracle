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
