"""Unit tests for Custom Date and Time Window Analysis terminal."""

from mdk_trading_oracle.api.resolvers import get_time_window_analysis
from mdk_trading_oracle.api.schema import schema


def test_time_window_analysis_akbnk():
    """Test custom time window analysis for AKBNK with opening and closing auctions."""
    result = get_time_window_analysis(
        symbol="AKBNK",
        broker_id="MLB",
        start_datetime="2026-09-14 09:55:00",
        end_datetime="2026-09-14 18:08:00",
        timeframe="5m",
    )
    assert result.symbol == "AKBNK"
    assert result.broker_id == "MLB"
    assert result.window_open_price == 72.0
    assert result.window_close_price == 72.55
    assert result.price_change_tl == 0.55
    assert result.price_change_pct == 0.76
    assert result.window_high_price >= result.window_low_price
    assert result.total_volume > 0
    assert result.total_turnover_tl > 0
    assert len(result.candles) > 0

    # Opening auction check (09:55 match)
    assert result.opening_auction is not None
    assert result.opening_auction.match_price == 72.0
    assert result.opening_auction.total_volume > 0
    assert result.opening_auction.broker_net_flow_tl is not None

    # Closing auction check (18:05 match)
    assert result.closing_auction is not None
    assert result.closing_auction.match_price == 72.55
    assert result.closing_auction.total_volume > 0
    assert result.closing_auction.broker_net_flow_tl > 0  # BofA net bought at close match


def test_time_window_analysis_graphql_query():
    """Test GraphQL timeWindowAnalysis query execution."""
    query = """
        query {
            timeWindowAnalysis(
                symbol: "AKBNK",
                brokerId: "MLB",
                startDatetime: "2026-09-14 09:55:00",
                endDatetime: "2026-09-14 18:08:00",
                timeframe: "5m"
            ) {
                symbol
                symbolName
                windowOpenPrice
                windowClosePrice
                priceChangeTl
                priceChangePct
                openingAuction {
                    matchTime
                    matchPrice
                    totalVolume
                    brokerNetFlowTl
                }
                closingAuction {
                    matchTime
                    matchPrice
                    totalVolume
                    brokerNetFlowTl
                }
                candles {
                    bucketStart
                    open
                    close
                    volume
                    brokerNetFlowTl
                }
            }
        }
    """
    res = schema.execute_sync(query)
    assert res.errors is None
    assert res.data is not None
    data = res.data["timeWindowAnalysis"]
    assert data["symbol"] == "AKBNK"
    assert data["windowOpenPrice"] == 72.0
    assert data["windowClosePrice"] == 72.55
    assert data["openingAuction"] is not None
    assert data["closingAuction"] is not None
    assert len(data["candles"]) > 0


def test_time_window_analysis_xu030_aggregate():
    """Test custom time window analysis for XU030 aggregate index."""
    result = get_time_window_analysis(
        symbol="XU030",
        broker_id="MLB",
        start_datetime="2026-09-14 09:55:00",
        end_datetime="2026-09-14 18:08:00",
        timeframe="5m",
    )
    assert result.symbol == "XU030"
    assert result.opening_auction is not None
    assert result.opening_auction.total_volume > 0
    assert result.closing_auction is not None
    assert result.closing_auction.total_volume > 0
