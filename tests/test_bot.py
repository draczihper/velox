import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bot import BotConfig, DirectedEdge, TriangularArbitrageBot


class DummyExchange:
    async def close(self):
        return None


@pytest.fixture
def make_bot(monkeypatch):
    monkeypatch.setattr(TriangularArbitrageBot, "_build_exchange", lambda self: DummyExchange())

    def _make(**kwargs):
        cfg = BotConfig(**kwargs)
        return TriangularArbitrageBot(cfg)

    return _make


def _edges():
    return (
        DirectedEdge(src="USDT", dst="BTC", symbol="BTC/USDT", side="buy"),
        DirectedEdge(src="BTC", dst="ETH", symbol="ETH/BTC", side="buy"),
        DirectedEdge(src="ETH", dst="USDT", symbol="ETH/USDT", side="sell"),
    )


def test_select_assets_filters_quote_and_volume(make_bot):
    bot = make_bot(
        start_asset="USDT",
        min_symbol_quote_volume=100.0,
        top_liquidity_count=10,
        top_volatility_count=10,
        allowed_quote_assets=("USDT",),
    )
    bot.markets = {
        "BTC/USDT": {"spot": True, "base": "BTC", "quote": "USDT"},
        "ETH/BTC": {"spot": True, "base": "ETH", "quote": "BTC"},
        "ADA/USDT": {"spot": True, "base": "ADA", "quote": "USDT"},
    }
    tickers = {
        "BTC/USDT": {"quoteVolume": 1000, "percentage": 1.2},
        "ETH/BTC": {"quoteVolume": 999999, "percentage": 12.0},
        "ADA/USDT": {"quoteVolume": 150, "percentage": 0.5},
    }

    selected = bot._select_assets(tickers)
    assert "USDT" in selected
    assert "BTC" in selected
    assert "ADA" in selected
    assert "ETH" not in selected


def test_find_candidate_cycles_requires_three_distinct_symbols(make_bot):
    bot = make_bot(start_asset="USDT")
    bot.edges_by_src = {
        "USDT": [DirectedEdge(src="USDT", dst="A", symbol="PAIR1", side="buy")],
        "A": [DirectedEdge(src="A", dst="B", symbol="PAIR2", side="buy")],
        "B": [
            DirectedEdge(src="B", dst="USDT", symbol="PAIR1", side="sell"),  # should be rejected
            DirectedEdge(src="B", dst="USDT", symbol="PAIR3", side="sell"),  # should be accepted
        ],
    }

    cycles = bot._find_candidate_cycles({"USDT", "A", "B"})
    assert len(cycles) == 1
    assert [edge.symbol for edge in cycles[0][1]] == ["PAIR1", "PAIR2", "PAIR3"]


def test_find_candidate_cycles_respects_intermediate_allowlist(make_bot):
    bot = make_bot(start_asset="USDT", allowed_intermediate_assets=("BTC",))
    bot.edges_by_src = {
        "USDT": [
            DirectedEdge(src="USDT", dst="BTC", symbol="BTC/USDT", side="buy"),
            DirectedEdge(src="USDT", dst="ETH", symbol="ETH/USDT", side="buy"),
        ],
        "BTC": [DirectedEdge(src="BTC", dst="ETH", symbol="ETH/BTC", side="buy")],
        "ETH": [
            DirectedEdge(src="ETH", dst="USDT", symbol="ETH/USDT", side="sell"),
            DirectedEdge(src="ETH", dst="BTC", symbol="ETH/BTC", side="sell"),
        ],
    }

    cycles = bot._find_candidate_cycles({"USDT", "BTC", "ETH"})
    assert cycles == []

    bot.cfg.allowed_intermediate_assets = ("BTC", "ETH")
    cycles = bot._find_candidate_cycles({"USDT", "BTC", "ETH"})
    assert len(cycles) == 1
    assert cycles[0][0] == ("USDT", "BTC", "ETH", "USDT")


def test_evaluate_cycle_rejects_wide_spread(make_bot):
    bot = make_bot(
        start_asset="USDT",
        start_amount=10,
        max_leg_spread_pct=0.001,
        max_leg_impact_pct=0.05,
        taker_fee=0.0,
        slippage_safety_buffer=0.0,
    )
    bot.markets = {
        "BTC/USDT": {"limits": {}},
        "ETH/BTC": {"limits": {}},
        "ETH/USDT": {"limits": {}},
    }
    orderbooks = {
        "BTC/USDT": {"bids": [[99.0, 10]], "asks": [[101.0, 10]]},
        "ETH/BTC": {"bids": [[0.5, 10]], "asks": [[0.5002, 10]]},
        "ETH/USDT": {"bids": [[200.0, 10]], "asks": [[200.1, 10]]},
    }

    ev = bot._evaluate_cycle(("USDT", "BTC", "ETH", "USDT"), _edges(), orderbooks)
    assert ev.status == "error"
    assert "spread_too_wide BTC/USDT" in ev.error


def test_evaluate_cycle_rejects_high_impact(make_bot):
    bot = make_bot(
        start_asset="USDT",
        start_amount=10,
        max_leg_spread_pct=0.01,
        max_leg_impact_pct=0.01,
        taker_fee=0.0,
        slippage_safety_buffer=0.0,
    )
    bot.markets = {
        "BTC/USDT": {"limits": {}},
        "ETH/BTC": {"limits": {}},
        "ETH/USDT": {"limits": {}},
    }
    orderbooks = {
        "BTC/USDT": {"bids": [[99.9, 100]], "asks": [[100.0, 0.01], [110.0, 100]]},
        "ETH/BTC": {"bids": [[0.5, 100]], "asks": [[0.5001, 100]]},
        "ETH/USDT": {"bids": [[250.0, 100]], "asks": [[250.1, 100]]},
    }

    ev = bot._evaluate_cycle(("USDT", "BTC", "ETH", "USDT"), _edges(), orderbooks)
    assert ev.status == "error"
    assert "impact_too_high BTC/USDT" in ev.error


def test_evaluate_cycle_respects_market_limits(make_bot):
    bot = make_bot(
        start_asset="USDT",
        start_amount=10,
        max_leg_spread_pct=0.01,
        max_leg_impact_pct=0.1,
        taker_fee=0.0,
        slippage_safety_buffer=0.0,
    )
    bot.markets = {
        "BTC/USDT": {"limits": {"cost": {"min": 50}}},
        "ETH/BTC": {"limits": {}},
        "ETH/USDT": {"limits": {}},
    }
    orderbooks = {
        "BTC/USDT": {"bids": [[99.9, 100]], "asks": [[100.0, 100]]},
        "ETH/BTC": {"bids": [[0.5, 100]], "asks": [[0.5001, 100]]},
        "ETH/USDT": {"bids": [[250.0, 100]], "asks": [[250.1, 100]]},
    }

    ev = bot._evaluate_cycle(("USDT", "BTC", "ETH", "USDT"), _edges(), orderbooks)
    assert ev.status == "error"
    assert "below_min_cost BTC/USDT" in ev.error


def test_evaluate_cycle_successful_with_fee_and_buffer(make_bot):
    bot = make_bot(
        start_asset="USDT",
        start_amount=10,
        max_leg_spread_pct=0.01,
        max_leg_impact_pct=0.01,
        taker_fee=0.001,
        slippage_safety_buffer=0.0004,
    )
    bot.markets = {
        "BTC/USDT": {"limits": {}},
        "ETH/BTC": {"limits": {}},
        "ETH/USDT": {"limits": {}},
    }
    orderbooks = {
        "BTC/USDT": {"bids": [[9.99, 100]], "asks": [[10.0, 100]]},
        "ETH/BTC": {"bids": [[0.4995, 100]], "asks": [[0.5, 100]]},
        "ETH/USDT": {"bids": [[5.2, 100]], "asks": [[5.21, 100]]},
    }

    ev = bot._evaluate_cycle(("USDT", "BTC", "ETH", "USDT"), _edges(), orderbooks)
    assert ev.status == "ok"
    assert ev.expected_profit_pct > 0
