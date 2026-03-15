import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bot import AdaptiveSpotBot, BotConfig, PositionState, SignalSnapshot


class DummyExchange:
    async def load_markets(self):
        return {}

    async def close(self):
        return None


@pytest.fixture
def make_bot(monkeypatch):
    monkeypatch.setattr(AdaptiveSpotBot, "_build_exchange", lambda self: DummyExchange())

    def _make(**kwargs):
        return AdaptiveSpotBot(BotConfig(**kwargs))

    return _make


def test_decide_buy_on_trend(make_bot):
    bot = make_bot()
    action, reason = bot._decide(
        SignalSnapshot(last_price=100.0, ema_fast=101.0, ema_slow=99.0, ema_confirm_fast=110.0, ema_confirm_slow=100.0, rsi=50.0, volatility=0.01, breakout_bias=-0.002, trend_gap=0.02, confirm_trend_gap=0.03)
    )
    assert action == "buy"
    assert reason == "trend_continuation"


def test_decide_sell_on_hard_stop(make_bot):
    bot = make_bot(stop_loss_pct=0.01)
    bot.position = PositionState(base_qty=0.1, entry_price=100.0, peak_price=102.0)

    action, reason = bot._decide(
        SignalSnapshot(last_price=98.5, ema_fast=99.0, ema_slow=100.0, ema_confirm_fast=101.0, ema_confirm_slow=100.0, rsi=45.0, volatility=0.02, breakout_bias=-0.01, trend_gap=-0.01, confirm_trend_gap=0.01)
    )
    assert action == "sell"
    assert reason == "hard_stop"


def test_adaptive_position_fraction_decreases_with_volatility(make_bot):
    bot = make_bot(max_position_fraction=0.8)
    low_vol = bot._adaptive_position_fraction(0.005)
    high_vol = bot._adaptive_position_fraction(0.05)

    assert low_vol >= high_vol
    assert 0.2 <= high_vol <= 0.8


def test_paper_enter_and_exit_updates_balances(make_bot):
    bot = make_bot(
        dry_run=True,
        starting_capital_quote=10.0,
        min_order_quote=1.0,
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    asyncio.run(bot._enter_position(price=100.0, volatility=0.01, reason="unit"))
    assert bot.position.base_qty > 0
    assert bot.paper_quote < 10.0

    asyncio.run(bot._exit_position(price=102.0, reason="unit"))
    assert bot.position.base_qty == 0
    assert bot.paper_base == 0
    assert bot.paper_quote > 10.0


def test_risk_guard_stops_after_drawdown(make_bot):
    bot = make_bot(max_equity_drawdown_pct=0.05)
    bot.best_equity = 10.0
    bot._running = True

    triggered = bot._risk_guard(9.4)
    assert triggered is True
    assert bot._running is False


def test_decide_honors_cooldown(make_bot):
    bot = make_bot(cooldown_bars_after_exit=3)
    bot.bars_since_exit = 1
    action, reason = bot._decide(
        SignalSnapshot(last_price=100.0, ema_fast=101.0, ema_slow=99.0, ema_confirm_fast=110.0, ema_confirm_slow=100.0, rsi=50.0, volatility=0.01, breakout_bias=-0.002, trend_gap=0.02, confirm_trend_gap=0.03)
    )
    assert action == "hold"
    assert reason == "cooldown"


def test_invalid_config_rejected(monkeypatch):
    monkeypatch.setattr(AdaptiveSpotBot, "_build_exchange", lambda self: DummyExchange())
    with pytest.raises(ValueError):
        AdaptiveSpotBot(BotConfig(starting_capital_quote=0))


def test_available_quote_balance_dry_run(make_bot):
    bot = make_bot(dry_run=True, starting_capital_quote=11.0)
    assert asyncio.run(bot._available_quote_balance()) == 11.0


def test_decide_breakout_pullback_entry(make_bot):
    bot = make_bot()
    action, reason = bot._decide(
        SignalSnapshot(
            last_price=100.0,
            ema_fast=101.0,
            ema_slow=100.0,
            ema_confirm_fast=110.0,
            ema_confirm_slow=107.0,
            rsi=63.0,
            volatility=0.01,
            breakout_bias=-0.0005,
            trend_gap=0.01,
            confirm_trend_gap=0.02,
        )
    )
    assert action == "buy"
    assert reason == "breakout_pullback"


def test_decide_sell_when_higher_timeframe_lost(make_bot):
    bot = make_bot()
    bot.position = PositionState(base_qty=0.1, entry_price=100.0, peak_price=101.0)

    action, reason = bot._decide(
        SignalSnapshot(
            last_price=100.8,
            ema_fast=101.0,
            ema_slow=100.0,
            ema_confirm_fast=99.0,
            ema_confirm_slow=100.0,
            rsi=45.0,
            volatility=0.01,
            breakout_bias=-0.002,
            trend_gap=0.01,
            confirm_trend_gap=-0.01,
        )
    )
    assert action == "sell"
    assert reason == "higher_timeframe_lost"
