import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from strategy_lab import (
    Candle,
    _signals_ema_trend,
    _signals_rsi_mean_reversion,
    _timeframe_seconds,
    run_backtest,
    walk_forward_evaluate,
)


def _make_candles(closes):
    candles = []
    ts = 1_700_000_000_000
    for close in closes:
        candles.append(
            Candle(
                timestamp_ms=ts,
                open=float(close),
                high=float(close),
                low=float(close),
                close=float(close),
                volume=1.0,
            )
        )
        ts += 60_000
    return candles


def test_timeframe_seconds_parser():
    assert _timeframe_seconds("5m") == 300
    assert _timeframe_seconds("1h") == 3600
    assert _timeframe_seconds("1d") == 86400


def test_backtest_profitable_on_simple_uptrend():
    closes = [100 + i for i in range(30)]
    candles = _make_candles(closes)
    entries = [False] * len(candles)
    exits = [False] * len(candles)
    entries[1] = True
    exits[-1] = True

    result = run_backtest(
        name="unit",
        candles=candles,
        entries=entries,
        exits=exits,
        start_cash=1000.0,
        fee_rate=0.0005,
        slippage_rate=0.0,
        periods_per_year=365.0,
    )
    assert result.trades == 1
    assert result.total_return_pct > 0
    assert result.max_drawdown_pct <= 0


def test_ema_signal_shapes():
    closes = [100 + i * 0.2 for i in range(120)]
    entries, exits = _signals_ema_trend(closes)
    assert len(entries) == len(closes)
    assert len(exits) == len(closes)


def test_rsi_reversion_triggers_after_drop():
    closes = [100.0] * 40 + [95.0, 92.0, 89.0, 87.0, 90.0, 92.0, 94.0]
    entries, exits = _signals_rsi_mean_reversion(closes)
    assert any(entries)


def test_walk_forward_returns_consistency_metrics():
    closes = [100 + i for i in range(120)]
    candles = _make_candles(closes)
    entries = [False] * len(candles)
    exits = [False] * len(candles)
    for i in range(0, len(candles), 40):
        if i + 1 < len(candles):
            entries[i + 1] = True
        if i + 39 < len(candles):
            exits[i + 39] = True

    wf = walk_forward_evaluate(
        strategy="unit",
        candles=candles,
        entries=entries,
        exits=exits,
        split_bars=40,
        start_cash=1000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        periods_per_year=365.0,
    )

    assert wf.splits == 3
    assert wf.consistency_pct == 100.0
    assert wf.median_return_pct > 0


def test_walk_forward_requires_aligned_lengths():
    candles = _make_candles([100 + i for i in range(60)])
    entries = [False] * 59
    exits = [False] * 60

    try:
        walk_forward_evaluate(
            strategy="unit",
            candles=candles,
            entries=entries,
            exits=exits,
            split_bars=30,
            start_cash=1000.0,
            fee_rate=0.001,
            slippage_rate=0.0,
            periods_per_year=365.0,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised
