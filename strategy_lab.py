#!/usr/bin/env python3
"""
Strategy research lab for spot-market strategies.

This tool fetches historical candles from a CCXT exchange and compares
long-only strategy variants by return and risk metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import time
from typing import Iterable, List, Sequence, Tuple

import ccxt


@dataclasses.dataclass(frozen=True)
class Candle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclasses.dataclass(frozen=True)
class Trade:
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    pnl_pct: float


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    name: str
    trades: int
    win_rate_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe_like: float
    stability_score: float


def _timeframe_seconds(timeframe: str) -> int:
    timeframe = timeframe.strip().lower()
    if len(timeframe) < 2:
        raise ValueError(f"Invalid timeframe '{timeframe}'")

    amount = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    if unit == "d":
        return amount * 86400
    if unit == "w":
        return amount * 7 * 86400
    raise ValueError(f"Unsupported timeframe '{timeframe}'")


def _ema(values: Sequence[float], period: int) -> List[float | None]:
    if period <= 1:
        return [float(v) for v in values]
    out: List[float | None] = [None] * len(values)
    if len(values) < period:
        return out

    k = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] * k) + (prev * (1.0 - k))
        out[i] = prev
    return out


def _sma(values: Sequence[float], period: int) -> List[float | None]:
    out: List[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out

    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def _rolling_std(values: Sequence[float], period: int) -> List[float | None]:
    out: List[float | None] = [None] * len(values)
    if period <= 1 or len(values) < period:
        return out

    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        out[i] = statistics.pstdev(window)
    return out


def _rsi(values: Sequence[float], period: int = 14) -> List[float | None]:
    out: List[float | None] = [None] * len(values)
    if len(values) <= period:
        return out

    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    out[period] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))

    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def _signals_ema_trend(closes: Sequence[float]) -> Tuple[List[bool], List[bool]]:
    fast = _ema(closes, 20)
    slow = _ema(closes, 55)
    entries = [False] * len(closes)
    exits = [False] * len(closes)

    for i in range(1, len(closes)):
        f = fast[i]
        s = slow[i]
        pf = fast[i - 1]
        if f is None or s is None or pf is None:
            continue

        trend_up = f > s
        cross_up = closes[i] > f and closes[i - 1] <= pf
        entries[i] = trend_up and cross_up
        exits[i] = (f < s) or (closes[i] < f)
    return entries, exits


def _signals_rsi_mean_reversion(closes: Sequence[float]) -> Tuple[List[bool], List[bool]]:
    rsi = _rsi(closes, 14)
    entries = [False] * len(closes)
    exits = [False] * len(closes)
    for i, v in enumerate(rsi):
        if v is None:
            continue
        entries[i] = v < 30.0
        exits[i] = v > 55.0
    return entries, exits


def _signals_bollinger_reversion(closes: Sequence[float]) -> Tuple[List[bool], List[bool]]:
    mid = _sma(closes, 20)
    std = _rolling_std(closes, 20)
    entries = [False] * len(closes)
    exits = [False] * len(closes)

    for i in range(len(closes)):
        m = mid[i]
        s = std[i]
        if m is None or s is None:
            continue
        lower = m - (2.0 * s)
        entries[i] = closes[i] < lower
        exits[i] = closes[i] > m
    return entries, exits


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak <= 0:
            continue
        drawdown = (eq / peak) - 1.0
        max_dd = min(max_dd, drawdown)
    return max_dd


def _profit_factor(trade_pnls: Sequence[float]) -> float:
    wins = sum(p for p in trade_pnls if p > 0)
    losses = sum(-p for p in trade_pnls if p < 0)
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _sharpe_like(equity_curve: Sequence[float], periods_per_year: float) -> float:
    if len(equity_curve) < 3:
        return 0.0

    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        cur = equity_curve[i]
        if prev <= 0:
            continue
        returns.append((cur / prev) - 1.0)
    if len(returns) < 2:
        return 0.0

    mean_ret = statistics.mean(returns)
    std_ret = statistics.pstdev(returns)
    if std_ret <= 1e-12:
        return 0.0
    return (mean_ret / std_ret) * math.sqrt(periods_per_year)


def run_backtest(
    name: str,
    candles: Sequence[Candle],
    entries: Sequence[bool],
    exits: Sequence[bool],
    start_cash: float,
    fee_rate: float,
    slippage_rate: float,
    periods_per_year: float,
) -> BacktestResult:
    cash = start_cash
    base = 0.0
    in_position = False
    entry_cash = 0.0
    entry_price = 0.0
    entry_index = -1

    trades: List[Trade] = []
    equity_curve: List[float] = []

    for i, candle in enumerate(candles):
        close = candle.close
        if close <= 0:
            equity_curve.append(cash)
            continue

        if in_position and exits[i]:
            sell_price = close * (1.0 - slippage_rate)
            proceeds = base * sell_price
            fee = proceeds * fee_rate
            cash = proceeds - fee
            pnl_pct = (cash / entry_cash) - 1.0 if entry_cash > 0 else 0.0
            trades.append(
                Trade(
                    entry_index=entry_index,
                    exit_index=i,
                    entry_price=entry_price,
                    exit_price=sell_price,
                    pnl_pct=pnl_pct,
                )
            )
            base = 0.0
            in_position = False

        if (not in_position) and entries[i] and cash > 0:
            buy_price = close * (1.0 + slippage_rate)
            fee = cash * fee_rate
            net_cash = cash - fee
            qty = net_cash / buy_price
            if qty > 0:
                entry_cash = cash
                entry_price = buy_price
                entry_index = i
                base = qty
                cash = 0.0
                in_position = True

        equity = cash if not in_position else (base * close)
        equity_curve.append(equity)

    if in_position and candles:
        last_close = candles[-1].close
        sell_price = last_close * (1.0 - slippage_rate)
        proceeds = base * sell_price
        fee = proceeds * fee_rate
        cash = proceeds - fee
        pnl_pct = (cash / entry_cash) - 1.0 if entry_cash > 0 else 0.0
        trades.append(
            Trade(
                entry_index=entry_index,
                exit_index=len(candles) - 1,
                entry_price=entry_price,
                exit_price=sell_price,
                pnl_pct=pnl_pct,
            )
        )
        equity_curve[-1] = cash

    total_return = (equity_curve[-1] / start_cash) - 1.0 if equity_curve else 0.0
    mdd = _max_drawdown(equity_curve)
    trade_pnls = [t.pnl_pct for t in trades]
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = (wins / len(trade_pnls)) if trade_pnls else 0.0
    pf = _profit_factor(trade_pnls)
    sharpe_like = _sharpe_like(equity_curve, periods_per_year)
    denom = abs(mdd) if abs(mdd) > 1e-12 else 1.0
    stability_score = total_return / denom

    return BacktestResult(
        name=name,
        trades=len(trades),
        win_rate_pct=win_rate * 100.0,
        total_return_pct=total_return * 100.0,
        max_drawdown_pct=mdd * 100.0,
        profit_factor=pf,
        sharpe_like=sharpe_like,
        stability_score=stability_score,
    )


def fetch_candles(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    max_bars: int,
) -> List[Candle]:
    candles: List[Candle] = []
    cursor = since_ms
    batch_limit = min(1000, max_bars)

    while len(candles) < max_bars:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=batch_limit)
        if not batch:
            break
        for row in batch:
            if len(row) < 6:
                continue
            ts = int(row[0])
            if candles and ts <= candles[-1].timestamp_ms:
                continue
            candles.append(
                Candle(
                    timestamp_ms=ts,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
            if len(candles) >= max_bars:
                break
        cursor = candles[-1].timestamp_ms + 1
        if len(batch) < batch_limit:
            break
        time.sleep(max(exchange.rateLimit, 100) / 1000.0)
    return candles


def _print_results(results: Iterable[BacktestResult]):
    header = (
        f"{'strategy':<24}"
        f"{'trades':>8}"
        f"{'win%':>10}"
        f"{'return%':>12}"
        f"{'mdd%':>10}"
        f"{'pf':>10}"
        f"{'sharpe':>10}"
        f"{'score':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        pf_display = "inf" if math.isinf(r.profit_factor) else f"{r.profit_factor:.2f}"
        print(
            f"{r.name:<24}"
            f"{r.trades:>8d}"
            f"{r.win_rate_pct:>10.2f}"
            f"{r.total_return_pct:>12.2f}"
            f"{r.max_drawdown_pct:>10.2f}"
            f"{pf_display:>10}"
            f"{r.sharpe_like:>10.2f}"
            f"{r.stability_score:>10.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Backtest spot strategies on exchange OHLCV data")
    parser.add_argument("--exchange", default="binance", help="CCXT exchange id")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", default="15m", help="OHLCV timeframe (e.g. 5m, 15m, 1h)")
    parser.add_argument("--lookback-days", type=int, default=45, help="Historical window in days")
    parser.add_argument("--max-bars", type=int, default=3000, help="Maximum bars to fetch")
    parser.add_argument("--start-cash", type=float, default=1000.0, help="Starting balance in quote currency")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Fee per side as fraction")
    parser.add_argument("--slippage-rate", type=float, default=0.0002, help="Execution slippage per side")
    args = parser.parse_args()

    if not hasattr(ccxt, args.exchange):
        raise SystemExit(f"Unsupported exchange '{args.exchange}'")
    exchange_cls = getattr(ccxt, args.exchange)
    exchange = exchange_cls({"enableRateLimit": True})

    try:
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - (args.lookback_days * 24 * 60 * 60 * 1000)
        candles = fetch_candles(
            exchange=exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            max_bars=args.max_bars,
        )
        if len(candles) < 120:
            raise SystemExit("Not enough candles fetched for robust comparison")

        closes = [c.close for c in candles]
        bar_seconds = _timeframe_seconds(args.timeframe)
        periods_per_year = (365.0 * 24.0 * 3600.0) / bar_seconds

        strategy_defs = [
            ("ema_trend_pullback", _signals_ema_trend(closes)),
            ("rsi_mean_reversion", _signals_rsi_mean_reversion(closes)),
            ("bollinger_reversion", _signals_bollinger_reversion(closes)),
        ]

        results = []
        for name, (entries, exits) in strategy_defs:
            results.append(
                run_backtest(
                    name=name,
                    candles=candles,
                    entries=entries,
                    exits=exits,
                    start_cash=args.start_cash,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                    periods_per_year=periods_per_year,
                )
            )

        results.sort(key=lambda x: x.stability_score, reverse=True)
        print(
            f"Exchange={args.exchange} Symbol={args.symbol} Timeframe={args.timeframe} "
            f"Candles={len(candles)}"
        )
        _print_results(results)
    finally:
        try:
            exchange.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
