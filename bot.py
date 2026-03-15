#!/usr/bin/env python3
"""
Adaptive spot bot focused on small-account growth.

Design goals:
- Replace triangular arbitrage logic with a simpler single-symbol strategy.
- Prioritize capital protection first (drawdown and loss guards).
- Keep a clean start/stop lifecycle for automatic operation.

This code does NOT guarantee profit. Always paper trade first.
"""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import datetime as dt
import os
import signal
import sys
import time
from typing import Optional, Sequence

import ccxt.async_support as ccxt
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclasses.dataclass
class BotConfig:
    exchange_id: str = os.getenv("EXCHANGE_ID", "kucoin")
    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")
    api_password: str = os.getenv("EXCHANGE_API_PASSWORD", "")

    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    confirm_timeframe: str = os.getenv("CONFIRM_TIMEFRAME", "1h")
    lookback_bars: int = int(os.getenv("LOOKBACK_BARS", "200"))

    starting_capital_quote: float = float(os.getenv("STARTING_CAPITAL_QUOTE", "10"))
    min_order_quote: float = float(os.getenv("MIN_ORDER_QUOTE", "5"))
    max_position_fraction: float = float(os.getenv("MAX_POSITION_FRACTION", "0.95"))

    fee_rate: float = float(os.getenv("FEE_RATE", "0.001"))
    slippage_rate: float = float(os.getenv("SLIPPAGE_RATE", "0.0005"))

    target_take_profit_pct: float = float(os.getenv("TARGET_TAKE_PROFIT_PCT", "0.008"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.005"))
    trailing_stop_pct: float = float(os.getenv("TRAILING_STOP_PCT", "0.004"))

    max_equity_drawdown_pct: float = float(os.getenv("MAX_EQUITY_DRAWDOWN_PCT", "0.12"))
    max_consecutive_losses: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "4"))
    min_volatility: float = float(os.getenv("MIN_VOLATILITY", "0.0015"))
    max_volatility: float = float(os.getenv("MAX_VOLATILITY", "0.03"))
    min_confirm_trend_gap: float = float(os.getenv("MIN_CONFIRM_TREND_GAP", "0.0005"))

    scan_interval_seconds: float = float(os.getenv("SCAN_INTERVAL_SECONDS", "20"))
    cooldown_bars_after_exit: int = int(os.getenv("COOLDOWN_BARS_AFTER_EXIT", "2"))
    run_seconds: float = float(os.getenv("RUN_SECONDS", "0"))
    dry_run: bool = _env_bool("DRY_RUN", "true")
    health_check: bool = _env_bool("HEALTH_CHECK", "true")

    csv_log_path: str = os.getenv("CSV_LOG_PATH", "bot_log.csv")
    log_every_n_scans: int = int(os.getenv("LOG_EVERY_N_SCANS", "0"))
    log_every_seconds: float = float(os.getenv("LOG_EVERY_SECONDS", "0"))


@dataclasses.dataclass
class SignalSnapshot:
    last_price: float
    ema_fast: float
    ema_slow: float
    ema_confirm_fast: float
    ema_confirm_slow: float
    rsi: float
    volatility: float
    breakout_bias: float
    trend_gap: float
    confirm_trend_gap: float


@dataclasses.dataclass
class PositionState:
    base_qty: float = 0.0
    entry_price: float = 0.0
    peak_price: float = 0.0


class AdaptiveSpotBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self._validate_config()
        self.exchange = self._build_exchange()
        self.position = PositionState()
        self.paper_quote = cfg.starting_capital_quote
        self.paper_base = 0.0
        self.best_equity = cfg.starting_capital_quote
        self.consecutive_losses = 0
        self.bars_since_exit = cfg.cooldown_bars_after_exit
        self._running = True
        self._scan_count = 0
        self._last_snapshot: Optional[SignalSnapshot] = None
        self._last_equity = cfg.starting_capital_quote
        self._heartbeat_task: Optional[asyncio.Task] = None

        self.log_header = [
            "timestamp_utc",
            "mode",
            "symbol",
            "action",
            "reason",
            "price",
            "base_qty",
            "quote_spent_or_received",
            "equity_quote",
            "drawdown_pct",
        ]

    def _validate_config(self):
        if self.cfg.starting_capital_quote <= 0:
            raise ValueError("STARTING_CAPITAL_QUOTE must be > 0")
        if not (0 < self.cfg.max_position_fraction <= 1):
            raise ValueError("MAX_POSITION_FRACTION must be in (0, 1]")
        if self.cfg.min_order_quote <= 0:
            raise ValueError("MIN_ORDER_QUOTE must be > 0")
        if self.cfg.lookback_bars < 60:
            raise ValueError("LOOKBACK_BARS must be >= 60")
        if self.cfg.cooldown_bars_after_exit < 0:
            raise ValueError("COOLDOWN_BARS_AFTER_EXIT must be >= 0")
        if self.cfg.max_consecutive_losses < 1:
            raise ValueError("MAX_CONSECUTIVE_LOSSES must be >= 1")
        if self.cfg.min_volatility < 0 or self.cfg.max_volatility <= self.cfg.min_volatility:
            raise ValueError("Volatility bounds must satisfy 0 <= MIN_VOLATILITY < MAX_VOLATILITY")

    def _build_exchange(self):
        if not hasattr(ccxt, self.cfg.exchange_id):
            raise ValueError(f"Unsupported exchange '{self.cfg.exchange_id}'")

        kwargs = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        if self.cfg.api_key:
            kwargs["apiKey"] = self.cfg.api_key
        if self.cfg.api_secret:
            kwargs["secret"] = self.cfg.api_secret
        if self.cfg.api_password:
            kwargs["password"] = self.cfg.api_password

        return getattr(ccxt, self.cfg.exchange_id)(kwargs)

    async def initialize(self):
        self._ensure_log_file()
        if self.cfg.health_check:
            await self._health_check()
        else:
            await self.exchange.load_markets()
        if self.cfg.log_every_seconds > 0:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def _ensure_log_file(self):
        if os.path.exists(self.cfg.csv_log_path):
            return
        with open(self.cfg.csv_log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.log_header)

    async def _health_check(self):
        print(f"[INFO] Health check: {self.cfg.exchange_id}")
        try:
            status = await self.exchange.fetch_status()
            status_value = status.get("status") if isinstance(status, dict) else status
            print(f"[INFO] fetch_status OK: {status_value}")
        except Exception as exc:
            print(f"[WARN] fetch_status failed: {exc}")

        try:
            await self.exchange.load_markets()
            print(f"[INFO] load_markets OK: {len(self.exchange.markets)} symbols")
        except Exception as exc:
            print(f"[ERROR] load_markets failed: {exc}")
            raise

    async def _heartbeat_loop(self):
        while self._running:
            await asyncio.sleep(self.cfg.log_every_seconds)
            if not self._running:
                break
            snapshot = self._last_snapshot
            if snapshot is None:
                continue
            self._log("hold", "heartbeat", snapshot.last_price, 0.0, 0.0, self._last_equity)

    async def run(self):
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self.stop)
            loop.add_signal_handler(signal.SIGTERM, self.stop)
        except NotImplementedError:
            pass

        deadline = time.monotonic() + self.cfg.run_seconds if self.cfg.run_seconds > 0 else None

        print(f"[INFO] Running adaptive bot on {self.cfg.exchange_id} {self.cfg.symbol} dry_run={self.cfg.dry_run}")
        while self._running:
            if deadline is not None and time.monotonic() >= deadline:
                print("[INFO] Run window complete.")
                self.stop()
                break

            try:
                await self.scan_once()
            except Exception as exc:
                print(f"[ERROR] scan failure: {exc}")
            await asyncio.sleep(self.cfg.scan_interval_seconds)

    def stop(self):
        print("[INFO] Stop signal received")
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def scan_once(self):
        candles = await self.exchange.fetch_ohlcv(
            self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            limit=max(self.cfg.lookback_bars, 80),
        )
        confirm_candles = await self.exchange.fetch_ohlcv(
            self.cfg.symbol,
            timeframe=self.cfg.confirm_timeframe,
            limit=120,
        )
        if len(candles) < 60:
            return
        if len(confirm_candles) < 40:
            return

        closes = [_safe_float(row[4]) for row in candles]
        highs = [_safe_float(row[2]) for row in candles]
        lows = [_safe_float(row[3]) for row in candles]
        confirm_closes = [_safe_float(row[4]) for row in confirm_candles]
        signal_snapshot = self._build_signal(closes, highs, lows, confirm_closes)
        self.bars_since_exit += 1
        if self.position.base_qty > 0:
            self.position.peak_price = max(self.position.peak_price, signal_snapshot.last_price)

        action, reason = self._decide(signal_snapshot)
        if action == "buy":
            await self._enter_position(signal_snapshot.last_price, signal_snapshot.volatility, reason)
        elif action == "sell":
            await self._exit_position(signal_snapshot.last_price, reason)

        equity = await self._current_equity(signal_snapshot.last_price)
        kill_switch_triggered = self._risk_guard(equity)
        if kill_switch_triggered and self.position.base_qty > 0:
            await self._exit_position(signal_snapshot.last_price, "kill_switch_exit")
        self._scan_count += 1
        self._last_snapshot = signal_snapshot
        if kill_switch_triggered and self.position.base_qty > 0:
            equity = await self._current_equity(signal_snapshot.last_price)
        self._last_equity = equity
        if (
            action == "hold"
            and not kill_switch_triggered
            and self.cfg.log_every_n_scans > 0
            and (self._scan_count % self.cfg.log_every_n_scans == 0)
        ):
            self._log("hold", reason, signal_snapshot.last_price, 0.0, 0.0, equity)

    @staticmethod
    def _ema(values: Sequence[float], period: int) -> list[Optional[float]]:
        out: list[Optional[float]] = [None] * len(values)
        if period <= 1 or len(values) < period:
            return out
        seed = sum(values[:period]) / period
        out[period - 1] = seed
        k = 2.0 / (period + 1)
        prev = seed
        for i in range(period, len(values)):
            prev = (values[i] * k) + (prev * (1.0 - k))
            out[i] = prev
        return out

    @staticmethod
    def _rsi(values: Sequence[float], period: int = 14) -> list[Optional[float]]:
        out: list[Optional[float]] = [None] * len(values)
        if len(values) <= period:
            return out

        gains, losses = [], []
        for i in range(1, period + 1):
            d = values[i] - values[i - 1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        out[period] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))

        for i in range(period + 1, len(values)):
            d = values[i] - values[i - 1]
            gain = max(d, 0.0)
            loss = max(-d, 0.0)
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            out[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
        return out

    @staticmethod
    def _atr_like(closes: Sequence[float], highs: Sequence[float], lows: Sequence[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        true_ranges = []
        start = len(closes) - period
        for i in range(start, len(closes)):
            prev_close = closes[i - 1]
            tr = max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
            true_ranges.append(max(tr, 0.0))
        mean_close = sum(closes[-period:]) / period
        if mean_close <= 0:
            return 0.0
        return (sum(true_ranges) / period) / mean_close

    def _build_signal(
        self,
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        confirm_closes: Sequence[float],
    ) -> SignalSnapshot:
        ema_fast = self._ema(closes, 12)[-1] or closes[-1]
        ema_slow = self._ema(closes, 34)[-1] or closes[-1]
        ema_confirm_fast = self._ema(confirm_closes, 12)[-1] or confirm_closes[-1]
        ema_confirm_slow = self._ema(confirm_closes, 34)[-1] or confirm_closes[-1]
        rsi = self._rsi(closes, 14)[-1] or 50.0
        vol = self._atr_like(closes, highs, lows, period=14)
        range_high = max(closes[-20:]) if len(closes) >= 20 else max(closes)
        breakout_bias = (closes[-1] / range_high) - 1.0 if range_high > 0 else 0.0
        trend_gap = ((ema_fast / ema_slow) - 1.0) if ema_slow > 0 else 0.0
        confirm_trend_gap = ((ema_confirm_fast / ema_confirm_slow) - 1.0) if ema_confirm_slow > 0 else 0.0
        return SignalSnapshot(
            last_price=closes[-1],
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_confirm_fast=ema_confirm_fast,
            ema_confirm_slow=ema_confirm_slow,
            rsi=rsi,
            volatility=vol,
            breakout_bias=breakout_bias,
            trend_gap=trend_gap,
            confirm_trend_gap=confirm_trend_gap,
        )

    def _decide(self, s: SignalSnapshot) -> tuple[str, str]:
        trend_up = s.ema_fast > s.ema_slow
        confirm_trend_up = s.ema_confirm_fast > s.ema_confirm_slow
        oversold = s.rsi < 38.0
        overbought = s.rsi > 67.0
        breakout_ready = s.breakout_bias >= -0.001 and s.rsi < 74.0
        regime_ok = self.cfg.min_volatility <= s.volatility <= self.cfg.max_volatility
        confirm_strength_ok = s.confirm_trend_gap >= self.cfg.min_confirm_trend_gap

        if self.position.base_qty <= 0:
            if self.bars_since_exit < self.cfg.cooldown_bars_after_exit:
                return "hold", "cooldown"
            if trend_up and confirm_trend_up and confirm_strength_ok and regime_ok and s.rsi < 62.0:
                return "buy", "trend_continuation"
            if oversold and confirm_trend_up and regime_ok and s.volatility < 0.02:
                return "buy", "mean_reversion"
            if trend_up and confirm_trend_up and confirm_strength_ok and regime_ok and breakout_ready:
                return "buy", "breakout_pullback"
            return "hold", "no_entry"

        stop_price = self.position.entry_price * (1.0 - self.cfg.stop_loss_pct)
        take_price = self.position.entry_price * (1.0 + self.cfg.target_take_profit_pct)
        trailing_floor = self.position.peak_price * (1.0 - self.cfg.trailing_stop_pct)

        if s.volatility >= self.cfg.max_volatility * 1.5:
            return "sell", "volatility_spike_protect"
        if s.last_price <= stop_price:
            return "sell", "hard_stop"
        if s.last_price <= trailing_floor and self.position.peak_price > self.position.entry_price:
            return "sell", "trailing_stop"
        if s.last_price >= take_price and overbought:
            return "sell", "take_profit"
        if not trend_up and s.rsi > 52.0:
            return "sell", "trend_lost"
        if not confirm_trend_up:
            return "sell", "higher_timeframe_lost"
        return "hold", "manage_open_position"

    async def _current_equity(self, mark_price: float) -> float:
        if self.cfg.dry_run:
            return self.paper_quote + (self.paper_base * mark_price)

        base_asset, quote_asset = self.cfg.symbol.split("/")
        bal = await self.exchange.fetch_balance()
        free_base = _safe_float((bal.get("free") or {}).get(base_asset, 0.0))
        free_quote = _safe_float((bal.get("free") or {}).get(quote_asset, 0.0))
        return free_quote + (free_base * mark_price)

    async def _available_quote_balance(self) -> float:
        if self.cfg.dry_run:
            return self.paper_quote
        _, quote_asset = self.cfg.symbol.split("/")
        bal = await self.exchange.fetch_balance()
        return _safe_float((bal.get("free") or {}).get(quote_asset, 0.0))

    def _adaptive_position_fraction(self, volatility: float) -> float:
        if volatility <= 0.0:
            return min(self.cfg.max_position_fraction, 0.5)
        vol_scalar = max(0.25, min(1.0, 0.015 / volatility))
        return max(0.2, min(self.cfg.max_position_fraction, 0.45 * vol_scalar))

    async def _enter_position(self, price: float, volatility: float, reason: str):
        if self.position.base_qty > 0:
            return

        fraction = self._adaptive_position_fraction(volatility=volatility)
        if self.cfg.dry_run:
            spend_quote = self.paper_quote * fraction
            if spend_quote < self.cfg.min_order_quote:
                return
            buy_price = price * (1.0 + self.cfg.slippage_rate)
            fee = spend_quote * self.cfg.fee_rate
            net_quote = spend_quote - fee
            qty = net_quote / buy_price
            if qty <= 0:
                return

            self.paper_quote -= spend_quote
            self.paper_base += qty
            self.position.base_qty = qty
            self.position.entry_price = buy_price
            self.position.peak_price = buy_price
            equity = self.paper_quote + (self.paper_base * price)
            self._log("buy", reason, price, qty, -spend_quote, equity)
            return

        ticker = await self.exchange.fetch_ticker(self.cfg.symbol)
        ask = _safe_float(ticker.get("ask"), default=price)
        spend_quote = (await self._available_quote_balance()) * fraction
        if spend_quote < self.cfg.min_order_quote:
            return

        qty = spend_quote / max(ask, 1e-12)
        qty = float(self.exchange.amount_to_precision(self.cfg.symbol, qty))
        if qty <= 0:
            return

        order = await self.exchange.create_order(self.cfg.symbol, "market", "buy", qty)
        filled = _safe_float(order.get("filled"), qty)
        avg = _safe_float(order.get("average"), ask)
        self.position.base_qty = filled
        self.position.entry_price = avg
        self.position.peak_price = avg
        equity = await self._current_equity(price)
        self._log("buy", reason, price, filled, -spend_quote, equity)

    async def _exit_position(self, price: float, reason: str):
        if self.position.base_qty <= 0:
            return

        if self.cfg.dry_run:
            qty = self.position.base_qty
            sell_price = price * (1.0 - self.cfg.slippage_rate)
            gross = qty * sell_price
            fee = gross * self.cfg.fee_rate
            net = gross - fee

            self.paper_base -= qty
            self.paper_quote += net
            pnl_pct = (sell_price / self.position.entry_price) - 1.0 if self.position.entry_price > 0 else 0.0
            self._update_loss_counter(pnl_pct)

            self.position = PositionState()
            self.bars_since_exit = 0
            equity = self.paper_quote
            self._log("sell", reason, price, qty, net, equity)
            return

        qty = float(self.exchange.amount_to_precision(self.cfg.symbol, self.position.base_qty))
        if qty <= 0:
            return
        order = await self.exchange.create_order(self.cfg.symbol, "market", "sell", qty)
        avg = _safe_float(order.get("average"), price)
        pnl_pct = (avg / self.position.entry_price) - 1.0 if self.position.entry_price > 0 else 0.0
        self._update_loss_counter(pnl_pct)
        self.position = PositionState()
        self.bars_since_exit = 0

        proceeds = _safe_float(order.get("cost"), qty * avg)
        equity = await self._current_equity(price)
        self._log("sell", reason, price, qty, proceeds, equity)

    def _update_loss_counter(self, pnl_pct: float):
        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def _risk_guard(self, equity: float) -> bool:
        self.best_equity = max(self.best_equity, equity)
        drawdown = 0.0
        if self.best_equity > 0:
            drawdown = 1.0 - (equity / self.best_equity)

        if drawdown >= self.cfg.max_equity_drawdown_pct:
            print("[KILL SWITCH] Max drawdown reached. Stopping bot.")
            self.stop()
            return True
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            print("[KILL SWITCH] Too many consecutive losses. Stopping bot.")
            self.stop()
            return True
        return False

    def _log(self, action: str, reason: str, price: float, qty: float, quote_flow: float, equity: float):
        drawdown_pct = 0.0 if self.best_equity <= 0 else (1.0 - (equity / self.best_equity)) * 100.0
        row = [
            dt.datetime.now(dt.timezone.utc).isoformat(),
            "dry_run" if self.cfg.dry_run else "live",
            self.cfg.symbol,
            action,
            reason,
            f"{price:.8f}",
            f"{qty:.8f}",
            f"{quote_flow:.8f}",
            f"{equity:.8f}",
            f"{drawdown_pct:.5f}",
        ]
        with open(self.cfg.csv_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    async def shutdown(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None


async def main():
    cfg = BotConfig()
    bot = AdaptiveSpotBot(cfg)
    try:
        await bot.initialize()
        await bot.run()
    finally:
        await bot.shutdown()
        await bot.exchange.close()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        pass
    asyncio.run(main())
