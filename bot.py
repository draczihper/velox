#!/usr/bin/env python3
"""
Triangular arbitrage bot (intra-exchange) using CCXT async support.

API key safety:
- Never hardcode secrets in this file.
- Set environment variables in your shell/profile or a local .env loader:
    export EXCHANGE_ID=binance
    export EXCHANGE_API_KEY='your_key'
    export EXCHANGE_API_SECRET='your_secret'
    export EXCHANGE_API_PASSWORD='optional_for_some_exchanges'
- Keep secrets out of Git and logs.
"""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import datetime as dt
import math
import os
import signal
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import ccxt.async_support as ccxt


@dataclasses.dataclass
class BotConfig:
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")
    api_password: str = os.getenv("EXCHANGE_API_PASSWORD", "")

    start_asset: str = os.getenv("START_ASSET", "USDT")
    start_amount: float = float(os.getenv("START_AMOUNT", "10"))

    # Fee model (default 0.1% taker/maker)
    taker_fee: float = float(os.getenv("TAKER_FEE", "0.001"))
    maker_fee: float = float(os.getenv("MAKER_FEE", "0.001"))

    min_profit_threshold: float = float(os.getenv("MIN_PROFIT_THRESHOLD", "0.002"))  # 0.2%
    max_orderbook_symbols_per_scan: int = int(os.getenv("MAX_ORDERBOOK_SYMBOLS_PER_SCAN", "240"))
    orderbook_depth_limit: int = int(os.getenv("ORDERBOOK_DEPTH_LIMIT", "25"))

    top_liquidity_count: int = int(os.getenv("TOP_LIQUIDITY_COUNT", "200"))
    top_volatility_count: int = int(os.getenv("TOP_VOLATILITY_COUNT", "120"))

    scan_interval_seconds: float = float(os.getenv("SCAN_INTERVAL_SECONDS", "6"))
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() in {"1", "true", "yes", "y"}

    max_consecutive_losses: int = 3
    csv_log_path: str = os.getenv("CSV_LOG_PATH", "arb_log.csv")


@dataclasses.dataclass(frozen=True)
class DirectedEdge:
    src: str
    dst: str
    symbol: str
    side: str  # buy/sell on `symbol`


@dataclasses.dataclass
class CycleEvaluation:
    cycle_assets: Tuple[str, str, str, str]
    edges: Tuple[DirectedEdge, DirectedEdge, DirectedEdge]
    expected_final: float
    expected_profit_pct: float
    status: str
    error: str = ""


class TriangularArbitrageBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.exchange = self._build_exchange()
        self.markets: Dict[str, dict] = {}
        self.edges_by_src: Dict[str, List[DirectedEdge]] = {}
        self.log_header = [
            "timestamp_utc",
            "mode",
            "status",
            "cycle",
            "start_asset",
            "start_amount",
            "expected_final",
            "expected_profit_pct",
            "realized_final",
            "realized_profit_pct",
            "error",
        ]
        self.consecutive_losses = 0
        self._running = True

    def _build_exchange(self):
        if not hasattr(ccxt, self.cfg.exchange_id):
            raise ValueError(f"Unsupported exchange '{self.cfg.exchange_id}' in ccxt")

        exchange_cls = getattr(ccxt, self.cfg.exchange_id)
        kwargs = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        if self.cfg.api_key:
            kwargs["apiKey"] = self.cfg.api_key
        if self.cfg.api_secret:
            kwargs["secret"] = self.cfg.api_secret
        if self.cfg.api_password:
            kwargs["password"] = self.cfg.api_password

        return exchange_cls(kwargs)

    async def initialize(self):
        self._ensure_log_file()
        self.markets = await self.exchange.load_markets()
        self._build_directed_edges()

    def _ensure_log_file(self):
        if os.path.exists(self.cfg.csv_log_path):
            return
        with open(self.cfg.csv_log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.log_header)

    def _build_directed_edges(self):
        edges_by_src: Dict[str, List[DirectedEdge]] = {}
        for symbol, market in self.markets.items():
            if not market.get("active", True):
                continue
            if market.get("spot") is False:
                continue

            base = market["base"]
            quote = market["quote"]

            # base -> quote uses SELL
            edges_by_src.setdefault(base, []).append(
                DirectedEdge(src=base, dst=quote, symbol=symbol, side="sell")
            )
            # quote -> base uses BUY
            edges_by_src.setdefault(quote, []).append(
                DirectedEdge(src=quote, dst=base, symbol=symbol, side="buy")
            )

        self.edges_by_src = edges_by_src

    async def run(self):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self.stop)
        loop.add_signal_handler(signal.SIGTERM, self.stop)

        print(f"[INFO] Bot started on {self.cfg.exchange_id}. dry_run={self.cfg.dry_run}")
        while self._running:
            try:
                await self.scan_once()
            except Exception as exc:
                print(f"[ERROR] scan failure: {exc}")
            await asyncio.sleep(self.cfg.scan_interval_seconds)

    def stop(self):
        print("[INFO] Stop signal received")
        self._running = False

    async def scan_once(self):
        tickers = await self.exchange.fetch_tickers()
        selected_assets = self._select_assets(tickers)
        cycles = self._find_candidate_cycles(selected_assets)
        if not cycles:
            return

        symbols = sorted({edge.symbol for _, edges in cycles for edge in edges})
        symbols = symbols[: self.cfg.max_orderbook_symbols_per_scan]
        orderbooks = await self._fetch_orderbooks(symbols)

        best: Optional[CycleEvaluation] = None
        for cycle_assets, edges in cycles:
            if any(e.symbol not in orderbooks for e in edges):
                continue
            ev = self._evaluate_cycle(cycle_assets, edges, orderbooks)
            self._log_attempt(ev, realized_final=None)
            if ev.status != "ok":
                continue
            if ev.expected_profit_pct < self.cfg.min_profit_threshold:
                continue
            if best is None or ev.expected_profit_pct > best.expected_profit_pct:
                best = ev

        if not best:
            return

        print(
            f"[INFO] candidate {best.cycle_assets} exp={best.expected_profit_pct*100:.3f}%"
        )
        await self._execute_cycle(best, orderbooks)

    def _select_assets(self, tickers: Dict[str, dict]) -> set[str]:
        rows = []
        for symbol, t in tickers.items():
            if "/" not in symbol:
                continue
            market = self.markets.get(symbol)
            if not market or market.get("spot") is False:
                continue

            volume = float(t.get("quoteVolume") or 0.0)
            if volume <= 0:
                continue
            pct = abs(float(t.get("percentage") or 0.0))
            base = market["base"]
            quote = market["quote"]
            rows.append((symbol, base, quote, volume, pct))

        rows.sort(key=lambda x: x[3], reverse=True)
        liquid = rows[: self.cfg.top_liquidity_count]
        liquid.sort(key=lambda x: x[4], reverse=True)
        picked = liquid[: self.cfg.top_volatility_count]

        assets = {self.cfg.start_asset}
        for _, base, quote, _, _ in picked:
            assets.add(base)
            assets.add(quote)
        return assets

    def _find_candidate_cycles(
        self, selected_assets: set[str]
    ) -> List[Tuple[Tuple[str, str, str, str], Tuple[DirectedEdge, DirectedEdge, DirectedEdge]]]:
        start = self.cfg.start_asset
        cycles = []
        seen = set()

        for e1 in self.edges_by_src.get(start, []):
            if e1.dst not in selected_assets:
                continue
            for e2 in self.edges_by_src.get(e1.dst, []):
                if e2.dst not in selected_assets or e2.dst == start:
                    continue
                for e3 in self.edges_by_src.get(e2.dst, []):
                    if e3.dst != start:
                        continue
                    if not (e1.symbol != e2.symbol != e3.symbol):
                        continue
                    cycle_assets = (start, e1.dst, e2.dst, start)
                    cycle_key = (cycle_assets, tuple(sorted([e1.symbol, e2.symbol, e3.symbol])))
                    if cycle_key in seen:
                        continue
                    seen.add(cycle_key)
                    cycles.append((cycle_assets, (e1, e2, e3)))
        return cycles

    async def _fetch_orderbooks(self, symbols: Sequence[str]) -> Dict[str, dict]:
        tasks = [self.exchange.fetch_order_book(s, limit=self.cfg.orderbook_depth_limit) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ob = {}
        for s, result in zip(symbols, results):
            if isinstance(result, Exception):
                continue
            ob[s] = result
        return ob

    def _evaluate_cycle(
        self,
        cycle_assets: Tuple[str, str, str, str],
        edges: Tuple[DirectedEdge, DirectedEdge, DirectedEdge],
        orderbooks: Dict[str, dict],
    ) -> CycleEvaluation:
        amount = self.cfg.start_amount

        for edge in edges:
            book = orderbooks.get(edge.symbol)
            if not book:
                return CycleEvaluation(cycle_assets, edges, amount, -1.0, "error", "missing orderbook")

            if edge.side == "buy":
                received = self._simulate_buy_with_quote(amount, book.get("asks") or [])
            else:
                received = self._simulate_sell_base(amount, book.get("bids") or [])

            if received is None or received <= 0:
                return CycleEvaluation(cycle_assets, edges, amount, -1.0, "error", f"insufficient depth {edge.symbol}")

            amount = received * (1.0 - self.cfg.taker_fee)

        profit_pct = (amount / self.cfg.start_amount) - 1.0
        return CycleEvaluation(cycle_assets, edges, amount, profit_pct, "ok")

    @staticmethod
    def _simulate_buy_with_quote(quote_amount: float, asks: Sequence[Sequence[float]]) -> Optional[float]:
        remaining_quote = quote_amount
        base_bought = 0.0
        for ask in asks:
            if len(ask) < 2:
                continue
            price, qty_base = float(ask[0]), float(ask[1])
            if price <= 0 or qty_base <= 0:
                continue
            max_quote_here = price * qty_base
            spend = min(remaining_quote, max_quote_here)
            base_bought += spend / price
            remaining_quote -= spend
            if remaining_quote <= 1e-12:
                break

        if remaining_quote > 1e-9:
            return None
        return base_bought

    @staticmethod
    def _simulate_sell_base(base_amount: float, bids: Sequence[Sequence[float]]) -> Optional[float]:
        remaining_base = base_amount
        quote_gained = 0.0
        for bid in bids:
            if len(bid) < 2:
                continue
            price, qty_base = float(bid[0]), float(bid[1])
            if price <= 0 or qty_base <= 0:
                continue
            sold = min(remaining_base, qty_base)
            quote_gained += sold * price
            remaining_base -= sold
            if remaining_base <= 1e-12:
                break

        if remaining_base > 1e-9:
            return None
        return quote_gained

    async def _execute_cycle(self, evaluation: CycleEvaluation, orderbooks: Dict[str, dict]):
        if self.cfg.dry_run:
            realized_final = evaluation.expected_final
            realized_profit_pct = (realized_final / self.cfg.start_amount) - 1.0
            status = "paper_executed"
            self._log_attempt(evaluation, realized_final=realized_final, status_override=status)
            self._track_loss(realized_profit_pct)
            return

        balance_ok = await self._has_sufficient_start_balance(self.cfg.start_asset, self.cfg.start_amount)
        if not balance_ok:
            self._log_attempt(evaluation, realized_final=None, status_override="rejected", error_override="insufficient wallet balance")
            return

        cur_amount = self.cfg.start_amount
        status = "executed"
        error = ""

        try:
            for edge in evaluation.edges:
                market = self.markets[edge.symbol]
                if edge.side == "sell":
                    amount = self.exchange.amount_to_precision(edge.symbol, cur_amount)
                    order = await self.exchange.create_order(edge.symbol, "market", "sell", amount)
                    cur_amount = self._extract_received_quote(order, market)
                else:
                    # Buy using quote currency budget; normalize to expected base amount via orderbook
                    est_base = self._simulate_buy_with_quote(cur_amount, (orderbooks[edge.symbol].get("asks") or []))
                    if est_base is None:
                        raise RuntimeError(f"No depth for buy {edge.symbol}")
                    amount = self.exchange.amount_to_precision(edge.symbol, est_base)
                    order = await self.exchange.create_order(edge.symbol, "market", "buy", amount)
                    cur_amount = self._extract_received_base(order, market)

                cur_amount = cur_amount * (1.0 - self.cfg.taker_fee)

            realized_final = cur_amount
            realized_profit_pct = (realized_final / self.cfg.start_amount) - 1.0
            self._log_attempt(evaluation, realized_final=realized_final, status_override=status)
            self._track_loss(realized_profit_pct)
        except Exception as exc:
            status = "failed_execution"
            error = str(exc)
            self._log_attempt(evaluation, realized_final=None, status_override=status, error_override=error)
            self._track_loss(-1.0)

    async def _has_sufficient_start_balance(self, asset: str, needed: float) -> bool:
        bal = await self.exchange.fetch_balance()
        free = float((bal.get("free") or {}).get(asset, 0.0))
        return free >= needed

    @staticmethod
    def _extract_received_quote(order: dict, market: dict) -> float:
        # prefer explicit fields from exchange response
        if order.get("cost") is not None:
            return float(order["cost"])
        average = float(order.get("average") or 0.0)
        filled = float(order.get("filled") or 0.0)
        if average > 0 and filled > 0:
            return average * filled
        raise RuntimeError(f"Cannot infer received quote from order {order.get('id')}")

    @staticmethod
    def _extract_received_base(order: dict, market: dict) -> float:
        if order.get("filled") is not None:
            return float(order["filled"])
        raise RuntimeError(f"Cannot infer received base from order {order.get('id')}")

    def _track_loss(self, profit_pct: float):
        if profit_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            print("[KILL SWITCH] 3 consecutive losses detected. Stopping bot.")
            self.stop()

    def _log_attempt(
        self,
        evaluation: CycleEvaluation,
        realized_final: Optional[float],
        status_override: Optional[str] = None,
        error_override: Optional[str] = None,
    ):
        status = status_override or evaluation.status
        err = error_override if error_override is not None else evaluation.error
        realized_profit_pct = ""
        if realized_final is not None and self.cfg.start_amount > 0:
            realized_profit_pct = (realized_final / self.cfg.start_amount) - 1.0

        row = [
            dt.datetime.utcnow().isoformat(),
            "dry_run" if self.cfg.dry_run else "live",
            status,
            " -> ".join(evaluation.cycle_assets),
            self.cfg.start_asset,
            f"{self.cfg.start_amount:.8f}",
            f"{evaluation.expected_final:.8f}",
            f"{evaluation.expected_profit_pct:.8f}",
            "" if realized_final is None else f"{realized_final:.8f}",
            "" if realized_profit_pct == "" else f"{realized_profit_pct:.8f}",
            err,
        ]

        with open(self.cfg.csv_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


async def main():
    cfg = BotConfig()
    bot = TriangularArbitrageBot(cfg)

    try:
        await bot.initialize()
        await bot.run()
    finally:
        await bot.exchange.close()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        # signal handlers differ on Windows; still run without them
        pass
    asyncio.run(main())
