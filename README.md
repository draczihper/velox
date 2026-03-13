# velox

Refactored to an **adaptive single-symbol spot bot** for small accounts.

## What changed

- Removed triangular arbitrage logic.
- Added a momentum + mean-reversion hybrid strategy with risk guards.
- Added position sizing for low-capital accounts (default starting balance: 10 quote units).
- Kept start/stop controls with signal handling and run windows.
- Added basic config validation and live sizing based on available quote balance.

## Run

```bash
pip install -r requirements.txt
DRY_RUN=true STARTING_CAPITAL_QUOTE=10 SYMBOL=BTC/USDT python bot.py
```

## Key env vars

- `DRY_RUN` (default `true`)
- `SYMBOL` (default `BTC/USDT`)
- `TIMEFRAME` (default `5m`)
- `STARTING_CAPITAL_QUOTE` (default `10`)
- `TARGET_TAKE_PROFIT_PCT` (default `0.008`)
- `STOP_LOSS_PCT` (default `0.005`)
- `MAX_EQUITY_DRAWDOWN_PCT` (default `0.12`)
- `COOLDOWN_BARS_AFTER_EXIT` (default `2`)
- `RUN_SECONDS` (default `0`, run forever)

## Important

No strategy can guarantee stable or consistent profits. Use paper mode first and only risk funds you can afford to lose.
