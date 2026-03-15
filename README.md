# velox

Refactored to an **adaptive single-symbol spot bot** for small accounts.

## What changed

- Removed triangular arbitrage logic.
- Added a multi-strategy engine (trend continuation, mean-reversion, breakout pullback) with risk guards.
- Added higher-timeframe trend confirmation (multi-timeframe filtering).
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
- `CONFIRM_TIMEFRAME` (default `1h`)
- `STARTING_CAPITAL_QUOTE` (default `10`)
- `MIN_VOLATILITY` (default `0.0015`)
- `MAX_VOLATILITY` (default `0.03`)
- `MIN_CONFIRM_TREND_GAP` (default `0.0005`)
- `TARGET_TAKE_PROFIT_PCT` (default `0.008`)
- `STOP_LOSS_PCT` (default `0.005`)
- `MAX_EQUITY_DRAWDOWN_PCT` (default `0.12`)
- `COOLDOWN_BARS_AFTER_EXIT` (default `2`)
- `RUN_SECONDS` (default `0`, run forever)

## Important

No strategy can guarantee stable or consistent profits. Use paper mode first and only risk funds you can afford to lose.


## Strategy research loop

Use the strategy lab to compare baseline strategies and optionally run a hybrid parameter search:

```bash
python strategy_lab.py --exchange kucoin --symbol BTC/USDT --timeframe 15m --lookback-days 90 --optimize-hybrid
```

Treat results as research only; do walk-forward validation and paper trading before any live deployment.
