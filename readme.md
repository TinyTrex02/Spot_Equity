# Spot Equity - MA Trend Following Strategy

A simple spot portfolio backtester using Moving Average trend following.

## Strategy

- **LONG** when Close > MA (fully invested)
- **CASH/ALT ASSET** when Close < MA (hold cash or alternative asset like Gold)

No leverage, no shorting - just long or cash.

## Features

- Configurable MA type (SMA/EMA), period, and timeframe
- Optional alternative asset holding during idle periods (e.g., Gold)
- Trading fees simulation
- Buy & Hold comparison with risk metrics (Sharpe, Sortino, Calmar, Max DD)
- Anti-forward bias implementation
- Performance dashboard generation

## Usage

```bash
python3 spot.py --data_file "Data/NIFTY.csv" --symbol NIFTY --output Results
```

## Configuration

Edit `DEFAULT_CONFIG` in `spot.py` to adjust:
- MA period/type/timeframe
- Trading fees
- Alternative asset settings
- Backtest date range
