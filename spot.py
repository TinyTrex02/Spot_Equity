#!/usr/bin/env python3
"""
Moving Average Trend Following Strategy - SPOT PORTFOLIO VERSION

This script backtests a trend-following strategy as a spot portfolio:
- LONG (fully invested) when price closes above the Moving Average (SMA or EMA)
- CASH (0% invested) when price closes below the Moving Average
- No leverage, no shorting - just long or cash

Entry: Close > MA → Buy with full capital
Exit: Close < MA → Sell entire position, go to cash

ANTI-FORWARD BIAS IMPLEMENTATION:
- Processes data strictly chronologically
- All decisions use only data available up to the current bar
- MA calculated using only historical data
- No data leakage or forward-looking decisions allowed

Usage:
    python ma_trend_spot_portfolio.py --data_file path/to/data.csv --symbol BTCUSDT --output /path/to/output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import argparse
import sys
import traceback

# Run your code using this below : 

# use YF for data 

# python3 /Users/rexy/Desktop/EthResearch/Strategies/spx_spot.py --data_file "/Users/rexy/Desktop/EthResearch/Data/NIFTY.csv" --symbol NIFTY --output /Users/rexy/Desktop/EthResearch/Results


# ==================== CONFIGURATION SECTION ====================
# Edit these values directly - they will be used when you run the script

DEFAULT_CONFIG = {
    'initial_capital': 100000,          # Starting capital ($)

    # Moving Average Configuration
    'ma_type': 'EMA',                   # 'SMA' or 'EMA'
    'ma_period': 80,                   # Moving average period
    'ma_timeframe': '1D',               # Timeframe for MA: '1H', '4H', '1D'

    # Trading Fees Configuration
    'use_trading_fees': False,           # Enable/disable trading fees
    'entry_fee_percentage': 0.04,       # Entry fee % (maker/taker)
    'exit_fee_percentage': 0.04,        # Exit fee %

    # Backtest Period
    'start_date': '2010-01-01',         # Start date for backtest
    'end_date': None,                   # End date (None = latest available)

    # Warm-up period for indicator calculation (days)
    'warmup_days': 30,

    # Alternative Asset During Idle (e.g., Gold when not in SPX)
    'hold_alt_asset_when_idle': True,   # Enable/disable holding alt asset during idle
    'alt_asset_file': '/Users/rexy/Desktop/EthResearch/Data/GOLDBEES.csv',  # Path to alt asset CSV
    'alt_asset_symbol': 'GOLD',         # Symbol name for alt asset (for display)
    'alt_asset_allocation_pct': 100,    # % of portfolio to allocate to alt asset (0-100)
}

# ==================== END CONFIGURATION SECTION ====================


def calculate_moving_average(df, period=50, ma_type='EMA', timeframe='1D'):
    """
    Calculate Moving Average (SMA or EMA)
    
    Args:
        df: DataFrame with OHLC data (must be sorted chronologically)
        period: MA period
        ma_type: 'SMA' or 'EMA'
        timeframe: Timeframe for resampling ('1H', '4H', '1D')
        
    Returns:
        Series with MA values aligned to original index
    """
    try:
        # Resample to desired timeframe
        if timeframe != 'original':
            df_resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        else:
            df_resampled = df.copy()
        
        # Calculate MA
        if ma_type.upper() == 'SMA':
            ma = df_resampled['close'].rolling(window=period).mean()
        elif ma_type.upper() == 'EMA':
            ma = df_resampled['close'].ewm(span=period, adjust=False).mean()
        else:
            raise ValueError(f"Invalid MA type: {ma_type}. Use 'SMA' or 'EMA'")
        
        # Forward fill MA values to align with original index
        ma_aligned = ma.reindex(df.index, method='ffill')
        
        return ma_aligned
        
    except Exception as e:
        print(f"Error calculating Moving Average: {e}")
        return pd.Series(index=df.index, data=np.nan)


def load_and_prepare_data(data_file, start_date='2021-01-01', end_date=None):
    """
    Load and prepare price data for backtesting
    
    Args:
        data_file: Path to CSV file (OHLCV data)
        start_date: Start date for backtest
        end_date: End date for backtest (None = latest)
        
    Returns:
        df: DataFrame with OHLC data
    """
    print(f"\n{'='*80}")
    print("LOADING AND PREPARING DATA")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    # Create datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.set_index('datetime', inplace=True)
    elif 'Date' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'])
        df.set_index('datetime', inplace=True)
    else:
        raise ValueError("CSV must have 'datetime' column or 'date' and 'time' columns or 'Date' column")
    
    # Sort chronologically
    df = df.sort_index()
    
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    
    print(f"Data loaded: {len(df):,} rows from {df.index.min()} to {df.index.max()}")
    
    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    if end_date:
        end_dt = pd.to_datetime(end_date)
    else:
        end_dt = df.index.max()
    
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    print(f"\nFiltered data from {start_dt.date()} to {end_dt.date()}")
    print(f"Rows after filtering: {len(df):,}")
    print(f"{'='*80}\n")
    
    return df


def backtest_spot_portfolio(
    df,
    symbol='BTCUSDT',
    initial_capital=100000,
    ma_type='EMA',
    ma_period=50,
    ma_timeframe='1D',
    use_trading_fees=True,
    entry_fee_percentage=0.04,
    exit_fee_percentage=0.04,
    warmup_days=60,
    hold_alt_asset_when_idle=False,
    alt_asset_file=None,
    alt_asset_symbol='GOLD',
    alt_asset_allocation_pct=100
):
    """
    Backtest the Moving Average Trend Following Strategy as SPOT PORTFOLIO
    
    Strategy:
    - Go LONG (fully invested) when close > MA
    - Go to CASH when close < MA
    - No leverage, no shorting
    
    Anti-Forward Bias Implementation:
    - Process data chronologically
    - MA calculated using only data available up to each point
    """
    print(f"\n{'='*80}")
    print(f"STARTING SPOT PORTFOLIO BACKTEST: {symbol}")
    print(f"{'='*80}")
    print(f"Initial Capital:           ${initial_capital:,.2f}")
    print(f"\n📈 MOVING AVERAGE CONFIG:")
    print(f"   MA Type:                {ma_type}")
    print(f"   MA Period:              {ma_period}")
    print(f"   MA Timeframe:           {ma_timeframe}")
    print(f"\n💰 TRADING FEES:")
    if use_trading_fees:
        print(f"   Entry Fee:              {entry_fee_percentage}%")
        print(f"   Exit Fee:               {exit_fee_percentage}%")
    else:
        print(f"   Trading Fees:           Disabled")
    print(f"\n🏆 ALTERNATIVE ASSET DURING IDLE:")
    if hold_alt_asset_when_idle and alt_asset_file:
        print(f"   Enabled:                YES")
        print(f"   Alt Asset:              {alt_asset_symbol}")
        print(f"   Allocation:             {alt_asset_allocation_pct}%")
        print(f"   Data File:              {alt_asset_file}")
    else:
        print(f"   Enabled:                NO (hold cash when idle)")
    print(f"\nWarmup Period:             {warmup_days} days")
    print(f"Strategy:                  LONG when Close > MA, {'hold ' + alt_asset_symbol if hold_alt_asset_when_idle else 'CASH'} when Close < MA")
    print(f"{'='*80}\n")
    
    # Calculate warmup date
    warmup_date = df.index.min() + pd.Timedelta(days=warmup_days)
    print(f"Warmup period ends: {warmup_date.date()}")
    print(f"Trading starts: {warmup_date.date()}\n")

    # Load alternative asset data if enabled
    alt_asset_df = None
    if hold_alt_asset_when_idle and alt_asset_file:
        try:
            print(f"Loading alternative asset data from: {alt_asset_file}")
            alt_asset_df = pd.read_csv(alt_asset_file)

            # Create datetime index for alt asset
            if 'datetime' in alt_asset_df.columns:
                alt_asset_df['datetime'] = pd.to_datetime(alt_asset_df['datetime'])
                alt_asset_df.set_index('datetime', inplace=True)
            elif 'date' in alt_asset_df.columns and 'time' in alt_asset_df.columns:
                alt_asset_df['datetime'] = pd.to_datetime(alt_asset_df['date'] + ' ' + alt_asset_df['time'])
                alt_asset_df.set_index('datetime', inplace=True)
            elif 'Date' in alt_asset_df.columns:
                alt_asset_df['datetime'] = pd.to_datetime(alt_asset_df['Date'])
                alt_asset_df.set_index('datetime', inplace=True)

            alt_asset_df = alt_asset_df.sort_index()
            alt_asset_df.columns = alt_asset_df.columns.str.lower()

            # Resample to the same timeframe as main strategy
            alt_asset_df = alt_asset_df.resample(ma_timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()

            print(f"Alt asset loaded: {len(alt_asset_df):,} bars from {alt_asset_df.index.min()} to {alt_asset_df.index.max()}")
        except Exception as e:
            print(f"WARNING: Could not load alt asset data: {e}")
            print("Falling back to cash during idle periods.")
            alt_asset_df = None
            hold_alt_asset_when_idle = False
    
    # Pre-calculate MA
    print(f"Calculating {ma_type} {ma_period} on {ma_timeframe} timeframe...")
    ma_values = calculate_moving_average(df, period=ma_period, ma_type=ma_type, timeframe=ma_timeframe)
    print(f"MA calculated for {len(ma_values.dropna())} timestamps")
    
    # Resample to MA timeframe for signal generation
    print(f"\nResampling to {ma_timeframe} for signal generation...")
    df_ma_tf = df.resample(ma_timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Get MA values on the MA timeframe
    ma_on_tf = calculate_moving_average(df, period=ma_period, ma_type=ma_type, timeframe=ma_timeframe)
    ma_tf_values = ma_on_tf.reindex(df_ma_tf.index, method='ffill')
    
    print(f"Signal bars: {len(df_ma_tf)}")
    
    # Initialize trading records
    trades = []
    alt_asset_trades = []  # Track alt asset trades separately

    # Portfolio state
    cash = initial_capital
    shares = 0
    in_position = False
    entry_time = None
    entry_price = None

    # Alternative asset state
    alt_shares = 0
    in_alt_position = False
    alt_entry_time = None
    alt_entry_price = None

    # Track equity curve
    equity_curve = []
    equity_dates = []

    # Statistics
    total_fees_paid = 0
    alt_asset_pnl = 0  # Track alt asset P&L separately
    
    # Process each bar on MA timeframe
    for i, (bar_time, bar) in enumerate(df_ma_tf.iterrows()):
        current_close = bar['close']

        # Get alt asset price for this bar (if available)
        alt_price = None
        if alt_asset_df is not None and bar_time in alt_asset_df.index:
            alt_price = alt_asset_df.loc[bar_time, 'close']
        elif alt_asset_df is not None:
            # Try to get the closest previous price
            prior_prices = alt_asset_df.loc[alt_asset_df.index <= bar_time]
            if len(prior_prices) > 0:
                alt_price = prior_prices.iloc[-1]['close']

        # Calculate current portfolio value
        if in_position:
            portfolio_value = shares * current_close
        elif in_alt_position and alt_price is not None:
            portfolio_value = alt_shares * alt_price + cash
        else:
            portfolio_value = cash

        # Record equity
        equity_curve.append(portfolio_value)
        equity_dates.append(bar_time)
        
        # Skip warmup period
        if bar_time < warmup_date:
            continue
        
        # Get current MA value
        if bar_time not in ma_tf_values.index:
            continue
        current_ma = ma_tf_values.loc[bar_time]
        
        if pd.isna(current_ma):
            continue
        
        # === SIGNAL LOGIC ===
        
        # EXIT: Close < MA and in position
        if in_position and current_close < current_ma:
            exit_price = current_close
            exit_time = bar_time
            
            # Calculate proceeds
            gross_proceeds = shares * exit_price
            
            # Calculate exit fee
            exit_fee = 0
            if use_trading_fees:
                exit_fee = gross_proceeds * exit_fee_percentage / 100
                total_fees_paid += exit_fee
            
            net_proceeds = gross_proceeds - exit_fee
            
            # Calculate P&L
            entry_cost = entry_price * shares
            if use_trading_fees:
                entry_fee = entry_cost * entry_fee_percentage / 100
            else:
                entry_fee = 0
            
            pnl = net_proceeds - entry_cost - entry_fee
            return_pct = (pnl / (entry_cost + entry_fee)) * 100
            
            # Calculate trade duration
            duration_hours = (exit_time - entry_time).total_seconds() / 3600
            
            # Update cash
            cash = net_proceeds
            
            print(f"\n{'='*60}")
            print(f"SELL: {exit_time.date()}")
            print(f"Close: ${current_close:.2f} < MA: ${current_ma:.2f}")
            print(f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
            print(f"Shares: {shares:.6f}")
            print(f"P&L: ${pnl:,.2f} ({return_pct:.2f}%)")
            print(f"Duration: {duration_hours/24:.1f} days")
            print(f"Portfolio: ${cash:,.2f} (CASH)")
            print(f"{'='*60}")
            
            # Record trade
            trade_record = {
                'date': entry_time.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'direction': 'LONG',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'entry_cost': entry_cost,
                'exit_proceeds': net_proceeds,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'pnl': pnl,
                'return_pct': return_pct,
                'trade_duration_hours': duration_hours,
                'exit_reason': 'Close < MA',
                'portfolio_value': cash
            }
            trades.append(trade_record)

            # Reset position
            shares = 0
            in_position = False
            entry_time = None
            entry_price = None

            # Buy alternative asset if enabled
            if hold_alt_asset_when_idle and alt_asset_df is not None and alt_price is not None:
                # Calculate amount to invest in alt asset
                alt_invest_amount = cash * (alt_asset_allocation_pct / 100)

                # Calculate entry fee for alt asset
                alt_entry_fee = 0
                if use_trading_fees:
                    alt_entry_fee = alt_invest_amount * entry_fee_percentage / 100
                    total_fees_paid += alt_entry_fee

                investable_alt = alt_invest_amount - alt_entry_fee
                alt_shares = investable_alt / alt_price
                alt_entry_price = alt_price
                alt_entry_time = bar_time
                in_alt_position = True

                # Keep remaining cash (if allocation < 100%)
                cash = cash - alt_invest_amount

                print(f"\n   🥇 BUYING {alt_asset_symbol}: {bar_time.date()}")
                print(f"   Alt Asset Price: ${alt_price:.2f}")
                print(f"   Invested: ${investable_alt:,.2f} ({alt_asset_allocation_pct}% of portfolio)")
                print(f"   Alt Shares: {alt_shares:.6f}")
                print(f"   Remaining Cash: ${cash:,.2f}")
        
        # ENTRY: Close > MA and not in position
        elif not in_position and current_close > current_ma:
            # First, sell alternative asset if holding
            if in_alt_position and alt_price is not None:
                alt_gross_proceeds = alt_shares * alt_price

                # Calculate exit fee for alt asset
                alt_exit_fee = 0
                if use_trading_fees:
                    alt_exit_fee = alt_gross_proceeds * exit_fee_percentage / 100
                    total_fees_paid += alt_exit_fee

                alt_net_proceeds = alt_gross_proceeds - alt_exit_fee

                # Calculate alt asset P&L
                alt_entry_cost = alt_entry_price * alt_shares
                alt_pnl = alt_net_proceeds - alt_entry_cost
                alt_return_pct = (alt_pnl / alt_entry_cost) * 100 if alt_entry_cost > 0 else 0
                alt_duration_hours = (bar_time - alt_entry_time).total_seconds() / 3600

                alt_asset_pnl += alt_pnl
                cash += alt_net_proceeds

                print(f"\n   🥇 SELLING {alt_asset_symbol}: {bar_time.date()}")
                print(f"   Entry: ${alt_entry_price:.2f} → Exit: ${alt_price:.2f}")
                print(f"   {alt_asset_symbol} P&L: ${alt_pnl:,.2f} ({alt_return_pct:.2f}%)")
                print(f"   Duration: {alt_duration_hours/24:.1f} days")
                print(f"   Proceeds: ${alt_net_proceeds:,.2f}")

                # Record alt asset trade
                alt_trade_record = {
                    'date': alt_entry_time.strftime('%Y-%m-%d'),
                    'symbol': alt_asset_symbol,
                    'direction': 'LONG',
                    'entry_time': alt_entry_time,
                    'exit_time': bar_time,
                    'entry_price': alt_entry_price,
                    'exit_price': alt_price,
                    'shares': alt_shares,
                    'pnl': alt_pnl,
                    'return_pct': alt_return_pct,
                    'trade_duration_hours': alt_duration_hours,
                    'exit_reason': 'Re-entering main asset'
                }
                alt_asset_trades.append(alt_trade_record)

                # Reset alt position
                alt_shares = 0
                in_alt_position = False
                alt_entry_time = None
                alt_entry_price = None

            entry_price = current_close
            entry_time = bar_time

            # Calculate entry fee
            entry_fee = 0
            if use_trading_fees:
                entry_fee = cash * entry_fee_percentage / 100
                total_fees_paid += entry_fee

            # Buy shares with available cash minus fee
            investable_cash = cash - entry_fee
            shares = investable_cash / entry_price

            # Update cash (all invested)
            cash = 0
            in_position = True

            print(f"\n{'='*60}")
            print(f"BUY: {bar_time.date()}")
            print(f"Close: ${current_close:.2f} > MA: ${current_ma:.2f}")
            print(f"Entry: ${entry_price:.2f}")
            print(f"Shares: {shares:.6f}")
            print(f"Invested: ${investable_cash:,.2f}")
            print(f"Portfolio: ${shares * entry_price:,.2f} (INVESTED)")
            print(f"{'='*60}")
    
    # Close any open position at end of data
    if in_position:
        exit_time = df_ma_tf.index[-1]
        exit_price = df_ma_tf.iloc[-1]['close']
        
        gross_proceeds = shares * exit_price
        exit_fee = 0
        if use_trading_fees:
            exit_fee = gross_proceeds * exit_fee_percentage / 100
            total_fees_paid += exit_fee
        
        net_proceeds = gross_proceeds - exit_fee
        
        entry_cost = entry_price * shares
        if use_trading_fees:
            entry_fee = entry_cost * entry_fee_percentage / 100
        else:
            entry_fee = 0
        
        pnl = net_proceeds - entry_cost - entry_fee
        return_pct = (pnl / (entry_cost + entry_fee)) * 100
        duration_hours = (exit_time - entry_time).total_seconds() / 3600
        
        cash = net_proceeds
        
        print(f"\n{'='*60}")
        print(f"FINAL POSITION CLOSED: {exit_time.date()}")
        print(f"Exit: ${exit_price:.2f}")
        print(f"Reason: End of Data")
        print(f"P&L: ${pnl:,.2f} ({return_pct:.2f}%)")
        print(f"{'='*60}")
        
        trade_record = {
            'date': entry_time.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'direction': 'LONG',
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'entry_cost': entry_cost,
            'exit_proceeds': net_proceeds,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'pnl': pnl,
            'return_pct': return_pct,
            'trade_duration_hours': duration_hours,
            'exit_reason': 'End of Data',
            'portfolio_value': cash
        }
        trades.append(trade_record)
        
        shares = 0
        in_position = False

    # Close any open alt asset position at end of data
    if in_alt_position and alt_asset_df is not None:
        exit_time = df_ma_tf.index[-1]

        # Get final alt asset price
        if exit_time in alt_asset_df.index:
            alt_exit_price = alt_asset_df.loc[exit_time, 'close']
        else:
            prior_prices = alt_asset_df.loc[alt_asset_df.index <= exit_time]
            alt_exit_price = prior_prices.iloc[-1]['close'] if len(prior_prices) > 0 else alt_entry_price

        alt_gross_proceeds = alt_shares * alt_exit_price
        alt_exit_fee = 0
        if use_trading_fees:
            alt_exit_fee = alt_gross_proceeds * exit_fee_percentage / 100
            total_fees_paid += alt_exit_fee

        alt_net_proceeds = alt_gross_proceeds - alt_exit_fee

        alt_entry_cost = alt_entry_price * alt_shares
        alt_pnl = alt_net_proceeds - alt_entry_cost
        alt_return_pct = (alt_pnl / alt_entry_cost) * 100 if alt_entry_cost > 0 else 0
        alt_duration_hours = (exit_time - alt_entry_time).total_seconds() / 3600

        alt_asset_pnl += alt_pnl
        cash += alt_net_proceeds

        print(f"\n{'='*60}")
        print(f"FINAL {alt_asset_symbol} POSITION CLOSED: {exit_time.date()}")
        print(f"Exit: ${alt_exit_price:.2f}")
        print(f"Reason: End of Data")
        print(f"{alt_asset_symbol} P&L: ${alt_pnl:,.2f} ({alt_return_pct:.2f}%)")
        print(f"{'='*60}")

        alt_trade_record = {
            'date': alt_entry_time.strftime('%Y-%m-%d'),
            'symbol': alt_asset_symbol,
            'direction': 'LONG',
            'entry_time': alt_entry_time,
            'exit_time': exit_time,
            'entry_price': alt_entry_price,
            'exit_price': alt_exit_price,
            'shares': alt_shares,
            'pnl': alt_pnl,
            'return_pct': alt_return_pct,
            'trade_duration_hours': alt_duration_hours,
            'exit_reason': 'End of Data'
        }
        alt_asset_trades.append(alt_trade_record)

        alt_shares = 0
        in_alt_position = False

    # Final portfolio value
    final_value = cash
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Create equity curve DataFrame
    equity_df = pd.DataFrame({
        'date': equity_dates,
        'portfolio_value': equity_curve
    })
    equity_df.set_index('date', inplace=True)
    
    # Calculate daily returns for equity curve
    equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
    
    # Calculate buy & hold benchmark
    first_price = df_ma_tf.loc[df_ma_tf.index >= warmup_date].iloc[0]['close']
    last_price = df_ma_tf.iloc[-1]['close']
    buy_hold_return = ((last_price / first_price) - 1) * 100
    buy_hold_final = initial_capital * (last_price / first_price)
    
    # Create buy & hold equity curve
    df_after_warmup = df_ma_tf.loc[df_ma_tf.index >= warmup_date]
    buy_hold_equity = initial_capital * (df_after_warmup['close'] / first_price)
    equity_df['buy_hold'] = buy_hold_equity.reindex(equity_df.index, method='ffill')
    equity_df['buy_hold_return'] = equity_df['buy_hold'].pct_change()
    
    # Print summary
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY - SPOT PORTFOLIO")
    print(f"{'='*80}")
    
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = trades_df[trades_df['pnl'] < 0]['pnl'].sum()
        net_pnl = trades_df['pnl'].sum()
        
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        total_return = ((final_value / initial_capital) - 1) * 100
        
        # Calculate time in market
        total_time_in_market = trades_df['trade_duration_hours'].sum()
        total_backtest_hours = (df_ma_tf.index[-1] - warmup_date).total_seconds() / 3600
        time_in_market_pct = (total_time_in_market / total_backtest_hours) * 100 if total_backtest_hours > 0 else 0
        
        # === CALCULATE RISK METRICS ===
        
        # Strategy metrics
        strategy_returns = equity_df['daily_return'].dropna()
        
        # Annualization factor (assuming daily data on trading timeframe)
        if ma_timeframe == '1D':
            ann_factor = 252
        elif ma_timeframe == '4H':
            ann_factor = 252 * 6
        elif ma_timeframe == '1H':
            ann_factor = 252 * 24
        else:
            ann_factor = 252
        
        # Strategy Sharpe Ratio (assuming 0% risk-free rate)
        strategy_mean = strategy_returns.mean()
        strategy_std = strategy_returns.std()
        strategy_sharpe = (strategy_mean / strategy_std) * np.sqrt(ann_factor) if strategy_std > 0 else 0
        
        # Strategy Sortino Ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        strategy_sortino = (strategy_mean / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0
        
        # Strategy Max Drawdown
        equity_values = equity_df['portfolio_value'].dropna().values
        if len(equity_values) > 0:
            peak = np.maximum.accumulate(equity_values)
            drawdown = (peak - equity_values) / peak * 100
            strategy_max_dd = np.nanmax(drawdown) if len(drawdown) > 0 else 0
        else:
            strategy_max_dd = 0
        
        # Strategy Calmar Ratio (annualized return / max drawdown)
        years = len(strategy_returns) / ann_factor
        strategy_ann_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        strategy_calmar = strategy_ann_return / strategy_max_dd if strategy_max_dd > 0 else 0
        
        # Buy & Hold metrics
        bh_returns = equity_df['buy_hold_return'].dropna()
        
        # B&H Sharpe
        bh_mean = bh_returns.mean()
        bh_std = bh_returns.std()
        bh_sharpe = (bh_mean / bh_std) * np.sqrt(ann_factor) if bh_std > 0 else 0
        
        # B&H Sortino
        bh_downside = bh_returns[bh_returns < 0]
        bh_downside_std = bh_downside.std() if len(bh_downside) > 0 else 0
        bh_sortino = (bh_mean / bh_downside_std) * np.sqrt(ann_factor) if bh_downside_std > 0 else 0
        
        # B&H Max Drawdown - drop NaN values first
        bh_equity = equity_df['buy_hold'].dropna().values
        if len(bh_equity) > 0:
            bh_peak = np.maximum.accumulate(bh_equity)
            bh_drawdown = (bh_peak - bh_equity) / bh_peak * 100
            bh_max_dd = np.nanmax(bh_drawdown) if len(bh_drawdown) > 0 else 0
        else:
            bh_max_dd = 0
        
        # B&H Calmar
        bh_ann_return = ((buy_hold_final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        bh_calmar = bh_ann_return / bh_max_dd if bh_max_dd > 0 else 0
        
        # === PRINT RESULTS ===
        
        print(f"\n📊 STRATEGY PERFORMANCE:")
        print(f"   Initial Capital:      ${initial_capital:,.2f}")
        print(f"   Final Portfolio:      ${final_value:,.2f}")
        print(f"   Net P&L:              ${net_pnl:,.2f}")
        print(f"   Total Return:         {total_return:.2f}%")
        print(f"   Annualized Return:    {strategy_ann_return:.2f}%")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"   Total Trades:         {total_trades}")
        print(f"   Winning Trades:       {winning_trades}")
        print(f"   Losing Trades:        {losing_trades}")
        print(f"   Win Rate:             {win_rate:.2f}%")
        print(f"   Profit Factor:        {profit_factor:.2f}")
        print(f"   Avg Win:              ${avg_win:,.2f}")
        print(f"   Avg Loss:             ${avg_loss:,.2f}")
        
        print(f"\n⏱️ TIME ANALYSIS:")
        print(f"   Time in Market:       {time_in_market_pct:.1f}%")
        print(f"   Time in Cash:         {100 - time_in_market_pct:.1f}%")
        print(f"   Backtest Duration:    {years:.2f} years")
        
        if use_trading_fees:
            print(f"\n💸 FEE ANALYSIS:")
            print(f"   Total Fees Paid:      ${total_fees_paid:,.2f}")
            print(f"   Fees as % of Capital: {(total_fees_paid/initial_capital)*100:.2f}%")

        # Alternative Asset Summary
        if hold_alt_asset_when_idle and len(alt_asset_trades) > 0:
            alt_trades_df = pd.DataFrame(alt_asset_trades)
            alt_total_trades = len(alt_trades_df)
            alt_winning = len(alt_trades_df[alt_trades_df['pnl'] > 0])
            alt_losing = len(alt_trades_df[alt_trades_df['pnl'] < 0])
            alt_win_rate = alt_winning / alt_total_trades * 100 if alt_total_trades > 0 else 0
            alt_total_pnl = alt_trades_df['pnl'].sum()
            alt_time_in_market = alt_trades_df['trade_duration_hours'].sum()
            alt_time_pct = (alt_time_in_market / total_backtest_hours) * 100 if total_backtest_hours > 0 else 0

            print(f"\n🥇 ALTERNATIVE ASSET ({alt_asset_symbol}) SUMMARY:")
            print(f"   Total {alt_asset_symbol} Trades: {alt_total_trades}")
            print(f"   Winning Trades:       {alt_winning}")
            print(f"   Losing Trades:        {alt_losing}")
            print(f"   Win Rate:             {alt_win_rate:.2f}%")
            print(f"   Total {alt_asset_symbol} P&L:    ${alt_total_pnl:,.2f}")
            print(f"   Time in {alt_asset_symbol}:      {alt_time_pct:.1f}%")
            print(f"   Avg Return per Trade: {alt_trades_df['return_pct'].mean():.2f}%")

        # === STRATEGY VS BUY & HOLD COMPARISON TABLE ===
        print(f"\n{'='*80}")
        print("📊 STRATEGY vs BUY & HOLD COMPARISON")
        print(f"{'='*80}")
        print(f"")
        print(f"{'METRIC':<25} {'STRATEGY':>15} {'BUY & HOLD':>15} {'DIFFERENCE':>15}")
        print(f"{'-'*70}")
        print(f"{'Total Return':<25} {total_return:>14.2f}% {buy_hold_return:>14.2f}% {total_return - buy_hold_return:>+14.2f}%")
        print(f"{'Annualized Return':<25} {strategy_ann_return:>14.2f}% {bh_ann_return:>14.2f}% {strategy_ann_return - bh_ann_return:>+14.2f}%")
        print(f"{'Final Value':<25} ${final_value:>13,.0f} ${buy_hold_final:>13,.0f} ${final_value - buy_hold_final:>+13,.0f}")
        print(f"{'-'*70}")
        print(f"{'Sharpe Ratio':<25} {strategy_sharpe:>15.2f} {bh_sharpe:>15.2f} {strategy_sharpe - bh_sharpe:>+15.2f}")
        print(f"{'Sortino Ratio':<25} {strategy_sortino:>15.2f} {bh_sortino:>15.2f} {strategy_sortino - bh_sortino:>+15.2f}")
        print(f"{'Calmar Ratio':<25} {strategy_calmar:>15.2f} {bh_calmar:>15.2f} {strategy_calmar - bh_calmar:>+15.2f}")
        print(f"{'-'*70}")
        print(f"{'Max Drawdown':<25} {strategy_max_dd:>14.2f}% {bh_max_dd:>14.2f}% {strategy_max_dd - bh_max_dd:>+14.2f}%")
        print(f"{'='*80}")
        
        # Winner summary
        print(f"\n🏆 SUMMARY:")
        strategy_wins = 0
        if total_return > buy_hold_return:
            print(f"   ✅ Strategy OUTPERFORMS on Total Return by {total_return - buy_hold_return:+.2f}%")
            strategy_wins += 1
        else:
            print(f"   ❌ Strategy UNDERPERFORMS on Total Return by {total_return - buy_hold_return:.2f}%")
        
        if strategy_sharpe > bh_sharpe:
            print(f"   ✅ Strategy has BETTER Sharpe Ratio ({strategy_sharpe:.2f} vs {bh_sharpe:.2f})")
            strategy_wins += 1
        else:
            print(f"   ❌ Strategy has WORSE Sharpe Ratio ({strategy_sharpe:.2f} vs {bh_sharpe:.2f})")
        
        if strategy_max_dd < bh_max_dd:
            print(f"   ✅ Strategy has LOWER Max Drawdown ({strategy_max_dd:.2f}% vs {bh_max_dd:.2f}%)")
            strategy_wins += 1
        else:
            print(f"   ❌ Strategy has HIGHER Max Drawdown ({strategy_max_dd:.2f}% vs {bh_max_dd:.2f}%)")
        
        print(f"\n   Strategy wins {strategy_wins}/3 key metrics")
        print(f"{'='*80}\n")
        
        # Store metrics in equity_df for dashboard
        equity_df.attrs['strategy_sharpe'] = strategy_sharpe
        equity_df.attrs['strategy_sortino'] = strategy_sortino
        equity_df.attrs['strategy_calmar'] = strategy_calmar
        equity_df.attrs['strategy_max_dd'] = strategy_max_dd
        equity_df.attrs['bh_sharpe'] = bh_sharpe
        equity_df.attrs['bh_sortino'] = bh_sortino
        equity_df.attrs['bh_calmar'] = bh_calmar
        equity_df.attrs['bh_max_dd'] = bh_max_dd
        equity_df.attrs['strategy_ann_return'] = strategy_ann_return
        equity_df.attrs['bh_ann_return'] = bh_ann_return
        equity_df.attrs['alt_asset_trades'] = alt_asset_trades
        equity_df.attrs['alt_asset_pnl'] = alt_asset_pnl
        equity_df.attrs['alt_asset_symbol'] = alt_asset_symbol

        return trades_df, equity_df, total_return, net_pnl, buy_hold_return
    else:
        print("No trades executed during backtest period")
        equity_df.attrs['alt_asset_trades'] = alt_asset_trades
        equity_df.attrs['alt_asset_pnl'] = alt_asset_pnl
        equity_df.attrs['alt_asset_symbol'] = alt_asset_symbol
        return pd.DataFrame(), equity_df, 0, 0, buy_hold_return


def create_performance_dashboard(trades_df, equity_df, initial_capital, symbol, config_summary, buy_hold_return):
    """Create comprehensive performance visualization dashboard"""
    if len(trades_df) == 0:
        print(f"No trades to create dashboard for {symbol}")
        return None
    
    # Get metrics from equity_df attributes
    strategy_sharpe = equity_df.attrs.get('strategy_sharpe', 0)
    strategy_sortino = equity_df.attrs.get('strategy_sortino', 0)
    strategy_calmar = equity_df.attrs.get('strategy_calmar', 0)
    strategy_max_dd = equity_df.attrs.get('strategy_max_dd', 0)
    bh_sharpe = equity_df.attrs.get('bh_sharpe', 0)
    bh_sortino = equity_df.attrs.get('bh_sortino', 0)
    bh_calmar = equity_df.attrs.get('bh_calmar', 0)
    bh_max_dd = equity_df.attrs.get('bh_max_dd', 0)
    strategy_ann_return = equity_df.attrs.get('strategy_ann_return', 0)
    bh_ann_return = equity_df.attrs.get('bh_ann_return', 0)
    
    # Create figure with more rows for comparison table
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'{symbol} Spot Portfolio - MA Trend Following Strategy', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # 1. Equity Curve with Buy & Hold - Top (spanning full width)
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    
    equity_curve = equity_df['portfolio_value'].values
    equity_dates = equity_df.index
    
    # Get actual buy & hold curve from equity_df
    if 'buy_hold' in equity_df.columns:
        buy_hold_curve = equity_df['buy_hold'].values
    else:
        buy_hold_curve = np.linspace(initial_capital, initial_capital * (1 + buy_hold_return/100), len(equity_curve))
    
    # Plot equity curves with thicker lines
    ax1.plot(equity_dates, equity_curve, 'b-', linewidth=2.5, label=f'Strategy (Return: {((equity_curve[-1]/initial_capital)-1)*100:.1f}%)')
    ax1.plot(equity_dates, buy_hold_curve, color='orange', linestyle='-', linewidth=2.5, label=f'Buy & Hold (Return: {buy_hold_return:.1f}%)')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Initial Capital')
    
    # Fill between for outperformance/underperformance
    ax1.fill_between(equity_dates, equity_curve, buy_hold_curve, 
                     where=(equity_curve >= buy_hold_curve), 
                     color='green', alpha=0.3, interpolate=True)
    ax1.fill_between(equity_dates, equity_curve, buy_hold_curve, 
                     where=(equity_curve < buy_hold_curve), 
                     color='red', alpha=0.3, interpolate=True)
    
    ax1.set_title('Equity Curve: Strategy vs Buy & Hold', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add text annotation for final values
    ax1.annotate(f'${equity_curve[-1]:,.0f}', xy=(equity_dates[-1], equity_curve[-1]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10, color='blue', fontweight='bold')
    ax1.annotate(f'${buy_hold_curve[-1]:,.0f}', xy=(equity_dates[-1], buy_hold_curve[-1]), 
                xytext=(10, -15), textcoords='offset points', fontsize=10, color='orange', fontweight='bold')
    
    # 2. Strategy vs Buy & Hold Metrics Comparison - Bar Chart
    ax2 = plt.subplot2grid((4, 3), (1, 0), colspan=2)
    
    metrics = ['Total\nReturn %', 'Annual\nReturn %', 'Sharpe\nRatio', 'Sortino\nRatio', 'Calmar\nRatio', 'Max DD %\n(lower=better)']
    strategy_values = [
        ((equity_curve[-1]/initial_capital)-1)*100,
        strategy_ann_return,
        strategy_sharpe,
        strategy_sortino,
        strategy_calmar,
        strategy_max_dd
    ]
    bh_values = [
        buy_hold_return,
        bh_ann_return,
        bh_sharpe,
        bh_sortino,
        bh_calmar,
        bh_max_dd
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, strategy_values, width, label='Strategy', color='#3498db', edgecolor='black')
    bars2 = ax2.bar(x + width/2, bh_values, width, label='Buy & Hold', color='#e67e22', edgecolor='black')
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Strategy vs Buy & Hold - Key Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars1, strategy_values):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold', color='#2980b9')
    
    for bar, val in zip(bars2, bh_values):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold', color='#d35400')
    
    # 3. Win/Loss Distribution - Middle right
    ax3 = plt.subplot2grid((4, 3), (1, 2))
    winning = len(trades_df[trades_df['pnl'] > 0])
    losing = len(trades_df[trades_df['pnl'] < 0])
    breakeven = len(trades_df[trades_df['pnl'] == 0])
    
    labels = ['Wins', 'Losses', 'Breakeven']
    sizes = [winning, losing, breakeven]
    colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.1, 0, 0)
    
    non_zero_labels = [l for l, s in zip(labels, sizes) if s > 0]
    non_zero_sizes = [s for s in sizes if s > 0]
    non_zero_colors = [c for c, s in zip(colors_pie, sizes) if s > 0]
    non_zero_explode = [e for e, s in zip(explode, sizes) if s > 0]
    
    if non_zero_sizes:
        ax3.pie(non_zero_sizes, explode=non_zero_explode, labels=non_zero_labels, 
                colors=non_zero_colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax3.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
    
    # 4. Drawdown Comparison - Bottom left
    ax4 = plt.subplot2grid((4, 3), (2, 0), colspan=2)
    
    # Strategy drawdown
    peak = np.maximum.accumulate(equity_curve)
    strategy_dd = (peak - equity_curve) / peak * 100
    
    # Buy & Hold drawdown
    bh_peak = np.maximum.accumulate(buy_hold_curve)
    bh_dd = (bh_peak - buy_hold_curve) / bh_peak * 100
    
    ax4.fill_between(equity_dates, 0, strategy_dd, color='blue', alpha=0.3, label=f'Strategy (Max: {max(strategy_dd):.1f}%)')
    ax4.fill_between(equity_dates, 0, bh_dd, color='orange', alpha=0.3, label=f'Buy & Hold (Max: {max(bh_dd):.1f}%)')
    ax4.plot(equity_dates, strategy_dd, 'b-', linewidth=1.5)
    ax4.plot(equity_dates, bh_dd, color='orange', linestyle='-', linewidth=1.5)
    
    ax4.set_title('Drawdown Comparison: Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Drawdown (%)', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower left', fontsize=11)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Return Distribution - Bottom middle-right
    ax5 = plt.subplot2grid((4, 3), (2, 2))
    ax5.hist(trades_df['return_pct'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Breakeven')
    ax5.axvline(x=trades_df['return_pct'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {trades_df["return_pct"].mean():.1f}%')
    ax5.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Return %', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    
    # 6. Cumulative P&L
    ax6 = plt.subplot2grid((4, 3), (3, 0))
    cumulative_pnl = trades_df['pnl'].cumsum()
    ax6.plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2)
    ax6.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                     where=(cumulative_pnl >= 0), color='green', alpha=0.3)
    ax6.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                     where=(cumulative_pnl < 0), color='red', alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Trade Number', fontsize=12)
    ax6.set_ylabel('Cumulative P&L ($)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # 7. Trade Duration Distribution
    ax7 = plt.subplot2grid((4, 3), (3, 1))
    ax7.hist(trades_df['trade_duration_hours'] / 24, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax7.axvline(x=(trades_df['trade_duration_hours'] / 24).mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {(trades_df["trade_duration_hours"] / 24).mean():.1f} days')
    ax7.set_title('Trade Duration Distribution', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Duration (days)', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=10)
    
    # 8. Trade Duration vs Return scatter
    ax8 = plt.subplot2grid((4, 3), (3, 2))
    scatter = ax8.scatter(trades_df['trade_duration_hours'] / 24, trades_df['return_pct'],
                         c=trades_df['return_pct'], cmap='RdYlGn', alpha=0.6, 
                         s=50, edgecolors='black', linewidths=0.5)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax8.set_title('Duration vs Return', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Duration (days)', fontsize=12)
    ax8.set_ylabel('Return (%)', fontsize=12)
    ax8.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax8, label='Return %')
    
    # Summary text at bottom
    total_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / total_trades * 100
    net_pnl = trades_df['pnl'].sum()
    total_return = ((equity_curve[-1] / initial_capital) - 1) * 100
    profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                       trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
    
    # Determine winner
    outperformance = total_return - buy_hold_return
    winner_text = f"Strategy OUTPERFORMS by {outperformance:+.1f}%" if outperformance > 0 else f"Strategy UNDERPERFORMS by {outperformance:.1f}%"
    winner_color = 'green' if outperformance > 0 else 'red'
    
    summary_text = (f"Trades: {total_trades} | Win Rate: {win_rate:.1f}% | PF: {profit_factor:.2f} | "
                   f"Strategy: {total_return:.1f}% | Buy&Hold: {buy_hold_return:.1f}% | {winner_text}")
    
    plt.figtext(0.5, 0.02, summary_text, ha='center', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=12, fontweight='bold')
    
    plt.figtext(0.5, 0.005, config_summary, ha='center', fontsize=10,
               style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    return fig


def process_backtest(
    data_file,
    symbol='BTCUSDT',
    output_dir=None,
    config=None
):
    """Process a single backtest run"""
    
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        full_config = DEFAULT_CONFIG.copy()
        full_config.update(config)
        config = full_config
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SPOT PORTFOLIO BACKTEST: {symbol}")
    print(f"Data File: {data_file}")
    print(f"{'='*80}")
    
    try:
        # Load data
        df = load_and_prepare_data(
            data_file,
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        
        # Run backtest
        trades_df, equity_df, total_return, net_pnl, buy_hold_return = backtest_spot_portfolio(
            df=df,
            symbol=symbol,
            initial_capital=config['initial_capital'],
            ma_type=config['ma_type'],
            ma_period=config['ma_period'],
            ma_timeframe=config['ma_timeframe'],
            use_trading_fees=config['use_trading_fees'],
            entry_fee_percentage=config['entry_fee_percentage'],
            exit_fee_percentage=config['exit_fee_percentage'],
            warmup_days=config['warmup_days'],
            hold_alt_asset_when_idle=config.get('hold_alt_asset_when_idle', False),
            alt_asset_file=config.get('alt_asset_file', None),
            alt_asset_symbol=config.get('alt_asset_symbol', 'GOLD'),
            alt_asset_allocation_pct=config.get('alt_asset_allocation_pct', 100)
        )
        
        # Save results if trades were executed
        if not trades_df.empty:
            if output_dir is None:
                output_dir = os.path.dirname(os.path.abspath(data_file))
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save trade log
            trade_log_file = os.path.join(output_dir, 
                                         f'{symbol}_spot_portfolio_trades_{timestamp}.csv')
            trades_df.to_csv(trade_log_file, index=False)
            print(f"Trade log saved: {trade_log_file}")
            
            # Save equity curve
            equity_file = os.path.join(output_dir,
                                      f'{symbol}_spot_portfolio_equity_{timestamp}.csv')
            equity_df.to_csv(equity_file)
            print(f"Equity curve saved: {equity_file}")

            # Save alt asset trades if any
            alt_asset_trades = equity_df.attrs.get('alt_asset_trades', [])
            if len(alt_asset_trades) > 0:
                alt_trades_df = pd.DataFrame(alt_asset_trades)
                alt_asset_symbol = config.get('alt_asset_symbol', 'GOLD')
                alt_trade_log_file = os.path.join(output_dir,
                                                 f'{symbol}_{alt_asset_symbol}_trades_{timestamp}.csv')
                alt_trades_df.to_csv(alt_trade_log_file, index=False)
                print(f"Alt asset trade log saved: {alt_trade_log_file}")

            # Create config summary
            alt_config_str = ""
            if config.get('hold_alt_asset_when_idle', False):
                alt_config_str = f" | Alt Asset: {config.get('alt_asset_symbol', 'GOLD')} ({config.get('alt_asset_allocation_pct', 100)}%)"
            config_summary = (f"Config: {config['ma_type']}{config['ma_period']} on {config['ma_timeframe']} | "
                            f"Fees: {'ON' if config['use_trading_fees'] else 'OFF'} | "
                            f"Strategy: LONG when Close > MA{alt_config_str}")
            
            # Create dashboard
            fig = create_performance_dashboard(
                trades_df, 
                equity_df,
                config['initial_capital'], 
                symbol,
                config_summary,
                buy_hold_return
            )
            
            if fig:
                dashboard_file = os.path.join(output_dir,
                                             f'{symbol}_spot_portfolio_dashboard_{timestamp}.png')
                fig.savefig(dashboard_file, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"Performance dashboard saved: {dashboard_file}")
        
        return trades_df, total_return, net_pnl, symbol, buy_hold_return
        
    except Exception as e:
        print(f"ERROR processing backtest: {e}")
        traceback.print_exc()
        return None, None, None, symbol, None


def main():
    """Main function"""
    print("="*80)
    print("SPOT PORTFOLIO - MA TREND FOLLOWING STRATEGY BACKTEST")
    print("="*80)
    
    parser = argparse.ArgumentParser(
        description='Spot Portfolio MA Trend Following Strategy Backtester'
    )
    
    parser.add_argument('--data_file', required=True, help='Price data CSV file path')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--output', help='Output directory')
    
    args = parser.parse_args()
    
    # Use DEFAULT_CONFIG (edit at top of file)
    config = DEFAULT_CONFIG.copy()
    
    result = process_backtest(
        data_file=args.data_file,
        symbol=args.symbol,
        output_dir=args.output,
        config=config
    )
    
    trades_df, total_return, net_pnl, symbol, buy_hold_return = result
    
    if trades_df is not None and len(trades_df) > 0:
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Symbol:              {symbol}")
        print(f"Strategy Return:     {total_return:.2f}%")
        print(f"Buy & Hold Return:   {buy_hold_return:.2f}%")
        print(f"Outperformance:      {total_return - buy_hold_return:+.2f}%")
        print(f"Net P&L:             ${net_pnl:,.2f}")
        print(f"Total Trades:        {len(trades_df)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


# Usage:
# python3 ma_trend_spot_portfolio.py --data_file "/path/to/data.csv" --symbol SPX --output /path/to/output

# FEATURES:
# ✓ Spot portfolio (no leverage, no shorting)
# ✓ LONG when Close > MA, CASH when Close < MA
# ✓ Configurable MA type (SMA/EMA), period, and timeframe
# ✓ Trading fees (optional)
# ✓ Buy & Hold comparison
# ✓ Time in market analysis
# ✓ Anti-forward bias implementation
# ✓ Comprehensive performance dashboard
# ✓ Equity curve tracking