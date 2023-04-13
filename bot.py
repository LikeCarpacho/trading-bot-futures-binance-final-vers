import math
import time
import numpy as np
import requests
from binance.client import Client
import talib
import json

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Initialize Binance client
client = Client(api_key, api_secret)

def get_account_balance():
    """
    Get account balance for USDT
    """
    account_balance = client.futures_account_balance()
    for balance in account_balance:
        if balance['asset'] == 'USDT':
            return float(balance['balance'])
    return 0.0  # return 0 if balance not found

# Constants
TRADE_SYMBOL = "BTCUSDT"
TRADE_SIZE = get_account_balance()  # Entire USDT futures balance
TRADE_TYPE = "MARKET"
TRADE_LVRG = 20
STOP_LOSS_THRESHOLD = 0.025  # 2.5% stop loss threshold
TAKE_PROFIT_THRESHOLD = 0.05 # 5% take profit threshold
SINEWAVE_PERIOD = 12  # 12 periods for sinewave
ENTRY_MOMENTUM_THRESHOLD = 3  # 3 consecutive candles above or below sinewave
REVERSAL_KEY_POINT = 6  # 6 periods after entry momentum for reversal key point

# Variables
closed_positions = []

# Print account balance
print("USDT Futures balance:", get_account_balance())

def get_talib_poly_channel(data, degree):
    """
    Returns polynomial regression channel using data and degree
    """
    # Talib.LINEARREG outputs an array with `degree-1` NaN values at the beginning
    # We need to remove these NaNs before continuing
    pr = talib.LINEARREG(data, timeperiod=degree)
    pr = pr[~np.isnan(pr)]
    
    # We need to calculate the standard deviation using the same number of values
    # as the polynomial regression line, so we use `len(pr)` instead of `len(data)`
    upper_channel = pr + np.std(data[-len(pr):]) * 2
    lower_channel = pr - np.std(data[-len(pr):]) * 2
    return upper_channel, lower_channel

# Define timeframes
timeframes = ['8h', '4h', '2h', '1h', '30m', '15m', '5m', '3m', '1m']

# Define start and end time for historical data
start_time = int(time.time()) - (86400 * 30)  # 30 days ago
end_time = int(time.time())

# Fetch historical data for BTCUSDT pair
candles = {}
for interval in timeframes:
    tf_candles = client.futures_klines(symbol=TRADE_SYMBOL, interval=interval.lower(), startTime=start_time * 1000, endTime=end_time * 1000)
    candles[interval] = []
    for candle in tf_candles:
        candles[interval].append({
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        })

def get_mtf_signal(candles, timeframes):
    """
    Calculates the MTF signals using polynomial regression channels for each timeframe.
    Returns a dictionary containing the MTF signals for each timeframe.
    """
    mtf_signal = {}
    neutral_tf = []

    # Loop through each timeframe
    for tf in timeframes:
        data = np.array([c['close'] for c in candles[tf]])
        upper_channel, lower_channel = get_talib_poly_channel(data, 20)  # Use 20 degree polynomial regression

        # Calculate the signal based on whether the close is below the lowest channel or above the highest channel
        close = candles[tf][-1]['close']
        if close <= lower_channel[-1] or all([close <= get_talib_poly_channel(np.array([c['close'] for c in candles[tf]]), 20)[1][-1] for tf in timeframes]):
            signal = "LONG"
        elif close >= upper_channel[-1] or all([close >= get_talib_poly_channel(np.array([c['close'] for c in candles[tf]]), 20)[0][-1] for tf in timeframes]):
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
            neutral_tf.append(tf)
        
        # Add signal to dictionary
        mtf_signal[tf] = signal

    # If all timeframes are neutral, return "NO TRADE" signal
    if len(neutral_tf) == len(timeframes):
        return "NO TRADE"
    
    return mtf_signal

def get_all_lows(candles):
    lows = []
    for i in range(len(candles[timeframes[0]])):
        low = float('inf')
        for tf in timeframes:
            low = min(low, candles[tf][i]['low'])
        lows.append(low)
    return lows

def get_all_highs(candles):
    highs = []
    for i in range(len(candles[timeframes[0]])):
        high = float('-inf')
        for tf in timeframes:
            high = max(high, candles[tf][i]['high'])
        highs.append(high)
    return highs

def check_long_entry(candles, stop_loss_threshold, take_profit_threshold):
    close_prices = [candle['close'] for candle in candles]
    diff = sliding_window_diff(close_prices)

    momentum_counter = 0
    for i in range(len(diff)-1):
        if diff[i] < 0 and diff[i+1] > 0:
            momentum_counter -= 1
        elif diff[i] > 0 and diff[i+1] < 0:
            momentum_counter += 1
    if all(price < lower for lower, price in zip(get_all_lows(candles), close_prices)) and momentum_counter == -ENTRY_MOMENTUM_THRESHOLD:
        return True
    return False

def check_short_entry(candles, stop_loss_threshold, take_profit_threshold):
    close_prices = [candle['close'] for candle in candles]
    diff = sliding_window_diff(close_prices)

    momentum_counter = 0
    for i in range(len(diff)-1):
        if diff[i] > 0 and diff[i+1] < 0:
            momentum_counter -= 1
        elif diff[i] < 0 and diff[i+1] > 0:
            momentum_counter += 1
    if all(price > upper for upper, price in zip(get_all_highs(candles), close_prices)) and momentum_counter == ENTRY_MOMENTUM_THRESHOLD:
        return True
    return False

def cancel_all_positions(symbol):
    """
    Cancels all open orders and positions for the specified symbol.
    """
    # cancel all open orders
    client.futures_cancel_all_orders(symbol=symbol)

    # close all open positions
    positions = client.futures_position_information(symbol=symbol)
    for position in positions:
        if position['positionAmt'] != '0':
            if position['positionSide'] == 'LONG':
                client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=position['positionAmt'])
            elif position['positionSide'] == 'SHORT':
                client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=position['positionAmt'])

def log_trade_details(trade_type, entry_price, stop_loss, take_profit, quantity):
    """Logs the details of a trade"""
    print(f"{trade_type.upper()} TRADE DETAILS")
    print(f"Entry Price: {entry_price}")
    print(f"Stop Loss: {stop_loss}")
    print(f"Take Profit: {take_profit}")
    print(f"Quantity: {quantity}\n")


def get_sinewave(ticker, interval, period):
    """Calculates the sinewave indicator for the specified ticker and interval"""
    prices = client.futures_historical_klines(ticker, interval, f"{period * 25} minutes ago UTC")
    closes = [float(price[4]) for price in prices]

    # Calculate the sinewave
    ema1 = sum(closes[-period:]) / period
    ema2 = sum(closes[-period * 2:]) / (period * 2)
    sinewave = ema1 - ema2

    return sinewave

def get_market_price():
    ticker = client.get_ticker(symbol=TRADE_SYMBOL)
    return float(ticker['lastPrice'])

def main():
    # initialize variables
    position_size = 100
    entry_price = 1.2000
    stop_loss = 0.0050
    take_profit = 0.0100
    open_positions = []
    is_reversal_key_point = False
    
    while True:
        # get current market price
        current_price = get_market_price()
        
        # calculate momentum of sine function
        momentum = calculate_sine_momentum(current_price)
        
        if len(open_positions) == 0:
            # if there are no open positions, check for entry conditions
            if momentum < 0 and current_price < entry_price - stop_loss:
                # enter long position
                open_positions.append(('long', position_size, current_price))
                print(f'Entered long position at {current_price}')
            elif momentum > 0 and current_price > entry_price + stop_loss:
                # enter short position
                open_positions.append(('short', position_size, current_price))
                print(f'Entered short position at {current_price}')
            elif is_reversal_key_point:
                # reload script from beginning
                is_reversal_key_point = False
                print('Reloaded script from beginning')
                
        else:
            # if there are open positions, check for exit conditions
            for position in open_positions:
                side = position[0]
                if side == 'long':
                    if current_price > position[2] + take_profit:
                        # close long position
                        close_price = current_price
                        pnl = (close_price - position[2]) * position[1]
                        print(f'Closed long position at {close_price} for a profit of {pnl}')
                        open_positions.remove(position)
                    elif current_price < position[2] - stop_loss:
                        # close long position
                        close_price = current_price
                        pnl = (close_price - position[2]) * position[1]
                        print(f'Closed long position at {close_price} for a loss of {pnl}')
                        open_positions.remove(position)
                        # init new short trade with reversed position size
                        position_size *= -1
                        open_positions.append(('short', position_size, current_price))
                        print(f'Entered new short position at {current_price}')
                elif side == 'short':
                    if current_price < position[2] - take_profit:
                        # close short position
                        close_price = current_price
                        pnl = (position[2] - close_price) * position[1]
                        print(f'Closed short position at {close_price} for a profit of {pnl}')
                        open_positions.remove(position)
                    elif current_price > position[2] + stop_loss:
                        # close short position
                        close_price = current_price
                        pnl = (position[2] - close_price) * position[1]
                        print(f'Closed short position at {close_price} for a loss of {pnl}')
                        open_positions.remove(position)
                        # init new long trade with reversed position size
                        position_size *= -1
                        open_positions.append(('long', position_size, current_price))
                        print(f'Entered new long position at {current_price}')

            if is_reversal_key_point:
                # reload script from beginning
                is_reversal_key_point = False
                print('Reloaded script from beginning')

            elif any(abs(current_price - position[2]) > take_profit for position in open_positions):
                # hit take profit momentum, close all positions and end trade
                total_pnl = 0
                for position in open_positions:
                    if current_price > position[2]: # long position
                        pnl = (current_price - position[2]) * position[1]
                        print(f"Closed {position[1]} contracts at {current_price} for a profit of {pnl}")
                    else: # short position
                        pnl = (position[2] - current_price) * position[1]
                        print(f"Closed {position[1]} contracts at {current_price} for a profit of {pnl}")
                    total_pnl += pnl
                print(f"Total PnL for this trade: {total_pnl}")
                open_positions = []
                print('Exited trade at take profit momentum')
                # reload script from beginning
                is_reversal_key_point = False
                print('Reloaded script from beginning')

            elif any(abs(current_price - position[2]) > stop_loss for position in open_positions):
                # hit stop loss, close all positions and initiate new trade with opposite side
                total_pnl = 0
                for position in open_positions:
                    if current_price > position[2]:  # long position
                        pnl = (current_price - position[2]) * position[1]
                        print(f"Closed {position[1]} contracts at {current_price} for a loss of {pnl}")
                    else:  # short position
                        pnl = (position[2] - current_price) * position[1]
                        print(f"Closed {position[1]} contracts at {current_price} for a loss of {pnl}")
                    total_pnl += pnl
                print(f"Total PnL for this trade: {total_pnl}")
                open_positions = []
                print('Exited trade at stop loss')
                # initiate new trade with opposite side
                entry_price = current_price
                if open_positions[0][0] == 'long':
                    trade_side = 'short'
                    position_size *= -1
                else:
                    trade_side = 'long'
                    position_size *= -1
                open_positions.append((trade_side, position_size, entry_price))
                print(f"Entered new trade at {entry_price} with {position_size} contracts, {trade_side} side")
                # reload script from beginning
                is_reversal_key_point = False
                print('Reloaded script from beginning')

if __name__ == '__main__':
    main()
