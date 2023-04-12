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
TAKE_PROFIT_THRESHOLD = 0.05 # 5% stop loss threshold
SINEWAVE_PERIOD = 12  # 12 periods for sinewave
ENTRY_MOMENTUM_THRESHOLD = 3  # 3 consecutive candles above or below sinewave
REVERSAL_KEY_POINT = 6  # 6 periods after entry momentum for reversal key point

# Variables
closed_positions = []

def get_talib_poly_channel(data, degree):
    """
    Returns polynomial regression channel using data and degree
    """
    # Get polynomial regression values
    pr, _, _ = talib.LINEARREG(data, timeperiod=degree)
    # Calculate upper and lower channel lines
    upper_channel = pr + np.std(data) * 2
    lower_channel = pr - np.std(data) * 2
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
    neutral_tf = None
    neutral_tf_typo = None
    for tf in timeframes:
        slope, poly, lower_band, upper_band = get_talib_poly_channel(np.array(candles[tf]['close']), 3)
        channel_width = upper_band - lower_band
        signal = None
        if candles[tf]['close'][-1] > upper_band:
            signal = 1
        elif candles[tf]['close'][-1] < lower_band:
            signal = -1
        elif candles[tf]['close'][-1] < (upper_band + lower_band) / 2:
            signal = -1
        elif candles[tf]['close'][-1] > (upper_band + lower_band) / 2:
            signal = 1
        mtf_signal[tf] = signal
        if neutral_tf is None and channel_width > 0:
            neutral_tf = tf
        elif channel_width > 0 and abs(channel_width - (upper_band[-2] - lower_band[-2])) < 0.05 * channel_width:
            neutral_tf_typo = tf
    if neutral_tf is not None:
        mtf_signal[neutral_tf] = 0
    if neutral_tf_typo is not None:
        mtf_signal[neutral_tf_typo] = 0
    return mtf_signal

def long_condition(candles, stop_loss, take_profit):
    """
    Determines if a long position should be opened based on the sinewave talib from 1min with close price data from 3min tf.
    Returns True if a long position should be opened, False otherwise.
    """
    print("Running short_condition()...")
    tf_3m_close = np.array(candles['3m'])[:, 3]
    sw = talib.SINWMA(tf_3m_close, timeperiod=5)
    sw_diff = np.diff(sw)
    max_sw = np.max(sw)
    min_sw = np.min(sw)
    exit_point = (max_sw + min_sw) / 2 + (max_sw - min_sw) / 4  # Exit point is 3/4 of the way up the sine wave
    entry_point = (max_sw + min_sw) / 2 - (max_sw - min_sw) / 4  # Entry point is 3/4 of the way down the sine wave
    stop_loss_price = candles['1m'][-1]['close'] * (1 - stop_loss)
    take_profit_price = candles['1m'][-1]['close'] * (1 + take_profit)
    
    print("Sine Wave:", sw)
    print("Sine Wave Differences:", sw_diff)
    print("Max Sine Wave Value:", max_sw)
    print("Min Sine Wave Value:", min_sw)
    print("Entry Point:", entry_point)
    print("Exit Point:", exit_point)
    print("Stop loss percentage:", stop_loss)
    print("Take profit percentage:", take_profit)
    print("Stop loss price:", stop_loss_price)
    print("Take profit price:", take_profit_price)

    if (sw[-1] < entry_point) and (sw_diff[-1] < sw_diff[-2]):
        print("Long position condition met")
        return True, stop_loss_price, take_profit_price
    else:
        print("Long position condition not met")
        return False, None, None

def short_condition(candles, stop_loss, take_profit):
    """
    Determines if a short position should be opened based on the sinewave talib from 1min with close price data from 3min tf.
    Returns True if a short position should be opened, False otherwise.
    """
    print("Running short_condition()...")
    tf_3m_close = np.array(candles['3m'])[:, 3]
    sw = talib.SINWMA(tf_3m_close, timeperiod=5)
    sw_diff = np.diff(sw)
    max_sw = np.max(sw)
    min_sw = np.min(sw)
    entry_point = (max_sw + min_sw) / 2 + (max_sw - min_sw) / 4  # Entry point is 3/4 of the way up the sine wave
    exit_point = (max_sw + min_sw) / 2 - (max_sw - min_sw) / 4  # Exit point is 3/4 of the way down the sine wave
    stop_loss_price = candles['1m'][-1]['close'] * (1 + stop_loss)
    take_profit_price = candles['1m'][-1]['close'] * (1 - take_profit)
    print("Sine wave values:", sw)
    print("Sine wave differences:", sw_diff)
    print("Max sine wave value:", max_sw)
    print("Min sine wave value:", min_sw)
    print("Entry point for short position:", entry_point)
    print("Exit point for short position:", exit_point)
    print("Stop loss percentage:", stop_loss)
    print("Take profit percentage:", take_profit)
    print("Current candle close price:", candles['1m'][-1]['close'])
    print("Stop loss price:", stop_loss_price)
    print("Take profit price:", take_profit_price)

    if (sw[-1] > entry_point) and (sw_diff[-1] > sw_diff[-2]):
        print("Short condition met!")
        return True, stop_loss_price, take_profit_price
    else:
        print("Short condition not met.")
        return False, None, None

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

def main():
    open_positions = client.futures_position_information()
    side = get_trade_side()
    
    # Check for stop loss or take profit
    else:
        closed_positions = []
        for position in open_positions:
            if position['positionAmt'] != '0' and position['positionSide'] == side and position['symbol'] == TRADE_SYMBOL:
                current_price = float(client.futures_symbol_ticker(symbol=TRADE_SYMBOL)['price'])
                stop_loss_price = float(position['entryPrice']) * (1 - STOP_LOSS_THRESHOLD)
                take_profit_price = float(position['entryPrice']) * (1 + TAKE_PROFIT_THRESHOLD)
                if current_price <= stop_loss_price:
                    print(f"{trade_type.upper()} TRADE - STOP LOSS HIT - Closing position...")
                    order = client.futures_create_order(
                        symbol=TRADE_SYMBOL,
                        side=Client.SIDE_SELL if side == Client.SIDE_BUY else Client.SIDE_BUY,
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=abs(float(position['positionAmt'])),
                        reduceOnly=True
                    )
                    closed_positions.append(position['orderId'])
                elif current_price >= take_profit_price:
                    print(f"{trade_type.upper()} TRADE - TAKE PROFIT HIT - Closing position...")
                    order = client.futures_create_order(
                        symbol=TRADE_SYMBOL,
                        side=Client.SIDE_SELL if side == Client.SIDE_BUY else Client.SIDE_BUY,
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=abs(float(position['positionAmt'])),
                        reduceOnly=True
                    )
                    closed_positions.append(position['orderId'])

        # Close all positions in case of reversal of sine
        if reversal_of_sine():
            print(f"REVERSAL OF SINE - CLOSING ALL POSITIONS...")
            for position in open_positions:
                if position['positionAmt'] != '0' and position['symbol'] == TRADE_SYMBOL:
                    order = client.futures_create_order(
                        symbol=TRADE_SYMBOL,
                        side=Client.SIDE_SELL if position['positionSide'] == Client.SIDE_BUY else Client.SIDE_BUY,
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=abs(float(position['positionAmt'])),
                        reduceOnly=True
                    )
                    closed_positions.append(position['orderId'])

        if closed_positions:
            update_trade_log(closed_positions)
