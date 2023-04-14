import math
import time
import numpy as np
import requests
import talib
import json

# binance module imports
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Initialize Binance client
client = Client(api_key, api_secret)

def get_account_balance():
    """
    Get account balance for USDT in the futures wallet
    """
    account_balance = client.futures_account_balance()
    for balance in account_balance:
        if balance['asset'] == 'USDT':
            return float(balance['balance'])
    return 0.0  # return 0 if balance not found

# Constants
TRADE_SYMBOL = "BTCUSDT"
TRADE_SIZE = 0.01  # Trade size in BTC
TRADE_TYPE = "MARKET"
TRADE_LVRG = 20
STOP_LOSS_THRESHOLD = 0.025  # 2.5% stop loss threshold
TAKE_PROFIT_THRESHOLD = 0.050 # 5% take profit threshold
SINEWAVE_PERIOD = 12  # 12 periods for sinewave
ENTRY_MOMENTUM_THRESHOLD = 3  # 3 consecutive candles above or below sinewave
REVERSAL_KEY_POINT = 6  # 6 periods after entry momentum for reversal key point

print()

# Variables
closed_positions = []

# Print account balance
print("USDT Futures balance:", get_account_balance())

# Define timeframes
timeframes = ['1d', '12h', '8h', '4h', '2h', '1h', '30m', '15m', '5m', '3m', '1m']
print(timeframes)

# Define start and end time for historical data
start_time = int(time.time()) - (86400 * 30)  # 30 days ago
end_time = int(time.time())

print()

# Fetch historical data for BTCUSDT pair
candles = {}
for interval in timeframes:
    tf_candles = client.futures_klines(symbol=TRADE_SYMBOL, interval=interval.lower(), startTime=start_time * 1000, endTime=end_time * 1000)
    candles[interval.lower()] = []
    for candle in tf_candles:
        candles[interval.lower()].append({
            'timestamp': candle[0],
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        })

# Print the historical data for BTCUSDT pair
for interval in timeframes:
    print(f"Data for {interval} interval:")
    print(candles[interval.lower()])

print()

def get_talib_poly_channel(data, degree):
    data = data.flatten()
    pr = talib.LINEARREG(data, timeperiod=degree)
    upper_channel = pr + 2 * talib.STDDEV(pr, timeperiod=degree)
    lower_channel = pr - 2 * talib.STDDEV(pr, timeperiod=degree)
    return upper_channel, lower_channel

def get_mtf_signal(candles, timeframes):
    signals = {}
    for tf in timeframes:
        # Get the OHLCV data for the desired timeframe
        data = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles[tf]], dtype=np.double)

        # Get the upper and lower channels using a 20-degree polynomial regression
        upper_channel, lower_channel = get_talib_poly_channel(data, 20)

        # Get the last close price for the current timeframe
        close = data[-1][-2]

        # Check if the close price is above the lower channel on the current timeframe, or if it is above the upper channel on all timeframes
        if (isinstance(lower_channel, np.ndarray) and close <= lower_channel[-1]):
            # Calculate percentage distance from the close to the ideal dip for dips
            dip_distance = ((lower_channel[-1] - close) / close) * 100
            signals[tf] = f"BUY (Dip: {dip_distance:.2f}%)"
        elif all([close >= get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[0][-1] if isinstance(get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[0], np.ndarray) else False for tf in timeframes]):
            # Calculate percentage distance from the close to the highest line from poly channel for tops
            tops_distance = ((close - upper_channel[-1]) / close) * 100
            signals[tf] = f"BUY (Tops: {tops_distance:.2f}%)"
        # Check if the close price is below the upper channel on the current timeframe, or if it is below the lower channel on all timeframes
        elif (isinstance(upper_channel, np.ndarray) and close >= upper_channel[-1]):
            # Calculate percentage distance from the close to the lowest line from poly channel for dips
            dip_distance = ((close - lower_channel[-1]) / close) * 100
            signals[tf] = f"SELL (Dip: {dip_distance:.2f}%)"
        elif all([close <= get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[1][-1] if isinstance(get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[1], np.ndarray) else False for tf in timeframes]):
            # Calculate percentage distance from the close to the lowest line from poly channel for tops
            tops_distance = ((upper_channel[-1] - close) / close) * 100
            signals[tf] = f"SELL (Tops: {tops_distance:.2f}%)"
        else:
            signals[tf] = "NEUTRAL"

    # If there are neutral signals on some timeframes, filter them out and return the remaining signal
    signals = {tf: signal for tf, signal in signals.items() if signal != "NEUTRAL"}

    # Return the signals dictionary along with the buy or sell signal for each timeframe
    return signals, "BUY" if all([signal.startswith("BUY") for signal in signals.values()]) else "SELL"


# Get the MTF signals
signals, mtf_signal = get_mtf_signal(candles, timeframes)

# Print the signals for all timeframes
print("MTF signals:")
for tf, signal in signals.items():
    print(f"{tf} - {signal}")

# Print the buy/sell signal based on the MTF signals
print("MTF buy/sell signal:", mtf_signal)


