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

print()

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

def load_data(symbol, interval, start_time, end_time):
    """
    Fetches historical candlestick data for a symbol and interval
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time * 1000}&endTime={end_time * 1000}"
    response = requests.get(url)
    response_json = json.loads(response.text)
    return response_json

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
            'open_time': candle[0],
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5],
            'close_time': candle[6],
            'quote_asset_volume': candle[7],
            'number_of_trades': candle[8],
            'taker_buy_base_asset_volume': candle[9],
            'taker_buy_quote_asset_volume': candle[10]
        })

# Print candles for all timeframes
for interval in candles:
    print(f"{interval} candles:")
    for candle in candles[interval]:
        print(candle)
        print("--------")

def get_mtf_signal(candles, timeframes):
    """
    Calculates the MTF signals using polynomial regression channels for each timeframe.
    Returns a dictionary containing the MTF signals for each timeframe.
    """
    mtf_signal = {}
    neutral_tf = []

    # Loop through each timeframe
    for tf in timeframes:
        data = np.array([c['close'] for c in candles[timeframes.index(tf)]])
        upper_channel, lower_channel = get_talib_poly_channel(data, 20)  # Use 20 degree polynomial regression

        # Calculate the signal based on whether the close is below the lowest channel or above the highest channel
        close = candles[timeframes.index(tf)][-1]['close']
        if close <= lower_channel[-1] or all([close <= get_talib_poly_channel(np.array([c['close'] for c in candles[timeframes.index(tf)]]), 20)[1][-1] for tf in timeframes]):
            signal = "LONG"
        elif close >= upper_channel[-1] or all([close >= get_talib_poly_channel(np.array([c['close'] for c in candles[timeframes.index(tf)]]), 20)[0][-1] for tf in timeframes]):
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
            neutral_tf.append(tf)
        
def get_all_lows(prices):
    lows = []
    for i in range(len(prices)):
        if i == 0:
            lows.append(prices[i])
        else:
            if prices[i] < lows[-1]:
                lows.append(prices[i])
            else:
                lows.append(lows[-1])
    return lows

def get_all_highs(prices):
    highs = []
    for i in range(len(prices)):
        if i == 0:
            highs.append(prices[i])
        else:
            if prices[i] > highs[-1]:
                highs.append(prices[i])
            else:
                highs.append(highs[-1])
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

def calculate_sine_momentum(candles, timeframe, current_price):
    """
    Calculates the sine wave momentum
    """
    highs = np.array([c['high'] for c in candles[timeframe]])
    lows = np.array([c['low'] for c in candles[timeframe]])
    close_prices = np.array([c['close'] for c in candles[timeframe]])

    # Calculate Talib's sine wave
    sinewave = talib.HT_SINE(close_prices)

    sinewave_mom = np.zeros_like(close_prices)

    for i in range(1, len(close_prices)):
        if close_prices[i] < np.min(close_prices) + np.min(highs - lows) * 0.1 or sinewave[i] == np.max(sinewave):
            # Bullish entry signal
            sinewave_mom[i] = 1
        elif sinewave[i] == np.min(sinewave):
            # Bearish entry signal
            sinewave_mom[i] = -1
        elif sinewave[i] == np.max(sinewave):
            # Bullish exit signal
            sinewave_mom[i] = -1
        elif sinewave[i] == np.min(sinewave):
            # Bearish exit signal
            sinewave_mom[i] = 1
        else:
            sinewave_mom[i] = 0

    return sinewave_mom[-1] * current_price

def get_latest_price(client, symbol):
    """Get the latest price of a given symbol."""
    ticker = client.futures_ticker(symbol=symbol)
    return float(ticker['lastPrice'])


print()
print("init main() function...")
print()

def main():
    symbol = 'BTCUSDT'
    initial_trade_type = 'BUY'
    quantity = 0.01
    stop_loss_percent = 2
    take_profit_percent = 4
    reversal_key_points = [np.pi/2, 3*np.pi/2]
    trade_count = 0

    client = Client(api_key, api_secret)

    while True:
        if trade_count == 0:
            entry_price = get_latest_price(client, symbol)
            if initial_trade_type == 'BUY':
                stop_loss_price = entry_price - (entry_price * stop_loss_percent / 100)
                take_profit_price = entry_price + (entry_price * take_profit_percent / 100)
            else:
                stop_loss_price = entry_price + (entry_price * stop_loss_percent / 100)
                take_profit_price = entry_price - (entry_price * take_profit_percent / 100)

            print(f"Initial {initial_trade_type} Entry Price: {entry_price:.2f}")
            print(f"Stop Loss Price: {stop_loss_price:.2f}")
            print(f"Take Profit Price: {take_profit_price:.2f}")

            order = client.futures_create_order(
                symbol=symbol,
                side=initial_trade_type,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            print(f"Initial {initial_trade_type} Entry Order Placed: {order}")

            latest_price = get_latest_price(client, symbol)

            if initial_trade_type == 'BUY':
                if latest_price >= take_profit_price:
                    close_position(client, symbol, 'SELL', quantity)
                    trade_count += 1
                elif latest_price <= stop_loss_price:
                    close_position(client, symbol, 'SELL', quantity)
                    initial_trade_type = 'SELL'
                    trade_count += 1
            else:
                if latest_price <= take_profit_price:
                    close_position(client, symbol, 'BUY', quantity)
                    trade_count += 1
                elif latest_price >= stop_loss_price:
                    close_position(client, symbol, 'BUY', quantity)
                    initial_trade_type = 'BUY'
                    trade_count += 1

        else:
            sine_wave = get_sine_wave(reversal_key_points)
            entry_price = get_latest_price(client, symbol)

            if sine_wave[0] < entry_price < sine_wave[1]:
                latest_price = get_latest_price(client, symbol)

                if initial_trade_type == 'BUY':
                    if latest_price >= take_profit_price:
                        close_position(client, symbol, 'SELL', quantity)
                        trade_count += 1
                        break
                    elif latest_price <= stop_loss_price:
                        close_position(client, symbol, 'SELL', quantity)
                        initial_trade_type = 'SELL'
                        trade_count += 1
                        break
                else:
                    if latest_price <= take_profit_price:
                        close_position(client, symbol, 'BUY', quantity)
                        trade_count += 1
                        break
                    elif latest_price >= stop_loss_price:
                        close_position(client, symbol, 'BUY', quantity)
                        initial_trade_type = 'BUY'
                        trade_count += 1
                        break

                time.sleep(10)
            else:
                if initial_trade_type == 'BUY':
                    close_position(client, symbol, 'SELL', quantity)
                    initial_trade_type = 'SELL'
                    trade_count += 1
                else:
                    close_position(client, symbol, 'BUY', quantity)
                    initial_trade_type = 'BUY'
                    trade_count += 1

                time.sleep(10)



if __name__ == '__main__':
    main()
