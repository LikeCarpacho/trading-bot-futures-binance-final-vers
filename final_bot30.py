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
TAKE_PROFIT_THRESHOLD = 0.050 # 5% take profit threshold
SINEWAVE_PERIOD = 12  # 12 periods for sinewave
ENTRY_MOMENTUM_THRESHOLD = 3  # 3 consecutive candles above or below sinewave
REVERSAL_KEY_POINT = 6  # 6 periods after entry momentum for reversal key point

print()

# Variables
closed_positions = []

# Print account balance
print("USDT Futures balance:", get_account_balance())

def get_talib_poly_channel(data, degree):
    data = data.flatten()
    pr = talib.LINEARREG(data, timeperiod=degree)
    upper_channel = pr + 2 * talib.STDDEV(pr, timeperiod=degree)
    lower_channel = pr - 2 * talib.STDDEV(pr, timeperiod=degree)
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
    signals = {}
    for tf in timeframes:
        # Get the OHLCV data for the desired timeframe
        data = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles[tf]], dtype=np.double)

        # Get the upper and lower channels using a 20-degree polynomial regression
        upper_channel, lower_channel = get_talib_poly_channel(data, 20)
        
        # Get the last close price for the current timeframe
        close = data[-1][-2]
        
        # Check if the close price is above the lower channel on the current timeframe, or if it is above the upper channel on all timeframes
        if (isinstance(lower_channel, np.ndarray) and close <= lower_channel[-1]) or all([close >= get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[0][-1] if isinstance(get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[0], np.ndarray) else False for tf in timeframes]):
            signals[tf] = "BUY"
        # Check if the close price is below the upper channel on the current timeframe, or if it is below the lower channel on all timeframes
        elif (isinstance(upper_channel, np.ndarray) and close >= upper_channel[-1]) or all([close <= get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[1][-1] if isinstance(get_talib_poly_channel(np.array([c['close'] for c in candles[tf]], dtype=np.double), 20)[1], np.ndarray) else False for tf in timeframes]):
            signals[tf] = "SELL"
        else:
            signals[tf] = "NEUTRAL"

    # If there are neutral signals on some timeframes, filter them out and return the remaining signal
    signals = {tf: signal for tf, signal in signals.items() if signal != "NEUTRAL"}
    return signals





get_mtf_signal(candles, timeframes)

time.sleep(5)
      
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

def close_position(client, symbol, side, quantity):
    """Close the open position with a market order"""
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        return True
    except Exception as e:
        print(e)
        return False


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
    # Read API key and secret from credentials file
    with open('credentials.txt') as f:
        api_key = f.readline().strip()
        api_secret = f.readline().strip()

    # Create client object
    client = Client(api_key, api_secret)

    # Trading variables
    last_position_type = None
    stop_loss_price = None
    take_profit_price = None

    STOP_LOSS = 0.025  # Stop loss percentage
    TAKE_PROFIT = 0.05  # Stop loss percentage
    LEVERAGE = 20


    # Define the candle and timeframe variables
    candles = 50
    timeframes = ['1d', '12h', '8h', '4h', '2h', '1h', '30m', '15m', '5m', '3m', '1m']

    # Call the get_mtf_signal() function with the necessary arguments
    #trigger = get_mtf_signal(candles, timeframes)

    # Print the trigger
    #print(trigger)

    # Infinite loop for continuous trading
    while True:
        try:
            # Get current price of symbol
            ticker = client.futures_symbol_ticker(symbol=TRADE_SYMBOL)
            current_price = float(ticker['price'])

            # Check if there is an open position
            if last_position_type is None:
                # No open position, check for entry
                if current_price < (1 - STOP_LOSS) * current_price:
                    # Enter LONG position
                    order = client.futures_create_order(symbol=TRADE_SYMBOL,
                                                        side='BUY',
                                                        type='MARKET',
                                                        quantity=TRADE_QUANTITY,
                                                        leverage=LEVERAGE)
                    last_position_type = 'LONG'
                    stop_loss_price = current_price * (1 - STOP_LOSS)
                    take_profit_price = current_price * (1 + TAKE_PROFIT)
                    print(f'Entered LONG position at {current_price}')
                elif current_price > (1 + STOP_LOSS) * current_price:
                    # Enter SHORT position
                    order = client.futures_create_order(symbol=TRADE_SYMBOL,
                                                        side='SELL',
                                                        type='MARKET',
                                                        quantity=TRADE_QUANTITY,
                                                        leverage=LEVERAGE)
                    last_position_type = 'SHORT'
                    stop_loss_price = current_price * (1 + STOP_LOSS)
                    take_profit_price = current_price * (1 - TAKE_PROFIT)
                    print(f'Entered SHORT position at {current_price}')
            else:
                # There is an open position, check for exit
                if last_position_type == 'LONG' and (current_price >= take_profit_price or current_price <= stop_loss_price):
                    # Close LONG position
                    order = client.futures_create_order(symbol=TRADE_SYMBOL,
                                                        side='SELL',
                                                        type='MARKET',
                                                        quantity=TRADE_QUANTITY,
                                                        leverage=LEVERAGE)
                    last_position_type, stop_loss_price, take_profit_price = None, None, None
                    print(f'Closed LONG position at {current_price}')
                    
                    # Open new position using opposite side if stop loss is hit
                    if current_price <= stop_loss_price:
                        # Enter SHORT position
                        order = client.futures_create_order(symbol=TRADE_SYMBOL,
                                                            side='SELL',
                                                            type='MARKET',
                                                            quantity=TRADE_QUANTITY,
                                                            leverage=LEVERAGE)
                        last_position_type = 'SHORT'
                        stop_loss_price = current_price * (1 + STOP_LOSS)
                        take_profit_price = current_price * (1 - TAKE_PROFIT)
                        print(f'Entered SHORT position at {current_price}')

                elif last_position_type == 'SHORT' and (current_price <= take_profit_price or current_price >= stop_loss_price):
                    # Close SHORT position
                    order = client.futures_create_order(symbol=TRADE_SYMBOL,
                                                    side='BUY',
                                                    type='MARKET',
                                                    quantity=TRADE_QUANTITY,
                                                    leverage=LEVERAGE)
                    last_position_type, stop_loss_price, take_profit_price = None, None, None
                    print(f'Closed SHORT position at {current_price}')

                # Open new trade in the opposite direction
                if last_position_type is None:
                    if current_price < (1 - STOP_LOSS) * current_price:
                        # Enter LONG position
                        order = client.futures_create_order(symbol=TRADE_SYMBOL,
                                                            side='BUY',
                                                            type='MARKET',
                                                            quantity=TRADE_QUANTITY,
                                                            leverage=LEVERAGE)
                        last_position_type = 'LONG'
                        stop_loss_price = current_price * (1 - STOP_LOSS)
                        take_profit_price = current_price * (1 + TAKE_PROFIT)
                        print(f'Entered LONG position at {current_price}')
                    elif current_price > (1 + STOP_LOSS) * current_price:
                        # Enter SHORT position
                        order = client.futures_create_order(symbol=TRADE_SYMBOL,
                                                            side='SELL',
                                                            type='MARKET',
                                                            quantity=TRADE_QUANTITY,
                                                            leverage=LEVERAGE)
                        last_position_type = 'SHORT'
                        stop_loss_price = current_price * (1 + STOP_LOSS)
                        take_profit_price = current_price * (1 - TAKE_PROFIT)
                        print(f'Entered SHORT position at {current_price}')

            # Wait for a bit before repeating the loop
            print('btc current price at: ', current_price)
            time.sleep(5)

        except BinanceAPIException as e:
            print(e)
            time.sleep(5)
        except BinanceOrderException as e:
            print(e)
            time.sleep(5)
        except Exception as e:
            print(e)
            time.sleep(5)


if __name__ == '__main__':
    main()