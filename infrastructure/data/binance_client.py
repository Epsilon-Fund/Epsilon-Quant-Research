from binance.client import Client
import pandas as pd
import yaml
import os

def get_binance_client():    ### Authentication for Binance API
    
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    return Client(
        api_key=config['binance']['api_key'],
        api_secret=config['binance']['api_secret']
    )

def get_data(client, symbol, interval, lookback):   ### Fetch historical data for a given symbol, interval, and lookback period

    klines = client.get_historical_klines(symbol, interval, f'{lookback} days ago UTC')
    
    df = pd.DataFrame(klines, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                       'Close_time', 'Quote_volume', 'Trades', 'Taker_base', 
                                       'Taker_quote', 'Ignore'])
    
    df = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    
    df = df.set_index('Time')
    
    df = df.astype(float)
    
    return df

def get_multiple_data(client, symbols, intervals, lookback):   ### Fetch historical data for multiple symbols and intervals, returning a dictionary of DataFrames

    if isinstance(intervals, str):
        intervals = [intervals] * len(symbols)
    
    data_dict = {}
    for symbol, interval in zip(symbols, intervals):
        key = f"{symbol}_{interval}"
        data_dict[key] = get_data(client, symbol, interval, lookback)
        print(f"✓ Fetched {key}")
    
    return data_dict

