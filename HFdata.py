import requests
import pandas as pd
from polygonKEY import polygonkey

def fetch_data():
    api_key = polygonkey  # Ensure your API key is correctly referenced
    symbol = 'AMD'
    date = '2024-02-05/2024-02-09'
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date}?apiKey={api_key}'
    
    response = requests.get(url)
    data = response.json()
    
    if data and 'results' in data:
        df_midas = pd.DataFrame(data['results'])
        df_midas['timestamp'] = pd.to_datetime(df_midas['t'], unit='ms')
        df_midas.set_index('timestamp', inplace=True)
        df_midas.drop(columns=['t'], inplace=True)
        df_midas.columns = ['volume', 'vwap', 'open', 'close', 'high', 'low', 'number_of_trades']
        print(df_midas)
    else:
        print(f"Error fetching data: {data}")
        return None
    
    # Check for pagination and fetch additional data if necessary
    while 'next_url' in data:
        response = requests.get(data['next_url'])
        data = response.json()
        
        if data and 'results' in data:
            new_data = pd.DataFrame(data['results'])
            new_data['timestamp'] = pd.to_datetime(new_data['t'], unit='ms')
            new_data.set_index('timestamp', inplace=True)
            new_data.drop(columns=['t'], inplace=True)
            new_data.columns = ['volume', 'vwap', 'open', 'close', 'high', 'low', 'number_of_trades']
            df_midas = pd.concat([df_midas, new_data])
        else:
            print(f"Error fetching additional data: {data}")
    
    return df_midas

if __name__ == "__main__":
    df_midas = fetch_data()
    if df_midas is not None:
        df_midas.to_csv('hf_data.csv', index=True)
        print(df_midas)
    else:
        print("Failed to fetch data.")


