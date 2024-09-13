import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from upcomingevents import Main  

indx_tickers = []

def read_tickers_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def fetch_yahoo_finance_data(ticker, start_date, end_date):
    try:
        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.history(start=start_date, end=end_date)
        if data.empty:
            print(f"No data available for {ticker}")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        data['Ticker'] = ticker
        # Check if 'Adj Close' is in the columns, if not, use 'Close' instead
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        return data[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    except Exception as exc:
        print(f"{ticker} generated an exception: {exc}")
        return pd.DataFrame()

def process_and_fetch(use_events=False, file_path=None):
    print(f"process_and_fetch: use_events={use_events}, file_path={file_path}\n")
    all_tickers = indx_tickers.copy()
    
    if use_events:
        df_events = Main()
        event_tickers = df_events['Company'].unique()
        event_tickers = [ticker for ticker in event_tickers if ticker != "MeetingN/A"]
        all_tickers.extend(event_tickers)
    
    if file_path is not None:
        file_tickers = read_tickers_from_file(file_path)
        all_tickers.extend(file_tickers)
    
    # Remove duplicates
    all_tickers = list(dict.fromkeys(all_tickers))
    print(f"All Tickers: {all_tickers}")

    start_date = '2015-07-01'
    end_date = '2024-06-30'

    all_data = []
    failed_tickers = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_ticker = {executor.submit(fetch_yahoo_finance_data, ticker, start_date, end_date): ticker for ticker in all_tickers}

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if not data.empty:
                    all_data.append(data)
                else:
                    failed_tickers.append(ticker)
            except Exception as exc:
                print(f"{ticker} generated an exception: {exc}")
                failed_tickers.append(ticker)

    if all_data:
        df_all_data = pd.concat(all_data, ignore_index=True)
        # Ensure Date is in datetime format
        df_all_data['Date'] = pd.to_datetime(df_all_data['Date'])
        # Sort by Ticker first, then by Date in descending order
        df_all_data = df_all_data.sort_values(['Ticker', 'Date'], ascending=[True, False])
        # Format the date after sorting
        df_all_data['Date'] = df_all_data['Date'].dt.strftime('%b %d, %Y')

        df_all_data.to_csv('scraped_yahoo_finance_datatest.csv', index=False)
        print('Fetched data has been saved to scraped_yahoo_finance_data.csv')
    else:
        print('No data was fetched successfully.')

    if failed_tickers:
        print(f"\nFailed to fetch data for the following tickers: {', '.join(failed_tickers)}")

if __name__ == "__main__":
    process_and_fetch(False, file_path="tickerslist.txt")
    #process_and_fetch(file_path="tickerslist.txt")