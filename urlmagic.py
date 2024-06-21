from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from upcomingevents import Main  # Import the Main function

# date to Unix timestamp 
def date_to_unix_timestamp(year, month, day):
    return int(datetime(year, month, day).timestamp())

# generate Yahoo Finance URL
def generate_yahoo_finance_url(ticker, start_date, end_date):
    period1 = date_to_unix_timestamp(*start_date)
    period2 = date_to_unix_timestamp(*end_date)
    url = f'https://finance.yahoo.com/quote/{ticker}/history/?period1={period1}&period2={period2}'
    return url


def scrape_yahoo_finance(ticker, start_date, end_date):
    url = generate_yahoo_finance_url(ticker, start_date, end_date)
    options = FirefoxOptions()
    options.add_argument('--headless')  
    options.set_preference('permissions.default.image', 2)
    options.page_load_strategy = 'eager'
    service = Service()
    driver = webdriver.Firefox(service=service, options=options)

    driver.get(url)

    
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'table.table.svelte-ewueuo'))
    )

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', {'class': 'table svelte-ewueuo'})

    rows = []
    if table:
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 7:  # Ensure we have all the columns we need
                row_data = [cell.get_text(strip=True) for cell in cells]
                rows.append(row_data)

    driver.quit()
    return rows

# acquire price data from YF
def process_and_scrape():
    # DataFrame from the Main function of upcomingevents.py
    df_events = Main()

    # filter out invalid tickers, mostly just meetings/conferences
    tickers = df_events['Company'].unique()
    tickers = [ticker for ticker in tickers if ticker != "MeetingN/A"]

    start_date = (2023, 6, 20)
    end_date = (2024, 6, 20)

    all_data = []

    
    with ThreadPoolExecutor(max_workers=None) as executor:
        future_to_ticker = {executor.submit(scrape_yahoo_finance, ticker, start_date, end_date): ticker for ticker in tickers}

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                for row in data:
                    all_data.append([ticker] + row)
            except Exception as exc:
                print(f"{ticker} generated an exception: {exc}")

    
    columns = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_all_data = pd.DataFrame(all_data, columns=columns)

    
    df_all_data.to_csv('scraped_yahoo_finance_data.csv', index=False)
    print('Scraped data has been saved to scraped_yahoo_finance_data.csv')

if __name__ == "__main__":
    process_and_scrape()
