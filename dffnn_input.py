import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import matplotlib.pyplot as plt

def prepare_data(df):
    if df.index.name == 'Date':
        df = df.reset_index()

   
    if 'Date' not in df.columns:
        print("Warning: 'Date' column not found in the DataFrame. Using index as date.")
        df['Date'] = df.index

    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.sort_values('Date')
    
    # Set 'Date' as index and drop the column
    df.set_index('Date', inplace=True)

    if 'Adj Close' not in df.columns:
        print("Warning: 'Adj Close' column not found. Using 'Close' column.")
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' column found in the DataFrame.")
    
    return df[['Adj Close']]
    
    

def calculate_log_returns(df):
    log_returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # calc historical volatility (20-day rolling standard deviation)
    historical_volatility = log_returns.rolling(window=30).std() * np.sqrt(252)
    
    return log_returns, historical_volatility

def fit_univariate_garch_models(returns, ticker):
   
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    results = model.fit(disp='off')
    
    
    egarch_model = arch_model(returns, vol='EGARCH', p=1, q=1)
    egarch_results = egarch_model.fit(disp='off')
    
    
    gjr_model = arch_model(returns, vol='GARCH', p=1, o=1, q=1)
    gjr_results = gjr_model.fit(disp='off')
    
    
    models = [results, egarch_results, gjr_results]
    best_model = min(models, key=lambda x: x.aic)
    
    model_names = ['GARCH(1,1)', 'EGARCH(1,1)', 'GJR-GARCH(1,1)']
    best_model_name = model_names[models.index(best_model)]
    
    print(f"Best model for {ticker}: {best_model_name}")
    print(best_model.summary())
    
    return best_model, best_model_name

def testmaybe(ticker_data):
    ticker_data = prepare_data(ticker_data)
    log_returns, historical_volatility = calculate_log_returns(ticker_data)
    
   
    best_model, best_model_name = fit_univariate_garch_models(log_returns.dropna(), 'Ticker')
    
    
    garch_volatility = pd.Series(best_model.conditional_volatility, index=log_returns.dropna().index)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(historical_volatility, label='Historical Volatility')
    plt.plot(garch_volatility, label=f'{best_model_name} Volatility')
    plt.title('Historical vs GARCH Volatility')
    plt.legend()
    plt.show()
    
    # Perform Augmented Dickey-Fuller test
    adf_result = adfuller(log_returns.dropna())
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    
    # Perform Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(log_returns.dropna())
    print('Jarque-Bera statistic:', jb_stat)
    print('Jarque-Bera p-value:', jb_pvalue)
    
    return log_returns, historical_volatility, garch_volatility


def analyze_multiple_tickers(df):
    results = {}
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        log_returns, hist_vol, garch_vol = testmaybe(ticker_data)
        results[ticker] = {
            'log_returns': log_returns,
            'historical_volatility': hist_vol,
            'garch_volatility': garch_vol
        }
    return results

if __name__ == "__main__":
    df = pd.read_csv('scraped_yahoo_finance_data_minimal.csv')
    results = analyze_multiple_tickers(df)
