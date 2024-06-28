import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import matplotlib.pyplot as plt

def prepare_data(df):
    
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Set 'Date' as index
    df.set_index('Date', inplace=True)
    
    
    return df[['Adj Close']]

def calculate_log_returns(df):
    log_returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # Calculate historical volatility (20-day rolling standard deviation)
    historical_volatility = log_returns.rolling(window=20).std() * np.sqrt(252)
    
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
    # Assuming ticker_data is a DataFrame with 'Date' and 'Adj Close' columns
    ticker_data = prepare_data(ticker_data)
    log_returns, historical_volatility = calculate_log_returns(ticker_data)
    
    # Fit GARCH model
    best_model, best_model_name = fit_univariate_garch_models(log_returns.dropna(), 'Ticker')
    
    # Generate GARCH volatility forecast
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

# This function can be used for multiple tickers if needed
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
    # Example usage
    df = pd.read_csv('scraped_yahoo_finance_data_minimal.csv')
    results = analyze_multiple_tickers(df)
    # You can now access results for each ticker
    # e.g., results['NKTX']['log_returns'], results['NKTX']['historical_volatility'], etc.
