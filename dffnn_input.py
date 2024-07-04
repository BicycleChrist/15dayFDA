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
    
    return df['Adj Close']



def calculate_log_returns_alt(adjclose):
    log_returns = np.log(adjclose / adjclose.shift(1))
    
    # calc historical volatility (20-day rolling standard deviation)
    historical_volatility = log_returns.rolling(window=30).std() * np.sqrt(252)
    
    return log_returns, historical_volatility


def calculate_log_returns(df):
    log_returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # calc historical volatility (20-day rolling standard deviation)
    historical_volatility = log_returns.rolling(window=30).std() * np.sqrt(252)
    
    return log_returns, historical_volatility


def fit_multiple_garch_models(returns):
    models = {
        'APGARCH': {
            'ged': arch_model(returns, vol='APARCH', p=1, q=1, dist='ged'),
            'normal': arch_model(returns, vol='APARCH', p=1, q=1, dist='normal'),
            'studentst': arch_model(returns, vol='APARCH', p=1, q=1, dist='studentst'),
            'skewt': arch_model(returns, vol='APARCH', p=1, q=1, dist='skewt'),
        },
        'EGARCH': {
            'ged': arch_model(returns, vol='EGARCH', p=1, q=1, dist='ged'),
            'normal': arch_model(returns, vol='EGARCH', p=1, q=1, dist='normal'),
            'studentst': arch_model(returns, vol='EGARCH', p=1, q=1, dist='studentst'),
            'skewt': arch_model(returns, vol='EGARCH', p=1, q=1, dist='skewt'),
        },
        'GARCH': {
            'ged': arch_model(returns, vol='GARCH', p=1, q=1, dist='ged'),
            'normal': arch_model(returns, vol='GARCH', p=1, q=1, dist='normal'),
            'studentst': arch_model(returns, vol='GARCH', p=1, q=1, dist='studentst'),
            'skewt': arch_model(returns, vol='GARCH', p=1, q=1, dist='skewt'),
        }
    }

    results = {}
    for model_type, dist_models in models.items():
        for dist, model in dist_models.items():
            try:
                result = model.fit(disp='off')
                results[f'{model_type}_{dist}'] = result
            except:
                print(f"Failed to fit {model_type} model with {dist} distribution")

    best_model = min(results.values(), key=lambda x: x.aic)
    best_model_name = [k for k, v in results.items() if v == best_model][0]

    print(f"Best model: {best_model_name}")
    print(best_model.summary())

    return results, best_model, best_model_name

def prepare_garch_features(returns):
    models, best_model, best_model_name = fit_multiple_garch_models(returns)
    
    features = pd.DataFrame(index=returns.index)
    for model_name, model in models.items():
        features[f'HV_{model_name}'] = pd.Series(model.conditional_volatility, index=returns.index)
    
    return features, best_model, best_model_name

def testmaybe(ticker_data):
    ticker_data = prepare_data(ticker_data)
    log_returns, historical_volatility = calculate_log_returns(ticker_data)
    
    garch_features, best_model, best_model_name = prepare_garch_features(log_returns.dropna())
    
    # Combine historical volatility with GARCH features
    all_features = pd.concat([historical_volatility, garch_features], axis=1).dropna()
    all_features.columns = ['HV'] + list(garch_features.columns)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(all_features['HV'], label='Historical Volatility')
    plt.plot(all_features[f'HV_{best_model_name}'], label=f'Best Model: {best_model_name}')
    plt.title('Historical vs Best GARCH Model Volatility')
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
    
    return log_returns, all_features

def analyze_multiple_tickers(df):
    results = {}
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        log_returns, features = testmaybe(ticker_data)
        results[ticker] = {
            'log_returns': log_returns,
            'features': features
        }
    return results

if __name__ == "__main__":
    df = pd.read_csv('scraped_yahoo_finance_data_minimal.csv')
    results = analyze_multiple_tickers(df)
