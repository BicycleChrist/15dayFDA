import os
import pandas as pd
import numpy as np
from arch import arch_model
from mvgarch.ugarch import UGARCH
from mvgarch.mgarch import DCCGARCH
import matplotlib.pyplot as plt

def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.pivot(columns='Ticker', values='Close')
    df.dropna(inplace=True)
    return df

def calculate_log_returns(df):
    log_returns = np.log(df / df.shift(1)).dropna()
    return log_returns

def fit_univariate_garch_models(df, ticker):
    ticker_data = df[ticker].dropna() * 100
    models = {'GARCH(1,1)': arch_model(ticker_data, vol='GARCH', p=1, q=1),
              'EGARCH(1,1)': arch_model(ticker_data, vol='EGARCH', p=1, q=1),
              'GJR-GARCH(1,1)': arch_model(ticker_data, vol='GARCH', p=1, q=1, o=1),
              'GARCH(2,1)': arch_model(ticker_data, vol='EGARCH', p=2, q=1),
              'GARCH(1,2)': arch_model(ticker_data, vol='EGARCH', p=1, q=2),
              }

    results = {}

    for model_name, model in models.items():
        model_fitted = model.fit(disp="off")
        aic = model_fitted.aic
        bic = model_fitted.bic
        results[model_name] = {'model': model_fitted, 'AIC': aic, 'BIC': bic}

        # Save conditional vs realized volatility plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model_fitted.conditional_volatility, label='Conditional Volatility')
        ax.plot(np.sqrt(252) * ticker_data.std(), label='Realized Volatility (annualized)')
        ax.set_title(f'{ticker} - {model_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.legend()

        if not os.path.exists('Volanalysisresults'):
            os.makedirs('Volanalysisresults')
        plt.savefig(os.path.join('Volanalysisresults', f'{ticker}_{model_name}_volatility.png'))
        plt.close()

    best_model_name = min(results, key=lambda k: results[k]['AIC'])
    best_model = results[best_model_name]['model']

    return best_model, results

#TODO: DCC model fit runs (CPU 100 %), but nothing else seems to work afterwards. Likely cuz I suck
#def fit_higher_order_models(df):
#    returns = df * 100
#    ugarch_models = {}
#    for ticker in df.columns:
#        ticker_data = returns[ticker].dropna()
#        garch = UGARCH(order=(1, 1))
#        garch.spec(returns=ticker_data)
#        garch.fit()
#        ugarch_models[ticker] = garch
#
#    if len(returns.columns) > 1:
#        garch_specs = [ugarch_models[ticker] for ticker in returns.columns]
#        dcc_garch = DCCGARCH()
#        dcc_garch.spec(ugarch_objs=garch_specs, returns=returns)
#        dcc_garch.fit()
#        log_likelihood = dcc_garch.qllf()
#
#        return dcc_garch, log_likelihood
#
#    return None, None

def main():
    df = pd.read_csv('scraped_yahoo_finance_data.csv')
    df = prepare_data(df)
    log_returns = calculate_log_returns(df)
    tickers = df.columns

    for ticker in tickers:
        print(f'Analyzing volatility for {ticker}...')
        best_model, results = fit_univariate_garch_models(log_returns, ticker)
        print(f'Best model for {ticker} based on AIC: {best_model}')
        for model_name, result in results.items():
            print(f'{model_name} - AIC: {result["AIC"]}, BIC: {result["BIC"]}')

   # dcc_garch, log_likelihood = fit_higher_order_models(log_returns)
   # if dcc_garch:
   #     print('DCC-GARCH Model Summary:')
   #     print(f'Log-Likelihood: {log_likelihood}')
   #     # Optionally, you can also save plots or other outputs for DCC-GARCH here.
   # else:
   #     print('DCC-GARCH model fitting failed or not enough data for DCC-GARCH.')

if __name__ == "__main__":
    main()
