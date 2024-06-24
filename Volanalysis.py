from mvgarch.ugarch import UGARCH
from mvgarch.mgarch import DCCGARCH
import os
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import concurrent.futures


# balance between responsiveness and stability, standard GARCH models (1,2) or (2,1) seeem to be slightly better off
# concerned about overestimating risk, the GJR-GARCH model might be appropriate. 
# one would think that EGARCH is a preferred model for forecasting during periods of heightned vol (I.E. leading up to or following a clincial event)  
# as it can differentiate between the impact of positive and negative shocks on volatility (leverage effect)
# TODO: Give the FIGARCH a shot, implement some degree of forecasting 


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
              'GARCH(2,1)': arch_model(ticker_data, vol='GARCH', p=2, q=1),
              'GARCH(1,2)': arch_model(ticker_data, vol='GARCH', p=1, q=2),
              }

    results = {}

    for model_name, model in models.items():
        model_fitted = model.fit(disp="off")
        aic = model_fitted.aic
        bic = model_fitted.bic
        results[model_name] = {'model': model_fitted, 'AIC': aic, 'BIC': bic}

        # compute realized volatility using a 30-day rolling window
        realized_volatility = ticker_data.rolling(window=30).std()

        # generate/save conditional vs realized volatility plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(realized_volatility, label='Realized Volatility', alpha=0.7, color='blue')
        ax.plot(model_fitted.conditional_volatility, label='Conditional Volatility', alpha=0.7, color="red")
        ax.set_title(f'{ticker} - {model_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True)

        if not os.path.exists('Volanalysisresults'):
            os.makedirs('Volanalysisresults')
        plt.savefig(os.path.join('Volanalysisresults', f'{ticker}_{model_name}_volatility.png'))
        plt.close()

    best_model_name = min(results, key=lambda k: results[k]['AIC'])
    best_model = results[best_model_name]['model']

    return best_model, results

def fit_higher_order_models(df):
    returns = df * 100
    ugarch_models = {}
    for ticker in df.columns:
        ticker_data = returns[ticker].dropna()
        garch = UGARCH(order=(1, 1))
        garch.spec(returns=ticker_data)
        garch.fit()
        ugarch_models[ticker] = garch

    if len(returns.columns) > 1:
        garch_specs = [ugarch_models[ticker] for ticker in returns.columns]
        dcc_garch = DCCGARCH()
        dcc_garch.spec(ugarch_objs=garch_specs, returns=returns)
        dcc_garch.fit()
        return dcc_garch, None
    
    return None, None

def testmaybe(df):
    returns = df * 100
    garch_specs = [UGARCH(order=(1, 1)) for _ in range(len(returns.columns))]
    
    dcc_garch = DCCGARCH()
    dcc_garch.spec(ugarch_objs=garch_specs, returns=returns)
    dcc_garch.fit()
    # DCC GARCH fit
    
    
    return dcc_garch, None

def GetDynamicCorrelation(dcc_garch_model):
    dynamic_correlation_results = DCCGARCH.dynamic_corr(dcc_garch_model.returns, dcc_garch_model.cond_vols, dcc_garch_model.dcc_a, dcc_garch_model.dcc_b)
    return dynamic_correlation_results


def plot_conditional_volatilities(dcc_garch_model, log_returns):
    cond_vols = dcc_garch_model.cond_vols
    
    plt.figure(figsize=(12, 8))
    for i, ticker in enumerate(log_returns.columns):
        plt.plot(log_returns.index, cond_vols[:, i], label=ticker)
    
    plt.title('Conditional Volatilities from DCC-GARCH')
    plt.xlabel('Date')
    plt.ylabel('Conditional Volatility')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists('Volanalysisresults'):
        os.makedirs('Volanalysisresults')
    plt.savefig(os.path.join('dcc_garch_output', 'dcc_garch_conditional_volatilities.png'))
    plt.close()


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

    dcc_garch, log_likelihood = testmaybe(log_returns)
    if dcc_garch:
        print(f"alpha: {dcc_garch.dcc_a}, beta: {dcc_garch.dcc_b}")
        print(f"alpha + beta: {dcc_garch.dcc_a + dcc_garch.dcc_b}")
        dynamic_correlation = GetDynamicCorrelation(dcc_garch)
        print("\ndynamic correlation: \n")
        print(dynamic_correlation)
        plot_conditional_volatilities(dcc_garch, log_returns)
    else:
        print('DCC-GARCH model fitting failed or not enough data for DCC-GARCH.')

if __name__ == "__main__":
    main()


