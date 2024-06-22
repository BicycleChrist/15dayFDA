import os
import pandas as pd
import numpy as np
from arch import arch_model
from mvgarch.mgarch import DCCGARCH
from mvgarch.ugarch import UGARCH
import matplotlib.pyplot as plt
import warnings



def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def fit_garch_model(df, ticker, p=1, q=1):
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data['LogReturn'] = np.log(ticker_data['Close']).diff()
    ticker_data.dropna(inplace=True)

    
    print(f'\nDescriptive statistics for {ticker}:')
    print(ticker_data['LogReturn'].describe())

    
    plt.figure(figsize=(10, 6))
    plt.plot(ticker_data.index, ticker_data['LogReturn'])
    plt.title(f'Log Returns for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Log Return')

    
    if not os.path.exists('Volanalysisresults'):
        os.makedirs('Volanalysisresults')
    plt.savefig(os.path.join('Volanalysisresults', f'{ticker}_log_returns.png'))
    plt.close()

    
    ticker_data['LogReturn'] *= 100

    model = arch_model(ticker_data['LogReturn'], vol='EGARCH', p=p, q=q)
    model_fitted = model.fit(disp="off")

    
    print(f'\nGARCH({p},{q}) Model Summary for {ticker}:')
    print(model_fitted.summary())

    return model_fitted


#TODO: Figure out data correct options for each higher order model
def fit_higher_order_models(df, tickers):
    returns = pd.DataFrame()

    for ticker in tickers:
        ticker_data = df[df['Ticker'] == ticker].copy()
        ticker_data['LogReturn'] = np.log(ticker_data['Close']).diff()
        ticker_data.dropna(inplace=True)

        if len(ticker_data) < 1000:
            print(f'Not enough data for higher order models for {ticker}. Skipping...')
            continue

        
        ticker_data['LogReturn'] *= 100

        returns[ticker] = ticker_data['LogReturn']

        # fit UGARCH 
        asset = ticker_data['LogReturn']
        garch = UGARCH(order=(1, 1))
        garch.spec(returns=asset)
        garch.fit()
        
        
        params = garch.garch_params
        UGconvol = garch.cond_vol

        print(f'\nGJR-GARCH Model Summary for {ticker}:')
        print(f'Parameters: {params}')
        print(f'{UGconvol}') # Plot this

    if len(returns.columns) > 1:
        # Fit DCC GARCH model
        garch_specs = [UGARCH(order=(1, 1)) for _ in range(len(returns.columns))]
        dcc_garch = DCCGARCH()
        dcc_garch.spec(ugarch_objs=garch_specs, returns=returns)
        dcc_garch.fit()
        #dcc_garch.qllf()
        dcc_garch.forecast(n_ahead=6)
        print(f'\nDCC-GARCH Model Summary:')
        print(f'Parameters: {dcc_garch.estimate_params()}')
        print(f'Log-Likelihood: {dcc_garch.qllf}')
        print(f'{dcc_garch.forecast(n_ahead=6)}')
def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv('scraped_yahoo_finance_data.csv')
    df = prepare_data(df)
    tickers = df['Ticker'].unique()
    pq_combinations = [(1, 1)]  # example combinations (2, 1), (1, 2), (2, 2)

    for ticker in tickers:
        for p, q in pq_combinations:
            print(f'Analyzing volatility for {ticker} with GARCH({p},{q})...')
            model = fit_garch_model(df, ticker, p, q)
            print(f'Volatility analysis for {ticker} with GARCH({p},{q}) completed.')

    print(f'Analyzing higher order models for tickers...')
    fit_higher_order_models(df, tickers)
    print(f'Higher order models analysis completed.')

if __name__ == "__main__":
    main()
