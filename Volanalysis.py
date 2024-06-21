import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def fit_garch_model(df, ticker):
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data['LogReturn'] = np.log(ticker_data['Close']).diff()
    ticker_data.dropna(inplace=True)

    # Print descriptive statistics
    print(f'\nDescriptive statistics for {ticker}:')
    print(ticker_data['LogReturn'].describe())

    # Plot log returns
    plt.figure(figsize=(10, 6))
    plt.plot(ticker_data.index, ticker_data['LogReturn'])
    plt.title(f'Log Returns for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.show()

    # Rescale the data
    ticker_data['LogReturn'] *= 100

    model = arch_model(ticker_data['LogReturn'], vol='Garch', p=1, q=1)
    model_fitted = model.fit(disp="off")

    # Print model summary
    print(f'\nGARCH Model Summary for {ticker}:')
    print(model_fitted.summary())

    return model_fitted

def analyze_volatility(model, ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model.conditional_volatility, label='Conditional Volatility')
    ax.set_title(f'Conditional Volatility for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.legend()
    plt.show()

def main():
    df = pd.read_csv('scraped_yahoo_finance_data.csv')
    df = prepare_data(df)
    tickers = df['Ticker'].unique()
    for ticker in tickers:
        print(f'Analyzing volatility for {ticker}...')
        model = fit_garch_model(df, ticker)
        analyze_volatility(model, ticker)
        print(f'Volatility analysis for {ticker} completed.')

if __name__ == "__main__":
    main()
