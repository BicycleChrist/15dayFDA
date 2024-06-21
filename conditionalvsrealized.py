import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import os

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

    # Rescale the data
    ticker_data['LogReturn'] *= 100

    model = arch_model(ticker_data['LogReturn'], vol='Garch', p=1, q=1)
    model_fitted = model.fit(disp="off")

    # Print model summary
    print(f'\nGARCH Model Summary for {ticker}:')
    print(model_fitted.summary())

    return model_fitted, ticker_data

def analyze_volatility(model, ticker_data, ticker, output_folder):
    realized_vol = ticker_data['LogReturn'].rolling(window=30).std()
    conditional_vol = model.conditional_volatility

    plt.figure(figsize=(10, 6))
    plt.plot(realized_vol, label='Realized Volatility', alpha=0.7)
    plt.plot(conditional_vol, label='Conditional Volatility', alpha=0.7)
    plt.title(f'Volatility for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(output_folder, f'{ticker}_volatility.png'))
    plt.close()

def main():
    df = pd.read_csv('scraped_yahoo_finance_data.csv')
    df = prepare_data(df)
    tickers = df['Ticker'].unique()

    output_folder = 'Volanalysisresults'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for ticker in tickers:
        print(f'Analyzing volatility for {ticker}...')
        model, ticker_data = fit_garch_model(df, ticker)
        analyze_volatility(model, ticker_data, ticker, output_folder)
        print(f'Volatility analysis for {ticker} completed.')

if __name__ == "__main__":
    main()
