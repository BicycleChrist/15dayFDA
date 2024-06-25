import mpl_toolkits.mplot3d
import numpy
from mvgarch.ugarch import UGARCH
from mvgarch.mgarch import DCCGARCH
import os
import pandas as pd
import numpy as np
from arch import arch_model
from arch.univariate import *
import matplotlib.dates as mdates

import matplotlib.pyplot as plt

def _to_unmasked_float_array(x):
    """
    Convert a sequence to a float array; if input was a masked array, masked
    values are converted to nans.
    """
    if hasattr(x, 'mask'):
        return np.ma.asarray(x, float).filled(np.nan)
    else:
        return np.asarray(x, float)


# balance between responsiveness and stability, standard GARCH models (1,2) or (2,1) seeem to be slightly better off
# concerned about overestimating risk, the GJR-GARCH model might be appropriate. 
# one would think that EGARCH is a preferred model for forecasting during periods of heightned vol (I.E. leading up to or following a clincial event)  
# as it can differentiate between the impact of positive and negative shocks on volatility (leverage effect)
# TODO: Give the FIGARCH a shot, implement some degree of forecasting 


def plot_dcc_garch_3d_surface(dcc_garch_model, log_returns):
    cond_vols = dcc_garch_model.cond_vols

    fig = plt.figure(figsize=(20, 20))  # Increased figure size
    ax = fig.add_subplot(111, projection='3d')

    num_assets = cond_vols.shape[1]
    time_points = log_returns.index
    assets = np.arange(num_assets)

    X, Y = np.meshgrid(assets, mdates.date2num(time_points))
    Z = cond_vols

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Asset Index', labelpad=14)
    ax.set_ylabel('Date', labelpad=20)
    ax.set_zlabel('Volatility', labelpad=20)
    ax.set_title('DCC-GARCH Conditional Volatilities', pad=10)

    # Format the date ticks
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Volatility', rotation=270, labelpad=20)

    # Set tick labels for all assets
    ax.set_xticks(np.arange(num_assets))
    ax.set_xticklabels(log_returns.columns, rotation=90, ha='right')

    # Adjust subplot parameters
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)

    # Adjust the viewing angle for better visibility
    ax.view_init(elev=20, azim=30)

    if not os.path.exists('Volanalysisresults'):
        os.makedirs('Volanalysisresults', exist_ok=True)
    plt.savefig(os.path.join('Volanalysisresults', 'dcc_garch_3d_surface.png'), bbox_inches='tight', dpi=300)
    plt.close()



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

def GarchEverything(df: pd.DataFrame):
    models = {
        'GARCH(1,1)':     [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='GARCH'  , p=1, q=1)      for ticker in df.columns],
        'EGARCH(1,1)':    [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='EGARCH' , p=1, q=1)      for ticker in df.columns],
        'GJR-GARCH(1,1)': [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='GARCH'  , p=1, q=1, o=1) for ticker in df.columns],
        'GARCH(2,1)':     [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='GARCH'  , p=2, q=1)      for ticker in df.columns],
        'GARCH(1,2)':     [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='FIGARCH', p=1, q=1)      for ticker in df.columns],
    }

    results = {
        model_name: {
            "result"   : model.fit(disp=True),
            "forecasts": model.fit(disp=False).forecast(method="bootstrap", simulations=5, horizon=5),
        }
        for model_name, model_list in models.items()
        for model in model_list
    }
    
    dcc_garch, log_likelihood = testmaybe(df)
    
    dccgarch_dict = {
        "model": dcc_garch,
        "dynCorr": GetDynamicCorrelation(dcc_garch),
    }
    plot_dcc_garch_3d_surface(dcc_garch, calculate_log_returns(df))
    
    #results["DCCGARCH"] = dccgarch_dict
    fig, ax = plt.subplots(figsize=(10, 6))
    #plt.ion()
    for column, row in df.items():
        ax.plot(row, label=column, alpha=0.25)
    ax.legend()
    ax.grid(True)
    #mapped_tickers = dict(zip(df.columns, dcc_garch.cond_vols))
    #ax.plot(mapped_tickers.values(), label=mapped_tickers.keys(), alpha=0.25)
    #for ticker, cond_vol in mapped_tickers.items():
    #    ax.plot(cond_vol, label=ticker, alpha=0.25)
    plt.show(block=True)
    return



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
        model_fitted = model.fit(disp=True)
        aic = model_fitted.aic
        bic = model_fitted.bic
        
        results[model_name] = {'model': model_fitted, 'AIC': aic, 'BIC': bic}

        # compute realized volatility using a 30-day rolling window
        #realized_volatility = np.sqrt(df.rolling(window=30).apply(lambda x: (x**2).sum()))
        #realized_vol_ewma = ticker_data.ewm(span=30).std()
        #realized_volatility = realized_vol_ewma
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
            os.makedirs('Volanalysisresults', exist_ok=True)
        plt.savefig(os.path.join('Volanalysisresults', f'{ticker}_{model_name}_volatility.png'))
        plt.close()

    best_model_name = min(results, key=lambda k: results[k]['AIC'])
    best_model = results[best_model_name]['model']
    forecast = best_model.forecast(horizon=1)
    print(forecast)

    return best_model, results


def fit_higher_order_models(df):
    returns = df
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

def testmaybe(log_returns):
    returns = log_returns * 10
    garch_specs = [UGARCH(order=(1, 1)) for _ in range(len(returns.columns))]
    
    dcc_garch = DCCGARCH()
    dcc_garch.spec(ugarch_objs=garch_specs, returns=returns)
    dcc_garch.fit()
    #dcc_garch.volatility = GARCH(1, 0, 1)
    #dcc_garch.distribution = StudentsT()
    # DCC GARCH fit
    return dcc_garch, None

def GetDynamicCorrelation(dcc_garch_model):
    dynamic_correlation_results = DCCGARCH.dynamic_corr(dcc_garch_model.returns, dcc_garch_model.cond_vols, dcc_garch_model.dcc_a, dcc_garch_model.dcc_b)
    return dynamic_correlation_results


def plot_conditional_volatilities(dcc_garch_model, log_returns):
    cond_vols = dcc_garch_model.cond_vols
    
    plt.figure(figsize=(12, 96))
    for i, ticker in enumerate(log_returns.columns):
        plt.plot(log_returns.index, cond_vols[:, i], label=ticker, alpha=0.25)
    
    plt.title('Conditional Volatilities from DCC-GARCH')
    plt.xlabel('Date')
    plt.ylabel('Conditional Volatility')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists('Volanalysisresults'):
        os.makedirs('Volanalysisresults', exist_ok=True)
    plt.savefig(os.path.join('dcc_garch_output', 'dcc_garch_conditional_volatilities.png'))
    plt.close()


import arch.data.core_cpi
# https://arch.readthedocs.io/en/latest/univariate/introduction.html#arch.univariate.arch_model
def test_arch_stuff(df, ticker):
    ticker_data = df[ticker].dropna() # don't multiply by 100 here
    # rescale prevents convergence issues
    am = ConstantMean(ticker_data, rescale=True)
    am.volatility = GARCH(1, 0, 1)
    am.distribution = Normal()
    res = am.fit()
    res.summary()
    
    core_cpi = arch.data.core_cpi.load()
    ann_inflation = 100 * core_cpi.CPILFESL.pct_change().dropna()
    #ar = ARX(100 * ann_inflation, rescale=True, lags=list(range(1,12)))
    ar = ARX(100 * ann_inflation, rescale=True, lags=[1,3,12])
    ar.distribution = StudentsT()
    # supposedly, StudentsT distribution is superior to normal-distribution; degree-of-freedom is approx. 8
    res = ar.fit(update_freq=0, disp="off")
    forecasts = res.forecast(horizon=5, method='simulation', simulations=50)

    y = np.random.randn(100)
    x = np.random.randn(100,2)
    ls = LS(y, x)
    ls.volatility = GARCH(1, 0, 1)
    ls.distribution = Normal()
    res = ls.fit()
    res.summary()



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
        plot_dcc_garch_3d_surface(dcc_garch, log_returns)
        plot_conditional_volatilities(dcc_garch, log_returns)


if __name__ == "__main__":
    main()


