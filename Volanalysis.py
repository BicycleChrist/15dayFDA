import mpl_toolkits.mplot3d
from mvgarch.ugarch import UGARCH
from mvgarch.mgarch import DCCGARCH
import os
import pandas as pd
import numpy as np
from arch.univariate import *
from arch.univariate import FIGARCH
import matplotlib.dates as mdates
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import os

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
# TODO: implement some degree of forecasting, properly backtest it 
# TODO: Rescale data back to original values after parameter estimation 

# interactive plotly plot with visible dates
def plot_dcc_garch_3d_surface_plotly(dcc_garch_model, log_returns):
    cond_vols = dcc_garch_model.cond_vols
    num_assets = cond_vols.shape[1]
    time_points = log_returns.index
    assets = np.arange(num_assets)
    X, Y = np.meshgrid(assets, mdates.date2num(time_points))
    Z = cond_vols
    
    date_strings = [d.strftime('%Y-%m-%d') for d in time_points]
    
    # plotly object, not matplotlib
    fig = go.FigureWidget(data=[go.Surface(z=Z, x=X, y=time_points, colorscale='Rainbow')])
    # colorscale presets are:
    # Blackbody,Bluered,Blues,C ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd
    
    fig.update_layout(
        title='DCC-GARCH Conditional Volatilities',
        # autosize=True,
        #width=1920,
        #height=1080,
        scene=dict(
            xaxis_title='Asset Index',
            yaxis_title='',
            zaxis_title='Volatility',
            xaxis=dict(
                ticktext=log_returns.columns,
                tickvals=assets,
            ),
            #yaxis=dict(
            #    ticktext=date_strings[::len(date_strings)//10],  # Show fewer dates to avoid overcrowding
            #    tickvals=list(range(0, len(time_points), len(time_points)//10)),
            #),
        )
    )
    
    fig.show()
    fig.write_html(os.path.join('Volanalysisresults', 'dcc_garch_3d_surface_interactive.html'))
    return fig

# Original matplotlib plot
def plot_dcc_garch_3d_surface_matplotlib(dcc_garch_model, log_returns, isInteractive=False):
    cond_vols = dcc_garch_model.cond_vols
    num_assets = cond_vols.shape[1]
    time_points = log_returns.index
    assets = np.arange(num_assets)
    X, Y = np.meshgrid(assets, mdates.date2num(time_points))
    Z = cond_vols
    
    if isInteractive: plt.ioff() # this is somehow inverted???????
    fig = plt.figure("3D_DCC_GARCH", figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
    ax.plot_wireframe(X, Y, Z, color='black', alpha=0.1)
    
    ax.set_xlabel('Asset Index', labelpad=14)
    ax.set_ylabel('Date', labelpad=20)
    ax.set_zlabel('Volatility', labelpad=20)
    ax.set_title('DCC-GARCH Conditional Volatilities', pad=10)
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    fig.autofmt_xdate()
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Volatility', rotation=270, labelpad=20)
    
    ax.set_xticks(np.arange(num_assets))
    ax.set_xticklabels(log_returns.columns, rotation=45, ha='right')
    
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
    ax.view_init(elev=20, azim=30)
    
    if isInteractive: plt.show()
    plt.savefig(os.path.join('Volanalysisresults', 'dcc_garch_3d_surface.png'), bbox_inches='tight', dpi=300)
    return fig


# size of the nodes during the animation represent conditonal volatility of the individual stocks at the given point in the time series
def animate_correlation_network(dcc_garch_model, log_returns, threshold=0.25, save_path='Volanalysisresults/animated_cor_network.mp4', frame_interval=1):
    dynamic_correlation_result = GetDynamicCorrelation(dcc_garch_model)
    dynamic_correlation = dynamic_correlation_result[0]

    num_timepoints = dynamic_correlation.shape[2]
    used_indices = range(0, num_timepoints, frame_interval)

    G = nx.Graph()
    for asset in log_returns.columns:
        G.add_node(asset)

    fig, ax = plt.subplots(figsize=(12, 8))

    def update(num):
        ax.clear()
        correlation_matrix = dynamic_correlation[:, :, used_indices[num]]
        G.clear_edges()

        for i, asset1 in enumerate(log_returns.columns):
            for j, asset2 in enumerate(log_returns.columns):
                if i < j:
                    correlation = correlation_matrix[i, j]
                    if abs(correlation) > threshold:
                        G.add_edge(asset1, asset2, weight=abs(correlation))

        pos = nx.kamada_kawai_layout(G)
        node_sizes = [dcc_garch_model.cond_vols[used_indices[num], i] * 1000 for i in range(len(log_returns.columns))]
        edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='red')
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
        nx.draw_networkx_labels(G, pos)

        ax.set_title(f'Asset Correlation Network at time {used_indices[num]}')
        ax.axis('off')

    num_frames = len(used_indices)
    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)

    if not os.path.exists('Volanalysisresults'):
        os.makedirs('Volanalysisresults', exist_ok=True)

    ani.save(save_path, writer='ffmpeg', fps=30)
    plt.close()

    print(f"Created animation with {num_frames} frames")




def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.pivot(columns='Ticker', values='Close')
    df.dropna(inplace=True)
    return df

def calculate_log_returns(df):
    df = df.astype("float")
    log_returns = np.log(df / df.shift(1)).dropna()*100
    historical_volatility = log_returns.rolling(window=30).std() * np.sqrt(252)
    return log_returns, historical_volatility

def GarchEverything(df: pd.DataFrame):
    models = {
        'GARCH(1,1)':     [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='GARCH'  , p=1, q=1)               for ticker in df.columns],
        'EGARCH(1,1)':    [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='EGARCH' , p=1, q=1)               for ticker in df.columns],
        'GJR-GARCH(1,1)': [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='GARCH'  , p=1, q=1, o=1)          for ticker in df.columns],
        'T-GARCH(1,1)':   [arch_model(df[ticker].dropna(), mean='LS', dist="studentst", rescale=False, vol='GARCH',   p=1, q=1, o=1, power=1) for ticker in df.columns],
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


"""
 FIGARCH model has to be constructed directly as guy who wrote the package instructed 
 https://github.com/bashtage/arch/issues/735
"""
#TODO: Implement mean ["Zero", "LS", "HAR", "Constant"] & dist ["normal", "t", "skewt", "ged"] for each model being fit

def fit_univariate_garch_models(df, ticker):
    ticker_data = df[ticker].dropna() 
    #figarch_model = ConstantMean(ticker_data, volatility=FIGARCH(p=1,q=1,power=2,truncation=1000))
    #figarch_res = figarch_model.fit()
    
    
    #aparch_res = aparch_model.fit()
    #print(aparch_res.summary())
    #print(figarch_res.summary())
    models = {'GARCH(1,1)': arch_model(ticker_data, vol='GARCH', p=1, q=1, dist="studentst", mean="Constant"),
              'GARCH(2,2)': arch_model(ticker_data, vol='GARCH', p=2, q=2, dist="studentst", mean="Constant"),
              'EGARCH(1,1)': arch_model(ticker_data, vol='EGARCH', p=1, q=1, dist="studentst", mean="Constant"),
              'GJR-GARCH(1,1,1)': arch_model(ticker_data, vol='GARCH', p=1, q=1, o=1, mean="Constant"),
              'T-GARCH(1,1)': arch_model(ticker_data, vol='GARCH', p=1, q=1, o=1, power=1, mean="Constant", dist="studentst"),
              'FIGARCH(1,1,2,d)': ConstantMean(ticker_data, volatility=FIGARCH(p=1,q=1,power=2,truncation=2000)),
              'APARCH:(1,1,1,d)':ConstantMean(ticker_data, volatility=APARCH(p=1,q=1,o=1,delta=0.7)),
              #'MIDAShyperbolic(m,asym)':ConstantMean(ticker_data, volatility=MIDASHyperbolic(m=5, asym=True))
              }

    results = {}

    for model_name, model in models.items():
        model_fitted = model.fit(disp=True)
        aic = model_fitted.aic
        bic = model_fitted.bic

        results[model_name] = {'model': model_fitted, 'AIC': aic, 'BIC': bic}

        # compute Historical volatility using a 30-day rolling window
        #realized_volatility = np.sqrt(df.rolling(window=30).apply(lambda x: (x**2).sum()))
        #realized_vol_ewma = ticker_data.ewm(span=30).std()
        #realized_volatility = realized_vol_ewma
        historical_volatility = ticker_data.rolling(window=30).std()

        # generate/save conditional vs Historical volatility plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(historical_volatility, label='Historical Volatility', alpha=0.7, color='blue')
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


def plot_conditional_volatilities(dcc_garch_model, log_returns, isInteractive=False):
    if isInteractive: plt.ioff() # this is somehow inverted???????
    plt.figure("ConditionalVolGraph", figsize=(96, 12))
    plt.title('Conditional Volatilities from DCC-GARCH')
    plt.xlabel('Date')
    plt.ylabel('Conditional Volatility')
    plt.grid(True)
    
    cond_vols = dcc_garch_model.cond_vols
    for i, ticker in enumerate(log_returns.columns):
        plt.plot(log_returns.index, cond_vols[:, i], label=ticker)
    
    if isInteractive: plt.legend().set_draggable(True)
    #plt.interactive(isInteractive)
    
    if isInteractive: plt.show(block=True)
    # for some reason saving the plot only works when interactive-mode is off
    plt.savefig(os.path.join('dcc_garch_output', 'dcc_garch_conditional_volatilities.png'))
    plt.close()
    return

#import arch.data.core_cpi
# https://arch.readthedocs.io/en/latest/univariate/introduction.html#arch.univariate.arch_model



def main():
    os.makedirs('Volanalysisresults', exist_ok=True)
    df = pd.read_csv('scraped_yahoo_finance_data.csv')
    df = prepare_data(df)
    log_returns, historical_volatility = calculate_log_returns(df)
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
        plot_dcc_garch_3d_surface_matplotlib(dcc_garch, log_returns, isInteractive=True)
        plot_dcc_garch_3d_surface_plotly(dcc_garch, log_returns)
        plot_conditional_volatilities(dcc_garch, log_returns, isInteractive=True)
        #animate_correlation_network(dcc_garch, log_returns, threshold=0.25)


if __name__ == "__main__":
    main()


