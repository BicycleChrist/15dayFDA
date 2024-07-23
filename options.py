import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.interpolate import griddata

def get_options_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.history(period="1d")['Close'].iloc[-1]
    
    all_calls = []
    all_puts = []
    for date in stock.options:
        options = stock.option_chain(date)
        expiration = datetime.strptime(date, "%Y-%m-%d")
        days_to_expiry = (expiration - datetime.now()).days
        
        for option_type in ['calls', 'puts']:
            df = getattr(options, option_type)
            df['DaysToExpiry'] = days_to_expiry
            df['Moneyness'] = np.log(df['strike'] / current_price)
            df['ExpirationDate'] = date
            if option_type == 'calls':
                all_calls.append(df)
            else:
                all_puts.append(df)
    
    calls = pd.concat(all_calls)
    puts = pd.concat(all_puts)
    return calls, puts, current_price

def save_to_csv(data, filename):
    data.to_csv(filename, index=False)

def plot_vol_surface(data, title, current_price, z_scale=1.0, min_days_to_expiry=7, strike_range=0.30):
    # Filter data based on minimum days to expiry and strike range
    min_strike = current_price * (1 - strike_range)
    max_strike = current_price * (1 + strike_range)
    filtered_data = data[(data['DaysToExpiry'] >= min_days_to_expiry) & 
                         (data['strike'] >= min_strike) & 
                         (data['strike'] <= max_strike)]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = filtered_data['strike']
    y = filtered_data['DaysToExpiry']
    z = filtered_data['impliedVolatility'] * z_scale
    
    # gridski
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    
    # interpolate Z values so we can have cool surface
    Z = griddata((x, y), z, (X, Y), method='cubic')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='rainbow', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Days to Expiration')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f"{title}\n(Min {min_days_to_expiry} days to expiry, Strike range: {strike_range*100}%)")
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, label='Implied Volatility', pad=0.1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ticker = "CRWD"  
    calls, puts, current_price = get_options_data(ticker)
    
    save_to_csv(calls, f"{ticker}_calls_{datetime.now().strftime('%Y-%m-%d')}.csv")
    save_to_csv(puts, f"{ticker}_puts_{datetime.now().strftime('%Y-%m-%d')}.csv")
    print(f"CSV files saved for {ticker} calls and puts.")
    
    # Plot with default settings specified in vol surface function
    plot_vol_surface(calls, f'{ticker} Call Options Volatility Surface', current_price)
    plot_vol_surface(puts, f'{ticker} Put Options Volatility Surface', current_price)
    
    # min 30 day to expiry, 20% range around current price
    plot_vol_surface(calls, f'{ticker} Call Options Volatility Surface (30+ days, 20% range)', 
                     current_price, min_days_to_expiry=30, strike_range=0.20)
    plot_vol_surface(puts, f'{ticker} Put Options Volatility Surface (30+ days, 20% range)', 
                     current_price, min_days_to_expiry=30, strike_range=0.20)
    
    # Plot with adjusted z_scale
    
    #plot_vol_surface(calls, f'{ticker} Call Options Volatility Surface (0.5x scale)', 
    #                 current_price, z_scale=0.5)
    #plot_vol_surface(puts, f'{ticker} Put Options Volatility Surface (0.5x scale)', 
    #                 current_price, z_scale=0.5)
    
    print(f"Current price of {ticker}: ${current_price:.2f}")
