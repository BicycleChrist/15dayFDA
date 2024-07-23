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

def plot_vol_surface(data, x_column, y_column, z_column, title, option_type):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = data[x_column]
    y = data[y_column]
    z = data[z_column]
    
    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate Z values on the grid
    Z = griddata((x, y), z, (X, Y), method='cubic')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'{title} - {option_type.capitalize()}')
    
    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, label='Implied Volatility', pad=0.1)
    
    plt.tight_layout()
    plt.show()

# Example usage
ticker = "AAPL"  # You can change this to any ticker you're interested in
calls, puts, current_price = get_options_data(ticker)

# Save CSV files
save_to_csv(calls, f"{ticker}_calls_{datetime.now().strftime('%Y-%m-%d')}.csv")
save_to_csv(puts, f"{ticker}_puts_{datetime.now().strftime('%Y-%m-%d')}.csv")
print(f"CSV files saved for {ticker} calls and puts.")

# Plot using Days to Expiry
plot_vol_surface(calls, 'strike', 'DaysToExpiry', 'impliedVolatility', f'{ticker} Volatility Surface (Days to Expiry)', 'calls')
plot_vol_surface(puts, 'strike', 'DaysToExpiry', 'impliedVolatility', f'{ticker} Volatility Surface (Days to Expiry)', 'puts')

# Plot using Moneyness
plot_vol_surface(calls, 'strike', 'Moneyness', 'impliedVolatility', f'{ticker} Volatility Surface (Moneyness)', 'calls')
plot_vol_surface(puts, 'strike', 'Moneyness', 'impliedVolatility', f'{ticker} Volatility Surface (Moneyness)', 'puts')

print(f"Current price of {ticker}: ${current_price:.2f}")
