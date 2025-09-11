import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.interpolate import griddata
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go

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

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """Calculate implied volatility using Brent's method"""
    if T <= 0:
        return 0

    # Use intrinsic value for bounds checking
    if option_type == 'call':
        intrinsic = max(S - K, 0)
        if market_price <= intrinsic:
            return 0
    else:
        intrinsic = max(K - S, 0)
        if market_price <= intrinsic:
            return 0

    def objective(sigma):
        if option_type == 'call':
            return black_scholes_call(S, K, T, r, sigma) - market_price
        else:
            return black_scholes_put(S, K, T, r, sigma) - market_price

    try:
        # Use reasonable bounds for volatility (0.1% to 500%)
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        return np.nan

def calculate_implied_volatilities(data, current_price, risk_free_rate=0.05):
    """Calculate implied volatilities for options data"""
    data = data.copy()
    data['CalculatedIV'] = np.nan

    for idx, row in data.iterrows():
        if pd.isna(row['lastPrice']) or row['lastPrice'] <= 0:
            continue

        if row['DaysToExpiry'] <= 0:
            continue

        T = row['DaysToExpiry'] / 365.0  # Convert to years
        market_price = row['lastPrice']

        # Determine option type from contract name or assume based on data
        option_type = 'call' if 'C' in str(row.get('contractSymbol', '')) else 'put'

        iv = implied_volatility(
            market_price, current_price, row['strike'],
            T, risk_free_rate, option_type
        )

        data.at[idx, 'CalculatedIV'] = iv

    return data



def plot_vol_surface(data, title, current_price, z_scale=1.0, max_days_to_expiry=180, interactive=True, use_calculated_iv=True, show_immediately=True, show_scatter=True, show_contours=True, show_contour_projections=True, show_atm_marker=True):
    # Use calculated IV if available, otherwise fall back to yfinance
    iv_column = 'CalculatedIV' if use_calculated_iv and 'CalculatedIV' in data.columns else 'impliedVolatility'

    # Filter by expiry and IV validity first
    filtered_data = data[(data['DaysToExpiry'] <= max_days_to_expiry) &
                         (data['DaysToExpiry'] >= 1) &
                         (data[iv_column].notna()) &
                         (data[iv_column] > 0) &
                         (data[iv_column] < 5)]  # Allow higher IVs for volatile stocks
    
    # Smart strike filtering based on number of unique strikes
    unique_strikes = sorted(filtered_data['strike'].unique())
    num_strikes = len(unique_strikes)
    
    print(f"Initial filtering {title}: {len(data)} -> {len(filtered_data)} points")
    print(f"Unique strikes found: {num_strikes}")
    
    # Apply strike range filtering for stocks with many strikes (threshold: 15 strikes)
    if num_strikes > 15:
        # Filter to strikes within 50% of ATM (25% below to 50% above)
        lower_bound = current_price * 0.75
        upper_bound = current_price * 1.5
        
        strike_filtered_data = filtered_data[
            (filtered_data['strike'] >= lower_bound) & 
            (filtered_data['strike'] <= upper_bound)
        ]
        
        strikes_in_range = sorted(strike_filtered_data['strike'].unique())
        print(f"Strike filtering applied: {num_strikes} -> {len(strikes_in_range)} strikes")
        print(f"Strike range: ${lower_bound:.2f} - ${upper_bound:.2f} (ATM: ${current_price:.2f})")
        print(f"Strikes used: {strikes_in_range}")
        
        filtered_data = strike_filtered_data
    else:
        print(f"Using all {num_strikes} strikes (below threshold of 15)")
        print(f"Available strikes: {unique_strikes}")
    
    print(f"Final data points: {len(filtered_data)}")
    print(f"Days range: {filtered_data['DaysToExpiry'].min()} - {filtered_data['DaysToExpiry'].max()}")
    print(f"IV range: {filtered_data[iv_column].min():.1%} - {filtered_data[iv_column].max():.1%}")

    if len(filtered_data) < 5:
        print(f"Insufficient data points for {title} (only {len(filtered_data)} valid points)")
        return

    x = filtered_data['strike']
    y = filtered_data['DaysToExpiry']
    z = filtered_data[iv_column] * z_scale

    # Get unique strikes and expiries for proper grid
    unique_strikes = sorted(x.unique())
    unique_days = sorted(y.unique())

    print(f"Unique strikes: {len(unique_strikes)}, Unique expiries: {len(unique_days)}")

    # Create smooth surface using interpolation
    if len(unique_strikes) > 3 and len(unique_days) > 3:
        # Create denser grid for smooth surface
        strike_range = np.linspace(min(unique_strikes), max(unique_strikes), 25)
        days_range = np.linspace(min(unique_days), max(unique_days), 25)
        X, Y = np.meshgrid(strike_range, days_range)

        # Interpolate Z values using actual data
        Z = griddata((x, y), z, (X, Y), method='cubic')

        # Fill NaN values with nearest neighbor if needed
        if np.isnan(Z).sum() > Z.size * 0.3:
            Z_nearest = griddata((x, y), z, (X, Y), method='nearest')
            Z = np.where(np.isnan(Z), Z_nearest, Z)

        print(f"Smooth grid shape: {X.shape}, Z has {np.isnan(Z).sum()} NaN values out of {Z.size}")
    else:
        X, Y, Z = None, None, None

    if interactive:
        # Plotly interactive version
        fig = go.Figure()

        # Add surface if we have a proper grid
        if X is not None and Y is not None and Z is not None:
            # Smooth connected surface
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Turbo',  # Beautiful rainbow colorscale
                opacity=0.9,
                name='Vol Surface',
                hovertemplate='Strike: $%{x:.1f}<br>Days: %{y:.0f}<br>Vol: %{z:.1%}<extra></extra>',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Implied Volatility", font=dict(size=14)),
                    tickformat='.0%',
                    len=0.8,
                    thickness=20
                ),
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    fresnel=0.2,
                    specular=1.2,
                    roughness=0.05
                ),
                contours=dict(
                    z=dict(
                        show=show_contours,
                        usecolormap=True,
                        highlightcolor="rgba(226,232,240,0.9)",  # Light contour lines for dark theme
                        project_z=show_contour_projections,
                        width=1.5
                    )
                )
            ))
            print(f"Added enhanced surface with {np.sum(~np.isnan(Z))} data points")
        else:
            print("Not enough data for surface, showing scatter only")

        # Show actual data points if toggled on
        if show_scatter:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=6,
                    color=z,
                    colorscale='Viridis',
                    showscale=False,
                    line=dict(width=1, color='white')
                ),
                name='Actual Data',
                hovertemplate='Strike: $%{x:.0f}<br>Days: %{y:.0f}<br>Vol: %{z:.1%}<extra></extra>'
            ))

        # Add subtle ATM marker at the intersection if toggled on
        if show_atm_marker:
            fig.add_trace(go.Scatter3d(
                x=[current_price],
                y=[y.mean()],
                z=[z.mean()],
                mode='markers',
                marker=dict(
                    size=12,
                    color='gold',
                    symbol='diamond',
                    line=dict(width=2, color='rgba(255,255,255,0.8)')
                ),
                name='ATM Reference',
                hovertemplate=f'At-The-Money<br>Strike: ${current_price:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>{len(filtered_data)} data points, ≤{max_days_to_expiry} days to expiry</sub>",
                font=dict(size=18),
                x=0.5
            ),
            scene=dict(
                xaxis_title='Strike Price ($)',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility',
                xaxis=dict(
                    tickmode='array',
                    tickvals=unique_strikes,
                    showgrid=True,
                    gridcolor='rgba(148,163,184,0.4)',  # Muted blue-gray grid
                    gridwidth=1
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(148,163,184,0.4)',
                    gridwidth=1
                ),
                zaxis=dict(
                    tickformat='.0%',
                    showgrid=True,
                    gridcolor='rgba(148,163,184,0.4)',
                    gridwidth=1
                ),
                bgcolor='rgba(15,23,42,0.95)',  # Deep slate background - like storm clouds
                camera=dict(
                    eye=dict(x=1.5, y=1.8, z=1.3),  # Better viewing angle
                    center=dict(x=0, y=0, z=0)
                )
            ),
            width=1200,
            height=800,
            paper_bgcolor='rgba(30,41,59,1)',  # Dark charcoal - volatile market vibes
            plot_bgcolor='rgba(30,41,59,1)',
            font=dict(color='rgba(226,232,240,1)', size=12)  # Light text
        )
        if show_immediately:
            print(f"Showing plot with {len(fig.data)} traces")

            # Force plotly to use browser and add delay
            import plotly.io as pio
            pio.renderers.default = "browser"

            try:
                fig.show()
                print("Plot displayed in browser")
            except Exception as e:
                print(f"Browser display failed: {e}")

            # Save HTML backup
            filename = f"{title.replace(' ', '_').replace('-', '_')}.html"
            fig.write_html(filename)
            print(f"Saved plot as {filename}")

            # Small delay to ensure rendering completes
            import time
            time.sleep(0.5)
        else:
            # Return figure for later display
            return fig

    else:
        # Matplotlib version
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Days to Expiration')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f"{title}\n(Min {min_days_to_expiry} days, Strike range: {strike_range*100}%)")
        fig.colorbar(surf, ax=ax, label='Implied Volatility', pad=0.1)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ticker = "MU"
    print(f"Fetching options data for {ticker}...")
    calls, puts, current_price = get_options_data(ticker)

    print(f"Current price of {ticker}: ${current_price:.2f}")
    print(f"Raw data - Calls: {len(calls)}, Puts: {len(puts)}")

    # Calculate proper implied volatilities
    print("Calculating implied volatilities...")
    calls = calculate_implied_volatilities(calls, current_price, risk_free_rate=0.05)
    puts = calculate_implied_volatilities(puts, current_price, risk_free_rate=0.05)

    # Show some stats
    valid_call_ivs = calls['CalculatedIV'].dropna()
    valid_put_ivs = puts['CalculatedIV'].dropna()
    print(f"Calculated IVs - Calls: {len(valid_call_ivs)}, Puts: {len(valid_put_ivs)}")

    if len(valid_call_ivs) > 0:
        print(f"Call IV range: {valid_call_ivs.min():.1%} - {valid_call_ivs.max():.1%}")
    if len(valid_put_ivs) > 0:
        print(f"Put IV range: {valid_put_ivs.min():.1%} - {valid_put_ivs.max():.1%}")

    # Save enhanced data
    save_to_csv(calls, f"{ticker}_calls_{datetime.now().strftime('%Y-%m-%d')}.csv")
    save_to_csv(puts, f"{ticker}_puts_{datetime.now().strftime('%Y-%m-%d')}.csv")
    print(f"CSV files saved for {ticker} calls and puts with calculated IVs.")

    # Interactive plots with calculated IVs - with proper timing
    print("Creating volatility surfaces...")

    # Create all figures first, then display them
    figures = []

    print("Processing call options...")
    fig1 = plot_vol_surface(calls, f'{ticker} Call Options - Calculated IV', current_price, max_days_to_expiry=180, show_immediately=False, show_scatter=True, show_contours=True, show_contour_projections=True, show_atm_marker=True)
    if fig1:
        figures.append((fig1, f'{ticker}_Call_Options_Calculated_IV'))

    print("Processing put options...")
    fig2 = plot_vol_surface(puts, f'{ticker} Put Options - Calculated IV', current_price, max_days_to_expiry=180, show_immediately=False, show_scatter=True, show_contours=True, show_contour_projections=True, show_atm_marker=True)
    if fig2:
        figures.append((fig2, f'{ticker}_Put_Options_Calculated_IV'))

    print("Processing short-term call options...")
    fig3 = plot_vol_surface(calls, f'{ticker} Call Options - Short Term', current_price, max_days_to_expiry=60, show_immediately=False, show_scatter=True, show_contours=False, show_contour_projections=False, show_atm_marker=True)
    if fig3:
        figures.append((fig3, f'{ticker}_Call_Options_Short_Term'))

    print("Processing short-term put options...")
    fig4 = plot_vol_surface(puts, f'{ticker} Put Options - Short Term', current_price, max_days_to_expiry=60, show_immediately=False, show_scatter=False, show_contours=True, show_contour_projections=False, show_atm_marker=True)
    if fig4:
        figures.append((fig4, f'{ticker}_Put_Options_Short_Term'))

    # Now display all figures with proper timing
    print(f"Displaying {len(figures)} volatility surfaces...")
    for i, (fig, filename) in enumerate(figures):
        fig.write_html(f"{filename}.html")
        import time
        time.sleep(1)  # Give each plot time to load
