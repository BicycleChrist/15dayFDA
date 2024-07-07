import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.univariate import ConstantMean, MIDASHyperbolic

def prepare_hf_data(df):
    # Ensure the DataFrame has a 'timestamp' column
    if 'timestamp' not in df.columns:
        raise ValueError("The DataFrame must have a 'timestamp' column.")

    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Ensure the DataFrame has a 'close' column
    if 'close' not in df.columns:
        raise ValueError("The DataFrame must have a 'close' column.")

    return df['close']

def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1))

def fit_midas_model(returns):
    model = ConstantMean(returns, volatility=MIDASHyperbolic(m=5, asym=True))
    results = model.fit(disp='off', rescale=False)
    return model, results

def analyze_midas_model(data_path):
    # Load and prepare data
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')

    # Resample to regular intervals and forward fill
    df = df.resample('1T').last().ffill()

    # Calculate returns
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()

    # Fit MIDAS model with different initial values
    model = ConstantMean(returns, volatility=MIDASHyperbolic(m=22))  # Using 22 for about a month of trading days
    try:
        midas_results = model.fit(disp='off', options={'maxiter': 1000})
    except:
        print("MIDAS model failed to converge. Trying with different initial values.")
        model.volatility.theta = np.array([0.1, 0.1])  # Example of setting different initial values
        midas_results = model.fit(disp='off', options={'maxiter': 1000})

    # Extract conditional volatility
    conditional_volatility = midas_results.conditional_volatility

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(conditional_volatility, label='MIDAS Hyperbolic Volatility')
    plt.title('MIDAS Hyperbolic Volatility Model')
    plt.xlabel('Date')
    plt.ylabel('Conditional Volatility')
    plt.legend()
    plt.show()

    # Print summary
    print("MIDAS Hyperbolic Model Summary:")
    print(midas_results.summary())

    return returns, conditional_volatility, midas_results

if __name__ == "__main__":
    # This allows the file to be run independently for testing
    log_returns, volatility, midas_results = analyze_midas_model('hf_data.csv')
