import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Import necessary functions from dffnn_input.py
from dffnn_input import prepare_data, calculate_log_returns_alt, prepare_garch_features

class DFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DFFNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()

        # Input layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.ln_layers.append(nn.LayerNorm(hidden_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.ln_layers.append(nn.LayerNorm(hidden_sizes[i]))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # Activation function
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        for layer, ln in zip(self.hidden_layers, self.ln_layers):
            x = self.activation(ln(layer(x)))
        return self.output_layer(x)

def prepare_features(log_returns, all_features, lookback=10):
    features = []
    targets = []

    for i in range(lookback, len(all_features)):
        feature = all_features.iloc[i-lookback:i].values.flatten()
        target = all_features['HV'].iloc[i]

        features.append(feature)
        targets.append(target)

    return np.array(features), np.array(targets)

def prepare_forecast_features(all_features, lookback=10):
    feature = all_features.iloc[-lookback:].values.flatten()
    return feature.reshape(1, -1)

def train_dffnn(X_train, y_train, model, epochs=30, batch_size=32, learning_rate=0.002):
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_test)).numpy()
    
    if np.isnan(predictions).any():
        print("Warning: NaN values in predictions")
        predictions = np.nan_to_num(predictions)
    
    mse = np.mean((predictions - y_test.reshape(-1, 1))**2)
    print(f"Mean Squared Error: {mse:.4f}")
    
    return predictions

def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Volatility')
    plt.plot(y_pred, label='Predicted Volatility')
    plt.title('Actual vs Predicted Volatility')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

def forecast_single_ticker(model, df, forecast_horizon=10):
    #df = prepare_data(df)
    log_returns, historical_volatility = calculate_log_returns_alt(df)
    garch_features, _, _ = prepare_garch_features(log_returns.dropna())
    
    all_features = pd.concat([historical_volatility, garch_features], axis=1).dropna()
    all_features.columns = ['HV'] + list(garch_features.columns)
    
    forecasts = []
    
    for _ in range(forecast_horizon):
        features = prepare_forecast_features(all_features)
        
        model.eval()
        with torch.no_grad():
            prediction = model(torch.FloatTensor(features)).item()
        
        forecasts.append(prediction)
        
        # update all_features for the next iteration
        new_date = all_features.index[-1] + pd.Timedelta(days=1)
        new_row = pd.DataFrame({col: [prediction] for col in all_features.columns}, index=[new_date])
        all_features = pd.concat([all_features, new_row])
    
    forecast_dates = pd.date_range(start=all_features.index[-forecast_horizon], periods=forecast_horizon)
    forecast_df = pd.DataFrame({'Forecasted_Volatility': forecasts}, index=forecast_dates)
    print(f"forecast_dates: {forecast_dates}")
    
    return forecast_df

def forecast_volatility(model, df, forecast_horizon=10):
    try:
        if 'Ticker' in df.columns:
            forecasts = {}
            for ticker, group in df.groupby('Ticker'):
                print(f"Processing ticker: {ticker}")
                ticker_df = prepare_data(group)
                ticker_forecast = forecast_single_ticker(model, ticker_df, forecast_horizon)
                forecasts[ticker] = ticker_forecast
            return forecasts
        else:
            df = prepare_data(df)
            return forecast_single_ticker(model, df, forecast_horizon)
    except Exception as e:
        print(f"An error occurred during forecasting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

import pathlib

def DoTheThing(df):
    log_returns, historical_volatility = calculate_log_returns_alt(df['Adj Close'])
    
    garch_features, best_model, best_model_name = prepare_garch_features(log_returns.dropna())
    print(f"best_model_name: {best_model_name}")
    
    all_features = pd.concat([historical_volatility, garch_features], axis=1).dropna()
    all_features.columns = ['HV'] + list(garch_features.columns)
    
    features, targets = prepare_features(log_returns, all_features)
    return features, targets


def LoadFiles():
    cwd = pathlib.Path.cwd()
    inputfolder = cwd / "inputs"
    inputfiles = inputfolder.glob("input*.csv")
    # input data must come from 'scraped_yahoo_finance_dataR.csv'; with the headers 
    # (Ticker, Date, Open, High, Low, Close, Adj Close, Volume)
    
    loaded_stuff = []
    
    for inputfile in inputfiles:
        df = pd.read_csv(inputfile)
        #df = prepare_data(df)  # don't do this
        log_returns, historical_volatility = calculate_log_returns_alt(df['Adj Close'])
        
        garch_features, best_model, best_model_name = prepare_garch_features(log_returns.dropna())
        print(f"best_model_name: {best_model_name}")
        
        all_features = pd.concat([historical_volatility, garch_features], axis=1).dropna()
        all_features.columns = ['HV'] + list(garch_features.columns)
        
        features, targets = prepare_features(log_returns, all_features)
        
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        loaded_stuff.append((df, features, targets))
    
    return loaded_stuff


def backtest_model(model, features, targets, window_size=30, step_size=1):
    predictions = []
    actual_values = []
    
    for i in range(0, len(features) - window_size, step_size):
        train_features = features[i:i+window_size]
        train_targets = targets[i:i+window_size]
        
        # Retrain the model on this window
        model = train_dffnn(train_features, train_targets, model, epochs=5)
        
        # Make a prediction for the next day
        with torch.no_grad():
            next_day_prediction = model(torch.FloatTensor(features[i+window_size].reshape(1, -1))).item()
        
        predictions.append(next_day_prediction)
        actual_values.append(targets[i+window_size])
    
    return np.array(predictions), np.array(actual_values)

def plot_backtest_results(predictions, actual_values, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual Volatility', alpha=0.7)
    plt.plot(predictions, label='Predicted Volatility', alpha=0.7)
    plt.title(f'Backtesting Results: Predicted vs Actual Volatility for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    loadedStuff = LoadFiles()
    
    for idx, (df, features, targets) in enumerate(loadedStuff):
        print(f"Processing dataset {idx + 1}")
        
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=42)
        
        input_size = features.shape[1]
        hidden_sizes = [64, 32, 16]
        output_size = 1
        
        model = DFFNN(input_size, hidden_sizes, output_size)
        trained_model = train_dffnn(X_train, y_train, model)
        
        predictions = evaluate_model(trained_model, X_test, y_test)
        
        plot_results(y_test, predictions.flatten())
        
        # Perform backtesting
        backtest_predictions, backtest_actual = backtest_model(trained_model, features, targets)
        
        # Plot backtesting results
        ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else f"Dataset {idx + 1}"
        plot_backtest_results(backtest_predictions, backtest_actual, ticker)
        
        # Forecast future volatility
        forecasts = forecast_volatility(trained_model, df, forecast_horizon=10)
        if forecasts is not None:
            if isinstance(forecasts, dict):
                for ticker, forecast_df in forecasts.items():
                    print(f"10-day Volatility Forecast for {ticker}:")
                    print(forecast_df)
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(forecast_df.index, forecast_df['Forecasted_Volatility'], marker='o')
                    plt.title(f'10-day Volatility Forecast for {ticker}')
                    plt.xlabel('Date')
                    plt.ylabel('Forecasted Volatility')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
            else:
                print("10-day Volatility Forecast:")
                print(forecasts)
                
                plt.figure(figsize=(12, 6))
                plt.plot(forecasts.index, forecasts['Forecasted_Volatility'], marker='o')
                plt.title('10-day Volatility Forecast')
                plt.xlabel('Date')
                plt.ylabel('Forecasted Volatility')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
        else:
            print("Forecasting failed. Please check your data and model.")

if __name__ == "__main__":
    main()

