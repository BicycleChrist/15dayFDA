import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Import necessary functions from Volanalysis.py
from dffnn_input import calculate_log_returns, fit_univariate_garch_models, prepare_data

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

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

def prepare_features(log_returns, historical_volatility, garch_volatility, lookback=10):
    features = []
    targets = []

    # index check
    common_index = historical_volatility.index.intersection(garch_volatility.index)
    historical_volatility = historical_volatility.loc[common_index]
    garch_volatility = garch_volatility.loc[common_index]
    log_returns = log_returns.loc[common_index]

    for i in range(lookback, len(historical_volatility)):
        feature = np.concatenate([
            historical_volatility.iloc[i-lookback:i].values,
            garch_volatility.iloc[i-lookback:i].values
        ])
        target = historical_volatility.iloc[i]

        features.append(feature)
        targets.append(target)

    return np.array(features), np.array(targets)

def prepare_forecast_features(historical_volatility, garch_volatility, lookback=10):
    # ensure we're working with the most recent data
    historical_volatility = historical_volatility.iloc[-lookback:]
    garch_volatility = garch_volatility.iloc[-lookback:]
    
    feature = np.concatenate([
        historical_volatility.values,
        garch_volatility.values
    ])
    return feature.reshape(1, -1)  # Reshape to match the model's input shape

def train_dffnn(features, targets, model, epochs=400, batch_size=32, learning_rate=0.001):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_test)).numpy()
    
    if np.isnan(predictions).any():
        print("Warning: NaN values in predictions")
        predictions = np.nan_to_num(predictions)  # Replace NaNs with 0
    
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
    log_returns, historical_volatility = calculate_log_returns(df)
    
    best_model, _ = fit_univariate_garch_models(log_returns.dropna(), 'Ticker')
    garch_volatility = pd.Series(best_model.conditional_volatility, index=log_returns.dropna().index)
    
    aligned_data = pd.concat([historical_volatility, garch_volatility], axis=1).dropna()
    aligned_data.columns = ['historical', 'garch']
    
    forecasts = []
    
    for _ in range(forecast_horizon):
        features = prepare_forecast_features(aligned_data['historical'], aligned_data['garch'])
        
        model.eval()
        with torch.no_grad():
            prediction = model(torch.FloatTensor(features)).item()
        
        forecasts.append(prediction)
        
        # Update aligned_data for the next iteration
        new_date = aligned_data.index[-1] + pd.Timedelta(days=1)
        new_row = pd.DataFrame({'historical': [prediction], 'garch': [prediction]}, index=[new_date])
        aligned_data = pd.concat([aligned_data, new_row])
    
    forecast_dates = pd.date_range(start=aligned_data.index[-forecast_horizon], periods=forecast_horizon)
    forecast_df = pd.DataFrame({'Forecasted_Volatility': forecasts}, index=forecast_dates)
    
    return forecast_df

def forecast_volatility(model, df, forecast_horizon=10):
    try:
        print("DataFrame columns before processing:", df.columns)
        print("DataFrame index name before processing:", df.index.name)
        
        # Check if there are multiple tickers
        if 'Ticker' in df.columns:
            # Group by ticker and forecast for each
            forecasts = {}
            for ticker, group in df.groupby('Ticker'):
                print(f"Processing ticker: {ticker}")
                ticker_df = prepare_data(group)
                print("Ticker DataFrame columns after prepare_data:", ticker_df.columns)
                print("Ticker DataFrame index name after prepare_data:", ticker_df.index.name)
                ticker_forecast = forecast_single_ticker(model, ticker_df, forecast_horizon)
                forecasts[ticker] = ticker_forecast
            return forecasts
        else:
            # Single ticker case
            df = prepare_data(df)
            print("DataFrame columns after prepare_data:", df.columns)
            print("DataFrame index name after prepare_data:", df.index.name)
            return forecast_single_ticker(model, df, forecast_horizon)
    except Exception as e:
        print(f"An error occurred during forecasting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    df = pd.read_csv('scraped_yahoo_finance_data_minimal.csv')
    df = prepare_data(df)
    log_returns, historical_volatility = calculate_log_returns(df)
    
    best_model, _ = fit_univariate_garch_models(log_returns.dropna(), 'Ticker')
    garch_volatility = pd.Series(best_model.conditional_volatility, index=log_returns.dropna().index)
    
    # Align the indices of historical_volatility and garch_volatility
    aligned_data = pd.concat([historical_volatility, garch_volatility], axis=1).dropna()
    
    # Prepare features for DFFNN
    features, targets = prepare_features(log_returns, aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
    
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Create and train DFFNN model
    input_size = features.shape[1]
    hidden_sizes = [64, 32, 16, 8]
    output_size = 1
    
    model = DFFNN(input_size, hidden_sizes, output_size)
    trained_model, X_test, y_test = train_dffnn(features, targets, model)
    
    # model eval
    predictions = evaluate_model(trained_model, X_test, y_test)
    
    # Plot results
    plot_results(y_test, predictions.flatten())

    #  forecast
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
