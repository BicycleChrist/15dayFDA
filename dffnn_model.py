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

    # Ensure all inputs have the same index
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



def train_dffnn(features, targets, model, epochs=500, batch_size=32, learning_rate=0.001):
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
    
    # Evaluate model
    predictions = evaluate_model(trained_model, X_test, y_test)
    
    # Plot results
    plot_results(y_test, predictions.flatten())

if __name__ == "__main__":
    main()
