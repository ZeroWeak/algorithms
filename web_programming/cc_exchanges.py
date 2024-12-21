import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Step 1: Fetch Data from Kraken API
def fetch_cryptocurrency_data():
    url = "https://api.kraken.com/0/public/AssetPairs"
    response = requests.get(url)
    pairs = response.json().get('result', {})
    active_pairs = [pair for pair in pairs if 'USD' in pair]
    
    # Fetch ticker data for all active USD pairs
    tickers = {}
    for pair in active_pairs:
        ticker_url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
        response = requests.get(ticker_url)
        if response.status_code == 200:
            tickers[pair] = response.json().get('result', {}).get(pair, {})
    
    # Get top 100 pairs based on transaction volume in the last 60 minutes
    volumes = []
    for pair, data in tickers.items():
        if 'v' in data:
            volume = float(data['v'][1])  # Transaction volume in the last 24 hours
            volumes.append((pair, volume))
    volumes.sort(key=lambda x: x[1], reverse=True)
    top_100_pairs = volumes[:100]
    
    # Fetch ticker data for top 100 pairs in the last 60 minutes
    top_100_tickers = {}
    for pair, _ in top_100_pairs:
        ticker_url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
        response = requests.get(ticker_url)
        if response.status_code == 200:
            top_10_tickers[pair] = response.json().get('result', {}).get(pair, {})
    
    return top_10_tickers

# Step 2: Prepare Dataset
class CryptoDataset(Dataset):
    def __init__(self, data, window_size=60):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        # Ensure the length is non-negative
        return max(0, len(self.data) - self.window_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size, :-1]
        y = self.data[idx + self.window_size, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Step 3: Build LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)  # Initial hidden state
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)  # Initial cell state
        output, _ = self.lstm(x, (h_0, c_0))
        output = self.dropout(output)
        output = self.fc(output[:, -1, :])
        return output

# Main Execution
if __name__ == "__main__":
    # Fetch data
    tickers = fetch_cryptocurrency_data()
    data = []
    symbols = []
    for symbol, ticker in tickers.items():
        if 'c' in ticker:
            price = float(ticker['c'][0])  # Last trade price
            volume = float(ticker['v'][0])  # 24h volume
            data.append([price, volume, price])  # Assuming price as target for simplicity
            symbols.append(symbol)
    
    # Convert to numpy array
    data = np.array(data)
    
    # Handle missing or NaN values
    data = np.nan_to_num(data)
    
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Create dataset and dataloader
    dataset = CryptoDataset(data, window_size=60)
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    else:
        print("Data set is too small for the given window size.")

    # Initialize model, loss, and optimizer
    input_size = 2
    hidden_size = 50
    output_size = 1
    num_layers = 1
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model (for simplicity, using the mock dataset)
    epochs = 50  # Increase number of epochs
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Predict price change for each cryptocurrency
    predictions = []
    with torch.no_grad():
        for i, symbol in enumerate(symbols):
            sample = data[i:i + 60, :-1]  # Last 60 data points
            if len(sample) < 60:
                continue  # Skip incomplete windows
            sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            predicted_price = model(sample).item()
            current_price = data[i + 59, 0]
            percent_change = ((predicted_price - current_price) / current_price) * 100
            predictions.append((symbol, percent_change))

    # Filter and display cryptocurrencies predicted to increase in price
    increasing_cryptos = [(symbol, round(change, 2)) for symbol, change in predictions if change > 0]
    increasing_cryptos.sort(key=lambda x: x[1], reverse=True)
    print("Cryptocurrencies predicted to increase in price within the next hour:")
    for symbol, change in increasing_cryptos:
        print(f"{symbol}: {change}%")
