# coding=utf-8

import os
import ccxt

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta


#print(ccxt.exchanges)
#indodax   = ccxt.indodax({
    #'apiKey': os.environ['ACCESS_API'],
    #'secret': os.environ['ACCESS_KEY'],
#})

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
    return tickers

# Step 2: Prepare Dataset
class CryptoDataset(Dataset):
    def __init__(self, data, window_size=60):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size, :-1]
        y = self.data[idx + self.window_size, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Step 3: Build LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size)  # Initial hidden state
        c_0 = torch.zeros(1, x.size(0), hidden_size)  # Initial cell state
        output, _ = self.lstm(x, (h_0, c_0))
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
    
    # Create dataset and dataloader
    dataset = CryptoDataset(data, window_size=60)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    input_size = 2
    hidden_size = 50
    output_size = 1
    num_layers = 1
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model (for simplicity, using the mock dataset)
    epochs = 10
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
    print("Cryptocurrencies predicted to increase in price within the next hour:")
    for symbol, change in increasing_cryptos:
        print(f"{symbol}: {change}%")
