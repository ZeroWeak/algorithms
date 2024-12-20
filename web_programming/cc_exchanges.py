import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Step 1: Fetch OHLC Historical Data
def fetch_historical_data(pair, interval=60):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json().get('result', {})
        pair_key = list(result.keys())[0]  # Pair key
        data = result[pair_key]
        return pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    return None

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
    # List of cryptocurrency pairs (example subset)
    pairs = ["BTCUSD", "ETHUSD", "ADAUSD", "XRPUSD", "SOLUSD"]

    predictions = []
    for pair in pairs:
        print(f"Fetching data for {pair}...")
        ohlc_data = fetch_historical_data(pair)
        if ohlc_data is None or ohlc_data.empty:
            print(f"No data for {pair}, skipping.")
            continue

        # Preprocess data
        ohlc_data['close'] = ohlc_data['close'].astype(float)
        ohlc_data['volume'] = ohlc_data['volume'].astype(float)
        data = ohlc_data[['close', 'volume']].values

        # Normalize data
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data = (data - data_mean) / data_std

        # Create dataset and dataloader
        dataset = CryptoDataset(data, window_size=60)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize model
        input_size = 2
        hidden_size = 50
        output_size = 1
        num_layers = 1
        model = LSTMModel(input_size, hidden_size, output_size, num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 5  # Reduced for simplicity
        for epoch in range(epochs):
            for x, y in dataloader:
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.squeeze(), y)
                loss.backward()
                optimizer.step()

        # Predict next hour price change
        with torch.no_grad():
            sample = data[-60:, :-1]  # Last 60 data points
            sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            predicted_price = model(sample).item()
            current_price = ohlc_data.iloc[-1]['close']
            percent_change = ((predicted_price - current_price) / current_price) * 100
            predictions.append((pair, round(percent_change, 2)))

    # Filter and sort results
    increasing_cryptos = [(symbol, change) for symbol, change in predictions if change > 0]
    increasing_cryptos.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print("\nCryptocurrencies predicted to increase in price within the next hour:")
    for symbol, change in increasing_cryptos:
        print(f"{symbol}: {change}%")
