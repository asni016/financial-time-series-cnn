import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import stft
import matplotlib.pyplot as plt

# Load stock data
def load_data():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    data = []

    for t in tickers:
        df = yf.download(t, start="2018-01-01", end="2023-01-01")
        data.append(df['Close'])

    df = pd.concat(data, axis=1)
    df.columns = tickers
    df.dropna(inplace=True)

    return df

# Normalize data
def normalize(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

# Create sliding windows
def create_windows(data, window=64):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][0])
    return np.array(X), np.array(y)

# Generate spectrograms
def create_spectrograms(X):
    specs = []
    for sample in X:
        channels = []
        for i in range(sample.shape[1]):
            f, t, Z = stft(sample[:, i], nperseg=16)
            S = np.abs(Z)**2
            channels.append(S)
        specs.append(np.stack(channels, axis=-1))
    return np.array(specs)

# Plot time series
def plot_time_series(data):
    data.plot(figsize=(10,5), title="Stock Price Time Series")
    plt.savefig("../outputs/time_series.png")
    plt.show()

# Plot spectrogram from raw signal
def plot_spectrogram(sample):
    f, t, Z = stft(sample[:, 0], nperseg=16)
    S = np.abs(Z)**2

    plt.figure(figsize=(8,5))
    plt.pcolormesh(t, f, S, shading='gouraud')
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.savefig("../outputs/spectrogram.png")
    plt.show()

# Plot training spectrogram
def plot_training_spectrogram(X_spec):
    plt.imshow(X_spec[0][:,:,0], aspect='auto', origin='lower')
    plt.title("Training Spectrogram")
    plt.colorbar()
    plt.savefig("../outputs/training_spectrogram.png")
    plt.show()