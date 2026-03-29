from utils import load_data, normalize, create_windows, create_spectrograms
from utils import plot_time_series, plot_spectrogram, plot_training_spectrogram
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Create outputs folder
os.makedirs("../outputs", exist_ok=True)

# Step 1: Load data
data = load_data()

# Step 2: Plot time series
plot_time_series(data)

# Step 3: Normalize
scaled, scaler = normalize(data)

# Step 4: Create windows
X, y = create_windows(scaled)

# Step 5: Plot raw spectrogram
plot_spectrogram(X[0])

# Step 6: Convert to spectrogram
X_spec = create_spectrograms(X)

# Step 7: Plot training spectrogram
plot_training_spectrogram(X_spec)

# Step 8: Split
X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.2)

# Step 9: Build model
model = build_model(X_spec.shape[1:])

# Step 10: Train
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Step 11: Predict
preds = model.predict(X_test)

# Step 12: Evaluate
mse = mean_squared_error(y_test, preds)
print("MSE:", mse)

# Step 13: Plot prediction
plt.plot(y_test, label="Actual")
plt.plot(preds, label="Predicted")
plt.legend()
plt.title("Prediction vs Actual")
plt.savefig("../outputs/prediction.png")
plt.show()

# Step 14: Save model
model.save("../outputs/model.keras")