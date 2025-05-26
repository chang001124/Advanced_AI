# nn_model_youbike_daily.py
"""
Train & validate a neural‑network model to predict daily YouBike transfer rentals.
Data: 2023_youbike_daily_features.csv
Split: first 80 % of chronological data → train, last 20 % → test.
Model: Simple feed‑forward (Dense) network using Keras.
Note: You can switch to LSTM by reshaping X and modifying model_fn.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

# ------------------------------------------------------------------
# 1. Load feature data
features_path = Path("2023_youbike_daily_features.csv")
assert features_path.exists(), "Feature file not found. Run feature_engineering_youbike_daily.py first."

df = pd.read_csv(features_path, parse_dates=["rent_date"])

# Target and feature matrix
Y = df["daily_rent_count"].values.astype("float32")
X = df.drop(["rent_date", "daily_rent_count"], axis=1).values.astype("float32")

# ------------------------------------------------------------------
# 2. Chronological split 80/20
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# ------------------------------------------------------------------
# 3. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Build NN model (Dense → Dense → output)
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print(model.summary())

# ------------------------------------------------------------------
# 5. Train
callbacks = [EarlyStopping(patience=20, restore_best_weights=True)]
history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    epochs=300,
    batch_size=16,
    callbacks=callbacks,
    verbose=0
)

# ------------------------------------------------------------------
# 6. Evaluate
pred = model.predict(X_test).flatten()
mae = mean_absolute_error(Y_test, pred)
rmse = np.sqrt(mean_squared_error(Y_test, pred))
print(f"Test MAE  = {mae:.2f}")
print(f"Test RMSE = {rmse:.2f}")

# Optional: save model & scaler
model.save("youbike_nn_model.h5")
import joblib; joblib.dump(scaler, "youbike_scaler.save")
print("✅ Model + scaler saved. Use them for future inference.")
