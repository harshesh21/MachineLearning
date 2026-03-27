import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# --- 1. Setup & Hardware ---
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
raw_df = pd.read_csv('./realtor-data.zip.csv')
df = raw_df[(raw_df['city'] == 'Houston') & (raw_df['state'] == 'Texas')].copy()

# Drop rows with missing crucial info
df.dropna(subset=['price', 'bed', 'bath', 'house_size', 'zip_code'], inplace=True)

# Feature Engineering
df['bath_bed_ratio'] = df['bath'] / df['bed']
df['house_size_per_room'] = df['house_size'] / (df['bed'] + df['bath'])
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Categorical Encoding (Zip Codes)
df['zip_code'] = df['zip_code'].astype(int).astype(str)
df = pd.get_dummies(df, columns=['zip_code'], drop_first=True)

# Define Target (Scaled to $100k units for NN stability)
y_raw = df['price'].values / 100000.0

# Define Features
drop_cols = ['price', 'status', 'city', 'state', 'prev_sold_date']
X_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
X_raw = X_df.astype(float).values
input_features = X_raw.shape[1] 

print(f"Total Features: {input_features}")

# Split and Scale
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Move Tensors to GPU
X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1).to(device)

# --- 2. Neural Network Definition ---
class HoustonMarketNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = HoustonMarketNet(input_size=input_features).to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25)

# --- 3. Training Loop ---
print("Starting Neural Network Training...")
epochs = 2000
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
    if (epoch + 1) % 500 == 0:
        model.eval()
        with torch.no_grad():
            v_pred = model(X_test)
            v_loss = criterion(v_pred, y_test)
        mae_val = mean_absolute_error(y_test_raw, v_pred.cpu().numpy().flatten()) * 100000
        print(f"Epoch {epoch+1:4} | Train Loss: {loss.item():.4f} | Val MAE: ${mae_val:,.2f}")

# --- 4. Ensemble Logic ---
print("\n--- Initializing Ensemble Engine ---")

# Train XGBoost
print("Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6)
xgb_model.fit(X_train_scaled, y_train_raw)
xgb_preds = xgb_model.predict(X_test_scaled)

# Get Final PyTorch Predictions (Vectors, not scalars)
model.eval()
with torch.no_grad():
    pytorch_preds_vector = model(X_test).cpu().numpy().flatten()

# Voting Ensemble Calculation (Averaging the two prediction vectors)
ensemble_preds = (xgb_preds + pytorch_preds_vector) / 2

# Final Scoreboard (Scaling back to actual dollars)
xgb_mae = mean_absolute_error(y_test_raw, xgb_preds) * 100000
pytorch_mae = mean_absolute_error(y_test_raw, pytorch_preds_vector) * 100000
ensemble_mae = mean_absolute_error(y_test_raw, ensemble_preds) * 100000

print("\n--- FINAL HOUSTON MARKET EVALUATION ---")
print(f"XGBoost MAE:        ${xgb_mae:,.2f}")
print(f"Neural Network MAE: ${pytorch_mae:,.2f}")
print(f"Ensemble MAE:       ${ensemble_mae:,.2f}")

improvement = min(xgb_mae, pytorch_mae) - ensemble_mae
print(f"Ensemble Improvement: ${improvement:,.2f} over best standalone model")

import joblib

# 1. Save the Scaler
joblib.dump(scaler, 'houston_scaler.bin')

# 2. Save the exact column order from the training DataFrame
joblib.dump(X_df.columns.tolist(), 'houston_features.bin')

# 3. Save the Models
xgb_model.save_model('houston_xgb_model.json')
torch.save(model.state_dict(), 'houston_nn_weights.pth')

print("\nAll models, scalers, and feature maps exported to disk.")