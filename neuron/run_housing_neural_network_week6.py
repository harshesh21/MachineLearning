import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Processing with Location ---
raw_df = pd.read_csv('realtor-data.zip.csv')
df = raw_df[(raw_df['city'] == 'Houston') & (raw_df['state'] == 'Texas')].copy()

# Add zip_code to the dropna subset
df.dropna(subset=['price', 'bed', 'bath', 'house_size', 'zip_code'], inplace=True)

# Engineered Features
df['bath_bed_ratio'] = df['bath'] / df['bed']
df['house_size_per_room'] = df['house_size'] / (df['bed'] + df['bath'])

# --- THE FIX ---
# Convert any Infinity values (caused by dividing by zero) to 0
df.replace([np.inf, -np.inf], 0, inplace=True)
# Fill any resulting nulls with 0
df.fillna(0, inplace=True)
# ---------------

# One-Hot Encode Zip Codes
df['zip_code'] = df['zip_code'].astype(int).astype(str)
df = pd.get_dummies(df, columns=['zip_code'], drop_first=True)

# Define Target
y_raw = df['price'].values / 100000.0

# Define Features (Drop text/date columns and the target)
drop_cols = ['price', 'status', 'city', 'state', 'prev_sold_date']
X_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Ensure all data is float32 for PyTorch
X_raw = X_df.astype(float).values
input_features = X_raw.shape[1] 

print(f"Total Features (Including Zip Codes): {input_features}")

# --- 2. Scaling and Tensors ---
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1).to(device)

# --- 3. Dynamic Architecture ---
class HoustonMarketNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(),
            nn.Dropout(0.1), # Randomly disables 20% of neurons
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Pass the feature count into the model initialization
model = HoustonMarketNet(input_size=input_features).to(device)

criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25)
# --- 4. Training Loop ---
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
            val_pred = model(X_test)
            val_loss = criterion(val_pred, y_test)
        print(f"Epoch {epoch+1:4} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")


import xgboost as xgb
from sklearn.metrics import mean_absolute_error




xgb_model = xgb.XGBRegressor(
    n_estimators=1000,          # Maximum possible trees
    learning_rate=0.05,         # Speed of learning (lower = more precise)
    max_depth=6,                # Complexity of the neighborhood logic
    early_stopping_rounds=20,   # Stop if 20 trees are built without improvement
    random_state=42,
    n_jobs=-1                   # Max out CPU threads
)

# 3. The Training Execution
# We pass the scaled test data so the model can monitor it in real-time.
# verbose=False keeps your terminal clean.
xgb_model.fit(
    X_train_scaled, y_train_raw,
    eval_set=[(X_test_scaled, y_test_raw)],
    verbose=False
)

# Predict and convert back to dollars
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_mae = mean_absolute_error(y_test_raw, xgb_preds) * 100000

print(f"True XGBoost Houston Baseline (MAE): ${xgb_mae:,.2f}")
