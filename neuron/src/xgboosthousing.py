import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load and Filter Data
raw_df = pd.read_csv('../realtor-data.zip.csv')
df = raw_df[(raw_df['city'] == 'Houston') & (raw_df['state'] == 'Texas')].copy()
df.dropna(subset=['price', 'bed', 'bath', 'house_size', 'zip_code'], inplace=True)

# 2. Engineered Features (Same as PyTorch)
df['bath_bed_ratio'] = df['bath'] / df['bed']
df['house_size_per_room'] = df['house_size'] / (df['bed'] + df['bath'])
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# 3. One-Hot Encode Zip Codes
df['zip_code'] = df['zip_code'].astype(int).astype(str)
df = pd.get_dummies(df, columns=['zip_code'], drop_first=True)

# 4. Define Target and Features
y_raw = df['price'].values  # Keeping target in true dollars for direct MAE calculation
drop_cols = ['price', 'status', 'city', 'state', 'prev_sold_date']
X_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
X_raw = X_df.astype(float).values

# 5. Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training XGBoost on {X_train.shape[1]} features...")

# 6. Train XGBoost
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)

# 7. Evaluate
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print(f"True XGBoost Houston Baseline (MAE): ${xgb_mae:,.2f}")