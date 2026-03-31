import torch
import torch.nn as nn
import joblib
import pandas as pd
from xgboost import XGBRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. REBUILD ARCHITECTURE ---
class HoustonMarketNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): 
        return self.network(x)

# --- 2. LOAD PRODUCTION ARTIFACTS ---
scaler = joblib.load('houston_scaler.bin')
feature_columns = joblib.load('houston_features.bin')
num_features = len(feature_columns)

# Load PyTorch Weights
nn_model = HoustonMarketNet(input_size=num_features).to(device)
nn_model.load_state_dict(torch.load('houston_nn_weights.pth', map_location=device, weights_only=True))
nn_model.eval() # Set to evaluation mode (turns off Dropout)

# Load XGBoost
xgb_model = XGBRegressor()
xgb_model.load_model('houston_xgb_model.json')

# --- 3. THE PREDICTION ENGINE ---
# --- 3. THE PREDICTION ENGINE ---
def price_houston_house(beds, baths, sqft, zip_code):
    # Calculate Engineered Features
    ratio = baths / beds if beds > 0 else 0
    per_room = sqft / (beds + baths) if (beds + baths) > 0 else 0
    
    # 1. Initialize ALL training columns to 0.0 to prevent KeyErrors
    input_dict = {col: [0.0] for col in feature_columns}
    
    # 2. Overwrite the specific core features
    if 'bed' in input_dict: input_dict['bed'] = [beds]
    if 'bath' in input_dict: input_dict['bath'] = [baths]
    if 'house_size' in input_dict: input_dict['house_size'] = [sqft]
    if 'bath_bed_ratio' in input_dict: input_dict['bath_bed_ratio'] = [ratio]
    if 'house_size_per_room' in input_dict: input_dict['house_size_per_room'] = [per_room]
    
    # 3. Flip the requested one-hot encoded zip code to 1.0
    target_zip_col = f'zip_code_{zip_code}'
    if target_zip_col in input_dict:
        input_dict[target_zip_col] = [1.0]
    else:
        print(f"[Warning] Zip code {zip_code} not in training data. Defaulting to base market logic.")
        
    # 4. Convert to DataFrame ensuring the exact column order used during training
    df_input = pd.DataFrame(input_dict)[feature_columns]
    
    # 5. Scale Data
    input_scaled = scaler.transform(df_input.values)
    
    # 6. Get Predictions
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        nn_val = nn_model(input_tensor).cpu().item()
        
    xgb_val = xgb_model.predict(input_scaled)[0]
    
    # Ensemble Average and convert back from $100k scale
    final_price = ((nn_val + xgb_val) / 2) * 100000
    return final_price

# --- 4. TEST DEPLOYMENT ---
print("Pricing a 4-Bed, 3-Bath, 2500 SqFt house in 77494 (Katy)...")
estimated_value = price_houston_house(beds=4, baths=3, sqft=2500, zip_code=77494)

print(f"Final Ensemble Prediction: ${estimated_value:,.2f}")