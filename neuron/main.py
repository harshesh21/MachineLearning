import torch
import torch.nn as nn
import joblib
import pandas as pd
from xgboost import XGBRegressor
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. API & HARDWARE SETUP ---
app = FastAPI(title="Houston Real Estate AI Engine")
# Add this block to allow your frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. INPUT SCHEMA ---
# This defines exactly what data the API expects to receive
class HouseSpecs(BaseModel):
    beds: int
    baths: int
    sqft: int
    zip_code: int

# --- 3. REBUILD ARCHITECTURE ---
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

# --- 4. LOAD PRODUCTION ARTIFACTS ---
# Loading these globally so they only process once at startup
scaler = joblib.load('houston_scaler.bin')
feature_columns = joblib.load('houston_features.bin')
num_features = len(feature_columns)

nn_model = HoustonMarketNet(input_size=num_features).to(device)
nn_model.load_state_dict(torch.load('houston_nn_weights.pth', weights_only=True))
nn_model.eval() 

xgb_model = XGBRegressor()
xgb_model.load_model('houston_xgb_model.json')

# --- 5. THE PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_price(specs: HouseSpecs):
    # Calculate Engineered Features
    ratio = specs.baths / specs.beds if specs.beds > 0 else 0
    per_room = specs.sqft / (specs.beds + specs.baths) if (specs.beds + specs.baths) > 0 else 0
    
    # Initialize all columns to 0.0
    input_dict = {col: [0.0] for col in feature_columns}
    
    # Overwrite the specific core features
    if 'bed' in input_dict: input_dict['bed'] = [specs.beds]
    if 'bath' in input_dict: input_dict['bath'] = [specs.baths]
    if 'house_size' in input_dict: input_dict['house_size'] = [specs.sqft]
    if 'bath_bed_ratio' in input_dict: input_dict['bath_bed_ratio'] = [ratio]
    if 'house_size_per_room' in input_dict: input_dict['house_size_per_room'] = [per_room]
    
    # Flip the requested one-hot encoded zip code
    target_zip_col = f'zip_code_{specs.zip_code}'
    zip_warning = None
    if target_zip_col in input_dict:
        input_dict[target_zip_col] = [1.0]
    else:
        zip_warning = f"Zip code {specs.zip_code} not in training data. Defaulting to base market logic."
        
    # Convert to DataFrame ensuring exact column order
    df_input = pd.DataFrame(input_dict)[feature_columns]
    
    # Scale Data
    input_scaled = scaler.transform(df_input.values)
    
    # Get Predictions
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        nn_val = nn_model(input_tensor).cpu().item()
        
    xgb_val = xgb_model.predict(input_scaled)[0]
    
    # Ensemble Average
    final_price = float(((nn_val + xgb_val) / 2) * 100000)
    
    # Return JSON response
    response = {
        "beds": specs.beds,
        "baths": specs.baths,
        "sqft": specs.sqft,
        "zip_code": specs.zip_code,
        "predicted_price": final_price
    }
    
    if zip_warning:
        response["warning"] = zip_warning
        
    return response