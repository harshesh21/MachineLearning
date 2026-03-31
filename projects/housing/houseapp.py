import joblib
import numpy as np

print("Booting up the valuation engine...")

# 1. Load the compiled Pipeline (Scaler + 300-Tree Gradient Boosting Model)
app_model = joblib.load('optimized_real_estate_model.pkl')

# 2. Input a brand new, unseen property listing
# Features: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
# Note: This is dummy data formatted for the California dataset architecture.
new_listing = np.array([[6.5, 10.0, 7.1, 3.2, 900.0, 2.8, 34.2, -118.1]])

# 3. Generate the prediction
# The Pipeline catches this raw array, mathematically scales it in the background, 
# and feeds it through the 300 sequential trees.
predicted_price = app_model.predict(new_listing)[0] * 100000

# 4. Run the business logic against the target parameters
print("-" * 40)
print(f"Fair Market Value Prediction: ${predicted_price:,.2f}")

target_budget = 500000

if predicted_price <= target_budget:
    print("Status: 🟢 WITHIN BUDGET. Action: Shortlist for review.")
else:
    print("Status: 🔴 OVER BUDGET. Action: Discard or flag for aggressive negotiation.")
print("-" * 40)