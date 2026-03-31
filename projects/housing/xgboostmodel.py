import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Fetch the data
california = fetch_california_housing()

# 2. Create the Features DataFrame (X)
X = pd.DataFrame(california.data, columns=california.feature_names)

# 3. Create the Target Series (y) - This is the "MedHouseVal"
y = pd.Series(california.target, name='MedHouseVal')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

print("Initiating XGBoost Training Sequence...")

# 1. The Pre-Processing Stage
# We fit the scaler ONLY on the training data to prevent data leakage,
# then we apply that exact math to the test data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. The Engine Configuration (Modern XGBoost 2.0+ API)
# early_stopping_rounds is strictly defined in the constructor now.
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
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# 4. The Evaluation
y_pred = xgb_model.predict(X_test_scaled)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("-" * 40)
print(f"✅ Training Complete.")
print(f"🌲 Total Trees Built: {xgb_model.best_iteration}")
print(f"🎯 Final XGBoost RMSE: ${final_rmse * 100000:,.0f}")
print("-" * 40)

import shap
import matplotlib.pyplot as plt

print("Reverse-engineering the AI with SHAP...")

# 1. Initialize the Tree Explainer with your trained XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# 2. Calculate the SHAP values for your scaled test data
# This calculates exactly how much money each feature added/subtracted for every house
shap_values = explainer.shap_values(X_test_scaled)



# 3. Generate the Global Feature Importance Plot
# Note: If X_test is a Pandas DataFrame, pass X_test to keep the column names. 
# If it's a NumPy array, it will just show "Feature 0, Feature 1", etc.
print("Generating visualization. Close the popup window to continue...")
# We use X_test.columns to grab the original headers from your Pandas DataFrame
# and map them over the raw NumPy matrix.
shap.summary_plot(
    shap_values, 
    X_test_scaled, 
    feature_names=X_test.columns, 
    plot_type="bar"
)


print("Generating the financial receipt for House #0...")

# 1. To use the modern Waterfall plot, SHAP needs an 'Explanation' object, 
# not just the raw numpy array we used for the summary plot.
explainer = shap.TreeExplainer(xgb_model)
shap_values_exp = explainer(X_test_scaled)


# 2. Attach your original Pandas column names to the Explanation object
shap_values_exp.feature_names = list(X_test.columns)

# 3. Generate the Waterfall Plot for a single house (Index 0)
shap.plots.waterfall(shap_values_exp[0])