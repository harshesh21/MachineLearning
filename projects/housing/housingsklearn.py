import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# 1. Fetch the data
california = fetch_california_housing()

# 2. Create the Features DataFrame (X)
X = pd.DataFrame(california.data, columns=california.feature_names)

# 3. Create the Target Series (y) - This is the "MedHouseVal"
y = pd.Series(california.target, name='MedHouseVal')

# 4. Look at the first 5 rows to understand the "Schema"
# print(X.head())
# print("\nTarget (y) - First 5 rows:")
# print(y.head())
# print("\nTarget name ",california.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data is ready! X_train shape:", X_train.shape)
print("Data is ready! X_train shape:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Test Set Error (MSE): {mse:.2f}")
linear_mse_sqrt = np.sqrt(mse)

# 3. Print the comparison
print(f"Training Error using Linear Regression (The Practice Test): ${linear_mse_sqrt * 100000:,.2f}")


# 1. Initialize the Forest (Regression version!)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 2. Train it
rf_reg.fit(X_train, y_train)

# 3. Predict and Calculate Error
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

df= pd.DataFrame({'Feature': X.columns, 'Importance': rf_reg.feature_importances_}).sort_values(by='Importance', ascending=False)
print(df)
print(f"Random Forest MSE: {mse_rf:.4f}")
print(f"Random Forest RMSE: ${np.sqrt(mse_rf) * 100000:,.2f}")

# we want to run the model on the trained set, and if the error is very low, it means the model has "memorized" the training data, and is likely overfitting.
# if so we need to give it max depth to prevent if from memorizing and guessing the price of the house after the depth is reached.

# 1. Have the model predict on the data it ALREADY trained on
y_pred_train = rf_reg.predict(X_train)

# 2. Calculate the error for those training predictions
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)

# 3. Print the comparison
print(f"Random forest Training Error on trained set (The Practice Test): ${rmse_train * 100000:,.2f}")
print(f"Testing Error (The Final Exam):     $50,534.00")



#here we will give max depth and see if training error is close to our initial test error
# The depths we want to test (None means infinite depth / the default)
depths_to_test = [5, 10, 20, None]

for depth in depths_to_test:
    # 1. Build a new model with the specific depth limit
    rf_tuned = RandomForestRegressor(max_depth=depth, random_state=42)
    
    # 2. Train it
    rf_tuned.fit(X_train, y_train)
    
    # 3. Predict on BOTH Training and Testing data
    train_preds = rf_tuned.predict(X_train)
    test_preds = rf_tuned.predict(X_test)
    
    # 4. Calculate RMSE for both
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds)) * 100000
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds)) * 100000
    
    # 5. Print the results
    print(f"--- Max Depth: {depth} ---")
    print(f"Train Error: ${train_rmse:,.0f}")
    print(f"Test Error:  ${test_rmse:,.0f}")
    print(f"Gap:         ${test_rmse - train_rmse:,.0f}\n")



import joblib

# 1. Train your final chosen model (using None for the best Test Score)
final_model = RandomForestRegressor(max_depth=None, random_state=42)
final_model.fit(X_train, y_train)

# 2. Save the model to your hard drive
joblib.dump(final_model, 'real_estate_model.pkl')

print("Model saved to disk as 'real_estate_model.pkl'!")