from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import joblib

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



# 1. Define the steps in the pipeline
# The Pipeline takes a list of tuples: ('name_of_step', Component())
production_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),                                    # Step 1: Scale the data
    ('model', RandomForestRegressor(max_depth=10, random_state=42))  # Step 2: Train/Predict
])

# 2. Train the ENTIRE pipeline at once
# This scales X_train, remembers the scaling math, and then trains the forest.
production_pipeline.fit(X_train, y_train)

# 3. Predict on test data
# The pipeline AUTOMATICALLY applies the saved scaling math to X_test before predicting!
y_pred = production_pipeline.predict(X_test)

# 4. Check the error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Pipeline RMSE: ${rmse * 100000:,.2f}")

joblib.dump(production_pipeline, 'real_estate_final_model.pkl')

print("Model saved to disk as 'real_estate_model.pkl'!")


from sklearn.model_selection import cross_val_score
#This fold is only to check if our model is stable, it doesn't train our model. just checks if the model is stable across different subsets of the training data.
#If the scores are similar across folds, it suggests that the model is not overfitting and is likely to perform well on unseen data.


# 1. Run the 5-fold evaluation on your training data
# Note: We use 'neg_root_mean_squared_error' because Scikit-Learn expects "higher is better", so it makes errors negative.
scores = cross_val_score(production_pipeline, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5)

# 2. Convert the negative scores back to positive and print them
rmse_scores = -scores
print(f"Scores for each of the 5 folds: {rmse_scores}")

# 3. Print the final, mathematically reliable average
print(f"Average Cross-Validation RMSE: {rmse_scores.mean():.4f}")