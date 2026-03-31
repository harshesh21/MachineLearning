from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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



# 1. Build the new pipeline with the Gradient Boosting engine
gb_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    # We use a learning_rate to control how aggressively it fixes mistakes
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)) 
])

gb_pipeline.fit(X_train, y_train)
y_pred = gb_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Gradient Boosting Pipeline RMSE: ${rmse * 100000:,.2f}")


# 2. Run the 5-Fold Cross-Validation directly on the new pipeline
gb_scores = cross_val_score(gb_pipeline, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5)

# 3. Calculate and print the results
gb_rmse_scores = -gb_scores
print(f"Gradient Boosting 5-Fold Scores: {gb_rmse_scores}")
print(f"Average GB RMSE: {gb_rmse_scores.mean():.4f}")



#we are going to create 27 different variations of the model by changing the number of trees, learning rate, and max depth.
#Then we will run each variation through 5-fold cross validation to see which one performs best on average across the folds.
#This is called a "Grid Search" because we are searching through a grid of possible settings to find the best one.
# THIS IS THE ULTIMATE BEST AVERAGE TO GET

from sklearn.model_selection import GridSearchCV

# 1. Define the dictionary of settings to test
# We are testing 3 tree counts, 3 learning rates, and 3 max depths (3x3x3 = 27 combinations)
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [3, 4, 5]
}

# 2. Configure the Grid Search
# cv=5 means 5-Fold Cross Validation. 
# 27 combinations * 5 folds = 135 total training runs!
grid_search = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1  # Developer trick: -1 tells Scikit-Learn to use all your CPU cores to run in parallel
)

# 3. Launch the automated factory
print("Starting Grid Search... this might take a minute as it runs 135 training cycles.")
grid_search.fit(X_train, y_train)

# 4. Extract the winning results
best_rmse = -grid_search.best_score_
print(f"Winning Architecture RMSE: ${best_rmse * 100000:,.0f}")
print(f"Best Settings: {grid_search.best_params_}")

print(f"Best Cross-Validation Score: {grid_search.cv_results_['mean_test_score'][grid_search.best_index_]:.4f}")

import joblib

# 1. Save the winning architecture (Scaler + Tuned Model) to disk
joblib.dump(grid_search, 'optimized_real_estate_model.pkl')

print("Production Build Completed: 'optimized_real_estate_model.pkl' saved to disk.")