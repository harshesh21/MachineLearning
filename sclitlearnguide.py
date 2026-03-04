import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generate 100 random numbers between 0 and 2 for "X"
# rand generates number between 0-1, so we multiply by 2
X = 2 * np.random.rand(100, 1) 

# 2. Generate "y" using the formula y = 3x + 4 + noise
# randn is "Standard Normal" distribution (Gaussian noise)
y = 4 + 3 * X + np.random.randn(100, 1)

# 1. Instantiate the model
lin_reg = LinearRegression()

# 2. Fit (Train) the model
# The model looks at X and y and tries to figure out the "3" and the "4"
lin_reg.fit(X, y)

# 3. Check what it learned
print(f"Intercept (b): {lin_reg.intercept_}") # Should be close to 4
print(f"Slope (m): {lin_reg.coef_}")      # Should be close to 3

# 3. Visualize it
plt.scatter(X, y)
plt.title("Generated Data (y = 3x + 4 + noise)")
plt.xlabel("X Input")
plt.ylabel("y Output")
plt.show()


# Create a new data point (Needs to be a 2D array, hence the double brackets)
X_new = np.array([[0],[2]]) 
print (f"New data points for prediction:\n{X_new}")
# Predict
y_predict = lin_reg.predict(X_new)

print(f"Predictions for X=0 and X=2:\n{y_predict}")


# --- Train/Test Split Example ---

# 2. Split the data
# test_size=0.2 means 20% of data is saved for the "Final Exam"
# random_state=42 is just a seed so we get the same shuffle every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data shape: {X_train.shape}") # Should be (80, 1)
print(f"Testing Data shape:  {X_test.shape}")  # Should be (20, 1)

# 3. Train ONLY on the Training set (Homework)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate ONLY on the Test set (The Exam)
# We hide the real answers (y_test) and ask the model to guess based on X_test
predictions = model.predict(X_test)

# 5. Score the Exam
# Compare the model's guesses (predictions) vs the actual answers (y_test)
mse = mean_squared_error(y_test, predictions)
print(f"Test Set Error (MSE): {mse:.2f}")