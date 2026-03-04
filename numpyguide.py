import numpy as np
import time

# Create a massive list of 10 million numbers
size = 10_000_000
my_list = list(range(size))
my_array = np.arange(size)

# METHOD 1: The "Developer" way (Python Loop)
start = time.time()
result_list = [x * 2 for x in my_list]
print(f"Python Loop time: {time.time() - start:.4f} seconds")

# METHOD 2: The "ML" way (Vectorization)
start = time.time()
result_array = my_array * 2  # <--- Look at this syntax. No loop.
print(f"Numpy Array time: {time.time() - start:.4f} seconds")

# Check the speedup
# You will likely see NumPy is 50x-100x faster.


#METHOD 3: DOT PRODUCT EXAMPLE
# Imagine you have input data and learned weights in a simple AI model
X = np.array([5, 1])  # Input Data
W = np.array([0.8, 1.5]) # Learned Weights 

weighted_score = np.dot(X, W)

print(f"Final Weighted Score (X dot W): {weighted_score}")