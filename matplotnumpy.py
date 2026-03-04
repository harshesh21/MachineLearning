import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Vector 'v' (The data input, e.g., features x=2, y=1)
v = np.array([2, 1])

# 2. Define the Transformation Matrix 'R' (The "weights" learned by the AI)
# This matrix performs a 90-degree counter-clockwise rotation.
# R = [[cos(90)  -sin(90)]
#      [sin(90)   cos(90)]]
rotation_matrix = np.array([
    [0, -1],
    [1, 0]
])

# 3. Apply the transformation using the DOT PRODUCT (Matrix * Vector)
# This is the core operation of deep learning!
v_rotated = np.dot(rotation_matrix, v)

print(f"Original Vector v: {v}")
print(f"Rotated Vector R*v: {v_rotated}")
# Expected output: Rotated Vector R*v: [-1  2] (The arrow moved from (2,1) to (-1,2))

# 4. Optional: Visualize the result
plt.figure(figsize=(5, 5))
plt.axvline(x=0, color='grey', lw=1)
plt.axhline(y=0, color='grey', lw=1)
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector', zorder=2)
plt.quiver(0, 0, v_rotated[0], v_rotated[1], angles='xy', scale_units='xy', scale=1, color='red', label='Rotated Vector', zorder=2)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(alpha=0.5)
plt.legend()
plt.title("Matrix Multiplication as a Transformation")
plt.show()