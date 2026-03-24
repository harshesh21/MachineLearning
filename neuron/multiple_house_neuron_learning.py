import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. THE SEED LOCK (Absolute Reproducibility)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define the Architecture
class HoustonDeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

model = HoustonDeepNet().to(device)

# 3. Raw Data (NumPy Arrays instead of Tensors initially)
X_raw = np.array([
    [2500.0, 4.0, 10.0],
    [1500.0, 3.0, 40.0],
    [3500.0, 5.0, 2.0],
    [2000.0, 3.0, 15.0],
    [4000.0, 6.0, 50.0]
])

y_target = torch.tensor([[5.0], [2.5], [8.5], [3.5], [6.0]], dtype=torch.float32).to(device)

# 4. THE SCALER BRIDGE 
# Scale the data using Scikit-Learn, THEN convert to a GPU Tensor
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_train = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# 5. The Training Engine
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("--- Training Stable Neural Network ---")

epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1:4} | Loss: {loss.item():.4f}")

print("\n--- Final Deterministic Predictions ---")
final_preds = model(X_train)
for i in range(5):
    print(f"House {i+1} | Target: ${y_target[i].item()*100:,.0f}k | Prediction: ${final_preds[i].item()*100:,.0f}k")

print("\n--- The Reality Check (Generalization Test) ---")

# 1. Create a brand new house
# 2600 SqFt, 4 Beds, 10 Years Old
new_house_raw = np.array([[8000.0, 15.0, 10.0]])

# 2. Scale it using the EXACT SAME rules as the training data
new_house_scaled = scaler.transform(new_house_raw)

# 3. Convert to Tensor and predict
new_house_tensor = torch.tensor(new_house_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    new_prediction = model(new_house_tensor)
    
print(f"Target logic: ~ $520k")
print(f"AI Prediction for New House: ${new_prediction.item()*100:,.0f}k")