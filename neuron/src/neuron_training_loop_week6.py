import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. The Architecture
class SingleNeuronNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(3, 1)

    def forward(self, x):
        return self.layer_1(x)

model = SingleNeuronNet().to(device)

# 2. The Data (X) and The Target (y)
# House: 2500 SqFt, 4 Beds, 10 Years Old
X_train = torch.tensor([[2500.0, 4.0, 10.0]]).to(device)
# Target Price: $500k (represented as 500.0)
y_target = torch.tensor([[500.0]]).to(device)

# 3. The Loss and The Optimizer
criterion = nn.MSELoss() # Mean Squared Error
# Adam is the industry-standard optimizer. 'lr' is the Learning Rate (how big of a step to take)
optimizer = optim.Adam(model.parameters(), lr=0.1)

print("--- Starting GPU Training Loop ---")

# 4. The Loop (Epochs)
epochs = 500
for epoch in range(epochs):
    
    # Step A: The Forward Pass (Make a guess)
    y_pred = model(X_train)
    
    # Step B: Calculate the Loss (How wrong is the guess?)
    loss = criterion(y_pred, y_target)
    
    # Step C: Zero the Gradients 
    # (PyTorch accumulates math by default, we have to wipe the slate clean every loop)
    optimizer.zero_grad()
    
    # Step D: Backpropagation
    # This calculates the exact calculus derivative for every weight in the network
    loss.backward()
    
    # Step E: The Optimizer Step
    # The mechanic reaches in and tweaks the weights based on the derivatives
    optimizer.step()
    
    # Print progress every 100 loops
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:3} | Prediction: ${y_pred.item():,.2f}k | Loss: {loss.item():,.2f}")

print("\n--- Final Result ---")
print(f"Target Price: $500.00k")
print(f"AI Final Prediction: ${model(X_train).item():,.2f}k")