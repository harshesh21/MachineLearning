import torch
import torch.nn as nn

# 1. Define the Device (Your GTX 1080)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Architect the Neural Network
class SingleNeuronNet(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(in_features, out_features)
        # We are passing in 3 features (SqFt, Beds, Age) and outputting 1 (Price)
        self.layer_1 = nn.Linear(3, 1) 

    def forward(self, x):
        # Define how the data flows. In this case, straight through layer 1.
        return self.layer_1(x)

# 3. Instantiate the model and send it to the GPU
model = SingleNeuronNet().to(device)
print("--- Network Architecture ---")
print(model)

# 4. Create a "Fake House" Tensor
# Let's say: [2500 SqFt, 4 Bedrooms, 10 Years Old]
# Note: We must send the data to the same device as the model!
fake_house = torch.tensor([[2500.0, 4.0, 10.0]]).to(device)

# 5. The Forward Pass (Making a prediction)
# We haven't trained this network yet, so the weights are completely random.
with torch.no_grad(): # Tells PyTorch not to calculate gradients just yet (saves memory)
    random_prediction = model(fake_house)

print("\n--- The Forward Pass ---")
print(f"Input House Data (GPU): {fake_house}")
print(f"Random Untrained Prediction (GPU): {random_prediction}")