import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from displacement_net import DisplacementNet
# Load the data
data = np.load(sys.argv[1], allow_pickle=True)

# Convert data to PyTorch tensors
features = torch.tensor([d[0] for d in data], dtype=torch.float32).to("cuda")
labels = torch.tensor([d[1] for d in data], dtype=torch.float32).to("cuda")

# Normalize last dimension of labels
labels = labels / torch.norm(labels, dim=-1, keepdim=True)

# Normalize features
WH = 8
features /= 8

# Flatten the last 2 dimensions of features
features = features.view(features.shape[0], -1)
num_features = features.shape[1]
print("Features shape", features.shape)

# Create a dataset and dataloader
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)

# Define the network architecture
# The input size depends on how many blocks you have and how you represent them
# Here, we assume each block is represented by a 4-element tuple and we have 10 blocks
NETWORK_WIDTH=128


model = DisplacementNet()
# Load model.pt if it exists
try:
    model.load_state_dict(torch.load("model.pt"))
except:
    pass
model = model.to("cuda")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-3)

losses = []

# Train the model
num_epochs = 100
ema_loss = torch.tensor(0.0).to("cuda")
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Forward pass
        inputs += torch.randn_like(inputs) * 0.0001
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)

        # Update ema_loss
        ema_loss = 0.99 * ema_loss + 0.01 * loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(ema_loss.item())
    
    # Print loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {ema_loss.item()}')
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "model.pt")

# Save the model
torch.save(model.state_dict(), "model.pt")


# Plot losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.savefig("losses.png")

