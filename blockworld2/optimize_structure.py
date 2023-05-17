import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from displacement_net import DisplacementNet
from utils import draw_world_torch
# Load the data
data = np.load(sys.argv[1], allow_pickle=True)

# Convert data to PyTorch tensors
features = torch.tensor([d[0] for d in data], dtype=torch.float32)[int(sys.argv[2])].to("cuda")
features /= 8
print(features.shape)

# Define the network architecture
model = DisplacementNet().to("cuda")
# Load model weights
model.load_state_dict(torch.load("model.pt"))

criterion = nn.L1Loss()

# Dump initial world state to png
draw_world_torch(features*8, "initial.png")

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False


num_steps = 10000

xy_offsets = nn.Parameter(torch.zeros((5,2)).to("cuda"), requires_grad=True)
optimizer = optim.Adam([xy_offsets], lr=3e-2)

features = features.view(5,4)
print("Initial features: " + str(features*8))
for i in range(num_steps):
    offset_features = features.clone()
    offset_features[:,0:2] += xy_offsets
    integrated_velocities = model(offset_features.view(1,-1))
    loss = torch.exp(torch.abs(integrated_velocities.norm())) + (torch.abs(xy_offsets[:].norm()))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 1000 == 0:
        print("Step: " + str(i))
        print("Loss: " + str(loss.item()))
        print("xy_offsets: " + str(xy_offsets))
        print("integrated_velocities: " + str(integrated_velocities))        
    


print("Final loss: " + str(loss.item()))
print("Final xy_offsets: " + str(xy_offsets))

features[:,0:2] += xy_offsets
print("Final features: " + str(features))

draw_world_torch(features*8, "final.png")



