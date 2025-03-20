
# %%


import pytorch as torch

# %%
import torch.nn as nn
import torch.optim as optim

# Sample data
inputs = torch.randn(5, 10)
targets = torch.tensor([1, 0, 3, 2, 1])

# Define a simple neural network with a softmax output
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 4)
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')