from dataloader.dataloader import MNISTLTDataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from swae.models.mnist import MNISTAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = MNISTLTDataLoader(data_dir="data/mnist_lt", train_batch_size=1000, test_batch_size=128)
train_loader, test_loader = data_loader.create_dataloader()

model = MNISTAutoencoder().to(device)

all_z = list()
all_y = list()
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)
    x_recon, z = model(x)
    all_z.append(z.detach())
    all_y.append(y)

X = torch.cat(all_z, dim=0)
y = torch.cat(all_y, dim=0)

print(X.shape, y.shape)

class MLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

classifier = MLP(input_size=2, hidden_size=100, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in dataloader:

        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    pred = list()
    corr = list()
    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)
        x_recon, z = model(x)
        z = z.detach()
        outputs = classifier(z)
        _, predicted = torch.max(outputs, 1)

        pred.append(predicted)
        corr.append(y)
    
    pred = torch.cat(pred, dim=0)
    corr = torch.cat(corr, dim=0)

    print(pred.shape, corr.shape)
    print(torch.sum(pred == corr) / pred.shape[0])
