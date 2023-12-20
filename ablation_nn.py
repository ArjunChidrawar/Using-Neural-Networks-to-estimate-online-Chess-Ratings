import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

#converting pandas dataframe into tensor data
data = pd.read_csv('ablation_data.csv')
Xmat = data.drop(columns=['avg_rating', 'Unnamed: 0.1']).to_numpy(dtype=np.float64)
Y = data['avg_rating'].to_numpy(dtype=np.float64)

Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.1, random_state=4)
Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.125, random_state=4)

X_train = torch.tensor(Xmat_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(Xmat_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(Xmat_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)


train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
val_dataset = TensorDataset(X_val, Y_val)

#Creating tensor data loaders
batch_size = 100 #specify as 1 for SGD (I just increased it to make it run faster)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
torch.manual_seed(4)

#**COMMENT OUT DROPOUT LINES 
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.act1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.Tanh()
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(hidden2, hidden3)
        self.act3 = nn.Tanh()
        self.dropout3 = nn.Dropout(0.5)
        self.output = nn.Linear(hidden3, output)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.act3(x)
        x = self.output(x)
        return x
    
#layer sizes
input_size = 1801
hidden1 = 128
hidden2 = 64
hidden3 = 32
output = 1
model = NeuralNetwork(input_size, hidden1, hidden2, hidden3, output)


learning_rate = 0.01
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
def train_loop(dataloader, model, loss_fn, optimizer, lamda = 0.0001):
    model.train()
    running_loss = 0
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        #compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, Y.view(-1, 1))

        #L2 regularization
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += lamda*l2_reg

        #Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"Mean Squared Error: {loss.item():.4f}  [{current:>5d}/{size:>5d}]")
    average_loss = running_loss / len(dataloader)
    losses.append(average_loss)


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, Y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, Y.view(-1, 1)).item()

    test_loss /= num_batches
    print(f"Test Error: \n Mean Squared Error: {test_loss:>8f} \n")
    print(f"Test Error: \n Root Mean Squared Error: {math.sqrt(test_loss):>8f} \n")


epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_function, optimizer)
    test_loop(test_dataloader, model, loss_function)
print("Done!")

#For Actual vs Predicted Visualization
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    Y_pred = model(X_test)
Y_pred = Y_pred.detach().numpy()
Y_test = Y_test.numpy()

plt.plot(losses, label='Ablation Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Ablation Training Loss Curve')
plt.legend()
plt.show()

