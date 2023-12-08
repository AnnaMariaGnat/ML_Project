import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


class NeuralNetwork(nn.Module):

    def __init__(self,ks=3, pad=1):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=ks, padding=pad),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=ks, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")



    def forward(self, x):
        logits = self.stack(x)
        return logits
    


    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

    def runCNN(self, epochs, learning_rate, batch_size, ks, pad, tensor_train_X, tensor_train_y, tensor_test_X, tensor_test_y):

        # Define model
        model = NeuralNetwork(ks, pad).to(self.device)
        print(model)

        # Define loss function
        loss_fn = nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Define data loaders
        train_loader = DataLoader(list(zip(tensor_train_X, tensor_train_y)), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(list(zip(tensor_test_X, tensor_test_y)), batch_size=batch_size, shuffle=False)
        
        for X, y in train_loader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        # Train and test the model
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train(train_loader, model, loss_fn, optimizer)
            self.test(test_loader, model, loss_fn)
        print("Done!")