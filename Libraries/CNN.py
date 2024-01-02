''' Needed libraries '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
from sklearn.base import BaseEstimator
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
home_directory = os.path.dirname(parent_directory)
exported_data_path = os.path.join(home_directory, 'Exported_Data')

class NeuralNetwork(nn.Module): # original 00
    # Our initial model
    ''' Neural Network class for CNN '''
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return F.log_softmax(self.seq(x), dim=1)



class CNN(BaseEstimator):
    ''' CNN class '''
    def __init__(self, epochs=28, lr=0.0025474653904364957, batch_size=100, step_size=10, gamma=0.08156406080977326):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        # Model initialization:
        self.model = NeuralNetwork()     
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Model parameters:
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        self.criterion = nn.CrossEntropyLoss()
        # History:
        self.history = {'loss': [], 'val_loss': []}


    def load_training(self, trainX, trainY):
        ''' Converts data to tensor and loads into the model '''
        tensor_train_X = torch.Tensor(trainX).reshape(-1, 1, 28, 28)
        tensor_train_y = torch.Tensor(trainY).long()
        self.train_loader = DataLoader(list(zip(tensor_train_X, tensor_train_y)), batch_size=self.batch_size, shuffle=True)


    def load_testing(self, testX):
        ''' Converts data to tensor and loads into the model '''
        tensor_test_X = torch.Tensor(testX).reshape(-1, 1, 28, 28)
        self.test_loader = DataLoader(tensor_test_X, batch_size=self.batch_size, shuffle=False)


    def fit(self, trainingX, trainingY, validationX=None, validationY=None):
        ''' Trains the model, returns the fitted model '''
        try:
            # Validate input data
            if not isinstance(trainingX, np.ndarray):
                raise ValueError("trainingX must be a numpy array")
            if not isinstance(trainingY, np.ndarray):
                raise ValueError("trainingY must be a numpy array")
            # Data loading
            self.load_training(trainX=trainingX, trainY=trainingY)
            self.model.train()
            for epoch in range(1, self.epochs + 1):
                epoch_losses = []
                for _, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    epoch_losses.append(loss.item()) # save loss
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                # Record training loss
                epoch_loss = np.mean(epoch_losses)
                self.history['loss'].append(epoch_loss)

                # Record validation loss and accuracy
                if validationX is not None and validationY is not None:
                    val_loss = self.evaluate(validationX, validationY)
                    self.history['val_loss'].append(val_loss)
                
        except Exception as e:
            print(f"An error occurred: {e}")
        return self.model
    

    def predict(self, testingX):
        ''' Predicts the class of the testing data,
        returns the classes of the testing data '''
        # Data loading
        self.load_testing(testX=testingX)

        self.model.eval()
        classes = torch.Tensor()
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                if self.device == "cuda":
                    classes = torch.cat((classes, pred.cuda()), dim=0)
                else:
                    classes = torch.cat((classes, pred.cpu()), dim=0)
        
        # Convert to numpy array
        classes = np.array(classes)
        classes = classes.astype(int)
        classes = classes.reshape(classes.shape[0],)
        return classes


    def evaluate(self, testingX, testingY):
        ''' Evaluates the model, returns the accuracy '''
        # Data loading
        testingX = torch.Tensor(testingX).reshape(-1, 1, 28, 28)
        testingY = torch.Tensor(testingY).long()
        loaderx = DataLoader(testingX, batch_size=self.batch_size, shuffle=False)
        loadery = DataLoader(testingY, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(zip(loaderx, loadery)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        self.accuracy = correct / total
        return self.accuracy
    

    def save_model(self, model_name='CNNmodel'):
        ''' Saves the trained model '''
        if self.model == None:
            print("Model is not trained yet!")
            return
        else:
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'history': self.history}, os.path.join(exported_data_path, model_name))


    def load_model(self, model_name='CNNmodel'):
        ''' Loads the model '''
        checkpoint = torch.load(os.path.join(exported_data_path, model_name))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.model.eval()


    def summary(self):
        ''' Prints a detailed summary of the model '''
        print("Model summary:")
        print("Epochs:", self.epochs)
        print("Learning rate:", self.lr)
        print("Layers:")
        summary(self.model, input_size=(1, 28, 28))  # Assuming input size is (1, 28, 28)
        print("Criterion:")
        print(self.criterion)