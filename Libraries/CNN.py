''' Needed libraries '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    ''' Neural Network class for CNN '''
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(self.dropout1(x)))
        x = self.fc2(self.dropout2(x))
        return F.log_softmax(x, dim=1)

class CNN:
    ''' CNN class '''
    def __init__(self, epochs=20, lr=0.01, batch_size=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        # Model initialization:
        self.model = NeuralNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Model parameters:
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

    # def load_data(self, trainX, trainY, testX):
    #     ''' Converts data to tensor and loads into the model '''
    #     if testX == []:
    #         tensor_train_X = torch.Tensor(trainX).reshape(-1, 1, 28, 28)
    #         tensor_train_y = torch.Tensor(trainY).long()
    #         self.train_loader = DataLoader(list(zip(tensor_train_X, tensor_train_y)), batch_size=self.batch_size, shuffle=True)
    #     if trainX == [] and trainY == []:
    #         tensor_test_X = torch.Tensor(testX).reshape(-1, 1, 28, 28)
    #         self.test_loader = DataLoader(tensor_test_X, batch_size=self.batch_size, shuffle=False)

    def load_training(self, trainX, trainY):
        ''' Converts data to tensor and loads into the model '''
        tensor_train_X = torch.Tensor(trainX).reshape(-1, 1, 28, 28)
        tensor_train_y = torch.Tensor(trainY).long()
        self.train_loader = DataLoader(list(zip(tensor_train_X, tensor_train_y)), batch_size=self.batch_size, shuffle=True)

    def load_testing(self, testX):
        ''' Converts data to tensor and loads into the model '''
        tensor_test_X = torch.Tensor(testX).reshape(-1, 1, 28, 28)
        self.test_loader = DataLoader(tensor_test_X, batch_size=self.batch_size, shuffle=False)
    

    def fit(self, trainingX, trainingY):
        ''' Trains the model, returns the fitted model '''
        # Data loading
        self.load_data(trainX=trainingX, trainY=trainingY)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for _, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        return self.model


    def predict(self, testingX):
        ''' Predicts the class of the testing data,
        returns the classes of the testing data '''
        # Data loading
        self.load_data(testX=testingX)

        self.model.eval()
        classes = []
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                classes.append(pred)
        return classes


    def save_model(self, model_name):
        ''' Saves the trained model '''
        if self.model == None:
            print("Model is not trained yet!")
            return
        else:
            torch.save(self.model.state_dict(), model_name)
    

    def load_model(self, model_name):
        ''' Loads the model '''
        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()