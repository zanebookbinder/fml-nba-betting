import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split


class NeuralNetRegressor(nn.Module):
    def __init__(self, input_features):
        super(NeuralNetRegressor, self).__init__()
        self.input_layer = nn.Linear(input_features, 64)  
        self.hidden_layer1 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)  # Output a single spread value

        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()
        self.optimizer = Adam(self.parameters())

    def forward(self, x):
        x = torch.tensor(x.values).float() 
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.output_layer(x)
        return x

    def train(self, X_train, y_train, epochs=10):
        for _ in range(epochs):
            y_pred = self(X_train)
            y_pred_adj = y_pred * abs(y_train).max() / abs(y_pred).max() # normalize to spread range
            y_pred_adj *= torch.sign(y_pred)
            loss = self.mse_loss(y_pred_adj, torch.tensor(y_train.values).float())
            print()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad() 

    def test(self, X_test):
        with torch.no_grad():
            y_pred = self(X_test)
            return y_pred.numpy() 

