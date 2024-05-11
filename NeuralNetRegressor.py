import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split


class NeuralNetRegressor(nn.Module):
    def __init__(self, input_features):
        super(NeuralNetRegressor, self).__init__()
        self.input_layer = nn.Linear(input_features, 32)  
        self.hidden_layer1 = nn.Linear(32, 16)
        self.hidden_layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)  # Output a single spread value

        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()
        self.optimizer = Adam(self.parameters())

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

    def train(self, X_train, y_train, epochs=10000):
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)

        X_train = F.normalize(X_train, p=2, dim=1)

        losses = []
        for i in range(epochs):
            if not i % 500:
                print('epoch', i, ' loss:', losses[-1] if losses else None)
            y_pred = self(X_train)
            y_pred_adj = y_pred
            loss = self.mse_loss(y_pred_adj.squeeze(), y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return losses

    def test(self, X_test):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        X_test = F.normalize(X_test, p=2, dim=1)

        with torch.no_grad():
            y_pred = self(X_test)
            return y_pred.numpy()

