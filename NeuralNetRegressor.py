import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class NeuralNetRegressor(nn.Module):
    def __init__(self, input_features, dropout_prob=0.3, lr=0.0001):
        super(NeuralNetRegressor, self).__init__()
        self.input_layer = nn.Linear(input_features, 32)  
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.hidden_layer1 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.hidden_layer2 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.hidden_layer3 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)  # Output a single spread value

        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # self.device = torch.device("cpu")
        
        # Uncomment to train on GPU
        self.device = torch.device("mps")
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.dropout1(x)
        x = self.relu(self.hidden_layer1(x))
        x = self.dropout2(x)
        x = self.relu(self.hidden_layer2(x))
        x = self.dropout3(x)
        x = self.relu(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x

    def train(self, X_train, y_train, epochs=200, batch_size=256):
        X_train = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).to(self.device)

        # X_train = F.normalize(X_train, p=2, dim=1)
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for i in range(epochs):
            epoch_losses = []
            if not i % 100:
                print('epoch', i, ' loss:', losses[-1] if losses else None)
            for X, y in loader:
                y_pred = self(X)
                y_pred_adj = y_pred
                loss = self.mse_loss(y_pred_adj.squeeze(), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(sum(epoch_losses) / len(epoch_losses))

        return losses

    def test(self, X_test):
        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        # X_test = F.normalize(X_test, p=2, dim=1)

        with torch.no_grad():
            y_pred = self(X_test)
            return y_pred.cpu().numpy()

