from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from utils import sigmoid
import seaborn as sns
import numpy as np
import time


class Model_AutoEncoder(nn.Module):

    def __init__(self, in_features, hidden_units, Batch):
        super(Model_AutoEncoder, self).__init__()

        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.linear_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units // 2)
        self.linear_3 = nn.Linear(in_features=hidden_units // 2, out_features=hidden_units)
        self.linear_4 = nn.Linear(in_features=hidden_units, out_features=in_features)

        self.in_features = in_features
        self.costs = []
        self.criterion = nn.MSELoss()
        self.activation = nn.ReLU()
        self.hidden_units = hidden_units
        self.Batch = Batch
        self.zeros = torch.zeros(self.in_features)
        self.criterion_mean = nn.MSELoss()
        self.reg_mean = 0
        self.batchnorm_1 = nn.BatchNorm1d(num_features=self.hidden_units)
        self.batchnorm_2 = nn.BatchNorm1d(num_features=self.hidden_units // 2)
        self.batchnorm_3 = nn.BatchNorm1d(num_features=self.hidden_units)

    def forward(self, X):

        output = self.linear_1(X)
        if self.Batch:
            output = self.batchnorm_1(output)
        output_1 = self.activation(output)

        output_encode = self.linear_2(output_1)
        if self.Batch:
            output_encode = self.batchnorm_2(output_encode)
        output_2 = self.activation(output_encode)

        output = self.linear_3(output_2)
        if self.Batch:
            output = self.batchnorm_3(output)
        output_3 = self.activation(output)

        output = self.linear_4(output_3)

        return output, output_encode

    def one_step_train(self, X_train, Y_train, optimizer):

        optimizer.zero_grad()

        output, _ = self.forward(X_train)

        loss = self.criterion(output,
                              Y_train)  # + self.reg_mean * self.criterion_mean(torch.mean(output, dim = 0) , self.zeros)

        self.costs.append(self.in_features * loss.item())

        loss.backward()

        optimizer.step()

    def Train(self, X_train, Y_train, num_epochs, batch_size, optimizer):
        for i in range(num_epochs):
            loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)
            for X in loader:
                self.one_step_train(X, X, optimizer)
            if i % 50 == 0:
                print("Epoch =", i)
        plt.figure("Cost")
        plt.plot(self.costs)
        # plt.show()

    def predict(self, X):
        predictions, encoding = self.forward(X)
        return predictions.detach(), encoding.detach()


def train_AutoEncoder(X, y, num_epochs, batch_size, learning_rate, reg_lambda, hidden_units, Batch, device):
    in_features = X.shape[1]
    model = Model_AutoEncoder(in_features, hidden_units, Batch)
    optimizer = optim.Adam(model.parameters(), weight_decay=reg_lambda)  # , lr = learning_rate)

    model.to(device)
    X = X.to(device)
    y = y.to(device)

    initial_time = time.time()
    model.Train(X_train=X, Y_train=y, num_epochs=num_epochs, batch_size=batch_size, optimizer=optimizer)
    final_time = time.time()
    print("Training time  =", final_time - initial_time)

    model.eval()
    train_pred, train_encoding = model.predict(X)
    train_MSE = mean_squared_error(y.cpu().numpy(), train_pred.cpu().numpy()) * in_features
    print("Training MSE   =", train_MSE)

    return train_pred, train_encoding, model, in_features


def AutoEncoder(X, y, num_epochs, batch_size, learning_rate, reg_lambda, hidden_units, Batch, device):
    print('Num epochs   =', num_epochs)
    print('Lambda reg   =', reg_lambda)
    print("Hidden units =", hidden_units)
    print("Batch        =", Batch)
    print()

    X_train, X_val_AE, y_train, y_val_AE = train_test_split(X, y, test_size=0.2)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(X_train)
    X_val_AE = scaler_x.transform(X_val_AE)

    y_train = scaler_y.fit_transform(y_train)
    y_val_AE = scaler_y.transform(y_val_AE)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))

    X_val_AE = torch.from_numpy(X_val_AE.astype(np.float32))
    y_val_AE = torch.from_numpy(y_val_AE.astype(np.float32))

    train_pred, train_encoding, model, in_features = train_AutoEncoder(X_train, y_train, num_epochs, batch_size,
                                                                               learning_rate,
                                                                               reg_lambda, hidden_units, Batch,
                                                                               device)

    X_val_AE = X_val_AE.to(device)
    y_val_AE = y_val_AE.to(device)

    val_pred, val_encoding = model.predict(X_val_AE)
    val_MSE = mean_squared_error(y_val_AE.cpu().numpy(), val_pred.cpu().numpy()) * in_features
    print("Validation MSE =", val_MSE)
    print()

    return train_pred.cpu().numpy(), val_pred.cpu().numpy(), scaler_x, model  # train_encoding.cpu().numpy(), val_encoding.cpu().numpy(), scaler_y


def compute_distance(predictions, X, y, threshold, mode):
    sq_difference = torch.sum((predictions - X) ** 2, dim=1, keepdim=True)
    sq_difference = sq_difference.cpu().numpy()
    sq_difference = np.sqrt(sq_difference)

    # plt.figure("Distribution of errors", figsize=(7, 7))
    # plt.title("Distribution of errors")
    # sns.distplot(sq_difference)
    # plt.xlabel("Reconstruction error")
    # print()
    # print()
    #
    # plt.figure("Scatterplot of errors", figsize=(7, 7))
    # plt.title("Scatterplot of errors")
    # sns.scatterplot(sq_difference[:, 0], sq_difference[:, 0], hue=y, alpha=0.8)
    # plt.xlabel("Reconstruction error")
    # plt.ylabel("Reconstruction error")

    if mode == "train":
        y_prob = (sigmoid(sq_difference - threshold))
        # plt.figure("Distribution of probabilities")
        # sns.distplot(y_prob)
        train_ROC_AUC = roc_auc_score(y, y_prob[:, 0])
        print("Training   AUC =", train_ROC_AUC)
    return sq_difference  # , y_prob


def predict_Anomalies(X, y, threshold, model, mode):
    X_pred, _ = model.predict(X)
    sq_difference_val = compute_distance(X_pred, X, y, threshold, mode)
    pred_class = (sq_difference_val > threshold).astype(int)
    return pred_class
