# <1155182964>
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn import linear_model
from kneed import KneeLocator
from sklearn.cluster import KMeans


# Problem 2
def problem_2(filename, predictors, target):
    # write your logic here, model is the NN model
    model, test_precision, test_recall = 0, 0, 0
    batch_size, learning_rate = 10, 0.01
    # load file
    df = pd.read_csv(filename)
    df = df.iloc[1:]
    # set manual seed
    torch.manual_seed(5726)
    x = df[predictors]
    y = df[target]
    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5726)
    # normalize Data
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    # 转为tensor
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train.to_numpy())
    # 序列初始化模型 输出层 - hidden层- 输出层
    model = nn.Sequential(nn.Linear(len(predictors), 5), nn.ReLU(), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1),
                          nn.Sigmoid())
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train model
    print("===> start training...")
    losses = []
    for epoch in range(5000):
        pred_y = model(x_train)  # Forward propagation
        pred_y = pred_y.squeeze(-1)
        loss = loss_function(pred_y, y_train)
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()  # Backward propagation
        optimizer.step()  # Gradient descent
    print("===> training end.")
    return model, test_precision, test_recall


print(problem_2("IEMS5726_Assignment3_Data/winequality-white-binary.csv",
                ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
                "quality"))


# Problem 3
def problem_3(filename, predictors, target):
    # write your logic here, model is the RF model
    model, mean_cv_acc, sd_cv_acc = 0, 0, 0

    return model, mean_cv_acc, sd_cv_acc


# Problem 4
def problem_4(filename, predictors, target):
    # write your logic here, model is the SVR model
    model, test_mae, test_rmse = 0, 0, 0

    return model, test_mae, test_rmse


# Problem 5
def problem_5(filename, predictors, target):
    # write your logic here, model is the MLR model
    model, mean_cv_mse, sd_cv_mse = 0, 0, 0

    return model, mean_cv_mse, sd_cv_mse


# Problem 6
def problem_6(train_filename, predictors, test_filename):
    # write your logic here, model is the k-mean model
    model, k, result = 0, 0, []

    return model, k, result
