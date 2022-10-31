# <1155182964>
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    batch_size, learning_rate = 10, 0.01
    # 载入文件
    df = pd.read_csv(filename)
    # 设置manual_seed
    torch.manual_seed(5726)
    x = df[predictors]
    y = df[target]
    # 数据划分，70%为训练集，30%为测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5726)
    # 数据标准化
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    # 转为tensor
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train.to_numpy())
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test.to_numpy())
    # 序列初始化模型 输出层 - hidden层- 输出层
    model = nn.Sequential(nn.Linear(len(predictors), 5), nn.ReLU(), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1),
                          nn.Sigmoid())
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 训练数据
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

    pred_y_test = model(x_test)
    pred_y_test = pred_y_test.detach().numpy()

    # 将预测的概率转化为0-1
    pred_y_test = np.around(pred_y_test, 0).astype(int)
    test_precision = metrics.precision_score(y_test, pred_y_test)
    test_recall = metrics.recall_score(y_test, pred_y_test)

    return model, test_precision, test_recall


# print(problem_2("IEMS5726_Assignment3_Data/winequality-white-binary.csv",
#                 ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
#                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
#                 "quality"))


# Problem 3
def problem_3(filename, predictors, target):
    # 数据加载
    df = pd.read_csv(filename)
    x = df[predictors]
    y = df[target]
    # 数据标准化
    sc = StandardScaler()
    x = sc.fit_transform(x)
    # 模型训练
    model = RandomForestClassifier(random_state=5726)
    model.fit(x, y)
    # 8折交叉
    score = cross_val_score(model, x, y, cv=8)
    # 求均值和标准差
    mean_cv_acc = np.mean(score)
    sd_cv_acc = np.std(score)
    return model, mean_cv_acc, sd_cv_acc


# print(problem_3("IEMS5726_Assignment3_Data/winequality-white.csv",
#                 ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
#                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"], "quality"))


# Problem 4
def problem_4(filename, predictors, target):
    model, test_mae, test_rmse = 0, 0, 0
    # 加载数据
    df = pd.read_csv(filename)
    x = df[predictors]
    y = df[target].values.reshape(-1, 1)
    # 数据标准化
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)
    # 数据划分，70%为训练集，30%为测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5726)
    # 初始化分类器
    model = SVR(kernel='poly')
    # 模型训练
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = sqrt(mean_squared_error(y_test, y_pred))
    return model, test_mae, test_rmse


print(problem_4("IEMS5726_Assignment3_Data/Fish.csv", ["Length1", "Length2", "Length3", "Height", "Width"], "Weight"))


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
