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
    x_test = sc.fit_transform(x_test)
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

    pred_y_test = model(x_test)
    pred_y_test = pred_y_test.detach().numpy()

    # 将预测的概率转化为0-1
    pred_y_test = np.around(pred_y_test, 0).astype(int)
    test_recall = metrics.recall_score(y_test, pred_y_test)
    test_precision = metrics.precision_score(y_test, pred_y_test)

    return model, test_precision, test_recall


print(problem_2("IEMS5726_Assignment3_Data/winequality-white-binary.csv",
                ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
                "quality"))


# Problem 3
def problem_3(filename, predictors, target):
    # write your logic here, model is the RF model
    model, mean_cv_acc, sd_cv_acc = 0, 0, 0
    # 数据加载
    df = pd.read_csv(filename)
    x = df[predictors]
    y = df[target]

    return model, mean_cv_acc, sd_cv_acc


# print(problem_3("IEMS5726_Assignment3_Data/winequality-white.csv",
#                 ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
#                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"], "quality"))


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
