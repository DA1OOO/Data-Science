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
    learning_rate = 0.01
    # 载入文件
    df = pd.read_csv(filename)
    # 设置manual_seed
    torch.manual_seed(5726)
    x = df[predictors].values
    y = df[target].values.reshape(-1, 1)
    # 数据标准化
    sc_x = StandardScaler()
    x = sc_x.fit_transform(x)
    # 数据划分，70%为训练集，30%为测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5726)
    # 转为tensor
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    # 序列初始化模型 输出层 - hidden层- 输出层
    model = nn.Sequential(nn.Linear(len(predictors), 5), nn.ReLU(), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1),
                          nn.Sigmoid())
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 训练数据
    print("===> start training...")
    losses = []
    for epoch in range(500):
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


# print(problem_2("Assignment3_Data/winequality-white-binary.csv",
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


# print(problem_3("Assignment3_Data/winequality-white.csv",
#                 ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
#                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"], "quality"))


# Problem 4
def problem_4(filename, predictors, target):
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


# print(problem_4("Assignment3_Data/Fish.csv", ["Length1", "Length2", "Length3", "Height", "Width"], "Weight"))


# Problem 5
def problem_5(filename, predictors, target):
    # 加载数据
    df = pd.read_csv(filename)
    x = df[predictors]
    y = df[target].values.reshape(-1, 1)
    # 数据标准化
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)
    # 模型初始化
    model = linear_model.LinearRegression()
    model.fit(x, y)
    # 交叉验证
    score = cross_val_score(model, x, y, cv=8, scoring='neg_mean_squared_error')
    score = score * (-1)
    mean_cv_mse = np.mean(score)
    sd_cv_mse = np.std(score)
    return model, mean_cv_mse, sd_cv_mse


# print(problem_5("Assignment3_Data/Fish.csv", ["Length1", "Length2", "Length3", "Height", "Width"], "Weight"))


# Problem 6
def problem_6(train_filename, predictors, test_filename):
    # 加载数据
    train_data = pd.read_csv(train_filename)
    train_data = train_data[predictors]
    test_data = pd.read_csv(test_filename)
    test_data = test_data[predictors]
    # 数据标准化
    sc = StandardScaler()
    train_data = sc.fit_transform(train_data)
    # 确定k值
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_data)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    k = kl.elbow
    # 模型训练
    model = KMeans(init="random", n_clusters=k, n_init=5, max_iter=300, random_state=5726)
    model.fit(train_data)
    # 测试数据结果预测
    result = model.predict(test_data)
    return model, k, result

# print(problem_6("Assignment3_Data/sample1.csv", ["x", "y"], "Assignment3_Data/sample2.csv"))
