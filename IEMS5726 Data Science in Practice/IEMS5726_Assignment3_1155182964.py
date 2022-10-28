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
    # set manual seed
    torch.manual_seed(5726)
    x_train, x_test, y_train, y_test = train_test_split()
    print(df.to_string())
    return model, test_precision, test_recall


print(problem_2("IEMS5726_Assignment3_Data/winequality-white-binary.csv",
                ["fixed acidity", "volatile acidity", "citric acid", "residual", "sugar", "chlorides",
                 "free sulfur dioxide", "total sulfur", "dioxide", "density", "pH", "sulphates", "alcohol"],
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
