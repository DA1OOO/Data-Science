# <Your student ID>
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn import metrics
# Problem 2
def problem_2(filename,predictors,target):
    # write your logic here, model is the NN model
    model, test_precision, test_recall = 0, 0, 0
    batch_size, learning_rate = 10, 0.01

    return model, test_precision, test_recall


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# Problem 3
def problem_3(filename,predictors,target):
    # write your logic here, model is the RF model
    model, mean_cv_acc, sd_cv_acc = 0, 0, 0

    return model, mean_cv_acc, sd_cv_acc


from sklearn.svm import SVR
# Problem 4
def problem_4(filename,predictors,target):
    # write your logic here, model is the SVR model
    model, test_mae, test_rmse = 0, 0, 0

    return model, test_mae, test_rmse


from sklearn import linear_model
# Problem 5
def problem_5(filename,predictors,target):
    # write your logic here, model is the MLR model
    model, mean_cv_mse, sd_cv_mse = 0, 0, 0

    return model, mean_cv_mse, sd_cv_mse


from kneed import KneeLocator
from sklearn.cluster import KMeans
# Problem 6
def problem_6(train_filename,predictors,test_filename):
    # write your logic here, model is the k-mean model
    model, k, result = 0, 0, []

    return model, k, result
