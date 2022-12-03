# <1155182964 daiyayuan>
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# Problem 2
def problem_2(a, b):
    # 向量内积
    # 如果向量维度不一致
    if a.shape != b.shape:
        return 'err dimension'
    # for i in range(0, a.size):
    #     output += a[i] * b[i]
    # return output
    output = np.dot(a, b)
    return output


# Problem 3
def problem_3(a, b):
    # 向量外积
    if a.shape != b.shape:
        return 'err dimension'
    output = np.outer(a, b)
    return output


# Problem 4
def problem_4(a, b):
    if a.shape != b.shape:
        return 'err dimension'
    output = a * b
    return output

# Problem 5
def problem_5(filename, col):
    full_col = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'MEDV']
    ############### Step A B C ###############
    temp_df = pd.read_csv(filename).dropna().reset_index(drop=True)
    ###############   Step D   ###############
    # 对输入列名进行数据清洗,读取指定列
    for i in col:
        if i not in full_col:
            col.remove(i)
    # 如果输入列名列表为空
    if len(col) == 0:
        col = full_col
    # 读取temp_df的指定列col
    df = temp_df[col]
    return df


# Problem 6
def problem_6(filename, threshold):
    ###############   Step A   ###############
    temp_df = pd.read_csv(filename)
    df = temp_df.fillna(temp_df.mean())
    #############   Step A B C   #############
    df = df[threshold.keys()]
    for key, value in threshold.items():
        df = df[df[key] > value]
    ###############   Step D   ###############
    df = df.reset_index(drop=True)
    ###############   Step E   ###############
    # 对给定的门槛取keys转换为list，然后对list进行排序
    temp_list = list(threshold.keys())
    list.sort(temp_list)
    df = df[temp_list]
    ###############   Step F   ###############
    # 以temp_list作为排序的优先级，对数值进行排序
    df.sort_values(by=temp_list, inplace=True, ascending=True)
    return df


# Problem 7
def problem_7(filename, n, col, threshold):
    ###############  Step A  ###############
    temp_df = pd.read_csv(filename)
    # list存取要读取的行索引
    list = []
    df_size = temp_df.shape[0]
    for i in range(0, 99):
        if i * n < df_size:
            list.append(i * n)
    df = pd.read_csv(filename)
    df = df.iloc[list]
    ###############   Step B   ###############
    df = df[col].dropna()
    ###############   Step C   ###############
    df[col[0]] = np.where(df[col[0]] > threshold, 'high', 'low')
    ###############   Step D   ###############
    meanhigh, meanlow = df.groupby(col[0])[col[1]].mean()
    return df, meanhigh, meanlow


# Problem 8
def problem_8(df1, df2):
    # df1.insert(0, 1, 1)
    #
    # temp_k = dot(dot(inv(dot(df1.T, df1)), df1.T), df2)
    # k = np.array([temp_k[0][0],temp_k[1][0]])
    #
    # temp_b = df2.values - np.matmul(df1.values, k)
    # temp_b = np.abs(temp_b)
    # b = temp_b.min()
    # return k, b
    df1.insert(loc=0, column='TEMP', value=1)
    x = np.linalg.lstsq(df1, df2, rcond=None)
    k = []
    b = x[0][0][0]
    for i in range(1, x[0].size):
        k.append(x[0][i][0])
    return k, b


# Problem 9
def problem_9(df):
    # s = df.corr('spearman').iat[0,1]
    # p = df.corr('pearson').iat[0,1]
    s = spearmanr(df[df.columns.values[0]], df[df.columns.values[1]])[0]
    p = pearsonr(df[df.columns.values[0]], df[df.columns.values[1]])[0]
    return s, p