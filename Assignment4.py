
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib import gridspec
from wordcloud import WordCloud


# Problem 2
def problem_2(filenames):
    # write your logic here
    data1 = list()
    data2 = list()
    for i in range(0, len(filenames)):
        df = pd.read_csv(filenames[i], header=None)
        data1.append(df.iloc[:, 0])
        print(df.iloc[:, 0])
        data2.append(df.iloc[:, 1])
        print(df.iloc[:, 1])
    ymin = 0
    ymax = 100
    fig = plt.figure(tight_layout=True)
    # 分为一行二列
    gs = gridspec.GridSpec(1, 2)
    plt.suptitle("Test result", fontsize=13, x=0.5, y=0.98)
    # test 1
    # 该figure的位置
    ax = fig.add_subplot(gs[0, 0])
    # ax.boxplot(data1, labels=filenames)
    ax.boxplot(data1, labels=['Class A', 'Class B', 'Class C'])
    ax.set_title('Test1')
    ax.set_ylim([ymin, ymax])
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.grid()
    # test 2
    # 该figure的位置
    ax = fig.add_subplot(gs[0, 1])
    # ax.boxplot(data2, labels=filenames)
    ax.boxplot(data2, labels=['Class A', 'Class B', 'Class C'])
    ax.set_title('Test2')
    ax.set_ylim([ymin, ymax])
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.grid()

    plt.savefig("problem2")


# problem_2(["Assignment4_Data/classA.csv", "Assignment4_Data/classB.csv",
#                  "Assignment4_Data/classC.csv"])


# Problem 3
def problem_3(filenames):
    filenames_num = len(filenames)
    if filenames_num % 2 != 0:
        gs_row = (filenames_num + 1) / 2
    else:
        gs_row = filenames_num / 2
    gs = gridspec.GridSpec(int(gs_row), 2)
    fig = plt.figure(tight_layout=True)
    for i in range(0, filenames_num):
        # 读取数据
        f = open(filenames[i], encoding="utf-8")
        # 转小写
        txt = f.read().lower()
        # 匹配数字、字母、空格
        txt = re.sub(r'[^\w\s]', '', txt)
        # 匹配非数字
        txt = re.sub(r'[^\D]', '', txt)
        # 生成词云
        word_cloud = WordCloud(collocations=False, background_color='white', random_state=5726).generate(txt)
        fig.add_subplot(gs[int(i / 2), int(i % 2)])
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")

    plt.savefig("problem3")


# problem_3(["Assignment4_Data/paragraph1.txt", "Assignment4_Data/paragraph2.txt",
#                  "Assignment4_Data/paragraph3.txt"])


# Problem 4
def problem_4(filename, start, end, target):
    start_time = datetime.strptime(start, "%d/%m/%Y")
    end_time = datetime.strptime(end, "%d/%m/%Y")
    df = pd.read_csv(filename)
    # 删除日期为空的行
    df = df.dropna(subset=['date'])
    # 将str格式时期转为datetime格式
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['Name'].isin(target)]
    df = df[(df['date'] <= end_time) & (df['date'] >= start_time)]
    plt.title("Close value")
    sns.lineplot(x="date", y="close", hue="Name", data=df)
    plt.grid()
    plt.savefig("problem4")


# problem_4("Assignment4_Data/all_stocks_5yr.csv", "1/1/2018", "14/1/2018", ["ABBV", "AIV", "DFS"])


# Problem 5
def problem_5(df):
    new_df = pd.DataFrame(index=df.index, columns=df.columns)

    new_df.loc['First Year'] = (df.loc['First Year'] / df.iloc[0, 0:].sum()).round(decimals=4)
    new_df.loc['Second Year'] = (df.loc['Second Year'] / df.iloc[1, 0:].sum()).round(decimals=4)
    new_df.plot(kind='barh', stacked=True, figsize=(10, 5))
    # 1 Year / Boys
    plt.text(new_df.iloc[0, 0], 0, new_df.iloc[0, 0])
    # 1 Year / Girls
    plt.text(1, 0, new_df.iloc[0, 1])
    # 2 Year / Boys
    plt.text(new_df.iloc[1, 0], 1, new_df.iloc[1, 0])
    # 2 Year / Girls
    plt.text(1, 1, new_df.iloc[1, 1])
    plt.title("Passing Percentage")
    plt.ylabel('Years')
    plt.savefig("problem5")


# problem_5(pd.DataFrame({'Boys': [67, 78], 'Girls': [72, 80], },
#                        index=['First Year', 'Second Year']))


# Problem 6
def problem_6(filename, start, end, column):
    df = pd.read_csv(filename)
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df_grouped = df.groupby("Date").agg({"Temperature": "mean",
                                         "Fuel_Price": "mean",
                                         "CPI": "mean",
                                         "Unemployment": "mean",
                                         "IsHoliday": "min"}).reset_index()
    df_grouped = df_grouped.fillna(method='ffill')
    count_columns_ex_date = len(df_grouped.columns[1:])
    for idx, col in enumerate(df_grouped.columns[1:]):
        plt.subplot(count_columns_ex_date, 1, idx + 1)
        plt.plot(df_grouped["Date"], df_grouped[col])
        plt.ylabel(col)
    plt.savefig("problem6")


# problem_6("Assignment4_Data/Features data set.csv", "1/1/2010", "31/7/2013",
#           ["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"])
