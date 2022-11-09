# <1155182964 DAI Yayuan>
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
    plt.show()


# print(problem_2(["IEMS5726_Assignment4_Data/classA.csv", "IEMS5726_Assignment4_Data/classB.csv",
#                  "IEMS5726_Assignment4_Data/classC.csv"]))


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
        ax = fig.add_subplot(gs[int(i / 2), int(i % 2)])
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")

    plt.savefig("problem3")
    plt.show()


# print(problem_3(["IEMS5726_Assignment4_Data/paragraph1.txt", "IEMS5726_Assignment4_Data/paragraph2.txt",
#                  "IEMS5726_Assignment4_Data/paragraph3.txt"]))


# Problem 4
def problem_4(filename, start, end, target):
    # write your logic here

    plt.savefig("problem4")  # do not call plt.show()


# Problem 5
def problem_5(df):
    # write your logic here

    plt.savefig("problem5")  # do not call plt.show()


# Problem 6
def problem_6(filename, start, end, column):
    # write your logic here

    plt.savefig("problem6")  # do not call plt.show()
