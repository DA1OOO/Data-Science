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
    data1 = pd.DataFrame()
    data2 = pd.DataFrame()
    for i in range(0, len(filenames)):
        df = pd.read_csv(filenames[i])
        data1[filenames[i]] = df.iloc[:, 0]
        data2[filenames[i]] = df.iloc[:, 1]
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
    ax.boxplot(data1, labels=['Class A', 'Class B', 'Class C'])
    ax.set_title('Test2')
    ax.set_ylim([ymin, ymax])
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.grid()

    plt.savefig("problem2")  # do not call plt.show()


print(problem_2(["IEMS5726_Assignment4_Data/classA.csv", "IEMS5726_Assignment4_Data/classB.csv",
                 "IEMS5726_Assignment4_Data/classC.csv"]))


# Problem 3
def problem_3(filenames):
    # write your logic here
    f = open(filenames[0], encoding="utf-8")
    word_cloud = WordCloud(collocations=False, background_color=
    'white').generate(f.read())
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("problem3")  # do not call plt.show()
    plt.show()


print(problem_3(["IEMS5726_Assignment4_Data/paragraph1.txt", "IEMS5726_Assignment4_Data/paragraph2.txt", "IEMS5726_Assignment4_Data/paragraph3.txt"]))


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
