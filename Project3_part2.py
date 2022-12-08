import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud


def main():
    # 数据读取
    df = pd.read_csv('Project_Data/news.csv')
    # 数据清洗
    df = data_clean(df)
    word_cloud(df['text'].to_string())
    labels = df.label
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    # 原始数据转TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    # 用TF-IDF矩阵训练分类器
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    # 用分类器进行预测，并得到预测精准度
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    # 得到混淆矩阵
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))


# 数据清洗
def data_clean(df):
    return df


# 生成词云
def word_cloud(data):
    print(data)
    plt.imshow(WordCloud(collocations=False, background_color='white', random_state=5726).generate(data),
               interpolation='bilinear')
    plt.savefig('news_word_cloud.png')
    plt.show()


if __name__ == '__main__':
    main()
