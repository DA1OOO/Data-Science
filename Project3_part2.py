import re
import numpy as np
import pandas as pd
import seaborn as sn
from nltk.corpus import stopwords
from matplotlib import pyplot as plt, gridspec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 数据清洗
def data_clean(df):
    print('===> Data cleaning...')
    # 删除空列
    df = df.dropna()
    # 根据text列进行去重
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    # 去掉标点
    p = re.compile(r'[^\w\s]+')
    df['text'] = [p.sub('', x) for x in df['text'].tolist()]
    # 去掉数字
    p = re.compile(r'[^\D]+')
    df['text'] = [p.sub('', x) for x in df['text'].tolist()]
    # 去除停用词
    stop_word = stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_word)]))
    print('===> Data cleaning finished!')
    return df


# 生成词云
def word_cloud(data):
    real_news = data[data['label'] == 'REAL']
    fake_news = data[data['label'] == 'FAKE']
    gs = gridspec.GridSpec(2, 1)
    fig = plt.figure(tight_layout=True)
    real_word_cloud = WordCloud(collocations=False, background_color='white', random_state=5726).generate(
        real_news['text'].to_string().lower())
    fig.add_subplot(gs[0, 0])
    plt.imshow(real_word_cloud, interpolation='bilinear')
    plt.title('Real News')

    fake_word_cloud = WordCloud(collocations=False, background_color='white', random_state=5726).generate(
        fake_news['text'].to_string().lower())
    fig.add_subplot(gs[1, 0])
    plt.imshow(fake_word_cloud, interpolation='bilinear')
    plt.title('Fake News')

    plt.savefig('news_word_cloud.png')
    print('===> Word cloud generated path: /news_word_cloud.png')


def draw_confusion_matrix(cm, name):
    plt.close()
    df_cm = pd.DataFrame(cm)
    ax = sn.heatmap(df_cm, annot=True, fmt='.20g')
    ax.set_title(name)  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig(name + '.png')
    print('===> Confusion Matrix file path: /' + name + '.png')


def passive_aggressive_classify(x_train, y_train, x_test, y_test):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    # 用TF-IDF矩阵训练分类器
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    # 用分类器进行预测，并得到预测精准度
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    # 得到混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    draw_confusion_matrix(cm, 'Passive Aggressive Classifier Confusion Matrix')
    print(f'===> Passive Aggressive Accuracy: {round(score * 100, 2)}%')


# 逻辑回归
def logic_regression_classify(X_train, y_train, X_test, y_test):
    # Vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model',
                      LogisticRegression())])  # Fitting the model
    model = pipe.fit(X_train, y_train)  # Accuracy
    prediction = model.predict(X_test)
    score = accuracy_score(y_test, prediction)
    # 得到混淆矩阵
    cm = confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL'])
    draw_confusion_matrix(cm, 'Logic Regression Classifier Confusion Matrix')
    print(f'===> Logic Regression Accuracy: {round(score * 100, 2)}%')


# 决策树
def decision_tree_classify(X_train, y_train, X_test, y_test):
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', DecisionTreeClassifier(criterion='entropy',
                                                      max_depth=20,
                                                      splitter='best',
                                                      random_state=42))])
    # Fitting the model
    model = pipe.fit(X_train, y_train)  # Accuracy
    prediction = model.predict(X_test)
    score = accuracy_score(y_test, prediction)
    # 得到混淆矩阵
    cm = confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL'])
    draw_confusion_matrix(cm, 'Decision Tree Classifier Confusion Matrix')
    print(f'===> Decision Tree Accuracy: {round(score * 100, 2)}%')


def random_forest_classify(X_train, y_train, X_test, y_test):
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model',
                      RandomForestClassifier(n_estimators=50,
                                             criterion="entropy"))])
    model = pipe.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = accuracy_score(y_test, prediction)
    # 得到混淆矩阵
    cm = confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL'])
    draw_confusion_matrix(cm, 'Random Forest Classifier Confusion Matrix')
    print(f'===> Random Forest Accuracy: {round(score * 100, 2)}%')


def main():
    # 数据读取
    df = pd.read_csv('Project_Data/news.csv')
    # 数据清洗
    df = data_clean(df)
    # 取label列
    labels = df.label
    # 清洗后的数据生成词云
    word_cloud(df)
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    # 原始数据转TF-IDF

    # RandomForestClassifier
    random_forest_classify(x_train, y_train, x_test, y_test)
    # PassiveAggressiveClassifier
    passive_aggressive_classify(x_train, y_train, x_test, y_test)
    # DecisionTreeClassifier
    decision_tree_classify(x_train, y_train, x_test, y_test)
    # LogicRegressionClassifier
    logic_regression_classify(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
