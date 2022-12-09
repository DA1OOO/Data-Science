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


def draw_confusion_matrix(cm1, cm2, cm3, cm4):
    plt.close()
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(tight_layout=True)

    df_cm = pd.DataFrame(cm1)
    fig.add_subplot(gs[0, 0])
    ax = sn.heatmap(df_cm, annot=True, fmt='.20g')
    ax.set_title('Random Forest Confusion Matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴

    df_cm = pd.DataFrame(cm1)
    fig.add_subplot(gs[0, 1])
    ax = sn.heatmap(df_cm, annot=True, fmt='.20g')
    ax.set_title('Passive Aggressive Confusion Matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴

    df_cm = pd.DataFrame(cm1)
    fig.add_subplot(gs[1, 0])
    ax = sn.heatmap(df_cm, annot=True, fmt='.20g')
    ax.set_title('Decision Tree Confusion Matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴

    df_cm = pd.DataFrame(cm1)
    fig.add_subplot(gs[1, 1])
    ax = sn.heatmap(df_cm, annot=True, fmt='.20g')
    ax.set_title('Logic Regression Confusion Matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴

    plt.savefig('confusion_matrix.png')
    print('===> Confusion Matrix Picture file path: /confusion_matrix.png')


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
    print(f'===> Passive Aggressive Accuracy: {round(score * 100, 2)}%')
    # 得到混淆矩阵
    return confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']), round(score * 100, 2)


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

    print(f'===> Logic Regression Accuracy: {round(score * 100, 2)}%')
    # 得到混淆矩阵
    return confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL']), round(score * 100, 2)


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
    print(f'===> Decision Tree Accuracy: {round(score * 100, 2)}%')
    # 得到混淆矩阵
    return confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL']), round(score * 100, 2)


def random_forest_classify(X_train, y_train, X_test, y_test):
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model',
                      RandomForestClassifier(n_estimators=50,
                                             criterion="entropy"))])
    model = pipe.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = accuracy_score(y_test, prediction)
    print(f'===> Random Forest Accuracy: {round(score * 100, 2)}%')
    # 得到混淆矩阵
    return confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL']), round(score * 100, 2)


def draw_accuracy_compare(accuracy):
    plt.close()
    accuracy.plot(kind='barh', stacked=False, figsize=(10, 5))
    plt.title("Different Classifier Accuracy")
    plt.yticks([])
    plt.xlabel('%')
    plt.ylabel('Classifier')
    plt.grid()
    plt.savefig('accuracy_compare.png')


def main():
    # 数据读取
    df = pd.read_csv('Project_Data/news.csv')
    print(df['text'].head().to_string())
    # 数据清洗
    df = data_clean(df)
    print(df['text'].head().to_string())
    # 取label列
    labels = df.label
    # 清洗后的数据生成词云
    word_cloud(df)
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    # RandomForestClassifier
    cm1, accuracy1 = random_forest_classify(x_train, y_train, x_test, y_test)
    # PassiveAggressiveClassifier
    cm2, accuracy2 = passive_aggressive_classify(x_train, y_train, x_test, y_test)
    # DecisionTreeClassifier
    cm3, accuracy3 = decision_tree_classify(x_train, y_train, x_test, y_test)
    # LogicRegressionClassifier
    cm4, accuracy4 = logic_regression_classify(x_train, y_train, x_test, y_test)
    # 画出混淆矩阵
    draw_confusion_matrix(cm1, cm2, cm3, cm4)
    # 画出accuracy对比
    accuracy = pd.DataFrame(columns=['random forest', 'passive aggressive', 'decision tree', 'logic regression'])
    accuracy.loc[0] = [accuracy1, accuracy2, accuracy3, accuracy4]
    draw_accuracy_compare(accuracy)


if __name__ == '__main__':
    main()
