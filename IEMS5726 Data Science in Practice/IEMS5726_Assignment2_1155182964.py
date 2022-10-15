# <1155182964>
import numpy as np
import pandas as pd
import torch
from nltk import RegexpTokenizer, PorterStemmer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
import nltk
import cv2 as cv
import soundfile as sf
import torchaudio


# Problem 2
def problem_2(filename, name):
    # write your logic here, df is a dataframe
    df = pd.read_csv(filename, index_col=0)
    # 计算哑变量
    list = [name]
    temp = pd.get_dummies(df, columns=list, prefix="", prefix_sep="")
    # 删除red列
    temp.drop('red', axis=1, inplace=True)
    # 与原始数据合并
    df = df.join(temp)
    return df


# Problem 3
def problem_3(filename, k):
    # write your logic here, pc is a numpy array
    # 取消科学计数法
    np.set_printoptions(suppress=True)
    df = pd.read_csv(filename)
    # PCA算法
    df = StandardScaler().fit_transform(df)
    pca = PCA(n_components=0.85, whiten=True)
    temp_pc = pca.fit_transform(df)
    # 精确到小数点后三位
    temp_pc = np.around(temp_pc, 3)
    # 取top k 个成分
    pc = temp_pc[:, :k]
    return pc


# Problem 4
def problem_4(sentence):
    # write your logic here
    # 去除标点后分词
    output = RegexpTokenizer('\w+').tokenize(sentence)
    stemmer = PorterStemmer()
    # 遍历获得词根
    stemSentence = []
    for token in output:
        stemSentence.append(stemmer.stem(token))
    # 生成二元组
    temp_bigrams = list(nltk.bigrams(stemSentence))
    # 拼接为新格式
    for bigram in temp_bigrams:
        stemSentence.append(bigram[0] + ' ' + bigram[1])
    output = stemSentence
    return output


# print(problem_4("Antoni Gaudí was a Spanish architect from Catalonia."))

# Problem 5
def problem_5(doc):
    # write your logic here, df is a dataframe, instead of number
    df = 0
    bigramsSet = set()
    for sentence in doc:
        for word in problem_4(sentence):
            bigramsSet.add(word)
    # 构建词汇表
    word_list = list(bigramsSet)
    word_list.sort()
    print(word_list)
    # 计算TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    print(vectorizer.get_feature_names)
    X = vectorizer.fit_transform(doc)
    print(X.toarray())
    # vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
    # X = vectorizer.fit_transform(doc)
    # print(X.toarray())
    # transform = TfidfTransformer()
    # Y = transform.fit_transform(X)  # 这里的输入是上面文档的计数矩阵
    # print(Y.toarray())  # 输出转换为tf-idf后的 Y 矩阵
    return df


# print(problem_5(["CUHK is located in Shatin", "CUHK has a large campus", "Shatin is a district in the New Territories"]))

# Problem 6
def problem_6(image_filename):
    # write your logic here, keypoint and descriptor are BRISK object
    image = cv.imread(filename=image_filename, flags=cv.IMREAD_GRAYSCALE)
    BRISK = cv.BRISK_create()
    keypoint, descriptor = BRISK.detectAndCompute(image, None)
    return keypoint, descriptor


# print(problem_6('assignment2_data/sample1.jpg'))

# Problem 7
def problem_7(image1_filename, image2_filename):
    # write your logic here, common_descriptor is the common desc.
    keypoint1, descriptors1 = problem_6(image1_filename)
    keypoint2, descriptors2 = problem_6(image2_filename)
    BFMatcher = cv.BFMatcher(normType=cv.NORM_HAMMING, crossCheck=True)
    # Matching descriptor vectors using Brute Force Matcher
    matches = BFMatcher.match(queryDescriptors=descriptors1,
                              trainDescriptors=descriptors2)
    # Sort them in the order of their distance
    common_descriptor = sorted(matches, key=lambda x: x.distance)
    return common_descriptor


# print(problem_7("assignment2_data/sample1.jpg", "assignment2_data/sample2.jpg"))

# Problem 8
def problem_8(audio_filename, sr, n_mels, n_fft):
    # write your logic here, spec is a tensor
    sig, old_sr = torchaudio.load(audio_filename)
    if old_sr != sr:
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(old_sr, sr)(sig[:1, :])
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(old_sr, sr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
    else:
        resig = sig
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=None, n_mels=n_mels)(resig)
    spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)
    return spec


print(problem_8("assignment2_data/StarWars60.wav", 1000, 64, 1024))


# Problem 9
def problem_9(spec, max_mask_pct, n_freq_masks, n_time_masks):
    # write your logic here
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec
    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    return aug_spec

# print(problem_9(problem_8("assignment2_data/StarWars60.wav", 1000, 64, 1024), 0.1, 2, 2))