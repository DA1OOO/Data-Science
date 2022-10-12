# <1155182964>
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Problem 2
def problem_2(filename,name):
    # write your logic here, df is a dataframe
    df = pd.read_csv(filename, index_col=0)
    # 计算哑变量
    temp = pd.get_dummies(df, prefix="", prefix_sep="")
    # 删除red列
    temp.drop('red', axis=1, inplace=True)
    # 与原始数据合并
    df = df.join(temp)
    return df
# Problem 3
def problem_3(filename,k):
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
    pc = temp_pc[:,:k]
    return pc

print(problem_3("assignment2_data/num.csv",6))

# Problem 4
def problem_4(sentence):
    # write your logic here
    output = []

    return output

# Problem 5
def problem_5(doc):
    # write your logic here, df is a dataframe, instead of number
    df = 0

    return df

# Problem 6
def problem_6(image_filename):
    # write your logic here, keypoint and descriptor are BRISK object
    keypoint = 0
    descriptor = 0

    return keypoint, descriptor

# Problem 7
def problem_7(image1_filename, image2_filename):
    # write your logic here, common_descriptor is the common desc.
    common_descriptor = 0

    return common_descriptor

# Problem 8
def problem_8(audio_filename, sr, n_mels, n_fft):
    # write your logic here, spec is a tensor
    spec = 0
 
    return spec

# Problem 9
def problem_9(spec, max_mask_pct, n_freq_masks, n_time_masks):
    # write your logic here
    aug_spec = spec
 
    return aug_spec


