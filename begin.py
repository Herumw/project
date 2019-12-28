import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import jieba.posseg as pseg
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

datafile = "data/DMSC.csv"

# 停用词表路径
stop_words_path = 'stop_words/stop_words.txt'

# 加载数据
raw_data = pd.read_csv(datafile)

print('数据集有{}条记录。'.format(len(raw_data)))
print('数据集包含{}部电影。'.format(len(raw_data['Movie_Name_CN'].unique())))
print(raw_data['Movie_Name_CN'].unique())

movie_mean_score = raw_data.groupby('Movie_Name_CN')['Star'].mean().sort_values(ascending=False)
movie_mean_score.plot(kind='bar')
plt.tight_layout()
#plt.show()

cln_data = raw_data.dropna().copy()

# 建立新的一列，如果打分>=3.0，为正面评价1，否则为负面评价0
cln_data['Positively Rated'] = np.where(cln_data['Star'] >= 3, 1, 0)

# 数据预览
#print(cln_data.head())

stopwords = [line.rstrip() for line in open(stop_words_path, 'r')]
#print(stopwords)

import re
import jieba.posseg as pseg

def proc_text(raw_line):
    """
        处理文本数据
        返回分词结果
    """

    # 1. 使用正则表达式去除非中文字符
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_line)

    # 2. 结巴分词+词性标注
    word_list = pseg.cut(chinese_only)

    # 3. 去除停用词，保留有意义的词性
    # 动词，形容词，副词
    used_flags = ['v', 'a', 'ad']
    meaninful_words = []
    for word, flag in word_list:
        #
        if (word not in stopwords) and (flag in used_flags):
            meaninful_words.append(word)
    return ' '.join(meaninful_words)

# 测试一条记录
test_text = cln_data.loc[5, 'Comment']
# print('原文本：', test_text)
# print('\n\n处理后：', proc_text(test_text))


cln_data=cln_data[:100]
cln_data['Words']=cln_data['Comment'].apply(proc_text)
print(cln_data.head())

# cln_data['Words'] = cln_data['Comment'].apply(proc_text)
# print(cln_data.head())


saved_data = cln_data[['Words', 'Positively Rated']].copy()
saved_data.dropna(subset=['Words'], inplace=True)
saved_data.to_csv('data/douban_cln_data.csv', encoding='utf-8', index=False)



from sklearn.model_selection import train_test_split
print("3")
X_train_data, X_test_data, y_train, y_test = train_test_split(saved_data['Words'], saved_data['Positively Rated'], test_size=1/4, random_state=0)
print("---------------")
print(X_train_data)
#print(X_train_data)

# print('X__train_data 第一条记录：\n\n', X_train_data.iloc[1])
# print('\n\n训练集样本数: {}，测试集样本数：{}'.format(len(X_train_data), len(X_test_data)))
from sklearn.feature_extraction.text import TfidfVectorizer

# max_features指定语料库中频率最高的词
n_dim = 10000
vectorizer = TfidfVectorizer(max_features=n_dim)
X_train = vectorizer.fit_transform(X_train_data.values)
X_test = vectorizer.transform(X_test_data.values)
print("++++++++++++")
print(X_train[1].toarray)

print('特征维度：', len(vectorizer.get_feature_names()))
print('语料库中top{}的词：'.format(n_dim))
print(vectorizer.get_feature_names())

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=100)
lr_model.fit(X_train, y_train)

LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

from sklearn.metrics import roc_auc_score

predictions = lr_model.predict(X_test)
print('AUC: ', roc_auc_score(y_test, predictions))