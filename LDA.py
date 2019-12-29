##知识处理讨论班 Work1-3
## 线性判别分析
## @author: 邢琛聪 51184506047
## github: @DeepTrial

import numpy as np
import re,json,os
from tqdm import tqdm
import pandas as pd
import scipy as sp
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#######################################################
## A.数据预处理
#######################################################


### 处理res数据
with open("./Data/Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res",'r') as fileRead:
    lines=fileRead.readlines()              #读取res数据

yActual=[]   #相关度值
queryID=[]   #对应的QueryID
queryNum=[]  #同一QueryID下的query数量

for line in tqdm(lines,desc="处理res:"):    #依照格式解析res数据
    dataLine=line.split(' ')
    if int(dataLine[0])>210:                #只处理前10个query
        break
    queryID.append(dataLine[0])
    queryNum.append(dataLine[3])
    yActual.append(dataLine[4])

    
## 处理documents数据
with open("./Data/documents.txt",'r',encoding="UTF-8") as fileRead:
    lines=fileRead.readlines()             #加载数据(逐行方式，可能会截断回答)
    
count=0                                    #控制读取回答数量(要求与提取相关度数量一致)
prevLine=''                                #处理同一回答但包含多行的情况
Answer=[]                                  #保存提取的回答
for line in tqdm(lines,desc="处理doc:"):
    temp=re.split('[<>]',prevLine+line)    #使用正则方式，以<>为分隔符，分割字符串
    while '' in temp:                      #去除空字符串
        temp.remove('')
    if "/article" not in temp:             #如果为检测到回答的结束标志 则说明此回答有多行
        prevLine=prevLine+line             #将分为多行的回答合并
        continue
    else:
        prevLine=''
        reply=''                           #query回答的暂存变量
        try:                                                                #提取回答的逻辑：
            if temp.index('/title')-temp.index('title')>=2:                 #保存<title></title>与<body></body>中最长的一个内容
                reply=temp[temp.index('title')+1]
            if temp.index('/body')-temp.index('body')>=2:
                if len(reply)<len(temp[temp.index('body')+1]):
                    reply=temp[temp.index('body')+1]
        except:                                                             #如果<title></title>与<body></body>不成对出现
            if "/body" in temp and temp.index('/body')>0:                   #则匹配任意一个标签，并保存最长的回答
                reply=temp[temp.index('/body')-1]
            if "body" in temp and temp.index('body')+1<len(temp):
                if len(reply)<len(temp[temp.index('body')+1]):
                    reply=temp[temp.index('body')+1]
            if "/title" in temp and temp.index('/title')>0:
                if len(reply)<len(temp[temp.index('/title')-1]):
                    reply=temp[temp.index('/title')-1]
            if "title" in temp and temp.index('title')+1<len(temp):
                if len(reply)<len(temp[temp.index('title')+1]):
                    reply=temp[temp.index('title')+1]
        if reply!='':                                                      #当回答不为空时则放入队伍，否则存 空标志
            Answer.append((reply.replace('\n',' ')).replace('\r',' '))
        else:
            Answer.append(np.nan)
        
        if count>=len(yActual):                 #当保存足够的回答(10*10000)时，终止
            break
        else:
            count=count+1

## 保存格式化数据
for i in tqdm(range(10),desc="保存数据:"):                  #暂存数据，避免多次重新计算
    data={'queryID':queryID[i*10000:(i+1)*10000],
          'queryNum':queryNum[i*10000:(i+1)*10000],
          'score':yActual[i*10000:(i+1)*10000],
          'content':Answer[i*10000:(i+1)*10000]}
    dataFrame=pd.DataFrame(data,columns=['queryID','queryNum','score','content'])
    dataFrame.to_csv("./Data/corpus/%d.csv"%(201+i),index=0)


#######################################################
## B.清洗文本数据，训练词向量
#######################################################


#启动点，读取上一步暂存的格式化数据
for corpusID in  tqdm(range(10),desc="分批处理"):
    ansData=pd.read_csv("./Data/corpus/%d.csv"%(201+corpusID))
    ansData = ansData.fillna(method="pad")
    totalSentence=ansData["content"].shape[0]

    #创建路径
    if not os.path.exists("./Data/model/%d/"%(201+corpusID)):
        os.makedirs(("./Data/model/%d/"%(201+corpusID)))

#根据文本内容清洗数据
    sentence=[]                 #保存清洗后的数据
    porter = PorterStemmer()    #波特词干提取器 避免时态，单复数影响
    for i in tqdm(range(totalSentence),desc="清洗%d文本"%(201+corpusID)):
        sentenceTemp=word_tokenize(ansData['content'][i])        #根据空格划分单词

        table = str.maketrans('', '', string.punctuation)        #去除标点符号
        sentenceTemp =[w.translate(table) for w in sentenceTemp]

        sentenceTemp = [word.lower() for word in sentenceTemp if word.isalpha()]   #去除 非纯字符单词

        stop_words = set(stopwords.words('english'))             #去除常见停用词 如a an of the before after等
        sentenceTemp = [porter.stem(w) for w in sentenceTemp if not w in stop_words and len(w)<20]  #去除长度超过20的字符，并提取词干
        if len(sentenceTemp)>=1:
            sentence.append(sentenceTemp)

#暂存清洗后的数据
    jsonfile=json.dumps(sentence)
    with open("./Data/model/%d/processtrain.json"%(201+corpusID),"w",encoding='UTF-8') as writeTrain:
        writeTrain.write(jsonfile)


#训练Word2Vec
vecDim=700               #词向量维度数
for corpusID in  tqdm(range(10),desc="分批处理"):

    model=Word2Vec(sentence,size=vecDim,min_count=1)                      #训练词向量
    model.save("./Data/model/%d/w2v_final.model"%(corpusID+201))

    postiveX = []
    negtiveX = []
    for i in tqdm(range(totalSentence),desc="转化%d"%(201+corpusID)):     #根据清洗的规则，生成词向量，最终得到句向量
        
        sentenceTemp=word_tokenize(ansData['content'][i])
        table = str.maketrans('', '', string.punctuation)
        sentenceTemp =[w.translate(table) for w in sentenceTemp]
        sentenceTemp = [word.lower() for word in sentenceTemp if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        sentenceTemp = [porter.stem(w) for w in sentenceTemp if not w in stop_words and len(w)<20]
        
        vecTemp=np.zeros((1,vecDim))
        for word in sentenceTemp:
            try:
                temp = np.array(model[word])
                temp = np.reshape(temp, (1, vecDim))
                vecTemp = vecTemp + temp
            except:
                pass
        if (i > 9700):
            if np.max(vecTemp) > 0:
                negtiveX.append(vecTemp)

        if (i <= 300):
            if np.max(vecTemp) > 0:
                postiveX.append(vecTemp)

        postiveX = np.reshape(np.asarray(postiveX), (len(postiveX), vecDim))
        negtiveX = np.reshape(np.asarray(negtiveX), (len(negtiveX), vecDim))

        np.save("./Data/model/%d/finalDataSetPostive.npy" % (201 + corpusID), postiveX)
        np.save("./Data/model/%d/finalDataSetNegtive.npy" % (201 + corpusID), negtiveX)

#######################################################
#C.LDA算法
#######################################################
def within_class(dataset):
    matDim = dataset.shape[1]
    conv = np.mat(np.zeros((matDim, matDim)))
    mean = np.mat(np.mean(dataset, axis=0))
    for dataline in dataset:
        temp = dataline - mean
        conv = conv + temp.T * temp
    return conv


def lda(c1, c2):
    # c1 第一类样本，每行是一个样本
    # c2 第二类样本，每行是一个样本
    sigmaC1 = within_class(c1)
    sigmaC2 = within_class(c2)
    # 计算各类样本的均值和所有样本均值
    u1 = np.mat(np.mean(c1, axis=0))  # 第一类样本均值
    u2 = np.mat(np.mean(c2, axis=0))  # 第二类样本均值

    Sw = sigmaC1 + sigmaC2            # 类内散度
    W = Sw.I * (u1 - u2).T
    return W,u1,u2


for corpusID in tqdm(range(10)):
    postiveX = np.load("./Data/model/%d/finalDataSetPostive.npy" % (201 + corpusID))
    negtiveX = np.load("./Data/model/%d/finalDataSetNegtive.npy" % (201 + corpusID))

    posTrain = postiveX[100:300]
    posValid = postiveX[:100]

    negTrain = negtiveX[:200]
    negValid = negtiveX[200:]

    w,u1,u2 = lda(posTrain, negTrain)
    print("---------Query: %d---------" % (201 + corpusID))
    print("100条正样本分类正确数量:", np.sum((posValid-0.5*(u1+u2)) * w > 0))
    print("100条负样本分类正确数量:", np.sum((negValid-0.5*(u1+u2)) * w < 0))

