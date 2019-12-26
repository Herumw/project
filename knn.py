#coding=utf-8
import numpy as np
import operator

def classify(inX,dataSet,labels,k):
    m,n=dataSet.shape
    diffMat=np.tile(inX,(m,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndecies=distances.argsort()#计算inX与所有点的距离

    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndecies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=lambda student:student[1],reverse=True) #字典排序后变成了列表的形式，列表里面是元组
    return sortedClassCount[0][0]



def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def file2matrix(filename):
    fr=open(filename)
    lines=fr.readlines()
    #dataMatrix=np.zeros((len,3))
    dataMatrix=[]
    labelMatrix=[]
    index=0
    for line in lines:
        line=line.strip()
        array=line.split('\t')
        dataMatrix.append(array[:3])  #这里读进来的数据都是字符串类型numpy_string
        labelMatrix.append(int(array[3]))

    dataMatrix=np.array(dataMatrix)
    dataMatrix=dataMatrix.astype(np.float64) #注意这里的array.astype是一个copy，并不是视图，所以这里一定要dataMatrix=dataMatrix.astype...
    labelMatrix=np.array(labelMatrix)
    return dataMatrix,labelMatrix


def autoNorm(dataSet):
    minVals=dataSet.min(0) #压缩行,所以是
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(dataSet.shape)
    m,n=normDataSet.shape
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?")) #这里输入所要查询人的三个特征
    # 书中raw_input在python3中修改为input（）
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    dataMatrix,labelMatrix=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(dataMatrix)
    inArr=np.array([ffMiles,percentTats,iceCream])
    classifierResult=classify((inArr-minVals)/ranges,normMat,labelMatrix,3)
    print("You will probably like this person:",resultList[classifierResult-1])



if __name__=="__main__":
    classifyPerson()