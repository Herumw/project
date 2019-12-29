#coding=utf-8
from math import log
import operator
import numpy as np
import pickle
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#计算香农熵 度量数据集无序程度
def calcShannonEnt(dataSet):
    m=len(dataSet)
    dict={}
    for i in range(m):
        feature=dataSet[i][-1]
        dict[feature]=dict.get(feature,0)+1
    sum=0
    for i in dict:
        p=1.0*dict[i]/m
        sum-=p*log(p,2)
    return sum

#划分数据集
def splitDataSet(dataSet, axis, value):#待划分的数据集 数据集特征 需要返回的特征值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
             reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
             reducedFeatVec.extend(featVec[axis + 1:])#extend方法是讲添加元素融入集合
             retDataSet.append(reducedFeatVec)#append将添加元素作为一个元素加入
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    bestfeature=-1
    bestShang=1000000
    m,n=np.array(dataSet).shape
    for feature in range(n-1):
        shang=0
        valueList=[]
        for i in range(m):
            valueList.append(dataSet[i][feature])
        valueSet=set(valueList)
        for value in valueSet:
            retDataSet=splitDataSet(dataSet,feature,value)
            size=np.array(retDataSet).shape[0]
            p=size*1.0/m
            shang+=p*calcShannonEnt(retDataSet)
        if shang<bestShang:
            bestfeature=feature
            bestShang=shang
    return bestfeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        classCount=classCount.get(vote,0)+1
    sortedClassCount=sorted(classCount.items(),key=lambda t:(t[1]))
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]#所有的类别都是一样的
    if len(dataSet[0])==1:#使用完了所有特征
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])#del用于list列表操作，删除一个或者连续几个元素
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def storeTree(inputTree,filename):
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr=open(filename,"rb")
    return pickle.load(fr)

#构造注解树 在python字典形式中如何存储树
def getNumLeafs(myTree):
    numLeafs=0 #初始化结点数
    firstSides = list(myTree.keys())#!!!python3中myTree.keys()得到的数据类型是dict_keys(['no surfacint']),需要用list()转换为列表形式
    firstStr = firstSides[0]  # 找到输入的第一个元素,第一个关键词为划分数据集类别的标签
    secondDict = myTree[firstStr]
    #firstStr = list(myTree)
    #secondDict=myTree[firstStr]
    for key in secondDict.keys(): #测试数据是否为字典形式
        if type(secondDict[key]).__name__=='dict': #!!!type判断子结点是否为字典类型
            numLeafs+=getNumLeafs(secondDict[key])
            #若子节点也为字典，则也是判断结点，需要递归获取num
        else:  numLeafs+=1
    return numLeafs #返回整棵树的结点数
def getTreeDepth(myTree):
    maxDepth=0
    # 下面三行为代码 python3 替换注释的两行代码
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    #firstStr=myTree.keys()[0]
    #secondDict=myTree[firstStr]#获取划分类别的标签
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:  #!!!这是另外一种判断方式，对应上面的100行
           thisDepth = 1 + getTreeDepth(secondDict[key])
    else:
        thisDepth = 1
    if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

if __name__=="__main__":
    dataSet,labels=createDataSet()
    #print(chooseBestFeatureToSplit(dataSet))
    myTree=createTree(dataSet,labels)
    #print(myTree)
    #storeTree(myTree, "myTree.txt")
    #print(getTreeDepth(myTree))
    print(getNumLeafs(myTree))