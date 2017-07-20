# -*- coding: utf-8 -*-
from math import log
import operator
def calcShannonEnt(dataSet):  #输入训练数据集
    numEntries = len(dataSet) #计算训练数据集中样例的数量
    labelCounts={} #创建一个字典，它的键值是最后一列的数值
    for featVec in dataSet:
        currentLabel=featVec[-1] #获得数据集的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1 #当前标签实例数+1
    shannonEnt=0.0
  #  print labelCounts #{'yes': 2, 'no': 3}
  #  print numEntries
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt -=prob*log(prob,2) #计算信息熵
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

#data ：待划分的数据集 axis:划分数据集的特征 value:需要返回的特征的值
def splitDataSet(dataSet,axis,value): #划分属性，获得去掉axis位置的属性value剩下的样本
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
           # print featVec[0:]
            reducedFeatVec = featVec[:axis] #去除每一行第axis之后的字符
            reducedFeatVec.extend(featVec[axis+1:])#extend（）方法接受一个列表作为参数，并将该参数的每个元素都添加到原有的列表中 去掉axis+1之前
            retDataSet.append(reducedFeatVec) #append()方法向列表的尾部添加一个新的元素，只接受一个参数
    return retDataSet

def chooseBestFeatureToSplit(dataSet): #选择最好的特征
    numFeatures = len(dataSet[0]) - 1 #判定当前数据集包含多少特征属性
    baseEntropy=calcShannonEnt(dataSet)#计算了真个数据集的原始香农熵
    bestInfoGain=0.0;bestFeature= -1
    for i in range(numFeatures): #遍历数据集中所有的特征
        featList=[example[i] for example in dataSet] #取每一列的所有特征值 被称为列表表达式
        uniqueVals = set(featList) #将特征值放到一个集合中，消除重复的特征值 set:集合 可以消除重复元素
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            #print subDataSet
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy - newEntropy  #计算信息增益
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList): #计算最大所属类别
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels): #构建分类树
    classList = [example[-1] for example in dataSet] #获得类别列
    if classList.count(classList[0])==len(classList):#如果所有样本属于同一类别
        return classList[0]
    if len(dataSet[0])==1:#如果只有类别列，没有属性列
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #获得最优属性下标
    bestFeatLabel = labels[bestFeat]#获得最优属性
    myTree ={bestFeatLabel:{}}
    del(labels[bestFeat])#删除最优属性
    #print [example[bestFeat] for example in dataSet]
    featValues = [example[bestFeat]for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] #复制了类标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels) #递归计算分类树
    labels.insert(bestFeat, bestFeatLabel)
    return myTree
#inputTree：是输入的决策树对象
#featLabels：是我们要预测的特征值得label
#testVec:是要预测的特征值向量
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0] #存储决策树第一个节点
    secondDict = inputTree[firstStr]#将第一个节点的值存到字典中
    featIndex=featLabels.index(firstStr)#将标签字符串转换为索引  建立索引，知道对应到第几种特征值
    print featIndex
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            classLabel = classify(secondDict[key],featLabels,testVec)
        else:classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree,filename):  #保存决策树模型（字典的保存） inputTree:已学习的决策树模型，字典格式
    import pickle
    fw =open(filename,'w')  #filename ：存储文件名
    pickle.dump(inputTree,fw)
    fw.close()

def gradTree(filename):  #读取保存的决策树（字典的读取）
    import pickle
    fr = open(filename)
    return pickle.load(fr)