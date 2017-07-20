#coding:utf-8
from numpy import *


def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #将X0设置为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))
#梯度上升算法
#dataMatIn是一个2维数组，每一列分别代表每个不同的特征，每行代表一个训练样本
#dataMatIn是一个100*3的矩阵
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat=mat(classLabels).transpose() #标签向量转置为列矩阵
    m,n=shape(dataMatrix) #得到行和列
    alpha=0.001 #步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat -h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

def plotBestFit(weights): ###画出最佳拟合直线#######
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat) # 矩阵转化为数组
    n = shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01 #j是迭代次数，i是样本点的下标
            randIndex=int(random.uniform(0,len(dataIndex))) #通过随机选取样本来更新回归系数
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#inX: 特征向量 weights:回归系数
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:return 1.0
    else:return 0.0
#用来打测试集和训练集
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500) #训练集的回归系数
    errorCount=0 #错误率
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0 #测试集的样本个数
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int (classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print errorRate
    return errorRate
#求十次平均
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print (numTests,errorSum/float(numTests))


