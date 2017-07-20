#coding:utf-8
from numpy import *

 # 输入:
 # dataMatrix:    输入数据
 # dimen:    划分特征下标
 # threshVal:    划分阈值
 # threshIneq:    划分方向(是左1右0分类还是左0右1分类)
# 输出:
  # retArray:    分类结果
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq =='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]= -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
 # 输入:
 #        dataArr:    输入数据
 #        classLabels:    分类标签集
 #        D:    权重向量
 # 输出:
 #        bestStump:    决策树信息
 #        minError:    带权错误(用于生成分类器权重值 alpha)
 #        bestClasEst:    分类结果

def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr) #把输入数据变成矩阵
    labelMat=mat(classLabels).T #把标签变成向量
    m,n=shape(dataMatrix) #m=5 ,n=2
    numSteps=10.0 # 特征值阈值步长
    bestStump={}  # 当前最佳决策树信息集 空字典
    bestClasEst=mat(zeros((m,1)))  # 分类结果
    minError=inf # 最小带权错误初始化为无穷大
    for i in range(n): # 遍历所有的特征选取最佳划分特征
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        # 遍历所有的特征值选取最佳划分特征值 stepSize为探测步长
        for j in range(-1,int(numSteps)+1): #+1才能包含numSteps的最大值 (-1到10)
            for inequal in ['lt','gt']: # 对于 左1右0 和 左0右1 两种分类方式
                # 当前划分阈值
                threshVal=(rangeMin+float(j)*stepSize)
                # 分类
                # dataMatrix:    输入数据
                # dimen:    划分特征下标
                # threshVal:    划分阈值
                # threshIneq:    划分方向(是左1右0分类还是左0右1分类)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 统计分类错误信息
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat] =0
                weightedError=D.T*errArr #计算加权错误率
               # print "split:dim %d,thresh %.2f,thresh ineqal: %s,the weighted error is %.3f" %(i,threshVal,inequal,weightedError)
                # 更新最佳决策树的信息
                if weightedError<minError:
                    minError = weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return  bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[] #单层决策树的数组
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m) #初始化所有样本的权值一样
    aggClassEst=mat(zeros((m,1)))#每个数据点的估计值 列向量
    for i in range(numIt):#迭代40次
        #        bestStump:    决策树信息
        #        error:    带权错误(用于生成分类器权重值 alpha)
        #        ClassEst:    分类结果
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        #print i
       # print "D:",D.T
        # 计算alpha，max(error,1e-16)保证没有错误的时候不出现除零溢出
        # alpha表示的是这个分类器的权重，错误率越低分类器权重越高
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print "classEst:",classEst.T
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)#更新D
        D=multiply(D,exp(expon))#更新D
        D=D/D.sum()  #更新D
        aggClassEst+=alpha*classEst #构建基本分类器的线性组合

        print "aggClassEst:" ,aggClassEst.T
        # 所有分类器的计算训练误差，如果这是0退出循环早（使用中断）
        #sign(aggClassEst) 为最终分类器
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        #print "total error:",errorRate,"\n"
        if errorRate==0.0:break
    return weakClassArr,aggClassEst
#datToClass:待分类的样例
#classifierArr：多个弱分类组成的数组
def adaClassify(datToClass,classifierArr):
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))#构建基本分类器的线性组合
    for i in range(len(classifierArr)): #遍历所有的弱分类器 得到每一种分类器的分类结果
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                               classifierArr[i]['thresh'],\
                               classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print classEst

    return sign(aggClassEst)

def loadDataSet(fileName):	  #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #自动检测特征的数目
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#第一个参数代表的是一个向量或者矩阵，代表的是分类器的预测强度
def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur=(1.0,1.0) #保留的是绘制光标的位置
    ySum=0.0 #用于计算AUC的值
    numPosClas=sum(array(classLabels)==1.0)  #计算正例的数目
    yStep=1/float(numPosClas)  #是真阳率
    xStep=1/float(len(classLabels)-numPosClas) #是假阳率
    sortedIndicies=predStrengths.argsort() #获得排序好的索引,由小到大
    print predStrengths
    print sortedIndicies
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0: #每得到一个标签为1的类，则要沿着y轴的方向下降一个补偿
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is:",ySum*xStep


