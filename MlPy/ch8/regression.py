#coding:utf-8
from numpy import *
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr): #求回归系数
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0: #判断行列式是否为0
        print "This matrix is singular,cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

#testPoint 是待测点
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m))) #创建对角权重矩阵
    for j in range(m):
        diffMat=testPoint-xMat[j,:]#计算样本点与预测值的距离
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2)) #计算每个样本点对应的权重值
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2): #用于计算回归系数
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = denom.I*(xMat.T*yMat)
    return ws

def redgeTest(xArr,yArr): #用于在一组lam上测试结果
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0) #求平均值
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0) #求方差
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

 #esp:每次迭代需要调整的步长
def stageWise(xArr,yArr,eps=0.01,numIt=100): #前向逐步线性回归
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat) #按照均值为0，方差为1做标准化处理
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1)) #用来保存#的值
    wsTest=ws.copy()
    wsMax=ws.copy() #建立了w的2份副本
    for i in range(numIt): #迭代100次
        print ws.T
        lowestError=inf #现将
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMat=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat





