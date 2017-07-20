#coding:utf-8
from numpy import*

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=map(float,curLine) #将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat
#参数分别为数据结合，待切分的特征和该特征的某个值
def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet): #负责生成叶节点
    return mean(dataSet[:,-1]) #得到最后一列的均值

def regErr(dataSet): #在给定数据上计算目标变量的平方误差
    return var(dataSet[:, -1]) * shape(dataSet)[0] #var()为均方差函数

#leafType:是对创建叶节点的函数的引用
#errType:是对前面介绍的总方差计算函数的引用
#ops:是一个用户定义的参数构成的元组

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)): #找到数据的最佳二元切分方式
    # 用于控制函数的停止时机
    tolS=ops[0]#容许的误差下降值
    tolN=ops[1]#切分的最少样本数
    # 用于控制函数的停止时机
    if len(set(dataSet[:,-1].T.tolist()[0]))==1: #统计不同剩余特征值的数目
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet) #计算误差
    bestS=inf
    bestIndex=0
    bestValue=0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN):continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if(S-bestS)<tolS:
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

#leafType:给出建立叶节点的函数
#errType:误差计算函数
#ops：包含树构建所需其他参数的元组
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj): #测试是否是一棵树（用于判断当前处理的节点是否是叶节点）
    return(type(obj).__name__=='dict')

def getMean(tree):#找到两个叶节点则计算他们的平均值
    if isTree(tree['right']):tree['right']=getMean(tree['right'])
    if isTree(tree['left']):tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
#参数为待剪枝的树与所需的测试数据
def prune(tree,testData): #回归树剪枝函数
    if shape(testData)[0]==0:#先确认测试集是否为空
        return getMean(tree)
    if(isTree(tree['right'])or isTree(tree['left'])):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    #如果两个分支已经不再是子树，合并它们
    if not isTree(tree['left'])and not isTree(tree['right']):
            lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
            errorNoMerge=sum(power(lSet[:,-1]-tree['left'],2))+\
                sum(power(rSet[:,-1]-tree['right'],2))
            treeMean=(tree['left'])+tree['right']/2.0
            errorMerge=sum(power(testData[:,-1]-treeMean,2))
            if errorMerge<errorNoMerge:
                print "merging"
                return treeMean
            else:return tree
    else:return tree

def linearSolve(dataSet):
    m,n=shape(dataSet)
    X=mat(ones((m,n)))
    Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('This matrix is singular,cannot do inverse,\n\
                        try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))


def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):return modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i],modelEval))
    return yHat



