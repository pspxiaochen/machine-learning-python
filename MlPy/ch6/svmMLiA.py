#coding:utf-8
from numpy import *

def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

 #选择两个不同的alpha值，如果一个选为第alpha[i]，则另一个alpha值选择除了i随机的一个
 #i是第一个alpha的下标，m是所有alpha的数目
def selectJrand(i,m):
    j=i
    while(j==i):
        j= int(random.uniform(0,m))
    return j
#由于aj的取值范围的限制，H为上限，L为下限  用于调整大于H或者小于L的alpha值
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
#参数分别为数据集，样本标签 常数C 容错率 最大循环次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    b=0
    m,n=shape(dataMatrix)
    alphas=mat(zeros((m,1))) #初始化为0
    iter=0 #存储在没有任何alpha改变你的情况下遍历数据集的次数，当该变量达到输入值maxIter时，函数结束运行
    while(iter<maxIter):
        alphaPairsChanged=0 #用于记录alpha是否已经进行优化
        for i in range(m):
            #第I样本的预测类别
            #multiply（）对应元素相乘
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            # multiply是numpy的乘法函数，.T是转置
            # 第i个样本对应的判别结果
            # (dataMatrix*dataMatrix[i,:].T)是一个核函数计算
            Ei=fXi-float(labelMat[i]) #计算误差EI
            # 如果误差大于容错率或者alpha值不符合约束，则进入优化
            if ((labelMat[i]*Ei<-toler)and (alphas[i]<C))or((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                # 随机选择第二个alpha
                j=selectJrand(i,m)
                # 计算第二个alpha的值
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                # 得到两个样本对应的两个alpha对应的误差值
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                # 存储原本的alpha值
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print "L==G";continue
                # 计算上下阈值
                # 针对y1,y2的值相同与否，上下值也不同
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-\
                dataMatrix[j, :] * dataMatrix[j,:].T
                # 最优修改量
                if eta>=0:
                    print "eta>=0";continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print "j not moving enough";continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 更新两个b值
                if(0<alphas[i])and (C>alphas[i]):b=b2
                elif(0<alphas[j])and (C>alphas[j]):b=b2
                else:b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print "iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged)
            if (alphaPairsChanged==0):iter+=1
            else:iter=0
            print "iteration number:%d" %iter
        return b,alphas


class optStruct:
    def __init__(self,dataMatIn,classLabels,c,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))

def calcEk(oS,k): #计算E值
    fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei): #用于选择第二个alpha
    #在确定好第一个alpha的情况下，确定第二个
    #求找最大步长，E1-E2
    maxK=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei]
    #将Ei设置为有效
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    #nonzero返回一个列表，这个列表中包含以输入列表为目录的列标识
    #返回非零E值所对应的alpha值
    #因为在eCache的第一列代表是否有效，非0代表有效
    if(len(validEcacheList))>1:
         for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
         return maxK,Ej
    else:
         #如果都不满足要求，直接随机选一个
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]


