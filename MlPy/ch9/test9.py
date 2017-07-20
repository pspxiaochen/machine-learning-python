import regTrees
from numpy import*
myDat=regTrees.loadDataSet('ex00.txt')
myMat=mat(myDat)
print regTrees.createTree(myMat)
ops=(1,4)
print (ops)