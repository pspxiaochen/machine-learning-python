import boost
import adaboost
from numpy import *
datMat,classLabels=boost.loadDataSet('horseColicTraining2.txt')
classifierArr,aggClassEst=boost.adaBoostTrainDS(datMat,classLabels,10)
boost.plotROC(aggClassEst.T,classLabels)
