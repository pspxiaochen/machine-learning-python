import regression
from numpy import *
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

abX,abY=regression.loadDataSet('abalone.txt')
yHat01=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

print regression.rssError(abY[0:99],yHat01.T)
print regression.rssError(abY[0:99],yHat1.T)
print regression.rssError(abY[0:99],yHat10.T)





