#coding:utf-8
from numpy import *
import boost
def loadSimpData():
    datMat=matrix([ [1. ,2.1],
                    [2. ,1.1],
                    [1.3,1.],
                    [1.,1.],
                    [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

