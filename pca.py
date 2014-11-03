#!/usr/bin/python

from numpy import *
import operator

def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=999999):
    meanVals=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)
    eigVals,eigVects=linalg.eig(mat(covMat))
    eigValInd=argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat):-1]
    redBigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redBigVects
    reconMat=(lowDDataMat*redBigVects.T)+meanVals
    return lowDDataMat,reconMat
