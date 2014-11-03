#!/usr/bin/python

from numpy import *
import operator

def loadExData():
    return[[4,4,0,2,2],
           [4,0,0,3,3],
           [4,0,0,1,1],
           [1,1,1,2,0],
           [2,2,2,0,0],
           [1,1,1,0,0],
           [5,5,5,0,0]]


def ecludSim(inA,inB):
    return 1.0/(1.0+linalg.norm(inA-inB))

def pearsSim(inA,inB):
    if len(inA)<3:return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=linalg.norm(inA)*linalg.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0;ratSimTotal=0.0;
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:continue
        overLap=nonzero(logical_and(dataMat[:,item].A>0,\
                                    dataMat[:,j].A>0))[0]
        if len(overLap)==0:similarity=0
        else:similarity=simMeas(dataMat[overLap,item],\
                                 dataMat[overLap,j])
        print 'the %d and %d similarity is: %f' %(item,j,similarity)
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:return 0
    else: return ratSimTotal/simTotal

def svdEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0;ratSimTotal=0.0
    U,Sigma,VT=linalg.svd(dataMat)
    Sig4=mat(eye(4)*Sigma[:4])
    xformedItems=dataMat.T*U[:,:4]*Sig4.I
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:continue
        similarity=simMeas(xformedItems[item,:].T,\
                           xformedItems[j,:].T)
        print 'the %d and %d similarity is:%f'%(item,j,similarity)
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=svdEst):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:return 'you rated everything'
    itemScores=[]
    for item in unratedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,\
                    key=lambda jj:jj[1],reverse=True)[:N]



def test():
    aa=mat(loadExData());
    c=recommend(aa,2);
    return c

