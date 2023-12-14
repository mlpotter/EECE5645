from MFspark import *
import numpy as np
from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.random import RandomRDDs
from MFspark import pred_diff,SE,normSqRDD
data_path = "small_data/fold0"

N = 20

R = readRatings(data_path,sc).repartition(N)

d = 5
seed = 123
V = generateItemProfiles(data,d,seed,sc,N)
U = generateUserProfiles(data,d,seed,sc,N)
lam = 0.1
mu = 0.1;
gamma = 0.1

ijru = R.map(lambda tuple: (tuple[0], (tuple[1], tuple[2]))).join(U,
                                                                  numPartitions=N)  # create tuples of (i,((j,rij),ui))

jiru = ijru.map(
    lambda tuple: (tuple[1][0][0], (tuple[0], tuple[1][0][1], tuple[1][1])))  # create tuples of (j,(i,rij,ui))

jiruv = jiru.join(V, numPartitions=N)  # create tuples of (j,((i,rij,ui),vj)))

joinedRDD = jiruv.map(lambda tuple: (tuple[1][0][0],tuple[0],pred_diff(tuple[1][0][1],tuple[1][0][2],tuple[1][1]),tuple[1][0][2],tuple[1][1])) # create tuples of (i,j,d,ui,vj)

#
# ijuv = U.cartesian(V,numPartitions=N).map(lambda tuple: ((tuple[0][0],tuple[1][0]),(tuple[0][1],tuple[1][1])))
# ijr = data.map(lambda tuple: ((tuple[0],tuple[1]),tuple[2]))
# ijruv = ijr.join(ijuv,numPartitions=N).map(lambda tuple: (tuple[0][0],tuple[0][1],pred_diff(tuple[1][0],tuple[1][1][0],tuple[1][1][1]),tuple[1][1][0],tuple[1][1][1]))

SE(ijduv)


normSqRDD(U, lam)

grad_u = joinedRDD.map(lambda tuple: (tuple[0], gradient_u(tuple[2], tuple[3], tuple[4]))).reduceByKey(
    lambda grad1, grad2: grad1 + grad2)

iui = joinedRDD.map(lambda tuple: (tuple[0], tuple[3])).reduceByKey(lambda ui, empty: ui)

iui.join(grad_u).mapValues(lambda tuple: tuple[0] - gamma * (tuple[1] + 2 * lam * tuple[0] @ tuple[0]))