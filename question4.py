from LogisticRegression import gradTotalLoss, totalLoss,logisticLoss,getAllFeatures,basicMetrics
from helpers import estimateGrad
from SparseVector import SparseVector

from ParallelLogisticRegression import readDataRDD,groupDataRDD,mapFeaturesToPartitionsRDD,getAllFeaturesRDD,identityHash,sendToPartitions


from SparseVector import SparseVector as SV
import numpy as np

N = 100
lam = 1.0
mushroom_rdd = readDataRDD(r"newsgroups/news.train", sc)

groupedDataRDD = groupDataRDD(mushroom_rdd, 100)

groupedDataRDD.flatMapValues(lambda datalist: list(getAllFeatures(datalist))).map(lambda x,y: (y,x))

# trainFeaturesToPartitionsRDD =  grouped_RDD.flatMapValues(lambda datalist: list(getAllFeatures(datalist))) \
#     .map(lambda tup: (tup[1], tup[0])) \
#     .distinct().partitionBy(N).cache()

featuresToPartitionsRDD = mapFeaturesToPartitionsRDD(groupedDataRDD, 20).cache()

betaRDD = getAllFeaturesRDD(groupedDataRDD).map(lambda x: (x, 0.0)).partitionBy(20).cache()

featuresToPartitionsRDD.groupByKey(numPartitions=20).mapValues(list).take(1)
featuresToPartitionsRDD.join(betaRDD).map(lambda pair: (pair[1][0], [pair[0],pair[1][1]])).reduceByKey(lambda x,y: x+y).values().map(SparseVector)

joined = betaRDD.join(featuresToPartitionsRDD).map(lambda pair: (pair[1][1], SV({pair[0]:pair[1][0]})))
joined.reduceByKey(lambda x,y: x+y).partitionBy(20,identityHash).cache()

g = sendToPartitions(betaRDD,featuresToPartitionsRDD,20)

beta_small = sendToPartitions(betaRDD, featuresToPartitionsRDD, N)
groupedDataRDD.join(beta_small).values().flatMap(lambda tupl: totalLoss(tupl[0],tupl[1],lam=0)).reduce(lambda x,y: x+y)

groupedDataRDD.join(beta_small).values().map(lambda tupl: totalLoss(tupl[0],tupl[1])).\
    sum()  

beta_sv = SV(dict(betaRDD.collect()))

data_loss = groupedDataRDD.join(beta_small).values().map(lambda tupl: totalLoss(tupl[0], tupl[1])).reduce(lambda x, y: x + y)
data_grad = groupedDataRDD.join(beta_small).values().flatMap(lambda tupl: list(gradTotalLoss(tupl[0], tupl[1], lam=lam / N).items())).reduceByKey(lambda x, y: x + y)

beta_small = sendToPartitions(betaRDD, featuresToPartitionsRDD, N)
data_grad = groupedDataRDD.join(beta_small).values().flatMap(lambda tupl: list(gradTotalLoss(tupl[0], tupl[1], lam=0.0).items())).reduceByKey(lambda x, y: x + y)

metrics_ = groupedDataRDD.join(beta_small).values().map(lambda tupl: basicMetrics(tupl[0], tupl[1]))
