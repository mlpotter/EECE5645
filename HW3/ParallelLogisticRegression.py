# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import argparse
import findspark
findspark.init()
import os
import shutil
from time import time
from SparseVector import SparseVector
from LogisticRegression import totalLoss,gradTotalLoss,getAllFeatures,basicMetrics,metrics
from operator import add
from pyspark import SparkContext


def readBetaRDD(input,spark_context):
    """ Read a vector β from file input. Each line contains pairs of the form:
                (feature,value)

        The return value is an RDD containing the above pairs.
    """
    return spark_context.textFile(input)\
                        .map(eval)

def writeBetaRDD(output,beta):
    """ Write a vector β to a file output.  Each line contains pairs of the form:
                (feature,value)
    """
    if os.path.exists(output):
        shutil.rmtree(output)
    beta.saveAsTextFile(output)

def readDataRDD(input_file,spark_context):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The result is stored in an RDD containing tuples of the form
                 (SparseVector(x),y)             
    """ 
    return spark_context.textFile(input_file)\
                        .map(eval)\
                        .map(lambda datapoint:(SparseVector(datapoint[0]),datapoint[1]))

def identityHash(num):
    """ Hash a number to itself 
    """
    return num

def groupDataRDD(dataRDD,N):
    """ Partition the data in dataRDD into N partitions and collect the data in
        each partition into a list. The rdd data should contain inputs of the type:
                (SparseVector(x),y)

        The result is an RDD containing tuples of the type

                (partitionID,dataList)
        
        where i is the index of the partition and dataList is a list of (SparseVector(x),y) values
        containing the data assigned to this partition

        Inputs are: 
            - dataRDD: The RDD containing the data
            - N: the number of partitions of the returned RDD

        The return value is the grouped RDD, partitioned using identityHash as a partition function.
    """
    return dataRDD.repartition(N)\
               .mapPartitionsWithIndex(  
                   lambda partitionID, elements: [(partitionID, [x for x in elements])])\
               .partitionBy(N,identityHash).cache()

def basicStatistics(groupedDataRDD):
    """ Return some basic statistics about the data in each partition in groupedDataRDD
    """
    num_datapoints=groupedDataRDD.values().map(lambda dataList: len(dataList))
    num_features=groupedDataRDD.values().map(lambda dataList: len(getAllFeatures(dataList)))
    
    datapoint_stats = (num_datapoints.min(),num_datapoints.max(),num_datapoints.mean())
    feature_stats = (num_features.min(),num_features.max(),num_features.mean())
    
    return datapoint_stats,feature_stats

def getAllFeaturesRDD(groupedDataRDD):                
    """ Get all the features present in grouped dataset groupedDataRDD.
 
        The input is:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values

        The return value is an RDD containing the above features.
    """                
    return groupedDataRDD.values()\
                         .flatMap(lambda dataList:getAllFeatures(dataList))\
                         .distinct()

def mapFeaturesToPartitionsRDD(groupedDataRDD,N):
    """ Given a groupedDataRDD, construct an RDD connecting the partitionID
        to all the features present in the data list of this partition. That is,
        given a groupedDataRDD containing pairs of the form

              (partitionID,dataList)
        
        return an RDD containing *all* pairs of the form

              (feat,partitionID)

        where feat is a feature label appearing in a datapoint inside dataList associated with partitionID.

        The inputs are:
            - groupedDataRDD:  RDD containing the grouped data
            - N: Number of partitions of the returned RDD
        
        The returned RDD is partitioned with the default hash function and cached.
    """

    return groupedDataRDD.flatMapValues(lambda data: getAllFeatures(data)) \
        .map(lambda x: (x[1],x[0])) \
        .partitionBy(N).cache()

def sendToPartitions(betaRDD,featuresToPartitionsRDD,N):
    """ Given a betaRDD and a featuresToPartitionsRDD, create an RDD that contains pairs of the form 
                   (partitionID, small_beta)
        
        where small_beta is a SparseVector containing only the features present in the partition partitionID. 
        
        The inputs are:
            - betaRDD: RDD storing β
            - featuresToPartitionsRDD:  RDD mapping features to partitions, generated by mapFeaturesToPartitionsRDD
            - N: Number of partitions of the returned RDD

        The returned RDD is partitioned with the identityHash function and cached.
    """
     # (feature, (value, partitionID))
    return betaRDD.join(featuresToPartitionsRDD) \
        .map(lambda x: (x[1][1], SparseVector({x[0]: x[1][0]}))) \
        .reduceByKey(add) \
        .partitionBy(N, identityHash).cache()

def totalLossRDD(groupedDataRDD,featuresToPartitionsRDD,betaRDD,N,lam = 0.0):
    """  Given a β represented by RDD betaRDD and a grouped dataset data represented by groupedDataRDD  compute 
         the regularized total logistic loss:

            L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2             
         
         Inputs are:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
            - featuresToPartitionsRDD: RDD mapping features to partitions, generated by mapFeaturesToPartitionsRDD
            - betaRDD: a vector β represented as an RDD of (feature,value) pairs
            - N: Number of partitions of RDDs
            - lam (optional): the regularization parameter λ (default: 0.0)

         The return value is the scalar L(β).
    """
    # (partitionID, (small_beta, dataList))
    loss = sendToPartitions(betaRDD, featuresToPartitionsRDD, N).join(groupedDataRDD) \
        .values() \
        .map(lambda x: totalLoss(x[1],x[0],lam=0.0)) \
        .sum()

    return loss + lam*betaRDD.values().map(lambda x: x**2).sum()

def gradTotalLossRDD(groupedDataRDD,featuresToPartitionsRDD,betaRDD,N,lam = 0.0):
    """  Given a β represented by RDD betaRDD and a grouped dataset data represented by groupedDataRDD  compute 
         the regularized total logistic loss :

            ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β                
        
         Inputs are:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
            - featuresToPartitionsRDD: an RDD mapping features to relevant partitionIDs, created by mapFeaturesToPartitionsRDD
            - betaRDD: a vector β represented as an RDD of (feature,value) pairs
            - lam: the regularization parameter λ

         The return value is an RDD storing ∇L(β) in key value pairs of the form:
               (feature,value)
    """
    # (partitionID, (small_beta, dataList))
    grad = sendToPartitions(betaRDD, featuresToPartitionsRDD, N).join(groupedDataRDD) \
        .values() \
        .flatMap(lambda x: gradTotalLoss(x[1],x[0],lam=0.0).items()) \
        .reduceByKey(add)

    if lam == 0.0:
        return grad
    else:
        return betaRDD.mapValues(lambda x: 2*x*lam) \
                .join(grad) \
                .mapValues(lambda x: np.sum(x))

def lineSearch(fun,xRDD,gradRDD,a=0.2,b=0.6):
    """ Given function fun, a current argument xRDD, and gradient gradRDD, 
        perform backtracking line search to find the next point to move to.
        (see Boyd and Vandenberghe, page 464).

        Both x and y are presumed to be RDDs containing key-value pairs of the form:
                 (feature,value)
 
        Parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient  ∇fun(x), the function finds a t such that
        fun(x - t * grad) <= fun(x) - a t <grad,grad>

        The return value is the resulting value of t.
    """
    t = 1.0
   
    fatx = fun(xRDD)
    gradSq = gradRDD.mapValues(lambda x:x*x).values().reduce(add)
    x_min_t_grad = xRDD.join(gradRDD).mapValues(lambda pair: pair[0]-t*pair[1]) 
    while fun(x_min_t_grad) > fatx - a * t * gradSq :
        t = b * t
        x_min_t_grad = xRDD.join(gradRDD).mapValues(lambda pair: pair[0]-t*pair[1] ) 
    return t 

def trainRDD(groupedDataRDD,featuresToPartitionsRDD,betaRDD_0,lam,max_iter,eps,N):
    """ Train a logistic model over a grouped dataset.
        
        Inputs are:
            - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
            - featuresToPartitionsRDD: an RDD mapping features to relevant partitionIDs, created by mapFeaturesToPartitionsRDD()
            - betaRDD_0: an initial vector β represented as an RDD of (feature,value) pairs
            - lam: the regularization parameter λ

            - max_iter: the maximum number of iterations
            - eps: the ε-tolerance
            - N: the number of partitions
    """
    k = 0
    gradNorm = 2*eps
    betaRDD = betaRDD_0
    data_total = pd.DataFrame()
    path = './results_norm_parallel' + '_lam_' + str(lam) + '_max_iter_' + str(max_iter) + '.csv'
    start = time()
    while k<max_iter and gradNorm > eps:
        
        gradRDD = gradTotalLossRDD(groupedDataRDD,featuresToPartitionsRDD,betaRDD,N,lam).cache()
        fun = lambda  xRDD: totalLossRDD(groupedDataRDD,featuresToPartitionsRDD,xRDD,N,lam)
        gamma = lineSearch(fun,betaRDD,gradRDD)
        betaRDD = betaRDD.join(gradRDD).mapValues(lambda pair: pair[0]-gamma*pair[1]).cache()

        obj = fun(betaRDD)
        gradSq = gradRDD.mapValues(lambda x:x*x).values().reduce(add)
        gradNorm = np.sqrt(gradSq)
        print('k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_{k-1})||_2 = ',gradNorm,'\tγ = ',gamma)

        data = pd.DataFrame({'Time': [time()-start], 'gradNorm': [gradNorm]})
        data_total = pd.concat((data_total,data))

        k = k + 1
    if os.path.isfile(path):
        os.remove(path)
    data_total.to_csv(path, mode='x', header=['Time','gradNorm'], index=False)

    return betaRDD,gradNorm,k         

def basicMetricsRDD(groupedDataRDD,featuresToPartitionsRDD,betaRDD,N):
    """ Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.
        
        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0
                 
                 TP = datapoints in (x,y) in P for which y=+1  
                 FP = datapoints in (x,y) in P for which y=-1  
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall of parameter vector β over data are defined as:
         
                 ACC(β,data) = ( #TP+#TN ) / (#P + #N)
                 PRE(β,data) = #TP / (#TP + #FP)
                 REC(β,data) = #TP/ (#TP + #FN)

        Inputs are:
             - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
              partitionID is an integer and dataList is a list of (SparseVector(x),y) values
             - featuresToPartitionsRDD: an RDD mapping features to relevant partitionIDs, created by mapFeaturesToPartitionsRDD()
             - betaRDD: a vector β represented as an RDD of (feature,value) pairs

        The return values are 
             - #P,#N,#TP,#FP,#TN,#FN
    """
    # (partitionID, (small_beta, dataList))
    return sendToPartitions(betaRDD, featuresToPartitionsRDD, N).join(groupedDataRDD) \
        .values() \
        .map(lambda x: basicMetrics(x[1], x[0])) \
        .reduce(lambda x,y: np.add(x,y))

def testRDD(groupedDataRDD,featuresToPartitionsRDD,betaRDD,N):
    """ Output the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.
        
        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0
                 
                 TP = datapoints in (x,y) in P for which y=+1  
                 FP = datapoints in (x,y) in P for which y=-1  
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall of parameter vector β over data are defined as:

                 ACC(β,data) = ( #TP+#TN ) / (#P + #N)
                 PRE(β,data) = #TP / (#TP + #FP)
                 REC(β,data) = #TP/ (#TP + #FN)

        Inputs are:
             - groupedDataRDD: a groupedRDD containing pairs of the form (partitionID,dataList), where 
               partitionID is an integer and dataList is a list of (SparseVector(x),y) values
             - featuresToPartitionsRDD: an RDD mapping features to relevant partitionIDs, created by mapFeaturesToPartitionsRDD()
             - betaRDD: a vector β represented as an RDD of (feature,value) pairs

        The return values are a tuple containing
             - ACC,PRE,REC 
    """
    P,N,TP,FP,TN,FN = basicMetricsRDD(groupedDataRDD,featuresToPartitionsRDD,betaRDD,N)
    return metrics(P,N,TP,FP,TN,FN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Sparse Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int,default=40, help='Maximum number of iterations')
    parser.add_argument('--N',type=int,default=20,help='Level of parallelism/number of partitions')
    parser.add_argument('--eps', type=float, default=0.1, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.') 

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true',help="Print Spark warning/info messages.")
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false',help="Suppress Spark warning/info messages.")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
  
    sc = SparkContext(appName='Parallel Sparse Logistic Regression')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")        

    if args.traindata is not None:
        print('Reading training data from',args.traindata)
        traindataRDD = readDataRDD(args.traindata,sc)
        groupedTrainDataRDD = groupDataRDD(traindataRDD,args.N) #identityhash
        trainFeaturesToPartitionsRDD = mapFeaturesToPartitionsRDD(groupedTrainDataRDD,args.N).cache() #defaulthash

        (dp_stats,f_stats) = basicStatistics(groupedTrainDataRDD)

        print('Read',traindataRDD.count(),'training data points')
        print('Created',args.N,'partitions with statistics:')
        print('Datapoints per partition: \tmin = %f \tmax = %f \tavg = %f ' % dp_stats)
        print('Features per partition: \tmin = %f \tmax = %f \tavg = %f ' % f_stats)

        betaRDD0 = getAllFeaturesRDD(groupedTrainDataRDD).map(lambda x:(x,0.0)).partitionBy(args.N).cache() #defaulthash

        print('Initial beta has',betaRDD0.count(),'features')

        print('Training on data from',args.traindata,'with λ =',args.lam,', ε = ',args.eps,', max iter = ',args.max_iter)
        beta, gradNorm, k = trainRDD(groupedTrainDataRDD,trainFeaturesToPartitionsRDD,betaRDD0,args.lam,args.max_iter,args.eps,args.N) 
        print('Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps)
        print('Saving trained β in',args.beta + '_parallel_lam_' + str(args.lam) + '_max_iter_' + str(args.max_iter))
        writeBetaRDD(args.beta + '_parallel_lam_' + str(args.lam) + '_max_iter_' + str(args.max_iter),beta)
    
    if args.testdata is not None:
        print('Reading test data from',args.testdata)
        testdataRDD = readDataRDD(args.testdata,sc)
        groupedTestDataRDD = groupDataRDD(testdataRDD,args.N).cache()
        testFeaturesToPartitionsRDD = mapFeaturesToPartitionsRDD(groupedTestDataRDD,args.N).cache()
        (dp_stats,f_stats) = basicStatistics(groupedTestDataRDD)

        print('Read',testdataRDD.count(),'test data points')
        print('Created',args.N,'partitions with statistics:')
        print('Datapoints per partition: \tmin = %f \tmax = %f \tavg = %f ' % dp_stats)
        print('Features per partition: \tmin = %f \tmax = %f \tavg = %f ' % f_stats)

        print('Reading β from', args.beta + '_parallel_lam_' + str(args.lam) + '_max_iter_' + str(args.max_iter))
        betaRDD = readBetaRDD(args.beta + '_parallel_lam_' + str(args.lam) + '_max_iter_' + str(args.max_iter),sc).partitionBy(args.N)
        print('Read beta with',betaRDD.count(),'features')
        print('Testing on data from',args.testdata)
        acc,pre,rec = testRDD(groupedTestDataRDD,testFeaturesToPartitionsRDD,betaRDD,args.N)
        print('\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec)
        if os.path.isfile('./results_acc_parallel.csv'):
            data = pd.DataFrame({'Max_iter': [args.max_iter], 'Lambda': [args.lam], 'ACC': [acc], 'PRE': [pre], 'REC': [rec]})
            data.to_csv('results_acc_parallel.csv', mode='a', header=False, index=False)
        else:
            columns = pd.DataFrame(columns = ['Max_iter','Lambda','ACC','PRE','REC'])
            result = pd.DataFrame({'Max_iter': [args.max_iter], 'Lambda': [args.lam], 'ACC': [acc], 'PRE': [pre], 'REC': [rec]})
            data = pd.concat((columns,result))
            data.to_csv('results_acc_parallel.csv', mode='x', header=['Max_iter','Lambda','ACC','PRE','REC'], index=False)

