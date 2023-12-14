# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import shutil
from time import time
from SparseVector import SparseVector
from LogisticRegression import train,test,readData,writeBeta,readBeta,getAllFeatures
from ParallelLogisticRegression import trainRDD,testRDD,readDataRDD,groupDataRDD,writeBetaRDD,readBetaRDD,mapFeaturesToPartitionsRDD,basicStatistics,getAllFeaturesRDD
from operator import add
from pyspark import SparkContext
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Sparse Logistic Regression.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata', default="newsgroups/news.train",
                        help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata', default="newsgroups/news.test",
                        help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta',
                        help='File where beta is stored (when training) and read from (when testing)')

    parser.add_argument('--max_iter', type=int, default=40, help='Maximum number of iterations')
    parser.add_argument('--N', type=int, default=20, help='Level of parallelism/number of partitions')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true',
                                 help="Print Spark warning/info messages.")
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false',
                                 help="Suppress Spark warning/info messages.")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Sparse Logistic Regression')

    if not args.verbose:
        sc.setLogLevel("ERROR")

    # lambdas = [10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1,10,10**2,10**3,10**4,10**5]
    # lambdas = [10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1]
    lambdas = [0.1,0.01]

    print("PARALLEL LOGISTIC DATA")

    print('Reading training data from', args.traindata)
    traindataRDD = readDataRDD(args.traindata, sc)
    groupedTrainDataRDD = groupDataRDD(traindataRDD, args.N)
    trainFeaturesToPartitionsRDD = mapFeaturesToPartitionsRDD(groupedTrainDataRDD, args.N).cache()

    (dp_stats, f_stats) = basicStatistics(groupedTrainDataRDD)

    print('Read', traindataRDD.count(), 'training data points')
    print('Created', args.N, 'partitions with statistics:')
    print('Datapoints per partition: \tmin = %f \tmax = %f \tavg = %f ' % dp_stats)
    print('Features per partition: \tmin = %f \tmax = %f \tavg = %f ' % f_stats)

    print('Reading test data from', args.testdata)
    testdataRDD = readDataRDD(args.testdata, sc)
    groupedTestDataRDD = groupDataRDD(testdataRDD, args.N).cache()
    testFeaturesToPartitionsRDD = mapFeaturesToPartitionsRDD(groupedTestDataRDD, args.N).cache()
    (dp_stats, f_stats) = basicStatistics(groupedTestDataRDD)

    print('Read', testdataRDD.count(), 'test data points')
    print('Created', args.N, 'partitions with statistics:')
    print('Datapoints per partition: \tmin = %f \tmax = %f \tavg = %f ' % dp_stats)
    print('Features per partition: \tmin = %f \tmax = %f \tavg = %f ' % f_stats)

    print("LOGISTIC DATA")
    print('Reading training data from', args.traindata)
    traindata = readData(args.traindata)
    print('Read', len(traindata), 'data points with', len(getAllFeatures(traindata)), 'features in total.')

    print('Reading test data from', args.testdata)
    testdata = readData(args.testdata)
    print('Read', len(testdata), 'data points with', len(getAllFeatures(testdata)), 'features in total.')

    accuracies_pp = [];
    precisions_pp = [];
    recalls_pp = [];
    time_pp = [];
    norm_pp = [];

    accuracies = [];
    precisions = [];
    recalls = [];
    time_ = [];
    norm = [];

    from time import time
    for lam in lambdas:
        args.lam = lam

        print("PARALLEL LOGISTIC REGRESSION")
        betaRDD0 = getAllFeaturesRDD(groupedTrainDataRDD).map(lambda x: (x, 0.0)).partitionBy(args.N).cache()

        print('Initial beta has', betaRDD0.count(), 'features')

        print('Training on data from', args.traindata, 'with λ =', args.lam, ', ε = ', args.eps, ', max iter = ',
              args.max_iter)
        start_time = time()
        beta, gradNorm, k = trainRDD(groupedTrainDataRDD, trainFeaturesToPartitionsRDD, betaRDD0, args.lam,
                                     args.max_iter, args.eps, args.N)
        end_time = time()
        time_pp.append(end_time-start_time)
        norm_pp.append(gradNorm)
        print('Algorithm ran for', k, 'iterations. Converged:', gradNorm < args.eps)
        print('Saving trained β in', args.beta)
        writeBetaRDD(args.beta+"_pp", beta)


        print('Reading β from', args.beta+"_pp")
        betaRDD = readBetaRDD(args.beta+"_pp", sc).partitionBy(args.N)
        print('Read beta with', betaRDD.count(), 'features')
        print('Testing on data from', args.testdata)
        acc, pre, rec = testRDD(groupedTestDataRDD, testFeaturesToPartitionsRDD, betaRDD, args.N)
        print('\tACC = ', acc, '\tPRE = ', pre, '\tREC = ', rec)

        accuracies_pp.append(acc); precisions_pp.append(pre); recalls_pp.append(rec);

        print("NORMAL LOGISTIC REGRESSION")
        beta0 = SparseVector({})
        start_time = time()
        beta, gradNorm, k = train(traindata, beta_0=beta0, lam=args.lam, max_iter=args.max_iter, eps=args.eps)
        end_time = time()
        time_.append(end_time-start_time)
        norm.append(gradNorm)
        print('Algorithm ran for', k, 'iterations. Converged:', gradNorm < args.eps)
        print('Saving trained β in', args.beta)
        writeBeta(args.beta+"_normal"+f"lam={lam}", beta)

        acc, pre, rec = test(testdata, beta)
        print("Final test performance at trained β is:", 'ACC = ', acc, '\tPRE = ', pre, '\tREC = ', rec)
        accuracies.append(acc); precisions.append(pre); recalls.append(rec);
        print("="*25)
        print()


    metric_df = pd.DataFrame({"lambda":lambdas,
                              "acc_pp":accuracies_pp,
                              "precision_pp":precisions_pp,
                              "recall_pp":recalls_pp,
                                "time_pp": time_pp,
                                "norm_pp": norm_pp,
                            "acc": accuracies,
                            "precision": precisions,
                            "recall": recalls,
                            "time": time_,
                            "norm": norm
                              })

    metric_df.to_excel("question7a_useless_tome.xlsx",index=False)

