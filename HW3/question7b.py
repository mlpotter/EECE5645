# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import shutil
from time import time
from SparseVector import SparseVector
from LogisticRegression import totalLoss, gradTotalLoss, getAllFeatures, basicMetrics, metrics
from ParallelLogisticRegression import *
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

    lambdas = [10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1,10,10**2,10**3,10**4,10**5]

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

    accuracies = []
    precisions = []
    recalls = []
    for lam in lambdas:
        args.lam = lam

        betaRDD0 = getAllFeaturesRDD(groupedTrainDataRDD).map(lambda x: (x, 0.0)).partitionBy(args.N).cache()

        print('Initial beta has', betaRDD0.count(), 'features')

        print('Training on data from', args.traindata, 'with λ =', args.lam, ', ε = ', args.eps, ', max iter = ',
              args.max_iter)
        beta, gradNorm, k = trainRDD(groupedTrainDataRDD, trainFeaturesToPartitionsRDD, betaRDD0, args.lam,
                                     args.max_iter, args.eps, args.N)
        print('Algorithm ran for', k, 'iterations. Converged:', gradNorm < args.eps)
        print('Saving trained β in', args.beta)
        writeBetaRDD(args.beta, beta)


        print('Reading β from', args.beta)
        betaRDD = readBetaRDD(args.beta, sc).partitionBy(args.N)
        print('Read beta with', betaRDD.count(), 'features')
        print('Testing on data from', args.testdata)
        acc, pre, rec = testRDD(groupedTestDataRDD, testFeaturesToPartitionsRDD, betaRDD, args.N)
        print('\tACC = ', acc, '\tPRE = ', pre, '\tREC = ', rec)

        accuracies.append(acc); precisions.append(pre); recalls.append(rec);

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].plot(lambdas,accuracies)
    axes[0].set_xlabel("$\lambda$")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xscale('log')

    axes[1].plot(lambdas,precisions)
    axes[1].set_xlabel("$\lambda$")
    axes[1].set_ylabel("Precision")
    axes[1].set_xscale('log')

    axes[2].plot(lambdas,recalls)
    axes[2].set_xlabel("$\lambda$")
    axes[2].set_ylabel("Recall")
    axes[2].set_xscale('log')

    plt.savefig("question7b.png")

