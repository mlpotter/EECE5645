# -*- coding: utf-8 -*-
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
from operator import add
from time import time

import findspark
findspark.init()

from pyspark import SparkContext

from ParallelRegression import train_FW,train_GD,readData,writeBeta,test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Regression.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata', default="data/medium.train",
                        help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata', default="data/medium.test",
                        help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta',
                        help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')
    parser.add_argument('--N', type=int, default=25, help='Level of parallelism')
    parser.add_argument('--solver', default='GD', choices=['GD', 'FW'],
                        help='GD learns β  via gradient descent, FW learns β using the Frank Wolfe algorithm')

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Regression')
    sc.setLogLevel('warn')

    beta = None

    # Train a linear model β from data, and store it in beta
    print('Reading training data from', args.traindata)
    data = readData(args.traindata, sc)
    data = data.repartition(args.N).cache()
    print("# Records: ",data.count())

    x, y = data.take(1)[0]
    dim = len(x)

    # plt.figure()
    # plt.hist(data.map(lambda data_tuple: data_tuple[1]).collect())
    # plt.savefig("question8.png")
    # plt.show()

    # Read beta from args.beta, and evaluate its MSE over data
    print('Reading test data from', args.testdata)
    test_data = readData(args.testdata, sc)
    test_data = test_data.repartition(args.N).cache()

    print("# Records: ",test_data.count())




    if args.solver == 'GD':
        lambdas = np.array([0.125,0.25,0.5,1.0,2.0,4.0,8.0])
        MSEs = np.zeros_like(lambdas)
        for i,lambdai in enumerate(lambdas):
            start = time()
            print('Gradient descent training on data from', args.traindata, 'with λ =', lambdai, ', ε =', args.eps,
                  ', max iter = ', args.max_iter)
            beta0 = np.zeros(dim)
            beta, gradNorm, k = train_GD(data, beta_0=beta0, lam=lambdai, max_iter=args.max_iter, eps=args.eps)
            print('Algorithm ran for', k, 'iterations. Converged:', gradNorm < args.eps, 'Training time:',
                  time() - start)
            print('Saving trained β in', args.beta)
            writeBeta(args.beta+f"lambda={lambdai}", beta)

            MSE = test(test_data, beta)
            print('MSE is:', MSE)
            MSEs[i] = MSE


        plt.figure()
        plt.plot(lambdas,MSEs,'bo-')
        for lambdai,msei in zip(lambdas,MSEs):
            plt.text(lambdai,msei+3,"$\lambda={:.4f}$".format(lambdai),rotation=90,horizontalalignment="center",fontsize="x-small")
        plt.xlabel("$\lambda$"); plt.ylabel("MSE"); plt.title(args.testdata + " Learning Curve")
        plt.savefig("MSE_GD")
        plt.show()


    else:
        Ks = np.array([1.0,5.0,10.0,20.0,30.0,40.0,50.0])
        # Ks = np.array([50.0])
        MSEs = np.zeros_like(Ks)

        for i,Ki in enumerate(Ks):

            start = time()
            print('Frank-Wolfe training on data from', args.traindata, 'with K =', Ki, ', ε =', args.eps,
                  ', max iter = ', args.max_iter)
            beta0 = np.zeros(dim)
            beta, criterion, k = train_FW(data, beta_0=beta0, K=Ki, max_iter=args.max_iter, eps=args.eps)
            print('Algorithm ran for', k, 'iterations. Converged:', criterion < args.eps, 'Training time:',
                  time() - start)
            print('Saving trained β in', args.beta)

            writeBeta(args.beta+f"K={Ki}", beta)

            MSE = test(test_data, beta)
            print('MSE is:', MSE)
            MSEs[i] = MSE

        plt.figure()
        plt.plot(Ks,MSEs,'bo-')
        for Ki,msei in zip(Ks,MSEs):
            plt.text(Ki,msei+3,"K={:.4f}".format(Ki),rotation=90,horizontalalignment="center",fontsize="x-small")
        plt.xlabel("K"); plt.ylabel("MSE"); plt.title(args.testdata + " Learning Curve")
        plt.savefig("MSE_FW")
        plt.show()



