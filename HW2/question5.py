# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from operator import add
from time import time
import findspark
findspark.init()
import ParallelRegression as PR
from pyspark import SparkContext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Regression.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--samples' ,type=int ,default=10 ,help='Number of random x,y, beta values')
    parser.add_argument('--delta' ,type=float ,default=1e-8 ,help='Numerical Derivative delta size')
    parser.add_argument('--N', type=int, default=25, help='Level of parallelism')
    parser.add_argument('--testdata',default="data/small.test", help='Input file containing (x,y) pairs, used to test a linear model')

    args = parser.parse_args()

    sc = SparkContext(f"local[{args.N}]",'Parallel Regression')
    sc.setLogLevel('warn')

    args = parser.parse_args()

    np.random.seed(42)

    dataset =  PR.readData(args.testdata,sc)
    data_tuple = dataset.take(1)
    print(data_tuple)
    dim = len(data_tuple[0][0])
    print(dim)
    lam = 0.1
    for lam in [0,0.1,0.5,5]:
        print("Lam={:.3f}".format(lam))
        for i in range(args.samples):
            beta = np.random.randn(dim,).astype(np.double)

            partial_f = lambda beta: PR.F(data=dataset,beta=beta,lam=lam) #partial(PR.f,x=x,y=y)

            # PR.gradient(data, beta, lam=0)

            print("Test i={:4d}".format(i))
            print("beta={}".format(np.round(beta,5)))
            print("Numerical Gradient: ", PR.estimateGrad(partial_f,beta,args.delta))
            print("Analytical Gradient: ", PR.gradient(dataset, beta, lam=lam))
            norm = np.linalg.norm(PR.gradient(dataset, beta, lam=lam) - PR.estimateGrad(partial_f,beta,args.delta),ord=np.inf)
            print('Norm={:.4f}'.format(norm))
            assert norm < 1e-4
        print("\n")