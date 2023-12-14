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
    parser.add_argument('--N', type=int, default=25, help='Level of parallelism')
    parser.add_argument('--testdata',default="data/small.test", help='Input file containing (x,y) pairs, used to test a linear model')

    args = parser.parse_args()

    sc = SparkContext(f"local[{args.N}]",'Parallel Regression')
    sc.setLogLevel('warn')

    args = parser.parse_args()

    np.random.seed(42)

    dataset =  PR.readData(args.testdata,sc)
    data_tuple = dataset.take(1)
    dim = len(data_tuple[0][0])


    gammas =  np.random.uniform(0,2,args.samples).tolist() + [0,1,1]
    lams = np.random.uniform(0,5,args.samples).tolist() + [1,0,1]

    for i,(gamma,lam) in enumerate(zip(gammas,lams)):
        beta1 = np.random.randn(dim, ).astype(np.double)
        beta2 = np.random.randn(dim, ).astype(np.double)

        a, b, c = PR.hcoeff(dataset, beta1, beta2, lam)

        print("\nTest {:5d}, gamma={:.3f} lam={:.3f}".format(i,gamma,lam))
        print("a={:.3f}, b={:.3f}, c={:.3f}".format(a,b,c))
        h = a * gamma ** 2 + b * gamma + c
        hf = PR.F(dataset, beta=beta1 + gamma * beta2, lam=lam)
        print("h={:.3f} hf={:.3f}".format(h,hf))
        assert np.abs(h-hf) < 1e-8