# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from operator import add
from time import time
import findspark
findspark.init()
import ParallelRegression as PR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Regression.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--samples' ,type=int ,default=100 ,help='Number of random x,y, beta values')
    parser.add_argument('--noise_std' ,type=float ,default=0.1 ,help='Standard Deviation of noise')
    parser.add_argument('--delta' ,type=float ,default=1e-8 ,help='Numerical Derivative delta size')

    args = parser.parse_args()

    np.random.seed(42)

    for i in range(args.samples):
        dim = np.random.randint(1,10)
        x = np.random.randn(dim,).astype(np.double)
        beta = np.random.randn(dim,).astype(np.double)
        y = x.T @ beta + np.random.randn(1,).astype(np.double)*args.noise_std

        partial_f = lambda beta: PR.f(x=x,beta=beta,y=y) #partial(PR.f,x=x,y=y)

        print("i={:4d} y={} x.T@beta ={}".format(i,np.round(y,4),np.round(x.T @ beta,4)))
        print("Numerical Gradient: ", PR.estimateGrad(partial_f,x,args.delta))
        print("Analytical Gradient: ", PR.localGradient(x,y,beta))
        norm = np.linalg.norm(PR.localGradient(x,y,beta) - PR.estimateGrad(partial_f,beta,args.delta),ord=np.inf)
        print('Norm={:.4f}'.format(norm))
        assert norm < 1e-4