# -*- coding: utf-8 -*-
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
from operator import add
from time import time
from pyspark import SparkContext


def readData(input_file,spark_context):
    """  Read data from an input file and return rdd containing pairs of the form:
                         (x,y)
         where x is a numpy array and y is a real value. The input file should be a 
         'comma separated values' (csv) file: each line of the file should contain x
         followed by y. For example, line:

         1.0,2.1,3.1,4.5

         should be converted to tuple:
        
         (array(1.0,2.1,3.1),4.5)
    """ 
    return spark_context.textFile(input_file)\
        	.map(lambda line: line.split(','))\
        	.map(lambda words: (words[:-1],words[-1]))\
        	.map(lambda inp: (np.array([ float(x) for x in inp[0]]),float(inp[1])))

def readBeta(input):
    """ Read a vector β from CSV file input
    """
    with open(input,'r') as fh:
        str_list = fh.read().strip().split(',')
        return np.array( [float(val) for val in str_list] )           

def writeBeta(output,beta):
    """ Write a vector β to a CSV file ouptut
    """
    with open(output,'w') as fh:
        fh.write(','.join(map(str, beta.tolist()))+'\n')
    
def estimateGrad(fun,x,delta):
     """ Given a real-valued function fun, estimate its gradient numerically.
     """
     d = len(x)
     grad = np.zeros(d)
     for i in range(d):
         e = np.zeros(d)
         e[i] = 1.0
         grad[i] = (fun(x+delta*e) - fun(x))/delta
     return grad


    
def predict(x,beta):
    """ Given vector x containing features and parameter vector β, 
        return the predicted value: 

                        y = <x,β>   
    """
    return x.T @ beta

def f(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector β, return the square error:

                 f(β;x,y) =  (y - <x,β>)^2	
    """
    return (y-predict(x,beta))**2

def localGradient(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector β, return the gradient ∇f of f:

                ∇f(β;x,y) =  -2 * (y - <x,β>) * x	

        with respect to parameter vector β.

        The return value is  ∇f.
    """
    return -2 * (y- predict(x,beta)) * x

def F(data,beta,lam = 0):
    """  Compute the regularized mean square error:

             F(β) = 1/n Σ_{(x,y) in data}    f(β;x,y)  + λ ||β ||_2^2   
                  = 1/n Σ_{(x,y) in data} (y- <x,β>)^2 + λ ||β ||_2^2 

         where n is the number of (x,y) pairs in RDD data. 

         Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: vector β
            - lam:  the regularization parameter λ

         The return value is F(β).
    """
    return data.map(lambda data_i: f(data_i[0],data_i[1],beta)).mean() + lam * beta.T @ beta
     
def gradient(data,beta,lam = 0):
    """ Compute the gradient  ∇F of the regularized mean square error 
                F(β) = 1/n Σ_{(x,y) in data} f(β;x,y) + λ ||β ||_2^2   
                     = 1/n Σ_{(x,y) in data} (y- <x,β>)^2 + λ ||β ||_2^2   
                 
        where n is the number of (x,y) pairs in data. 

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β
             - lam:  the regularization parameter λ

        The return value is an array containing ∇F.
    """
    return data.map(lambda data_i: localGradient(data_i[0],data_i[1],beta)).mean() + 2*lam * beta

def hcoeff(data,beta1, beta2, lam = 0):
    """ Compute the coefficients a,b,c of quadratic function h, defined as :           
                       h(γ) = F(β_1 + γβ_2) = aγ^2 + bγ + c
        where F is the reqularized mean square error function.

        Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta1: vector β_1
            - beta2: vector β_2
            - lam: the regularization parameter λ

        The return value is a tuple containing (a,b,c).    
    """
    N = data.count()
    beta1 = np.expand_dims(beta1,1)
    beta2 = np.expand_dims(beta2,1)

    a = data.map(lambda data_tuple: (beta2.T @ (np.einsum("i,j->ij",data_tuple[0],data_tuple[0])) @ beta2)).mean() + lam*beta2.T @ beta2
    b = 2*lam*beta1.T @ beta2 - 2/N * data.map(lambda data_tuple: (data_tuple[1]*np.dot(data_tuple[0],beta2) - beta1.T @ np.einsum("i,j->ij",data_tuple[0],data_tuple[0]) @ beta2)).sum() # - 2*lam*beta1.T @ beta2
    c = data.map(lambda data_tuple: (data_tuple[1]-np.dot(data_tuple[0],beta1))**2).sum()*1/N + lam*beta1.T @ beta1

    return a.item(),b.item(),c.item()
def exactLineSearch(data,beta,g,lam = 0):
    """ Given  data, a vector x, and a direction g, return
                   γ = argmin_{γ} F(data, β-γg)

        The solution is found by first computing the coefficients of the quadratic
        polynomial 
                   h(γ) = F(data, β-γg) = aγ^2 + bγ + c
        The return value is γ* = -b/(2*a)

        Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: vector β
            - g: direction vector g
            - lam: the regularization parameter λ

        The return value is γ*     

    """
    a,b,c = hcoeff(data,beta,-g,lam)
    gamma = -b/(2*a)

    return gamma

def test(data,beta):
    """ Compute the mean square error  

        	 MSE(β) =  1/n Σ_{(x,y) in data} (y- <x,β>)^2

        of parameter vector β over the dataset contained in RDD data, where n is the size of RDD data.
        
        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return value is MSE(β).  
    """
    return F(data,beta,lam=0)

def train_GD(data,beta_0, lam,max_iter,eps):
    """ Perform gradient descent to  minimize F given by
  
             F(β) = 1/n Σ_{(x,y) in data} f(β;x,y) + λ ||β ||_2^2   

        where
             - data: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector β
             - lam:  is the regularization parameter λ
             - max_iter: maximum number of iterations of gradient descent
             - eps: upper bound on the l2 norm of the gradient

        The function performs gradient descent with a gain found through 
        exact line search. That is, it computes
                   
                   β_k+1 = β_k - γ_k ∇F(β_k) 
        	
        where the gain γ_k is given by
        
                   γ_k = argmin_{γ} F(β_k - γ ∇F(β_k))

        and terminates after max_iter iterations or when ||∇F(β_k)||_2<ε.   

        The function returns:
             -beta: the trained β, 
             -gradNorm: the norm of the gradient at the trained β, and
             -k: the number of iterations performed
    """
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:
        grad = gradient(data,beta,lam)  
        obj = F(data,beta,lam)   
        gamma = exactLineSearch(data,beta,grad,lam)
        gradNorm = np.linalg.norm(grad)
        print('k =',k,'\tt =',time()-start,'\tF(β_k) =',obj,'\t||∇F(β_k)||_2=',gradNorm,'\tγ_k =',gamma)
        beta = beta - gamma * grad
        k = k + 1

    return beta,gradNorm,k


def solveLin(z,K):
    """ Solve problem
           Minimize:  z^T β
           subject to:  ||β||_1 <=Κ
        
        The return value is the optimal β*.
    """
    abs_max = np.max(np.abs(z))
    # print(-K*np.sign(z)*(np.abs(z)==abs_max))
    return -K*np.sign(z)*(np.abs(z)==abs_max)

def exactLineSearchFW(data,beta,s):
    """ Given  data, a vector x, and a direction g, return
                   γ' = argmin_{γ in [0,1]} F(data, (1-γ)β+γs)

        The solution is found by first computing the coefficients of the quadratic
        polynomial 
                   h(γ) = F(data, (1-γ)β + γ s) = aγ^2 + bγ + c

        Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: first interpolation vector β
            - s: second interpolation vector s

        The return value is γ'     

    """
    a, b, c = hcoeff(data, beta, s-beta, lam=0)
    if a != 0:
        gamma = np.clip(-b / (2 * a),0,1)
    else:
        if b > 0:
            gamma = 0
        else:
            gamma = 1

    return gamma





def train_FW(data,beta_0, K,max_iter,eps):
    """ Use the Frank-Wolfe algorithm   minimize F_0 given by
  
             F_0(β) = 1/n Σ_{(x,y) in data} f(β;x,y)    
        
        Subject to:
             ||β||_1 <= K

        Inputs are:
             - data: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector β
             - K:  the bound K
             - max_iter: maximum number of iterations
             - eps: upper bound on the convergence criterion

        The function runs the Frank-Wolfe algorithm with a step-size found through 
        exact line search. That is, it computes
                   
                   s_k =  argmin_{s:||s||_1<=K} s^T ∇F_0(β_k) 
                   β_k+1 = (1-γ_κ)β_k + γ_k s_k 
        	
        where the gain γ_k is given by
        
                   γ_k = argmin_{γ in [0,1]} F_0((1-γ_κ)β_k + γ_κ s_k))

        and terminates after max_iter iterations or when (β_k-s_k)^T∇F(β_k)<ε.   

        The function returns:
             -beta: the trained β, 
             -criterion: the condition (β_k-s_k)^T ∇F(β_k)
             -k: the number of iterations performed
    """



    k = 0
    criterion = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and criterion > eps:
        obj = F(data,beta,lam=0)
        
        grad = gradient(data, beta, lam=0)
        s = solveLin(grad, K)

        criterion = (beta-s) @ grad

        gamma = exactLineSearchFW(data, beta, s)
        beta = (1 - gamma) * beta + gamma * s

        feasibility = np.linalg.norm(beta,ord=1)/K

        print('k =',k,'\tt =',time()-start,'\tF(β_k) =',obj,'\t|β_k)|/K=',feasibility,'\t(β_k-s_k)^T ∇F(β_k) =',criterion,'\tγ_k =',gamma)

        k = k + 1

    return beta,criterion,k



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Regression.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--K', type=float,default=100.00, help='L1 norm threshold')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')
    parser.add_argument('--N',type=int,default=25,help='Level of parallelism')
    parser.add_argument('--solver',default='GD',choices=['GD', 'FW'],help='GD learns β  via gradient descent, FW learns β using the Frank Wolfe algorithm')

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Regression')
    sc.setLogLevel('warn')
    
    beta = None
                            
    if args.traindata is not None:
        # Train a linear model β from data, and store it in beta
        print('Reading training data from',args.traindata)
        data = readData(args.traindata,sc)
        data = data.repartition(args.N).cache()
        
        x,y = data.take(1)[0]
        dim = len(x)

        plt.figure()
        plt.hist(data.map(lambda data_tuple: data_tuple[1]).collect())
        plt.savefig("question8.png")
        plt.show()
        
        if args.solver == 'GD':
            start = time()
            print('Gradient descent training on data from',args.traindata,'with λ =',args.lam,', ε =',args.eps,', max iter = ',args.max_iter)
            beta0 = np.zeros(dim)
            beta, gradNorm, k = train_GD(data,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps) 
            print('Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps, 'Training time:', time()-start)
            print('Saving trained β in',args.beta)
            writeBeta(args.beta,beta)

        else:
            start = time()
            print('Frank-Wolfe training on data from',args.traindata,'with K =',args.K,', ε =',args.eps,', max iter = ',args.max_iter)
            beta0 = np.zeros(dim)
            beta, criterion, k = train_FW(data,beta_0=beta0,K=args.K,max_iter=args.max_iter,eps=args.eps) 
            print('Algorithm ran for',k,'iterations. Converged:',criterion<args.eps, 'Training time:', time()-start)
            print('Saving trained β in',args.beta)
            writeBeta(args.beta,beta)
     
    if args.testdata is not None:
        # Read beta from args.beta, and evaluate its MSE over data
        print('Reading test data from',args.testdata)
        data = readData(args.testdata,sc)
        data = data.repartition(args.N).cache()
        
        print('Reading β from',args.beta)
        beta = readBeta(args.beta)

        print('Computing MSE on data',args.testdata)
        MSE = test(data,beta)
        print('MSE is:', MSE)

