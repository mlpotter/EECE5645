import numpy as np
import pandas as pd
from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.random import RandomRDDs
from MFspark import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallele Matrix Factorization.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', help='Directory containing folds. The folds should be named fold0, fold1, ..., foldK.')
    parser.add_argument('folds', type=int, help='Number of folds')
    parser.add_argument('--gain', default=0.001, type=float, help="Gain")
    parser.add_argument('--power', default=0.2, type=float, help="Gain Exponent")
    parser.add_argument('--epsilon', default=1.e-99, type=float, help="Desired objective accuracy")
    parser.add_argument('--lam', default=0, type=float, help="Regularization parameter for user features")
    parser.add_argument('--mu', default=0, type=float, help="Regularization parameter for item features")
    parser.add_argument('--d', default=10, type=int, help="Number of latent features")
    parser.add_argument('--outputfile', help='Output file')
    parser.add_argument('--maxiter', default=20, type=int, help='Maximum number of iterations')
    parser.add_argument('--N', default=20, type=int, help='Parallelization Level')
    parser.add_argument('--seed', default=1234567, type=int, help='Seed used in random number generator')
    parser.add_argument('--output', default=None,
                        help='If not None, cross validation is skipped, and U,V are trained over entire dataset and store it in files output_U and output_V')
    parser.add_argument('--grid_size', default=20, type=int, help='grid spacing')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)



    args = parser.parse_args()

    reg_strength = np.linspace(0,3000,args.grid_size)
    cv_rmse = np.zeros((args.grid_size,))

    sc = SparkContext(appName='Parallel MF')

    for m,regi in enumerate(reg_strength):
        args.lam = regi
        args.mu = regi

        print(f"Lambda=Mu={regi}")


        if not args.verbose:
            sc.setLogLevel("ERROR")

        folds = {}

        if args.output is None:
            for k in range(args.folds):
                folds[k] = readRatings(args.data + "/fold" + str(k), sc)
        else:
            folds[0] = readRatings(args.data, sc)

        cross_val_rmses = []
        for k in folds:
            train_folds = [folds[j] for j in folds if j is not k]

            if len(train_folds) > 0:
                train = train_folds[0]
                for fold in train_folds[1:]:
                    train = train.union(fold)
                train.repartition(args.N).cache()
                test = folds[k].repartition(args.N).cache()
                Mtrain = train.count()
                Mtest = test.count()

                print("Initiating fold %d with %d train samples and %d test samples" % (k, Mtrain, Mtest))
            else:
                train = folds[k].repartition(args.N).cache()
                test = train
                Mtrain = train.count()
                Mtest = test.count()
                print(
                    "Running single training over training set with %d train samples. Test RMSE computes RMSE on training set" % Mtrain)

            i = 0
            change = 1.e99
            obj = 1.e99
            # rd.seed(args.seed)

            # Generate user profiles
            U = generateUserProfiles(train, args.d, args.seed, sc, args.N).cache()
            V = generateItemProfiles(train, args.d, args.seed, sc, args.N).cache()

            print("Training set contains %d users and %d items" % (U.count(), V.count()))

            start = time()
            gamma = args.gain

            while i < args.maxiter and change > args.epsilon:
                i += 1

                joinedRDD = joinAndPredictAll(train, U, V, args.N).cache()

                oldObjective = obj
                obj = SE(joinedRDD) + normSqRDD(U, args.lam) + normSqRDD(V, args.mu)
                change = np.abs(obj - oldObjective)

                testRMSE = np.sqrt(1. * SE(joinAndPredictAll(test, U, V, args.N)) / Mtest)

                gamma = args.gain / i ** args.power

                U.unpersist()
                V.unpersist()
                U = adaptU(joinedRDD, gamma, args.lam, args.N).cache()
                V = adaptV(joinedRDD, gamma, args.mu, args.N).cache()

                now = time() - start
                print("Iteration: %d\tTime: %f\tObjective: %f\tTestRMSE: %f" % (i, now, obj, testRMSE))
                joinedRDD.unpersist()

            cross_val_rmses.append(testRMSE)

            train.unpersist()
            test.unpersist()

        if args.output is None:
            print("%d-fold cross validation error is: %f " % (args.folds, np.mean(cross_val_rmses)))
        else:
            print("Saving U and V RDDs")
            U.saveAsTextFile(args.output + '_U')
            V.saveAsTextFile(args.output + '_V')

        cv_rmse[m] = np.mean(cross_val_rmses)

df = pd.DataFrame({"reg":reg_strength,"cv_rmse":cv_rmse})
df.to_excel("question4e.xlsx",index=False)