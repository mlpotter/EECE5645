import sys
import argparse
import findspark

findspark.init()
from time import time
from pyspark import SparkContext
from helpers import *
from TextAnalyzer import *
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Analysis via the Dale Chall Formula',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',
                        help='Text file to be processed. This file contains text over several lines, with each line corresponding to a different sentence.')
    parser.add_argument('--master', default="local[25]", help="Spark Master")
    parser.add_argument('--simple_words', default="/courses/EECE5645.202410/data/HW1/Data/DaleChallEasyWordList.txt",
                        help="File containing Dale Chall simple word list. Each word appears in one line.")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Text Analysis')
    # sc.setLogLevel('warn')
    easy_list = create_list_from_file(args.simple_words)

    N =  [2,5,10,15,20]
    # N = [15,20]
    N_times = []
    for n in N:


        # Add tour code here
        corpus = sc.textFile(args.input)#.repartition(n)
        start = time()

        dalechallformula(corpus,easy_list,numPartitions = n)

        end = time()
        N_times.append(end-start)
        print(f'Total execution time for {n}:', str(end - start) + 'sec')


    pd.DataFrame({"N":N,"times":N_times}).to_csv("question.csv")

    plt.figure()
    plt.bar(x=N,height=N_times,color='blue')
    plt.xlabel("N Partitions",fontsize=15); plt.ylabel("Time (sec)",fontsize=15)
    plt.savefig("question5.png")
    plt.close()