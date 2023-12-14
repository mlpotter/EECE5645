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
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Analysis via the Dale Chall Formula',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--N',type=int,default=30,help="Number of partitions to be used in RDDs containing word counts.")
    parser.add_argument('--master', default="local[25]", help="Spark Master")
    parser.add_argument('--simple_words', default="/courses/EECE5645.202410/data/HW1/Data/DaleChallEasyWordList.txt",
                        help="File containing Dale Chall simple word list. Each word appears in one line.")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Text Analysis')
    # sc.setLogLevel('warn')
    easy_list = create_list_from_file(args.simple_words)

    data_folder = "Data"
    directories = ["Books","Movies","News"]

    df = pd.DataFrame(columns=["Category","File","DCF"])
    for directory in directories:
        print("Directory: ",directory)
        for file in os.listdir(os.path.join(data_folder,directory)):
        # Add tour code here
            corpus = sc.textFile(os.path.join(data_folder,directory,file),minPartitions=args.N)#.repartition(n)

            dcf = dalechallformula(corpus,easy_list,numPartitions = args.N)
            print("File: ", os.path.join(data_folder, directory, file), "DCF={:.4f}".format(dcf))

            temp_df = pd.DataFrame({"Category":[directory],"File":[file],"DCF":[dcf]})

            df = pd.concat((df,temp_df),axis=0,ignore_index=True)

    corpus = sc.textFile(os.path.join(data_folder, directory, file), minPartitions=args.N)  # .repartition(n)


    file = "fireandice.txt"
    corpus = sc.textFile(os.path.join(data_folder, "fireandice.txt"), minPartitions=args.N)  # .repartition(n)
    dcf = dalechallformula(corpus, easy_list, numPartitions=args.N)
    print("File: ", os.path.join(data_folder, file), "DCF={:.4f}".format(dcf))

    temp_df = pd.DataFrame({"Category": ["NONE"], "File": [file], "DCF": [dcf]})

    df = pd.concat((df, temp_df), axis=0, ignore_index=True)

    df.to_excel("question6.xlsx",index=False)