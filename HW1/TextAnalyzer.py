import sys
import argparse
import findspark
findspark.init()
from time import time
from pyspark import SparkContext
from helpers import *

def count_sentences(rdd):
    """ Count the sentences in a file.

    Input:
    - rdd: an RDD containing the contents of a file, with one sentence in each element.

    
    Return value: The total number of sentences in the file.
    """
    return rdd.count()

def count_words(rdd):
    """ Count the number of words in a file.

    Input:
    - rdd: an RDD containing the contents of a file, with one sentence in each element.

    
    Return value: The total number of words in the file.
    """

    return rdd.flatMap(lambda s: s.split()).count()

def compute_counts(rdd,numPartitions = 10):
    """ Produce an rdd that contains the number of occurences of each word in a file.

    Each word in the file is converted to lowercase and then stripped of leading and trailing non-alphabetic
    characters before its occurences are counted.

    Input:
    - rdd: an RDD containing the contents of a file, with one sentence in each element.

    
    Return value: an RDD containing pairs of the form (word,count), where word is is a lowercase string, 
    without leading or trailing non-alphabetic characters, and count is the number of times it appears
    in the file. The returned RDD should have a number of partitions given by numPartitions.

    """

    word_counts = rdd.repartition(numPartitions).flatMap(lambda s: s.split())  \
        .map(lambda word: (strip_non_alpha(to_lower_case(word)),1))  \
        .reduceByKey(lambda x,y: x+y) #\
        # .sortBy(lambda pair: pair[1],ascending=False)

    return word_counts
    

def count_difficult_words(counts,easy_list):
    """ Count the number of difficult words in a file.

    Input:
    - counts: an RDD containing pairs of the form (word,count), where word is a lowercase string, 
    without leading or trailing non-alphabetic characters, and count is the number of times this word appears
    in the file.
    - easy_list: a list of words deemed 'easy'.


    Return value: the total number of 'difficult' words in the file represented by RDD counts. 

    A word should be considered difficult if is not the 'same' as a word in easy_list. Two words are the same
    if one is the inflection of the other, when ignoring cases and leading/trailing non-alphabetic characters. 
    """
    # print(counts.filter(lambda word_pair: find_match_stricter(word_pair[0],easy_list) is None).collect())

    difficult_count = counts.filter(lambda word_pair: find_match(word_pair[0],easy_list) is None)  \
                    .map(lambda x: x[1]) \
                    .reduce(lambda x,y: x+y)

    return difficult_count

def dalechallformula(rdd,easy_list,numPartitions = 10):
    """

    :param rdd: The  text file rdd
    :param easy_list: the list o f easy words to exclude from difficult words
    :return: the DCF
    """
    num_sentences = count_sentences(rdd)
    num_words = count_words(rdd)
    compute_counts_rdd = compute_counts(rdd,numPartitions)
    num_dff = count_difficult_words(compute_counts_rdd, easy_list)
    dcf = 0.1579 * (num_dff / num_words) * 100 + 0.0496 * (num_words / num_sentences)
    return dcf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis via the Dale Chall Formula',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['SEN','WRD','UNQ','TOP30','DFF','DCF'])
    parser.add_argument('input', help='Text file to be processed. This file contains text over several lines, with each line corresponding to a different sentence.')
    parser.add_argument('--master',default="local[25]",help="Spark Master")
    parser.add_argument('--N',type=int,default=30,help="Number of partitions to be used in RDDs containing word counts.")
    parser.add_argument('--simple_words',default="/courses/EECE5645.202410/data/HW1/Data/DaleChallEasyWordList.txt",help="File containing Dale Chall simple word list. Each word appears in one line.")
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')
    # sc.setLogLevel('warn')

    start = time()

    # Add tour code here
    corpus = sc.textFile(args.input) #.repartition(args.N)
    if args.mode == "SEN":
        print("Number of sentences: ",count_sentences(corpus))

    if args.mode == "WRD":
        print("Number of words: ",count_words(corpus))

    if args.mode == "UNQ":
        compute_counts_rdd = compute_counts(corpus,args.N)
        print("Number of partitions Used: ",compute_counts_rdd.getNumPartitions())
        print("Number of unique words: ",compute_counts_rdd.count())

    if args.mode == "TOP30":
        compute_counts_rdd = compute_counts(corpus,args.N)
        print("Number of partitions Used: ",compute_counts_rdd.getNumPartitions())
        print("Top 30 Words: ",compute_counts_rdd.takeOrdered(30,lambda x: -x[1]))

    if args.mode == "DFF":
        compute_counts_rdd = compute_counts(corpus,args.N)
        easy_list = create_list_from_file(args.simple_words)
        print("Number of partitions Used: ",compute_counts_rdd.getNumPartitions())
        dff = count_difficult_words(compute_counts_rdd,easy_list)
        print("Number of Difficult words: ",dff)

    if args.mode == "DCF":
        easy_list = create_list_from_file(args.simple_words)
        dcf = dalechallformula(corpus,easy_list,args.N)
        print("Dale-Chall Formula: {:.5f}".format(dcf))

    #

    end = time()
    print('Total execution time:',str(end-start)+'sec')
