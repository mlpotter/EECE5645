#!/bin/bash
#SBATCH --job-name=spark-pi      # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=20     # number of tasks per node
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                # memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
spark-submit --total-executor-cores 60 --executor-memory 1G --master spark://10.99.253.33:7077 --executor-memory 100G --driver-memory 100G MFspark.py med_data 4 --N 20 --gain 0.0001 --pow 0.2 --maxiter 20 --d 1