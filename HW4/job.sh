#!/bin/bash
#SBATCH --job-name=spark-pi      # create a short name for your job
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                # memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
spark-submit --total-executor-cores 208 --executor-memory 100G --master spark://10.99.253.38:7077  --driver-memory 100G /scratch/potter.mi/EECE5645/HW4/MFspark.py big_data 5 --N 208 --gain 0.0001 --pow 0.2 --maxiter 20 --d 10