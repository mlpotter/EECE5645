#!/bin/bash
#SBATCH --job-name=sparkith         # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
echo $MASTER

spark-submit --total-executor-cores 50 --executor-memory 100G --driver-memory 100G /scratch/potter.mi/EECE5645/HW4/MFspark.py big_data 5 --N 20 --gain 0.0001 --pow 0.2 --maxiter 20 --d 1