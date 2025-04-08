#!/bin/bash

#SBATCH --partition pvc       #name of partition either pvc, ampere, genoa, spr 
#SBATCH --nodes=1             #how many nodes (up to 3)
#SBATCH --ntasks=2            #up to 112 per node
#SBATCH --cpus-per-task=1     #set to >1 to use OpenMP threads
#SBATCH --time=00:10:00       
#SBATCH --gres=gpu:2          #necessary for GPU resources, can go up to 4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source pvc-env.sh   #This can be found in the Environment directory of this GitHub repo
cd /cephfs/home/jk945/idefix/test/MHD/disk #The is your own working directory

srun --mpi=pmi2 ./idefix
