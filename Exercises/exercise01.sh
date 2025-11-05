#!/bin/bash

#SBATCH --partition pvc                      #name of partition either pvc, ampere, genoa, spr 
#SBATCH --nodes=<number of nodes>            #how many nodes (up to 3)
#SBATCH --ntasks=<number of tasks>           #e.g. up to 112 per PVC node
#SBATCH --cpus-per-task=<number of cpus>     #set to >1 to use OpenMP threads
#SBATCH --time=hh:mm:ss       
#SBATCH --gres=gpu:<number of gpus>          #necessary for GPU resources, can go up to 4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MY_WORKING_DIRECTORY="please fill this in..."
export MY_EXEC="...also fill this in.."
export MY_OPTIONS="...fill in as well..."

#set up the environment
source ~/swirles-training/Environments/pvc-env.sh #This can be found from the Environment directory of this GitHub repo

cd ${MY_WORKING_DIRECTORY}

#Build the executable
make all

#Run the executable
srun --mpi=pmi2 ${MY_EXEC} ${MY_OPTIONS}
