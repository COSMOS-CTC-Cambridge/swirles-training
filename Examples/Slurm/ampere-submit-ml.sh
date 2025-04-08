#!/bin/bash

#SBATCH --partition ampere    #name of partition
#SBATCH --nodes=1             #how many nodes (up to 3)
#SBATCH --ntasks=1            #up to 48
#SBATCH --cpus-per-task=1     #set to >1 to use OpenMP threads
#SBATCH --time=00:30:00       
#SBATCH --gres=gpu:1          #up to 4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=1   #number of MPI ranks

### get the first node name as master address - customized for vgg slurm
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


export MY_WORKING_DIRECTORY=/cephfs/home/jk945/swirles-training/Examples/ML
export MY_EXEC="python ./resnet_ddp.py"
export MY_OPTIONS=""

source /cephfs/store/gr-eps1/jk945/.torch-cuda/bin/activate
cd ${MY_WORKING_DIRECTORY}


srun ${MY_EXEC} ${MY_OPTIONS}
