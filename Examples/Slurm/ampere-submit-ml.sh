#!/bin/bash

#SBATCH --partition ampere    #name of partition
#SBATCH --nodes=1             #how many nodes (up to 3)
#SBATCH --ntasks-per-node=2   #same as number of GPUs
#SBATCH --time=00:30:00       
#SBATCH --gres=gpu:2          #up to 4

export OMP_NUM_THREADS=1
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others

export MASTER_PORT=12340

### get the first node name as master address - customized for vgg slurm
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export RANK=$SLURM_PROCID
### WORLD_SIZE should be gpus/node * num_nodes
export WORLD_SIZE=$(echo "$SLURM_NTASKS_PER_NODE * $SLURM_NNODES" | bc -l)


export MY_WORKING_DIRECTORY=/cephfs/home/jk945/swirles-training/Examples/ML
export MY_PYTHON_SCRIPT="./resnet_ddp.py"
export MY_OPTIONS=""

source /cephfs/store/gr-eps1/jk945/.torch-cuda/bin/activate
cd ${MY_WORKING_DIRECTORY}


torchrun --nnode=${SLURM_NNODES} --nproc_per_node=${SLURM_NTASKS_PER_NODE} ${MY_PYTHON_SCRIPT} ${MY_OPTIONS}
