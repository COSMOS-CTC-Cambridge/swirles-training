#!/bin/bash

#SBATCH --partition <name>        #name of partition either pvc, ampere, genoa, spr 
#SBATCH --nodes=<#nodes>          #how many nodes
#SBATCH --ntasks=<#tasks>         #how many tasks (usu MPI ranks) in total 
#SBATCH --cpus-per-task=<#cpus>   #set to >1 to use OpenMP threads
#SBATCH --time=hh:mm:ss       


mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export application=<full path to your exec>
export options=<your arguments>

mpirun -np $SLURM_NTASKS -ppn $mpi_tasks_per_node $application $options
