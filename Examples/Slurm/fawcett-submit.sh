#!/bin/bash

#SBATCH --partition cosmosx       #name of partition either cosmosx,knl,skylake 
#SBATCH --nodes=1                 #there is only one node...
#SBATCH --ntasks=1                #how many tasks (usu MPI ranks) in total
#SBATCH --cpus-per-task=4         #set to >1 to use OpenMP threads
#SBATCH --time=hh:mm:ss


#Load any modules that you might need


#Set environment variables


mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export application="<full path to your exec>"
export options="<your arguments>"

mpirun -np $SLURM_NTASKS -ppn $mpi_tasks_per_node $application $options
