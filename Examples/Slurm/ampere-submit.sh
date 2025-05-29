#!/bin/bash
#SBATCH --partition ampere
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:2



export GRTECLYN_DIR=/cephfs/home/jk945/GRTeclyn/Examples/BinaryBH
export application=${GRTECLYN_DIR}/main3d.gnu.MPI.CUDA.ex
export options="${GRTECLYN_DIR}/params_profile.txt checkpoint_interval=-1 plot_interval=-1"

. /usr/share/modules/init/bash
module purge
module load openmpi/5.0.3/gcc/xs76solb

srun --mpi=pmix $application $options
