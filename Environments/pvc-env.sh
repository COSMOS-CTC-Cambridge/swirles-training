#!/bin/bash

module load intel-oneapi-compilers/2024.1.0/gcc/o3pkwb7f
module load intel-oneapi-mpi/2021.12.1/oneapi/gfbqixxv


export I_MPI_PMI_LIBRARY=/usr/lib/x86_64-linux-gnu/libpmi2.so
export I_MPI_OFFLOAD=1
unset ZE_AFFINITY_MASK
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export I_MPI_OFFLOAD_TOPOLIB=level_zero
export I_MPI_OFFLOAD_PIN=1
export I_MPI_DEBUG=3

