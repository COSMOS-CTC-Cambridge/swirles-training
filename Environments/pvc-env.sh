#!/bin/bash

module load intel/compiler/2025.1.0
module load intel/mpi/2021.15
module load intel/mkl/2025.1

export I_MPI_PMI_LIBRARY=/usr/lib/x86_64-linux-gnu/libpmi2.so
export I_MPI_OFFLOAD=1
unset ZE_AFFINITY_MASK
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export I_MPI_OFFLOAD_TOPOLIB=level_zero
export I_MPI_OFFLOAD_PIN=1
export I_MPI_DEBUG=3

