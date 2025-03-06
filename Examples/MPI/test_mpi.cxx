#include <mpi.h>
#include <iostream>

int main(int argc, char** argv)
{
  int rank = 0;
  int numprocs = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "This is processor " << rank << " out of " << numprocs << std::endl; 

  MPI_Finalize(); 
  
  return 0; 
}
