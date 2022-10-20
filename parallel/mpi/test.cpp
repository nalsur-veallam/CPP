#include "mpi.h"
#include <iostream>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::cout << "Hello from rank " << rank << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){
        std::cout << "System's size is " << size << std::endl;
    }
    MPI_Finalize();
    
    return 0;
}
