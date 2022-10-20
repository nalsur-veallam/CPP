#include "mpi.h"
#include <iostream>
#include <string.h>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size >= 2){        
        const char* msg = "Hello, process number 1";
        
        if (rank == 0){
            MPI_Send(msg, strlen(msg), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            std::cout << "Process with rank " << rank << " sent a message to the process with rank 1." << std::endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 1){
            char* new_msg;
            MPI_Recv(new_msg, strlen(msg), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Process with rank " << rank << " received the message: " << new_msg  << std::endl;
        }
    }
    
    MPI_Finalize();
    
    return 0;
}
