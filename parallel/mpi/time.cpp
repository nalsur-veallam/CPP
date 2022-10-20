#include "mpi.h"
#include <iostream>
#include <chrono>

#define N 100000
#define K 100

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size >= 2){
        for(int i = 1; i <= N; i++){
            double time = 0;
            for(int j = 0; j < K;j++){
            
                auto start = std::chrono::steady_clock::now();
                int msg[i];
                
                if (rank == 0){
                    MPI_Send(&msg, i, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }
                
                if (rank == 1){
                    int new_msg[i];
                    MPI_Recv(&new_msg, i, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                
                auto end = std::chrono::steady_clock::now(); 
                std::chrono::duration<double> elapsed_seconds = end-start;
                time += double(elapsed_seconds.count());
                            
            }
            std::cout << i << " " <<  1000/K*time << std::endl;
        }
        
    }
    
    MPI_Finalize();
    
    return 0;
}
