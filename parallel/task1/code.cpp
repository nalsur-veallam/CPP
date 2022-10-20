#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <mpi.h>

#define K 10

std::random_device randomizer;
std::mt19937 seed(randomizer());

const double lowRand = 0.0;
const double highRand = 1.0;

inline double getRandomReal(double low, double high)
{
    std::uniform_real_distribution<double> intDistrib(low, high);
    return intDistrib(seed);
}

void randomizeMatrix(double *A, int M, int N, double low, double high)
{
    for(int i = 0; i < M * N; ++i) {
        A[i] = getRandomReal(low, high);
    }
}

void MatrixMultiplicationMPI(double *&A, double *&B, double *&C, int &Size, int &ProcNum, int &ProcRank) {
	int dim = Size;
	int i, j, k, p, ind;
	double temp = 0;
	MPI_Status Status;
	int ProcPartSize = dim/ProcNum; 
	int ProcPartElem = ProcPartSize*dim; 
	double* bufA = new double[ProcPartElem];
	double* bufB = new double[ProcPartElem];
	double* bufC = new double[ProcPartElem];
	int ProcPart = dim/ProcNum, part = ProcPart*dim;
	
	MPI_Scatter(A, part, MPI_DOUBLE, bufA, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, part, MPI_DOUBLE, bufB, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
	for (i=0; i < ProcPartSize; i++) {
		for (j=0; j < ProcPartSize; j++) {
			for (k=0; k < dim; k++) 
				temp += bufA[i*dim+k]*bufB[j*dim+k];
			bufC[i*dim+j+ProcPartSize*ProcRank] = temp;
			temp = 0.0;
		}
	}

	int NextProc; int PrevProc;
	for (p=1; p < ProcNum; p++) {
		NextProc = ProcRank+1;
		if (ProcRank == ProcNum-1) 
			NextProc = 0;
		PrevProc = ProcRank-1;
		if (ProcRank == 0) 
			PrevProc = ProcNum-1;
		MPI_Sendrecv_replace(bufB, part, MPI_DOUBLE, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
		temp = 0.0;
		for (i=0; i < ProcPartSize; i++) {
			for (j=0; j < ProcPartSize; j++) {
				for (k=0; k < dim; k++) {
					temp += bufA[i*dim+k]*bufB[j*dim+k];
				}
				if (ProcRank-p >= 0 ) 
					ind = ProcRank-p;
				else ind = (ProcNum-p+ProcRank);
				bufC[i*dim+j+ind*ProcPartSize] = temp;
				temp = 0.0;
			}
		}
	}
	
	MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete []bufA;
	delete []bufB;
	delete []bufC;
}

int main(int argc, char *argv[])
{
    int Size = 1000;
    int ProcNum, ProcRank;
    double time = 0;
    
    double *A = new double[Size*Size];
    double *B = new double[Size*Size];
    double *C = new double[Size*Size];
    
    randomizeMatrix(A, Size, Size, lowRand, highRand);
    randomizeMatrix(B, Size, Size, lowRand, highRand);
    std::fill(C, C + Size*Size, 0);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);   
    
    auto start = std::chrono::steady_clock::now();
    
    for(int i = 0; i < K; i++){
    
    MatrixMultiplicationMPI(A, B, C, Size, ProcNum, ProcRank);
    
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0){
    auto end = std::chrono::steady_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end-start;
    time += double(elapsed_seconds.count());
    
    std::cout << ProcNum << " " << 1000/K*time << std::endl;
    }
              
    delete[] A;
    delete[] B;
    delete[] C;
    
    MPI_Finalize();
    
    return 0;
}







