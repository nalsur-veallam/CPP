#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <omp.h>

// g++ -fopenmp mul.cpp

void gemm(int N, const double *A,
          const double *B, double *C)
{
    int i,k;
    #pragma omp parallel for shared(A,B,C) private(i,k)
    for (int i = 0; i < N; i++){
        C[i%N] = 0;
        for (int k = 0; k < N; k++){
            C[i%N] += A[k*N+i%N]*B[k%N];
        }
    }
}

void gemmSerialVerifier(int N, const double *A,
                        const double *B, double *C)
{
    for (int i = 0; i < N; i++){
        for (int k = 0; k < N; k++){
            C[i] += A[k*N+i]*B[k];
        }
    }
}

double norm(int N, const double *A)
{
    double norm = 0;
    for (int i = 0; i < N; i++){
        norm += A[i];
    }
    
    return norm;
}

std::string helpMsg = "Usage: ./dgemm <matrixSize> <iterations> <threadsNum>";

#define errorChk(ans, msg)                 \
    {                                      \
        if (!ans) {                        \
            std::cerr << msg << std::endl; \
            exit(1);                       \
        }                                  \
    }                                      \


void customMatrix(double *A, int N)
{
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; j++){
            if (j%2 == 0)
                A[j*N + i] = double(i+1);
            else
                A[j*N + i] = -1*double(i+1);
        }
    }
}

int main(int argc, char *argv[])
{
    int matrixSize, elementsInMatrix, computeIters, threadsNum;

    if (argc == 4) {
        std::istringstream dimSizeStr(argv[1]);
        dimSizeStr >> matrixSize;
        errorChk((matrixSize > 0) && (matrixSize < std::numeric_limits<int>::max()),
                 "Matrix dimension should be greater than 0 and less than " +
                 std::to_string(std::numeric_limits<int>::max()));
        elementsInMatrix = matrixSize * matrixSize;

        std::istringstream computeItersStr(argv[2]);
        computeItersStr >> computeIters;
        errorChk(computeIters > 0, "Number of iterations should be greater than 0");

        std::istringstream threadsNumStr(argv[3]);
        threadsNumStr >> threadsNum;
        errorChk(threadsNum, "Number of OpenMP threads should be greater than 0");
        omp_set_num_threads(threadsNum);
    } else {
        errorChk(false, "Not enough arguments! " + helpMsg);
    }

    double *A = new double[elementsInMatrix];
    errorChk(A, "new failed!");
    double *B = new double[matrixSize];
    errorChk(B, "new failed!");
    double *C = new double[matrixSize];
    errorChk(C, "new failed!");
    
    int iter = 0;
    double elem;
    std::ifstream in("./vector.txt"); // окрываем файл для чтения
    if (in.is_open())
    {
        while (in >> elem && iter < matrixSize)
        {
            B[iter] = elem;
            iter += 1;
        }
    }
    in.close();

    double flop = 2.0 * static_cast<double>(matrixSize)
        * static_cast<double>(matrixSize)
        * static_cast<double>(matrixSize);

    customMatrix(A, matrixSize);
    std::fill(C, C + matrixSize, 0);

    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl
              << "Matrix size: " << "(" << matrixSize << "x" << matrixSize << ")"
              << "Compute iters: " << computeIters << std::endl
              << std::endl;

    auto startTime = omp_get_wtime();

    for (int i = 0; i < computeIters; ++i)
        gemm(matrixSize, A, B, C);

    auto endTime = omp_get_wtime();
    

    auto avgIterTime = (endTime - startTime) / static_cast<double>(computeIters);
    std::cout << "Sec/iter: " << avgIterTime << std::endl
              << "GFLOPS: " << flop / avgIterTime << std::endl
              << "Final vector norm: " << norm(matrixSize, C) << std::endl
              << std::endl;
/*
    std::cout << "Verifying results..." << std::endl;
    if (!verifyResults(matrixSize, matrixSize, matrixSize, A, B, C)) {
        std::cout << "Verification failed!" << std::endl;
    } else {
        std::cout << "Verification successful!" << std::endl;
    }
 */

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
