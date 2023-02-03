#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <omp.h>

void gemmSerialVerifier(int M, int N, int K, const double *A,
                        const double *B, double *C)
{
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            C[i*N+j%N] = 0;
            for (int k = 0; k < N; k++){
                C[i*N+j%N] += A[k*N+i%N]*B[j*N+k%N];
            }
        }
    }
}

void gemm(int M, int N, int K, const double *A,
          const double *B, double *C)
{
    #pragma omp parallel for shared(A,B,C)
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            C[i*N+j%N] = 0;
            for (int k = 0; k < N; k++){
                C[i*N+j%N] += A[k*N+i%N]*B[j*N+k%N];
            }
        }
    }
}

std::string helpMsg = "Usage: ./dgemm <matrixSize> <iterations> <threadsNum>";
const double lowRand = 0.0;
const double highRand = 1.0;

std::random_device randomizer;
std::mt19937 seed(randomizer());

#define errorChk(ans, msg)                 \
    {                                      \
        if (!ans) {                        \
            std::cerr << msg << std::endl; \
            exit(1);                       \
        }                                  \
    }                                      \

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

bool verifyResults(int M, int N, int K,
                   const double *A, const double *B, const double *C,
                   const double accuracy = 0.001)
{
    double *correctResultMatrix = new double[M * N];
    errorChk(correctResultMatrix, "new failed!");

    /*
     * We assume that the most simple serial gemm implementation is correct.
     */
    gemmSerialVerifier(M, N, K, A, B, correctResultMatrix);

    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(correctResultMatrix[i] - C[i]) > accuracy) {
            delete[] correctResultMatrix;
            return false;
        }
    }

    delete[] correctResultMatrix;

    return true;
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
    double *B = new double[elementsInMatrix];
    errorChk(B, "new failed!");
    double *C = new double[elementsInMatrix];
    errorChk(C, "new failed!");

    double flop = 2.0 * static_cast<double>(matrixSize)
        * static_cast<double>(matrixSize)
        * static_cast<double>(matrixSize);

    randomizeMatrix(A, matrixSize, matrixSize, lowRand, highRand);
    randomizeMatrix(B, matrixSize, matrixSize, lowRand, highRand);
    std::fill(C, C + elementsInMatrix, 0);

    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl
              << "Matrix size: " << "(" << matrixSize << "x" << matrixSize << ")"
              << "X" << "("<< matrixSize << "x" << matrixSize << ")" << std::endl
              << "Compute iters: " << computeIters << std::endl
              << std::endl;

    auto startTime = omp_get_wtime();

    for (int i = 0; i < computeIters; ++i)
        gemm(matrixSize, matrixSize, matrixSize, A, B, C);

    auto endTime = omp_get_wtime();

    auto avgIterTime = (endTime - startTime) / static_cast<double>(computeIters);
    std::cout << "Sec/iter: " << avgIterTime << std::endl
              << "GFLOPS: " << flop / avgIterTime << std::endl
              << std::endl;

    std::cout << "Verifying results..." << std::endl;
    if (!verifyResults(matrixSize, matrixSize, matrixSize, A, B, C)) {
        std::cout << "Verification failed!" << std::endl;
    } else {
        std::cout << "Verification successful!" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
