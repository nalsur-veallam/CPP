#include <stdio.h>
#include <sys/time.h>
#include <iostream>

#define SIZE (4*4)
#define THREADS_PER_BLOCK 16

__global__ void mat_mul(float* dA, float* dB, float* dC, int N){
    int idx = blockIdx.y*blockDim.y+threadIdx.y;
    int idy = blockIdx.x*blockDim.x+threadIdx.x;

    float blockSum = 0;

    if (idx < N && idy < N){
        for (int i = 0; i < N; i++)
            blockSum += dA[idx*N + i] * dB[i*N + idy];
    }
    dC[idx*N + idy] = blockSum;
}



int main (int argc, char *argv[])
{
    int N = int(sqrtf(SIZE));
    float *hA, *hB, *hC, *dA, *dB, *dC;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Global memory
    hA = (float*) malloc (SIZE * sizeof(float));
    hB = (float*) malloc (SIZE * sizeof(float));
    hC = (float*) malloc (SIZE * sizeof(float));
    cudaMalloc((void**) &dA, SIZE * sizeof(float));
    cudaMalloc((void**) &dB, SIZE * sizeof(float));
    cudaMalloc((void**) &dC, SIZE * sizeof(float));

    cudaEventRecord(start, 0);
    
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    
    if (SIZE > THREADS_PER_BLOCK){
        threadsPerBlock.x = THREADS_PER_BLOCK;
        threadsPerBlock.y = THREADS_PER_BLOCK;
        blocksPerGrid.x = N/THREADS_PER_BLOCK;
        blocksPerGrid.y = N/THREADS_PER_BLOCK;
    }
    
    cudaMemcpy(dA, hA, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*SIZE, cudaMemcpyHostToDevice);

    mat_mul<<< blocksPerGrid, threadsPerBlock >>>(dA, dB, dC, N);
    cudaMemcpy(hC, dC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for CUDA GPU (global memory):  %3.1f ms \n", time);
    
    for(int i = 0; i < SIZE; i++)
        std::cout << hC[i] << std::endl;
    
    free(hA);
    free(hB);
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    return 0;
}
