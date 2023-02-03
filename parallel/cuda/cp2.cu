#include <stdio.h>
#include <sys/time.h>
#include <iostream>

#define SIZE (4*4)
#define THREADS_PER_BLOCK 16
#define NSTREAMS 4
#define TILE_WIDTH 16

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

__global__ void mat_mul_shared(float* dA, float* dB, float* dC, int N){
    
//     __shared__ float dM[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float dN[TILE_WIDTH][TILE_WIDTH];
//     
//      int tx = threadIdx.x, ty = threadIdx.y;
//      int idx = blockIdx.y * TILE_WIDTH + ty;
//      int idy = blockIdx.x * TILE_WIDTH + tx;
//     float Pvalue = 0;
// 
//     for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
//        if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
//           ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
//        else
//           ds_M[ty][tx] = 0;
//        if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
//           ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
//        else
//           ds_N[ty][tx] = 0;
// 
//        __syncthreads();
//        for (int k = 0; k < TILE_WIDTH; ++k)
//           Pvalue += ds_M[ty][k] * ds_N[k][tx];
//        __syncthreads();
//     }
//     if (Row < numCRows && Col < numCColumns)
//        C[Row*numCColumns+Col] = Pvalue;
}

__global__ void mat_mul_async1(float* dA, float* dB, float* dC, int N, int id){
    int idx = blockIdx.y*blockDim.y+threadIdx.y;
    int idy = blockIdx.x*blockDim.x+threadIdx.x;

    float blockSum = 0;
    
    int STREAM_SIZE = N/NSTREAMS; 
    
    if (idx < N && idy < N && STREAM_SIZE%idx == STREAM_SIZE%idy == id){
        for (int i = 0; i < N; i++)
            blockSum += 2;//dA[idx*N + i] * dB[idy*N + i];
    }
    
    dC[idy*N + idx] = 2;
}

__global__ void mat_mul_async2(float* dA, float* dB, float* dC, int N, int id){
    int idx = blockIdx.y*blockDim.y+threadIdx.y;
    int idy = blockIdx.x*blockDim.x+threadIdx.x;

    float blockSum = 0;
    
    int STREAM_SIZE = N/NSTREAMS; 
    
    if (idx < N && idy < N && STREAM_SIZE%idx != STREAM_SIZE%idy && STREAM_SIZE%idx == id){
        for (int i = 0; i < N; i++)
            blockSum += dA[idx*N + i] * dB[idy*N + i];
    }
    
    dC[idy*N + idx] = blockSum;
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
    
    free(hA);
    free(hB);
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    // Pinned memory
    cudaMallocHost((void**) &hA, SIZE * sizeof(float));
    cudaMallocHost((void**) &hB, SIZE * sizeof(float));
    cudaMallocHost((void**) &hC, SIZE * sizeof(float));
    cudaMalloc( (void**) &dA, SIZE * sizeof(float));
    cudaMalloc((void**) &dB, SIZE * sizeof(float));
    cudaMalloc((void**) &dC, SIZE * sizeof(float));
    
    cudaEventRecord(start, 0);
    
    cudaMemcpy(dA, hA, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*SIZE, cudaMemcpyHostToDevice);

    mat_mul<<< blocksPerGrid, threadsPerBlock >>>(dA, dB, dC, N);
    cudaMemcpy(hC, dC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for CUDA GPU (pinned memory):  %3.1f ms \n", time);
    
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    // Pinned memory and Async    
    cudaMallocHost((void**) &hA, SIZE * sizeof(float));
    cudaMallocHost((void**) &hB, SIZE * sizeof(float));
    cudaMallocHost((void**) &hC, SIZE * sizeof(float));
    cudaMalloc( (void**) &dA, SIZE * sizeof(float));
    cudaMalloc((void**) &dB, SIZE * sizeof(float));
    cudaMalloc((void**) &dC, SIZE * sizeof(float));
    
    int STREAM_SIZE = int(SIZE/NSTREAMS);
    cudaStream_t stream[NSTREAMS];
    
    for (int i = 0; i < NSTREAMS; i++){
        cudaStreamCreate(&stream[i]);
    }
    
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < NSTREAMS; i++){
        int offset = i*STREAM_SIZE;
        cudaMemcpyAsync(&dA[offset], &hA[offset], sizeof(float)*STREAM_SIZE, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&dB[offset], &hB[offset], sizeof(float)*STREAM_SIZE, cudaMemcpyHostToDevice, stream[i]);
        mat_mul_async1<<< blocksPerGrid, threadsPerBlock, 0, stream[i] >>>(&dA[offset], &dB[offset], &dC[offset], N/NSTREAMS, i);
        mat_mul_async2<<< blocksPerGrid, threadsPerBlock, 0, stream[i] >>>(dA, dB, dC, N, i);
        cudaMemcpyAsync(&hC[offset], &dC[offset], sizeof(float)*STREAM_SIZE, cudaMemcpyDeviceToHost, stream[i]);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for CUDA GPU (pinned memory and async):  %3.1f ms \n", time);
    
    for(int i = 0; i < SIZE; i++)
        std::cout << hC[i] << std::endl;
    
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    for (int i = 0; i < NSTREAMS; i++){
        cudaStreamDestroy(stream[i]);
    }
    
    auto res = cudaGetLastError();
    if(res != cudaSuccess){
        printf("%s\n", cudaGetErrorString(res));
    }
    else{
        printf("OK \n");
    }
    
    return 0;
}

