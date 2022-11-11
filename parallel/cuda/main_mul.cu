#include <stdio.h>
#include <sys/time.h>

#define SIZE (4096*4096)
#define THREADS_PER_BLOCK 32
#define NSTREAMS 2
#define TILE_WIDTH 64

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
    
    __shared__ float dM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float dN[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = blockIdx.y * TILE_WIDTH + ty;
    int idy = blockIdx.x * TILE_WIDTH + tx;
    float blockSum = 0;

    for (int i = 0; i < (N-1)/TILE_WIDTH+1; i++) {
       if (idx < N && i*TILE_WIDTH + tx < N)
          dM[ty][tx] = dA[idx*N + i*TILE_WIDTH + tx];
       else
          dM[ty][tx] = 0;
       if (idy < N && i*TILE_WIDTH + ty < N)
          dN[ty][tx] = dB[(i*TILE_WIDTH + ty)*N + idy];
       else
          dN[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; k++)
          blockSum += dM[ty][k] * dN[k][tx];
       __syncthreads();
    }
    if (idx < N && idy < N)
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
    
    dim3 threadsPerBlock(N, N, 1);
    dim3 blocksPerGrid(1, 1, 1);
    
    if (SIZE > THREADS_PER_BLOCK){
        threadsPerBlock.x = THREADS_PER_BLOCK;
        threadsPerBlock.y = THREADS_PER_BLOCK;
        threadsPerBlock.z = 1;
        blocksPerGrid.x = N/THREADS_PER_BLOCK;
        blocksPerGrid.y = N/THREADS_PER_BLOCK;
        blocksPerGrid.z = 1;
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
    int STREAM_SIZE = int(SIZE/NSTREAMS);
    cudaStream_t stream[NSTREAMS];
    
    blocksPerGrid.x = N/THREADS_PER_BLOCK/NSTREAMS;
    blocksPerGrid.y = N/THREADS_PER_BLOCK/NSTREAMS;
    
    for (int i = 0; i < NSTREAMS; i++){
        cudaStreamCreate(&stream[i]);
    }
    
    cudaMallocHost((void**) &hA, SIZE * sizeof(float));
    cudaMallocHost((void**) &hB, SIZE * sizeof(float)); // Matrix B transposed
    cudaMallocHost((void**) &hC, SIZE * sizeof(float));
    cudaMalloc( (void**) &dA, SIZE * sizeof(float));
    cudaMalloc((void**) &dB, SIZE * sizeof(float));
    cudaMalloc((void**) &dC, SIZE * sizeof(float));
    
    
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < NSTREAMS; i++){
        int offset = i*STREAM_SIZE;
        cudaMemcpyAsync(dA + (int)offset, hA + (int)offset, sizeof(float)*STREAM_SIZE, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&dB[offset], &hB[offset], sizeof(float)*STREAM_SIZE, cudaMemcpyHostToDevice, stream[i]);
    }
    
    cudaDeviceSynchronize();
    
    mat_mul<<< blocksPerGrid, threadsPerBlock >>>(dA, dB, dC, N);
    
    for (int i = 0; i < NSTREAMS; i++){
        int offset = i*STREAM_SIZE;
        cudaMemcpyAsync(hC + (int)offset, dC + (int)offset, sizeof(float)*STREAM_SIZE, cudaMemcpyDeviceToHost, stream[i]);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for CUDA GPU (pinned memory and async):  %3.1f ms \n", time);
    
    for (int i = 0; i < NSTREAMS; i++){
        cudaStreamDestroy(stream[i]);
    }
    
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);    
    
    // Shared memory
    cudaMallocHost((void**) &hA, SIZE * sizeof(float));
    cudaMallocHost((void**) &hB, SIZE * sizeof(float));
    cudaMallocHost((void**) &hC, SIZE * sizeof(float));
    cudaMalloc( (void**) &dA, SIZE * sizeof(float));
    cudaMalloc((void**) &dB, SIZE * sizeof(float));
    cudaMalloc((void**) &dC, SIZE * sizeof(float));
    
    blocksPerGrid.x = (N-1)/TILE_WIDTH+1;
    blocksPerGrid.y = (N-1)/TILE_WIDTH+1;
    threadsPerBlock.x = TILE_WIDTH;
    threadsPerBlock.y = TILE_WIDTH;
    
    cudaEventRecord(start, 0);
    
    cudaMemcpy(dA, hA, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*SIZE, cudaMemcpyHostToDevice);

    mat_mul_shared<<< blocksPerGrid, threadsPerBlock >>>(dA, dB, dC, N);
    
    cudaThreadSynchronize();
    
    cudaMemcpy(hC, dC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for CUDA GPU (shared memory):  %3.1f ms \n", time);
    
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    auto res = cudaGetLastError();
    if(res != cudaSuccess){
        printf("%s\n", cudaGetErrorString(res));
    }
    else{
        printf("OK \n");
    }
    
    return 0;
}

