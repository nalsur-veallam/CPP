#include <stdio.h>
#include <sys/time.h>

#define SIZE (1024*1024)
#define TILE_WIDTH 32

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

void random_fill(float * hA, unsigned int N)
{
    for (int i = 0; i < N; ++i) {
        hA[i] = 1.0 * i;
    }
}

int main (int argc, char *argv[])
{
    int N = int(sqrtf(SIZE));
    float *hA, *hB, *hC, *dA, *dB, *dC;
    
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 blocksPerGrid((N / TILE_WIDTH) + 1, (N / TILE_WIDTH) + 1, 1);
    
    // Shared memory
    cudaMallocHost((void**) &hA, SIZE * sizeof(float));
    cudaMallocHost((void**) &hB, SIZE * sizeof(float));
    cudaMallocHost((void**) &hC, SIZE * sizeof(float));
    cudaMalloc( (void**) &dA, SIZE * sizeof(float));
    cudaMalloc((void**) &dB, SIZE * sizeof(float));
    cudaMalloc((void**) &dC, SIZE * sizeof(float));
    
    random_fill(hA, N*N);
    random_fill(hB, N*N);
    
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             printf("%10.3f %c", hA[i+4*j], ' ');
//         }
//         printf("\n");
//     }
//     printf("\n");
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             printf("%10.3f %c", hB[i+4*j], ' ');
//         }
//         printf("\n");
//     }
//     printf("\n");
    
    cudaMemcpy(dA, hA, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*SIZE, cudaMemcpyHostToDevice);

    mat_mul_shared<<< blocksPerGrid, threadsPerBlock >>>(dA, dB, dC, N);
    
    cudaThreadSynchronize();
    
    cudaMemcpy(hC, dC, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             printf("%10.3f %c", hC[i+4*j], ' ');
//         }
//         printf("\n");
//     }
    
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

