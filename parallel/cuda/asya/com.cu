#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <iostream>

using namespace std;


#define CUDA_FLOAT float
#define BLOCK_SIZE 512 
#define GRID_SIZE 2048
#define THREAD_SIZE 4 //number of trapezoids in one block, 
                      //so we don't count the same boundary value 2 times


__global__ void kern(CUDA_FLOAT *den, CUDA_FLOAT *nom)
{
int n = threadIdx.x + blockIdx.x * BLOCK_SIZE; //single thread index
CUDA_FLOAT x0 = n * 1.f / (BLOCK_SIZE * GRID_SIZE); //left bound of trapezoid
CUDA_FLOAT y0 = sqrtf(1 - x0 * x0); //left value of trapezoid for nominator
CUDA_FLOAT z0 = x0*sqrtf(1 - x0 * x0); //left value of trapezoid for denominator
CUDA_FLOAT dx = 1.f / (1.f * BLOCK_SIZE * GRID_SIZE * THREAD_SIZE); //integration step
CUDA_FLOAT den_s = 0; //sum of block of denominator trapezoids
CUDA_FLOAT nom_s = 0; //sum of block of nomenitor trapezoids
CUDA_FLOAT x1, y1, z1;
for (int i=0; i < THREAD_SIZE; ++i)
{
x1 = x0 + dx; //right bound of trapezoid
y1 = sqrtf(1 - x1 * x1); //right value of trapezoid for denomenitor
den_s += (y0 + y1) * dx / 2.f; //trapezoid area for denomenitor
z1 = x1*sqrtf(1 - x1 * x1); //right value of trapezoid for nomenitor
nom_s += (z0 + z1) * dx / 2.f; //trapezoid area for nomenitor
y0 = y1;
x0 = x1;
}
den[n] = den_s; //write result into the global memory
nom[n] = nom_s;
}


CUDA_FLOAT KahanSum(CUDA_FLOAT* input, int inputsize) {
    CUDA_FLOAT sum = 0.0;
    CUDA_FLOAT c = 0.0;
    for(int i = 1; i < inputsize; i++){
        CUDA_FLOAT y = input[i] - c;
        CUDA_FLOAT t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}


int main(int argc, char **argv)
{   
    printf("Result is ");
    CUDA_FLOAT* nom; //host memory pointer
    CUDA_FLOAT* den;
    CUDA_FLOAT* d_nom; //device memory pointer
    CUDA_FLOAT* d_den;
    CUDA_FLOAT nom_S; //meaning of integral for nomenator
    CUDA_FLOAT den_S; //meaning of integral for denomenator
    nom = (CUDA_FLOAT*)malloc (GRID_SIZE*BLOCK_SIZE*sizeof(CUDA_FLOAT)); //allocating host memory
    den = (CUDA_FLOAT*)malloc (GRID_SIZE*BLOCK_SIZE*sizeof(CUDA_FLOAT));
    cudaMalloc ((void **) &d_nom, GRID_SIZE*BLOCK_SIZE*sizeof(CUDA_FLOAT)); //alocating device memory
    cudaMalloc ((void **) &d_den, GRID_SIZE*BLOCK_SIZE*sizeof(CUDA_FLOAT));
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = dim3(GRID_SIZE);
    kern<<<grid, block>>> (d_den, d_nom);
    cudaMemcpy(nom, d_nom, GRID_SIZE*BLOCK_SIZE*sizeof( CUDA_FLOAT ),cudaMemcpyDeviceToHost );
    cudaMemcpy(den, d_den, GRID_SIZE*BLOCK_SIZE*sizeof( CUDA_FLOAT ),cudaMemcpyDeviceToHost );
    nom_S = KahanSum(nom, GRID_SIZE*BLOCK_SIZE);
    den_S = KahanSum(den, GRID_SIZE*BLOCK_SIZE);
    cout << nom_S/den_S << '\n';
    //accurate meaning is 0.42441318
    return 0;
}
