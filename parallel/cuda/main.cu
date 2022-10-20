#include <stdio.h>

#define N 100*1024*1024

__global__ void fx(float* dA){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	dA[idx] = sinf(sqrtf(2.0*3.14*(float)idx/(float)N));
}

void fx_cpu(float* hA){
	for (int idx = 0; idx < N; idx++)
	    hA[idx] = sinf(sqrtf(2.0*3.14*(float)idx/(float)N));
}

int main(){
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *hA, *dA;

	// Global memory
	hA = (float*) malloc(sizeof(float)*N);
	cudaMalloc( (void**) &dA, sizeof(float)*N);
	
	cudaEventRecord(start, 0);	
	
	fx <<<N/512, 512>>> (dA);
	

	cudaMemcpy(hA, dA, sizeof(float)*N, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start,stop);
	printf("Time for GPU Global = %3.1f ms\n", time);
	cudaFree(dA);
	free(hA);
	
	// Pinned memory
	cudaMallocHost((void**) &hA, sizeof(float)*N);
	cudaMalloc( (void**) &dA, sizeof(float)*N);
	
	cudaEventRecord(start, 0);	
	
	fx <<<N/512, 512>>> (dA);
	

	cudaMemcpy(hA, dA, sizeof(float)*N, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start,stop);
	printf("Time for GPU Pinned = %3.1f ms\n", time);
	
	cudaFree(dA);
	cudaFreeHost(hA);
	
	// UVA memory
	cudaMallocManaged((void**) &hA, sizeof(float)*N);
	
	cudaEventRecord(start, 0);	
	
	fx <<<N/512, 512>>> (hA);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start,stop);
	printf("Time for GPU UVA = %3.1f ms\n", time);
	
	cudaEventRecord(start, 0);	
	
	fx_cpu (hA);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start,stop);
	printf("Time for CPU UVA = %3.1f ms\n", time);
    
	cudaFree(hA);
	
	// Serial Method
	
	hA = (float*) malloc(sizeof(float)*N);
	cudaEventRecord(start, 0);	
	
	fx_cpu(hA);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start,stop);
	printf("Time for CPU Serial= %3.1f ms\n", time);
	free(hA); 
	return 0;
}
