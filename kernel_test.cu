#include <iostream>   
#include <cstdlib>   
#include <iomanip>
#include <cuda_runtime.h>
#include <chrono>
using namespace std::chrono;
using namespace std;


/* 
    //Allocate mem on device (LOCATION, SIZE); address, number of bytes to allocate | Allocating memory on the GPU, and storing the ADD of that GPU memory into host pointer d_c.
    cudaMalloc((void**)&d_c, sizeof(int));

    //Args (dst, src, numBytes, direction) , (point2addcopyto,point2addcopyfrom, N*sizeof(type), cudaMemcpyHosttoDevice/DevicetoHost)
    cudaMemcpy(d_c, h_c, sizeof(int), cudaMemcpyHostToDevice);
 */



__global__ void increment_gpu(int *a, int N) {
    int i = threadIdx.x;
    if (i < N) {
        a[i] = a[i] + 1;
    }
}

void increment_cpu(int *a, int N) {
    for (int i=0; i<N; i++) {
        if (i < N) {
            a[i] = a[i] + 1;
        }
    }
}


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Using device: "<< prop.name << endl;
    // ---------------------------------------------------------------------------//
    int N = 1000;
    int *h_a = (int*)malloc(N * sizeof(int));
    int *d_a = nullptr;
    for (int i = 0; i < N; i++) h_a[i] = 0;
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid_size(1);
    dim3 block_size(N);

    auto start = high_resolution_clock::now();
    increment_cpu(h_a, N);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time elapsed: " << duration.count() << " us" << endl;

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);
    
    increment_gpu<<<grid_size, block_size>>>(d_a, N);

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=0; i < N; i++) {
        cout << "a[" << i << "] = " << h_a[i] << endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, gpu_start, gpu_stop);
    cout << "GPU time: " << milliseconds * 1000 << " us" << endl;

    free(h_a);
    cudaFree(d_a); 
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    return 0;
} 
