// int i = threadIdx.x + blockDim.x * blockIdx.x
#include <iostream>   
#include <cstdlib>   
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

__global__ void display_threads1(float *a) {
    int i = threadIdx.x;
    if (i < 10) {
        a[i] = i;
    }
}

__global__ void display_threads2(float *a) {
    int i = threadIdx.x + blockIdx.x;
    if (i < 10) {
        a[i] = i;
    }
}

__global__ void display_threads3(float *a) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < 10) {
        a[i] = i;
    }
}


int main() {
    float *d_a, *d_b, *d_c;
    float *h_a = (float*)malloc(10*sizeof(float));
    float *h_b = (float*)malloc(10*sizeof(float));
    float *h_c = (float*)malloc(10*sizeof(float));

    for (int i = 0; i < 10; i++) {
        h_a[i] = 0;
        h_b[i] = 0;
        h_c[i] = 0;

    }
    cudaMalloc((void**)&d_a, 10*sizeof(float));
    cudaMemcpy(d_a, h_a, 10*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_b, 10*sizeof(float));
    cudaMemcpy(d_b, h_b, 10*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_c, 10*sizeof(float));
    cudaMemcpy(d_c, h_c, 10*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_size(3);
    dim3 block_size(3);
    display_threads1<<<grid_size, block_size>>>(d_a);
    display_threads2<<<grid_size, block_size>>>(d_b);
    display_threads3<<<grid_size, block_size>>>(d_c);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_a, d_a, 10*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, 10*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, 10*sizeof(float), cudaMemcpyDeviceToHost);

    cout << "\nResults from display_threads1:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "h_a[" << i << "]: " << h_a[i] << endl; 
    }
    cout << "\nResults from display_threads2:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "h_b[" << i << "]: " << h_b[i] << endl; 
    }
    cout << "\nResults from display_threads3:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "h_c[" << i << "]: " << h_c[i] << endl; 
    }


    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
} 
