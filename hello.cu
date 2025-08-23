#include <iostream>   
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel: runs on GPU
__global__ void helloGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    // Launch the kernel with 1 block and 1 thread
    helloGPU<<<1, 1>>>();
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Print from CPU
    cout << "Hello from CPU!\n" << endl;

    return 0;
}
