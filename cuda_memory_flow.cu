#include <iostream>   
#include <cstdlib>   
#include <iomanip>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel: runs on GPU
__global__ void kernelName1() {
    printf("Hello from GPU!\n");
}

__global__ void kernelName2(int *ptr) {
    printf("Changing contents of host pointer!\n");
    *ptr = 100;
}


int main() {
    //Host and Device pointer variables that hold memory addresses
    int *h_c, *d_c;
    //Allocate mem on host
    h_c = (int*)malloc(sizeof(int));
    *h_c = 200; 
    //Allocate mem on device (LOCATION, SIZE); address, number of bytes to allocate | Allocating memory on the GPU, and storing the ADD of that GPU memory into host pointer d_c.
    cudaMalloc((void**)&d_c, sizeof(int));

    //Copy data from host to device
    //Args (dst, src, numBytes, direction) , (point2addcopyto,point2addcopyfrom,N*sizeof(type), cudaMemcpyHosttoDevice/DevicetoHost)
    cudaMemcpy(d_c, h_c, sizeof(int), cudaMemcpyHostToDevice);
    cout << "Host pointer value before kernel launch: " << *h_c << endl;
    
    //Launch kernel
    dim3 grid_size(1,1);
    dim3 block_size(1,1);
     kernelName1<<<grid_size, block_size>>>();
     kernelName2<<<grid_size, block_size>>>(d_c);


    // Wait for GPU to finish
    cudaDeviceSynchronize();



    //Copy data from device to host
    cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Print from CPU
    cout << "Host pointer address: 0x" << hex << reinterpret_cast<uintptr_t>(h_c) << endl;
    cout << "Host pointer value after kernel launch: " << dec << *h_c << endl;
    cudaFree(d_c);
    free(h_c);
        
    return 0;
}
