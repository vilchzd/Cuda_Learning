#include <iostream>   
#include <cstdlib>   
#include <iomanip>
#include <cuda_runtime.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

__global__ void vector_add_gpu(float *a, float *b, float *c, int N) {
    int i = threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_cpu(float *a, float *b, float *c, int N) {
    cout << "--------------------------------------------" << endl;
    for (int i=0; i<N; i++) {
        if (i < N) {
            c[i] = a[i] + b[i];
        }
    
    }
    cout << "v_c[0] = " << c[0] << ", v_c[1] = " << c[1] << ", v_c[" << N-1 << "] = " << c[N-1] << " ...\n";
}


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Using device: "<< prop.name << endl;
    // ---------------------------------------------------------------------------//
    int N = 512;
    cout << "N= " << N << endl;
    float *h_va = (float*)malloc(N * sizeof(float));
    float *h_vb = (float*)malloc(N * sizeof(float));
    float *h_vc = (float*)malloc(N * sizeof(float));
    float *d_va, *d_vb, *d_vc;
    for (int i = 0; i < N; i++) {
        if (i == 0) {
            h_va[i] = 0;
        }
        h_va[i] = i;
        h_vb[i] = (2.0f + i)/3.0f;
        h_vc[i] = 0;
    }
    cudaMalloc((void**)&d_va, N*sizeof(float));
    cudaMalloc((void**)&d_vb, N*sizeof(float));
    cudaMalloc((void**)&d_vc, N*sizeof(float));
    cudaMemcpy(d_va, h_va, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, h_vb, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vc, h_vc, N*sizeof(float), cudaMemcpyHostToDevice);


    dim3 grid_size(1);
    dim3 block_size(N);

    auto start = high_resolution_clock::now();
    vector_add_cpu(h_va, h_vb, h_vc, N);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time elapsed: " << duration.count() << " us" << endl;
  
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);
    
    vector_add_gpu<<<grid_size, block_size>>>(d_va, d_vb, d_vc, N);

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    cudaMemcpy(h_vc, d_vc, N*sizeof(float), cudaMemcpyDeviceToHost);
    cout << "--------------------------------------------" << endl;
    cout << "v_c[0] = " << h_vc[0] << ", v_c[1] = " << h_vc[1] << ", v_c[" << N-1 << "] = " << h_vc[N-1] << " ...\n";

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, gpu_start, gpu_stop);
    cout << "GPU time: " << milliseconds * 1000 << " us" << endl;

    free(h_va);
    free(h_vb);
    free(h_vc);
    cudaFree(d_va); 
    cudaFree(d_vb); 
    cudaFree(d_vc);

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    return 0;
} 
