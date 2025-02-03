#include <cuda.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    }

__global__ 
void vec_add_device(float *a, float *b, float *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
        printf("GPU: c[%d] = %f (a[%d] = %f, b[%d] = %f)\n", index, c[index], index, a[index], index, b[index]);
    }
}

void vec_add_host(float *a, float *b, float *c, int n) {
    cudaDeviceReset();

    float *d_a, *d_b, *d_c;
    int size = n * sizeof(float);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("CUDA Bellek: %ld MB boş, %ld MB toplam\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vec_add_device<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("host c[%d] = %f\n", i, c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    float a[1000], b[1000], c[1000];
    int n = 1000;
    
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = (3 * i * i);
    }
    
    vec_add_host(a, b, c, n);
}
