#pragma once

#include <iostream>
#include "cuda_runtime.h"
#include <cub/cub.cuh>

#define CUDA_HELPER_STRINGIFY(x) #x
// do while statement ensures that macro behaves like a single statement
#define CHECK_CUDA_ERROR(cuda_call)                                                                                    \
    do {                                                                                                               \
        cudaError_t cuda_status = cuda_call;                                                                           \
        if (cuda_status != cudaSuccess) {                                                                              \
            std::cerr << "CUDA error in " << CUDA_HELPER_STRINGIFY(cuda_call) << ": "                                  \
                      << cudaGetErrorString(cuda_status) << std::endl;                                                 \
            std::terminate();                                                                                          \
        }                                                                                                              \
    } while (0)

namespace pattern
{
namespace cuda
{

template <class T> inline T* malloc(size_t count) {
    T* ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

inline void ExclusiveSum(void* dst, void* src, size_t count) {
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dst, count);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, src, dst, count);
    cudaFree(d_temp_storage);
}

inline void InclusiveSum(void* dst, void* src, size_t count) {
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, src, dst, count);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, src, dst, count);
    cudaFree(d_temp_storage);
}

template <class T> inline void memset(T* dst, T value, size_t count) {
    CHECK_CUDA_ERROR(cudaMemset(dst, value, count * sizeof(T)));
}

template <class T> inline void memcpy(T* dst, T* src, size_t count, cudaMemcpyKind kind) {
    CHECK_CUDA_ERROR(cudaMemcpy(dst, src, count * sizeof(T), kind));
}

// Copy from host to device
template <class T> inline void memcpy_host_dev(T* dst, T* src, size_t count) {
    memcpy<T>(dst, src, count, cudaMemcpyHostToDevice);
}

// Copy from device to host
template <class T> inline void memcpy_dev_host(T* dst, T* src, size_t count) {
    memcpy<T>(dst, src, count, cudaMemcpyDeviceToHost);
}

inline void free(void* dev_ptr) {
    CHECK_CUDA_ERROR(cudaFree(dev_ptr));
}

} // namespace cuda
} // namespace pattern
