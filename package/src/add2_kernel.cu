#include <torch/extension.h>
// #include <torch/extension.h>
#include <cuda_runtime.h>
// #include <iostream>
#include <stdint.h>
// #include "kernel.h"
// #include "cuda_runtime.h"
#include "unistd.h"
#include "iostream"
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

using namespace::std;

void launch_add2(float hitRatio,  uintptr_t stream_handle, at::Tensor base) {
// cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream_handle);
cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

int l2_cache_size;
cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
int num_bytes = 0;
num_bytes = min((int)(1.5 * base.numel() * base.element_size()), l2_cache_size);
hitRatio = max((float)1.0, (float)(num_bytes/ l2_cache_size));
cout<<"+++++++++++"<<endl<<num_bytes<<"  "<<l2_cache_size<<"  "<<hitRatio<<"  "<<endl;

cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(base.data_ptr()); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = 512;                    // Number of bytes for persisting accesses.
//                                                                               // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyPersisting;  // Type of access property on cache miss.
cudaError_t err = cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);    

std::cout << "Error setting stream attribute: " << cudaGetErrorString(err) << std::endl;
}