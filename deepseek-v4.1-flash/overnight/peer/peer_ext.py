"""Tiny CUDA helpers for cross-process GPU<->GPU signalling through pinned host memory."""
import os

import torch
from torch.utils.cpp_extension import load_inline

CUDA = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ unsigned long long ld_acquire(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Copy n bytes (multiple of 16) then publish value with release semantics.
__global__ void send_kernel(const int4* src, int4* dst, long n16, unsigned long long* flag, unsigned long long value) {
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = src[i];
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence_system(); st_release(flag, value); }
}

// Spin until flag >= value, then copy n bytes from src (host) to dst (device).
__global__ void recv_kernel(const int4* src, int4* dst, long n16, const unsigned long long* flag, unsigned long long value) {
  __shared__ int ok;
  if (threadIdx.x == 0) { while (ld_acquire(flag) < value) {} ok = 1; }
  __syncthreads();
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

// Emulate expert compute: stream `bytes` of device memory (weights) and reduce.
__global__ void stream_kernel(const int4* w, long n16, float* out) {
  float acc = 0.f;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
    int4 v = w[i]; acc += __int_as_float(v.x) + __int_as_float(v.w);
  }
  if (acc == 12345.f) out[0] = acc;
}

void send(int64_t src, int64_t dst, int64_t nbytes, int64_t flag, int64_t value, int64_t blocks) {
  auto s = c10::cuda::getCurrentCUDAStream();
  send_kernel<<<1, 256, 0, s>>>((const int4*)src, (int4*)dst, nbytes / 16, (unsigned long long*)flag, value);
}
void recv(int64_t src, int64_t dst, int64_t nbytes, int64_t flag, int64_t value, int64_t blocks) {
  auto s = c10::cuda::getCurrentCUDAStream();
  recv_kernel<<<1, 256, 0, s>>>((const int4*)src, (int4*)dst, nbytes / 16, (const unsigned long long*)flag, value);
}
void stream(int64_t w, int64_t nbytes, int64_t out, int64_t blocks) {
  auto s = c10::cuda::getCurrentCUDAStream();
  stream_kernel<<<blocks, 512, 0, s>>>((const int4*)w, nbytes / 16, (float*)out);
}
void host_register(int64_t ptr, int64_t nbytes) {
  TORCH_CHECK(cudaHostRegister((void*)ptr, nbytes, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess, "cudaHostRegister failed");
}
int64_t device_pointer(int64_t ptr) {
  void* d = nullptr;
  TORCH_CHECK(cudaHostGetDevicePointer(&d, (void*)ptr, 0) == cudaSuccess, "cudaHostGetDevicePointer failed");
  return (int64_t)d;
}
"""

CPP = """
void send(int64_t src, int64_t dst, int64_t nbytes, int64_t flag, int64_t value, int64_t blocks);
void recv(int64_t src, int64_t dst, int64_t nbytes, int64_t flag, int64_t value, int64_t blocks);
void stream(int64_t w, int64_t nbytes, int64_t out, int64_t blocks);
void host_register(int64_t ptr, int64_t nbytes);
int64_t device_pointer(int64_t ptr);
"""


def load(arch: str):
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return load_inline(
        name=f"peer_ext_{arch.replace('.', '')}", cpp_sources=CPP, cuda_sources=CUDA,
        functions=["send", "recv", "stream", "host_register", "device_pointer"],
        extra_cuda_cflags=["-O3"], verbose=False,
    )
