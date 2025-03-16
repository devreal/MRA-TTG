#ifndef MRA_DEVICE_PLATFORM_H
#define MRA_DEVICE_PLATFORM_H

#include <cstdlib>
#include <algorithm>

#if defined(MRA_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif defined(MRA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

/**
 * Utilities to achieve platform independence.
 * Provides wrappers for CUDA/HIP constructs to compile code
 * for both host and devices.
 */

namespace mra::detail {
  struct dim3
  {
      unsigned int x, y, z;
      constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz)
      { }
  };
} // namespace mra::detail

/* convenience macro to mark functions __device__ if compiling for CUDA */
#if defined(__CUDA_ARCH__)
#define SCOPE __device__ __host__
#define SYNCTHREADS() __syncthreads()
#define DEVSCOPE __device__
#define SHARED __shared__
#define LAUNCH_BOUNDS(__NT) __launch_bounds__(__NT)
#define HAVE_DEVICE_ARCH 1
#elif defined(__HIP__)
#define SCOPE __device__ __host__
#define SYNCTHREADS() __syncthreads()
#define DEVSCOPE __device__
#define SHARED __shared__
#define LAUNCH_BOUNDS(__NT) __launch_bounds__(__NT)
#define HAVE_DEVICE_ARCH 1
#else // __CUDA_ARCH__
#define SCOPE
#define SYNCTHREADS() do {} while(0)
#define DEVSCOPE inline
#define SHARED
#define LAUNCH_BOUNDS(__NT)
#endif // __CUDA_ARCH__

#if defined(__CUDACC__)
#define checkSubmit() \
  if (cudaPeekAtLastError() != cudaSuccess)                         \
    std::cout << "kernel submission failed at " << __FILE__ << ":" << __LINE__ << ": " \
    << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
#define CALL_KERNEL(name, block, thread, shared, stream, args) \
  name<<<block, thread, shared, stream>>> args
#elif defined(__HIPCC__)
#define checkSubmit() \
  if (hipPeekAtLastError() != hipSuccess)                           \
    std::cout << "kernel submission failed at " << __FILE__ << ":" << __LINE__ << ": " \
    << hipGetErrorString(hipPeekAtLastError()) << std::endl;
#define CALL_KERNEL(name, block, thread, shared, stream, args) \
  name<<<block, thread, shared, stream>>> args
#else  // __CUDACC__
#define checkSubmit() do {} while(0)
#define CALL_KERNEL(name, blocks, thread, shared, stream, args) \
  do { \
    blockIdx = {0, 0, 0};                       \
    for (std::size_t i = 0; i < blocks; ++i) {  \
      blockIdx.x = i;                           \
      name args;                                \
    }                                           \
  } while (0)
#endif // __CUDACC__

#if defined(__CUDA_ARCH__)
#define THROW(s) do { std::printf(s); __trap(); } while(0)
#elif defined(__HIP__)
/* TODO: how to error out on HIP? */
#define THROW(s) do { printf(s); } while(0)
#else  // __CUDA_ARCH__
#define THROW(s) do { throw std::runtime_error(s); } while(0)
#endif // __CUDA_ARCH__

#if defined(__CUDACC__)
#define GLOBALSCOPE __global__
#elif defined(__HIPCC__)
#define GLOBALSCOPE __global__
#else  // __CUDA_ARCH__
#define GLOBALSCOPE
#endif // __CUDA_ARCH__

#if defined(MRA_ENABLE_HOST)
#define MAX_THREADS_PER_BLOCK 1
#else
#define MAX_THREADS_PER_BLOCK 1024
#endif

#if defined(MRA_ENABLE_HOST)
using Dim3 = mra::detail::dim3;
#else
using Dim3 = dim3;
#endif // HAVE_DEVICE_ARCH

// current have to make sure we use scope::SyncIn for host
// this will be fixed once TTG supports coros for host tasks
#if defined(MRA_ENABLE_HOST)
#define TempScope ttg::scope::SyncIn
#else
#define TempScope ttg::scope::Allocate
#endif


#if defined(MRA_ENABLE_HOST)
namespace mra {
  /* define our own thread layout (single thread) */
  static constexpr const mra::detail::dim3 threadIdx = {0, 0, 0};
  static constexpr const mra::detail::dim3 blockDim  = {1, 1, 1};
  static constexpr const mra::detail::dim3 gridDim   = {1, 1, 1};
  inline thread_local    mra::detail::dim3 blockIdx  = {0, 0, 0};
} // namespace mra
#endif // MRA_ENABLE_HOST

/**
 * Function returning the thread ID in a flat ID space.
 */
namespace mra {
  DEVSCOPE int thread_id() {
#if defined(HAVE_DEVICE_ARCH)
    return blockDim.x * ((blockDim.y * threadIdx.z) + threadIdx.y) + threadIdx.x;
#else  // HAVE_DEVICE_ARCH
    return 0;
#endif // HAVE_DEVICE_ARCH
  }

  DEVSCOPE int block_size() {
#if defined(HAVE_DEVICE_ARCH)
    return blockDim.x * blockDim.y * blockDim.z;
#else  // HAVE_DEVICE_ARCH
    return 1;
#endif // HAVE_DEVICE_ARCH
  }

  SCOPE inline bool is_team_lead() {
#if defined(HAVE_DEVICE_ARCH)
    return (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
#else  // HAVE_DEVICE_ARCH
    return true;
#endif // HAVE_DEVICE_ARCH
  }

  constexpr inline Dim3 max_thread_dims(int K) {
    return Dim3(K, K, std::min(((MAX_THREADS_PER_BLOCK) / (K*K)), K));
  }

  constexpr inline int max_threads(int K) {
    Dim3 thread_dims = max_thread_dims(K);
    return thread_dims.x*thread_dims.y*thread_dims.z;
  }

} // namespace mra

namespace std {
  inline std::ostream& operator<<(std::ostream& os, Dim3& d) {
    os << "{" << d.x << ", " << d.y << ", " << d.z << "}";
    return os;
  }
} // namespace std


#endif // MRA_DEVICE_PLATFORM_H
