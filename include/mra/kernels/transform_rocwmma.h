#ifndef HAVE_TRANSFORM_ROCWMMA_H
#define HAVE_TRANSFORM_ROCWMMA_H

#include "mra/misc/platform.h"
#include "mra/ops/mxm.h"
//

#define ROCWMMA_NUM_THREADS 512

#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>


namespace mra {

namespace detail {

#if 0
template <size_type K, typename T>
__device__ void transform_klt16(
    const T* a,
    const T* b,
    T*& c)
{
  /* hold everything in shared memory */
  extern __shared__ char smem[];
  T* shmem = reinterpret_cast<T*>(smem);
  T* b_shmem = shmem;
  T* a_shmem = b_shmem + K * K;
  T* c_shmem = a_shmem + K * K * K;

  const size_type tid = thread_id();
  const size_type num_threads = block_size();

  /* load A and B into shared memory */
  for (int idx = tid; idx < K * K; idx += num_threads) {
    b_shmem[idx] = b[idx];
  }
  for (int idx = tid; idx < K * K * K; idx += num_threads) {
    a_shmem[idx] = a[idx];
  }
  __syncthreads();

  for (int d = 0; d < 3; ++d) {
    /* compute c = a * b, with c also in shared memory */
    for (int i = tid/K; i < K * K; i += num_threads/K) {
      T* ci = c_shmem + i * K;
      int j = tid % K;
      T sum = 0;
      for (long k = 0; k < K; ++k) { /* not parallelized */
        sum += a_shmem[k * K * K + i] * b_shmem[k * K + j];
      }
      if (d == 0) {
        ci[j] = sum;
      } else {
        ci[j] += sum;
      }
    }
    __syncthreads();

    /* swap A and C for the next iteration, so we always read from A and write to C */
    std::swap(a_shmem, c_shmem);
   }

   // write back result to global memory
   for (int idx = tid; idx < K * K * K; idx += num_threads) {
     c[idx] = a_shmem[idx]; // a_shmem is the final result after 3 iterations
   }

}
       //

  template <Dimension NDIM, size_type K, typename T>
  SCOPE void transform_klt16(
    const T* t,
    const T* c,
    T* result,
    T* workspace) {
    const T* pc = c;
    T *t0=workspace, *t1=result;
    if (NDIM & 0x1) std::swap(t0,t1);
    const size_type dimj = K;
    size_type dimi = 1;
    for (size_type n=1; n<NDIM; ++n) dimi *= dimj;
    mTxmq(dimi, dimj, dimj, t0, t, pc);
    for (size_type n=1; n<NDIM; ++n) {
      mTxmq(dimi, dimj, dimj, t1, t0, pc);
      std::swap(t0,t1);
    }
    /* no need to synchronize here, mTxmq synchronizes */
  }

#endif // 0

/**
 * This implementation only works on K=16. For other K values, we fall back to the Level-3 implementation.
 * The fragment size is 16x16x16.
 * The block dimension is 256 threads (one wavefront) to match the MFMA requirements.
 * We load B into a fragment and keep it there.
 * We load A into fragments. Each wave-front stores 4 input fragments and 4 output fragments.
 *
 */
template <size_type K, typename T>
__device__ void transform_rocwmma_k(
    const T* a,
    const T* b,
    T*& c,
    T* workspace)
{
  constexpr uint32_t WM = 16, WN = 16, WK = 16;
  constexpr uint32_t WAVE = 64;   // CDNA wavefront size
  constexpr const int ndim = 3; // fixed for benchmark

  using FragmentA = rocwmma::fragment<rocwmma::matrix_a, K, K, K, T, rocwmma::col_major>;
  using FragmentB = rocwmma::fragment<rocwmma::matrix_b, K, K, K, T, rocwmma::row_major>;
  using FragmentAcc = rocwmma::fragment<rocwmma::accumulator, K, K, K, T, rocwmma::row_major>;

  if constexpr (K < 16) {
    // Fallback to non mma implementation
    transform_klt16<3, K, T>(a, b, c, workspace);
    //transform(a, b, c, workspace);
    return;
  } else if constexpr (K > 16) {
    // Not supported, fallback to Level-3
    //transform_level3_k<T, K>(a, b, c, workspace);
    printf("WTF dude!");
    return;
  } else {

    /* single shared memory region, holds A and C */
    extern __shared__ char smem[];
    T* shmem = reinterpret_cast<T*>(smem);

    int wave_id = thread_id() / WAVE;
    constexpr int num_waves = (ROCWMMA_NUM_THREADS / WAVE);
    constexpr int frags_per_wave = (K / num_waves);

    // load b into a fragment
    FragmentB b_frag;
    rocwmma::load_matrix_sync(b_frag, b, K);

    /* load A into shared memory */
    for (int idx = thread_id(); idx < K * K; idx += block_size()) {
      shmem[idx] = a[idx];
    }
    __syncthreads();

    /* every wavefront handles 4 fragments */
    FragmentA a_frags[frags_per_wave];
    FragmentAcc acc_frags[frags_per_wave];

    for (int d = 0; d < ndim; ++d) {
      /* load all wavefront fragments */
      for (int i = 0; i < frags_per_wave; ++i)
      {
        /* load the current fragment */
        if (i < frags_per_wave - 1 || frags_per_wave == 1) {
          const T* c_ptr = (d == 0) ? a : shmem;
          rocwmma::load_matrix_sync(a_frags[i], c_ptr + (i + wave_id * frags_per_wave) * K, K*K);
          // TODO: is it worth prefetching the next fragment?
          if constexpr (frags_per_wave > 1) {
            rocwmma::load_matrix_sync(a_frags[i+1], c_ptr + (i+1 + wave_id * frags_per_wave) * K, K*K);
          }
        }
        rocwmma::fill_fragment(acc_frags[i], static_cast<T>(0));
        rocwmma::mma_sync(acc_frags[i], a_frags[i], b_frag, acc_frags[i]);
      }

      /* write back all fragments */
      if (d == ndim - 1) {
        /* last iteration, write back to global memory */
        for (int i = 0; i < frags_per_wave; ++i)
        {
          rocwmma::store_matrix_sync(c + (i + wave_id * frags_per_wave) * K * K,
                                    acc_frags[i], K);
        }
      } else {
        /* wait for all fragments to be loaded from shared memory */
        rocwmma::synchronize_workgroup();
        /* write back to shared memory */
        for (int i = 0; i < frags_per_wave; ++i)
        {
          rocwmma::store_matrix_sync(shmem + (i + wave_id * frags_per_wave) * K * K,
                                    acc_frags[i], K);
        }
      }

      rocwmma::synchronize_workgroup();
    }
  }
}

} // namespace detail

template<typename T>
__device__ bool transform_shared(
  int K,
  const T* a,
  const T* b,
  T* c,
  T* workspace)
{
  switch(K) {
#if 0
    case  4:
	    detail::transform_rocwmma_k<4, float>(a, b, c, workspace);
      break;
    case  8:
      detail::transform_rocwmma_k<8, float>(a, b, c, workspace);
      break;
    case 10:
      detail::transform_rocwmma_k<10, float>(a, b, c, workspace);
      break;
    case 12:
      detail::transform_rocwmma_k<12, float>(a, b, c, workspace);
      break;
#endif // 0
    case 16:
      detail::transform_rocwmma_k<16>(a, b, c, workspace);
      break;
    default:
      //printf("WARNING: transform_rocwmma does not support K=%d, falling back to reference implementation\n", K);
      return false;
  }

  return true;
}

} // namespace mra
  //
#else 

namespace mra {
template<typename T>
__device__ bool transform_shared(
  int K,
  const T* a,
  const T* b,
  T* c,
  T* workspace) {
  return false;
}
}

#endif // __HIP_DEVICE_COMPILE__
       //

namespace mra {

  inline Dim3 transform_blockdim(int K) {
    return {ROCWMMA_NUM_THREADS, 1, 1};
  }


  template<typename T>
  constexpr size_type transform_shmem_size(size_type K) {
    if (K <= 16) {
      return K*K*K*sizeof(T);
    } else {
      // For K > 16, we fall back to the Level-3 implementation, which doesn't use shared memory.
      return 0;
    }
  }
}

#undef ROCWMMA_NUM_THREADS

#define MRA_HAVE_TRANSFORM_SHARED 1


#endif // HAVE_TRANSFORM_ROCWMMA_H
