#ifndef MRA_MXM_H
#define MRA_MXM_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

namespace mra{


  namespace detail {
    /* mTxm that blocks on A (i.e., dimi is the largest dimension )*/
    template <typename aT, typename bT, typename cT, bool Q = false>
    SCOPE bool mTxm_block_a(size_type dimi, size_type dimj, size_type dimk,
                            cT* __restrict__ c, const aT* a, const bT* b) {

      auto tid = thread_id();
      SHARED aT block_a[MAX_THREADS_PER_BLOCK];
      SHARED cT block_c[MAX_THREADS_PER_BLOCK];

      size_type a_block_dimi;
      a_block_dimi = block_size() / dimk;

      /* A is transposed and we want to coalesce in dimi */
      auto a_trans_i = tid % a_block_dimi; // index in transposed block A(k, i)
      auto a_trans_k = tid / a_block_dimi;
      for (size_type i = 0; i < dimi; i += a_block_dimi) {

        if (i+a_trans_i < dimi && threadIdx.x < dimj) { /* in case dimi is not a multiple of Y*Z */
          /* transpose block of A into shared memory */
          block_a[a_trans_k*a_block_dimi + a_trans_i] = a[a_trans_k*dimi + i+a_trans_i];
        }

        /* make sure the block is written */
        SYNCTHREADS();

        cT *c_i = c + i*dimj;

        if (i+a_trans_i < dimi && threadIdx.x < dimj) { /* in case dimi is not a multiple of Y*Z */
          size_type c_idx = (threadIdx.z*blockDim.y+threadIdx.y)*dimj + threadIdx.x; // works!
          cT sum = 0.0;
          /* k is not parallel */
          for (size_type k = 0; k < dimk; ++k) {
            aT a_ = block_a[k*a_block_dimi+(threadIdx.z*blockDim.y+threadIdx.y)];
            bT b_ = b[k*dimj + threadIdx.x];
            sum += a_ * b_;
          }
          block_c[c_idx] = sum;
        }

        SYNCTHREADS();
        /* copy shared memory block to global memory */
        if (tid < a_block_dimi*dimj) {
          if constexpr (Q) {
            c_i[tid]  = block_c[tid];
          } else {
            c_i[tid] += block_c[tid];
          }
        }
      }
      /* synchronize before returning */
      SYNCTHREADS();
      return true;
    }
  } // namespace detail


  /**
   * reference implementation, adapted from madness
   * c(i,j) += sum(k) a(k,i)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mTxm(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    if (ldb == dimj && detail::mTxm_block_a<aT, bT, cT, Q>(dimi, dimj, dimk, c, a, b)) {
      return; // succesfully blocked on A
    }
    /* TODO: block on B */

    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *aik_ptr = a + i;
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }

        for (long k=0; k<dimk; ++k,aik_ptr+=dimi) { /* not parallelized */
          aT aki = *aik_ptr;
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] += aki*b[k*ldb+j];
          }
        }
      }
    }
    SYNCTHREADS();
  }

  // /**
  //  * reference implementation, adapted from madness
  //  * c(i,j) = sum(k) a(k,i)*b(k,j)
  //  */
  // template <typename aT, typename bT, typename cT>
  // SCOPE void mTxmq(size_type dimi, size_type dimj, size_type dimk,
  //         cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
  //   mTxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  // }
  template <typename aT, typename bT, typename cT>
  SCOPE
  void mTxmq(size_type dimi, size_type dimj, size_type dimk,
            cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {

    if (ldb == -1) ldb=dimj;
    mTxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(i,k)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxm(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *ai_ptr = a + i*dimk;
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          for (long k=0; k<dimk; ++k) { /* not parallelized */
            ci[j] += ai_ptr[k]*b[k*ldb+j];
          }
        }
      }
    }
    SYNCTHREADS();
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(i,k)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxmq(size_type dimi, size_type dimj, size_type dimk,
                  cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    mxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(i,k)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mxmT(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        const aT *ai_ptr = a + i*dimk;
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          cT sum = 0.0;
          for (size_type k=0; k<dimk; ++k) { /* not parallelized */
            /**
             * TODO: this is not optimal, we should transpose b first into shared memory
             */
            sum += ai_ptr[k]*b[j*dimk+k];
          }
          if constexpr (Q) {
            ci[j] = sum;
          } else {
            ci[j] += sum;
          }

        }
      }
    }
    SYNCTHREADS();
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(i,k)*b(j,k)
   */
  template <typename aT, typename bT, typename cT>
  SCOPE void mxmTq(size_type dimi, size_type dimj, size_type dimk,
                 cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    mxmT<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) += sum(k) a(k,i)*b(j,k)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mTxmT(size_type dimi, size_type dimj, size_type dimk,
                   cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
    /* trivial 2D implementation for devices */
    if (threadIdx.z == 0) {
      for (size_type i = threadIdx.y; i < dimi; i += blockDim.y) {
        cT* ci = c + i*dimj; // the row of C all threads in dim x work on
        if constexpr(Q) {
          for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
            ci[j] = 0.0;
          }
        }
        for (size_type j = threadIdx.x; j < dimj; j += blockDim.x) {
          const aT *aik_ptr = a + i;
          for (long k=0; k<dimk; ++k,aik_ptr+=dimj) { /* not parallelized */
            /**
             * TODO: this is not optimal, we should transpose a and b first into shared memory
             */
            ci[j] += *aik_ptr * b[j*ldb+k];
          }
        }
      }
    }
    SYNCTHREADS();
  }

  /**
   * reference implementation, adapted from madness
   *
   * c(i,j) = sum(k) a(k,i)*b(j,k)
   */
  template <typename aT, typename bT, typename cT>
  SCOPE void mTxmTq(size_type dimi, size_type dimj, size_type dimk,
                   cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    mTxmT<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }
}

#endif // MRA_MXM_H
