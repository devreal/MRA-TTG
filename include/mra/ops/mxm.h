#ifndef MRA_MXM_H
#define MRA_MXM_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

#include "mra/ops/mxm_cublasdx.h"

namespace mra{

  /**
   * reference implementation, adapted from madness
   * c(i,j) += sum(k) a(k,i)*b(k,j)
   */
  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE void mTxm(size_type dimi, size_type dimj, size_type dimk,
          cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
    if (ldb == -1) ldb=dimj;
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


#ifndef MRA_HAVE_MTXMQ

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
    mTxm<aT, bT, cT, true>(dimi, dimj, dimk, c, a, b, ldb);
  }


  template<typename T>
  constexpr size_type mTxmq_shmem_size(size_type K) {
    return 0;
  }

#endif // MRA_HAVE_MTXMQ

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
