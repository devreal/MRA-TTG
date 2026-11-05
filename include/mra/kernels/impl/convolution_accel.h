#ifndef MRA_KERNELS_IMPL_CONVOLUTION_ACCEL_H
#define MRA_KERNELS_IMPL_CONVOLUTION_ACCEL_H

/**
 * Selects the accelerated (tensor-core) implementation of the convolution
 * core used by mra/kernels/convolution.h, exported as mra::accel::apply_conv,
 * apply_conv_shmem_size and apply_conv_min_threads:
 *  - CUDA: mma.sync (convolution_cuda_mma.h), or WMMA
 *    (convolution_cuda_wmma.h) if MRA_CONV_USE_WMMA is defined.
 * Both only apply on sm_80+ with a compile-time K == 8 (mma also K == 6,
 * see mma::apply_term_needs); apply_conv returns false otherwise and the
 * generic implementation is used.
 */

#include "mra/kernels/impl/convolution_cuda_mma.h"
#include "mra/kernels/impl/convolution_cuda_wmma.h"

namespace mra {
  namespace accel {

#if defined(MRA_CONV_USE_WMMA)
    using wmma::apply_conv;
    using wmma::apply_conv_shmem_size;
    using wmma::apply_conv_min_threads;
    /* WMMA never splits off the S term (see mma::apply_conv_splits_s) */
    template<typename KT>
    SCOPE constexpr bool apply_conv_splits_s() { return false; }
#else
    using mma::apply_conv;
    using mma::apply_conv_shmem_size;
    using mma::apply_conv_min_threads;
    using mma::apply_conv_splits_s;
#endif
    using mma::apply_conv_s_warp;
    using mma::apply_conv_s_warps;

  } // namespace accel
} // namespace mra

#endif // MRA_KERNELS_IMPL_CONVOLUTION_ACCEL_H
