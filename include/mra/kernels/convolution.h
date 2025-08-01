#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include "mra/kernels.h"
#include "mra/kernels/transform.h"
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/ops/mxm.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"

namespace mra{

  template <Dimension NDIM>
  SCOPE size_type convolution_tmp_size(size_type K) {
    return 0;
  }

  template <typename T, Dimension NDIM>
  SCOPE void conv_transform(
    size_type K,
    size_type dimk,
    const TensorView<T, NDIM>& trans,
    const TensorView<T, 2>& f,
    TensorView<T, 2>& result,
    T* work1,
    T* work2)
  {
    // This function is a placeholder for the actual convolution transform logic.
    // It should be implemented to perform the convolution operation on tensors.
    // Analogue to madness apply_transformation()
    size_type rank = 2*K; // doing computation assuming full rank
    size_type size = 1;
    for (size_type i = 0; i < NDIM; ++i) size *= dimk;
    size_type dimi = size/dimk;

    mTxmq(dimi, rank /* need to define it here */, dimk, work1, f.data(), trans.data(), dimk);

    size = rank * size / dimk;
    dimi = size / dimk;

    for (size_type d = 0; d < NDIM; ++d) {
      mTxmq(dimi, rank, dimk, work2, work1, trans.data(), dimk);
      size = rank * size / dimk;
      dimi = size / dimk;
      std::swap(work1, work2);
    }

    // aligned_axpy();
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl()
    {
      // Implement the convolution kernel logic here
      // This is a placeholder for the actual implementation
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel()
    {
      // Call the implementation function
      convolution_kernel_impl<T, NDIM>();
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_compress_kernel()
  {

  }

  /* explicit instantiation */
  extern template
  void submit_compress_kernel<double, 3>();

}

#endif // MRA_KERNELS_CONVOLUTION_H
