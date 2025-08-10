#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include "mra/kernels.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/gaxpy.h"
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
    std::array<TensorView<T, 2>, NDIM>& trans,
    const TensorView<T, 2>& f,
    TensorView<T, 2>& result,
    TensorView<T, 2>& work1,
    TensorView<T, 2>& work2)
  {
    size_type rank = trans[0].dim(0); // doing computation assuming full rank
    size_type size = 1;
    for (size_type i = 0; i < NDIM; ++i) size *= dimk;
    size_type dimi = size/dimk;

    mTxmq(dimi, rank, dimk, work1.data(), f.data(), trans[0].data(), dimk);

    size = rank * size / dimk;
    dimi = size / dimk;

    for (size_type d = 1; d < NDIM; ++d) {
      mTxmq(dimi, rank, dimk, work2.data(), work1.data(), trans[d].data(), dimk);
      size = rank * size / dimk;
      dimi = size / dimk;
      std::swap(work1.data(), work2.data());
    }

    axpy_kernel_impl<T, NDIM>(work1, result, T(1.0));
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl()
    {
      T normthresh = 1e-20; // Can potentially be a parameter
      size_type TWOK = 2*K;
      T normr = 1.0;
      for (size_type i = 0; i < NDIM; ++i) normr *= op[i]->normR;
      if (normr > normthresh) {
        // assemble trans and call conv_transform
      }

      T norms = 1.0;
      for (size_type i = 0; i < NDIM; ++i) norms *= op[i]->normS;
      if (norms > normthresh) {
        // assemble trans and call conv_transform
      }
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
  void submit_convolution_kernel()
  {

  }

  /* explicit instantiation */
  extern template
  void submit_convolution_kernel<double, 3>();

}

#endif // MRA_KERNELS_CONVOLUTION_H
