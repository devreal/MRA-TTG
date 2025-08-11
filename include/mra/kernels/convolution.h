#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include "mra/ops/mxm.h"
#include "mra/kernels.h"
#include "mra/kernels/gaxpy.h"
#include "mra/kernels/transform.h"
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/convolutiondata.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"

namespace mra{

  template <Dimension NDIM>
  SCOPE size_type convolution_tmp_size(size_type K) {
    size_type K2NDIM = std::pow(K, NDIM);
    size_type TWOK2NDIM = std::pow(2*K, NDIM);
    return TWOK2NDIM + 5*K2NDIM; // resultf, resultc, tmpresult, f0, work1, work2
  }

  template <typename T, Dimension NDIM>
  SCOPE void conv_transform(
    const size_type dimk,
    const T mufac,
    std::array<TensorView<T, 2>, NDIM>& trans,
    const TensorView<T, NDIM>& f,
    TensorView<T, NDIM>& result,
    TensorView<T, NDIM>& work1,
    TensorView<T, NDIM>& work2)
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

    axpy_kernel_impl<T, NDIM>(work1, result, mufac);
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl(
      size_type K,
      const mra::OperatorData<T, NDIM>& op,
      const TensorView<T, NDIM>& f,
      TensorView<T, NDIM>& result,  // size K, stores the sum
      T* tmp)
    {
      std::array<Slice,NDIM> s0 = {Slice(0, K), Slice(0, K), Slice(0, K)}
      T normthresh = 1e-20; // Can potentially be a parameter

      SHARED std::array<TensorView<T, 2>, NDIM> trans;
      SHARED TensorView<T, NDIM> resultf, resultc, work1, work2, f0;

      size_type blockId = blockIdx.x;
      T* block_tmp_ptr = &tmp[blockId*convolution_tmp_size<NDIM>(K)];
      const size_type K2NDIM = std::pow(K, NDIM);
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);

      // construct temporaries and pass them to conv_transform
      resultf   = TensorView<T, NDIM>(&block_tmp_ptr[                    0], 2*K);
      resultc   = TensorView<T, NDIM>(&block_tmp_ptr[            TWOK2NDIM], K);
      tmpresult = TensorView<T, NDIM>(&block_tmp_ptr[   TWOK2NDIM + K2NDIM], K);
      f0        = TensorView<T, NDIM>(&block_tmp_ptr[ TWOK2NDIM + 2*K2NDIM], K);
      work1     = TensorView<T, NDIM>(&block_tmp_ptr[ TWOK2NDIM + 3*K2NDIM], K);
      work2     = TensorView<T, NDIM>(&block_tmp_ptr[ TWOK2NDIM + 4*K2NDIM], K);

      T normr = 1.0;
      for (size_type i = 0; i < NDIM; ++i) normr *= op->ops[i]->normR;

      if (normr > normthresh) {
        // assemble trans and call conv_transform
        for (size_type d = 0; d < NDIM; ++d){
          trans[d].current_view() = op->ops[d]->R;
        }
      }
      conv_transform<T, NDIM>(2*K, ops.fac, trans, f, resultf, work1, work2);

      T norms = 1.0;
      auto f0_view= f0.current_view();
      f0(s0) = f.current_view()(s0);
      for (size_type i = 0; i < NDIM; ++i) norms *= op->ops[i]->normS;
      if (norms > normthresh) {
        // assemble trans and call conv_transform
        for (size_type d = 0; d < NDIM; ++d){
          trans[d].current_view() = op->ops[d]->S;
        }
      }
      conv_transform<T, NDIM>(K, -ops.fac, trans, f0, resultc, work1, work2);;

      auto tmpresult_view = tmpresult.current_view();
      tmpresult_view(s0) = resultf.current_view()(s0);
      gaxpy_kernel_impl<T, NDIM>(
        tmpresult, resultc, result, 1.0, 1.0);
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
