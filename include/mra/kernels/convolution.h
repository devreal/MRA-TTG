#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include <algorithm>
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
    return 2*TWOK2NDIM + 4*K2NDIM; // resultf, resultc, tmpresult, result, f, work1, work2
  }

  template <typename T, Dimension NDIM>
  SCOPE void conv_transform(
    const size_type dimk,
    const T mufac,
    const std::array<TensorView<T, 2>, NDIM>& trans,
    const TensorView<T, NDIM>& f,
    TensorView<T, NDIM>& result,
    TensorView<T, NDIM>& work1,
    TensorView<T, NDIM>& work2)
  {
    size_type rank = trans[0].dim(0); // doing computation assuming full rank
    size_type size = 1;
    for (size_type i = 0; i < NDIM; ++i) size *= dimk;
    size_type dimi = size/dimk;

    T* work1ptr = work1.data();
    T* work2ptr = work2.data();

    mTxmq(dimi, rank, dimk, work1ptr, f.data(), trans[0].data());

    size = rank * size / dimk;
    dimi = size / dimk;

    for (size_type d = 1; d < NDIM; ++d) {
      mTxmq(dimi, rank, dimk, work2ptr, work1ptr, trans[d].data());
      size = rank * size / dimk;
      dimi = size / dimk;
      std::swap(work1ptr, work2ptr);
    }

    detail::axpy_kernel_impl<T, NDIM>(work1, result, mufac);
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl(
      size_type K,
      const T normr,
      const T norms,
      const T fac,
      TensorView<T, NDIM>& f,
      const std::array<TensorView<T, 2>, NDIM>& transr,
      const std::array<TensorView<T, 2>, NDIM>& transs,
      TensorView<T, NDIM>& resultf,
      TensorView<T, NDIM>& resultc,
      TensorView<T, NDIM>& tmpresult,
      TensorView<T, NDIM>& result,  // size K, stores the sum
      TensorView<T, NDIM>& work1,
      TensorView<T, NDIM>& work2)
    {
      std::array<Slice,NDIM> s0 = {Slice(0, K), Slice(0, K), Slice(0, K)};
      T normthresh = 1e-20; // Can potentially be a parameter

      if (normr > normthresh) {
        conv_transform<T, NDIM>(2*K, fac, transr, f, resultf, work1, work2);
      }

      SHARED TensorView<T, NDIM> f0;
      f0(s0) = f(s0);
      SYNCTHREADS();

      if (norms > normthresh) {
        conv_transform<T, NDIM>(K, -fac, transs, f0, resultc, work1, work2);
      }

      // auto tmpresult_view = tmpresult.current_view();
      tmpresult(s0) = resultf(s0);
      gaxpy_kernel_impl<T, NDIM>(
        tmpresult, resultc, result, 1.0, 1.0);
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel(
      size_type K,
      size_type N,
      const T normr,
      const T norms,
      const T fac,
      const TensorView<T, NDIM+1> f_view,
      TensorView<T, NDIM+1> result_view,
      const std::array<TensorView<T, 2>, (size_t)NDIM> transr,
      const std::array<TensorView<T, 2>, (size_t)NDIM> transs,
      T* tmp)
    {
      SHARED TensorView<T, NDIM> resultf, resultc, work1, work2;
      SHARED TensorView<T, NDIM> f, tmpresult, result;

      size_type blockId = blockIdx.x;
      T* block_tmp_ptr = &tmp[blockId*convolution_tmp_size<NDIM>(K)];
      const size_type K2NDIM = std::pow(K, NDIM);
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);

      // construct temporaries and pass them to conv_transform
      resultf   = TensorView<T, NDIM>(&block_tmp_ptr[                      0], 2*K);
      resultc   = TensorView<T, NDIM>(&block_tmp_ptr[              TWOK2NDIM], K);
      tmpresult = TensorView<T, NDIM>(&block_tmp_ptr[     TWOK2NDIM + K2NDIM], K);
      f         = TensorView<T, NDIM>(&block_tmp_ptr[     TWOK2NDIM + K2NDIM], K);
      work1     = TensorView<T, NDIM>(&block_tmp_ptr[ 2*TWOK2NDIM + 2*K2NDIM], K);
      work2     = TensorView<T, NDIM>(&block_tmp_ptr[ 2*TWOK2NDIM + 3*K2NDIM], K);

      for (size_type blockId = blockIdx.x; blockId < N; blockId += gridDim.x){
        if (is_team_lead()) {
          f      = f_view(blockId);
          result = result_view(blockId);
        }
      }
      SYNCTHREADS();

      convolution_kernel_impl<T, NDIM>(K, normr, norms, fac, f, transr, transs,
        resultf, resultc, tmpresult, result, work1, work2);
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_convolution_kernel(
    size_type K,
    size_type N,
    const T normr,
    const T norms,
    const T fac,
    const TensorView<T, NDIM+1>& f,
    TensorView<T, NDIM+1>& result,
    const std::array<TensorView<T, 2>, (size_t)NDIM>& transr,
    const std::array<TensorView<T, 2>, (size_t)NDIM>& transs,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    CONFIGURE_KERNEL((detail::convolution_kernel<T, NDIM>), smem_size);
    CALL_KERNEL((detail::convolution_kernel<T, NDIM>), N, thread_dims, smem_size, stream,
    (K, N, normr, norms, fac, f, result, transr, transs, tmp));
    checkSubmit();
  }

  /* explicit instantiation */
  extern template
  void submit_convolution_kernel<double, 3>(
    size_type K,
    size_type N,
    const double normr,
    const double norms,
    const double fac,
    const TensorView<double, 3+1>& f,
    TensorView<double, 3+1>& result,
    const std::array<TensorView<double, 2>, 3>& transr,
    const std::array<TensorView<double, 2>, 3>& transs,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_CONVOLUTION_H
