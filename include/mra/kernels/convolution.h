#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include <algorithm>
#include <cmath>
#include <numbers>
#include <iostream>
#include "mra/ops/mxm.h"
#include "mra/kernels/gaxpy.h"
#include "mra/ops/functions.h"
#include "mra/kernels/transform.h"
#include "mra/misc/conv_mad.h"
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"

namespace mra{

  template <Dimension NDIM>
  SCOPE size_type convolution_tmp_size(size_type K) {
    size_type K2NDIM = std::pow(K, NDIM);
    size_type TWOK2NDIM = std::pow(2*K, NDIM);
    return 3*TWOK2NDIM + 2*K2NDIM; // resultc, result, f, work1, work2
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    SCOPE void conv_transform(
      const size_type dimk,
      const size_type mu,
      const T mufac,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto& trans,
      const concepts::TensorView<NDIM> auto& f,
      concepts::TensorView<NDIM> auto& result,
      concepts::TensorView<NDIM> auto& work1,
      concepts::TensorView<NDIM> auto& work2)
    {
      size_type rank = dimk; // doing computation assuming full rank
      size_type size = 1;
      for (size_type i = 0; i < NDIM; ++i) size *= dimk;
      size_type dimi = size/dimk;

      // assume the tensors to be uninitialized
      result = 0.0;
      work1 = 0.0;
      work2 = 0.0;

      T* work1ptr = work1.data();
      T* work2ptr = work2.data();

      //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size
      //          << " norm f " << normf(f) << " trans " << 0 << normf(trans[0](mu)) << std::endl;
      mTxmq(dimi, rank, dimk, work1ptr, f.data(), trans[0](mu).data());

      size = rank * size / dimk;
      dimi = size / dimk;

      for (size_type d = 1; d < NDIM; ++d) {
        //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size  << " trans " << d << " norm " << norm(trans[d]) << std::endl;
        mTxmq(dimi, rank, dimk, work2ptr, work1ptr, trans[d](mu).data());
        size = rank * size / dimk;
        dimi = size / dimk;
        std::swap(work1ptr, work2ptr);
      }

      detail::axpy_kernel_impl<T, NDIM>(work1, result, mufac);
      //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size
      //          << " result " << normf(result) << std::endl;

    }



#if 0
    /// too lazy for extended calling lists
    struct Transformation {
      long r;             // Effective rank of transformation
      const Q* U;         // Ptr to matrix
      const Q* VT;
    };

    template<typename T, Dimension NDIM>
    DEVSCOPE void make_transformation(T Rnorm, size_type mu, const TensorView<T, 4>& ops,
                                      std::array<Transformation, (size_t)NDIM>& trans) {

      const auto tol_Rs = tol/(Rnorm*NDIM);  // Errors are relative within here

      // Determine rank of SVD to use or if to use the full matrix
      long twok = 2*k;
      // TODO: do we care about modified() operators?
      //if (modified()) twok=k;

      long break_even;
      if (NDIM==1) break_even = long(0.5*twok);
      else if (NDIM==2) break_even = long(0.6*twok);
      else if (NDIM==3) break_even=long(0.65*twok);
      else break_even=long(0.7*twok);
      bool rank_is_zero = false;
      for (std::size_t d=0; d<NDIM; ++d) {
        long r;
        for (r=0; r<twok; ++r) {
          if (ops_1d[d]->Rs[r] < tol_Rs) break;
        }
        if (r >= break_even) {
          trans[d].r = twok;
          trans[d].U = ops(mu, 0).ptr();
          trans[d].VT = nullptr;
        }
        else {
          if (r == 0) {
            rank_is_zero = true;
            break;
          }
          trans[d].r = r;
          trans[d].U = ops(mu, 1).ptr();
          trans[d].VT = ops(mu, 2).ptr();
        }
      }
    }
#endif // 0

    template<typename T, Dimension NDIM>
    void muopxv_fast(
      size_type K,
      const size_type mu,
      const T mufac,
      const T tol,
      const std::array<bool, 2>& at,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto& transs,
      const concepts::TensorView<3> auto& opnorms,
      concepts::TensorView<NDIM> auto& f,
      concepts::TensorView<NDIM> auto& f0,
      concepts::TensorView<NDIM> auto& resultc,
      concepts::TensorView<NDIM> auto& result,
      concepts::TensorView<NDIM> auto& work1,
      concepts::TensorView<NDIM> auto& work2,
      concepts::TensorView<NDIM> auto& work1_k,
      concepts::TensorView<NDIM> auto& work2_k)
    {
      // R term
      double Rnorm = 1.0;
      for (std::size_t d=0; d<NDIM; ++d) Rnorm *= opnorms(mu, d, (size_type)NormId::Rnorm);
      if (at[0] && Rnorm > 1.e-20) {

        conv_transform<T, NDIM>(2*K, mu, mufac, transr, f, result, work1, work2);
      }

      // S term
      double Snorm = 1.0;
      for (std::size_t d=0; d<NDIM; ++d) Snorm *= opnorms(mu, d, (size_type)NormId::Snorm);
      if (at[1] && Snorm > 0.0) {
        conv_transform<T, NDIM>(K, mu, -mufac, transs, f0, resultc, work1_k, work2_k);
      }

    }


    template<typename T, Dimension NDIM>
    DEVSCOPE void apply_conv(
      size_type K,
      const T fac,
      const T tol,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto& transs,
      const concepts::TensorView<3> auto& opnorms,
      const std::array<bool, 2>& at,
      concepts::TensorView<NDIM> auto& f,
      concepts::TensorView<NDIM> auto& f0,
      concepts::TensorView<NDIM> auto& resultc,
      concepts::TensorView<NDIM> auto& result,  // size K, stores the sum
      concepts::TensorView<NDIM> auto& work1,
      concepts::TensorView<NDIM> auto& work2)
    {
      SHARED DenseTensorView<T, NDIM> work1_k, work2_k;
      SHARED std::array<Slice,NDIM> s0;
      if (is_team_lead()) {
        s0 = std::array<Slice,NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
        work1_k = DenseTensorView<T, NDIM>(work1.data(), K);
        work2_k = DenseTensorView<T, NDIM>(work2.data(), K);
      }

      size_type rank = transr[0].dim(0); // doing computation assuming full rank

      T optol = 0.01*tol/rank; // can potentially be a parameter

      f0(s0) = f(s0);

      // TODO: do we care about modified() operators?

      // TODO: why does this fix correctness?!
      result = 0.0;
      resultc = 0.0;
      work1 = 0.0;
      work2 = 0.0;

      /**
       * TODO: split this out into two kernels:
       *  - one kernel that computes the contributions for each muop separately
       *  - one that accumulates the contributions and applies the aggressive screening analogous to MADNESS
       * That way we gain significant parallelism even for small N.
       */
      for (int mu = 0; mu < rank; ++mu) {
        T munorm = opnorms(mu, 0, (size_type)NormId::MUnorm);
        if (munorm > optol) {
          T fac = opnorms(mu, 0, (size_type)NormId::Fac);
          muopxv_fast<T, NDIM>(K, mu, fac, tol/std::abs(fac), at, transr, transs, opnorms, f, f0,
                               resultc, result, work1, work2, work1_k, work2_k);
        }
      }
      //r(s0).gaxpy(1.0,r0,1.0);
      // OR
      //foreach_idxs(resultc, [&](auto... idxs) {
      //  result(idxs...) += resultc(idxs...);
      //});
      result(s0) += resultc;

    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl(
      Key<NDIM> key,
      Key<NDIM> displacement,
      size_type K,
      const T opnorm,
      const T fac,
      const T tol,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto& transs,
      const concepts::TensorView<3> auto& opnorms,
      const std::array<bool, 2>& at,
      concepts::TensorView<NDIM> auto& in,
      concepts::TensorView<NDIM> auto& f,
      concepts::TensorView<NDIM> auto& f0,
      concepts::TensorView<NDIM> auto& resultc,
      concepts::TensorView<NDIM> auto& result,  // size K, stores the sum
      concepts::TensorView<NDIM> auto& work1,
      concepts::TensorView<NDIM> auto& work2,
      T* resnorm_out)
    {
      SYNCTHREADS();
      T normthresh = 1e-20; // Can potentially be a parameter
      const T cnorm = mra::normf(f);
      T resnorm = 0.0;

      //std::cout << "MRA-APPLY key " << key << " disp " << displacement << " cnorm " << cnorm
      //          << " opnorm " << opnorm << " tol " << tol << std::endl;
      if ((cnorm * opnorm) > (tol / fac)) {

        apply_conv<T, NDIM>(K, fac, (tol / fac / cnorm), transr, transs,
                   opnorms, at, f, f0, resultc,
                   result, work1, work2);

        resnorm = normf(result);
      }

      bool above_threshold = (resnorm > (0.3 * tol / fac));

      //std::cout << "MRA_OP_APPLY " << key << " disp " << displacement << " cnorm " << cnorm
      //          << " opnorm " << opnorm << " tol " << tol << " resnorm " << resnorm
      //          << (above_threshold ? " above threshold" : " below threshold, dropping result") << std::endl;

      // Accumulate input if not empty
      if (!in.empty()) {
        if (above_threshold) {
          /* add input values */
          result += in;
        } else {
          /* if input is empty, we can just copy the result to it */
          result = in;
        }
        resnorm = normf(result);
      } else if (!above_threshold) {
        /* if input is empty and result is below threshold, we can just leave it zero */
        result = 0.0;
        resnorm = 0.0;
      }
      if (resnorm_out != nullptr) {
        if (is_team_lead()) {
          *resnorm_out = resnorm;
        }
      }

      //std::cout << "MRA_OP_APPLY " << key << " disp " << displacement << " result " << resnorm << std::endl;

    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel(
      Key<NDIM> key,
      Key<NDIM> displacement,
      size_type K,
      size_type N,
      const T opnorm,
      const T fac,
      const T tol,
      const concepts::TensorView<NDIM+1> auto in_view,
      const concepts::TensorView<NDIM+1> auto f_view,
      concepts::TensorView<NDIM+1> auto result_view,
      concepts::TensorView<1> auto resnorms,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto transr,
      const concepts::TensorViewArray<3, (size_t)NDIM> auto transs,
      const concepts::TensorView<3> auto opnorms,
      const std::array<bool, 2> at,
      T* tmp)
    {
      SHARED DenseTensorView<T, NDIM> f0, resultc, work1, work2, result;
      SHARED DenseTensorView<const T, NDIM> f, in;

      size_type blockId = blockIdx.x;
      T* block_tmp_ptr = &tmp[blockId*convolution_tmp_size<NDIM>(K)];
      const size_type K2NDIM = std::pow(K, NDIM);
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);

      if (is_team_lead()) {
        // construct temporaries and pass them to conv_transform
        f0        = DenseTensorView<T, NDIM>(&block_tmp_ptr[                     0], K);
        resultc   = DenseTensorView<T, NDIM>(&block_tmp_ptr[                K2NDIM], K);
        work1     = DenseTensorView<T, NDIM>(&block_tmp_ptr[              2*K2NDIM], 2*K);
        work2     = DenseTensorView<T, NDIM>(&block_tmp_ptr[  TWOK2NDIM + 2*K2NDIM], 2*K);
      }

      for (size_type blockId = blockIdx.x; blockId < N; blockId += gridDim.x) {
        if (result_view.is_zero(blockId)) {
          // nothing to do
          continue;
        }
        if (is_team_lead()) {
          in     = in_view(blockId);
          f      = f_view(blockId);
          result = result_view(blockId);
        }
        SYNCTHREADS();
        if (f_view.is_zero(blockId)) {
          /* copy input to output */
          result = in;
          continue;
        }

        convolution_kernel_impl<T, NDIM>(key, displacement, K, opnorm, fac, tol,
                                         transr, transs, opnorms, at, in, f, f0,
                                         resultc, result, work1, work2,
                                         resnorms.empty() ? nullptr : &resnorms[blockId]);
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_convolution_kernel(
    Key<NDIM> key,
	Key<NDIM> displacement,
    size_type K,
    size_type N,
    const T opnorm,
    const T fac,
    const T tol,
    const concepts::TensorView<NDIM+1> auto& in_view,
    const concepts::TensorView<NDIM+1> auto& f_view,
    concepts::TensorView<NDIM+1> auto& result_view,
    concepts::TensorView<1> auto& resnorms,
    const concepts::TensorViewArray<3, (size_t)NDIM> auto& transr,
    const concepts::TensorViewArray<3, (size_t)NDIM> auto& transs,
    const concepts::TensorView<3> auto& opnorms,
    const std::array<bool, 2>& at,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    CONFIGURE_KERNEL((detail::convolution_kernel<T, NDIM>), smem_size);
    CALL_KERNEL((detail::convolution_kernel<T, NDIM>), N, thread_dims, smem_size, stream,
                (key, displacement, K, N, opnorm, fac, tol, in_view, f_view, result_view,
                 resnorms, transr, transs, opnorms, at, tmp));
    checkSubmit();
  }


  /* explicit instantiation */
  extern template
  void submit_convolution_kernel<double, 3>(
    Key<3> key,
    Key<3> displacement,
    size_type K,
    size_type N,
    const double opnorm,
    const double fac,
    const double tol,
    const SparseTensorView<double, 3+1>& in,
    const SparseTensorView<double, 3+1>& contribution,
    SparseTensorView<double, 3+1>& result,
    SparseTensorView<double, 1>& resnorms,
    const std::array<SparseTensorView<double, 3>, 3>& transr,
    const std::array<SparseTensorView<double, 3>, 3>& transs,
    const DenseTensorView<double, 3>& opnorms,
    const std::array<bool, 2>& at,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_CONVOLUTION_H
