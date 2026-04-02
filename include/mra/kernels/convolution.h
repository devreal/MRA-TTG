#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include <algorithm>
#include "mra/ops/mxm.h"
#include "mra/kernels/gaxpy.h"
#include "mra/ops/functions.h"
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
    return 4*TWOK2NDIM + 3*K2NDIM; // resultf, resultc, tmpresult, result, f, work1, work2
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

    // assume the tensors to be uninitialized
    result = 0.0;
    work1 = 0.0;
    work2 = 0.0;

    T* work1ptr = work1.data();
    T* work2ptr = work2.data();

    std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size << " norm f " << normf(f) << " trans " << 0 << normf(trans[0]) << std::endl;
    mTxmq(dimi, rank, dimk, work1ptr, f.data(), trans[0].data());

    size = rank * size / dimk;
    dimi = size / dimk;

    for (size_type d = 1; d < NDIM; ++d) {
      //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size  << " trans " << d << " norm " << norm(trans[d]) << std::endl;
      mTxmq(dimi, rank, dimk, work2ptr, work1ptr, trans[d].data());
      size = rank * size / dimk;
      dimi = size / dimk;
      std::swap(work1ptr, work2ptr);
    }

    detail::axpy_kernel_impl<T, NDIM>(work1, result, mufac);
    std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size << " result " << normf(result) << std::endl;

  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl(
      Key<NDIM> key,
      Key<NDIM> displacement,
      size_type K,
      const T opnorm,
      const T normr,
      const T norms,
      const T fac,
      const T tol,
      const std::array<TensorView<T, 2>, NDIM>& transr,
      const std::array<TensorView<T, 2>, NDIM>& transs,
      const std::array<bool, 2>& at,
      TensorView<T, NDIM>& in,
      TensorView<T, NDIM>& f,
      TensorView<T, NDIM>& f0,
      TensorView<T, NDIM>& resultf,
      TensorView<T, NDIM>& resultc,
      TensorView<T, NDIM>& tmpresult,
      TensorView<T, NDIM>& result,  // size K, stores the sum
      TensorView<T, NDIM>& work1,
      TensorView<T, NDIM>& work2,
      T* resnorm_out)
    {
      SHARED TensorView<T, NDIM> work1_k, work2_k;
      SHARED std::array<Slice,NDIM> s0;
      if (is_team_lead()) {
        s0 = std::array<Slice,NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
        work1_k = TensorView<T, NDIM>(work1.data(), K);
        work2_k = TensorView<T, NDIM>(work2.data(), K);
      }
      SYNCTHREADS();
      T normthresh = 1e-20; // Can potentially be a parameter
      const T cnorm = mra::normf(f);
      T resnorm = 0.0;

      std::cout << "MRA-APPLY key " << key << " disp " << displacement << " cnorm " << cnorm
                << " opnorm " << opnorm << " tol " << tol << std::endl;
      if ((cnorm * opnorm > tol / fac) && (cnorm * opnorm > tol / fac)) {

        // TODO: is the third condition valid?
        if (at[0] && normr > normthresh /*&& (normr > normthresh/(normr * NDIM))*/) {
          conv_transform<T, NDIM>(2*K, fac, transr, f, result, work1, work2);
        }

        f0(s0) = f(s0);

        if (at[1] && norms > normthresh /*&& (norms > normthresh/(norms * NDIM))*/) {
          conv_transform<T, NDIM>(K, -fac, transs, f0, resultc, work1_k, work2_k);
        }


        // TODO: this does not do the same thing as the MADNESS code!
        // MAD: r(s0).gaxpy(1.0,r0,1.0);
        //result(s0) += resultc;
        foreach_idxs(resultc, [&](auto... idxs) {
          result(idxs...) += resultc(idxs...);
        });
        //gaxpy_kernel_impl<T, NDIM>(
        //  tmpresult, resultc, result, 1.0, 1.0);

        // set to zero if norm is below threshold; this is the aggressive screening analogous to MADNESS
        resnorm = normf(result);
      }

      bool above_threshold = (resnorm > (0.3 * tol / fac));

      //tmpresult = resultf(s0);

      std::cout << "MRA_OP_APPLY BEFORE ACCUMULATE " << key << " disp " << displacement
                << ", in " << normf(in)
                << ", tol " << tol << ", fac " << fac
                << ", result " << resnorm
                << " above threshold " << above_threshold << std::endl;

      //std::cout << "MRA_OP_APPLY BEFORE ACCUMULATE " << key << " disp " << displacement << " result \n" << result << std::endl;

      if (!above_threshold) {
        /* reset result to 0 */
        result = 0.0;
      }
      if (!in.empty()) {
        /* add input values */
        result += in;
      }

      if (resnorm_out != nullptr) {
        resnorm = normf(result);
        if (is_team_lead()) {
          *resnorm_out = resnorm;
        }
      }

      std::cout << "MRA_OP_APPLY " << key << " disp " << displacement << " result " << resnorm << std::endl;

    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel(
      Key<NDIM> key,
      Key<NDIM> displacement,
      size_type K,
      size_type N,
      const T opnorm,
      const T normr,
      const T norms,
      const T fac,
      const T tol,
      const TensorView<T, NDIM+1> in_view,
      const TensorView<T, NDIM+1> f_view,
      TensorView<T, NDIM+1> result_view,
      TensorView<T, 1>& resnorms,
      const std::array<TensorView<T, 2>, (size_t)NDIM> transr,
      const std::array<TensorView<T, 2>, (size_t)NDIM> transs,
      const std::array<bool, 2> at,
      T* tmp)
    {
      SHARED TensorView<T, NDIM> f0, resultc, work1, work2;
      SHARED TensorView<T, NDIM> f, tmpresult, resultf, result, in;

      size_type blockId = blockIdx.x;
      T* block_tmp_ptr = &tmp[blockId*convolution_tmp_size<NDIM>(K)];
      const size_type K2NDIM = std::pow(K, NDIM);
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);

      if (is_team_lead()) {
        // construct temporaries and pass them to conv_transform
        f0        = TensorView<T, NDIM>(&block_tmp_ptr[                     0], K);
        resultc   = TensorView<T, NDIM>(&block_tmp_ptr[                K2NDIM], K);
        tmpresult = TensorView<T, NDIM>(&block_tmp_ptr[              2*K2NDIM], K);
        work1     = TensorView<T, NDIM>(&block_tmp_ptr[              3*K2NDIM], 2*K);
        work2     = TensorView<T, NDIM>(&block_tmp_ptr[  TWOK2NDIM + 3*K2NDIM], 2*K);
        resultf   = TensorView<T, NDIM>(&block_tmp_ptr[2*TWOK2NDIM + 3*K2NDIM], 2*K);
      }

      for (size_type blockId = blockIdx.x; blockId < N; blockId += gridDim.x){
        if (is_team_lead()) {
          in     = in_view(blockId);
          f      = f_view(blockId);
          result = result_view(blockId);
        }
        SYNCTHREADS();

        convolution_kernel_impl<T, NDIM>(key, displacement, K, opnorm, normr, norms, fac, tol,
                                         transr, transs, at, in, f, f0,
                                         resultf, resultc, tmpresult, result, work1, work2,
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
    const T normr,
    const T norms,
    const T fac,
    const T tol,
    const TensorView<T, NDIM+1>& in,
    const TensorView<T, NDIM+1>& f,
    TensorView<T, NDIM+1>& result,
    TensorView<T, 1>& resnorms,
    const std::array<TensorView<T, 2>, (size_t)NDIM>& transr,
    const std::array<TensorView<T, 2>, (size_t)NDIM>& transs,
    const std::array<bool, 2>& at,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    CONFIGURE_KERNEL((detail::convolution_kernel<T, NDIM>), smem_size);
    CALL_KERNEL((detail::convolution_kernel<T, NDIM>), N, thread_dims, smem_size, stream,
                (key, displacement, K, N, opnorm, normr, norms, fac, tol, in, f, result,
                 resnorms, transr, transs, at, tmp));
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
    const double normr,
    const double norms,
    const double fac,
    const double tol,
    const TensorView<double, 3+1>& in,
    const TensorView<double, 3+1>& contribution,
    TensorView<double, 3+1>& result,
    TensorView<double, 1>& resnorms,
    const std::array<TensorView<double, 2>, 3>& transr,
    const std::array<TensorView<double, 2>, 3>& transs,
    const std::array<bool, 2>& at,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_CONVOLUTION_H
