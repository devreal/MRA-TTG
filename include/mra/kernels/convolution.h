#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <numbers>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mra/misc/allocator.h"
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
    DEVSCOPE void conv_transform(
      int opid,
      const size_type dimk,
      const size_type mu,
      const T mufac,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& trans,
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
      work1 = 0.0;
      work2 = 0.0;

      T* work1ptr = work1.data();
      T* work2ptr = work2.data();

      //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size
      //          << " norm f " << normf(f) << " trans " << 0 << normf(trans[0](opid, mu)) << std::endl;
      mTxmq(dimi, rank, dimk, work1ptr, f.data(), trans[0](opid, mu).data());

      size = rank * size / dimk;
      dimi = size / dimk;

      for (size_type d = 1; d < NDIM; ++d) {
        //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size  << " trans " << d << " norm " << norm(trans[d]) << std::endl;
        mTxmq(dimi, rank, dimk, work2ptr, work1ptr, trans[d](opid, mu).data());
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
    DEVSCOPE void muopxv_fast(
      int opid,
      size_type K,
      const size_type mu,
      const T mufac,
      const T tol,
      const std::array<bool, 2>& at,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
      const concepts::TensorView<4> auto& opnorms,
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
      for (std::size_t d=0; d<NDIM; ++d) Rnorm *= opnorms(opid, mu, d, (size_type)NormId::Rnorm);
      if (at[0] && Rnorm > 1.e-20) {

        conv_transform<T, NDIM>(opid, 2*K, mu, mufac, transr, f, result, work1, work2);
      }

      // S term
      double Snorm = 1.0;
      for (std::size_t d=0; d<NDIM; ++d) Snorm *= opnorms(opid, mu, d, (size_type)NormId::Snorm);
      if (at[1] && Snorm > 0.0) {
        conv_transform<T, NDIM>(opid, K, mu, -mufac, transs, f0, resultc, work1_k, work2_k);
      }

      //auto rnorm = normf(result);
      //auto rcnorm = normf(resultc);
      //if (is_team_lead())
      //  printf("MRA APPLY CONV: opid %d mu %d result %e resultc %e\n",
      //       opid, mu, rnorm, rcnorm);

    }


    template<typename T, Dimension NDIM>
    DEVSCOPE void apply_conv(
      int opid,
      size_type K,
      const T fac,
      const T tol,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
      const concepts::TensorView<4> auto& opnorms,
      const std::array<bool, 2>& at,
      concepts::TensorView<NDIM> auto& f,
      concepts::TensorView<NDIM> auto& f0,
      concepts::TensorView<NDIM> auto& resultc,
      concepts::TensorView<NDIM> auto& result,  // size K, stores the sum
      concepts::TensorView<NDIM> auto& work1,
      concepts::TensorView<NDIM> auto& work2)
    {
      SHARED DenseTensorView<T, NDIM> work1_k, work2_k;
      // cannot be SHARED because ctors won't run
      std::array<Slice,NDIM> s0 = std::array<Slice,NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
      if (is_team_lead()) {
        work1_k = DenseTensorView<T, NDIM>(work1.data(), K);
        work2_k = DenseTensorView<T, NDIM>(work2.data(), K);
      }

      const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank); // doing computation assuming full rank

      T optol = 0.01*tol/rank; // can potentially be a parameter

      f0(s0) = f(s0);

      // TODO: do we care about modified() operators?

      // TODO: why does this fix correctness?!
      result = 0.0;
      resultc = 0.0;

      for (size_type mu = 0; mu < rank; ++mu) {
        T munorm = opnorms(opid, mu, 0, (size_type)NormId::MUnorm);
        if (munorm > optol) {
          T mufac = opnorms(opid, mu, 0, (size_type)NormId::Fac);
          muopxv_fast<T, NDIM>(opid, K, mu, mufac, tol/std::abs(mufac), at, transr, transs, opnorms, f, f0,
                               resultc, result, work1, work2, work1_k, work2_k);
        }
      }
      result(s0) += resultc;
    }

    /**
     * Combines a (fully summed) convolution result with the pre-existing `in` node and
     * applies the aggressive-screening threshold, writing the final per-function norm to
     * `resnorm_out` (if non-null).
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_finalize(
      const T fac,
      const T tol,
      const concepts::TensorView<NDIM> auto& in,
      concepts::TensorView<NDIM> auto& result,
      T* resnorm_out)
    {
      T resnorm = normf(result);
      bool above_threshold = (resnorm > (0.3 * tol / fac));

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
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl(
      Key<NDIM> key,
      int opid,
      Key<NDIM> displacement,
      size_type K,
      const T fac,
      const T tol,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
      const concepts::TensorView<4> auto& opnorms,
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
      const T cnorm = mra::normf(f);
      T opnorm = opnorms(opid, 0, 0, (size_type)NormId::Opnorm);

      //std::cout << "MRA-APPLY key " << key << " disp " << displacement << " cnorm " << cnorm
      //          << " opnorm " << opnorm << " tol " << tol << std::endl;
      if ((cnorm * opnorm) > (tol / fac)) {
        apply_conv<T, NDIM>(opid, K, fac, (tol / fac / cnorm), transr, transs,
                   opnorms, at, f, f0, resultc,
                   result, work1, work2);
      } else {
        result = 0.0;
      }

      convolution_finalize<T, NDIM>(fac, tol, in, result, resnorm_out);

      //std::cout << "MRA_OP_APPLY " << key << " disp " << displacement << " result " << resnorm << std::endl;

    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel(
      Key<NDIM> key,
      Key<NDIM> displacement,
      size_type K,
      size_type N,
      const T fac,
      const T tol,
      const concepts::TensorView<NDIM+1> auto in_view,
      const concepts::TensorView<NDIM+1> auto f_view,
      concepts::TensorView<NDIM+1> auto result_view,
      concepts::TensorView<1> auto resnorms,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto transs,
      const concepts::TensorView<4> auto opnorms_view,
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
          if (is_team_lead() && !resnorms.empty()) {
            resnorms[blockId] = 0.0;
          }
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
          if (!resnorms.empty()) {
            auto resnorm = normf(result);
            if (is_team_lead()) {
              resnorms[blockId] = resnorm;
            }
          }
          continue;
        }

        int opid = opnorms_view.dim(0) > 1 ? blockId : 0; // TODO: this is a bit hacky, can we do better?

        convolution_kernel_impl<T, NDIM>(key, opid, displacement, K, fac, tol,
                                         transr, transs, opnorms_view, at, in, f, f0,
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
    const T fac,
    const T tol,
    const concepts::TensorView<NDIM+1> auto& in_view,
    const concepts::TensorView<NDIM+1> auto& f_view,
    concepts::TensorView<NDIM+1> auto& result_view,
    concepts::TensorView<1> auto& resnorms,
    const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
    const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
    const concepts::TensorView<4> auto& opnorms,
    const std::array<bool, 2>& at,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    //CONFIGURE_KERNEL((detail::convolution_kernel<T, NDIM>), smem_size);
    CALL_KERNEL((detail::convolution_kernel<T, NDIM>), N, thread_dims, smem_size, stream,
                (key, displacement, K, N, fac, tol, in_view, f_view, result_view,
                 resnorms, transr, transs, opnorms, at, tmp));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the convolution kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/convolution.h. Batching is only ever
   * done across tasks at the same (level, displacement), which guarantees
   * K/fac/tol/transr/transs/opnorms/at are bitwise identical for every member
   * (see mra/misc/conv_mad.h::GaussianConvolutionOperator::get_op and
   * mra/ops/functions.h::truncate_tol) -- only in/f/result/resnorms/tmp and
   * the per-member function count N_m differ.
   */
  namespace detail {

#if defined(MRA_ENABLE_CUDA)
    inline void check_cuda_rt(cudaError_t err, const char* what) {
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("ConvolutionBatchPool: ") + what + " failed: " + cudaGetErrorString(err));
      }
    }
#elif defined(MRA_ENABLE_HIP)
    inline void check_hip_rt(hipError_t err, const char* what) {
      if (err != hipSuccess) {
        throw std::runtime_error(std::string("ConvolutionBatchPool: ") + what + " failed: " + hipGetErrorString(err));
      }
    }
#endif

    /**
     * Bundles everything convolution_kernel_batched needs for one batch member.
     * in_view/f_view are only ever read inside the kernel; result_view/resnorms_view
     * are written. Constness is enforced at the point of use (via a `const auto&`
     * local binding before calling operator()) rather than in this struct's field
     * types, since there is no converting constructor from SparseTensorView<T,...>
     * to SparseTensorView<const T,...> to build the latter from the views the
     * surrounding task already holds.
     */
    template <typename T, Dimension NDIM>
    struct ConvolutionBatchArg {
      SparseTensorView<T, NDIM+1> in_view;
      SparseTensorView<T, NDIM+1> f_view;
      SparseTensorView<T, NDIM+1> result_view;
      DenseTensorView<T, 1> resnorms_view;
      T* tmp = nullptr;
      size_type block_offset = 0; // exclusive prefix-sum start for this member within the launch
      size_type n = 0;            // number of blocks (functions) this member contributes

      ConvolutionBatchArg() = default;
      ConvolutionBatchArg(SparseTensorView<T, NDIM+1> in_view,
                          SparseTensorView<T, NDIM+1> f_view,
                          SparseTensorView<T, NDIM+1> result_view,
                          DenseTensorView<T, 1> resnorms_view,
                          T* tmp, size_type block_offset, size_type n)
      : in_view(in_view), f_view(f_view), result_view(result_view), resnorms_view(resnorms_view)
      , tmp(tmp), block_offset(block_offset), n(n)
      { }
    };

    /**
     * One combined launch covering `num_members` independent nodes sharing one
     * (K, fac, tol, transr, transs, opnorms, at). Grid size = total number of
     * blocks across all members (sum of each member's N_m); each block looks
     * up which member it belongs to via a prefix-sum `block_offset` field on
     * each arg (linear scan -- num_members is bounded by set_batch_matcher's
     * max_batch_size, so this is cheap compared to the mTxmq calls that follow).
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel_batched(
      size_type num_members,
      ConvolutionBatchArg<T, NDIM>* args,   // device ptr, size num_members
      size_type total_blocks,
      size_type K,
      const T fac,
      const T tol,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto transs,
      const concepts::TensorView<4> auto opnorms_view,
      const std::array<bool, 2> at)
    {
      SHARED DenseTensorView<T, NDIM> f0, resultc, work1, work2, result;
      SHARED DenseTensorView<const T, NDIM> f, in;
      SHARED size_type s_member, s_local;

      const size_type K2NDIM = std::pow(K, NDIM);
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);

      for (size_type blockId = blockIdx.x; blockId < total_blocks; blockId += gridDim.x) {
        if (is_team_lead()) {
          size_type m = 0;
          while ((m + 1) < num_members && args[m+1].block_offset <= blockId) ++m;
          s_member = m;
          s_local  = blockId - args[m].block_offset;
        }
        SYNCTHREADS();
        const size_type member = s_member;
        const size_type i      = s_local;
        auto& arg = args[member];

        if (arg.result_view.is_zero(i)) {
          // nothing to do
          if (is_team_lead() && !arg.resnorms_view.empty()) {
            arg.resnorms_view[i] = 0.0;
          }
          continue;
        }
        if (is_team_lead()) {
          // const& forces operator()'s read-only overload; result_view stays
          // mutable since the kernel writes through it.
          const auto& in_view = arg.in_view;
          const auto& f_view  = arg.f_view;
          auto& result_view   = arg.result_view;

          T* block_tmp_ptr = &arg.tmp[i*convolution_tmp_size<NDIM>(K)];
          f0        = DenseTensorView<T, NDIM>(&block_tmp_ptr[                     0], K);
          resultc   = DenseTensorView<T, NDIM>(&block_tmp_ptr[                K2NDIM], K);
          work1     = DenseTensorView<T, NDIM>(&block_tmp_ptr[              2*K2NDIM], 2*K);
          work2     = DenseTensorView<T, NDIM>(&block_tmp_ptr[  TWOK2NDIM + 2*K2NDIM], 2*K);
          in     = in_view(i);
          f      = f_view(i);
          result = result_view(i);
        }
        SYNCTHREADS();
        if (arg.f_view.is_zero(i)) {
          /* copy input to output */
          result = in;
          if (!arg.resnorms_view.empty()) {
            auto resnorm = normf(result);
            if (is_team_lead()) {
              arg.resnorms_view[i] = resnorm;
            }
          }
          continue;
        }

        // opid indexes opnorms_view by *local* function index within the member's
        // own node, not the flattened blockId -- do not conflate the two.
        int opid = opnorms_view.dim(0) > 1 ? static_cast<int>(i) : 0;

        convolution_kernel_impl<T, NDIM>(Key<NDIM>{}, opid, Key<NDIM>{}, K, fac, tol,
                                         transr, transs, opnorms_view, at, in, f, f0,
                                         resultc, result, work1, work2,
                                         arg.resnorms_view.empty() ? nullptr : &arg.resnorms_view[i]);
      }
    }

    /**
     * Per-device pool of pinned host / device arrays used to build ONE
     * convolution_kernel_batched launch from the ttg::device::coop-collected
     * batch. Memory is per-device, not per-stream: any slot may be reused for
     * any stream submitted on this pool's device. There is no fixed slot
     * count and no blocking wait -- acquire() scans for a slot whose previous
     * submission has completed (a non-blocking event query) and, if none is
     * free, grows the pool by one more slot rather than stalling the caller.
     * Host storage uses DeviceAllocator<T> (the same pinned-memory allocator
     * used for ttg::Buffer elsewhere in this codebase) instead of hand-rolled
     * cudaHostRegister calls.
     */
    template <typename T, Dimension NDIM>
    struct ConvolutionBatchPool {
      using arg_t = ConvolutionBatchArg<T, NDIM>;

      struct slot_t {
        std::vector<arg_t, DeviceAllocator<arg_t>> host_args; // pinned host storage
        arg_t* dev_args = nullptr;
        std::size_t dev_capacity = 0;
#if defined(MRA_ENABLE_CUDA)
        cudaEvent_t event;
#elif defined(MRA_ENABLE_HIP)
        hipEvent_t event;
#endif
        bool event_recorded = false; // false until this slot has been submitted at least once

        slot_t() {
#if defined(MRA_ENABLE_CUDA)
          check_cuda_rt(cudaEventCreate(&event), "cudaEventCreate");
#elif defined(MRA_ENABLE_HIP)
          check_hip_rt(hipEventCreate(&event), "hipEventCreate");
#endif
        }

        slot_t(const slot_t&) = delete;
        slot_t& operator=(const slot_t&) = delete;

        ~slot_t() {
#if defined(MRA_ENABLE_CUDA)
          if (dev_args) cudaFree(dev_args);
          cudaEventDestroy(event);
#elif defined(MRA_ENABLE_HIP)
          if (dev_args) hipFree(dev_args);
          hipEventDestroy(event);
#endif
        }
      };

      explicit ConvolutionBatchPool(int device) : device(device) { }

      ConvolutionBatchPool(const ConvolutionBatchPool&) = delete;
      ConvolutionBatchPool& operator=(const ConvolutionBatchPool&) = delete;

      /* Never blocks: returns a slot with device-side capacity for at least
       * num_members entries, ready to be filled via slot_t::host_args. */
      slot_t& acquire(std::size_t num_members) {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& sp : slots) {
          if (!sp->event_recorded || event_ready(sp->event)) {
            ensure_capacity(*sp, num_members);
            return *sp;
          }
        }
        slots.push_back(std::make_unique<slot_t>());
        auto& s = *slots.back();
        ensure_capacity(s, num_members);
        return s;
      }

      /* Call right after the H2D copy + kernel launch have been issued on `stream`. */
      void mark_submitted(slot_t& s, ttg::device::Stream stream) {
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaEventRecord(s.event, stream), "cudaEventRecord");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipEventRecord(s.event, stream), "hipEventRecord");
#endif
        s.event_recorded = true;
      }

      int device;

     private:
#if defined(MRA_ENABLE_CUDA)
      static bool event_ready(cudaEvent_t event) {
        cudaError_t err = cudaEventQuery(event);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        check_cuda_rt(err, "cudaEventQuery");
        return false; // unreachable
      }
#elif defined(MRA_ENABLE_HIP)
      static bool event_ready(hipEvent_t event) {
        hipError_t err = hipEventQuery(event);
        if (err == hipSuccess) return true;
        if (err == hipErrorNotReady) return false;
        check_hip_rt(err, "hipEventQuery");
        return false; // unreachable
      }
#endif

      void ensure_capacity(slot_t& s, std::size_t num_members) {
        if (num_members > s.dev_capacity) {
          if (s.dev_args) {
#if defined(MRA_ENABLE_CUDA)
            check_cuda_rt(cudaFree(s.dev_args), "cudaFree");
#elif defined(MRA_ENABLE_HIP)
            check_hip_rt(hipFree(s.dev_args), "hipFree");
#endif
          }
#if defined(MRA_ENABLE_CUDA)
          check_cuda_rt(cudaMalloc(&s.dev_args, num_members*sizeof(arg_t)), "cudaMalloc");
#elif defined(MRA_ENABLE_HIP)
          check_hip_rt(hipMalloc(&s.dev_args, num_members*sizeof(arg_t)), "hipMalloc");
#endif
          s.dev_capacity = num_members;
        }
        s.host_args.reserve(num_members);
      }

      std::mutex mtx;
      std::vector<std::unique_ptr<slot_t>> slots;
    };

    /**
     * Lazily constructs one ConvolutionBatchPool per device, the first time
     * that device is actually used (rather than eagerly allocating memory on
     * every device up front). Construction happens from inside a device task,
     * so ttg::device::current_device() -- used as the index -- already
     * reflects the correct CUDA/HIP context; no explicit
     * cudaSetDevice/hipSetDevice bookkeeping is needed here.
     */
    template <typename T, Dimension NDIM>
    struct ConvolutionBatchPoolRegistry {
      explicit ConvolutionBatchPoolRegistry(int num_devices) : entries(num_devices) { }

      ConvolutionBatchPool<T, NDIM>& get(int device) {
        auto& e = entries[device];
        std::call_once(e.once, [&]{ e.pool = std::make_unique<ConvolutionBatchPool<T, NDIM>>(device); });
        return *e.pool;
      }

     private:
      struct entry_t {
        std::once_flag once;
        std::unique_ptr<ConvolutionBatchPool<T, NDIM>> pool;
      };
      std::vector<entry_t> entries;
    };

  } // namespace detail

  /**
   * Batched counterpart of submit_convolution_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.host_args (by the
   * caller, via detail::submit_convolution_batch_leader below), sharing one
   * (K, fac, tol, transr, transs, opnorms, at) across the whole batch.
   */
  template <typename T, Dimension NDIM>
  void submit_convolution_kernel_batched(
    detail::ConvolutionBatchPool<T, NDIM>& pool,
    typename detail::ConvolutionBatchPool<T, NDIM>::slot_t& slot,
    size_type K,
    const T fac,
    const T tol,
    const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
    const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
    const concepts::TensorView<4> auto& opnorms,
    const std::array<bool, 2>& at,
    ttg::device::Stream stream)
  {
    using arg_t = typename detail::ConvolutionBatchPool<T, NDIM>::arg_t;
    const size_type num_members = static_cast<size_type>(slot.host_args.size());
    const auto& last_arg = slot.host_args.back();
    const size_type total_blocks = last_arg.block_offset + last_arg.n;

#if defined(MRA_ENABLE_CUDA)
    detail::check_cuda_rt(cudaMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    detail::check_hip_rt(hipMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    CALL_KERNEL((detail::convolution_kernel_batched<T, NDIM>), total_blocks, thread_dims, smem_size, stream,
                (num_members, slot.dev_args, total_blocks, K, fac, tol, transr, transs, opnorms, at));
    checkSubmit();

    pool.mark_submitted(slot, stream);
  }

  namespace detail {

    /**
     * Shared by shell0_tt and accumulate_tt in mra/tasks/convolution.h: given
     * the batch_view returned by their own `co_await ttg::device::coop<KeyT>(...)`
     * (which must stay inline in each coroutine -- only the ordinary,
     * non-suspending code below is worth sharing), marshal every member into
     * the current device's pool and submit one combined kernel launch if this
     * task is the batch's leader.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_convolution_batch_leader(
      BatchView& batch,
      ConvolutionBatchPoolRegistry<T, NDIM>& registry,
      size_type K,
      const T fac,
      const T tol,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
      const concepts::TensorView<4> auto& opnorms_view,
      const std::array<bool, 2>& at)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      auto& pool = registry.get(ttg::device::current_device());
      auto& slot = pool.acquire(nb);
      slot.host_args.clear();
      size_type running_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_in       = batch[m].template get<0>();
        auto& m_f        = batch[m].template get<1>();
        auto& m_result   = batch[m].template get<2>();
        auto& m_resnorms = batch[m].template get<3>();
        auto& m_tmp      = batch[m].template get<4>();
        size_type m_n = m_result.dim(0);
        slot.host_args.emplace_back(m_in, m_f, m_result, m_resnorms,
                                    m_tmp.current_device_ptr(), running_offset, m_n);
        running_offset += m_n;
      }
      submit_convolution_kernel_batched<T, NDIM>(pool, slot, K, fac, tol, transr, transs,
                                                  opnorms_view, at, ttg::device::current_stream());
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* explicit instantiation */
  extern template
  void submit_convolution_kernel<double, 3>(
    Key<3> key,
    Key<3> displacement,
    size_type K,
    size_type N,
    const double fac,
    const double tol,
    const SparseTensorView<double, 3+1>& in,
    const SparseTensorView<double, 3+1>& contribution,
    SparseTensorView<double, 3+1>& result,
    SparseTensorView<double, 1>& resnorms,
    const std::array<SparseTensorView<double, 4>, 3>& transr,
    const std::array<SparseTensorView<double, 4>, 3>& transs,
    const DenseTensorView<double, 4>& opnorms,
    const std::array<bool, 2>& at,
    double* tmp,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra

#endif // MRA_KERNELS_CONVOLUTION_H
