#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include <algorithm>
#include <cmath>
#include <numbers>
#include <iostream>
#include <tuple>
#include <vector>
#include "mra/misc/device_batch_pool.h"
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

    /**
     * Processes one (node, function-index) pair: the per-block body shared by
     * both the unbatched convolution_kernel below and the batched
     * convolution_kernel_batched further down -- there is exactly one copy of
     * this logic to maintain instead of two near-identical grid-stride loops.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void convolution_process_one(
      Key<NDIM> key,
      Key<NDIM> displacement,
      size_type K,
      const T fac,
      const T tol,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
      const concepts::TensorView<4> auto& opnorms_view,
      const std::array<bool, 2>& at,
      const concepts::TensorView<NDIM+1> auto& in_view,
      const concepts::TensorView<NDIM+1> auto& f_view,
      concepts::TensorView<NDIM+1> auto& result_view,
      concepts::TensorView<1> auto& resnorms,
      T* tmp,
      size_type i)
    {
      SHARED DenseTensorView<T, NDIM> f0, resultc, work1, work2, result;
      SHARED DenseTensorView<const T, NDIM> f, in;

      if (result_view.is_zero(i)) {
        // nothing to do
        if (is_team_lead() && !resnorms.empty()) {
          resnorms[i] = 0.0;
        }
        return;
      }
      if (is_team_lead()) {
        const size_type K2NDIM = std::pow(K, NDIM);
        const size_type TWOK2NDIM = std::pow(2*K, NDIM);
        T* block_tmp_ptr = &tmp[i*convolution_tmp_size<NDIM>(K)];
        // construct temporaries and pass them to conv_transform
        f0        = DenseTensorView<T, NDIM>(&block_tmp_ptr[                     0], K);
        resultc   = DenseTensorView<T, NDIM>(&block_tmp_ptr[                K2NDIM], K);
        work1     = DenseTensorView<T, NDIM>(&block_tmp_ptr[              2*K2NDIM], 2*K);
        work2     = DenseTensorView<T, NDIM>(&block_tmp_ptr[  TWOK2NDIM + 2*K2NDIM], 2*K);
        in     = in_view(i);
        f      = f_view(i);
        result = result_view(i);
      }
      SYNCTHREADS();
      if (f_view.is_zero(i)) {
        /* copy input to output */
        result = in;
        if (!resnorms.empty()) {
          auto resnorm = normf(result);
          if (is_team_lead()) {
            resnorms[i] = resnorm;
          }
        }
        return;
      }

      int opid = opnorms_view.dim(0) > 1 ? static_cast<int>(i) : 0; // TODO: this is a bit hacky, can we do better?

      convolution_kernel_impl<T, NDIM>(key, opid, displacement, K, fac, tol,
                                       transr, transs, opnorms_view, at, in, f, f0,
                                       resultc, result, work1, work2,
                                       resnorms.empty() ? nullptr : &resnorms[i]);
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
      for (size_type blockId = blockIdx.x; blockId < N; blockId += gridDim.x) {
        convolution_process_one<T, NDIM>(key, displacement, K, fac, tol, transr, transs, opnorms_view, at,
                                         in_view, f_view, result_view, resnorms, tmp, blockId);
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
   * done across tasks at the same tree level, which guarantees K/fac/tol/at
   * are bitwise identical for every member (see mra/ops/functions.h::truncate_tol
   * and accumulate_tt's `at = {true, source.level()>0}` -- source.level() ==
   * dest.level() always, so same dest level implies same `at` too). Matching
   * is deliberately NOT narrowed to same displacement as well: requiring the
   * same (level, displacement) -- and thus the same transr/transs/opnorms,
   * per mra/misc/conv_mad.h::GaussianConvolutionOperator::get_op's cache key
   * -- made accumulate_tt's batches mostly size 1 in practice (two tasks
   * rarely apply the same neighbor offset at the same moment). Instead,
   * transr/transs/opnorms travel PER MEMBER in the tuple below; the extra
   * data is just a few small view descriptors (~200 bytes/member, not the
   * underlying filter-matrix data, which TensorView only points to), a good
   * trade for letting same-level tasks with different displacements batch
   * together.
   */
  namespace detail {

    /**
     * Per-member argument bundle for convolution_kernel_batched. in_view/f_view
     * are only ever read inside the kernel; result_view/resnorms_view are
     * written. Constness is enforced at the point of use (via a `const auto&`
     * local binding before calling operator()) rather than in the tuple's
     * element types, since there is no converting constructor from
     * SparseTensorView<T,...> to SparseTensorView<const T,...> to build the
     * latter from the views the surrounding task already holds. transr/transs/
     * opnorms are per-member (see the batching-support comment above for why).
     */
    template <typename T, Dimension NDIM>
    using ConvolutionBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,             // in_view
      SparseTensorView<T, NDIM+1>,             // f_view
      SparseTensorView<T, NDIM+1>,             // result_view
      DenseTensorView<T, 1>,                   // resnorms_view
      T*,                                      // tmp
      size_type,                               // n: number of blocks (functions) this member contributes
      std::array<SparseTensorView<T, 4>, NDIM>, // transr (this member's own operator data)
      std::array<SparseTensorView<T, 4>, NDIM>, // transs
      DenseTensorView<T, 4>                    // opnorms
    >;

    /* Named indices into ConvolutionBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct ConvolutionBatchArgIdx {
      static constexpr std::size_t in_view       = 0;
      static constexpr std::size_t f_view        = 1;
      static constexpr std::size_t result_view   = 2;
      static constexpr std::size_t resnorms_view = 3;
      static constexpr std::size_t tmp           = 4;
      static constexpr std::size_t n             = 5;
      static constexpr std::size_t transr        = 6;
      static constexpr std::size_t transs        = 7;
      static constexpr std::size_t opnorms       = 8;
    };

    /**
     * One combined launch covering `num_members` independent nodes sharing one
     * (K, fac, tol, at) -- transr/transs/opnorms are per-member, not shared
     * (see the batching-support comment above). Grid is 3D: blockIdx.y selects
     * the batch member (gridDim.y == num_members), blockIdx.x the function
     * index within that member (gridDim.x == the largest N_m across the whole
     * batch); members with fewer than gridDim.x functions simply have their
     * higher-x blocks do nothing (the grid-stride loop below exits immediately
     * once i >= n). No block-to-member scan is needed since the 3D grid already
     * gives every block its (member, local index) pair directly. This makes
     * convolution_kernel_batched a thin wrapper: unpack one member's args and
     * hand off to the exact same per-(node, function) body convolution_kernel
     * itself uses (convolution_process_one, defined above with
     * convolution_kernel_impl).
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel_batched(
      ConvolutionBatchArg<T, NDIM>* args,   // device ptr, size == gridDim.y
      size_type K,
      const T fac,
      const T tol,
      const std::array<bool, 2> at)
    {
      using idx = ConvolutionBatchArgIdx;

      const size_type member = blockIdx.y;
      auto& arg = args[member];
      const size_type n = std::get<idx::n>(arg);

      for (size_type i = blockIdx.x; i < n; i += gridDim.x) {
        convolution_process_one<T, NDIM>(Key<NDIM>{}, Key<NDIM>{}, K, fac, tol,
                                         std::get<idx::transr>(arg), std::get<idx::transs>(arg),
                                         std::get<idx::opnorms>(arg), at,
                                         std::get<idx::in_view>(arg), std::get<idx::f_view>(arg),
                                         std::get<idx::result_view>(arg), std::get<idx::resnorms_view>(arg),
                                         std::get<idx::tmp>(arg), i);
      }
    }

  } // namespace detail

  /**
   * Batched counterpart of submit_convolution_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.host_args (by the
   * caller, via detail::submit_convolution_batch_leader below), sharing
   * (K, fac, tol, at) across the whole batch -- transr/transs/opnorms are
   * per-member, already inside slot.host_args. Grid is (max_n, num_members, 1)
   * -- see convolution_kernel_batched's comment for why.
   */
  template <typename T, Dimension NDIM>
  void submit_convolution_kernel_batched(
    detail::BatchPool<detail::ConvolutionBatchArg<T, NDIM>>& pool,
    typename detail::BatchPool<detail::ConvolutionBatchArg<T, NDIM>>::slot_t& slot,
    size_type K,
    const T fac,
    const T tol,
    const std::array<bool, 2>& at,
    ttg::device::Stream stream)
  {
    using idx = detail::ConvolutionBatchArgIdx;
    using arg_t = detail::ConvolutionBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.host_args.size());
    size_type max_n = 0;
    for (const auto& arg : slot.host_args) {
      max_n = std::max(max_n, std::get<idx::n>(arg));
    }

#if defined(MRA_ENABLE_CUDA)
    detail::check_cuda_rt(cudaMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    detail::check_hip_rt(hipMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    Dim3 grid_dims(max_n, num_members, 1);

    CALL_KERNEL((detail::convolution_kernel_batched<T, NDIM>), grid_dims, thread_dims, smem_size, stream,
                (slot.dev_args, K, fac, tol, at));
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
     * task is the batch's leader. Each member's OWN transr/transs/opnorms are
     * read from its own coop() args (get<5..7>), not shared across the batch --
     * see the batching-support comment on ConvolutionBatchArg for why.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_convolution_batch_leader(
      BatchView& batch,
      BatchPoolRegistry<ConvolutionBatchArg<T, NDIM>>& registry,
      size_type K,
      const T fac,
      const T tol,
      const std::array<bool, 2>& at)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      auto& pool = registry.get(ttg::device::current_device());
      auto& slot = pool.acquire(nb);
      slot.host_args.clear();
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_in       = batch[m].template get<0>();
        auto& m_f        = batch[m].template get<1>();
        auto& m_result   = batch[m].template get<2>();
        auto& m_resnorms = batch[m].template get<3>();
        auto& m_tmp      = batch[m].template get<4>();
        auto& m_transr   = batch[m].template get<5>();
        auto& m_transs   = batch[m].template get<6>();
        auto& m_opnorms  = batch[m].template get<7>();
        slot.host_args.emplace_back(m_in, m_f, m_result, m_resnorms,
                                    m_tmp.current_device_ptr(), static_cast<size_type>(m_result.dim(0)),
                                    m_transr, m_transs, m_opnorms);
      }
      submit_convolution_kernel_batched<T, NDIM>(pool, slot, K, fac, tol, at, ttg::device::current_stream());
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
