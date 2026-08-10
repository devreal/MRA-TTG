#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include <algorithm>
#include <cmath>
#include <numbers>
#include <iostream>
#include <tuple>
#include <vector>
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/ops/mxm.h"
#include "mra/kernels/gaxpy.h"
#include "mra/ops/functions.h"
#include "mra/kernels/transform.h"
#include "mra/misc/conv_mad.h"
#include "mra/misc/batch_size.h"
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

    /**
     * See the comment on multiply_kernel_impl in mra/kernels/multiply.h for
     * why this uses explicitly-named template parameters instead of the
     * abbreviated (concept-constrained `auto`) form used elsewhere: too many
     * compiler-synthesized template parameters in one function trips an nvcc
     * EDG front-end assertion ("check_name_hiding_by_template_parameters").
     */
    template<typename T, Dimension NDIM,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTransr,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTranss,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewF0,
             concepts::TensorView<NDIM> ViewResultc,
             concepts::TensorView<NDIM> ViewResult,
             concepts::TensorView<NDIM> ViewWork1,
             concepts::TensorView<NDIM> ViewWork2,
             concepts::TensorView<NDIM> ViewWork1k,
             concepts::TensorView<NDIM> ViewWork2k>
    DEVSCOPE void muopxv_fast(
      int opid,
      size_type K,
      const size_type mu,
      const T mufac,
      const T tol,
      const std::array<bool, 2>& at,
      const ViewTransr& transr,
      const ViewTranss& transs,
      const ViewOpnorms& opnorms,
      ViewF& f,
      ViewF0& f0,
      ViewResultc& resultc,
      ViewResult& result,
      ViewWork1& work1,
      ViewWork2& work2,
      ViewWork1k& work1_k,
      ViewWork2k& work2_k)
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

    /** See the comment on muopxv_fast above. */
    template <typename T, Dimension NDIM,
              concepts::TensorViewArray<4, (size_t)NDIM> ViewTransr,
              concepts::TensorViewArray<4, (size_t)NDIM> ViewTranss,
              concepts::TensorView<4> ViewOpnorms,
              concepts::TensorView<NDIM> ViewIn,
              concepts::TensorView<NDIM> ViewF,
              concepts::TensorView<NDIM> ViewF0,
              concepts::TensorView<NDIM> ViewResultc,
              concepts::TensorView<NDIM> ViewResult,
              concepts::TensorView<NDIM> ViewWork1,
              concepts::TensorView<NDIM> ViewWork2>
    DEVSCOPE void convolution_kernel_impl(
      Key<NDIM> key,
      int opid,
      Key<NDIM> displacement,
      size_type K,
      const T fac,
      const T tol,
      const ViewTransr& transr,
      const ViewTranss& transs,
      const ViewOpnorms& opnorms,
      const std::array<bool, 2>& at,
      ViewIn& in,
      ViewF& f,
      ViewF0& f0,
      ViewResultc& resultc,
      ViewResult& result,  // size K, stores the sum
      ViewWork1& work1,
      ViewWork2& work2,
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
      size_type N,
      size_type tmp_pos)
    {
      SHARED DenseTensorView<T, NDIM> f0, resultc, work1, work2, result;
      SHARED DenseTensorView<const T, NDIM> f, in;
      SHARED size_type i;

      if (is_team_lead()) {
        // result_view has exactly n_nonzero non-zero entries, so this
        // always finds a valid function id -- see submit_convolution_kernel.
        // Excluded (zero) functions' resnorms entries are pre-filled with
        // 0.0 host-side (see mra/tasks/convolution.h) since no block visits
        // them here.
        i = find_nth_nonzero(N, tmp_pos, result_view);

        const size_type K2NDIM = std::pow(K, NDIM);
        const size_type TWOK2NDIM = std::pow(2*K, NDIM);
        T* block_tmp_ptr = &tmp[tmp_pos*convolution_tmp_size<NDIM>(K)];
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

#if defined(MRA_CHECK_NORMS)
    /**
     * Debug-only, launched as its OWN kernel (grid of 1 block) immediately
     * before convolution_kernel below, on the same stream -- see
     * compress_verify_sparsity_kernel's comment (mra/kernels/compress.h) for
     * why this must not be inlined into convolution_kernel itself (a
     * same-launch race against find_nth_nonzero's assert in other blocks
     * would let this check's own diagnostic go unprinted).
     *
     * n_nonzero was computed host-side (mra/tasks/convolution.h's `sparsity`,
     * = nonzero_if_any(in_node, contribution) for accumulate_tt, or a
     * single-view scan of f/contribution for shell0_tt) and used to both
     * allocate result_view's (out's) own sparsity and size this launch's
     * grid/tmp buffer -- so a single-view scan of result_view's own device
     * bitfield (not a fresh union of in_view/f_view) is the right
     * cross-check here, mirroring compress's p_in pattern (a single view
     * that's already the union, rather than reconstruct's node/from_parent
     * pair). Also verifies f_view (the contribution being absorbed this
     * step) doesn't have non-zero positions outside result_view's coverage
     * -- those would never be visited by this launch.
     */
    template<typename T, Dimension NDIM>
    GLOBALSCOPE void convolution_verify_sparsity_kernel_single(
      Key<NDIM> key,
      size_type N,
      size_type n_nonzero,
      const concepts::TensorView<NDIM+1> auto f_view,
      concepts::TensorView<NDIM+1> auto result_view)
    {
      if (is_team_lead()) {
        const size_type actual = count_union_nonzero(N, result_view);
        if (actual != n_nonzero) {
          THROWF("convolution_kernel: n_nonzero mismatch at level %d: host=%llu device=%llu (N=%llu)\n",
                 (int)key.level(), (unsigned long long)n_nonzero, (unsigned long long)actual, (unsigned long long)N);
        }
        const size_type bad_f = find_nonzero_not_in_union(N, f_view, result_view);
        if (bad_f != N) {
          THROWF("convolution_kernel: f_view non-zero at fnid=%llu (level %d) outside "
                 "result_view's coverage -- that position is never visited by this launch\n",
                 (unsigned long long)bad_f, (int)key.level());
        }
      }
    }
#endif // MRA_CHECK_NORMS

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel(
      Key<NDIM> key,
      Key<NDIM> displacement,
      size_type K,
      size_type N,
      size_type n_nonzero,
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
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        convolution_process_one<T, NDIM>(key, displacement, K, fac, tol, transr, transs, opnorms_view, at,
                                         in_view, f_view, result_view, resnorms, tmp, N, pos);
      }
    }

  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_convolution_kernel(
    Key<NDIM> key,
	  Key<NDIM> displacement,
    size_type K,
    size_type N,
    size_type n_nonzero,
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
#if defined(MRA_CHECK_NORMS)
    // Separate, prior kernel on the same stream -- see
    // convolution_verify_sparsity_kernel_single's comment for why this must
    // not be inlined into convolution_kernel itself.
    CALL_KERNEL((detail::convolution_verify_sparsity_kernel_single<T, NDIM>), 1, 32, 0, stream,
                (key, N, n_nonzero, f_view, result_view));
    checkSubmit();
#endif // MRA_CHECK_NORMS

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    //CONFIGURE_KERNEL((detail::convolution_kernel<T, NDIM>), smem_size);
    CALL_KERNEL((detail::convolution_kernel<T, NDIM>), n_nonzero, thread_dims, smem_size, stream,
                (key, displacement, K, N, n_nonzero, fac, tol, in_view, f_view, result_view,
                 resnorms, transr, transs, opnorms, at, tmp));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the convolution kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/convolution.h. Batching is unrestricted:
   * set_batch_matcher's predicate always returns true, so any tasks of the
   * same TT can end up in the same batch (up to max_batch_size), regardless of
   * level or displacement. Only K and fac are truly global constants shared by
   * every possible member; everything else that could vary -- tol, at,
   * transr, transs, opnorms -- travels PER MEMBER in the tuple below.
   * Level-only matching (an earlier, narrower version of this) already made
   * accumulate_tt's batches mostly size 1 in practice (two tasks rarely reach
   * the same level, let alone the same displacement, at the same moment); the
   * unrestricted matcher maximizes batch sizes at the cost of a few hundred
   * extra bytes of view/scalar descriptors per member (not the underlying
   * filter-matrix data, which TensorView only points to) and a device kernel
   * that no longer gets to assume anything is uniform across a launch besides
   * K/fac.
   */
  namespace detail {

    /**
     * Per-member argument bundle for convolution_kernel_batched. in_view/f_view
     * are only ever read inside the kernel; result_view/resnorms_view are
     * written. Constness is enforced at the point of use (via a `const auto&`
     * local binding before calling operator()) rather than in the tuple's
     * element types, since there is no converting constructor from
     * SparseTensorView<T,...> to SparseTensorView<const T,...> to build the
     * latter from the views the surrounding task already holds. tol/at/transr/
     * transs/opnorms are all per-member (see the batching-support comment
     * above for why) -- only K/fac stay batch-wide kernel parameters.
     */
    template <typename T, Dimension NDIM>
    using ConvolutionBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,             // in_view
      SparseTensorView<T, NDIM+1>,             // f_view
      SparseTensorView<T, NDIM+1>,             // result_view
      DenseTensorView<T, 1>,                   // resnorms_view
      T*,                                      // tmp
      size_type,                               // n: number of blocks (functions) this member contributes
      std::array<DenseTensorView<T, 4>, NDIM>, // transr (this member's own operator data)
      std::array<DenseTensorView<T, 4>, NDIM>, // transs
      DenseTensorView<T, 4>,                   // opnorms
      T,                                       // tol: this member's own truncate_tol(...)
      std::array<bool, 2>,                     // at: this member's own apply-terms flags
      size_type                                // sparsity_offset: this member's byte range start in the aggregated sparsity staging buffer (see convolution_scatter_sparsity_kernel)
    >;

    /* Named indices into ConvolutionBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct ConvolutionBatchArgIdx {
      static constexpr std::size_t in_view         = 0;
      static constexpr std::size_t f_view          = 1;
      static constexpr std::size_t result_view     = 2;
      static constexpr std::size_t resnorms_view   = 3;
      static constexpr std::size_t tmp             = 4;
      static constexpr std::size_t n               = 5;
      static constexpr std::size_t transr          = 6;
      static constexpr std::size_t transs          = 7;
      static constexpr std::size_t opnorms         = 8;
      static constexpr std::size_t tol             = 9;
      static constexpr std::size_t at              = 10;
      static constexpr std::size_t sparsity_offset = 11;
    };

    /**
     * One combined launch covering every non-zero function position across
     * all members of the batch, flattened into a single 1D grid of size
     * total_nonzero -- no padding blocks for members with fewer functions
     * than others, no blocks wasted on positions already known to be zero.
     * member_offsets (size num_members+1) names, for a given global grid
     * position, which member a block belongs to and that member's own
     * compact local position (find_member_for_pos, an O(num_members) scan
     * -- cheap since num_members is small); convolution_process_one's team
     * lead then turns that local position into a real function id via an
     * on-device scan of that member's own result_view sparsity
     * (find_nth_nonzero). This makes convolution_kernel_batched a thin
     * wrapper: look up one work item and hand off to the exact same
     * per-(node, function) body convolution_kernel itself uses
     * (convolution_process_one, defined above with convolution_kernel_impl).
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel_batched(
      ConvolutionBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets,        // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      size_type K,
      const T fac)
    {
      using idx = ConvolutionBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;

      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        const size_type member_N = std::get<idx::n>(arg);

        convolution_process_one<T, NDIM>(Key<NDIM>{}, Key<NDIM>{}, K, fac, std::get<idx::tol>(arg),
                                         std::get<idx::transr>(arg), std::get<idx::transs>(arg),
                                         std::get<idx::opnorms>(arg), std::get<idx::at>(arg),
                                         std::get<idx::in_view>(arg), std::get<idx::f_view>(arg),
                                         std::get<idx::result_view>(arg), std::get<idx::resnorms_view>(arg),
                                         std::get<idx::tmp>(arg), member_N, local_pos);
      }
    }

    /**
     * Scatters pre-aggregated per-member sparsity bytes into each member's own
     * result tensor's inline bitfield. The bytes themselves were computed
     * host-side (from each member's RangeSparsityBase-backed Tensor, via
     * detail::sparsity_to_bytes in submit_convolution_batch_leader below),
     * assembled into one contiguous pinned buffer, and copied to `sparsity`
     * with a single H2D transfer -- replacing what would otherwise be one
     * SparsityManager/MockTensor allocation + copy per member. Launched on the
     * same stream immediately before convolution_kernel_batched, so by the
     * time that kernel reads result_view.is_zero(i)/is_nonzero(i) the bytes
     * are already in place; stream ordering alone makes this correct, no
     * extra synchronization needed.
     */
    template <typename T, Dimension NDIM>
    GLOBALSCOPE void convolution_scatter_sparsity_kernel(
      ConvolutionBatchArg<T, NDIM>* args,        // device ptr, size == gridDim.x
      const SparsityState* sparsity)             // device ptr, aggregated batch-wide staging buffer
    {
      using idx = ConvolutionBatchArgIdx;

      const size_type member = blockIdx.x;
      auto& arg = args[member];
      auto& result_view = std::get<idx::result_view>(arg);
      const size_type n = std::get<idx::n>(arg);
      const size_type offset = std::get<idx::sparsity_offset>(arg);

      for (size_type i = threadIdx.x; i < n; i += blockDim.x) {
        result_view.set_state(i, sparsity[offset + i]);
      }
    }

    /**
     * Prunes out_view's device-side sparsity bitfield for any function whose
     * *computed* resnorms entry is exactly zero. mra/tasks/convolution.h's
     * shell0_tt/accumulate_tt call out.set_zero(i) host-side for these same
     * positions right after resnorms comes back to host -- but
     * FunctionNodeBase::set_zero() only updates the host RangeSparsityBase
     * ranges (see mra/tensor/sparsity.h), never the device inline bitfield
     * that was already populated earlier (via SparsityManager or the batch
     * scatter kernel, before this node's actual coefficient values were even
     * computed). Without this device-side counterpart, out_view keeps
     * reporting non-zero for positions the host has since pruned, so a
     * downstream consumer's device-side union scan (e.g. reconstruct's,
     * treating this node as its "node" input) finds MORE non-zero positions
     * than the host ever expected. Must run while resnorms/out_view are
     * still device-resident, i.e. before co_await
     * ttg::device::wait(resnorms.buffer()) brings resnorms back to host.
     * Single block, N small -- not meant for the hot path beyond this use.
     */
    template <Dimension NDIM>
    GLOBALSCOPE void convolution_prune_zero_norm_kernel(
      size_type N,
      const concepts::TensorView<1> auto resnorms,
      concepts::TensorView<NDIM+1> auto out_view)
    {
      for (size_type i = threadIdx.x; i < N; i += blockDim.x) {
        if (out_view.is_nonzero(i) && resnorms[i] == 0.0) {
          out_view.set_state(i, SparsityState::ALLOCATED);
        }
      }
    }

#if defined(MRA_CHECK_NORMS)
    /**
     * Debug-only: cross-checks, for every member, that the flattened launch
     * grid's slice for that member (member_offsets[m+1] - member_offsets[m],
     * a running sum of each member's own host-computed n_nonzero -- see
     * submit_convolution_batch_leader) agrees with a fresh on-device scan of
     * that same member's result_view (a single-view scan suffices -- see
     * convolution_verify_sparsity_kernel_single's comment). One block per
     * member, same style as convolution_scatter_sparsity_kernel above.
     * Launched (gated by MRA_CHECK_NORMS) immediately AFTER
     * convolution_scatter_sparsity_kernel and before
     * convolution_kernel_batched in submit_convolution_kernel_batched --
     * unlike reconstruct's batched check, this one must run after the
     * scatter, since result_view's own device bitfield (what's being
     * checked here) is exactly what the scatter just wrote.
     */
    template <typename T, Dimension NDIM>
    GLOBALSCOPE void convolution_verify_sparsity_kernel(
      ConvolutionBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets)        // device ptr, size == num_members+1
    {
      using idx = ConvolutionBatchArgIdx;

      const size_type member = blockIdx.x;
      if (is_team_lead()) {
        auto& arg = args[member];
        const size_type member_N = std::get<idx::n>(arg);
        const size_type expected = member_offsets[member + 1] - member_offsets[member];
        auto& result_view = std::get<idx::result_view>(arg);
        const size_type actual = count_union_nonzero(member_N, result_view);
        if (actual != expected) {
          THROWF("convolution_kernel_batched: n_nonzero mismatch for batch member %llu: "
                 "host=%llu device=%llu (N=%llu)\n",
                 (unsigned long long)member, (unsigned long long)expected,
                 (unsigned long long)actual, (unsigned long long)member_N);
        }
        auto& f_view = std::get<idx::f_view>(arg);
        const size_type bad_f = find_nonzero_not_in_union(member_N, f_view, result_view);
        if (bad_f != member_N) {
          THROWF("convolution_kernel_batched: f_view non-zero at fnid=%llu for batch member %llu "
                 "outside result_view's coverage -- that position is never visited by this launch\n",
                 (unsigned long long)bad_f, (unsigned long long)member);
        }
      }
    }
#endif // MRA_CHECK_NORMS

  } // namespace detail

  /**
   * See detail::convolution_prune_zero_norm_kernel's comment. Called by both
   * shell0_tt and accumulate_tt (mra/tasks/convolution.h), on the same
   * stream as the kernel that just computed resnorms/out, before that
   * stream's data is brought back to host.
   */
  template <Dimension NDIM>
  void submit_convolution_prune_zero_norm_kernel(
    size_type N,
    const concepts::TensorView<1> auto& resnorms,
    concepts::TensorView<NDIM+1> auto& out_view,
    ttg::device::Stream stream)
  {
    CALL_KERNEL((detail::convolution_prune_zero_norm_kernel<NDIM>), 1, 32, 0, stream, (N, resnorms, out_view));
    checkSubmit();
  }

  /**
   * Batched counterpart of submit_convolution_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.host_args (by the
   * caller, via detail::submit_convolution_batch_leader below), sharing only
   * (K, fac) across the whole batch -- tol/at/transr/transs/opnorms are
   * per-member, already inside slot.host_args. Grid is 1D over total_nonzero
   * -- see convolution_kernel_batched's comment for why. `sparsity_pool`/
   * `sparsity_slot` carry the batch-wide aggregated sparsity bytes assembled
   * by submit_convolution_batch_leader; see convolution_scatter_sparsity_kernel.
   * `offset_pool`/`offset_slot` carry the small per-member offsets array
   * (size num_members+1), also assembled by submit_convolution_batch_leader.
   */
  template <typename T, Dimension NDIM>
  void submit_convolution_kernel_batched(
    detail::BatchPool<detail::ConvolutionBatchArg<T, NDIM>>& pool,
    typename detail::BatchPool<detail::ConvolutionBatchArg<T, NDIM>>::slot_t& slot,
    detail::BatchPool<detail::SparsityState>& sparsity_pool,
    typename detail::BatchPool<detail::SparsityState>::slot_t& sparsity_slot,
    detail::BatchPool<size_type>& offset_pool,
    typename detail::BatchPool<size_type>::slot_t& offset_slot,
    size_type total_nonzero,
    size_type K,
    const T fac,
    ttg::device::Stream stream)
  {
    using idx = detail::ConvolutionBatchArgIdx;
    using arg_t = detail::ConvolutionBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.host_args.size());

#if defined(MRA_ENABLE_CUDA)
    detail::check_cuda_rt(cudaMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    detail::check_cuda_rt(cudaMemcpyAsync(sparsity_slot.dev_args, sparsity_slot.host_args.data(),
                                          sparsity_slot.host_args.size()*sizeof(detail::SparsityState),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    detail::check_cuda_rt(cudaMemcpyAsync(offset_slot.dev_args, offset_slot.host_args.data(),
                                          offset_slot.host_args.size()*sizeof(size_type),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    detail::check_hip_rt(hipMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    detail::check_hip_rt(hipMemcpyAsync(sparsity_slot.dev_args, sparsity_slot.host_args.data(),
                                        sparsity_slot.host_args.size()*sizeof(detail::SparsityState),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    detail::check_hip_rt(hipMemcpyAsync(offset_slot.dev_args, offset_slot.host_args.data(),
                                        offset_slot.host_args.size()*sizeof(size_type),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif

    // Scatter each member's aggregated sparsity bytes into its own result
    // tensor's inline bitfield; same stream as the main kernel below, so
    // stream ordering guarantees it completes first.
    CALL_KERNEL((detail::convolution_scatter_sparsity_kernel<T, NDIM>), num_members, 32, 0, stream,
                (slot.dev_args, sparsity_slot.dev_args));
    checkSubmit();

#if defined(MRA_CHECK_NORMS)
    // Debug-only: verify the flattened grid's per-member slice (derived from
    // each member's host-computed n_nonzero) still matches a fresh on-device
    // scan of that member's result_view -- must run AFTER the scatter above,
    // since result_view's own device bitfield is exactly what that scatter
    // just wrote. See convolution_verify_sparsity_kernel's comment.
    CALL_KERNEL((detail::convolution_verify_sparsity_kernel<T, NDIM>), num_members, 32, 0, stream,
                (slot.dev_args, offset_slot.dev_args));
    checkSubmit();
#endif // MRA_CHECK_NORMS

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    CALL_KERNEL((detail::convolution_kernel_batched<T, NDIM>), total_nonzero, thread_dims, smem_size, stream,
                (slot.dev_args, offset_slot.dev_args, num_members, total_nonzero, K, fac));
    checkSubmit();

    pool.mark_submitted(slot, stream);
    sparsity_pool.mark_submitted(sparsity_slot, stream);
    offset_pool.mark_submitted(offset_slot, stream);
  }

  namespace detail {

    /**
     * Shared by shell0_tt and accumulate_tt in mra/tasks/convolution.h: given
     * the batch_view returned by their own `co_await ttg::device::coop<KeyT>(...)`
     * (which must stay inline in each coroutine -- only the ordinary,
     * non-suspending code below is worth sharing), marshal every member into
     * the current device's pool and submit one combined kernel launch if this
     * task is the batch's leader. Each member's OWN tol/transr/transs/opnorms/at
     * are read from its own coop() args (get<5..9>), not shared across the
     * batch -- see the batching-support comment on ConvolutionBatchArg for why.
     *
     * Sparsity: each member also passes its own `out` tensor (get<10>(), the
     * real Tensor -- not just its view) through coop(), so this leader can
     * read its RangeSparsityBase-backed sparsity directly (no per-member
     * SparsityManager/MockTensor allocation) and assemble every member's
     * bytes into one pinned staging buffer (from the same process-wide
     * sparsity pool used by SparsityManager, see sparsitymanager.h), copied
     * to the device in a single transfer by submit_convolution_kernel_batched
     * instead of one small H2D copy per member.
     *
     * Flattening: each member also passes its own n_nonzero (get<11>())
     * through coop() -- already computed independently of batching,
     * per-member, in mra/tasks/convolution.h. The leader turns those into a
     * tiny (num_members+1)-entry offsets array (a running sum of
     * n_nonzero), so the combined kernel can launch exactly total_nonzero
     * blocks and each one can find its member with an O(num_members) scan
     * (find_member_for_pos) instead of indexing a per-function list -- see
     * convolution_kernel_batched.
     *
     * `total_functions` is the whole FunctionSet's total function count
     * (fixed for this operation's entire run, unlike any single member's
     * own structural N) -- used only to size the sparsity-byte staging
     * pool's first allocation to a fixed upper bound
     * (max_batch_size * total_functions), so it never needs to grow after
     * that.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_convolution_batch_leader(
      BatchView& batch,
      BatchPoolRegistry<ConvolutionBatchArg<T, NDIM>>& registry,
      size_type K,
      const T fac,
      size_type total_functions)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      auto& pool = registry.get(ttg::device::current_device());
      auto& slot = pool.acquire(registry.get_max_batch_size()); // allocate space for full batch
      slot.host_args.clear();

      // Offsets slot: always acquired at max_batch_size+1 capacity (not
      // nb+1), so its device buffer is allocated once, on first use, and
      // never resized after that.
      auto& offset_pool = member_offset_pool_registry().get(ttg::device::current_device());
      auto& offset_slot = offset_pool.acquire(registry.get_max_batch_size() + 1);
      offset_slot.host_args.resize(nb + 1);
      offset_slot.host_args[0] = 0;

      // Sparsity-byte slot: acquired at a fixed upper bound (every member
      // contributes at most total_functions bytes, and there are at most
      // max_batch_size members), not the exact total_sparsity_bytes needed
      // this launch -- so its device buffer is allocated once and never
      // resized, even though the exact byte count varies launch to launch.
      const size_type max_sparsity_bytes =
          static_cast<size_type>(registry.get_max_batch_size()) * total_functions;
      size_type total_sparsity_bytes = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        total_sparsity_bytes += static_cast<size_type>(batch[m].template get<2>().dim(0));
      }
      auto& sparsity_pool = sparsity_pool_registry().get(ttg::device::current_device());
      auto& sparsity_slot = sparsity_pool.acquire(max_sparsity_bytes);
      sparsity_slot.host_args.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_in       = batch[m].template get<0>();
        auto& m_f        = batch[m].template get<1>();
        auto& m_result   = batch[m].template get<2>();
        auto& m_resnorms = batch[m].template get<3>();
        auto& m_tmp      = batch[m].template get<4>();
        auto& m_transr   = batch[m].template get<5>();
        auto& m_transs   = batch[m].template get<6>();
        auto& m_opnorms  = batch[m].template get<7>();
        auto& m_tol      = batch[m].template get<8>();
        auto& m_at       = batch[m].template get<9>();
        auto& m_out      = batch[m].template get<10>(); // real out tensor, for its RangeSparsityBase sparsity
        const size_type m_n_nonzero = batch[m].template get<11>();
        const size_type n = static_cast<size_type>(m_result.dim(0)); // structural N

        sparsity_to_bytes(m_out.sparsity(), &sparsity_slot.host_args[sparsity_offset], n);

        slot.host_args.emplace_back(m_in, m_f, m_result, m_resnorms,
                                    m_tmp.current_device_ptr(), n,
                                    m_transr, m_transs, m_opnorms, m_tol, m_at, sparsity_offset);
        sparsity_offset += n;

        offset_slot.host_args[m + 1] = offset_slot.host_args[m] + m_n_nonzero;
      }
      const size_type total_nonzero = offset_slot.host_args[nb];
      submit_convolution_kernel_batched<T, NDIM>(pool, slot, sparsity_pool, sparsity_slot,
                                                  offset_pool, offset_slot, total_nonzero,
                                                  K, fac, ttg::device::current_stream());
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
    size_type n_nonzero,
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
