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

  /**
   * Heuristic choosing how many thread-blocks ("groups") should split the rank-mu terms
   * of a convolution operator between them: convolution_kernel_partials computes each
   * group's contribution independently (in its own thread-block), and a second, cheap
   * kernel (convolution_kernel_finalize) sums the groups and applies the result.
   *
   * Capped so that nnz * num_groups never exceeds 512 total thread-blocks for
   * convolution_kernel_partials -- a soft cap to avoid over-subdividing when there's
   * already enough parallelism from the number of functions, not a hard occupancy limit
   * (unlike a cooperative launch, this kernel has no residency requirement). Never
   * parallelizes over mu on the host backend, where there is no thread-block concept to
   * split across.
   */
  SCOPE size_type convolution_num_groups(size_type nnz, size_type rank) {
#if defined(MRA_ENABLE_HOST)
    return 1;
#else
    if (nnz == 0) return 1;
    size_type occupancy_cap = std::max<size_type>(1, 512 / nnz);
    size_type groups = std::min(occupancy_cap, rank);
    size_type pow2 = 1;
    while (pow2 * 2 <= groups) pow2 *= 2; // round down to a power of two for a clean tree reduction
    return pow2;
#endif
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


    /**
     * Applies the mu-terms [mu_lo, mu_hi) of the operator's separated rank expansion to `f`,
     * accumulating the R-term sum into `result` and the S-term sum into `resultc` -- SEPARATELY,
     * without folding them together. `apply_conv` (below) calls this with the full [0, rank)
     * range and folds once, for the sequential per-function kernel; convolution_kernel_partials
     * calls it with a sub-range assigned to one thread-block "group" and leaves the fold to
     * convolution_kernel_finalize, which sums all groups' R- and S-sums separately first and
     * folds exactly once -- matching the sequential kernel's computation order term-for-term
     * (R-terms summed in mu order, S-terms summed in mu order, combined once at the end),
     * rather than interleaving each group's own R+S fold, which would reorder how the
     * (typically near-cancelling) R and S contributions combine.
     */
    template<typename T, Dimension NDIM>
    DEVSCOPE bool apply_conv_range(
      int opid,
      size_type K,
      size_type mu_lo,
      size_type mu_hi,
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
      bool result = false;
      // cannot be SHARED because ctors won't run
      std::array<Slice,NDIM> s0 = std::array<Slice,NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
      if (is_team_lead()) {
        work1_k = DenseTensorView<T, NDIM>(work1.data(), K);
        work2_k = DenseTensorView<T, NDIM>(work2.data(), K);
      }

      const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank); // full rank, used to scale the per-term error budget

      T optol = 0.01*tol/rank; // can potentially be a parameter

      f0(s0) = f(s0);

      // TODO: do we care about modified() operators?

      // TODO: why does this fix correctness?!
      result = 0.0;
      resultc = 0.0;

      for (size_type mu = mu_lo; mu < mu_hi; ++mu) {
        T munorm = opnorms(opid, mu, 0, (size_type)NormId::MUnorm);
        if (munorm > optol) {
          T mufac = opnorms(opid, mu, 0, (size_type)NormId::Fac);
          muopxv_fast<T, NDIM>(opid, K, mu, mufac, tol/std::abs(mufac), at, transr, transs, opnorms, f, f0,
                               resultc, result, work1, work2, work1_k, work2_k);
          result = true;
        }
      }
      return result;
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
      const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank);
      apply_conv_range<T, NDIM>(opid, K, 0, rank, fac, tol, transr, transs, opnorms, at,
                                f, f0, resultc, result, work1, work2);
      std::array<Slice,NDIM> s0 = std::array<Slice,NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
      result(s0) += resultc;
      //auto rnorm = normf(result);
      //auto rcnorm = normf(resultc);
      //if (is_team_lead()) printf("MRA APPLY CONV opid %d final result %e resultc %e\n", opid, rnorm, rcnorm);
    }

    /**
     * Combines a (fully summed) convolution result with the pre-existing `in` node and
     * applies the aggressive-screening threshold, writing the final per-function norm to
     * `resnorm_out` (if non-null). Shared by the sequential convolution_kernel_impl and by
     * convolution_kernel_finalize (the second half of the two-kernel grouped path).
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

    /**
     * Computes each function's contribution to `group_partials`, splitting the rank-mu
     * terms into gridDim.y independent "groups" (blockIdx.y) so they can be computed in
     * parallel thread-blocks. This kernel does no cross-block communication (each block
     * only ever touches its own group_partials slot), so unlike a cooperative-launch
     * design its grid size is not limited by device occupancy -- convolution_kernel_finalize
     * (below) sums the groups afterwards in a separate, ordinary kernel launch on the same
     * stream; ordering between the two is guaranteed by the stream, no explicit sync needed.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel_partials(
      size_type K,
      size_type N,
      const T fac,
      const T tol,
      const concepts::TensorView<NDIM+1> auto f_view,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto transr,
      const concepts::TensorViewArray<4, (size_t)NDIM> auto transs,
      const concepts::TensorView<4> auto opnorms_view,
      const std::array<bool, 2> at,
      concepts::TensorView<NDIM+2> auto group_partials,
      concepts::TensorView<NDIM+2> auto group_partials_s,
      concepts::TensorView<2> auto group_partials_mask,
      T* tmp)
    {
      SHARED DenseTensorView<T, NDIM> f0, resultc, work1, work2, group_slot, group_slot_s;
      SHARED DenseTensorView<const T, NDIM> f;

      const size_type num_groups = gridDim.y;
      const size_type groupId = blockIdx.y;
      const size_type K2NDIM = std::pow(K, NDIM);
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);

      for (size_type fnIdx = blockIdx.x; fnIdx < N; fnIdx += gridDim.x) {

        // false by default, set to true if we have a non-zero partial result
        if (is_team_lead()) {
          group_partials_mask(fnIdx, groupId) = false;
        }
        if (group_partials.is_zero(fnIdx)) {
          // no function here, or the function's output is entirely zero: nothing to do
          continue;
        }

        T* block_tmp_ptr = &tmp[(fnIdx * num_groups + groupId) * convolution_tmp_size<NDIM>(K)];
        if (is_team_lead()) {
          f0           = DenseTensorView<T, NDIM>(&block_tmp_ptr[                     0], K);
          resultc      = DenseTensorView<T, NDIM>(&block_tmp_ptr[                K2NDIM], K);
          work1        = DenseTensorView<T, NDIM>(&block_tmp_ptr[              2*K2NDIM], 2*K);
          work2        = DenseTensorView<T, NDIM>(&block_tmp_ptr[  TWOK2NDIM + 2*K2NDIM], 2*K);
          group_slot   = group_partials(fnIdx, groupId);
          group_slot_s = group_partials_s(fnIdx, groupId);
          f            = f_view(fnIdx);
        }
        SYNCTHREADS();

        if (f_view.is_zero(fnIdx)) {
          continue;
        }


        int opid = opnorms_view.dim(0) > 1 ? fnIdx : 0;
        const T opnorm = opnorms_view(opid, 0, 0, (size_type)NormId::Opnorm);
        const T cnorm = mra::normf(f);

        if ((cnorm * opnorm) > (tol / fac)) {
          const T effective_tol = tol / fac / cnorm;
          const size_type rank = (size_type)opnorms_view(opid, 0, 0, (size_type)NormId::Rank);
          const size_type mu_lo = (groupId * rank) / num_groups;
          const size_type mu_hi = ((groupId + 1) * rank) / num_groups;
          if (mu_lo < mu_hi) {
            // R-sum -> group_slot, S-sum -> group_slot_s, left UNFOLDED: convolution_kernel_finalize
            // sums each series across all groups separately and folds exactly once, matching the
            // sequential kernel's computation order (see apply_conv_range's doc comment).
            bool have_update = apply_conv_range<T, NDIM>(opid, K, mu_lo, mu_hi, fac, effective_tol,
                                      transr, transs, opnorms_view, at,
                                      f, f0, group_slot_s, group_slot, work1, work2);
            if (is_team_lead()) {
              group_partials_mask(fnIdx, groupId) = have_update;
            }
          }
        }
      }
    }

    /**
     * Sums the per-group partials computed by convolution_kernel_partials and applies the
     * aggressive-screening threshold / `in` accumulation (convolution_finalize) -- one thread
     * block per function, so no cross-block synchronization is needed here either. The extra
     * kernel-submission cost is negligible next to the parallelism gained in
     * convolution_kernel_partials for operators with large rank.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel_finalize(
      size_type K,
      size_type N,
      size_type num_groups,
      const T fac,
      const T tol,
      const concepts::TensorView<NDIM+1> auto in_view,
      concepts::TensorView<NDIM+1> auto result_view,
      concepts::TensorView<1> auto resnorms,
      concepts::TensorView<NDIM+2> auto group_partials,
      concepts::TensorView<NDIM+2> auto group_partials_s,
      concepts::TensorView<2> auto group_partials_mask)
    {
      SHARED DenseTensorView<T, NDIM> result;
      SHARED DenseTensorView<const T, NDIM> in;

      for (size_type fnIdx = blockIdx.x; fnIdx < N; fnIdx += gridDim.x) {
        if (result_view.is_zero(fnIdx)) {
          if (is_team_lead() && !resnorms.empty()) {
            resnorms[fnIdx] = 0.0;
          }
          continue;
        }
        if (is_team_lead()) {
          in     = in_view(fnIdx);
          result = result_view(fnIdx);
        }
        SYNCTHREADS();

        // Sum the R-series and S-series separately across all groups, then fold exactly
        // once -- matching the sequential kernel's computation order (see apply_conv_range).
        auto next_group = [&](int last_group) -> int {
          for (int g = last_group + 1; g < num_groups; ++g) {
            if (group_partials_mask(fnIdx, g)) return g;
          }
          return num_groups;
        };
        result = 0.0;
        for (int g = next_group(-1); g < num_groups; g = next_group(g)) {
          //auto rnorm = normf(result);
          //auto pgnorm = normf(group_partials(fnIdx, g));
          //if (is_team_lead())
          //  printf("MRA CONV FINALIZE: fnIdx %d group %d result %e group_partial %e\n",
          //         fnIdx, g, rnorm, pgnorm);
          result += group_partials(fnIdx, g);
        }

        int g = next_group(-1);
        if (g < num_groups) {
          auto resultc = group_partials_s(fnIdx, g);
          for (; g < num_groups; g = next_group(g)) {
            //auto rnorm = normf(resultc);
            //auto pgnorm = normf(group_partials_s(fnIdx, g));
            //if (is_team_lead())
            //  printf("MRA CONV FINALIZE: fnIdx %d group %d resultc %e group_partial_s %e\n",
            //         fnIdx, g, rnorm, pgnorm);

            resultc += group_partials_s(fnIdx, g);
          }

          std::array<Slice, NDIM> s0 = std::array<Slice, NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
          result(s0) += resultc;
          //auto rnorm = normf(result);
          //auto rcnorm = normf(resultc);
          //if (is_team_lead()) printf("MRA CONV FINALIZE: fnIdx %d final result %e resultc %e\n", fnIdx, rnorm, rcnorm);
        }

        convolution_finalize<T, NDIM>(fac, tol, in, result,
                                      resnorms.empty() ? nullptr : &resnorms[fnIdx]);
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

  /**
   * Launches convolution_kernel_partials, splitting each function's rank-mu terms across
   * `num_groups` independent thread-block groups (see convolution_num_groups()).
   * `group_partials` is the per-group scratch tensor (sized [N, num_groups, 2K, 2K, 2K],
   * sparse on dim0 -- allocated by the caller using the same SparsityInfo used for
   * `result_view`). Ordinary (non-cooperative) launch: grid size is only limited by the
   * heuristic in convolution_num_groups(), not by device occupancy.
   */
  template <typename T, Dimension NDIM>
  void submit_convolution_kernel_partials(
    size_type K,
    size_type N,
    size_type num_groups,
    const T fac,
    const T tol,
    const concepts::TensorView<NDIM+1> auto& f_view,
    const concepts::TensorViewArray<4, (size_t)NDIM> auto& transr,
    const concepts::TensorViewArray<4, (size_t)NDIM> auto& transs,
    const concepts::TensorView<4> auto& opnorms,
    const std::array<bool, 2>& at,
    concepts::TensorView<NDIM+2> auto& group_partials,
    concepts::TensorView<NDIM+2> auto& group_partials_s,
    concepts::TensorView<2> auto& group_partials_mask,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    Dim3 grid_dims(N, num_groups, 1);

    CALL_KERNEL((detail::convolution_kernel_partials<T, NDIM>), grid_dims, thread_dims, smem_size, stream,
                (K, N, fac, tol, f_view, transr, transs, opnorms, at, group_partials, group_partials_s,
                 group_partials_mask, tmp));
    checkSubmit();
  }

  /**
   * Launches convolution_kernel_finalize, summing the `num_groups` per-group partials
   * computed by submit_convolution_kernel_partials and applying the aggressive-screening
   * threshold / `in` accumulation. Must be submitted on the same stream as
   * submit_convolution_kernel_partials, after it -- stream ordering (not an explicit sync)
   * guarantees group_partials is fully written before this kernel reads it.
   */
  template <typename T, Dimension NDIM>
  void submit_convolution_kernel_finalize(
    size_type K,
    size_type N,
    size_type num_groups,
    const T fac,
    const T tol,
    const concepts::TensorView<NDIM+1> auto& in_view,
    concepts::TensorView<NDIM+1> auto& result_view,
    concepts::TensorView<1> auto& resnorms,
    concepts::TensorView<NDIM+2> auto& group_partials,
    concepts::TensorView<NDIM+2> auto& group_partials_s,
    concepts::TensorView<2> auto& group_partials_mask,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL((detail::convolution_kernel_finalize<T, NDIM>), N, thread_dims, 0, stream,
                (K, N, num_groups, fac, tol, in_view,
                 result_view, resnorms, group_partials,
                 group_partials_s, group_partials_mask));
    checkSubmit();
  }


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

  extern template
  void submit_convolution_kernel_partials<double, 3>(
    size_type K,
    size_type N,
    size_type num_groups,
    const double fac,
    const double tol,
    const SparseTensorView<double, 3+1>& f,
    const std::array<SparseTensorView<double, 4>, 3>& transr,
    const std::array<SparseTensorView<double, 4>, 3>& transs,
    const DenseTensorView<double, 4>& opnorms,
    const std::array<bool, 2>& at,
    SparseTensorView<double, 3+2>& group_partials,
    SparseTensorView<double, 3+2>& group_partials_s,
    DenseTensorView<bool, 2>& group_partials_mask,
    double* tmp,
    ttg::device::Stream stream);

  extern template
  void submit_convolution_kernel_finalize<double, 3>(
    size_type K,
    size_type N,
    size_type num_groups,
    const double fac,
    const double tol,
    const SparseTensorView<double, 3+1>& in,
    SparseTensorView<double, 3+1>& result,
    SparseTensorView<double, 1>& resnorms,
    SparseTensorView<double, 3+2>& group_partials,
    SparseTensorView<double, 3+2>& group_partials_s,
    DenseTensorView<bool, 2>& group_partials_mask,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra

#endif // MRA_KERNELS_CONVOLUTION_H
