#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include <algorithm>
#include <cmath>
#include <numbers>
#include <iostream>
#include <tuple>
#include <vector>
#include <cstdlib>
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/ops/mxm.h"
#include "mra/kernels/gaxpy.h"
#include "mra/ops/functions.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/impl/convolution_accel.h"
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
    // f0, resultc: K-sized. work1, work2, staged_in: 2K-sized. staged_in
    // holds a copy of this function's "in" (the running sum so far), taken
    // before any write into `result` -- which, when accumulate_tt reuses its
    // input node's own storage as the output (see its comment), aliases the
    // very same memory `in` reads from. See convolution_process_one's
    // staged_in comment for why the copy has to happen there.
    return 4*TWOK2NDIM + 2*K2NDIM; // f0, resultc, work1, work2, staged_in
  }

  namespace detail {

    /**
     * Returns the number of bytes of shared memory required to launch convolution_kernel with the given K.
     * Currently, only uses shared memory if using tensor cores. WE may want to put some tensors into shared memory
     * for other runs.
     */
    template<typename T>
    SCOPE constexpr size_type apply_conv_shmem_size(auto K) {
      size_type accel_smem_size = mra::accel::apply_conv_shmem_size<T>(K);
      if (accel_smem_size > 0) {
        return accel_smem_size;
      } else {
        return mTxmq_shmem_size<T>(2*K);
      }
    }

    /**
     * Thread block for the convolution kernels: max_thread_dims(2*K), with
     * the y dimension raised if needed so that the block holds at least
     * mra::accel::apply_conv_min_threads(K) threads (the tensor-core path in
     * apply_conv relies on it).
     */
    constexpr Dim3 conv_block_dims(size_type K) {
      Dim3 dims = max_thread_dims(2*K);
      const unsigned min_threads = mra::accel::apply_conv_min_threads(K);
      const unsigned xz = dims.x * dims.z;
      if (dims.y * xz < min_threads) {
        dims.y = (min_threads + xz - 1) / xz;
      }
      return dims;
    }

    /**
     * conv_block_dims, checked against MAX_THREADS_PER_BLOCK, the most any
     * convolution kernel is compiled for (see conv_launch_threads).
     */
    inline Dim3 conv_thread_dims(size_type K) {
      const Dim3 dims = conv_block_dims(K);
      if (dims.x * dims.y * dims.z > MAX_THREADS_PER_BLOCK) {
        throw std::runtime_error("convolution: MAX_THREADS_PER_BLOCK is too small for the "
                                 "tensor-core convolution (needs " +
                                 std::to_string(mra::accel::apply_conv_min_threads(K)) + " threads)");
      }
      return dims;
    }

    /**
     * The block size the convolution kernels are compiled for (their
     * LAUNCH_BOUNDS): conv_block_dims(K) for a compile-time K, so a block
     * smaller than MAX_THREADS_PER_BLOCK gets more registers per thread at
     * the same 2 blocks per SM; MAX_THREADS_PER_BLOCK for a run-time K.
     */
    template<typename KT>
    constexpr int conv_launch_threads() {
      if constexpr (is_ct_integral_v<KT>) {
        constexpr Dim3 dims = conv_block_dims(KT::value);
        return dims.x * dims.y * dims.z;
      } else {
        return MAX_THREADS_PER_BLOCK;
      }
    }

    /**
     * Calls launch(K) with K as a compile-time constant (mra::Int) for the
     * Ks that have a tensor-core path in the mma implementation (see
     * mra::accel::mma::apply_term_needs), and with K as a run-time value
     * otherwise. The kernels are compiled for these Ks on every backend;
     * where the tensor-core path does not apply (WMMA: K=6; HIP and host:
     * both), apply_conv falls back to the generic implementation.
     */
    template<typename LaunchFn>
    void with_ct_K(size_type K, LaunchFn&& launch) {
      switch (K) {
        case 8:  launch(Int<8>{}); break;
        case 6:  launch(Int<6>{}); break;
        default: launch(K);        break;
      }
    }


    template <typename T, Dimension NDIM,
              concepts::TensorViewArray<3, (size_t)NDIM> ViewTrans,
              concepts::TensorView<NDIM> ViewF,
              concepts::TensorView<NDIM> ViewResult,
              concepts::TensorView<NDIM> ViewWork1,
              concepts::TensorView<NDIM> ViewWork2>
    DEVSCOPE void conv_transform(
      const std::array<size_type, NDIM>& terms, // index of the term's matrix in each trans[d]
      const size_type dimk,
      const T mufac,
      const ViewTrans& trans,
      const ViewF& f,
      ViewResult& result,
      ViewWork1& work1,
      ViewWork2& work2)
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
      //          << " norm f " << normf(f) << " trans " << 0 << normf(trans[0](terms[0])) << std::endl;
      mTxmq(dimi, rank, dimk, work1ptr, f.data(), trans[0](terms[0]).data());

      size = rank * size / dimk;
      dimi = size / dimk;

      for (size_type d = 1; d < NDIM; ++d) {
        //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size  << " trans " << d << " norm " << norm(trans[d]) << std::endl;
        mTxmq(dimi, rank, dimk, work2ptr, work1ptr, trans[d](terms[d]).data());
        size = rank * size / dimk;
        dimi = size / dimk;
        std::swap(work1ptr, work2ptr);
      }

      detail::axpy_kernel_impl<T, NDIM>(work1, result, mufac);
      //std::cout << "CONV_TRANSFORM: dimk " << dimk << " rank " << rank << " size " << size
      //          << " result " << normf(result) << std::endl;

    }

    /**
     * See the comment on multiply_kernel_impl in mra/kernels/multiply.h for
     * why this uses explicitly-named template parameters instead of the
     * abbreviated (concept-constrained `auto`) form used elsewhere: too many
     * compiler-synthesized template parameters in one function trips an nvcc
     * EDG front-end assertion ("check_name_hiding_by_template_parameters").
     */
    template<typename T, Dimension NDIM,
             concepts::TensorViewArray<3, (size_t)NDIM> ViewTransr,
             concepts::TensorViewArray<3, (size_t)NDIM> ViewTranss,
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
      auto K,
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
      // the matrices of this term in the R/S views of each dimension
      std::array<size_type, NDIM> terms;
      for (Dimension d=0; d<NDIM; ++d) terms[d] = (size_type)opnorms(opid, mu, d, (size_type)NormId::Term);

      // R term
      double Rnorm = 1.0;
      for (Dimension d=0; d<NDIM; ++d) Rnorm *= opnorms(opid, mu, d, (size_type)NormId::Rnorm);
      if (at[0] && Rnorm > 1.e-20) {
        conv_transform<T, NDIM>(terms, 2*K, mufac, transr, f, result, work1, work2);
      }

      // S term
      double Snorm = 1.0;
      for (Dimension d=0; d<NDIM; ++d) Snorm *= opnorms(opid, mu, d, (size_type)NormId::Snorm);
      if (at[1] && Snorm > 0.0) {
        conv_transform<T, NDIM>(terms, K, -mufac, transs, f0, resultc, work1_k, work2_k);
      }
    }


    /**
     * Rebinds work1_k/work2_k to K-sized views of work1/work2's storage, on
     * behalf of the team lead only. Factored into its own function
     * for the same reason as compress_process_one_leader (see
     * mra/kernels/compress.h) and convolution_process_one_leader below:
     * nvcc was observed to miscompile a SHARED-tensor rebind directly
     * inside if(is_team_lead()){...} before a SYNCTHREADS() barrier. Uses
     * detail::broadcast_dims directly (bypassing TensorView's make_dims()
     * single-value-broadcast dispatch, itself a separate confirmed nvcc
     * miscompile site -- see reconstruct_process_one_leader's comment)
     * rather than passing K straight to the variadic-Dims constructor.
     */
    template<typename T, Dimension NDIM, typename KT, typename Work1T, typename Work2T,
             typename Work1KT, typename Work2KT>
    DEVSCOPE void apply_conv_leader(
      Work1T& work1,
      Work2T& work2,
      KT K,
      Work1KT& work1_k,
      Work2KT& work2_k)
    {
      work1_k = Work1KT(work1.data(), detail::broadcast_dims<typename Work1KT::dims_type>(K, std::make_index_sequence<NDIM>{}));
      work2_k = Work2KT(work2.data(), detail::broadcast_dims<typename Work2KT::dims_type>(K, std::make_index_sequence<NDIM>{}));
    }

    template<typename T, Dimension NDIM, bool SDone = false,
             concepts::TensorViewArray<3, (size_t)NDIM> ViewTransr,
             concepts::TensorViewArray<3, (size_t)NDIM> ViewTranss,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewF0,
             concepts::TensorView<NDIM> ViewResultc,
             concepts::TensorView<NDIM> ViewResult,
             concepts::TensorView<NDIM> ViewWork1,
             concepts::TensorView<NDIM> ViewWork2>
    DEVSCOPE void apply_conv(
      int opid,
      auto K,
      const T tol,
      const ViewTransr& transr,
      const ViewTranss& transs,
      const ViewOpnorms& opnorms,
      const std::array<bool, 2>& at,
      ViewF& f,
      ViewF0& f0,
      ViewResultc& resultc,
      ViewResult& result,  // size K, stores the sum
      ViewWork1& work1,
      ViewWork2& work2)
    {
      using dims_k_type = decltype(make_dims<NDIM>(K));
      using tensor_view_k_type = DenseTensorView<T, NDIM, dims_k_type>;
      SHARED tensor_view_k_type work1_k, work2_k;
      // cannot be SHARED because ctors won't run
      std::array<Slice,NDIM> s0 = std::array<Slice,NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
      if (is_team_lead()) {
        apply_conv_leader<T, NDIM>(work1, work2, K, work1_k, work2_k);
      }

      const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank); // doing computation assuming full rank

      T optol = 0.01*tol/rank; // can potentially be a parameter

      if constexpr (!SDone) {
        f0 = f(s0);
      }

      // TODO: do we care about modified() operators?
      // SDone: resultc already holds the S term; only mma::apply_conv takes SDone
      bool accel_done;
      if constexpr (SDone) {
        accel_done = mra::accel::apply_conv<T, NDIM, true>(opid, K, optol, transr, transs, opnorms, at, f, f0,
                                   resultc, result);
      } else {
        accel_done = mra::accel::apply_conv<T, NDIM>(opid, K, optol, transr, transs, opnorms, at, f, f0,
                                   resultc, result);
      }
      if (!accel_done) {

        result = 0.0;
        resultc = 0.0;

        for (size_type mu = 0; mu < rank; ++mu) {
          T munorm = opnorms(opid, mu, 0, (size_type)NormId::MUnorm);
          if (munorm > optol) {
            T mufac = opnorms(opid, mu, 0, (size_type)NormId::Fac);
            muopxv_fast<T, NDIM>(opid, K, mu, mufac, tol/std::abs(mufac), at,
                                 transr, transs, opnorms,
                                 f, f0,
                                 resultc, result,
                                 work1, work2,
                                 work1_k, work2_k);
          }
        }
      }

      result(s0) += resultc;
    }

    /**
     * Combines a (fully summed) convolution result with the pre-existing `in` node and
     * applies the aggressive-screening threshold, writing the final per-function norm to
     * `resnorm_out` (if non-null).
     */
    template <typename T, Dimension NDIM, concepts::TensorView<NDIM> ViewIn, concepts::TensorView<NDIM> ViewResult>
    DEVSCOPE void convolution_finalize(
      const T fac,
      const T tol,
      const ViewIn& in,
      ViewResult& result,
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
    template <typename T, Dimension NDIM, bool SDone = false,
              concepts::TensorViewArray<3, (size_t)NDIM> ViewTransr,
              concepts::TensorViewArray<3, (size_t)NDIM> ViewTranss,
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
      auto K,
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
      auto smem_size = detail::apply_conv_shmem_size<T>(2*K);

      //std::cout << "MRA-APPLY key " << key << " disp " << displacement << " cnorm " << cnorm
      //          << " opnorm " << opnorm << " tol " << tol << std::endl;
      if ((cnorm * opnorm) > (tol / fac)) {
        apply_conv<T, NDIM, SDone>(opid, K, (tol / fac / cnorm), transr, transs,
                   opnorms, at, f, f0, resultc,
                   result, work1, work2);
      } else {
        result = 0.0;
      }

      convolution_finalize<T, NDIM>(fac, tol, in, result, resnorm_out);

      //std::cout << "MRA_OP_APPLY " << key << " disp " << displacement << " result " << resnorm << std::endl;

    }

    /**
     * Does the actual per-fnid setup work (finding fnid, rebinding
     * f0/resultc/work1/work2/f/in/result to this function's slice of the
     * batch), on behalf of the team lead only. Deliberately factored into
     * its own function -- see compress_process_one_leader's
     * comment in mra/kernels/compress.h (and reconstruct_process_one_leader's
     * in mra/kernels/reconstruct.h) for the full story: nvcc was observed to
     * miscompile the surrounding if(is_team_lead()){...} SYNCTHREADS()
     * pattern when this body was inlined here, corrupting SHARED state
     * across the barrier. Uses detail::broadcast_dims directly (bypassing
     * TensorView's make_dims() single-value-broadcast dispatch, itself a
     * separate confirmed nvcc miscompile site) rather than passing K/TWOK
     * straight to the variadic-Dims constructor.
     */
    template<typename T, Dimension NDIM, typename KT, typename TwoKT,
             typename F0T, typename ResultcT, typename Work1T, typename Work2T, typename StagedInT, typename FT, typename InT, typename ResultT>
    DEVSCOPE void convolution_process_one_leader(
      const T* in_data,   // in_view(i).data() etc., see row_data
      const T* f_data,
      T* result_data,
      T* tmp,
      KT K,
      TwoKT TWOK,
      size_type N,
      size_type tmp_pos,
      size_type tmp_slot,
      F0T& f0,
      ResultcT& resultc,
      Work1T& work1,
      Work2T& work2,
      StagedInT& staged_in,
      FT& f,
      InT& in,
      ResultT& result,
      size_type fnid,
      size_type& i,
      T* resultc_ptr)
    {
      // fnid: the function id of position tmp_pos, see convolution_process_one
      i = fnid;

      const size_type K2NDIM = mra::pow(K, Int<NDIM>{});
      const size_type TWOK2NDIM = mra::pow(TWOK, Int<NDIM>{});
      T* block_tmp_ptr = &tmp[tmp_slot*convolution_tmp_size<NDIM>(K)];
      // construct temporaries and pass them to conv_transform
      f0        = F0T(&block_tmp_ptr[                      0], detail::broadcast_dims<typename F0T::dims_type>(K, std::make_index_sequence<NDIM>{}));
      // resultc_ptr: the S term computed by another kernel (SDone), else in tmp
      resultc   = ResultcT(resultc_ptr ? resultc_ptr : &block_tmp_ptr[K2NDIM], detail::broadcast_dims<typename ResultcT::dims_type>(K, std::make_index_sequence<NDIM>{}));
      work1     = Work1T(&block_tmp_ptr[              2*K2NDIM], detail::broadcast_dims<typename Work1T::dims_type>(TWOK, std::make_index_sequence<NDIM>{}));
      work2     = Work2T(&block_tmp_ptr[  TWOK2NDIM + 2*K2NDIM], detail::broadcast_dims<typename Work2T::dims_type>(TWOK, std::make_index_sequence<NDIM>{}));
      staged_in = StagedInT(&block_tmp_ptr[2*TWOK2NDIM + 2*K2NDIM], detail::broadcast_dims<typename StagedInT::dims_type>(TWOK, std::make_index_sequence<NDIM>{}));
      in        = InT(in_data, detail::broadcast_dims<typename InT::dims_type>(TWOK, std::make_index_sequence<NDIM>{}));
      f         = FT(f_data, detail::broadcast_dims<typename FT::dims_type>(TWOK, std::make_index_sequence<NDIM>{}));
      result    = ResultT(result_data, detail::broadcast_dims<typename ResultT::dims_type>(TWOK, std::make_index_sequence<NDIM>{}));
    }

    /**
     * view(i).data() without view(i)'s scan, given the number of allocated
     * entries before i (from find_nth_allocated_warp). Where view is zero at
     * i it calls view(i), which needs no scan there.
     */
    template <typename ViewT>
    DEVSCOPE auto row_data(ViewT& view, size_type i, size_type before) {
      return view.is_zero(i) ? view(i).data() : &view.data()[before * view.entry_size()];
    }

    /**
     * Processes one (node, function-index) pair: the per-block body shared by
     * both the unbatched convolution_kernel below and the batched
     * convolution_kernel_batched further down -- there is exactly one copy of
     * this logic to maintain instead of two near-identical grid-stride loops.
     * Returns the function index `i` it just processed (found via
     * find_nth_allocated_warp and set by convolution_process_one_leader; same
     * value on every thread in the block, courtesy of the SHARED `i`), so both callers
     * can demote result_view's sparsity state for that position right after
     * the call, on the same team-lead thread that just wrote resnorms[i], if
     * the computed norm turned out to be exactly zero -- no extra
     * synchronization needed, and no separate kernel launch either.
     */
    template <typename T, Dimension NDIM, bool SDone = false, typename TransrT, typename TranssT, typename OpnormsViewT,
              typename InViewT, typename FViewT, typename ResultViewT, typename ResnormsT>
    DEVSCOPE size_type convolution_process_one(
      Key<NDIM> key,
      Key<NDIM> displacement,
      auto K,
      const T fac,
      const T tol,
      const TransrT& transr,
      const TranssT& transs,
      const OpnormsViewT& opnorms_view,
      const std::array<bool, 2>& at,
      const InViewT& in_view,
      const FViewT& f_view,
      ResultViewT& result_view,
      ResnormsT& resnorms,
      T* tmp,
      size_type N,
      size_type tmp_pos,
      size_type tmp_slot,
      T* resultc_ptr = nullptr)  // with SDone: where resultc already is
    {
      auto TWOK = Int<2>{}*K;
      using dims_k_type = decltype(make_dims<NDIM>(K));
      using dims_2k_type = decltype(make_dims<NDIM>(TWOK));
      using tensor_view_k_type = DenseTensorView<T, NDIM, dims_k_type>;
      using tensor_view_2k_type = DenseTensorView<T, NDIM, dims_2k_type>;
      using const_tensor_view_2k_type = DenseTensorView<const T, NDIM, dims_2k_type>;
      SHARED tensor_view_k_type f0, resultc;
      SHARED tensor_view_2k_type work1, work2, result, staged_in;
      SHARED const_tensor_view_2k_type f, in;
      SHARED size_type i;

      // The function id of this position, searched for by the first warp.
      // result_view has exactly n_nonzero non-zero entries, so this
      // always finds a valid function id -- see submit_convolution_kernel.
      // Excluded (zero) functions' resnorms entries are pre-filled with
      // 0.0 host-side (see mra/tasks/convolution.h) since no block visits
      // them here.
      // The first warp finds the function and its data in the three views
      // in one pass.
      size_type fnid = N;
      const T* in_data = nullptr;
      const T* f_data = nullptr;
      T* result_data = nullptr;
#ifdef HAVE_DEVICE_ARCH
      if (thread_id() < MRA_WARP_SIZE)
#endif
      {
        size_type in_before, f_before;
        fnid = find_nth_allocated_warp(N, tmp_pos, result_view, in_view, in_before, f_view, f_before);
        in_data = row_data(in_view, fnid, in_before);
        f_data = row_data(f_view, fnid, f_before);
        result_data = row_data(result_view, fnid, tmp_pos);
      }
      if (is_team_lead()) {
        convolution_process_one_leader<T, NDIM>(in_data, f_data, result_data, tmp, K, TWOK, N, tmp_pos, tmp_slot,
                                                f0, resultc, work1, work2, staged_in, f, in, result, fnid, i, resultc_ptr);
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
        return i;
      }

      // Snapshot `in` into tmp-backed storage before anything below writes to
      // `result` -- result_view/out may be the SAME underlying storage as
      // in_view when accumulate_tt reuses its input node in place as the
      // output (see its comment in mra/tasks/convolution.h), and apply_conv/
      // convolution_kernel_impl's cheap-skip path both zero `result`
      // unconditionally before the final combination with `in` -- which would
      // otherwise read back zeros (or worse, the half-written new value)
      // instead of the original running sum.
      staged_in = in;

      int opid = opnorms_view.dim(0) > 1 ? static_cast<int>(i) : 0; // TODO: this is a bit hacky, can we do better?

      convolution_kernel_impl<T, NDIM, SDone>(key, opid, displacement, K, fac, tol,
                                       transr, transs, opnorms_view, at, staged_in, f, f0,
                                       resultc, result, work1, work2,
                                       resnorms.empty() ? nullptr : &resnorms[i]);
      return i;
    }

#if defined(MRA_CHECK_NORMS)
    /**
     * Debug-only, launched as its OWN kernel (grid of 1 block) immediately
     * before convolution_kernel below, on the same stream -- see
     * compress_verify_sparsity_kernel's comment (mra/kernels/compress.h) for
     * why this must not be inlined into convolution_kernel itself (a
     * same-launch race against find_nth_allocated's assert in other blocks
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
    template<typename T, Dimension NDIM, concepts::TensorView<NDIM+1> ViewF, concepts::TensorView<NDIM+1> ViewResult>
    GLOBALSCOPE void convolution_verify_sparsity_kernel_single(
      Key<NDIM> key,
      size_type N,
      size_type n_nonzero,
      const ViewF f_view,
      ViewResult result_view)
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

    template <typename T, Dimension NDIM,
              concepts::TensorView<NDIM+1> ViewIn,
              concepts::TensorView<NDIM+1> ViewF,
              concepts::TensorView<NDIM+1> ViewResult,
              concepts::TensorView<1> ViewResnorms,
              concepts::TensorViewArray<3, (size_t)NDIM> ViewTransr,
              concepts::TensorViewArray<3, (size_t)NDIM> ViewTranss,
              concepts::TensorView<4> ViewOpnorms,
              typename KT>
    LAUNCH_BOUNDS(conv_launch_threads<KT>())
    GLOBALSCOPE void convolution_kernel(
      Key<NDIM> key,
      Key<NDIM> displacement,
      KT K,
      size_type N,
      size_type n_nonzero,
      const T fac,
      const T tol,
      const ViewIn in_view,
      const ViewF f_view,
      ViewResult result_view,
      ViewResnorms resnorms,
      const ViewTransr transr,
      const ViewTranss transs,
      const ViewOpnorms opnorms_view,
      const std::array<bool, 2> at,
      T* tmp)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        size_type i = convolution_process_one<T, NDIM>(key, displacement, K, fac, tol, transr, transs, opnorms_view, at,
                                                        in_view, f_view, result_view, resnorms, tmp, N, pos, pos);
        // result_view's bitfield was populated optimistically (from the
        // structural union of the inputs' sparsity, via SparsityManager)
        // before this position's real value was known -- demote it now that
        // resnorms[i] says otherwise. Safe with no extra synchronization:
        // only this position's own block/team-lead thread ever touches
        // resnorms[i]/result_view's state for this i.
        if (!resnorms.empty() && is_team_lead() && resnorms[i] == 0.0) {
          result_view.set_state(i, SparsityState::ALLOCATED);
        }
      }

      // Finalize resnorms for positions convolution_process_one never visits
      // above (result_view.is_zero(i) -- structurally excluded from the
      // n_nonzero union, not just computed to zero). resnorms is
      // Allocate-scoped (see mra/tasks/convolution.h) so these entries hold
      // garbage otherwise; this set is disjoint from anything any block
      // writes above, so block 0 alone doing this is safe regardless of the
      // other blocks' progress. submit_convolution_kernel forces at least one
      // block whenever resnorms is non-empty, so this still runs even when
      // n_nonzero == 0.
      if (!resnorms.empty() && blockIdx.x == 0) {
        for (size_type i = threadIdx.x; i < N; i += blockDim.x) {
          if (result_view.is_zero(i)) {
            resnorms[i] = T(0);
          }
        }
      }
    }

  } // namespace detail

  template <typename T, Dimension NDIM,
            concepts::TensorView<NDIM+1> ViewIn,
            concepts::TensorView<NDIM+1> ViewF,
            concepts::TensorView<NDIM+1> ViewResult,
            concepts::TensorView<1> ViewResnorms,
            concepts::TensorViewArray<3, (size_t)NDIM> ViewTransr,
            concepts::TensorViewArray<3, (size_t)NDIM> ViewTranss,
            concepts::TensorView<4> ViewOpnorms>
  void submit_convolution_kernel(
    Key<NDIM> key,
	  Key<NDIM> displacement,
    size_type K,
    size_type N,
    size_type n_nonzero,
    const T fac,
    const T tol,
    const ViewIn& in_view,
    const ViewF& f_view,
    ViewResult& result_view,
    ViewResnorms& resnorms,
    const ViewTransr& transr,
    const ViewTranss& transs,
    const ViewOpnorms& opnorms,
    const std::array<bool, 2>& at,
    T* tmp,
    ttg::device::Stream stream)
#if defined(MRA_HAVE_KERNEL_DEFINITIONS)
  {
#if defined(MRA_CHECK_NORMS)
    // Separate, prior kernel on the same stream -- see
    // convolution_verify_sparsity_kernel_single's comment for why this must
    // not be inlined into convolution_kernel itself.
    CALL_KERNEL((detail::convolution_verify_sparsity_kernel_single<T, NDIM>), 1, 32, 0, stream,
                (key, N, n_nonzero, f_view, result_view));
    checkSubmit();
#endif // MRA_CHECK_NORMS

    Dim3 thread_dims = detail::conv_thread_dims(K);
    auto smem_size = detail::apply_conv_shmem_size<T>(K);

    // At least one block even when n_nonzero == 0, but only when resnorms is
    // actually in use (accumulate_tt's non-last_key calls pass an empty one
    // every time, the common case -- forcing a block there would be a wasted
    // launch for nothing): convolution_kernel's block 0 also finalizes
    // resnorms for positions never visited by the position loop (see its
    // comment), which must still happen when there's no real work otherwise.
    const size_type grid = (n_nonzero > 0 || resnorms.empty()) ? n_nonzero : size_type(1);

    //CONFIGURE_KERNEL((detail::convolution_kernel<T, NDIM>), smem_size);
    //auto in_view_k = make_ct_tensorview_from(in_view, in_view.dim(0), Int<16>{});
    detail::with_ct_K(K, [&](auto ctK) {
      CALL_KERNEL((detail::convolution_kernel<T, NDIM>), grid, thread_dims, smem_size, stream,
                  (key, displacement, ctK, N, n_nonzero, fac, tol, in_view, f_view, result_view,
                  resnorms, transr, transs, opnorms, at, tmp));
    });
    checkSubmit();
  }
#else
  ;
#endif // MRA_HAVE_KERNEL_DEFINITIONS

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
      // No per-member tmp field anymore: every member's scratch comes out of
      // one shared, occupancy-sized pool instead (see GroupedBatchPool::
      // ensure_scratch / submit_convolution_kernel_batched), indexed by
      // blockIdx.x rather than this member's own local position.
      size_type,                               // n: number of blocks (functions) this member contributes
      std::array<DenseTensorView<const T, 3>, NDIM>, // transr (this member's own operator data)
      std::array<DenseTensorView<const T, 3>, NDIM>, // transs
      DenseTensorView<T, 4>,                   // opnorms
      T,                                       // tol: this member's own truncate_tol(...)
      std::array<bool, 2>                      // at: this member's own apply-terms flags
      // No sparsity_offset field anymore: sparsity scatter and resnorms
      // zero-init are both done as part of the H2D transfer itself now (see
      // submit_convolution_batch_leader/submit_grouped_copy's comments), not
      // by a device kernel reading an offset out of this struct.
    >;

    /* Named indices into ConvolutionBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct ConvolutionBatchArgIdx {
      static constexpr std::size_t in_view         = 0;
      static constexpr std::size_t f_view          = 1;
      static constexpr std::size_t result_view     = 2;
      static constexpr std::size_t resnorms_view   = 3;
      static constexpr std::size_t n               = 4;
      static constexpr std::size_t transr          = 5;
      static constexpr std::size_t transs          = 6;
      static constexpr std::size_t opnorms         = 7;
      static constexpr std::size_t tol             = 8;
      static constexpr std::size_t at              = 9;
    };

    /**
     * The S term (k = K) of every non-zero position, one warp per position,
     * into s_result[pos * K^3], for the Ks of apply_conv_splits_s.
     * convolution_kernel_batched adds it to the R term afterwards (SDone).
     * Skipped: f zero or gated out (s_result is not read), and at[1] false
     * (the main kernel zeroes it).
     */
    template <typename T, Dimension NDIM, typename KT>
    LAUNCH_BOUNDS(mra::accel::apply_conv_s_warps * 32)
    GLOBALSCOPE void convolution_s_kernel_batched(
      ConvolutionBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets,        // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      T* s_result,                            // total_nonzero x K^3
      KT K,
      const T fac)
    {
      using idx = ConvolutionBatchArgIdx;
      constexpr int NW = mra::accel::apply_conv_s_warps;
      static_assert(conv_launch_threads<KT>() == 256,
                    "convolution_s_kernel_batched: the norm of f is replayed for 256 threads per block");
      const int warp = thread_id() / 32;
      const size_type nwarps = size_type(gridDim.x) * NW;
      const size_type K2NDIM = mra::pow(K, Int<NDIM>{});
      for (size_type pos = size_type(blockIdx.x) * NW + warp; pos < total_nonzero; pos += nwarps) {
        size_type local_pos;
        const size_type member = find_member_for_pos_warp(member_offsets, num_members, pos, &local_pos);
        auto& arg = args[member];
        if (!std::get<idx::at>(arg)[1]) continue;
        const auto& f_view = std::get<idx::f_view>(arg);
        size_type f_before;
        const size_type i = find_nth_allocated_warp(std::get<idx::n>(arg), local_pos, std::get<idx::result_view>(arg),
                                                    f_view, f_before);
        if (f_view.is_zero(i)) continue;
        const auto& opnorms = std::get<idx::opnorms>(arg);
        const int opid = opnorms.dim(0) > 1 ? static_cast<int>(i) : 0;
        T* resultc = &s_result[pos * K2NDIM];
        mra::accel::apply_conv_s_warp<T, NDIM>(opid, K, fac, std::get<idx::tol>(arg), std::get<idx::transs>(arg),
                                               opnorms, row_data(f_view, i, f_before), resultc);
      }
    }

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
     * (find_nth_allocated). This makes convolution_kernel_batched a thin
     * wrapper: look up one work item and hand off to the exact same
     * per-(node, function) body convolution_kernel itself uses
     * (convolution_process_one, defined above with convolution_kernel_impl).
     */
    template <typename T, Dimension NDIM, typename KT>
    LAUNCH_BOUNDS(conv_launch_threads<KT>())
    GLOBALSCOPE void convolution_kernel_batched(
      ConvolutionBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets,        // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      T* shared_tmp,                          // one shared, occupancy-sized scratch pool for
                                               // the whole launch -- indexed by blockIdx.x, not
                                               // per-member (see GroupedBatchPool::ensure_scratch)
      T* s_result,                            // with SDone: the S terms (convolution_s_kernel_batched)
      KT K,
      const T fac)
    {
      using idx = ConvolutionBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;

      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        // looked up by the whole first warp (see find_member_for_pos_warp)
        if (thread_id() < 32) {
          size_type lp;
          const size_type m = find_member_for_pos_warp(member_offsets, num_members, pos, &lp);
          if (is_team_lead()) {
            member = m;
            local_pos = lp;
          }
        }
        SYNCTHREADS();
        auto& arg = args[member];
        const size_type member_N = std::get<idx::n>(arg);
        auto& result_view = std::get<idx::result_view>(arg);
        auto& resnorms_view = std::get<idx::resnorms_view>(arg);

        constexpr bool SDone = mra::accel::apply_conv_splits_s<KT>();
        size_type i = convolution_process_one<T, NDIM, SDone>(Key<NDIM>{}, Key<NDIM>{}, K, fac, std::get<idx::tol>(arg),
                                         std::get<idx::transr>(arg), std::get<idx::transs>(arg),
                                         std::get<idx::opnorms>(arg), std::get<idx::at>(arg),
                                         std::get<idx::in_view>(arg), std::get<idx::f_view>(arg),
                                         result_view, resnorms_view,
                                         shared_tmp, member_N, local_pos, blockIdx.x,
                                         SDone ? &s_result[pos * mra::pow(K, Int<NDIM>{})] : nullptr);
        // See convolution_kernel's comment: demote this member's result_view
        // state for i now that resnorms_view[i] is known -- same inline
        // check as the unbatched path, replacing the old, separately
        // launched convolution_prune_zero_norm_kernel_batched.
        if (!resnorms_view.empty() && is_team_lead() && resnorms_view[i] == 0.0) {
          result_view.set_state(i, SparsityState::ALLOCATED);
        }
      }
    }

    /**
     * Load balancing (MRA_CONV_LB, see submit_convolution_lb). A launch's
     * work is a list of units: per position, c0 units for its fixed cost
     * (MRA_CONV_LB_C0), then one per applied k = 2K term. The list is split
     * into equal shares per SM, each share into R parts: one part per block
     * (mode 1) or per SM (mode 2, R = 1). A position whose terms span
     * several parts is summed up by the last of its pieces to finish, which
     * also does its epilogue.
     */

    /** What the pre-pass found for one position. */
    template <typename T>
    struct lb_rec {
      unsigned member, local_pos, fnid, in_before, f_before, nr;  // nr: applied k = 2K terms, 0 if none
      T cnorm;                                                    // ||f||
    };

    /** The buffers of one load-balanced launch, shared by its kernels. */
    template <typename T>
    struct lb_bufs {
      lb_rec<T>* recs;     // per position
      unsigned* units;     // per position: c0 + nr
      unsigned* prefix;    // exclusive prefix sum of units, then the total
      unsigned* arrive;    // per position: pieces that have finished
      unsigned* pos_lo;    // per part: its positions [pos_lo, pos_hi)
      unsigned* pos_hi;
      unsigned* counters;  // mode 1: a slot counter per SM, then a taken flag per part; mode 2: a cursor per SM
      T* partials;         // per part: 2 partial results of (2K)^3 (lb_slot)
      uint4* tterms;       // per member, tcap entries: the matrix indices {d0, d1, d2} of (opid, term)
      unsigned tcap;
      unsigned num_sms;
      unsigned c0;
    };

    /**
     * lb_part_lo: the first item of part p of n items in np parts (the first
     * n % np parts one larger); lb_part_of: the part of item x. lb_sub and
     * lb_sub_of: the same for part g = s * R + j (the j-th of SM s's R parts)
     * of W units.
     */
    SCOPE unsigned lb_part_lo(unsigned n, unsigned np, unsigned p) {
      const unsigned q = n / np, r = n % np;
      return p * q + (p < r ? p : r);
    }
    SCOPE unsigned lb_part_of(unsigned n, unsigned np, unsigned x) {
      const unsigned q = n / np, r = n % np;
      return x < r * (q + 1) ? x / (q + 1) : r + (x - r * (q + 1)) / q;  // q > 0 in the second case
    }
    struct lb_parts { unsigned W, S, R; };  // W units, S SMs, R parts per SM
    struct lb_range { unsigned lo, hi; };
    SCOPE lb_range lb_sub(const lb_parts& q, unsigned g) {
      const unsigned s = g / q.R, j = g % q.R;
      const unsigned slo = lb_part_lo(q.W, q.S, s), n = lb_part_lo(q.W, q.S, s + 1) - slo;
      return {slo + lb_part_lo(n, q.R, j), slo + lb_part_lo(n, q.R, j + 1)};
    }
    SCOPE unsigned lb_sub_of(const lb_parts& q, unsigned x) {
      const unsigned s = lb_part_of(q.W, q.S, x);
      const unsigned slo = lb_part_lo(q.W, q.S, s), n = lb_part_lo(q.W, q.S, s + 1) - slo;
      return s * q.R + lb_part_of(n, q.R, x - slo);
    }

    /** The last pos with prefix[pos] <= x; prefix is non-decreasing from prefix[0] = 0. */
    SCOPE unsigned lb_pos_of(const unsigned* prefix, unsigned n, unsigned x) {
      unsigned a = 0, b = n;
      while (b - a > 1) {
        const unsigned c = (a + b) / 2;
        if (prefix[c] <= x) a = c; else b = c;
      }
      return a;
    }

    /**
     * The piece of a position (units [u0, u1), nr terms) in a part (units
     * rg): whether there is one, whether it is the whole position, its terms
     * b_begin .. b_begin + b_count - 1, and for a split position the parts
     * g0 .. g1 its terms fall in.
     */
    struct lb_piece_t { bool work, whole; unsigned b_begin, b_count, g0, g1; };
    SCOPE lb_piece_t lb_piece_of(unsigned u0, unsigned u1, unsigned nr, unsigned c0, lb_range rg, const lb_parts& q) {
      lb_piece_t p{false, true, 0, 0, 0, 0};
      if (nr == 0) {
        p.work = (u0 >= rg.lo);  // no terms: the part with the first unit does it all
        return p;
      }
      const unsigned ub = u0 + c0;  // the first term's unit
      const unsigned bl = ub > rg.lo ? ub : rg.lo, bh = u1 < rg.hi ? u1 : rg.hi;
      if (bh <= bl) return p;  // only c0 units in this part
      p.work = true;
      p.b_begin = bl - ub;
      p.b_count = bh - bl;
      p.whole = (p.b_count == nr);
      if (!p.whole) {
        p.g0 = lb_sub_of(q, ub);
        p.g1 = lb_sub_of(q, u1 - 1);
      }
      return p;
    }

    /**
     * Part g's slot for the partial result of a position whose terms start in
     * part g0. 2 per part: a part can hold the end of one split position and
     * the start of another.
     */
    SCOPE size_type lb_slot(unsigned g, unsigned g0) { return 2 * size_type(g) + (g == g0 ? 1 : 0); }

    DEVSCOPE unsigned lb_smid() {
#if defined(__CUDA_ARCH__)
      unsigned s;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
      return s;
#else
      return 0;
#endif
    }

    /** Called by a whole warp: the first t < n with open(t), n if none. */
    template <typename Pred>
    DEVSCOPE unsigned lb_warp_first(unsigned n, Pred&& open) {
#if defined(__CUDA_ARCH__)
      const unsigned lane = thread_id() % 32;
      for (unsigned base = 0; base < n; base += 32) {
        const unsigned bits = __ballot_sync(0xffffffffu, base + lane < n && open(base + lane));
        if (bits) return base + __ffs(bits) - 1;
      }
#endif
      return n;
    }

    /**
     * Load-balancing pre-pass: the term table (b.tterms), then one warp per
     * position for its record (see mra::accel::lb_count_r), units and
     * arrive and, with apply_conv_splits_s, its S term.
     */
    template <typename T, Dimension NDIM, typename KT>
    LAUNCH_BOUNDS(mra::accel::apply_conv_s_warps * 32)
    GLOBALSCOPE void convolution_lb_prepass_batched(
      ConvolutionBatchArg<T, NDIM>* args,
      const size_type* member_offsets,
      size_type num_members,
      size_type total_nonzero,
      lb_bufs<T> b,
      T* s_result,         // with apply_conv_splits_s: total_nonzero x K^3
      KT K,
      const T fac)
    {
      using idx = ConvolutionBatchArgIdx;
      const size_type ntab = size_type(num_members) * b.tcap;
      for (size_type e = size_type(blockIdx.x) * blockDim.x + thread_id(); e < ntab; e += size_type(gridDim.x) * blockDim.x) {
        const auto& opn = std::get<idx::opnorms>(args[e / b.tcap]);
        const size_type i = e % b.tcap, nrank = opn.dim(1);
        if (i < opn.dim(0) * nrank) {
          const size_type c = i / nrank, mu = i % nrank;
          b.tterms[e] = make_uint4((unsigned)opn(c, mu, 0, (size_type)NormId::Term), (unsigned)opn(c, mu, 1, (size_type)NormId::Term),
                                   (unsigned)opn(c, mu, 2, (size_type)NormId::Term), 0u);
        }
      }

      constexpr int NW = mra::accel::apply_conv_s_warps;
      const int warp = thread_id() / 32;
      const size_type nwarps = size_type(gridDim.x) * NW;
      for (size_type pos = size_type(blockIdx.x) * NW + warp; pos < total_nonzero; pos += nwarps) {
        size_type local_pos;
        const size_type member = find_member_for_pos_warp(member_offsets, num_members, pos, &local_pos);
        auto& arg = args[member];
        const auto& in_view = std::get<idx::in_view>(arg);
        const auto& f_view = std::get<idx::f_view>(arg);
        const auto& opnorms = std::get<idx::opnorms>(arg);
        size_type in_before, f_before;
        const size_type i = find_nth_allocated_warp(std::get<idx::n>(arg), local_pos, std::get<idx::result_view>(arg),
                                                    in_view, in_before, f_view, f_before);
        const bool fzero = f_view.is_zero(i);
        const int opid = opnorms.dim(0) > 1 ? static_cast<int>(i) : 0;
        lb_rec<T> r{};
        r.member = member;
        r.local_pos = local_pos;
        r.fnid = i;
        r.in_before = in_before;
        r.f_before = f_before;
        if (!fzero) {
          const int n = mra::accel::lb_count_r<T, NDIM, conv_launch_threads<KT>()>(opid, K, fac, std::get<idx::tol>(arg), opnorms,
                                                                                    row_data(f_view, i, f_before), r.cnorm);
          r.nr = std::get<idx::at>(arg)[0] ? n : 0;
        }
        if (thread_id() % 32 == 0) {
          b.recs[pos] = r;
          b.units[pos] = b.c0 + r.nr;
          b.arrive[pos] = 0;
        }
        if constexpr (mra::accel::apply_conv_splits_s<KT>()) {
          if (!fzero && std::get<idx::at>(arg)[1]) {
            mra::accel::apply_conv_s_warp<T, NDIM>(opid, K, fac, std::get<idx::tol>(arg), std::get<idx::transs>(arg),
                                                   opnorms, row_data(f_view, i, f_before),
                                                   &s_result[pos * mra::pow(K, Int<NDIM>{})], r.cnorm);
          }
        }
      }
    }

    /**
     * One block: the exclusive prefix sum of units into prefix (prefix[n] is
     * the total), each part's positions [pos_lo, pos_hi), and zeroed
     * counters. Each thread scans 4 consecutive units per tile.
     */
    template <unsigned NT>
    GLOBALSCOPE void convolution_lb_scan(
      const unsigned* units,
      unsigned* prefix,     // n + 1
      unsigned n,
      unsigned S,
      unsigned R,
      unsigned* pos_lo,     // S * R
      unsigned* pos_hi,     // S * R
      unsigned* counters,
      unsigned ncounters)
    {
#if defined(__CUDA_ARCH__)
      constexpr unsigned IPT = 4, NWS = NT / 32;
      static_assert(NWS <= 32, "convolution_lb_scan: one warp scans the warp sums");
      __shared__ unsigned wsum[NWS];
      const unsigned tid = thread_id(), lane = tid % 32, warp = tid / 32;
      for (unsigned c = tid; c < ncounters; c += NT) counters[c] = 0;
      unsigned carry = 0;  // the units of the tiles before
      for (unsigned base = 0; base < n; base += NT * IPT) {
        unsigned v[IPT], s = 0;
        for (unsigned k = 0; k < IPT; ++k) {
          const unsigned x = base + tid * IPT + k;
          v[k] = x < n ? units[x] : 0u;
          s += v[k];
        }
        unsigned incl = s;  // inclusive sum over the warp's threads
        for (unsigned d = 1; d < 32; d <<= 1) {
          const unsigned y = __shfl_up_sync(0xffffffffu, incl, d);
          if (lane >= d) incl += y;
        }
        if (lane == 31) wsum[warp] = incl;
        SYNCTHREADS();
        if (warp == 0) {
          unsigned w = lane < NWS ? wsum[lane] : 0u;
          for (unsigned d = 1; d < 32; d <<= 1) {
            const unsigned y = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w += y;
          }
          if (lane < NWS) wsum[lane] = w;
        }
        SYNCTHREADS();
        unsigned run = carry + (warp > 0 ? wsum[warp - 1] : 0u) + incl - s;
        for (unsigned k = 0; k < IPT; ++k) {
          const unsigned x = base + tid * IPT + k;
          if (x < n) prefix[x] = run;
          run += v[k];
        }
        carry += wsum[NWS - 1];
        SYNCTHREADS();  // wsum is reused by the next tile
      }
      if (tid == 0) prefix[n] = carry;
      SYNCTHREADS();
      const lb_parts q{carry, S, R};
      for (unsigned g = tid; g < S * R; g += NT) {
        const lb_range r = lb_sub(q, g);
        const bool any = r.lo < r.hi;
        pos_lo[g] = any ? lb_pos_of(prefix, n, r.lo) : 0;
        pos_hi[g] = any ? lb_pos_of(prefix, n, r.hi - 1) + 1 : 0;
      }
#endif // __CUDA_ARCH__
    }

    /**
     * Piece p of position pos in part g, using the pre-pass's record. A whole
     * position goes as in convolution_process_one. A split position's piece
     * stores its partial result in its slot (lb_slot); the last piece to
     * finish adds the slots up in part order and does the epilogue.
     */
    template <typename T, Dimension NDIM, bool SDone, typename ArgT>
    DEVSCOPE void convolution_lb_piece(
      ArgT& arg,
      const lb_rec<T>& rec,
      size_type pos,
      auto K,
      const T fac,
      T* tmp,
      T* s_result,
      const lb_piece_t& p,
      unsigned g,
      const lb_parts& q,
      const lb_bufs<T>& b)
    {
#if defined(__CUDA_ARCH__)
      using idx = ConvolutionBatchArgIdx;
      auto TWOK = Int<2>{}*K;
      using dims_k_type = decltype(make_dims<NDIM>(K));
      using dims_2k_type = decltype(make_dims<NDIM>(TWOK));
      using tensor_view_k_type = DenseTensorView<T, NDIM, dims_k_type>;
      using tensor_view_2k_type = DenseTensorView<T, NDIM, dims_2k_type>;
      using const_tensor_view_2k_type = DenseTensorView<const T, NDIM, dims_2k_type>;
      SHARED tensor_view_k_type f0, resultc;
      SHARED tensor_view_2k_type work1, work2, result, staged_in;
      SHARED const_tensor_view_2k_type f, in;
      SHARED size_type i;
      SHARED int last;
      const size_type k3 = mra::pow(TWOK, Int<NDIM>{});

      const size_type N = std::get<idx::n>(arg);
      const auto& in_view = std::get<idx::in_view>(arg);
      const auto& f_view = std::get<idx::f_view>(arg);
      auto& result_view = std::get<idx::result_view>(arg);
      auto& resnorms = std::get<idx::resnorms_view>(arg);
      const auto& opnorms = std::get<idx::opnorms>(arg);
      const T tol = std::get<idx::tol>(arg);
      const std::array<bool, 2>& at = std::get<idx::at>(arg);

      if (is_team_lead()) {
        const size_type fnid = rec.fnid;
        convolution_process_one_leader<T, NDIM>(row_data(in_view, fnid, rec.in_before), row_data(f_view, fnid, rec.f_before),
                                                row_data(result_view, fnid, rec.local_pos), tmp, K, TWOK, N,
                                                rec.local_pos, blockIdx.x, f0, resultc, work1, work2, staged_in, f, in,
                                                result, fnid, i,
                                                SDone ? &s_result[pos * mra::pow(K, Int<NDIM>{})] : nullptr);
      }
      SYNCTHREADS();

      if (f_view.is_zero(i)) {  // no terms: whole
        result = in;
        if (!resnorms.empty()) {
          auto resnorm = normf(result);
          if (is_team_lead()) resnorms[i] = resnorm;
        }
      } else {
        const T cnorm = rec.cnorm;
        const int opid = opnorms.dim(0) > 1 ? static_cast<int>(i) : 0;
        const T opnorm = opnorms(opid, 0, 0, (size_type)NormId::Opnorm);
        const bool gate = (cnorm * opnorm) > (tol / fac);
        const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank);
        const T optol = 0.01*(tol / fac / cnorm)/rank;
        if (p.whole) staged_in = in;  // result may be in's storage, see convolution_process_one
        if (rec.nr > 0) {  // nr > 0 implies at[0] and the gate
          /* the slot of a split position's piece (not used if whole) */
          tensor_view_2k_type part(b.partials + lb_slot(g, p.g0) * k3,
                                   broadcast_dims<typename tensor_view_2k_type::dims_type>(TWOK, std::make_index_sequence<NDIM>{}));
          const uint4* tt = b.tterms + size_type(rec.member) * b.tcap + size_type(opid) * opnorms.dim(1);
          mra::accel::apply_term<T, NDIM, NormId::Rnorm, true>(opid, TWOK, rank, optol, std::get<idx::transr>(arg),
                                                              opnorms, f, p.whole ? result : part, p.b_begin, p.b_count, tt);
        } else {
          result = 0.0;  // whole: no k = 2K term applied
        }
        if (!p.whole) {
          /* publish this piece; the last one adds up the pieces (the non-empty parts g0 .. g1) */
          __threadfence();
          SYNCTHREADS();
          if (is_team_lead()) {
            unsigned pieces = 0;
            for (unsigned h = p.g0; h <= p.g1; ++h) {
              const lb_range r = lb_sub(q, h);
              pieces += (r.lo < r.hi) ? 1u : 0u;
            }
            last = (atomicAdd(&b.arrive[pos], 1u) + 1 == pieces);
          }
          SYNCTHREADS();
          if (!last) return;
          __threadfence();
          staged_in = in;
          T* rptr = result.data();
          bool first = true;
          for (unsigned h = p.g0; h <= p.g1; ++h) {
            const lb_range r = lb_sub(q, h);
            if (r.lo == r.hi) continue;
            const T* src = b.partials + lb_slot(h, p.g0) * k3;
            for (size_type e = thread_id(); e < k3; e += block_size()) rptr[e] = first ? src[e] : rptr[e] + src[e];
            first = false;
          }
          SYNCTHREADS();
        }
        if (gate) {
          std::array<Slice, NDIM> s0 = std::array<Slice, NDIM>{Slice(0, K), Slice(0, K), Slice(0, K)};
          if (!at[1]) {
            resultc = 0.0;
          } else if constexpr (!SDone) {
            f0 = f(s0);
            mra::accel::apply_term<T, NDIM, NormId::Snorm>(opid, K, rank, optol, std::get<idx::transs>(arg),
                                                          opnorms, f0, resultc);
          }
          result(s0) += resultc;
        }
        convolution_finalize<T, NDIM>(fac, tol, staged_in, result, resnorms.empty() ? nullptr : &resnorms[i]);
      }
      if (!resnorms.empty() && is_team_lead() && resnorms[i] == 0.0) {
        result_view.set_state(i, SparsityState::ALLOCATED);
      }
#endif // __CUDA_ARCH__
    }

    /** Runs part g's piece of position pos, if it has one (both modes). */
    template <typename T, Dimension NDIM, typename KT>
    DEVSCOPE void lb_run_pos(
      ConvolutionBatchArg<T, NDIM>* args,
      unsigned pos,
      unsigned g,
      lb_range rg,
      const lb_parts& q,
      const lb_bufs<T>& b,
      T* shared_tmp,
      T* s_result,
      KT K,
      const T fac)
    {
      const lb_rec<T> rec = b.recs[pos];
      const lb_piece_t p = lb_piece_of(b.prefix[pos], b.prefix[pos + 1], rec.nr, b.c0, rg, q);
      if (p.work) {
        convolution_lb_piece<T, NDIM, mra::accel::apply_conv_splits_s<KT>()>(args[rec.member], rec, pos, K, fac,
                                                                            shared_tmp, s_result, p, g, q, b);
      }
    }

    /**
     * Load balancing, mode 1: one part per block. A block takes the next part
     * of its SM (g = SM * R + slot) and runs its positions in order; a block
     * that is done takes a part no block has taken (an SM can get fewer than
     * R blocks).
     */
    template <typename T, Dimension NDIM, typename KT>
    LAUNCH_BOUNDS(conv_launch_threads<KT>())
    GLOBALSCOPE void convolution_kernel_batched_lb(
      ConvolutionBatchArg<T, NDIM>* args,
      size_type total_nonzero,
      T* shared_tmp,
      T* s_result,
      lb_bufs<T> b,
      KT K,
      const T fac)
    {
#if defined(__CUDA_ARCH__)
      const lb_parts q{b.prefix[total_nonzero], b.num_sms, gridDim.x / b.num_sms};
      const unsigned G = q.S * q.R;
      unsigned* claimed = b.counters + b.num_sms;
      SHARED unsigned g_sh;

      if (is_team_lead()) {  // home part: this SM's next slot
        const unsigned s = lb_smid();
        const unsigned j = s < b.num_sms ? atomicAdd(&b.counters[s], 1u) : q.R;
        const unsigned g = s * q.R + j;
        g_sh = (j < q.R && atomicCAS(&claimed[g], 0u, 1u) == 0u) ? g : G;
      }
      SYNCTHREADS();
      /* takes a part no block has taken, or returns G */
      auto steal = [&]() -> unsigned {
        SYNCTHREADS();  // every thread has read g_sh
        if (thread_id() < 32) {
          unsigned c;
          while (true) {
            c = lb_warp_first(G, [&](unsigned x) { return *reinterpret_cast<volatile unsigned*>(&claimed[x]) == 0u; });
            if (c == G) break;
            unsigned won = 0;
            if (thread_id() == 0) won = (atomicCAS(&claimed[c], 0u, 1u) == 0u);
            if (__shfl_sync(0xffffffffu, won, 0)) break;
          }
          if (thread_id() == 0) g_sh = c;
        }
        SYNCTHREADS();
        return g_sh;
      };
      unsigned g = g_sh;
      if (g == G) g = steal();
      while (g != G) {
        const lb_range rg = lb_sub(q, g);
        for (unsigned pos = b.pos_lo[g], end = b.pos_hi[g]; pos < end; ++pos) {
          lb_run_pos<T, NDIM>(args, pos, g, rg, q, b, shared_tmp, s_result, K, fac);
          SYNCTHREADS();
        }
        g = steal();
      }
#endif // __CUDA_ARCH__
    }

    /**
     * Load balancing, mode 2: one part per SM. The blocks on an SM take its
     * positions one at a time from a per-SM cursor (b.counters), so they
     * finish together even if one is faster. A block whose SM is done helps
     * another SM.
     */
    template <typename T, Dimension NDIM, typename KT>
    LAUNCH_BOUNDS(conv_launch_threads<KT>())
    GLOBALSCOPE void convolution_kernel_batched_lb_sm(
      ConvolutionBatchArg<T, NDIM>* args,
      size_type total_nonzero,
      T* shared_tmp,
      T* s_result,
      lb_bufs<T> b,
      KT K,
      const T fac)
    {
#if defined(__CUDA_ARCH__)
      const lb_parts q{b.prefix[total_nonzero], b.num_sms, 1};
      SHARED unsigned s_sh, k_sh;
      unsigned k_next = 0;  // team lead

      if (is_team_lead()) {
        const unsigned sm = lb_smid();
        s_sh = sm < b.num_sms ? sm : 0;
        k_sh = atomicAdd(&b.counters[s_sh], 1u);
      }
      SYNCTHREADS();
      while (true) {
        const unsigned s = s_sh, pos = b.pos_lo[s] + k_sh;
        SYNCTHREADS();  // every thread has read s_sh and k_sh
        if (pos < b.pos_hi[s]) {
          if (is_team_lead()) k_next = atomicAdd(&b.counters[s], 1u);  // the next position, claimed ahead
          lb_run_pos<T, NDIM>(args, pos, s, lb_sub(q, s), q, b, shared_tmp, s_result, K, fac);
          if (is_team_lead()) k_sh = k_next;
          SYNCTHREADS();
          continue;
        }
        /* this SM's part is done: help one that is not */
        if (thread_id() < 32) {
          const unsigned t = lb_warp_first(b.num_sms, [&](unsigned x) {
            return b.pos_lo[x] + *reinterpret_cast<volatile unsigned*>(&b.counters[x]) < b.pos_hi[x];
          });
          if (thread_id() == 0) {
            s_sh = t;
            if (t < b.num_sms) k_sh = atomicAdd(&b.counters[t], 1u);
          }
        }
        SYNCTHREADS();
        if (s_sh == b.num_sms) break;
      }
#endif // __CUDA_ARCH__
    }

#if defined(MRA_ENABLE_CUDA)
    /** MRA_CONV_LB: 0 off (convolution_kernel_batched), 1 convolution_kernel_batched_lb, 2 ..._lb_sm. Read once. */
    inline int lb_mode() {
      static const int m = [] { const char* e = std::getenv("MRA_CONV_LB"); return e != nullptr ? std::atoi(e) : 0; }();
      return m;
    }

    /**
     * The load-balanced launch: pre-pass, scan, then the mode's kernel at the
     * resident grid. Scratch: tmp | S terms | partial results | records |
     * units, prefix, arrive, pos_lo, pos_hi, counters | term table.
     */
    template <typename T, Dimension NDIM, typename KT, typename SlotT>
    void submit_convolution_lb(
      SlotT& slot,
      int device_id,
      size_type num_members,
      size_type total_nonzero,
      KT K,
      const T fac,
      Dim3 thread_dims,
      std::size_t smem_size,
      ttg::device::Stream stream)
    {
      // an empty grid is an invalid launch (see submit_convolution_kernel_batched)
      if (total_nonzero == 0) return;
      /* a position's fixed cost in units; >= 1 so every position has a unit */
      static const unsigned c0 = [] {
        const char* e = std::getenv("MRA_CONV_LB_C0");
        return static_cast<unsigned>(std::max(1, e != nullptr ? std::atoi(e) : 10));
      }();
      static const unsigned num_sms = [] {
        int device = 0, n = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
        return static_cast<unsigned>(n);
      }();
      constexpr bool SDone = mra::accel::apply_conv_splits_s<KT>();
      const bool per_sm = lb_mode() == 2;
      auto kernel = per_sm ? &convolution_kernel_batched_lb_sm<T, NDIM, KT> : &convolution_kernel_batched_lb<T, NDIM, KT>;
      const int threads = static_cast<int>(thread_dims.x * thread_dims.y * thread_dims.z);
      const unsigned grid = kernel_grid_cap(kernel, threads, smem_size, 1);
      const unsigned R = per_sm ? 1 : grid / num_sms;
      const std::size_t n = total_nonzero, NP = std::size_t(num_sms) * R;
      constexpr std::size_t kK = KT::value;
      const std::size_t k3 = mra::pow(2 * kK, NDIM);
      unsigned tcap = 0;  // term table entries per member: the largest operators x rank
      for (const auto& a : slot.args) {
        const auto& opn = std::get<ConvolutionBatchArgIdx::opnorms>(a);
        tcap = std::max<unsigned>(tcap, static_cast<unsigned>(opn.dim(0) * opn.dim(1)));
      }

      const std::size_t b_tmp = align_to_type<T>(grid * convolution_tmp_size<NDIM>(K) * sizeof(T));
      const std::size_t b_s = align_to_type<T>(SDone ? n * mra::pow(kK, NDIM) * sizeof(T) : 0);
      const std::size_t b_part = align_to_type<T>(2 * NP * k3 * sizeof(T));
      const std::size_t b_rec = align_to_type<T>(n * sizeof(lb_rec<T>));
      const std::size_t ncounters = num_sms + NP;
      const std::size_t b_u32 = align_to_type<T>((3 * n + 1 + 2 * NP + ncounters) * sizeof(unsigned));
      const std::size_t b_terms = std::size_t(num_members) * tcap * sizeof(uint4);
      slot.dev_scratch.grow(device_id, b_tmp + b_s + b_part + b_rec + b_u32 + b_terms);
      char* ptr = reinterpret_cast<char*>(slot.dev_scratch.get());
      T* shared_tmp = reinterpret_cast<T*>(ptr);
      T* s_result = SDone ? reinterpret_cast<T*>(ptr + b_tmp) : nullptr;
      lb_bufs<T> b;
      b.partials = reinterpret_cast<T*>(ptr + b_tmp + b_s);
      b.recs = reinterpret_cast<lb_rec<T>*>(ptr + b_tmp + b_s + b_part);
      b.units = reinterpret_cast<unsigned*>(ptr + b_tmp + b_s + b_part + b_rec);
      b.prefix = b.units + n;
      b.arrive = b.prefix + n + 1;
      b.pos_lo = b.arrive + n;
      b.pos_hi = b.pos_lo + NP;
      b.counters = b.pos_hi + NP;
      b.tterms = reinterpret_cast<uint4*>(ptr + b_tmp + b_s + b_part + b_rec + b_u32);
      b.tcap = tcap;
      b.num_sms = num_sms;
      b.c0 = c0;

      constexpr int pw = mra::accel::apply_conv_s_warps;
      const size_type grid_p = std::min<size_type>((total_nonzero + pw - 1) / pw,
          kernel_grid_cap(convolution_lb_prepass_batched<T, NDIM, KT>, pw * 32, 0, 1));
      CALL_KERNEL((convolution_lb_prepass_batched<T, NDIM, KT>), grid_p, pw * 32, 0, stream,
                  (slot.dev_args, slot.dev_offsets, num_members, total_nonzero, b, s_result, K, fac));
      CALL_KERNEL((convolution_lb_scan<1024>), 1, 1024, 0, stream,
                  (b.units, b.prefix, static_cast<unsigned>(total_nonzero), num_sms, R, b.pos_lo, b.pos_hi, b.counters,
                   static_cast<unsigned>(ncounters)));
      CALL_KERNEL(kernel, grid, thread_dims, smem_size, stream,
                  (slot.dev_args, total_nonzero, shared_tmp, s_result, b, K, fac));
    }
#endif // MRA_ENABLE_CUDA

#if defined(MRA_CHECK_NORMS)
    /**
     * Debug-only: cross-checks, for every member, that the flattened launch
     * grid's slice for that member (member_offsets[m+1] - member_offsets[m],
     * a running sum of each member's own host-computed n_nonzero -- see
     * submit_convolution_batch_leader) agrees with a fresh on-device scan of
     * that same member's result_view (a single-view scan suffices -- see
     * convolution_verify_sparsity_kernel_single's comment). One block per
     * member. Launched (gated by MRA_CHECK_NORMS) immediately AFTER
     * submit_grouped_copy and before convolution_kernel_batched in
     * submit_convolution_kernel_batched -- unlike reconstruct's batched
     * check, this one must run after that transfer, since result_view's own
     * device bitfield (what's being checked here) is exactly what it just
     * scattered (see submit_grouped_copy).
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
   * Batched counterpart of submit_convolution_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.args (by the caller,
   * via detail::submit_convolution_batch_leader below), sharing only (K, fac)
   * across the whole batch -- tol/at/transr/transs/opnorms are per-member,
   * already inside slot.args. Grid is 1D over total_nonzero -- see
   * convolution_kernel_batched's comment for why. slot.offsets carries the
   * small per-member offsets array (size num_members+1), assembled by
   * submit_convolution_batch_leader alongside slot.args and slot.extra_dsts/
   * extra_srcs/extra_sizes (the per-member sparsity-scatter and
   * resnorms-zero destinations -- see submit_grouped_copy's comment) in the
   * same GroupedBatchPool slot -- one event for the whole launch instead of
   * separate ones per region.
   */
  template <typename T, Dimension NDIM>
  void submit_convolution_kernel_batched(
    detail::GroupedBatchPool<detail::ConvolutionBatchArg<T, NDIM>>& pool,
    typename detail::GroupedBatchPool<detail::ConvolutionBatchArg<T, NDIM>>::slot_t& slot,
    size_type total_nonzero,
    size_type K,
    const T fac,
    ttg::device::Stream stream)
#if defined(MRA_HAVE_KERNEL_DEFINITIONS)
  {
    using idx = detail::ConvolutionBatchArgIdx;
    using arg_t = detail::ConvolutionBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.args.size());

    // Single combined H2D transfer for args/offsets, plus every member's
    // sparsity-scatter and resnorms-zero destination copy -- see
    // submit_grouped_copy.
    detail::submit_grouped_copy<arg_t>(slot, num_members, slot.offsets.size(), pool.device_id, stream);

#if defined(MRA_CHECK_NORMS)
    // Debug-only: verify the flattened grid's per-member slice (derived from
    // each member's host-computed n_nonzero) still matches a fresh on-device
    // scan of that member's result_view -- must run AFTER the scatter above,
    // since result_view's own device bitfield is exactly what that scatter
    // just wrote. See convolution_verify_sparsity_kernel's comment.
    CALL_KERNEL((detail::convolution_verify_sparsity_kernel<T, NDIM>), num_members, 32, 0, stream,
                (slot.dev_args, slot.dev_offsets));
    checkSubmit();
#endif // MRA_CHECK_NORMS

    Dim3 thread_dims = detail::conv_thread_dims(K);
    auto smem_size = detail::apply_conv_shmem_size<T>(K);
    const int threads = static_cast<int>(thread_dims.x) * static_cast<int>(thread_dims.y) * static_cast<int>(thread_dims.z);

    // total_nonzero can run into the thousands for a large batch, far beyond
    // what the device can run concurrently -- cap the grid at actual
    // occupancy and let convolution_kernel_batched's own grid-stride loop
    // (see its comment) pick up the rest, rather than submitting one
    // immediately-resident block per position.
    //
    // Sparsity demotion for a member position whose computed norm turns out
    // to be exactly zero happens inline inside convolution_kernel_batched
    // itself (see convolution_process_one's comment) -- no separate
    // post-launch prune kernel needed.
    // Scratch is sized from grid_cap -- a fixed bound for this kernel/device,
    // independent of total_nonzero -- not from grid_x, so ensure_scratch's
    // first allocation already covers every later launch and never needs to
    // grow (see its comment for why growing it would be unsafe).
    detail::with_ct_K(K, [&](auto ctK) {
#if defined(MRA_ENABLE_CUDA)
      // load balancing (MRA_CONV_LB, see detail::submit_convolution_lb), for the Ks with a tensor-core path
      if constexpr (mra::accel::apply_conv_has_path<decltype(ctK)>()) {
        if (detail::lb_mode() != 0) {
          detail::submit_convolution_lb<T, NDIM>(slot, pool.device_id, num_members, total_nonzero, ctK, fac,
                                                 thread_dims, smem_size, stream);
          return;
        }
      }
#endif // MRA_ENABLE_CUDA
      size_type grid_cap = detail::kernel_grid_cap(detail::convolution_kernel_batched<T, NDIM, decltype(ctK)>, threads, smem_size);
      size_type grid_x = std::min<size_type>(total_nonzero, grid_cap);
      const std::size_t tmp_elems = static_cast<std::size_t>(grid_cap) * convolution_tmp_size<NDIM>(K);
      T* shared_tmp = nullptr;
      T* s_result = nullptr;
      if constexpr (mra::accel::apply_conv_splits_s<decltype(ctK)>()) {
        // The S terms go after the blocks' scratch, so this launch's size sets
        // the region's size. grow() is safe: acquire() only hands out a slot
        // whose launches have finished (see FixedDeviceRegion).
        const std::size_t s_elems = static_cast<std::size_t>(total_nonzero) * mra::pow(K, Int<NDIM>{});
        slot.dev_scratch.grow(pool.device_id, (tmp_elems + s_elems) * sizeof(T));
        shared_tmp = reinterpret_cast<T*>(slot.dev_scratch.get());
        s_result = shared_tmp + tmp_elems;
        // the S term first, one warp per position (see convolution_s_kernel_batched)
        constexpr int s_warps = mra::accel::apply_conv_s_warps;
        constexpr int s_threads = s_warps * 32;
        size_type grid_s = std::min<size_type>((total_nonzero + s_warps - 1) / s_warps,
            detail::kernel_grid_cap(detail::convolution_s_kernel_batched<T, NDIM, decltype(ctK)>, s_threads, 0, 1));
        // an empty grid is an invalid launch (as below)
        if (grid_s > 0) {
          CALL_KERNEL((detail::convolution_s_kernel_batched<T, NDIM>), grid_s, s_threads, 0, stream,
                      (slot.dev_args, slot.dev_offsets, num_members, total_nonzero, s_result, ctK, fac));
        }
      } else {
        shared_tmp = pool.template ensure_scratch<T>(slot, tmp_elems);
      }
      // an empty grid (no nonzero function in the batch) is an invalid launch
      if (grid_x > 0) {
        CALL_KERNEL((detail::convolution_kernel_batched<T, NDIM>), grid_x, thread_dims, smem_size, stream,
                    (slot.dev_args, slot.dev_offsets, num_members, total_nonzero, shared_tmp, s_result, ctK, fac));
      }
    });
    checkSubmit();

    pool.mark_submitted(slot, stream);
  }
#else
  ;
#endif // MRA_HAVE_KERNEL_DEFINITIONS

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
     * SparsityManager/MockTensor allocation), stage every member's bytes
     * into the slot's own pinned sparsity region (part of the same
     * GroupedBatchPool slot as slot.args/slot.offsets, not the separate
     * process-wide pool SparsityManager uses for non-batched pushes -- see
     * sparsitymanager.h), and queue a direct copy from that staged slice
     * straight to `m_out`'s own device buffer -- which is exactly where its
     * inline sparsity bitfield starts (see submit_grouped_copy's comment) --
     * as one of slot.extra_dsts/extra_srcs/extra_sizes' entries, which
     * submit_convolution_kernel_batched's single submit_grouped_copy call
     * scatters to its final destination before the compute kernel runs.
     *
     * resnorms: each member also now passes its own `resnorms` Tensor
     * (get<12>(), the real Tensor -- separate from resnorms_view, get<3>(),
     * which is what the compute kernel itself uses) through coop(), purely
     * so this leader can queue a second extra-copy entry per member: a
     * zero-fill (nullptr source) of `m_resnorms`'s own device buffer,
     * covering positions
     * convolution_process_one won't visit for this member (mirroring
     * convolution_kernel's unbatched tail pass -- see its comment). Skipped
     * for a non-last-key accumulate_tt member, which passes an empty
     * resnorms_view/tensor through coop() (get<3>().empty()).
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
     * region's first allocation to a fixed upper bound
     * (max_batch_size * total_functions), so it never needs to grow after
     * that, and to size the shared zero-source buffer's eventual max extent
     * (every member's resnorms is at most total_functions elements).
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_convolution_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<ConvolutionBatchArg<T, NDIM>>& registry,
      size_type K,
      const T fac,
      size_type total_functions)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      const auto device = ttg::device::current_device();
      auto& pool = registry.get(device);

      // Args/offsets regions sized to a fixed upper bound (max_batch_size,
      // +1 for offsets), and the sparsity staging region to max_batch_size *
      // total_functions bytes (every member contributes at most
      // total_functions bytes) -- not the exact sizes needed this launch --
      // so the slot's device buffers are each allocated once, on first use,
      // and never resized after that.
      const size_type max_sparsity_bytes =
          static_cast<size_type>(registry.get_max_batch_size()) * total_functions;
      auto& slot = pool.acquire(registry.get_max_batch_size(), registry.get_max_batch_size() + 1, max_sparsity_bytes);
      slot.args.clear();
      slot.offsets.resize(nb + 1);
      slot.offsets[0] = 0;
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();
      slot.extra_dsts.reserve(2 * nb);  // sparsity + resnorms destination per member, upper bound
      slot.extra_srcs.reserve(2 * nb);
      slot.extra_sizes.reserve(2 * nb);

      // Logical (used) sparsity-byte size for this launch varies (different
      // nodes can have different structural N), unlike the slot's capacity
      // above.
      size_type total_sparsity_bytes = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        total_sparsity_bytes += static_cast<size_type>(batch[m].template get<2>().dim(0));
      }
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_in            = batch[m].template get<0>();
        auto& m_f             = batch[m].template get<1>();
        auto& m_result        = batch[m].template get<2>();
        auto& m_resnorms      = batch[m].template get<3>();
        // No per-member tmp extraction anymore -- see ConvolutionBatchArg's
        // comment: every member's scratch now comes from one shared,
        // occupancy-sized pool instead (ensure_scratch in
        // submit_convolution_kernel_batched).
        auto& m_transr        = batch[m].template get<4>();
        auto& m_transs        = batch[m].template get<5>();
        auto& m_opnorms       = batch[m].template get<6>();
        auto& m_tol           = batch[m].template get<7>();
        auto& m_at            = batch[m].template get<8>();
        auto& m_out           = batch[m].template get<9>(); // real out tensor: sparsity source AND this member's scatter destination
        const size_type m_n_nonzero = batch[m].template get<10>();
        auto& m_resnorms_tensor = batch[m].template get<11>(); // real resnorms tensor: this member's zero-fill destination
        const size_type n = static_cast<size_type>(m_result.dim(0)); // structural N

        sparsity_to_bytes(m_out.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), n);
        slot.extra_dsts.push_back(const_cast<T*>(m_out.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(n);
        sparsity_offset += n;

        if (!m_resnorms.empty()) {
          slot.extra_dsts.push_back(const_cast<T*>(m_resnorms_tensor.buffer().device_ptr_on(device)));
          slot.extra_srcs.push_back(nullptr); // zero-fill, see GroupedBatchPool::slot_t::extra_srcs
          slot.extra_sizes.push_back(n * sizeof(T));
        }

        slot.args.emplace_back(m_in, m_f, m_result, m_resnorms, n,
                               m_transr, m_transs, m_opnorms, m_tol, m_at);

        slot.offsets[m + 1] = slot.offsets[m] + m_n_nonzero;
      }
      const size_type total_nonzero = slot.offsets[nb];
      submit_convolution_kernel_batched<T, NDIM>(pool, slot, total_nonzero, K, fac, ttg::device::current_stream());
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
    const std::array<SparseTensorView<double, 3>, 3>& transr,
    const std::array<SparseTensorView<double, 3>, 3>& transs,
    const DenseTensorView<double, 4>& opnorms,
    const std::array<bool, 2>& at,
    double* tmp,
    ttg::device::Stream stream);

#ifndef MRA_ENABLE_HOST
  extern template
  void submit_convolution_kernel_batched<double, 3>(
    detail::GroupedBatchPool<detail::ConvolutionBatchArg<double, 3>>& pool,
    typename detail::GroupedBatchPool<detail::ConvolutionBatchArg<double, 3>>::slot_t& slot,
    size_type total_nonzero,
    size_type K,
    const double fac,
    ttg::device::Stream stream);
#endif // !MRA_ENABLE_HOST

#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra

#endif // MRA_KERNELS_CONVOLUTION_H
