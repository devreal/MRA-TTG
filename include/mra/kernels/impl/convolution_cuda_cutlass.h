#ifndef MRA_KERNELS_IMPL_CONVOLUTION_CUDA_CUTLASS_H
#define MRA_KERNELS_IMPL_CONVOLUTION_CUDA_CUTLASS_H

#include "mra/misc/types.h"
#include "mra/misc/integer.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/conv_mad.h" // for NormId, used as a template parameter below

/**
 * CUTLASS/CuTe-based alternative to convolution_cuda_wmma.h's hand-rolled
 * nvcuda::wmma implementation. See the design plan this file implements for
 * the full rationale; in short:
 *
 *  - convolution_cuda_wmma.h's apply_conv_k has every warp in the block
 *    independently re-load the ENTIRE per-(mu,dimension) operator tile
 *    (`trans[d](opid,mu)`) from global memory on every iteration (its own
 *    comment admits this: "All warps hold the full set of B fragments...
 *    because each warp will need to multiply by all of them") -- an
 *    apply_conv_min_warps(8)-fold redundant global/L1 load every (mu,d).
 *    This file fixes that by cooperatively staging that tile into shared
 *    memory ONCE per block (cute::cooperative_copy), read by every warp from
 *    there instead.
 *  - It also has no control over the shared-memory layout used to rotate
 *    the intermediate result between dimensions (a flat, unswizzled
 *    buffer), a classic FP64 bank-conflict setup. This file applies
 *    CUTLASS's own canonical FP64 tensorop shared-memory swizzle
 *    (cute::Swizzle<2,2,2> over a 16x4 atom tile -- confirmed against
 *    CUTLASS v3.8.0's own test suite, test/unit/cute/ampere/
 *    cooperative_gemm.cu's CooperativeGemm5_Double_MMA_SwizzledSmemLayouts)
 *    to the newly-staged operator tile.
 *
 * Scoping decision (deliberately NOT attempted here): the dimension-rotation
 * buffer itself (what was WMMA's `c_smem`, here three fixed-role buffers --
 * see apply_conv_k below) keeps the EXACT SAME unswizzled addressing as the
 * proven-correct WMMA/generic paths (row-major write at one dimension,
 * col-major reinterpretation as the next dimension's input). Swizzling that
 * buffer too would need a single swizzle simultaneously conflict-free for
 * both orientations, which is a stricter, harder-to-verify-without-a-GPU
 * constraint than the one-way B-tile staging above; it is left as a
 * follow-up once real hardware (Nsight Compute bank-conflict counters) is
 * available to validate any candidate swizzle there. Fixing the B-tile
 * redundant-global-load bug alone, with high confidence, was judged better
 * than risking a silently-wrong numerical kernel in a scientific code by
 * also guessing at the harder, dual-orientation swizzle.
 *
 * Another consequence of needing the operator tile in shared memory (not
 * redundantly in each warp's registers) is that the *coefficient* operand
 * (`f`, WMMA's register-resident `f_frags`) must now also live in shared
 * memory for CUTLASS's cute::cooperative_gemm to consume it -- it no longer
 * stays resident in registers across the whole `mu` loop the way WMMA's
 * hand-written fragments did. This trades away some of that register
 * residency in exchange for using CUTLASS's own tested, documented
 * primitives rather than hand-rolling an equivalent; it costs extra shared
 * memory *bandwidth* (not extra global memory traffic -- `f` is still
 * staged into shared memory only ONCE per block) and noticeably more
 * shared-memory *capacity* than the WMMA kernel needed (three operand-sized
 * buffers instead of one -- see apply_conv_shmem_size below). This is a
 * real, intentional trade-off, not an oversight.
 *
 * IMPORTANT: this file has been written against the real CUTLASS v3.8.0
 * headers (atom name, canonical swizzle recipe, and the cooperative_copy/
 * cooperative_gemm call shapes below were all confirmed by reading that
 * source, not guessed), but could not be compiled or run against an actual
 * GPU in the environment that produced it -- there was no CUDA toolchain or
 * GPU reachable. Per this project's CLAUDE.md, build it in a CUDA-enabled
 * container and treat functional validation (src/tests/mad_convolution_test,
 * which cross-checks against real MADNESS output) as still outstanding
 * before trusting this path.
 */

#if __has_include(<cutlass/cutlass.h>) && __has_include(<cute/tensor.hpp>)
#define MRA_HAVE_CUTLASS_HEADERS 1
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#endif

/* Device-side availability for the actual compute path: FP64 tensor cores
 * only exist on sm_80+ (same constraint as the WMMA backend). The sizing
 * helpers below (apply_conv_shmem_size, apply_conv_min_threads) must stay
 * available on BOTH nvcc compilation passes of a .cu translation unit (the
 * host pass, which needs the real size to configure CALL_KERNEL's launch,
 * and the device pass) -- __CUDA_ARCH__ is only defined on the device pass,
 * so (mirroring convolution_cuda_wmma.h's own MRA_ENABLE_CUDA-gated
 * apply_conv_shmem_size, as opposed to its __CUDA_ARCH__-gated mma_traits)
 * those two helpers are gated on MRA_HAVE_CUTLASS_HEADERS alone. Only the
 * actual device-compute bodies (apply_conv_k, apply_conv) need the stricter
 * arch-gated macro below, since they use real device-only intrinsics
 * (inline PTX inside the MMA atom, cp.async). */
#if defined(MRA_HAVE_CUTLASS_HEADERS) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define MRA_HAVE_CUTLASS_MMA 1
#endif

namespace mra {
  namespace accel {

    /** Same contract as convolution_cuda_wmma.h's apply_conv_min_warps. */
    inline constexpr int apply_conv_min_warps = 8;

    /** Same contract as convolution_cuda_wmma.h's apply_conv_min_threads. */
    SCOPE constexpr int apply_conv_min_threads(auto K) {
#ifdef MRA_ENABLE_CUDA
      if (K == 8) {
        return apply_conv_min_warps * 32;
      }
#endif // MRA_ENABLE_CUDA
      return 0;
    }

#ifdef MRA_HAVE_CUTLASS_HEADERS
    namespace detail {

      /**
       * CUTLASS's own canonical FP64 tensorop shared-memory swizzle recipe,
       * tiled to an arbitrary (rows, cols) logical shape -- confirmed
       * against CUTLASS v3.8.0's test/unit/cute/ampere/cooperative_gemm.cu,
       * test CooperativeGemm5_Double_MMA_SwizzledSmemLayouts (smem_a_atom_
       * layout/smem_b_atom_layout there), not derived from first principles.
       * Used ONLY for the operator-tile (B) staging buffer -- see this
       * file's header comment for why the rotation buffers are left
       * unswizzled.
       */
      template <class Shape2>
      CUTE_HOST_DEVICE constexpr auto fp64_swizzled_layout(Shape2 shape) {
        auto atom = cute::composition(cute::Swizzle<2, 2, 2>{},
                                       cute::Layout<cute::Shape<cute::_16, cute::_4>,
                                                     cute::Stride<cute::_1, cute::_16>>{});
        return cute::tile_to_shape(atom, shape);
      }

      /**
       * Element count of the shared-memory footprint apply_conv_k needs for
       * one local tile size Kc (the loop-fixed NDIM=3 version): three
       * M=Kc*Kc by Kc operand-sized buffers (the immutable f bootstrap plus
       * two fixed-role rotation stages -- see apply_conv_k) and one swizzled
       * Kc x Kc operator-tile staging buffer. A plain constexpr function
       * (no __CUDA_ARCH__ dependency), safe to call from host code sizing a
       * kernel launch.
       */
      template <typename T, int Kc>
      CUTE_HOST_DEVICE constexpr mra::size_type cutlass_conv_shmem_elems() {
        constexpr int M = Kc * Kc;
        auto bs_layout = fp64_swizzled_layout(cute::Shape<cute::Int<Kc>, cute::Int<Kc>>{});
        return static_cast<mra::size_type>(3 * M * Kc + cute::cosize(bs_layout));
      }

    } // namespace detail
#endif // MRA_HAVE_CUTLASS_HEADERS

    /**
     * Same contract as convolution_cuda_wmma.h's apply_conv_shmem_size:
     * called with the OUTER (S-term) K; internally sizes for the larger
     * R-term (Kc = 2K) call, since both apply_conv_k invocations inside one
     * apply_conv call share the same extern __shared__ region sequentially.
     * See the MRA_HAVE_CUTLASS_HEADERS-vs-MRA_HAVE_CUTLASS_MMA comment above
     * for why this is gated the way it is (must be correct on the host
     * compilation pass, not just __CUDA_ARCH__ device passes).
     */
    template<typename T>
    SCOPE constexpr size_type apply_conv_shmem_size(auto K) {
#ifdef MRA_HAVE_CUTLASS_HEADERS
      if (K == 8) {
        return detail::cutlass_conv_shmem_elems<T, 16>() * sizeof(T);
      }
#endif // MRA_HAVE_CUTLASS_HEADERS
      return 0;
    }

#ifdef MRA_HAVE_CUTLASS_MMA

    /**
     * One term's (R or S) worth of the fused, batched convolution GEMM
     * chain for one octree-node function: for each separable operator term
     * `mu` (0..rank-1, runtime trip count), applies the NDIM=3-fold chain
     * of small Kc^2-by-Kc times Kc-by-Kc GEMMs against `trans[d](opid,mu)`,
     * scaling the final dimension's result by `mufac` into the persistent
     * accumulator `result`. Structurally mirrors
     * convolution_cuda_wmma.h's apply_conv_k (same loop nest, same
     * mu-screening, same "scale-and-accumulate only at the last dimension"
     * semantics), but built on CUTLASS/CuTe's cute::cooperative_copy /
     * cute::cooperative_gemm instead of raw nvcuda::wmma intrinsics -- see
     * this file's header comment for exactly what that changes and why.
     *
     * Fixed-role shared-memory buffers (NDIM is static_assert'd == 3 below,
     * same restriction as the WMMA backend, so these are hardcoded rather
     * than indexed by a runtime/template d-loop):
     *   buf_f  -- immutable copy of `f`, staged ONCE before the mu loop,
     *             read as every mu's d=0 input. Never overwritten.
     *   buf_t1 -- output of d=0 / input of d=1 (col-major reinterpretation
     *             of the same bytes the d=0 GEMM wrote row-major, exactly
     *             mirroring the WMMA/generic paths' own dimension-cycling
     *             addressing trick -- see this file's header comment).
     *   buf_t2 -- output of d=1 / input of d=2 (the last dimension, whose
     *             result is consumed directly into the register
     *             accumulator rather than written back to shared memory).
     *   bstage -- swizzled staging buffer for trans[d](opid,mu), restaged
     *             cooperatively by the whole block on every (mu,d).
     */
    template<typename T, Dimension NDIM, NormId Term,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTrans,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewResult>
    DEVSCOPE void apply_conv_k(
      int opid,
      auto K,
      int rank,
      const T optol,
      const ViewTrans& trans,
      const ViewOpnorms& opnorms,
      ViewF& f,
      ViewResult& result)
    {
      static_assert(std::is_same_v<T, double>,
                    "Convolution CUTLASS backend currently only supports double precision "
                    "(the only type with an sm_80 tensor-core MMA atom)");
      static_assert(NDIM == 3, "Convolution CUTLASS backend only supports NDIM=3");

      using namespace cute;

      constexpr int Kc = static_cast<int>(decltype(K)::value);
      constexpr int M  = Kc * Kc;
      constexpr int Nn = Kc;
      constexpr int Kd = Kc;
      constexpr int kConvThreads = apply_conv_min_warps * MRA_WARP_SIZE;

      /**
       * MADNESS uses slightly different thresholds for R and S terms.
       */
      constexpr auto thresh = [&](){
        if (Term == NormId::Rnorm) return 1.e-20; else return 0.0;
      };

      using TiledMma = TiledMMA<
        MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,
        Layout<Shape<_8, _1, _1>>  // 8 atoms (warps) tiled along M only -- matches apply_conv_min_warps
      >;
      TiledMma tiled_mma;

      // A-operand layout: (M,K) col-major -- matches f.data()'s existing
      // layout exactly (see convolution_cuda_wmma.h's own "col-major ldm"
      // comment on the analogous load), left unswizzled (see header
      // comment).
      auto a_layout = make_layout(Shape<Int<M>, Int<Kd>>{});
      // C/rotation-output layout: (M,N) row-major, also left unswizzled.
      auto c_layout = make_layout(Shape<Int<M>, Int<Nn>>{}, GenRowMajor{});
      // B-operand (operator tile) layout: physically row-major (Kd rows, N
      // cols) in trans[d](opid,mu).data() -- presented to CuTe in its
      // (N,K) GEMM-operand convention as stride (1,N), i.e. the "N" ("not
      // transposed") side of the atom's "TN" naming -- and swizzled, the
      // actual fix this file makes.
      auto b_gmem_layout = make_layout(Shape<Int<Nn>, Int<Kd>>{}, Stride<_1, Int<Nn>>{});
      auto bs_layout = detail::fp64_swizzled_layout(Shape<Int<Nn>, Int<Kd>>{});

      extern SHARED T smem[];
      T* buf_f  = smem;
      T* buf_t1 = buf_f  + M * Kd;
      T* buf_t2 = buf_t1 + M * Kd;
      T* bstage = buf_t2 + M * Kd;

      Tensor sBstage = make_tensor(make_smem_ptr(bstage), bs_layout);

      // Bootstrap: stage f once, before the mu loop -- it never changes
      // across mu, so every mu's d=0 step re-reads this same buffer.
      {
        Tensor gF  = make_tensor(make_gmem_ptr(f.data()), a_layout);
        Tensor sAf = make_tensor(make_smem_ptr(buf_f), a_layout);
        cooperative_copy<kConvThreads, sizeof_bits_v<T>>(static_cast<uint32_t>(thread_id()), gF, sAf);
      }
      SYNCTHREADS();

      auto thr_mma = tiled_mma.get_thread_slice(static_cast<uint32_t>(thread_id()));
      Tensor gResult = make_tensor(make_gmem_ptr(result.data()), c_layout);
      Tensor tCgC = thr_mma.partition_C(gResult);
      auto tCrC_persist = thr_mma.make_fragment_C(tCgC);
      clear(tCrC_persist);
      auto tCrC_tmp = thr_mma.make_fragment_C(tCgC);

      for (int mu = 0; mu < rank; ++mu) {
        T munorm = opnorms(opid, mu, 0, (size_type)NormId::MUnorm);
        double dnorm = 1.0;
        for (Dimension d=0; d<NDIM; ++d) dnorm *= opnorms(opid, mu, d, (size_type)Term);
        if (munorm > optol && dnorm > thresh()) {
          T mufac = opnorms(opid, mu, 0, (size_type)NormId::Fac);
          if (NormId::Snorm == Term) mufac *= -1.0; // sign flip for Snorm

          // d = 0: buf_f (A) x trans[0](opid,mu) (B) -> buf_t1 (C)
          {
            Tensor sA = make_tensor(make_smem_ptr(buf_f), a_layout);
            Tensor gB = make_tensor(make_gmem_ptr(trans[0](opid, mu).data()), b_gmem_layout);
            cooperative_copy<kConvThreads, sizeof_bits_v<T>>(static_cast<uint32_t>(thread_id()), gB, sBstage);
            cp_async_fence();
            cp_async_wait<0>();
            SYNCTHREADS();
            Tensor sC = make_tensor(make_smem_ptr(buf_t1), c_layout);
            cooperative_gemm(static_cast<uint32_t>(thread_id()), tiled_mma, T(1.0), sA, sBstage, T(0.0), sC);
            SYNCTHREADS();
          }

          // d = 1: buf_t1 reinterpreted as A x trans[1](opid,mu) (B) -> buf_t2 (C)
          {
            Tensor sA = make_tensor(make_smem_ptr(buf_t1), a_layout);
            Tensor gB = make_tensor(make_gmem_ptr(trans[1](opid, mu).data()), b_gmem_layout);
            cooperative_copy<kConvThreads, sizeof_bits_v<T>>(static_cast<uint32_t>(thread_id()), gB, sBstage);
            cp_async_fence();
            cp_async_wait<0>();
            SYNCTHREADS();
            Tensor sC = make_tensor(make_smem_ptr(buf_t2), c_layout);
            cooperative_gemm(static_cast<uint32_t>(thread_id()), tiled_mma, T(1.0), sA, sBstage, T(0.0), sC);
            SYNCTHREADS();
          }

          // d = 2 (last): buf_t2 reinterpreted as A x trans[2](opid,mu) (B)
          // -> scaled by mufac and accumulated into the persistent
          // register accumulator (survives across the whole mu loop,
          // exactly mirroring convolution_cuda_wmma.h's
          // `c_frags[i][j].x[e] += mufac * acc.x[e]`).
          {
            Tensor sA = make_tensor(make_smem_ptr(buf_t2), a_layout);
            Tensor gB = make_tensor(make_gmem_ptr(trans[2](opid, mu).data()), b_gmem_layout);
            cooperative_copy<kConvThreads, sizeof_bits_v<T>>(static_cast<uint32_t>(thread_id()), gB, sBstage);
            cp_async_fence();
            cp_async_wait<0>();
            SYNCTHREADS();
            clear(tCrC_tmp);
            cooperative_gemm(static_cast<uint32_t>(thread_id()), tiled_mma, sA, sBstage, tCrC_tmp);

            CUTE_UNROLL
            for (int i = 0; i < size(tCrC_tmp); ++i) {
              tCrC_persist(i) += mufac * tCrC_tmp(i);
            }
            SYNCTHREADS(); // bstage/buf_t2 about to be reused by the next mu
          }
        }
      }

      // Done with all mu: store the accumulated result back to global
      // memory (mirrors convolution_cuda_wmma.h's final store_c loop).
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC_persist); ++i) {
        tCgC(i) = tCrC_persist(i);
      }
      SYNCTHREADS();
    }

    /**
     * Same contract as convolution_cuda_wmma.h's apply_conv: dispatches the
     * R-term (f/result, internal tile size 2K) and S-term (f0/resultc,
     * internal tile size K) through apply_conv_k above. Identical structure
     * to the WMMA version so convolution.h's call site (mra::accel::
     * apply_conv<T,NDIM>(...)) does not need to change.
     */
    template<typename T, Dimension NDIM,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTransr,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTranss,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewF0,
             concepts::TensorView<NDIM> ViewResultc,
             concepts::TensorView<NDIM> ViewResult>
    DEVSCOPE bool apply_conv(
      int opid,
      auto K,
      const T optol,
      const ViewTransr& transr,
      const ViewTranss& transs,
      const ViewOpnorms& opnorms,
      const std::array<bool, 2>& at,
      ViewF& f,
      ViewF0& f0,
      ViewResultc& resultc,
      ViewResult& result)
    {
      if constexpr (mra::is_ct_integral_v<decltype(K)>) {
        assert(block_size() % MRA_WARP_SIZE == 0 &&
               block_size() / MRA_WARP_SIZE >= apply_conv_min_warps);
        const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank);

        static_assert(NDIM == 3, "Convolution CUTLASS backend only supports NDIM=3");
        static_assert(decltype(K)::value == 8, "Convolution CUTLASS backend only supports K=8");

        if (at[0]) {
          apply_conv_k<T, NDIM, NormId::Rnorm>(
                      opid, mra::Int<2>{}*K, rank, optol, transr, opnorms, f, result);
        } else {
          result = 0.0;
        }
        if (at[1]) {
          apply_conv_k<T, NDIM, NormId::Snorm>(
                      opid, K, rank, optol, transs, opnorms, f0, resultc);
        } else {
          resultc = 0.0;
        }
        return true;
      }
      return false;
    }

#else // !MRA_HAVE_CUTLASS_MMA

    template<typename T, Dimension NDIM,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTransr,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTranss,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewF0,
             concepts::TensorView<NDIM> ViewResultc,
             concepts::TensorView<NDIM> ViewResult>
    DEVSCOPE bool apply_conv(
      int opid,
      auto K,
      const T optol,
      const ViewTransr& transr,
      const ViewTranss& transs,
      const ViewOpnorms& opnorms,
      const std::array<bool, 2>& at,
      ViewF& f,
      ViewF0& f0,
      ViewResultc& resultc,
      ViewResult& result)
    {
      return false;
    }

#endif // MRA_HAVE_CUTLASS_MMA

  } // namespace accel

} // namespace mra

#undef MRA_HAVE_CUTLASS_MMA
#undef MRA_HAVE_CUTLASS_HEADERS

#endif // MRA_KERNELS_IMPL_CONVOLUTION_CUDA_CUTLASS_H
