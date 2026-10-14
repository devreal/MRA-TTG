#ifndef MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H
#define MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H

/**
 * Convolution using FP64 tensor cores through raw PTX mma.sync (m8n8k4).
 *
 * Unlike the WMMA implementation (convolution_cuda_wmma.h, kept as a basis for
 * a rocWMMA port), the fragment layouts are documented, so operands are loaded
 * from and results written to memory directly:
 *  - the intermediate results are rotated through shared memory with plain
 *    shared loads/stores,
 *  - f is (re)loaded into shared memory with cp.async,
 *  - the final result is accumulated in L2 using fire-and-forget reductions
 *    (red.global.add) rather than being held in registers.
 *
 * Provides the same entry points as convolution_cuda_wmma.h; only one of the
 * two can be included.
 */

#include "mra/misc/types.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/conv_mad.h" // for NormId, used as a template parameter below


/* Device-side availability: only true while compiling for sm_80 or newer. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#  define MRA_HAVE_MMA 1
#endif

/* Must be visible in both host and device passes: it declares __device__
 * variables that the host-side registration stub refers to. */
#if defined(__CUDACC__)
#  include <cuda/pipeline>
#endif



namespace mra {
  namespace accel {

    /**
     * apply_conv_k keeps each warp's share of the row tiles in registers, so
     * it is compiled for at least this many warps per block (see
     * MAX_WARP_TILES there); kernels calling apply_conv must be launched with
     * at least apply_conv_min_threads(K) threads (see
     * detail::conv_thread_dims in kernels/convolution.h).
     */
    inline constexpr int apply_conv_min_warps = 8;

    /**
     * Minimum number of threads per block for kernels calling apply_conv
     * with this K: apply_conv_min_warps whole warps if the MMA path is used
     * for K (mirrors the condition in apply_conv_shmem_size), 0 otherwise.
     */
    SCOPE constexpr int apply_conv_min_threads(auto K) {
#ifdef MRA_ENABLE_CUDA
      if (K == 8) {
        // CUDA warps are 32 threads (MRA_WARP_SIZE is only defined in device code)
        return apply_conv_min_warps * 32;
      }
#endif // MRA_ENABLE_CUDA
      return 0;
    }

    template<typename T>
    struct mma_traits;

#ifdef MRA_HAVE_MMA
    /* FP64 tensor-core tile geometry (the only shape NVIDIA offers). */
    template<>
    struct mma_traits<double> {
      using T = double;
      static constexpr int WarpSize = MRA_WARP_SIZE;
      static constexpr int M = 8;
      static constexpr int N = 8;
      static constexpr int K = 4;
      /* Per-warp register capacity (see apply_conv_k) is sized for at least
       * this many warps per block (see apply_conv_min_warps). */
      static constexpr int MinWarps = apply_conv_min_warps;

      /**
       * m8n8k4 FP64 MMA (PTX mma.sync): C += A * B.
       * Per the PTX ISA, with g = lane / 4 and t = lane % 4, each thread holds
       *   A (8x4): a = A[g][t]
       *   B (4x8): b = B[t][g]
       *   C (8x8): c0 = C[g][2*t], c1 = C[g][2*t+1]
       */
      __device__ __forceinline__
      static void mma(double& c0, double& c1, double a, double b) {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0, %1}, {%2}, {%3}, {%0, %1};"
                     : "+d"(c0), "+d"(c1)
                     : "d"(a), "d"(b));
      }

      /** Fire-and-forget atomic add to global memory (RED), performed in L2. */
      __device__ __forceinline__
      static void red_add(double* ptr, double v) {
        asm volatile("red.relaxed.gpu.global.add.f64 [%0], %1;"
                     :: "l"(ptr), "d"(v) : "memory");
      }
    };

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
      ViewResult& result  // size K, stores the sum
      )
    {
      using mma = mma_traits<T>;
      constexpr int K2            = K * K;
      constexpr int K3            = K2 * K;

      // A is MxK, B is KxN, C is MxN
      constexpr int M             = K2;
      constexpr int N             = K;

      // we distribute among warps along the M dimension, not along K or N:
      // each warp owns tiles_per_warp consecutive row tiles, based on the
      // actual number of warps in the block. The A operands for a warp's tiles
      // are held in registers across each dimension, so their count is bounded at compile
      // time by MAX_WARP_TILES -- kernels are launched with at least
      // mma::MinWarps warps (see apply_conv_min_threads), which guarantees
      // tiles_per_warp <= MAX_WARP_TILES.
      constexpr int M_TILES            = (M + mma::M - 1) / mma::M;
      constexpr int MAX_WARP_TILES     = (M_TILES + mma::MinWarps - 1) / mma::MinWarps;
      constexpr int N_WARP_TILES       = N / mma::N;
      constexpr int K_WARP_TILES       = K / mma::K;

      /**
       * MADNESS uses slightlight different thresholds for R and S terms.
       */
      constexpr auto thresh = [&](){
        if (Term == NormId::Rnorm) return 1.e-20; else return 0.0;
      };

      const int num_warps       = block_size() / mma::WarpSize;
      const int warp_id         = thread_id() / mma::WarpSize;
      const int lane            = thread_id() % mma::WarpSize;
      const int lane_g          = lane / 4; // see mma::mma
      const int lane_t          = lane % 4;
      const int tiles_per_warp  = (M_TILES + num_warps - 1) / num_warps;
      const int warp_tile0      = warp_id * tiles_per_warp;
      assert(tiles_per_warp <= MAX_WARP_TILES);
      // whether the i'th of this warp's tiles exists; i is a compile-time
      // index in the unrolled loops below so the operands stay in registers
      auto tile_active = [&](int i) { return i < tiles_per_warp && warp_tile0 + i < M_TILES; };
      const int row0            = warp_tile0 * mma::M; // first row of this warp's tiles

      const bool has_work = tile_active(0);

      /**
       * Allocate shared memory for the intermediates results to rotate.
       * At the start of each mu, c_smem holds a copy of f (same layout as
       * f.data()), from which the first dimension's A operands are loaded.
       *
       * Viewed as A (M x K, col-major, ldm K2) when read and as C (M x N,
       * row-major, ldm K) when written, which rotates the dimensions.
       * The per-lane base pointers below cover the lane's position in the
       * fragments (see mma::mma); the remaining offsets are compile-time
       * constants that fold into the shared memory instructions.
       */
      extern __shared__ __align__(16) T c_smem[];
      T* const a_smem = c_smem + lane_t * K2 + row0 + lane_g;    // A[row0 + g][t]
      T* const c_lane = c_smem + (row0 + lane_g) * K + 2 * lane_t; // C[row0 + g][2t]
      auto load_a = [&](T (&a)[MAX_WARP_TILES][K_WARP_TILES]) {
        #pragma unroll
        for (int i = 0; i < MAX_WARP_TILES; ++i) {
          if (!tile_active(i)) continue;
          #pragma unroll
          for (int k = 0; k < K_WARP_TILES; ++k) {
            a[i][k] = a_smem[k * mma::K * K2 + i * mma::M];
          }
        }
      };

      /**
       * Asynchronously (re)load f from global memory into c_smem.
       * Each thread copies a strided subset of 16B chunks through its own
       * pipeline (f.data() is at least 128B aligned); wait_f() waits for this
       * thread's copies and then syncs the block so that all copies are visible.
       */
      constexpr int CHUNK_BYTES = 16;
      constexpr int CHUNK_ELEMS = CHUNK_BYTES / sizeof(T);
      static_assert(K3 % CHUNK_ELEMS == 0, "f must be a whole number of 16B chunks");
      auto pipe = cuda::make_pipeline();
      auto async_load_f = [&]() {
        pipe.producer_acquire();
        for (int i = thread_id() * CHUNK_ELEMS; i < K3; i += block_size() * CHUNK_ELEMS) {
          cuda::memcpy_async(&c_smem[i], &f.data()[i],
                             cuda::aligned_size_t<CHUNK_BYTES>(CHUNK_BYTES), pipe);
        }
        pipe.producer_commit();
      };
      auto wait_f = [&]() {
        pipe.consumer_wait();
        pipe.consumer_release();
        SYNCTHREADS();
      };

      async_load_f();
      bool f_in_flight = true;

      /**
       * The result is not held in registers: the first active mu stores its
       * contribution to result, later ones add theirs using fire-and-forget
       * reductions (RED) that are performed in L2, so result is never loaded.
       * Each element is owned by a single thread, so the order of additions
       * is fixed and the result is deterministic.
       */
      bool result_init = false;

      for (size_type mu = 0; mu < rank; ++mu) {
        T munorm = opnorms(opid, mu, 0, (size_type)NormId::MUnorm);
        double dnorm = 1.0;
        for (Dimension d=0; d<NDIM; ++d) dnorm *= opnorms(opid, mu, d, (size_type)Term);
        if (munorm > optol && dnorm > thresh()) {
          T mufac = opnorms(opid, mu, 0, (size_type)NormId::Fac);
          if (NormId::Snorm == Term) mufac *= -1.0; // sign flip for Snorm
          // make sure f has fully arrived in c_smem
          if (f_in_flight) {
            wait_f();
            f_in_flight = false;
          }
          T a_frags[MAX_WARP_TILES][K_WARP_TILES];
          // fill the A operands from f in c_smem
          if (has_work) load_a(a_frags);
          // iterate over all but the last dimension, rotating through c_smem
          #pragma unroll
          for (int d = 0; d < NDIM-1; ++d) {
            /**
             * The B operands are loaded when needed rather than held in
             * registers: the K x K matrix is small and stays in L1.
             * NOTE: making the global accesses explicit (__ldg, __stcg) lets
             *       the compiler keep more operands live across tiles, which
             *       causes spills at higher occupancy.
             * TODO: preload the next B matrix into SMEM using copy_async().
             */
            const T* b = trans[d](opid, mu).data() + lane_t * K + lane_g; // B[t][g]

            // sync to make sure that all warps have read the previous result from SMEM
            SYNCTHREADS();
            /* --- Multiply and store ------------------------------------------------ */
            if (has_work) {
              #pragma unroll
              for (int i = 0; i < MAX_WARP_TILES; ++i) {
                if (!tile_active(i)) continue;
                #pragma unroll
                for (int j = 0; j < N_WARP_TILES; ++j) {
                  T c0 = 0.0, c1 = 0.0;
                  #pragma unroll
                  for (int k = 0; k < K_WARP_TILES; ++k) {
                    mma::mma(c0, c1, a_frags[i][k], b[k * mma::K * K + j * mma::N]);
                  }
                  // store back to SMEM for the next dimension's convolution
                  T* c = c_lane + i * mma::M * K + j * mma::N;
                  c[0] = c0;
                  c[1] = c1;
                }
              }
            }
            // wait for all warps to finish writing to SMEM before the next dimension's convolution
            SYNCTHREADS();
            // rotate the result in SMEM for the next dimension's convolution
            // (the last dimension loads its operands itself, see below)
            if (has_work && d < NDIM-2) load_a(a_frags);
          }

          /**
           * Last dimension: add the result directly to result.
           * A is read per tile rather than held in registers, so f is only
           * reloaded into c_smem once all warps are done.
           */
          if (has_work) {
            const T* b = trans[NDIM-1](opid, mu).data() + lane_t * K + lane_g; // B[t][g]
            T* dst = result.data() + (row0 + lane_g) * K + 2 * lane_t;         // C[g][2t]
            /* --- Multiply and accumulate into result ----------------------------------- */
            #pragma unroll
            for (int i = 0; i < MAX_WARP_TILES; ++i) {
              if (!tile_active(i)) continue;
              #pragma unroll
              for (int j = 0; j < N_WARP_TILES; ++j) {
                T c0 = 0.0, c1 = 0.0;
                #pragma unroll
                for (int k = 0; k < K_WARP_TILES; ++k) {
                  mma::mma(c0, c1, a_smem[k * mma::K * K2 + i * mma::M],
                                   b[k * mma::K * K + j * mma::N]);
                }
                c0 *= mufac;
                c1 *= mufac;
                T* r = dst + i * mma::M * K + j * mma::N;
                if (!result_init) {
                  r[0] = c0;
                  r[1] = c1;
                } else {
                  mma::red_add(&r[0], c0);
                  mma::red_add(&r[1], c1);
                }
              }
            }
          }
          if (mu+1 < rank) {
            // make sure that all warps have read the last result from SMEM,
            // then reload f for the next mu
            SYNCTHREADS();
            async_load_f();
            f_in_flight = true;
          }
          result_init = true;
        }
      }

      // drain a pending reload of f (skipped trailing mu) before c_smem is reused
      if (f_in_flight) {
        wait_f();
      }

      // no mu contributed: the result is zero
      if (!result_init) {
        for (int i = thread_id(); i < K3; i += block_size()) {
          result.data()[i] = 0.0;
        }
      }
      // make the result visible to the whole block
      SYNCTHREADS();
    }
#endif // MRA_HAVE_MMA


    /**
     * Implementation of convolution using FP64 tensor cores (PTX mma.sync).
     * Currently only supports NDIM=3 and K=8.
     * Works only if K is a compile-time constant, because the tiling is
     * configured at compile time.
     *
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
      ViewResult& result)  // size K, stores the sum
    {
#ifdef MRA_HAVE_MMA
      if constexpr (mra::is_ct_integral_v<decltype(K)>) {
        // apply_conv_k's per-warp register capacity needs at least MinWarps
        // whole warps -- guaranteed by launching with apply_conv_min_threads(K)
        assert(block_size() % mma_traits<T>::WarpSize == 0 &&
               block_size() / mma_traits<T>::WarpSize >= mma_traits<T>::MinWarps);
        // get the rank of the operation
        const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank); // doing computation assuming full rank

        static_assert(NDIM == 3, "Convolution CUDA MMA only supports NDIM=3");
        static_assert(K == 8, "Convolution CUDA MMA only supports K=8");
        // convolution for f
        if (at[0]) {
          apply_conv_k<T, NDIM, NormId::Rnorm>(
                      opid, mra::Int<2>{}*K, rank, optol, transr, opnorms,
                      f, result);
        } else {
          result = 0.0;
        }
        // convolution for f0
        if (at[1]) {
          apply_conv_k<T, NDIM, NormId::Snorm>(
                      opid, K, rank, optol, transs, opnorms,
                      f0, resultc);
        } else {
          resultc = 0.0;
        }
        return true;
      }
#endif // MRA_HAVE_MMA
      return false;
    }

    /**
     * Returns the number of bytes of shared memory required for the
     * convolution kernel using CUDA MMA.
     *
     * NOTE: This needs to be updated if the kernel is changed to
     *       use more shared memory.
     */
    template<typename T>
    SCOPE constexpr size_type apply_conv_shmem_size(auto K) {
#ifdef MRA_ENABLE_CUDA
      if (K == 8) {
        auto K2 = mra::Int<2>{}*K;
        return K2*K2*K2*sizeof(T); // SMEM for the result matrix to rotate
      }
#endif // MRA_ENABLE_CUDA
      return 0;
    }

  } // namespace accel

} // namespace mra

#undef MRA_HAVE_MMA

#endif // MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H
