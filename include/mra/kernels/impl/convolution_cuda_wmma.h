#ifndef MRA_KERNELS_IMPL_CONVOLUTION_CUDA_WMMA_H
#define MRA_KERNELS_IMPL_CONVOLUTION_CUDA_WMMA_H

#include "mra/misc/stacked_allocator.h"
#include "mra/misc/types.h"
#include "mra/tensor/tensorview.h"


/* Device-side availability: only true while compiling for sm_80 or newer. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#  define MRA_HAVE_MMA 1
#  include <mma.h>
#endif



namespace mra {
  namespace accel {

    template<typename T>
    struct mma_traits;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    /* FP64 tensor-core tile geometry (the only shape NVIDIA offers). */
    template<>
    struct mma_traits<double> {
      using T = double;
      static constexpr int WarpSize = MRA_WARP_SIZE;
      static constexpr int M = 8;
      static constexpr int N = 8;
      static constexpr int K = 4;
      static constexpr int NumWarps = MAX_THREADS_PER_BLOCK/WarpSize;

      using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                          M, N, K,
                                          double, nvcuda::wmma::col_major>;
      using FragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                          M, N, K,
                                          double, nvcuda::wmma::row_major>;
      using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                          M, N, K,
                                          double>;

      /** Load one 8x4 tile of A^T starting at row `row`, contraction offset `k`. */
      __device__ __forceinline__
      static void load_a(FragA& frag, const double* a, int k, int row, int ldm) {
        nvcuda::wmma::load_matrix_sync(frag, a + (size_t)k * ldm + row, ldm);
      }

      /** Load one 4x8 tile of B starting at contraction offset `k`, column `col`. */
      __device__ __forceinline__
      static void load_b(FragB& frag, const double* b, int k, int col, int ldm) {
        nvcuda::wmma::load_matrix_sync(frag, b + (size_t)k * ldm + col, ldm);
      }

      /** Store one 8x8 accumulator tile to row-major C at [row][col]. */
      __device__ __forceinline__
      static void store_c(double* c, const FragC& frag, int row, int col, int ldm) {
        nvcuda::wmma::store_matrix_sync(c + (size_t)row * ldm + col, frag, ldm,
                                        nvcuda::wmma::mem_row_major);
      }
    };
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

#ifdef MRA_HAVE_MMA
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
      ViewResult& result,  // size K, stores the sum
      mra::BlockStackAllocator& smem_allocator
      )
    {
      using mma = mma_traits<T>;
      using FragA = typename mma::FragA;
      using FragB = typename mma::FragB;
      using FragC = typename mma::FragC;
      constexpr int K2            = K * K;
      constexpr int K3            = K2 * K;

      // A is MxK, B is KxN, C is MxN
      constexpr int M             = K2;
      constexpr int N             = K;

      // we distribute among warps along the M dimension, not along K or N
      constexpr int M_TILES            = (M + mma::M - 1) / mma::M;
      constexpr int ROWS_PER_WARP      = (M_TILES + mma::NumWarps-1) / mma::NumWarps * mma::M;
      constexpr int M_WARP_TILES       = ROWS_PER_WARP / mma::M;
      constexpr int N_WARP_TILES       = N / mma::N;
      constexpr int K_WARP_TILES       = K / mma::K;

      /**
       * MADNESS uses slightlight different thresholds for R and S terms.
       */
      constexpr auto thresh = [&](){
        if (Term == NormId::Rnorm) return 1.e-20; else return 0.0;
      };

      const int warp_id         = thread_id() / mma::WarpSize;
      const int warp_row_offset = warp_id * ROWS_PER_WARP;

      const bool has_work = (warp_row_offset < K2);

      /**
       * Allocate shared memory for the intermediates results to rotate.
       */
      //auto c_smem = smem_allocator.template alloc<T>(K3);
      extern __shared__ T c_smem[];


      /**
       * Pre-load the warp's A fragments (i.e., f).
       * The fragments will remain in registers for the entire convolution,
       * and will be reused for every mu.
       */
      FragA f_frags[M_WARP_TILES][K_WARP_TILES];
      /**
       * We accumulate the entire result in registers, and only write back
       * to memory at the end.
       */
      FragC c_frags[M_WARP_TILES][N_WARP_TILES];
      if (has_work) {
        #pragma unroll
        for (int i = 0; i < M_WARP_TILES; ++i) {
          #pragma unroll
          for (int j = 0; j < K_WARP_TILES; ++j) {
            mma::load_a(f_frags[i][j], f.data(),
                        j * mma::K,                            /* contraction offset */
                        warp_row_offset + i * mma::M,          /* row in A^T         */
                        K2);                                   /* col-major ldm      */
          }
          // zero out result fragments
          for (int j = 0; j < N_WARP_TILES; ++j) {
            nvcuda::wmma::fill_fragment(c_frags[i][j], 0.0);
          }
        }
      }


      for (size_type mu = 0; mu < rank; ++mu) {
        T munorm = opnorms(opid, mu, 0, (size_type)NormId::MUnorm);
        double dnorm = 1.0;
        for (Dimension d=0; d<NDIM; ++d) dnorm *= opnorms(opid, mu, d, (size_type)Term);
        if (munorm > optol && dnorm > thresh()) {
          T mufac = opnorms(opid, mu, 0, (size_type)NormId::Fac);
          if (NormId::Snorm == Term) mufac *= -1.0; // sign flip for Snorm
          FragA a_frags[M_WARP_TILES][K_WARP_TILES];
          // fill the a_frags from the loaded f_frags
          if (has_work) {
            #pragma unroll
            for (int i = 0; i < M_WARP_TILES; ++i) {
              #pragma unroll
              for (int k = 0; k < K_WARP_TILES; ++k) {
                #pragma unroll
                for(int e=0; e<f_frags[i][k].num_elements; e++)
                  a_frags[i][k].x[e] = f_frags[i][k].x[e];
              }
            }
          }
          // iterate over dimensions
          #pragma unroll
          for (int d = 0; d < NDIM; ++d) {
            /**
             * Load the mu B fragments for this dimension. All warps hold the full set of B fragments for the current mu,
             * because each warp will need to multiply by all of them.
             * The fragments will remain in registers for the entire convolution, and will be reused.
             * TODO: preload the next B fragment into SMEM using copy_async().
             */
            FragB b_frags[K_WARP_TILES][N_WARP_TILES];
            if (has_work) {
              #pragma unroll
              for (int i = 0; i < K_WARP_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < N_WARP_TILES; ++j) {
                  mma::load_b(b_frags[j][i], trans[d](opid, mu).data(),
                              i * mma::K,                            /* contraction offset */
                              j * mma::N,                            /* col in B           */
                              K);                                    /* row-major ldm      */
                }
              }
            }

            // sync to make sure that all warps have read the previous result from SMEM
            SYNCTHREADS();
            /* --- Accumulate and store ---------------------------------------------- */
            if (has_work) {
              #pragma unroll
              for (int i = 0; i < M_WARP_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < N_WARP_TILES; ++j) {
                  FragC acc;
                  nvcuda::wmma::fill_fragment(acc, 0.0);
                  #pragma unroll
                  for (int k = 0; k < K_WARP_TILES; ++k) {
                    nvcuda::wmma::mma_sync(acc, a_frags[i][k], b_frags[k][j], acc);
                  }
                  if (d < NDIM-1) {
                    // store back to SMEM for the next dimension's convolution
                    mma::store_c(c_smem, acc,
                                warp_row_offset + i * mma::M,   /* row in C   */
                                j * mma::N,                    /* col in C   */
                                K);                             /* row-major ldm */
                  } else {
                    // last dimension, scale and accumulate into result fragment
                    #pragma unroll
                    for(int e = 0; e < acc.num_elements; e++)
                      c_frags[i][j].x[e] += mufac * acc.x[e];
                  }
                }
              }
            }
            // wait for all warps to finish writing to SMEM before the next dimension's convolution
            SYNCTHREADS();
            if (has_work && d < NDIM-1) {
              // rotate the result in SMEM for the next dimension's convolution
              #pragma unroll
              for (int i = 0; i < M_WARP_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < K_WARP_TILES; ++j) {
                  mma::load_a(a_frags[i][j], c_smem,
                              j * mma::K,                            /* contraction offset */
                              warp_row_offset + i * mma::M,          /* row in A^T         */
                              K2);                                   /* col-major ldm      */
                }
              }
            }
          }
        }
      }

      // we're done with all mu, store the result back to global memory
      // TODO: stage through SMEM and use copy_async() to pipeline
      #pragma unroll
      for (int i = 0; i < M_WARP_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < N_WARP_TILES; ++j) {
          // store to global memory
          mma::store_c(result.data(), c_frags[i][j],
                      warp_row_offset + i * mma::M,   /* row in C   */
                      j * mma::N,                    /* col in C   */
                      K);                             /* row-major ldm */
        }
      }
      SYNCTHREADS();
    }
#endif // MRA_HAVE_MMA


    /**
     * Implementation of convolution using CUDA WMMA.
     * Currently only supports NDIM=3 and K=8.
     * Works only if K is a compile-time constant, because we need to compile-time configure
     * the wmma fragments.
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
      ViewResult& result,  // size K, stores the sum
      mra::BlockStackAllocator& smem_allocator)
    {
#ifdef MRA_HAVE_MMA
      if constexpr (mra::is_ct_integral_v<decltype(K)>) {
        // get the rank of the operation
        const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank); // doing computation assuming full rank

        static_assert(NDIM == 3, "Convolution CUDA WMMA only supports NDIM=3");
        static_assert(K == 8, "Convolution CUDA WMMA only supports K=8 or K=16");
        // convolution for f
        if (at[0]) {
          apply_conv_k<T, NDIM, NormId::Rnorm>(
                      opid, mra::Int<2>{}*K, rank, optol, transr, opnorms,
                      f, result, smem_allocator);
        } else {
          result = 0.0;
        }
        // convolution for f0
        if (at[1]) {
          apply_conv_k<T, NDIM, NormId::Snorm>(
                      opid, K, rank, optol, transs, opnorms,
                      f0, resultc, smem_allocator);
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
     * convolution kernel using CUDA WMMA.
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

#endif // MRA_KERNELS_IMPL_CONVOLUTION_CUDA_WMMA_H