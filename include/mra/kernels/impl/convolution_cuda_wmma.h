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
      constexpr int K2            = K * K;
      // need at least 1 fragment per warp
      constexpr int ROWS_PER_WARP = ((K2 / mma::NumWarps) + mma::M - 1) / mma::M * mma::M;
      constexpr int ROW_TILES     = ROWS_PER_WARP / mma::M;
      constexpr int NSTEPS        = K / mma::K;
      constexpr int COL_TILES     = K / mma::K;
      constexpr auto thresh = [&](){
        if (Term == NormId::Rnorm) return 1.e-20; else return 0.0;
      };

      const int warp_id         = thread_id() / mra::WarpSize;
      const int warp_row_offset = warp_id * ROWS_PER_WARP;

      const bool has_work = (warp_row_offset < K2);

      /**
       * Allocate shared memory for the intermediates results to rotate.
       */
      auto c_smem = smem_allocator.template alloc<T>(K2);


      /**
       * Pre-load the warp's A fragments (i.e., f).
       * The fragments will remain in registers for the entire convolution, and will be reused.
       */
      mma::FragA f_frags[ROW_TILES][NSTEPS];
      // we accumulate the result in registers, and only write back to memory at the end
      mma::FragC c_frags[ROW_TILES][COL_TILES];
      if (has_work) {
        #pragma unroll
        for (int i = 0; i < ROW_TILES; ++i) {
          #pragma unroll
          for (int j = 0; j < NSTEPS; ++j) {
            mma::load_a(f_frags[i][j], f.data(),
                        j * mma::K,                            /* contraction offset */
                        warp_row_offset + i * mma::M,          /* row in A^T         */
                        K2);                                   /* col-major ldm      */
            nvcuda::wmma::fill_fragment(c_frags[i][j], 0.0);
          }
        }
      }


      for (size_type mu = 0; mu < rank; ++mu) {
        T munorm = opnorms(opid, mu, 0, (size_type)NormId::MUnorm);
        double dnorm = 1.0;
        for (Dimension d=0; d<NDIM; ++d) dnorm *= opnorms(opid, mu, d, (size_type)normid);
        if (munorm > optol && dnorm > thresh()) {
          T mufac = opnorms(opid, mu, 0, (size_type)NormId::Fac);
          if (NormId::Snorm == normid) mufac *= -1.0; // sign flip for Snorm
          mma::FragA a_frags[ROW_TILES][NSTEPS];
          // fill the a_frags from the loaded f_frags
          if (has_work) {
            #pragma unroll
            for (int i = 0; i < ROW_TILES; ++i) {
              #pragma unroll
              for (int j = 0; j < NSTEPS; ++j) {
                #pragma unroll
                for(int e=0; e<f_frags[i][j].num_elements; e++)
                  a_frags[i][j].x[e] = f_frags[i][j].x[e];
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
            mma::FragB b_frags[NSTEPS][COL_TILES];
            if (has_work) {
              #pragma unroll
              for (int i = 0; i < COL_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < NSTEPS; ++j) {
                  mma::load_b(b_frags[j][i], trans[d](opid, mu).data(),
                              j * mma::K,                            /* contraction offset */
                              i * mma::K,                            /* col in B           */
                              K);                                    /* row-major ldm      */
                }
              }
            }

            // sync to make sure that all warps have read the previous result from SMEM
            SYNCTHREADS();
            /* --- Accumulate and store ---------------------------------------------- */
            if (has_work) {
              #pragma unroll
              for (int t = 0; t < ROW_TILES; ++t) {
                #pragma unroll
                for (int ct = 0; ct < COL_TILES; ++ct) {
                  mma::FragC acc;
                  nvcuda::wmma::fill_fragment(acc, 0.0);
                  #pragma unroll
                  for (int s = 0; s < NSTEPS; ++s) {
                    nvcuda::wmma::mma_sync(acc, a_frags[t][s], b_frags[s][ct], acc);
                  }
                  if (d < NDIM-1) {
                    // store back to SMEM for the next dimension's convolution
                    mma::store_c(c_smem.get(), acc,
                                warp_row_offset + t * mma::M,   /* row in C   */
                                ct * mma::N,                    /* col in C   */
                                K);                             /* row-major ldm */
                  } else {
                    // last dimension, scale and accumulate into result fragment
                    #pragma unroll
                    for(int e = 0; e < acc.num_elements; e++)
                      c_frags[t][ct].x[e] += mufac * acc.x[e];
                  }
                }
              }
            }
            // wait for all warps to finish writing to SMEM before the next dimension's convolution
            SYNCTHREADS();
            if (has_work && d < NDIM-1) {
              // rotate the result in SMEM for the next dimension's convolution
              #pragma unroll
              for (int t = 0; t < ROW_TILES; ++t) {
                #pragma unroll
                for (int s = 0; s < NSTEPS; ++s) {
                  mma::load_a(a_frags[t][s], c_smem.get(),
                              s * mma::K,                            /* contraction offset */
                              warp_row_offset + t * mma::M,          /* row in A^T         */
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
      for (int t = 0; t < ROW_TILES; ++t) {
        #pragma unroll
        for (int ct = 0; ct < COL_TILES; ++ct) {
          // store to global memory
          mma::store_c(result.data(), c_frags[t][ct],
                      warp_row_offset + t * mma::M,   /* row in C   */
                      ct * mma::N,                    /* col in C   */
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
                      opid, mra::Int<2>*K, rank, optol, transr, opnorms,
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
#ifdef MRA_HAVE_MMA
      if (K == 8) {
        auto K2 = mra::Int<2>{}*K;
        return K2*K2*K2*sizeof(T); // SMEM for the result matrix to rotate
      }
#endif // MRA_HAVE_MMA
      return 0;
    }

  } // namespace accel

} // namespace mra

#undef MRA_HAVE_MMA

#endif // MRA_KERNELS_IMPL_CONVOLUTION_CUDA_WMMA_H