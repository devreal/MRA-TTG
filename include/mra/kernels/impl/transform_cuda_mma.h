#ifndef MRA_KERNELS_IMPL_TRANSFORM_CUDA_MMA_H
#define MRA_KERNELS_IMPL_TRANSFORM_CUDA_MMA_H

/**
 * 3D transform on FP64 tensor cores (sm_80+) using mma.sync PTX directly:
 *
 *   result[z'][y'][x'] = sum_{z,y,x} f[z][y][x] c[z][z'] c[y][y'] c[x][x']
 *
 * for a D x D x D tensor f and a D x D matrix c (stored [in][out]), i.e.
 * mra::transform for NDIM == 3. This is the single-term case of the
 * convolution core in convolution_cuda_mma.h (apply_conv_k), relaxed so that
 * it runs with any number of whole warps and any D <= 20:
 *
 *  - Stages 1+2 (contract x, then y): warp w handles the z-planes
 *    w, w+nw, ... one at a time; the data of a plane is the B operand of both
 *    contractions and the stage-1 output is the stage-2 B operand as-is (see
 *    apply_conv_k). The stage-2 output goes to the exchange buffer W[n][z]
 *    (n = y'*D + x').
 *  - Stage 3 (contract z): warp w handles the 8-wide column tiles of W
 *    w, w+nw, ... and stores its result tile straight to global memory.
 *
 * D need not be a multiple of the 8x8x4 tile: the fragments are zero-padded,
 * the contraction to a multiple of 4 and the output to a multiple of 8.
 * The kernel is templated on the number of 4-deep contraction steps
 * KS = ceil(D/4) only, D itself is a runtime value.
 *
 * W lives in dynamic shared memory if the launch provides at least
 * transform_mma_shmem_size(D) bytes of it (rows padded to 4*KS+1 and rotated
 * to spread the banks), else in the caller's workspace (D^3 elements, which
 * mra::transform always provides). The matrix fragments are read straight
 * from c, which stays in L1. f may alias result: f is read completely before
 * a block-wide barrier that precedes the first store to result.
 *
 * Define MRA_DISABLE_TRANSFORM_DMMA to fall back to the generic mTxm path.
 */

#include <cstdint>
#include <type_traits>

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

namespace mra {
  namespace accel {
  namespace mma {

    /** largest D (= 2*MRA_MAX_K) the tensor-core transform handles */
    inline constexpr size_type transform_mma_max_dim = 20;

    /** contraction steps of 4 for dimension D */
    SCOPE constexpr size_type transform_mma_ks(size_type D) { return (D + 3) / 4; }

    /** row stride (in elements) of W in shared memory */
    SCOPE constexpr size_type transform_mma_smem_stride(size_type D) { return 4 * transform_mma_ks(D) + 1; }

    /**
     * Shared memory (elements of double) of the W buffer one warp needs for
     * transform_warp with dimension D.
     */
    SCOPE constexpr size_type transform_warp_w_size(size_type D) { return D * D * transform_mma_smem_stride(D); }

    /**
     * What transform_warp does with the transformed tensor v before it lands
     * in result: Scale stores alpha*v, Mul multiplies result by alpha*v, and
     * Square stores (alpha*v)^2.
     */
    enum class TransformEpilogue { Scale, Mul, Square };

    /**
     * Dynamic shared memory (bytes) for transform_warp of D^3 tensors of T by
     * nwarps warps of a block at once (each with its own W); 0 if there is no
     * tensor-core path for T, NDIM, and D.
     */
    template<typename T, Dimension NDIM>
    SCOPE constexpr size_type transform_warp_shmem_size(size_type D, size_type nwarps) {
#if defined(MRA_ENABLE_CUDA) && defined(MRA_CUDA_ARCH) && (MRA_CUDA_ARCH >= 80) && !defined(MRA_DISABLE_TRANSFORM_DMMA)
      if constexpr (std::is_same_v<T, double> && NDIM == 3) {
        if (D > 0 && D <= transform_mma_max_dim) {
          return nwarps * transform_warp_w_size(D) * sizeof(T);
        }
      }
#endif
      return 0;
    }

    /**
     * Dynamic shared memory (bytes) for the tensor-core transform of a D^3
     * tensor of T to keep W in shared memory; 0 if there is no tensor-core path
     * for T, NDIM, and D, in which case the launch needs none.
     */
    template<typename T, Dimension NDIM>
    SCOPE constexpr size_type transform_mma_shmem_size(size_type D) {
#if defined(MRA_ENABLE_CUDA) && defined(MRA_CUDA_ARCH) && (MRA_CUDA_ARCH >= 80) && !defined(MRA_DISABLE_TRANSFORM_DMMA)
      if constexpr (std::is_same_v<T, double> && NDIM == 3) {
        if (D > 0 && D <= transform_mma_max_dim) {
          return D * D * transform_mma_smem_stride(D) * sizeof(T);
        }
      }
#endif
      return 0;
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(MRA_DISABLE_TRANSFORM_DMMA)
#define MRA_HAVE_TRANSFORM_DMMA 1

    namespace detail {

      /** D += A*B for one 8x8x4 FP64 tile, see dmma_8x8x4 in convolution_cuda_mma.h */
      __device__ __forceinline__
      void tdmma_8x8x4(double (&d)[2], double a, double b) {
        asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
            : "+d"(d[0]), "+d"(d[1]) : "d"(a), "d"(b));
      }

      __device__ __forceinline__
      unsigned dynamic_smem_bytes() {
        unsigned r;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(r));
        return r;
      }

      /**
       * Rotation of the rows of W in shared memory, chosen per KS to minimize
       * bank conflicts of the stage-2 stores and the stage-3 loads.
       */
      template<int KS>
      inline constexpr int transform_mma_rot = (KS == 5) ? 7 : (KS >= 3) ? 1 : 0;

      /**
       * See the comment at the top of the file. Fragment layout (g = lane/4,
       * t = lane%4): lane holds A[g][t], B[t][g], and D[g][2t], D[g][2t+1].
       * Called by all threads of the block, or (WarpScope) by one warp alone.
       * f[z][y][x] is at f[z*ldz + y*ldy + x]. Epi: 0 stores the result as is,
       * otherwise the TransformEpilogue Epi-1 applies with alpha.
       */
      template<int KS, bool WInSmem, bool WarpScope = false, int Epi = 0>
      __device__ __noinline__
      void transform_mma_ks(const int D,
                            const double* f,
                            const int ldy,
                            const int ldz,
                            const double* __restrict__ c,
                            double* result,
                            double* W,
                            const double alpha = 1.0)
      {
        constexpr int MT  = (KS + 1) / 2;  // 8-wide output tiles per dimension
        constexpr int SW  = 4 * KS + 1;    // W row stride in shared memory
        constexpr int ROT = transform_mma_rot<KS>;
        const int NN = D * D;
        const int NT = (NN + 7) / 8;       // stage-3 column tiles

        /* WarpScope: the calling warp does the whole transform on its own */
        const int tid    = WarpScope ? thread_id() % MRA_WARP_SIZE : thread_id();
        const int warp   = tid / MRA_WARP_SIZE;
        const int lane   = tid % MRA_WARP_SIZE;
        const int nwarps = WarpScope ? 1 : block_size() / MRA_WARP_SIZE; // a partial last warp only joins the barriers
        const int g      = lane / 4;
        const int t      = lane % 4;
        const bool active = warp < nwarps;

        /* element (n, z) of W */
        auto widx = [&](int n, int z) {
          if constexpr (WInSmem) {
            int o = z + (ROT * n) % SW;
            if (o >= SW) o -= SW;
            return n * SW + o;
          } else {
            return n * D + z;
          }
        };

        /* ta[mt][ks] = c[KS*t+ks][g+8mt]: A fragments of stages 1 and 3 (Tx and Tz) */
        double ta[MT][KS];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          #pragma unroll
          for (int ks = 0; ks < KS; ++ks) {
            const int i = KS * t + ks, o = g + 8 * mt;
            ta[mt][ks] = (i < D && o < D) ? c[i * D + o] : 0.0;
          }
        }

        /* ty[mt2][nt][r] = c[2t+r+8nt][g+8mt2]: A fragments of stage 2 (Ty),
         * kept in registers unless that costs too many (then loaded per use, from L1) */
        constexpr bool HoistTy = (KS <= 4);
        auto load_ty = [&](int mt2, int nt, int r) {
          const int y = 2 * t + r + 8 * nt, yp = g + 8 * mt2;
          return (y < D && yp < D) ? c[y * D + yp] : 0.0;
        };
        double ty[HoistTy ? MT : 1][HoistTy ? MT : 1][2];
        if constexpr (HoistTy) {
          #pragma unroll
          for (int mt2 = 0; mt2 < MT; ++mt2) {
            #pragma unroll
            for (int nt = 0; nt < MT; ++nt) {
              ty[mt2][nt][0] = load_ty(mt2, nt, 0);
              ty[mt2][nt][1] = load_ty(mt2, nt, 1);
            }
          }
        }

        /* fr[nt][ks] = f[z][y = g+8nt][x = KS*t+ks]: stage-1 B fragments of plane z */
        auto load_plane = [&](double (&fr)[MT][KS], int z) {
          const double* fz = f + z * ldz;
          #pragma unroll
          for (int nt = 0; nt < MT; ++nt) {
            #pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
              const int y = g + 8 * nt, x = KS * t + ks;
              fr[nt][ks] = (y < D && x < D) ? fz[y * ldy + x] : 0.0;
            }
          }
        };

        /* f may have just been written by other threads, and W (in shared
         * memory) may still be in use by the caller */
        if constexpr (WarpScope) __syncwarp(); else SYNCTHREADS();

        if (active) {
          /* (prefetching the next plane before computing this one costs more
           * in register spills than it gains in latency) */
          for (int z = warp; z < D; z += nwarps) {
            double fr[MT][KS];
            load_plane(fr, z);

            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              /* Stage 1: yv[nt][r] = Y[x' = g+8mt][y = 2t+r+8nt] */
              double yv[MT][2];
              #pragma unroll
              for (int nt = 0; nt < MT; ++nt) {
                yv[nt][0] = yv[nt][1] = 0.0;
                #pragma unroll
                for (int ks = 0; ks < KS; ++ks) {
                  tdmma_8x8x4(yv[nt], ta[mt][ks], fr[nt][ks]);
                }
              }

              /* Stage 2: zz[r] = Z[y' = g+8mt2][x' = 2t+r+8mt] */
              #pragma unroll
              for (int mt2 = 0; mt2 < MT; ++mt2) {
                const int yp = g + 8 * mt2;
                double zz[2] = {0.0, 0.0};
                #pragma unroll
                for (int nt = 0; nt < MT; ++nt) {
                  #pragma unroll
                  for (int r = 0; r < 2; ++r) {
                    if constexpr (HoistTy) tdmma_8x8x4(zz, ty[mt2][nt][r], yv[nt][r]);
                    else                   tdmma_8x8x4(zz, load_ty(mt2, nt, r), yv[nt][r]);
                  }
                }
                if (yp < D) {
                  #pragma unroll
                  for (int r = 0; r < 2; ++r) {
                    const int xp = 2 * t + r + 8 * mt;
                    if (xp < D) W[widx(yp * D + xp, z)] = zz[r];
                  }
                }
              }
            }
          }
        }

        /* W is complete, and every warp is done reading f */
        if constexpr (WarpScope) __syncwarp(); else SYNCTHREADS();

        if (active) {
          /* the pairs (n, n+1) stored below are both in range or both out of it if D is even */
          const bool vec_store = (D % 2 == 0) && ((reinterpret_cast<std::uintptr_t>(result) % 16) == 0);
          for (int tile = warp; tile < NT; tile += nwarps) {
            /* Stage-3 B fragments: b[ks] = W[n = g+8*tile][z = KS*t+ks] */
            const int n = g + 8 * tile;
            double b[KS];
            #pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
              const int z = KS * t + ks;
              b[ks] = (n < NN && z < D) ? W[widx(n, z)] : 0.0;
            }
            /* Out[z' = g+8mt][n = 2t+r+8*tile] */
            const int nc = 2 * t + 8 * tile;
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              double acc[2] = {0.0, 0.0};
              #pragma unroll
              for (int ks = 0; ks < KS; ++ks) {
                tdmma_8x8x4(acc, ta[mt][ks], b[ks]);
              }
              const int zp = g + 8 * mt;
              if (zp < D) {
                double* out = result + zp * NN + nc;
                /* the elements (out[0], out[1]) of a Mul were stored by this lane itself */
                auto epi = [&](double v, double old) {
                  if constexpr (Epi == 0) return v;
                  else if constexpr (Epi == 1 + (int)TransformEpilogue::Scale) return alpha * v;
                  else if constexpr (Epi == 1 + (int)TransformEpilogue::Mul) return old * (alpha * v);
                  else { const double av = alpha * v; return av * av; }
                };
                constexpr bool reads = (Epi == 1 + (int)TransformEpilogue::Mul);
                if (vec_store) {
                  if (nc < NN) {
                    double2 o = reads ? *reinterpret_cast<const double2*>(out) : make_double2(0.0, 0.0);
                    *reinterpret_cast<double2*>(out) = make_double2(epi(acc[0], o.x), epi(acc[1], o.y));
                  }
                } else {
                  if (nc < NN)     out[0] = epi(acc[0], reads ? out[0] : 0.0);
                  if (nc + 1 < NN) out[1] = epi(acc[1], reads ? out[1] : 0.0);
                }
              }
            }
          }
        }

        /* result is complete, and W may be reused */
        if constexpr (WarpScope) __syncwarp(); else SYNCTHREADS();
      }

      template<int KS>
      __device__ __forceinline__
      void transform_mma_dispatch_w(int D, const double* f, const double* c, double* result, double* workspace) {
        extern __shared__ __align__(16) double transform_mma_smem[];
        const unsigned need = D * D * (4 * KS + 1) * sizeof(double);
        if (dynamic_smem_bytes() >= need) {
          transform_mma_ks<KS, true>(D, f, D, D * D, c, result, transform_mma_smem);
        } else {
          transform_mma_ks<KS, false>(D, f, D, D * D, c, result, workspace);
        }
      }

    } // namespace detail

    /**
     * Whether transform_warp can run for dimension D with every whole warp of
     * the block at once, i.e. the launch provided transform_warp_shmem_size(D,
     * block_size()/32) bytes of dynamic shared memory. Uniform across the block.
     */
    template<typename T, Dimension NDIM>
    __device__ bool transform_warp_available(size_type D) {
      if constexpr (std::is_same_v<T, double> && NDIM == 3) {
        const size_type nwarps = block_size() / MRA_WARP_SIZE;
        return D > 0 && D <= transform_mma_max_dim && nwarps > 0 &&
               detail::dynamic_smem_bytes() >= nwarps * transform_warp_w_size(D) * sizeof(T);
      }
      return false;
    }

    /**
     * The shared-memory W buffer of the calling warp for transform_warp, if
     * transform_warp_available(D).
     */
    template<typename T>
    __device__ T* transform_warp_buffer(size_type D) {
      extern __shared__ __align__(16) double transform_mma_smem[];
      return reinterpret_cast<T*>(transform_mma_smem) + (thread_id() / MRA_WARP_SIZE) * transform_warp_w_size(D);
    }

    /**
     * As transform below, but executed by the calling warp alone (all 32 lanes),
     * so that the warps of a block can transform different tensors at once,
     * with E applied to the result (see TransformEpilogue). f[z][y][x] is read
     * from f[z*ldz + y*ldy + x], e.g. the child slice of a larger cube; result
     * is dense. W is the warp's own transform_warp_w_size(D) elements of shared
     * memory, see transform_warp_buffer. Synchronizes the warp only. Requires
     * transform_warp_available(D).
     */
    template<typename T, Dimension NDIM, TransformEpilogue E = TransformEpilogue::Scale>
    __device__ void transform_warp(size_type D, const T* f, size_type ldy, size_type ldz,
                                   const T* c, T* result, T alpha, T* W) {
      static_assert(std::is_same_v<T, double> && NDIM == 3, "transform_warp: double, NDIM=3 only");
      constexpr int Epi = 1 + (int)E;
      switch (transform_mma_ks(D)) {
        case 1: detail::transform_mma_ks<1, true, true, Epi>(D, f, ldy, ldz, c, result, W, alpha); break;
        case 2: detail::transform_mma_ks<2, true, true, Epi>(D, f, ldy, ldz, c, result, W, alpha); break;
        case 3: detail::transform_mma_ks<3, true, true, Epi>(D, f, ldy, ldz, c, result, W, alpha); break;
        case 4: detail::transform_mma_ks<4, true, true, Epi>(D, f, ldy, ldz, c, result, W, alpha); break;
        case 5: detail::transform_mma_ks<5, true, true, Epi>(D, f, ldy, ldz, c, result, W, alpha); break;
        default: break;
      }
    }

    /**
     * The tensor-core transform of the D^3 tensor f by the D x D matrix c into
     * result (see the top of this file), called by all threads of the block.
     * workspace (D^3 elements) is used if the launch did not provide
     * transform_mma_shmem_size(D) bytes of dynamic shared memory. Returns false,
     * without touching anything, if there is no tensor-core path for T, NDIM,
     * D, or the block shape (which needs at least one whole warp).
     */
    template<typename T, Dimension NDIM>
    __device__ bool transform(size_type D, const T* f, const T* c, T* result, T* workspace) {
      if constexpr (std::is_same_v<T, double> && NDIM == 3) {
        if (D == 0 || D > transform_mma_max_dim || block_size() < MRA_WARP_SIZE) return false;
        switch (transform_mma_ks(D)) {
          case 1: detail::transform_mma_dispatch_w<1>(D, f, c, result, workspace); return true;
          case 2: detail::transform_mma_dispatch_w<2>(D, f, c, result, workspace); return true;
          case 3: detail::transform_mma_dispatch_w<3>(D, f, c, result, workspace); return true;
          case 4: detail::transform_mma_dispatch_w<4>(D, f, c, result, workspace); return true;
          case 5: detail::transform_mma_dispatch_w<5>(D, f, c, result, workspace); return true;
          default: return false;
        }
      }
      return false;
    }

#endif // __CUDA_ARCH__ >= 800

  } // namespace mma
  } // namespace accel
} // namespace mra

#endif // MRA_KERNELS_IMPL_TRANSFORM_CUDA_MMA_H
