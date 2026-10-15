#ifndef MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H
#define MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H

/**
 * Convolution on FP64 tensor cores (sm_80+) using mma.sync PTX directly.
 * Selected by mra/kernels/impl/convolution_accel.h.
 */

#include "mra/misc/types.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/conv_mad.h" // for NormId, used as a template parameter below


/* Device-side availability: only true while compiling for sm_80 or newer
 * (FP64 tensor-core mma). */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#  define MRA_HAVE_DMMA 1
#endif



namespace mra {
  namespace accel {
  namespace mma {

    /**
     * apply_conv_k splits each tensor among exactly this many warps, keeping
     * every warp's share in registers; kernels calling apply_conv must be
     * launched with at least apply_conv_min_threads(K) threads (see
     * detail::conv_thread_dims in kernels/convolution.h). Any further warps
     * only help with the copies.
     */
    inline constexpr int apply_conv_min_warps = 8;

    /**
     * Minimum number of threads per block for kernels calling apply_conv
     * with this K: apply_conv_min_warps whole warps if the tensor-core path
     * is used for K (mirrors the condition in apply_conv_shmem_size), 0 otherwise.
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

#ifdef MRA_HAVE_DMMA
    /**
     * D += A*B for one 8x8x4 FP64 tensor-core tile (mma.m8n8k4, A row-major,
     * B column-major). Fragment layout, with g = lane/4 and t = lane%4:
     * lane holds A[g][t], B[t][g], and D[g][2t], D[g][2t+1].
     */
    __device__ __forceinline__
    void dmma_8x8x4(double (&d)[2], double a, double b) {
      asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
          : "+d"(d[0]), "+d"(d[1]) : "d"(a), "d"(b));
    }

    /**
     * Applies all terms mu of one (R or S) separated operator to the k^3
     * tensor f (k = K, a compile-time 8 or 16) and stores the sum into result:
     *
     *   result[z'][y'][x'] = sum_mu mufac_mu sum_{z,y,x} f[z][y][x] Tz[z][z'] Ty[y][y'] Tx[x][x']
     *
     * with Tz = trans[0], Ty = trans[1], Tx = trans[2] (each stored [in][out]).
     *
     * Work distribution over the first apply_conv_min_warps (NW) warps:
     *  - Stages 1+2 (contract x, then y): warp w owns the z-planes
     *    [w*ZS, (w+1)*ZS) and does both contractions in registers. Both use
     *    the data as the B operand (C = T^T * X): the output's column index
     *    sits on lane%4 (pairs in d[0..1]), which is exactly where the next
     *    B operand expects its contraction index, so the stage-1 output is
     *    the stage-2 input as-is (with the contraction index permuted, which
     *    Ty's A fragments mirror).
     *  - Exchange: the stage-2 output is written to shared memory as
     *    W[n][z] (n = y'*k + x'), with 16-byte slots rotated by (x'+y') to
     *    avoid bank conflicts, and read back by
     *  - Stage 3 (contract z): warp w owns columns n in [w*k*k/NW, (w+1)*k*k/NW)
     *    and accumulates mufac*Tz^T * W into registers across all mu, so the
     *    result is written to global memory once.
     *
     * f stays in registers across all mu. Each term costs one shared-memory
     * round trip and two block-wide barriers. The transform matrices of the
     * next applied term are copied into shared memory with cp.async while
     * the current one is computed (see prefetch), in an order that lets each
     * warp load its A fragments directly.
     *
     * The shared memory (apply_conv_shmem_size) holds W followed by two sets
     * of {Tx, Ty, Tz}.
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
      const ViewF& f,
      ViewResult& result  // size K, stores the sum
      )
    {
      static_assert(std::is_same_v<T, double>, "apply_conv_k: only double is supported");
      static_assert(NDIM == 3, "apply_conv_k: only NDIM=3 is supported");
      constexpr int k     = decltype(K)::value;
      constexpr int NW    = apply_conv_min_warps;
      static_assert(k == 8 || k == 16, "apply_conv_k: k must be 8 or 16");
      constexpr int MT    = k / 8;             // 8-wide tiles per dimension
      constexpr int KS    = k / 4;             // 4-deep contraction steps per dimension
      constexpr int ZS    = k / NW;            // z-planes per warp in stages 1+2
      constexpr int NT3   = k * k / (8 * NW);  // 8-wide column tiles per warp in stage 3
      constexpr int SLOTS = k / 2;             // 16-byte slots per row of W

      /* MADNESS uses slightly different thresholds for R and S terms. */
      constexpr double thresh = (Term == NormId::Rnorm) ? 1.e-20 : 0.0;

      const int tid    = thread_id();
      const int warp   = tid / MRA_WARP_SIZE;
      const int lane   = tid % MRA_WARP_SIZE;
      const int g      = lane / 4;
      const int t      = lane % 4;
      const bool active = warp < NW;  // any extra warps only join the copies and barriers
      const int w      = active ? warp : 0;

      /* W[n][z] in shared memory, see above */
      extern __shared__ T c_smem[];
      auto widx = [](int xp, int yp, int z) {
        return (yp * k + xp) * k + 2 * (((z / 2) + xp + yp) % SLOTS) + (z % 2);
      };

      /* f may have just been written by other threads (f0, see detail::apply_conv),
       * and a previous call may still be using the shared memory */
      SYNCTHREADS();

      /* Stage-1 B fragments: fr[s][nt][ks] = f[z = ZS*w+s][y = g+8nt][x = KS*t+ks] */
      T fr[ZS][MT][KS];
      const T* fptr = f.data();
      #pragma unroll
      for (int s = 0; s < ZS; ++s) {
        #pragma unroll
        for (int nt = 0; nt < MT; ++nt) {
          #pragma unroll
          for (int ks = 0; ks < KS; ++ks) {
            fr[s][nt][ks] = fptr[((ZS * w + s) * k + g + 8 * nt) * k + KS * t + ks];
          }
        }
      }

      /* acc[mt][nt][r] = result[z' = g+8mt][n = 2t+r+8(NT3*w+nt)] */
      T acc[MT][NT3][2];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT3; ++nt) {
          acc[mt][nt][0] = acc[mt][nt][1] = 0.0;
        }
      }

      /**
       * Raw indexing into opnorms (opid, mu, d, NormId) and trans[d]
       * (opid, mu, k, k): TensorView's operator() keeps its strides in local
       * memory, which is too expensive in this loop.
       */
      const T* norms = opnorms.data();
      const size_type norms_rank = opnorms.dim(1);
      auto norm = [&](int mu, int d, NormId id) {
        return norms[((opid * norms_rank + mu) * NDIM + d) * (size_type)NormId::Count + (size_type)id];
      };
      auto kept = [&](int mu) {
        double dnorm = 1.0;
        for (Dimension d = 0; d < NDIM; ++d) dnorm *= norm(mu, d, Term);
        return norm(mu, 0, NormId::MUnorm) > optol && dnorm > thresh;
      };

      /**
       * Which terms are applied (uniform across the block), as a bitmask
       * computed once up front, so that finding the next term does not wait
       * for global memory. Terms beyond MaxMaskRank are tested on the fly.
       */
      constexpr int MaxMaskRank = 1024;
      __shared__ uint32_t kept_mask[MaxMaskRank / 32];
      const int mask_rank = rank < MaxMaskRank ? rank : MaxMaskRank;
      for (int base = warp * MRA_WARP_SIZE; base < mask_rank; base += (block_size() / MRA_WARP_SIZE) * MRA_WARP_SIZE) {
        const int mu = base + lane;
        const uint32_t bits = __ballot_sync(0xffffffffu, mu < mask_rank && kept(mu));
        if (lane == 0) kept_mask[base / 32] = bits;
      }
      SYNCTHREADS();
      /* the first applied term >= mu, or rank */
      auto next_kept = [&](int mu) {
        while (mu < mask_rank) {
          const uint32_t bits = kept_mask[mu / 32] >> (mu % 32);
          if (bits) return mu + __ffs(bits) - 1;
          mu = (mu / 32 + 1) * 32;
        }
        while (mu < rank && !kept(mu)) ++mu;
        return mu;
      };

      /**
       * Copy Tx, Ty, Tz of term mu into tbuf (asynchronously), stored in the
       * order their A fragments are used: fragment i of lane l at
       * [(i/2)*64 + 2*l + i%2], so that a lane reads two fragments with one
       * conflict-free 16-byte load. Tx and Tz fragments are indexed
       * i = mt*KS + ks, Ty fragments i = (mt2*MT + nt)*2 + r (see the
       * stages below for the element each fragment holds).
       */
      const size_type trans_rank = trans[0].dim(1);
      auto prefetch = [&](int mu, T* tbuf) {
        const size_type offset = (opid * trans_rank + mu) * k * k;
        for (int e = tid; e < 3 * k * k; e += block_size()) {
          const int m  = e / (k * k);
          const int i  = e % (k * k);
          const int fi = 2 * (i / 64) + i % 2;  // fragment index
          const int fl = (i % 64) / 2;          // lane holding it
          const int fg = fl / 4, ft = fl % 4;
          int row, col;
          const T* src;
          if (m == 1) { // Ty[y = 2t+r+8nt][y' = g+8mt2]
            row = 2 * ft + fi % 2 + 8 * ((fi / 2) % MT);
            col = fg + 8 * (fi / (2 * MT));
            src = trans[1].data();
          } else {      // Tx[x = KS*t+ks][x' = g+8mt], Tz[z = KS*t+ks][z' = g+8mt]
            row = KS * ft + fi % KS;
            col = fg + 8 * (fi / KS);
            src = (m == 0) ? trans[2].data() : trans[0].data();
          }
          const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(&tbuf[e]));
          asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src + offset + row * k + col) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      };
      /* the fragments (2*fp, 2*fp+1) of matrix m (0: Tx, 1: Ty, 2: Tz) in tbuf */
      auto tfrag2 = [&](const T* tbuf, int m, int fp) {
        return reinterpret_cast<const double2*>(tbuf + m * k * k)[fp * MRA_WARP_SIZE + lane];
      };

      T* tbufs = c_smem + k * k * k;  // 2 x {Tx, Ty, Tz}, after W
      int buf = 0;
      /* mufac of term mu (sign-flipped for S), loaded along with its prefetch */
      auto get_mufac = [&](int mu) {
        T mufac = norm(mu, 0, NormId::Fac);
        return (NormId::Snorm == Term) ? -mufac : mufac;
      };
      int mu = next_kept(0);
      T mufac = 0.0;
      if (mu < rank) {
        prefetch(mu, tbufs);
        mufac = get_mufac(mu);
      }

      while (mu < rank) {
        const T* tb = tbufs + buf * 3 * k * k;

        /* this term's matrices have landed, and all warps are done with W */
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        SYNCTHREADS();

        if (active) {
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            /* Stage-1 A fragments: tx[ks] = Tx[x = KS*t+ks][x' = g+8mt] */
            T tx[KS];
            #pragma unroll
            for (int h = 0; h < KS / 2; ++h) {
              double2 v = tfrag2(tb, 0, mt * KS / 2 + h);
              tx[2 * h] = v.x;
              tx[2 * h + 1] = v.y;
            }

            /* Stage 1: y[s][nt][r] = Y[x' = g+8mt][y = 2t+r+8nt] for z-plane s */
            T y[ZS][MT][2];
            #pragma unroll
            for (int s = 0; s < ZS; ++s) {
              #pragma unroll
              for (int nt = 0; nt < MT; ++nt) {
                y[s][nt][0] = y[s][nt][1] = 0.0;
                #pragma unroll
                for (int ks = 0; ks < KS; ++ks) {
                  dmma_8x8x4(y[s][nt], tx[ks], fr[s][nt][ks]);
                }
              }
            }

            /* Stage 2: zz[s][r] = Z[y' = g+8mt2][x' = 2t+r+8mt] for z-plane s */
            #pragma unroll
            for (int mt2 = 0; mt2 < MT; ++mt2) {
              /* Stage-2 A fragments: ty[nt][r] = Ty[y = 2t+r+8nt][y' = g+8mt2] */
              T ty[MT][2];
              #pragma unroll
              for (int nt = 0; nt < MT; ++nt) {
                double2 v = tfrag2(tb, 1, mt2 * MT + nt);
                ty[nt][0] = v.x;
                ty[nt][1] = v.y;
              }
              T zz[ZS][2];
              #pragma unroll
              for (int s = 0; s < ZS; ++s) {
                zz[s][0] = zz[s][1] = 0.0;
                #pragma unroll
                for (int nt = 0; nt < MT; ++nt) {
                  #pragma unroll
                  for (int r = 0; r < 2; ++r) {
                    dmma_8x8x4(zz[s], ty[nt][r], y[s][nt][r]);
                  }
                }
              }
              #pragma unroll
              for (int r = 0; r < 2; ++r) {
                const int xp = 2 * t + r + 8 * mt;
                const int yp = g + 8 * mt2;
                if constexpr (ZS == 2) {
                  *reinterpret_cast<double2*>(&c_smem[widx(xp, yp, 2 * w)]) = make_double2(zz[0][r], zz[1][r]);
                } else {
                  c_smem[widx(xp, yp, w)] = zz[0][r];
                }
              }
            }
          }
        }

        /* W is complete; everyone is done with the other T buffer, so the
         * next term's matrices can go there while this one finishes */
        SYNCTHREADS();

        const int mu_next = next_kept(mu + 1);
        T mufac_next = 0.0;
        if (mu_next < rank) {
          prefetch(mu_next, tbufs + (buf ^ 1) * 3 * k * k);
          mufac_next = get_mufac(mu_next);
        }

        if (active) {
          /* Stage-3 A fragments: tz[mt][ks] = Tz[z = KS*t+ks][z' = g+8mt] */
          T tz[MT][KS];
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            #pragma unroll
            for (int h = 0; h < KS / 2; ++h) {
              double2 v = tfrag2(tb, 2, mt * KS / 2 + h);
              tz[mt][2 * h] = v.x;
              tz[mt][2 * h + 1] = v.y;
            }
          }
          #pragma unroll
          for (int nt = 0; nt < NT3; ++nt) {
            /* Stage-3 B fragments: b[ks] = W[n = g+8(NT3*w+nt)][z = KS*t+ks] */
            const int n  = g + 8 * (NT3 * w + nt);
            const int xp = n % k;
            const int yp = n / k;
            T b[KS];
            #pragma unroll
            for (int h = 0; h < KS / 2; ++h) {
              double2 v = *reinterpret_cast<const double2*>(&c_smem[widx(xp, yp, KS * t + 2 * h)]);
              b[2 * h]     = v.x;
              b[2 * h + 1] = v.y;
            }
            /* sum up this term separately, then add it to acc scaled by mufac */
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              T c[2] = {0.0, 0.0};
              #pragma unroll
              for (int ks = 0; ks < KS; ++ks) {
                dmma_8x8x4(c, tz[mt][ks], b[ks]);
              }
              acc[mt][nt][0] = fma(mufac, c[0], acc[mt][nt][0]);
              acc[mt][nt][1] = fma(mufac, c[1], acc[mt][nt][1]);
            }
          }
        }

        mu = mu_next;
        mufac = mufac_next;
        buf ^= 1;
      }

      if (active) {
        T* rptr = result.data();
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          #pragma unroll
          for (int nt = 0; nt < NT3; ++nt) {
            const int zp = g + 8 * mt;
            const int n  = 2 * t + 8 * (NT3 * w + nt);
            *reinterpret_cast<double2*>(&rptr[zp * k * k + n]) = make_double2(acc[mt][nt][0], acc[mt][nt][1]);
          }
        }
      }
      /* the next call reuses the shared memory */
      SYNCTHREADS();
    }
#endif // MRA_HAVE_DMMA


    /**
     * Implementation of convolution using FP64 tensor cores (mma, sm_80+),
     * see apply_conv_k. Currently only supports NDIM=3 and K=8 (i.e., 2K=16
     * for the R term and K=8 for the S term). Works only if K is a
     * compile-time constant, because the work distribution is configured at
     * compile time. Returns false if it does not apply.
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
#ifdef MRA_HAVE_DMMA
      if constexpr (mra::is_ct_integral_v<decltype(K)>) {
        // apply_conv_k's per-warp register capacity needs at least MinWarps
        // whole warps -- guaranteed by launching with apply_conv_min_threads(K)
        assert(block_size() % MRA_WARP_SIZE == 0 &&
               block_size() / MRA_WARP_SIZE >= apply_conv_min_warps);
        // get the rank of the operation
        const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank); // doing computation assuming full rank

        static_assert(NDIM == 3, "Tensor-core convolution only supports NDIM=3");
        static_assert(K == 8, "Tensor-core convolution only supports K=8");
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
#endif // MRA_HAVE_DMMA
      return false;
    }

    /**
     * Returns the number of bytes of shared memory required for the
     * convolution kernel using tensor cores (see apply_conv_k).
     *
     * NOTE: This needs to be updated if the kernel is changed to
     *       use more shared memory.
     */
    template<typename T>
    SCOPE constexpr size_type apply_conv_shmem_size(auto K) {
#ifdef MRA_ENABLE_CUDA
      if (K == 8) {
        auto K2 = mra::Int<2>{}*K;
        // W (exchange buffer of apply_conv_k) and two sets of {Tx, Ty, Tz}
        return (K2*K2*K2 + 2*3*K2*K2)*sizeof(T);
      }
#endif // MRA_ENABLE_CUDA
      return 0;
    }

  } // namespace mma
  } // namespace accel

} // namespace mra

#undef MRA_HAVE_DMMA

#endif // MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H