#ifndef MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H
#define MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H

/**
 * Convolution on FP64 tensor cores (sm_80+) using mma.sync PTX directly.
 * Selected by mra/kernels/impl/convolution_accel.h.
 */

#include "mra/misc/types.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/conv_mad.h" // for NormId, used as a template parameter below
#include "mra/ops/functions.h"  // reduce_warp


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
     * Number of warps of the K=6 term functions: apply_conv_k6 (k = 12) gives
     * each warp 2 of the 12 z-planes and 3 of the 18 column tiles,
     * apply_conv_k6s (k = 6) every 6th term.
     */
    inline constexpr int apply_conv_k6_warps = 6;

    /**
     * Warps per block of the separate k = K launch (see
     * apply_conv_splits_s), one f per warp.
     */
    inline constexpr int apply_conv_s_warps = 4;

    /**
     * Warps and shared memory (in elements of T) that the term function for
     * k needs; {0, 0} if k has none. K takes the tensor-core path only if both
     * k = 2K (R term) and k = K (S term) have one. Block size, launch bounds
     * and shared memory are derived from this table.
     *
     * To add a K: write its term functions, call them from apply_term, list
     * their needs here, and add K to detail::with_ct_K (kernels/convolution.h).
     */
    struct term_needs {
      int warps;
      int smem;
    };

    SCOPE constexpr term_needs apply_term_needs(size_type k) {
      switch (k) {
        case 16:
        case 8:   // apply_conv_k: W (k^3) and two sets of {Tx, Ty, Tz} (k^2 each)
          return {apply_conv_min_warps, static_cast<int>(k * k * k + 2 * 3 * k * k)};
        case 12:  // apply_conv_k6: W (rows padded to 16) and two sets of {Tx, Ty, Tz} A fragments (6 per lane)
          return {apply_conv_k6_warps, 12 * 12 * 16 + 2 * 3 * 6 * 32};
        case 6:   // apply_conv_k6s: a W (k^3) per warp
          return {apply_conv_k6_warps, apply_conv_k6_warps * 6 * 6 * 6};
        default:
          return {0, 0};
      }
    }

    /**
     * Minimum number of threads per block for kernels calling apply_conv
     * with this K: enough whole warps for both term functions (see
     * apply_term_needs) if the tensor-core path is used for K, 0 otherwise.
     */
    SCOPE constexpr int apply_conv_min_threads(auto K) {
#ifdef MRA_ENABLE_CUDA
      const size_type k = K;
      const int r = apply_term_needs(2 * k).warps, s = apply_term_needs(k).warps;
      if (r > 0 && s > 0) {
        // CUDA warps are 32 threads (MRA_WARP_SIZE is only defined in device code)
        return (r > s ? r : s) * 32;
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

      /* Stage-1 B fragments: fr[s][nt][ks] = f[z = ZS*w+s][y = g+8nt][x = KS*t+ks],
       * read 16 bytes (ks, ks+1) at a time; f must be 16-byte aligned */
      T fr[ZS][MT][KS];
      const T* fptr = f.data();
      assert(reinterpret_cast<uintptr_t>(fptr) % alignof(double2) == 0);
      #pragma unroll
      for (int s = 0; s < ZS; ++s) {
        #pragma unroll
        for (int nt = 0; nt < MT; ++nt) {
          const double2* frow = reinterpret_cast<const double2*>(fptr + ((ZS * w + s) * k + g + 8 * nt) * k + KS * t);
          #pragma unroll
          for (int h = 0; h < KS / 2; ++h) {
            double2 v = frow[h];
            fr[s][nt][2 * h]     = v.x;
            fr[s][nt][2 * h + 1] = v.y;
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
       * Thread tid copies the elements e = tid + j*block_size() of tbuf (Tx,
       * Ty, Tz one after the other) from csrc[j] + mu*k*k; csrc[j], the
       * source for term 0, is computed once. Full pointers also keep the
       * compiler from reloading trans[].data() from global memory for every
       * copy (each cp.async has a "memory" clobber).
       */
      constexpr int CJ = (3 * k * k + NW * MRA_WARP_SIZE - 1) / (NW * MRA_WARP_SIZE);  // copies per thread
      const size_type trans_rank = trans[0].dim(1);
      const T* csrc[CJ];  // source of copy j for term 0, nullptr if none
      #pragma unroll
      for (int j = 0; j < CJ; ++j) {
        /* unsigned: all divisors are powers of 2, so / and % become shifts and masks */
        const unsigned e  = tid + j * block_size();
        const unsigned m  = e / (k * k);
        const unsigned i  = e % (k * k);
        const unsigned fi = 2 * (i / 64) + i % 2;  // fragment index
        const unsigned fl = (i % 64) / 2;          // lane holding it
        const unsigned fg = fl / 4, ft = fl % 4;
        unsigned row, col;
        if (m == 1) {  // Ty[y = 2t+r+8nt][y' = g+8mt2]
          row = 2 * ft + fi % 2 + 8 * ((fi / 2) % MT);
          col = fg + 8 * (fi / (2 * MT));
        } else {       // Tx[x = KS*t+ks][x' = g+8mt], Tz[z = KS*t+ks][z' = g+8mt]
          row = KS * ft + fi % KS;
          col = fg + 8 * (fi / KS);
        }
        const T* base = (m == 0) ? trans[2].data() : (m == 1) ? trans[1].data() : trans[0].data();
        csrc[j] = (e < 3u * k * k) ? base + opid * trans_rank * k * k + row * k + col : nullptr;
      }
      auto prefetch = [&](int mu, T* tbuf) {
        #pragma unroll
        for (int j = 0; j < CJ; ++j) {
          if (csrc[j] != nullptr) {
            const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(&tbuf[tid + j * block_size()]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(csrc[j] + mu * k * k) : "memory");
          }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      };
      /* the fragments (2*fp, 2*fp+1) of matrix m (0: Tx, 1: Ty, 2: Tz) in tbuf */
      auto tfrag2 = [&](const T* tbuf, int m, int fp) {
        return reinterpret_cast<const double2*>(tbuf + m * k * k)[fp * MRA_WARP_SIZE + lane];
      };

      T* tbufs = c_smem + k * k * k;  // 2 x {Tx, Ty, Tz}, after W
      int buf = 0;
      /* mufac of term mu, loaded along with its prefetch (the S term's sign is
       * applied in stage 3, see sfac) */
      auto get_mufac = [&](int mu) {
        return norm(mu, 0, NormId::Fac);
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
              /* the S term's sign goes on the DFMA operand, where it is free: a
               * separate negation is an extra FP64 op that waits behind the DMMAs */
              const T sfac = (NormId::Snorm == Term) ? -mufac : mufac;
              acc[mt][nt][0] = fma(sfac, c[0], acc[mt][nt][0]);
              acc[mt][nt][1] = fma(sfac, c[1], acc[mt][nt][1]);
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

    /**
     * The applied terms mu of one separated operator (R or S), as in
     * apply_conv_k: a term is applied if MADNESS's screening keeps it. The
     * first MaxMaskRank terms are decided once by the block, as a bitmask in
     * shared memory; later terms are tested one by one. opnorms
     * (opid, mu, d, NormId) is indexed by hand, as in apply_conv_k.
     */
    template<typename T, Dimension NDIM, NormId Term>
    class conv_terms {
    public:
      static constexpr int MaxMaskRank = 1024;
      using mask_type = uint32_t[MaxMaskRank / 32];

      /* Called by all threads of the block: builds the bitmask in mask (shared
       * memory, owned by the caller) and synchronizes the block. */
      template<concepts::TensorView<4> ViewOpnorms>
      DEVSCOPE conv_terms(int opid, int rank, const T optol, const ViewOpnorms& opnorms, mask_type& mask)
      : m_norms(opnorms.data())
      , m_norms_rank(opnorms.dim(1))
      , m_opid(opid)
      , m_rank(rank)
      , m_optol(optol)
      , m_mask_rank(rank < MaxMaskRank ? rank : MaxMaskRank)
      , m_mask(mask)
      {
        const int warp = thread_id() / MRA_WARP_SIZE;
        const int lane = thread_id() % MRA_WARP_SIZE;
        for (int base = warp * MRA_WARP_SIZE; base < m_mask_rank; base += (block_size() / MRA_WARP_SIZE) * MRA_WARP_SIZE) {
          const int mu = base + lane;
          const uint32_t bits = __ballot_sync(0xffffffffu, mu < m_mask_rank && kept(mu));
          if (lane == 0) mask[base / 32] = bits;
        }
        SYNCTHREADS();
      }

      /* Called by the 32 lanes of one warp alone: builds the bitmask in mask
       * (owned by this warp) and synchronizes the warp. */
      template<concepts::TensorView<4> ViewOpnorms>
      DEVSCOPE conv_terms(int opid, int rank, const T optol, const ViewOpnorms& opnorms, mask_type& mask, unsigned lane)
      : m_norms(opnorms.data())
      , m_norms_rank(opnorms.dim(1))
      , m_opid(opid)
      , m_rank(rank)
      , m_optol(optol)
      , m_mask_rank(rank < MaxMaskRank ? rank : MaxMaskRank)
      , m_mask(mask)
      {
        for (int base = 0; base < m_mask_rank; base += MRA_WARP_SIZE) {
          const int mu = base + lane;
          const uint32_t bits = __ballot_sync(0xffffffffu, mu < m_mask_rank && kept(mu));
          if (lane == 0) mask[base / 32] = bits;
        }
        __syncwarp();
      }

      /* the first applied term >= mu, or rank */
      DEVSCOPE int next(int mu) const {
        while (mu < m_mask_rank) {
          const uint32_t bits = m_mask[mu / 32] >> (mu % 32);
          if (bits) return mu + __ffs(bits) - 1;
          mu = (mu / 32 + 1) * 32;
        }
        while (mu < m_rank && !kept(mu)) ++mu;
        return mu;
      }

      /* the n-th applied term after mu, or rank */
      DEVSCOPE int advance(int mu, int n) const {
        for (int s = 0; s < n && mu < m_rank; ++s) mu = next(mu + 1);
        return mu;
      }

      /* the factor of term mu (sign-flipped for S) */
      DEVSCOPE T mufac(int mu) const {
        T mufac = fac(mu);
        return (NormId::Snorm == Term) ? -mufac : mufac;
      }

      /* the factor of term mu without the S sign (see apply_conv_k8w) */
      DEVSCOPE T fac(int mu) const {
        return norm(mu, 0, NormId::Fac);
      }

    private:
      /* MADNESS uses slightly different thresholds for R and S terms. */
      static constexpr double thresh = (Term == NormId::Rnorm) ? 1.e-20 : 0.0;

      DEVSCOPE T norm(int mu, int d, NormId id) const {
        return m_norms[((m_opid * m_norms_rank + mu) * NDIM + d) * (size_type)NormId::Count + (size_type)id];
      }

      DEVSCOPE bool kept(int mu) const {
        double dnorm = 1.0;
        for (Dimension d = 0; d < NDIM; ++d) dnorm *= norm(mu, d, Term);
        return norm(mu, 0, NormId::MUnorm) > m_optol && dnorm > thresh;
      }

      const T* m_norms;
      size_type m_norms_rank;
      int m_opid;
      int m_rank;
      T m_optol;
      int m_mask_rank;
      const mask_type& m_mask;
    };

    /**
     * apply_conv_k for k = 12, the R term of K=6 (the S term is
     * apply_conv_k6s). 12 is not a multiple of the 8-wide mma tile: tile rows
     * and columns beyond 12 are padding, filled with zeros and never
     * stored. Same structure as apply_conv_k, on NW = apply_conv_k6_warps
     * warps; warp w computes the z-planes 2w and 2w+1 (p = 0, 1).
     *  - Each contraction (x, y, z) takes three 4-deep steps j over the index
     *    c(t, j) = 2t+j for j < 2 and 8+t for j = 2 (see cidx): steps 0 and 1
     *    cover 0..7, step 2 covers 8..11.
     *  - Stage 1 uses three B tiles per warp: y = 0..7 of plane 0, y = 0..7
     *    of plane 1, and one tile with y = 8..11 of both planes (column n:
     *    plane n%2, y = 8+n/2). Its output column 2t+r (plane r, y = 8+t) is
     *    exactly step 2 of plane r in stage 2. The second output row tile
     *    holds x' = 8+g/2 (rows duplicated in pairs, see orow).
     *  - Stage 2: x' = 0..7 get one accumulator per plane; x' = 8..11 of
     *    both planes share one (column n: plane n%2, x' = 8+n/2). Every lane
     *    stores z pairs (2w, 2w+1), as apply_conv_k does with ZS == 2.
     *  - Stage 3: warp w computes the columns n = y'*k + x' in [24w, 24w+24).
     * W has rows of 16 (8 slots of 16 bytes); slot s of row (x', y') is
     * stored at s ^ h(x', y') (see widx), so no store or load has bank
     * conflicts.
     */
    template<typename T, Dimension NDIM, NormId Term,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTrans,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewResult>
    DEVSCOPE void apply_conv_k6(
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
      static_assert(std::is_same_v<T, double>, "apply_conv_k6: only double is supported");
      static_assert(NDIM == 3, "apply_conv_k6: only NDIM=3 is supported");
      constexpr int k     = decltype(K)::value;
      constexpr int NW    = apply_conv_k6_warps;
      static_assert(k == 12, "apply_conv_k6: k must be 12");
      constexpr int MT    = (k + 7) / 8;              // 8-wide tiles per output index
      constexpr int KS    = (k + 3) / 4;              // 4-deep contraction steps per dimension
      constexpr int NP    = KS / 2;                   // A fragment pairs per tile (one 16-byte load each)
      constexpr int NS    = KS % 2;                   // single A fragments per tile (k = 12: the 8+t step)
      constexpr int NF    = MT * KS;                  // A fragments per lane and matrix
      constexpr int NB1   = 3;                        // stage-1 B tiles per warp
      constexpr int NT3   = 3;                        // 8-wide column tiles per warp in stage 3
      static_assert(NW * NT3 * 8 == k * k, "apply_conv_k6: the warps' column tiles must cover n");

      /* shared memory: W, then two sets of {Tx, Ty, Tz} A fragments */
      constexpr int WROW  = 16;                       // row length of W (padded for the swizzle, see widx)
      constexpr int WSIZE = k * k * WROW;
      constexpr int PM    = NF * MRA_WARP_SIZE;       // A-fragment entries per matrix
      constexpr int TSET  = 3 * PM;                   // one set of {Tx, Ty, Tz}
      static_assert(apply_term_needs(k).warps == NW && apply_term_needs(k).smem == WSIZE + 2 * TSET,
                    "apply_conv_k6: apply_term_needs(12) does not match this layout");

      const int tid    = thread_id();
      const int warp   = tid / MRA_WARP_SIZE;
      const int lane   = tid % MRA_WARP_SIZE;
      const int g      = lane / 4;
      const int t      = lane % 4;
      const bool active = warp < NW;  // any extra warps only join the copies and barriers
      const int w      = active ? warp : 0;

      /* contraction index c(t, j) of step j for lane%4 = t; k means padding */
      auto cidx = [](int t, int j) {
        return j < 2 ? 2 * t + j : (j == 2 ? 8 + t : k);
      };
      /* output index of A-fragment row g of tile mt; k means padding.
       * dup: Tx in stage 1, whose second tile holds each row twice */
      auto orow = [](int mt, int g, bool dup) {
        if (mt == 0) return g < k ? g : k;
        return dup ? 8 + g / 2 : (g < 4 ? 8 + g : k);
      };

      /* index of W[n = y'*k + x'][z] in rows of WROW */
      extern __shared__ T c_smem[];
      auto widx = [](int xp, int yp, int z) {
        // h = 2 * (p[x'%4] ^ (x'/4)%2) + y'%2 with p = (0, 2, 3, 1)
        const int h = 2 * (((0x78 >> (2 * (xp % 4))) & 3) ^ ((xp / 4) % 2)) | (yp % 2);
        return (yp * k + xp) * WROW + 2 * ((z / 2) ^ h) + (z % 2);
      };

      /* f may have just been written by other threads (f0, see detail::apply_conv),
       * and a previous call may still be using the shared memory */
      SYNCTHREADS();

      /* Stage-1 B fragments: fr[b][j] = f[z][y][x = c(t,j)] with (z, y) the
       * plane and row of column g of tile b (see above) */
      T fr[NB1][KS];
      const T* fptr = f.data();
      #pragma unroll
      for (int b = 0; b < NB1; ++b) {
        const int z = b < 2 ? 2 * w + b : 2 * w + g % 2;
        const int y = b == 2 ? 8 + g / 2 : g;
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
          const int x = cidx(t, j);
          fr[b][j] = (y < k && x < k) ? fptr[(z * k + y) * k + x] : 0.0;
        }
      }

      /* acc[mt][nt][r] = result[z' = orow(mt, g)][n = 2t+r+8(NT3*w+nt)] */
      T acc[MT][NT3][2];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < NT3; ++nt) {
          acc[mt][nt][0] = acc[mt][nt][1] = 0.0;
        }
      }

      /* the applied terms; building their bitmask synchronizes the block */
      __shared__ typename conv_terms<T, NDIM, Term>::mask_type kept_mask;
      const conv_terms<T, NDIM, Term> terms(opid, rank, optol, opnorms, kept_mask);

      /**
       * prefetch(mu, tbuf) copies the A fragments of Tx, Ty, Tz of term mu
       * into tbuf with cp.async, in the order they are read: first the pairs
       * (j = 2h, 2h+1) of each tile mt, lane l's pair at
       * [(mt*NP + h)*64 + 2*l + j%2] (one 16-byte load), then the single
       * fragment j = KS-1 at [MT*NP*64 + mt*32 + l] (one 8-byte load).
       * Fragment j of tile mt holds T[c(t, j)][orow(mt, g)], zero for padding.
       * Each thread copies the entries tid + p*block_size() of each matrix.
       * Their sources are the same for every term up to an offset, so they
       * are computed once, one byte per entry: the offset in the k x k
       * matrix, ZFill (write zero) or None (no entry). Tx has its own plan
       * (rows duplicated), Ty and Tz share one.
       */
      constexpr int PJ = (PM + NW * MRA_WARP_SIZE - 1) / (NW * MRA_WARP_SIZE);  // entries per thread and matrix
      constexpr uint32_t ZFill = 0xfe, None = 0xff;
      static_assert(PJ <= 4 && k * k <= (int)ZFill, "apply_conv_k6: the prefetch plan must fit in 32 bits");
      auto plan_entry = [&](int i, bool dup) -> uint32_t {
        if (i >= PM) return None;
        int mt, l, j;  // the entry is fragment j of tile mt for lane l
        if (i < MT * NP * 64) {
          mt = (i / 64) / NP;
          l  = (i % 64) / 2;
          j  = 2 * ((i / 64) % NP) + i % 2;
        } else {
          const int s = i - MT * NP * 64;
          mt = s / 32;
          l  = s % 32;
          j  = KS - 1;
        }
        const int row = cidx(l % 4, j);
        const int col = orow(mt, l / 4, dup);
        return (row < k && col < k) ? row * k + col : ZFill;
      };
      uint32_t plan_x = 0, plan_yz = 0;
      #pragma unroll
      for (int p = 0; p < PJ; ++p) {
        const int i = tid + p * block_size();
        plan_x  |= plan_entry(i, true) << (8 * p);
        plan_yz |= plan_entry(i, false) << (8 * p);
      }
      const size_type trans_rank = trans[0].dim(1);
      auto prefetch = [&](int mu, T* tbuf) {
        const size_type offset = (opid * trans_rank + mu) * k * k;
        #pragma unroll
        for (int m = 0; m < 3; ++m) {  // 0: Tx, 1: Ty, 2: Tz
          const T* src = trans[2 - m].data() + offset;
          const uint32_t plan = (m == 0) ? plan_x : plan_yz;
          #pragma unroll
          for (int p = 0; p < PJ; ++p) {
            const uint32_t v = (plan >> (8 * p)) & 0xff;
            if (v != None) {
              const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(&tbuf[m * PM + tid + p * block_size()]));
              /* src-size 0: zero-fill, nothing is read */
              asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
                           :: "r"(dst), "l"(src + (v == ZFill ? 0 : v)), "r"(v == ZFill ? 0 : 8) : "memory");
            }
          }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      };
      /* the KS fragments of tile mt of matrix m (0: Tx, 1: Ty, 2: Tz) in tbuf */
      auto tfrags = [&](const T* tbuf, int m, int mt, T (&a)[KS]) {
        const T* tm = tbuf + m * PM;
        #pragma unroll
        for (int h = 0; h < NP; ++h) {
          double2 v = reinterpret_cast<const double2*>(tm)[(mt * NP + h) * MRA_WARP_SIZE + lane];
          a[2 * h] = v.x;
          a[2 * h + 1] = v.y;
        }
        if constexpr (NS > 0) {
          a[KS - 1] = tm[MT * NP * 64 + mt * MRA_WARP_SIZE + lane];
        }
      };

      T* tbufs = c_smem + WSIZE;  // 2 x {Tx, Ty, Tz}, after W
      int buf = 0;
      /* mufac is loaded along with the term's prefetch */
      int mu = terms.next(0);
      T mufac = 0.0;
      if (mu < rank) {
        prefetch(mu, tbufs);
        mufac = terms.mufac(mu);
      }

      while (mu < rank) {
        const T* tb = tbufs + buf * TSET;

        /* this term's matrices have landed, and all warps are done with W */
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        SYNCTHREADS();

        if (active) {
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            /* Stage-1 A fragments: tx[j] = Tx[x = c(t,j)][x' = orow(mt, g, dup)] */
            T tx[KS];
            tfrags(tb, 0, mt, tx);

            /* Stage 1: y[b][r] = Y[x' = orow(mt, g, dup)][column 2t+r of tile b] */
            T y[NB1][2];
            #pragma unroll
            for (int b = 0; b < NB1; ++b) {
              y[b][0] = y[b][1] = 0.0;
              #pragma unroll
              for (int j = 0; j < KS; ++j) {
                dmma_8x8x4(y[b], tx[j], fr[b][j]);
              }
            }

            /* Stage 2: Z[y' = orow(mt2, g)][x'] */
            #pragma unroll
            for (int mt2 = 0; mt2 < MT; ++mt2) {
              /* Stage-2 A fragments: ty[j] = Ty[y = c(t,j)][y' = orow(mt2, g)] */
              T ty[KS];
              tfrags(tb, 1, mt2, ty);
              const int yp = g + 8 * mt2;
              if (mt == 0) {
                /* zz[p][r] = Z[x' = 2t+r] of plane p */
                T zz[2][2];
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                  zz[p][0] = zz[p][1] = 0.0;
                  dmma_8x8x4(zz[p], ty[0], y[p][0]);
                  dmma_8x8x4(zz[p], ty[1], y[p][1]);
                  dmma_8x8x4(zz[p], ty[2], y[2][p]);
                }
                if (yp < k) {
                  #pragma unroll
                  for (int r = 0; r < 2; ++r) {
                    *reinterpret_cast<double2*>(&c_smem[widx(2 * t + r, yp, 2 * w)]) = make_double2(zz[0][r], zz[1][r]);
                  }
                }
              } else {
                /* zz[r] = Z[x' = 8+t] of plane r; column g is plane g%2 */
                const bool p = g % 2;
                T zz[2] = {0.0, 0.0};
                dmma_8x8x4(zz, ty[0], p ? y[1][0] : y[0][0]);
                dmma_8x8x4(zz, ty[1], p ? y[1][1] : y[0][1]);
                dmma_8x8x4(zz, ty[2], p ? y[2][1] : y[2][0]);
                if (yp < k) {
                  *reinterpret_cast<double2*>(&c_smem[widx(8 + t, yp, 2 * w)]) = make_double2(zz[0], zz[1]);
                }
              }
            }
          }
        }

        /* W is complete; everyone is done with the other T buffer, so the
         * next term's matrices can go there while this one finishes */
        SYNCTHREADS();

        const int mu_next = terms.next(mu + 1);
        T mufac_next = 0.0;
        if (mu_next < rank) {
          prefetch(mu_next, tbufs + (buf ^ 1) * TSET);
          mufac_next = terms.mufac(mu_next);
        }

        if (active) {
          /* Stage-3 A fragments: tz[mt][j] = Tz[z = c(t,j)][z' = orow(mt, g)] */
          T tz[MT][KS];
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            tfrags(tb, 2, mt, tz[mt]);
          }
          #pragma unroll
          for (int nt = 0; nt < NT3; ++nt) {
            const int tile = NT3 * w + nt;
            {
              /* Stage-3 B fragments: b[j] = W[n = g+8*tile][z = c(t,j)] */
              const int n  = g + 8 * tile;
              const int xp = n % k;
              const int yp = n / k;
              T b[KS];
              double2 v = *reinterpret_cast<const double2*>(&c_smem[widx(xp, yp, 2 * t)]);
              b[0] = v.x;
              b[1] = v.y;
              b[2] = c_smem[widx(xp, yp, 8 + t)];
              /* sum up this term separately, then add it to acc scaled by mufac */
              #pragma unroll
              for (int mt = 0; mt < MT; ++mt) {
                T c[2] = {0.0, 0.0};
                #pragma unroll
                for (int j = 0; j < KS; ++j) {
                  dmma_8x8x4(c, tz[mt][j], b[j]);
                }
                acc[mt][nt][0] = fma(mufac, c[0], acc[mt][nt][0]);
                acc[mt][nt][1] = fma(mufac, c[1], acc[mt][nt][1]);
              }
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
            const int zp = orow(mt, g, false);
            const int n  = 2 * t + 8 * (NT3 * w + nt);
            if (zp < k && n < k * k) {
              *reinterpret_cast<double2*>(&rptr[zp * k * k + n]) = make_double2(acc[mt][nt][0], acc[mt][nt][1]);
            }
          }
        }
      }
      /* the next call reuses the shared memory */
      SYNCTHREADS();
    }

    /**
     * The S term of K=6 (k = 6). Each of the NW warps applies whole terms on
     * its own, since one term split across the block would be only a few mma
     * per warp between two block barriers. Warp w takes every NW-th applied
     * term and keeps its own sum in registers; the NW sums are added in a
     * fixed order at the end. Per term, a warp computes stages 1+2 for all 6
     * z-planes, exchanges the result through its own W (z pairs stored
     * together, __syncwarp only), and computes stage 3 for the 5 column
     * tiles of n = y'*k + x'. Every dimension is zero-padded to 8
     * (c(t, j) = 2t+j for t < 3). The next term's A fragments are read into
     * registers while the current term is computed.
     */
    template<typename T, Dimension NDIM, NormId Term,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTrans,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewResult>
    DEVSCOPE void apply_conv_k6s(
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
      static_assert(std::is_same_v<T, double>, "apply_conv_k6s: only double is supported");
      static_assert(NDIM == 3, "apply_conv_k6s: only NDIM=3 is supported");
      constexpr int k     = decltype(K)::value;
      constexpr int NW    = apply_conv_k6_warps;
      static_assert(k == 6, "apply_conv_k6s: k must be 6");
      constexpr int NTILE = (k * k + 7) / 8;  // 8-wide column tiles of n = y'*k + x'
      constexpr int WS    = k * k * k;        // one warp's W
      static_assert(apply_term_needs(k).warps == NW && apply_term_needs(k).smem == NW * WS,
                    "apply_conv_k6s: apply_term_needs(6) does not match this layout");

      const int tid    = thread_id();
      const int warp   = tid / MRA_WARP_SIZE;
      const int lane   = tid % MRA_WARP_SIZE;
      const int g      = lane / 4;
      const int t      = lane % 4;
      const bool active = warp < NW;  // any extra warps only join the barriers
      const int w      = active ? warp : 0;
      const bool valid = t < 3 && g < k;  // the lane holds a contraction index 2t+j < k and an output row g < k

      /* index of this warp's W[n = y'*k + x'][z]; the three 16-byte slots of a
       * row are rotated by 2x' */
      extern __shared__ T c_smem[];
      T* wbuf = c_smem + w * WS;
      auto widx = [](int xp, int yp, int z) {
        return (yp * k + xp) * k + 2 * (((z / 2) + 2 * xp) % 3) + (z % 2);
      };

      /* f may have just been written by other threads (f0, see detail::apply_conv),
       * and a previous call may still be using the shared memory */
      SYNCTHREADS();

      /* Stage-1 B fragments of all planes: fr[z][j] = f[z][y = g][x = 2t+j] */
      T fr[k][2];
      const T* fptr = f.data();
      #pragma unroll
      for (int z = 0; z < k; ++z) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          fr[z][j] = valid ? fptr[(z * k + g) * k + 2 * t + j] : 0.0;
        }
      }

      /* acc[nt][r] = result[z' = g][n = 2t+r+8nt] */
      T acc[NTILE][2];
      #pragma unroll
      for (int nt = 0; nt < NTILE; ++nt) {
        acc[nt][0] = acc[nt][1] = 0.0;
      }

      /* the applied terms; building their bitmask synchronizes the block */
      __shared__ typename conv_terms<T, NDIM, Term>::mask_type kept_mask;
      const conv_terms<T, NDIM, Term> terms(opid, rank, optol, opnorms, kept_mask);

      /* this lane's A-fragment elements of term mu: a[m][j] = T[c(t,j)][g] of
       * Tx, Ty, Tz (m = 0, 1, 2), zero for padding */
      const size_type trans_rank = trans[0].dim(1);
      auto load_frags = [&](int mu, T (&a)[3][2]) {
        const size_type offset = (opid * trans_rank + mu) * k * k + 2 * t * k + g;
        #pragma unroll
        for (int m = 0; m < 3; ++m) {
          #pragma unroll
          for (int j = 0; j < 2; ++j) {
            a[m][j] = valid ? trans[2 - m].data()[offset + j * k] : 0.0;
          }
        }
      };
      /* warp w takes the applied terms w, w + NW, w + 2*NW, ...; the next
       * one's fragments and mufac are read one term ahead */
      int mu = terms.advance(terms.next(0), w);
      T a[3][2], an[3][2];
      T mufac = 0.0;
      if (active && mu < rank) {
        load_frags(mu, a);
        mufac = terms.mufac(mu);
      }

      while (active && mu < rank) {
        const int mu_next = terms.advance(mu, NW);
        T mufac_next = 0.0;
        if (mu_next < rank) {
          load_frags(mu_next, an);
          mufac_next = terms.mufac(mu_next);
        }

        /* Stages 1+2 for the plane pairs (2p, 2p+1): zz[s][r] = Z[y' = g][x' = 2t+r] of plane 2p+s */
        #pragma unroll
        for (int p = 0; p < k / 2; ++p) {
          T zz[2][2];
          #pragma unroll
          for (int s = 0; s < 2; ++s) {
            T y[2] = {0.0, 0.0};
            dmma_8x8x4(y, a[0][0], fr[2 * p + s][0]);
            dmma_8x8x4(y, a[0][1], fr[2 * p + s][1]);
            zz[s][0] = zz[s][1] = 0.0;
            dmma_8x8x4(zz[s], a[1][0], y[0]);
            dmma_8x8x4(zz[s], a[1][1], y[1]);
          }
          #pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int xp = 2 * t + r;
            if (xp < k && g < k) {
              *reinterpret_cast<double2*>(&wbuf[widx(xp, g, 2 * p)]) = make_double2(zz[0][r], zz[1][r]);
            }
          }
        }
        __syncwarp();

        /* Stage 3: B fragments b = W[n = g+8nt][z = 2t, 2t+1] */
        #pragma unroll
        for (int nt = 0; nt < NTILE; ++nt) {
          const int n = g + 8 * nt;
          double2 b = make_double2(0.0, 0.0);
          if (n < k * k && t < 3) b = *reinterpret_cast<const double2*>(&wbuf[widx(n % k, n / k, 2 * t)]);
          T c[2] = {0.0, 0.0};
          dmma_8x8x4(c, a[2][0], b.x);
          dmma_8x8x4(c, a[2][1], b.y);
          acc[nt][0] = fma(mufac, c[0], acc[nt][0]);
          acc[nt][1] = fma(mufac, c[1], acc[nt][1]);
        }
        /* all lanes are done with W before the next term overwrites it */
        __syncwarp();

        mu = mu_next;
        mufac = mufac_next;
        #pragma unroll
        for (int m = 0; m < 3; ++m) {
          a[m][0] = an[m][0];
          a[m][1] = an[m][1];
        }
      }

      /* add up the warps' sums: each warp stores its sum in its own W, then
       * every thread adds up its elements over the warps */
      SYNCTHREADS();
      if (active) {
        #pragma unroll
        for (int nt = 0; nt < NTILE; ++nt) {
          #pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int n = 2 * t + r + 8 * nt;
            if (g < k && n < k * k) wbuf[g * k * k + n] = acc[nt][r];
          }
        }
      }
      SYNCTHREADS();
      T* rptr = result.data();
      for (int i = tid; i < WS; i += block_size()) {
        T sum = 0.0;
        for (int ww = 0; ww < NW; ++ww) sum += c_smem[ww * WS + i];
        rptr[i] = sum;
      }
      /* the next call reuses the shared memory */
      SYNCTHREADS();
    }

    /**
     * ||f|| of the 4096 values at fptr, computed by one warp in the order
     * normf uses in a 256-thread block, so the result is bit for bit the
     * one convolution_kernel_impl gets: lane l plays threads l + 32j
     * (j = 0..7) of sumabssq, then reduce_block's two reduce_warp stages.
     */
    __device__ __forceinline__
    double normf_block256_by_warp(const double* fptr) {
      constexpr unsigned BS = 256, N = 4096, NJ = BS / MRA_WARP_SIZE;
      const unsigned lane = thread_id() % MRA_WARP_SIZE;
      const double* p = fptr + lane;
      double s[NJ];  // the sums of threads lane + 32j
      #pragma unroll
      for (unsigned j = 0; j < NJ; ++j) {
        double a = 0.0;
        #pragma unroll
        for (unsigned i = MRA_WARP_SIZE * j; i < N; i += BS) {
          double x = p[i];
          a += x * x;
        }
        s[j] = mra::detail::reduce_warp(a, lane, MRA_WARP_SIZE);  // warp j's sum, in lane 0
      }
      double v = 0.0;  // warp j's sum in lane j, zero elsewhere
      #pragma unroll
      for (unsigned j = 0; j < NJ; ++j) {
        const double sj = __shfl_sync(0xffffffffu, s[j], 0);
        if (lane == j) v = sj;
      }
      v = mra::detail::reduce_warp(v, lane, MRA_WARP_SIZE);
      return std::sqrt(__shfl_sync(0xffffffffu, v, 0));
    }

    /**
     * In-register transpose for apply_conv_k8w, from stage 2's layout (lane
     * (g, t) holds Z[y' = g][x' = 2t+r] of plane z in zz[z][r]) to stage 3's
     * B fragments (lane (g', t') needs W[n = g'+8nt][z = 2t'+ks] =
     * Z[y' = nt][x' = g'] of plane 2t'+ks). In bits: lane (y'2 y'1 y'0 x'2 x'1)
     * and register (x'0, z2 z1 z0) become lane (x'2 x'1 x'0 z2 z1) and register
     * (y'2 y'1 y'0, z0). Rounds 1-3 each swap a lane bit with a register bit:
     * a lane sends the register whose bit differs from its own lane bit and
     * receives the partner's into it. Round 4 permutes the lanes. Afterwards
     * W[n = g'+8nt][z = 2t'+ks] is zz[4*((nt/2)%2) + 2*(nt%2) + ks][nt/4].
     */
    __device__ __forceinline__
    void transpose_k8(double (&zz)[8][2], unsigned lane) {
      auto exch = [&](double& lo, double& hi, unsigned bit) {
        const bool b = (lane >> bit) & 1;
        const double recv = __shfl_xor_sync(0xffffffffu, b ? lo : hi, 1u << bit);
        if (b) lo = recv; else hi = recv;
      };
      /* round 1: lane bit 4 <-> x'0 */
      #pragma unroll
      for (int z = 0; z < 8; ++z) exch(zz[z][0], zz[z][1], 4);
      /* round 2: lane bit 3 <-> z2 */
      #pragma unroll
      for (int z = 0; z < 4; ++z) {
        exch(zz[z][0], zz[z + 4][0], 3);
        exch(zz[z][1], zz[z + 4][1], 3);
      }
      /* round 3: lane bit 2 <-> z1, over the planes with z1 = 0 */
      constexpr int z1lo[4] = {0, 1, 4, 5};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        exch(zz[z1lo[i]][0], zz[z1lo[i] + 2][0], 2);
        exch(zz[z1lo[i]][1], zz[z1lo[i] + 2][1], 2);
      }
      /* round 4: lane (x'2 x'1 x'0 z2 z1) reads lane (x'0 z2 z1 x'2 x'1) */
      const unsigned src = (((lane >> 2) & 1) << 4) | (((lane >> 1) & 1) << 3) | ((lane & 1) << 2)
                         | (((lane >> 4) & 1) << 1) | ((lane >> 3) & 1);
      #pragma unroll
      for (int z = 0; z < 8; ++z) {
        zz[z][0] = __shfl_sync(0xffffffffu, zz[z][0], src);
        zz[z][1] = __shfl_sync(0xffffffffu, zz[z][1], src);
      }
    }

    /**
     * apply_conv_k at k = 8 by one warp, with no block barriers: applies all
     * terms of the S operator to f (the k^3 corner of the (2k)^3 tensor at
     * fptr) and stores the sum into resultc (k^3, [z'][y'][x']). Each output
     * value gets apply_conv_k's mma and FMA sequence, so the sum is bit for
     * bit the same. Per term: stages 1+2 for all 8 z-planes, transpose_k8,
     * stage 3 for the 8 column tiles of n = y'*k + x'. f stays in registers;
     * the next term's fragments and mufac are read one term ahead.
     */
    template<typename T, Dimension NDIM, NormId Term,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTrans,
             concepts::TensorView<4> ViewOpnorms>
    DEVSCOPE void apply_conv_k8w(
      int opid,
      int rank,
      const T optol,
      const ViewTrans& trans,
      const ViewOpnorms& opnorms,
      const T* fptr,     // (2k)^3
      T* resultc)        // k^3
    {
      static_assert(std::is_same_v<T, double>, "apply_conv_k8w: only double is supported");
      static_assert(NDIM == 3, "apply_conv_k8w: only NDIM=3 is supported");
      constexpr int k = 8;
      const unsigned lane = thread_id() % MRA_WARP_SIZE;
      const int warp = thread_id() / MRA_WARP_SIZE;
      const int g = lane / 4;
      const int t = lane % 4;

      /* the applied terms; each warp builds its own bitmask */
      __shared__ typename conv_terms<T, NDIM, Term>::mask_type kept_mask[apply_conv_s_warps];
      const conv_terms<T, NDIM, Term> terms(opid, rank, optol, opnorms, kept_mask[warp], lane);

      /* Stage-1 B fragments of all planes: fr[z][ks] = f[z][y = g][x = 2t+ks], 16 bytes per plane */
      T fr[k][2];
      #pragma unroll
      for (int z = 0; z < k; ++z) {
        double2 v = *reinterpret_cast<const double2*>(fptr + (z * 2 * k + g) * 2 * k + 2 * t);
        fr[z][0] = v.x;
        fr[z][1] = v.y;
      }

      /* this lane's A fragments of term mu: a[m][ks] = T[2t+ks][g] of Tx, Ty, Tz (m = 0, 1, 2) */
      const size_type trans_rank = trans[0].dim(1);
      auto load_frags = [&](int mu, T (&a)[3][2]) {
        const size_type offset = (opid * trans_rank + mu) * k * k + 2 * t * k + g;
        #pragma unroll
        for (int m = 0; m < 3; ++m) {
          #pragma unroll
          for (int ks = 0; ks < 2; ++ks) {
            a[m][ks] = __ldg(trans[2 - m].data() + offset + ks * k);
          }
        }
      };

      /* acc[nt][r] = result[z' = g][n = 2t+r+8nt] */
      T acc[k][2];
      #pragma unroll
      for (int nt = 0; nt < k; ++nt) {
        acc[nt][0] = acc[nt][1] = 0.0;
      }

      T a[3][2], an[3][2];
      int mu = terms.next(0);
      T mufac = 0.0;
      if (mu < rank) {
        load_frags(mu, a);
        mufac = terms.fac(mu);
      }
      while (mu < rank) {
        const int mu_next = terms.next(mu + 1);
        T mufac_next = 0.0;
        if (mu_next < rank) {
          load_frags(mu_next, an);
          mufac_next = terms.fac(mu_next);
        }

        /* Stages 1+2 for every plane z: zz[z][r] = Z[y' = g][x' = 2t+r] */
        T zz[k][2];
        #pragma unroll
        for (int z = 0; z < k; ++z) {
          T y[2] = {0.0, 0.0};
          dmma_8x8x4(y, a[0][0], fr[z][0]);
          dmma_8x8x4(y, a[0][1], fr[z][1]);
          zz[z][0] = zz[z][1] = 0.0;
          dmma_8x8x4(zz[z], a[1][0], y[0]);
          dmma_8x8x4(zz[z], a[1][1], y[1]);
        }

        transpose_k8(zz, lane);

        /* Stage 3 for the column tiles nt, B fragments from zz (see
         * transpose_k8): sum up this term separately, then add it to acc
         * scaled by mufac (the S term's sign on the DFMA operand, as in
         * apply_conv_k) */
        const T sfac = (NormId::Snorm == Term) ? -mufac : mufac;
        #pragma unroll
        for (int nt = 0; nt < k; ++nt) {
          const int p = 4 * ((nt / 2) % 2) + 2 * (nt % 2);
          T c[2] = {0.0, 0.0};
          dmma_8x8x4(c, a[2][0], zz[p][nt / 4]);
          dmma_8x8x4(c, a[2][1], zz[p + 1][nt / 4]);
          acc[nt][0] = fma(sfac, c[0], acc[nt][0]);
          acc[nt][1] = fma(sfac, c[1], acc[nt][1]);
        }

        mu = mu_next;
        mufac = mufac_next;
        #pragma unroll
        for (int m = 0; m < 3; ++m) {
          a[m][0] = an[m][0];
          a[m][1] = an[m][1];
        }
      }

      #pragma unroll
      for (int nt = 0; nt < k; ++nt) {
        *reinterpret_cast<double2*>(&resultc[g * k * k + 8 * nt + 2 * t]) = make_double2(acc[nt][0], acc[nt][1]);
      }
    }

    /**
     * Applies all terms of one (R or S) separated operator to the k^3 tensor
     * f with the term function for k: apply_conv_k (k = 16 or 8, K = 8),
     * apply_conv_k6 (k = 12, the R term of K = 6) or apply_conv_k6s (k = 6,
     * the S term of K = 6). What each needs from the block is listed in
     * apply_term_needs.
     */
    template<typename T, Dimension NDIM, NormId Term,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTrans,
             concepts::TensorView<4> ViewOpnorms,
             concepts::TensorView<NDIM> ViewF,
             concepts::TensorView<NDIM> ViewResult>
    DEVSCOPE void apply_term(
      int opid,
      auto k,
      int rank,
      const T optol,
      const ViewTrans& trans,
      const ViewOpnorms& opnorms,
      const ViewF& f,
      ViewResult& result  // size k, stores the sum
      )
    {
      constexpr size_type kv = decltype(k)::value;
      if constexpr (kv == 16 || kv == 8) {
        apply_conv_k<T, NDIM, Term>(opid, k, rank, optol, trans, opnorms, f, result);
      } else if constexpr (kv == 12) {
        apply_conv_k6<T, NDIM, Term>(opid, k, rank, optol, trans, opnorms, f, result);
      } else {
        static_assert(kv == 6, "apply_term: no term function for this k (see apply_term_needs)");
        apply_conv_k6s<T, NDIM, Term>(opid, k, rank, optol, trans, opnorms, f, result);
      }
    }
#endif // MRA_HAVE_DMMA

    /**
     * Whether apply_conv takes the tensor-core path for a K of type KT: K is
     * a compile-time constant and both its terms have a term function (see
     * apply_term_needs).
     */
    template<typename KT>
    SCOPE constexpr bool apply_conv_has_path() {
      if constexpr (mra::is_ct_integral_v<KT>) {
        return apply_conv_min_threads(std::decay_t<KT>::value) > 0;
      } else {
        return false;
      }
    }

    /**
     * Implementation of convolution using FP64 tensor cores (mma, sm_80+):
     * the R term (k = 2K) applied to f and the S term (k = K) to f0, each by
     * apply_term. Currently only supports NDIM=3 and the Ks of
     * apply_term_needs (K=8: k = 16 and 8, K=6: k = 12 and 6). Works only if
     * K is a compile-time constant, because the work distribution is
     * configured at compile time. Returns false if it does not apply.
     * With SDone the S term is skipped: resultc already holds it (see
     * apply_conv_splits_s).
     */
    template<typename T, Dimension NDIM, bool SDone = false,
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
      if constexpr (apply_conv_has_path<decltype(K)>()) {
        // the term functions need at least apply_conv_min_threads(K) threads, in whole warps
        assert(block_size() % MRA_WARP_SIZE == 0 &&
               block_size() / MRA_WARP_SIZE >= apply_conv_min_threads(K) / MRA_WARP_SIZE);
        // get the rank of the operation
        const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank); // doing computation assuming full rank

        static_assert(NDIM == 3, "Tensor-core convolution only supports NDIM=3");
        // convolution for f
        if (at[0]) {
          apply_term<T, NDIM, NormId::Rnorm>(
                      opid, mra::Int<2>{}*K, rank, optol, transr, opnorms,
                      f, result);
        } else {
          result = 0.0;
        }
        // convolution for f0
        if (at[1]) {
          if constexpr (!SDone) {
            apply_term<T, NDIM, NormId::Snorm>(
                        opid, K, rank, optol, transs, opnorms,
                        f0, resultc);
          }
        } else {
          resultc = 0.0;
        }
        return true;
      }
#endif // MRA_HAVE_DMMA
      return false;
    }

    /**
     * The S term of one f by one warp (see apply_conv_splits_s): ||f||, the
     * gate and optol as in convolution_kernel_impl and detail::apply_conv,
     * then apply_conv_k8w into resultc.
     */
    template<typename T, Dimension NDIM,
             concepts::TensorViewArray<4, (size_t)NDIM> ViewTranss,
             concepts::TensorView<4> ViewOpnorms>
    DEVSCOPE void apply_conv_s_warp(
      int opid,
      auto K,
      const T fac,
      const T tol,
      const ViewTranss& transs,
      const ViewOpnorms& opnorms,
      const T* fptr,      // f, (2K)^3
      T* resultc)         // K^3
    {
#ifdef MRA_HAVE_DMMA
      static_assert(decltype(K)::value == 8, "apply_conv_s_warp: only K = 8 is split off");
      /* ||f|| and the gate, as in convolution_kernel_impl */
      const T cnorm = normf_block256_by_warp(fptr);
      const T opnorm = opnorms(opid, 0, 0, (size_type)NormId::Opnorm);
      if (!((cnorm * opnorm) > (tol / fac))) return;
      /* rank and optol, as in detail::apply_conv */
      const size_type rank = opnorms(opid, 0, 0, (size_type)NormId::Rank);
      const T tolc = tol / fac / cnorm;
      const T optol = 0.01*tolc/rank;
      apply_conv_k8w<T, NDIM, NormId::Snorm>(opid, rank, optol, transs, opnorms, fptr, resultc);
#endif // MRA_HAVE_DMMA
    }

    /**
     * Whether the S term (k = K) gets its own launch, one warp per f
     * (detail::convolution_s_kernel_batched): K = 8, compile-time K only.
     * Split across the block, a k = 8 term is only 6 mma per warp between
     * two block barriers; one warp does all 48 with none.
     */
    template<typename KT>
    SCOPE constexpr bool apply_conv_splits_s() {
#ifdef MRA_ENABLE_CUDA
      if constexpr (mra::is_ct_integral_v<KT>) {
        return std::decay_t<KT>::value == 8;
      }
#endif // MRA_ENABLE_CUDA
      return false;
    }

    /**
     * Returns the number of bytes of shared memory required for the
     * convolution kernel using tensor cores (see apply_term_needs).
     *
     * NOTE: This needs to be updated if the kernel is changed to
     *       use more shared memory.
     */
    template<typename T>
    SCOPE constexpr size_type apply_conv_shmem_size(auto K) {
#ifdef MRA_ENABLE_CUDA
      if (apply_conv_min_threads(K) > 0) {
        // the larger of the two term functions' needs (see apply_term_needs)
        const size_type k = K;
        const size_type r = apply_term_needs(2 * k).smem, s = apply_term_needs(k).smem;
        return (r > s ? r : s) * sizeof(T);
      }
#endif // MRA_ENABLE_CUDA
      return 0;
    }

  } // namespace mma
  } // namespace accel

} // namespace mra

#undef MRA_HAVE_DMMA

#endif // MRA_KERNELS_IMPL_CONVOLUTION_CUDA_MMA_H