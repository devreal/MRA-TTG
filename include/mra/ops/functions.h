#ifndef MRA_FUNCTIONS_H
#define MRA_FUNCTIONS_H

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace mra {


    /// In given box return the truncation tolerance for given threshold
    template <typename T, Dimension NDIM>
    T truncate_tol(const Key<NDIM>& key, const T thresh, T cell_min_width, int truncate_mode = 0) {

        // RJH ... introduced max level here to avoid runaway
        // refinement due to truncation threshold going down to
        // intrinsic numerical error
        const Level MAXLEVEL1 = 20; // 0.5**20 ~= 1e-6
        const Level MAXLEVEL2 = 10; // 0.25**10 ~= 1e-6

        if (truncate_mode == 0) {
            return thresh;
        }
        else if (truncate_mode == 1) {
            double L = cell_min_width;
            return thresh*std::min(1.0,pow(0.5,double(std::min(key.level(),MAXLEVEL1)))*L);
        }
        else if (truncate_mode == 2) {
            double L = cell_min_width;
            return thresh*std::min(1.0,pow(0.25,double(std::min(key.level(),MAXLEVEL2)))*L*L);
        }
        else if (truncate_mode == 3) {
            // similar to truncate mode 1, but with an additional factor to
            // account for an increased number of boxes in higher dimensions

            // here is our handwaving argument: this threshold will give each
            // FunctionNode an error of less than thresh. The total error can
            // then be as high as sqrt(#nodes) * thresh. Therefore in order to
            // account for higher dimensions: divide thresh by about the root of
            // number of siblings (2^NDIM) that have a large error when we
            // refine along a deep branch of the tree. FAB
            //
            // Nope ... it can easily be as high as #nodes * tol.  The real
            // fix for this is an end-to-end error analysis of the larger
            // application and if desired to include this factor into the
            // threshold selected by the application. RJH
            const static double fac=1.0/std::pow(2,NDIM*0.5);
            double L = cell_min_width;
            return thresh*fac*std::min(1.0,pow(0.5,double(std::min(key.level(),MAXLEVEL1)))*L);

        } else {
            throw std::runtime_error("truncate_tol: unknown truncate mode " + std::to_string(truncate_mode));
        }
    }

    // volume of n-dimensional sphere of radius R
    template<typename T>
    SCOPE T vol_nsphere(int NDIM, T R) {
        return std::pow(std::numbers::pi,NDIM*0.5)*std::pow(R,NDIM)/std::tgamma(1+0.5*NDIM);
    }

    /// Computes square of distance between two coordinates
    template <typename T>
    SCOPE T distancesq(const Coordinate<T,1>& p, const Coordinate<T,1>& q) {
        T x = p[0]-q[0];
        return x*x;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,2>& p, const Coordinate<T,2>& q) {
        T x = p[0]-q[0], y = p[1]-q[1];
        return x*x + y*y;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,3>& p, const Coordinate<T,3>& q) {
        T x = p[0]-q[0], y = p[1]-q[1], z=p[2]-q[2];
        return x*x + y*y + z*z;
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,1>& p, const concepts::TensorView<1> auto& q, T* rsq, size_type N) {
        const T x = p(0);
#ifdef HAVE_DEVICE_ARCH
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
        SYNCTHREADS();
#else  // HAVE_DEVICE_ARCH
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
#endif // HAVE_DEVICE_ARCH
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,2>& p, const concepts::TensorView<2> auto& q, T* rsq, size_type N) {
        const T x = p(0);
        const T y = p(1);
#ifdef HAVE_DEVICE_ARCH
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
        SYNCTHREADS();
#else  // HAVE_DEVICE_ARCH
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
#endif // HAVE_DEVICE_ARCH
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const concepts::TensorView<2> auto& q, T* rsq, size_type N) {
        const T x = p(0);
        const T y = p(1);
        const T z = p(2);
#ifdef HAVE_DEVICE_ARCH
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
        SYNCTHREADS();
#else  // HAVE_DEVICE_ARCH
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
#endif // HAVE_DEVICE_ARCH
    }

    template <typename T>
    SCOPE void distance(const Coordinate<T,1>& p, const concepts::TensorView<1> auto& q, T* rsq, size_type N) {
        const T x = p(0);
#ifdef HAVE_DEVICE_ARCH
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            rsq[i] = std::sqrt(xx*xx);
        }
        SYNCTHREADS();
#else  // HAVE_DEVICE_ARCH
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            rsq[i] = std::sqrt(xx*xx);
        }
#endif // HAVE_DEVICE_ARCH
    }

    template <typename T>
    SCOPE void distance(const Coordinate<T,2>& p, const concepts::TensorView<2> auto& q, T* rsq, size_type N) {
        const T x = p(0);
        const T y = p(1);
#ifdef HAVE_DEVICE_ARCH
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = std::sqrt(xx*xx + yy*yy);
        }
        SYNCTHREADS();
#else  // HAVE_DEVICE_ARCH
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = std::sqrt(xx*xx + yy*yy);
        }
#endif // HAVE_DEVICE_ARCH
    }

    template <typename T>
    SCOPE void distance(const Coordinate<T,3>& p, const concepts::TensorView<2> auto& q, T* rsq, size_type N) {
        const T x = p(0);
        const T y = p(1);
        const T z = p(2);
#ifdef HAVE_DEVICE_ARCH
        for (size_type i = thread_id(); i < N; i += block_size()) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = std::sqrt(xx*xx + yy*yy + zz*zz);
        }
        SYNCTHREADS();
#else  // HAVE_DEVICE_ARCH
        for (size_type i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = std::sqrt(xx*xx + yy*yy + zz*zz);
        }
#endif // HAVE_DEVICE_ARCH
    }


    namespace detail {
#ifdef HAVE_DEVICE_ARCH
      /**
       * Sum of val over the first nlanes lanes of the calling warp, valid in
       * lane 0. All existing lanes of the warp must call this; nlanes may be
       * less than MRA_WARP_SIZE for the last, partial warp of a block, whose
       * missing lanes are never read.
       */
      template <typename T>
      DEVSCOPE T reduce_warp(T val, unsigned lane, unsigned nlanes) {
#if defined(__CUDA_ARCH__)
        const unsigned mask = (nlanes >= 32) ? 0xffffffffu : ((1u << nlanes) - 1);
#endif
        for (unsigned offset = MRA_WARP_SIZE / 2; offset > 0; offset /= 2) {
#if defined(__CUDA_ARCH__)
          T other = __shfl_down_sync(mask, val, offset);
#else
          T other = __shfl_down(val, offset);
#endif
          if (lane + offset < nlanes) val += other;
        }
        return val;
      }
#endif // HAVE_DEVICE_ARCH

      /**
       * Reduce the contributions of each calling thread in a block into a single value.
       * On the host, we simply copy the result into the output value.
       * All threads of the block must call this; only the first blocksize
       * threads contribute (blocksize can be given explicitly in case not
       * all threads contribute values). Every thread receives the result.
       * Uses warp shuffles and one shared-memory element per warp (or, if
       * MRA_SMEM_REDUCE is defined, a shared-memory tree with block_size()
       * elements).
       */
      template <typename T>
      SCOPE void reduce_block(const T input, T* output, size_type blocksize = block_size()) {
#ifdef HAVE_DEVICE_ARCH

#ifndef MRA_SMEM_REDUCE
        constexpr unsigned MaxWarps = (MAX_THREADS_PER_BLOCK + MRA_WARP_SIZE - 1) / MRA_WARP_SIZE;
        /* per-warp partial sums, and the result in its own slot so that a
         * subsequent call cannot overwrite it before every thread has read it */
        __shared__ T sdata[MaxWarps + 1];
        const unsigned tid    = thread_id();
        const unsigned nthreads = block_size();
        const unsigned lane   = tid % MRA_WARP_SIZE;
        const unsigned warp   = tid / MRA_WARP_SIZE;
        const unsigned nwarps = (nthreads + MRA_WARP_SIZE - 1) / MRA_WARP_SIZE;
        const unsigned nlanes = (nthreads - warp * MRA_WARP_SIZE < MRA_WARP_SIZE)
                              ? nthreads - warp * MRA_WARP_SIZE : MRA_WARP_SIZE;
        assert(nwarps <= MaxWarps);

        T val = reduce_warp((tid < blocksize) ? input : T(0), lane, nlanes);
        if (lane == 0) sdata[warp] = val;
        SYNCTHREADS();

        if (warp == 0) {
          val = reduce_warp((lane < nwarps) ? sdata[lane] : T(0), lane, nlanes);
          if (lane == 0) sdata[MaxWarps] = val;
        }
        SYNCTHREADS();

        // every thread returns a result
        *output = sdata[MaxWarps];

#else // MRA_SMEM_REDUCE
        __shared__ T sdata[MAX_THREADS_PER_BLOCK];
        size_type tid = thread_id();
        sdata[tid] = input;
        SYNCTHREADS();

        /* handle odd number of elements */
        if (blocksize % 2 && blocksize > 1) {
          if (tid == 0) {
            sdata[0] += sdata[blocksize - 1];
          }
          SYNCTHREADS();
        }

        for (size_type s = blocksize / 2; s > 0; s /= 2) {
          if (tid < s) {
            sdata[tid] += sdata[tid + s];
          }
          SYNCTHREADS();
          /* handle odd sizes */
          if (s % 2 == 1 && s > 1 && tid == 0) {
            /* have thread 0 fold in the last (odd) element */
            sdata[0] += sdata[s-1];
            /* no need to synchronize here, thread 0 will just continue above */
          }
        }

        SYNCTHREADS();
        // every thread returns a result
        *output = sdata[0];
        SYNCTHREADS();
#endif // MRA_SMEM_REDUCE
#else  // HAVE_DEVICE_ARCH
        *output = input;
#endif // HAVE_DEVICE_ARCH
      }
    }

    /**
     * Accumulator type for sum of squares function.
     * Maps float to double for higher accuracy.
     */
    template<typename T>
    struct accumulator_type {
      using type = std::decay_t<T>;
    };
    template<>
    struct accumulator_type<float> {
      using type = double;
    };

    template<typename T>
    using accumulator_type_t = typename accumulator_type<T>::type;

    template <typename accumulatorT>
    SCOPE void sumabssq(const concepts::TensorView auto& a, accumulatorT* sum) {
      accumulatorT s = 0.0;
      /* every thread computes a partial sum */
      foreach_idx(a, [&](size_type i) mutable {
        accumulatorT x = a[i];
        s += x*x;
      });
      detail::reduce_block(s, sum, std::min(a.size(), static_cast<size_type>(block_size())));
    }

    /// Block-wide sum of a[i]*b[i] into *sum; a and b must have the same size.
    /// As with sumabssq, *sum is only valid after a SYNCTHREADS().
    template <typename accumulatorT>
    SCOPE void sumprod(const concepts::TensorView auto& a, const concepts::TensorView auto& b, accumulatorT* sum) {
      accumulatorT s = 0.0;
      /* every thread computes a partial sum */
      foreach_idx(a, [&](size_type i) mutable {
        s += accumulatorT(a[i])*accumulatorT(b[i]);
      });
      detail::reduce_block(s, sum, std::min(a.size(), static_cast<size_type>(block_size())));
    }


    /// Compute Frobenius norm ... still needs specializing for complex
    SCOPE auto normf(const concepts::TensorView auto& a) {
      using accumulatorT = accumulator_type_t<typename std::decay_t<decltype(a)>::value_type>;
#ifdef HAVE_DEVICE_ARCH
      __shared__ accumulatorT sum;
#else  // HAVE_DEVICE_ARCH
      accumulatorT sum;
#endif // HAVE_DEVICE_ARCH
      sumabssq(a, &sum);
#ifdef HAVE_DEVICE_ARCH
      /* wait for all threads to contribute */
      SYNCTHREADS();
#endif // HAVE_DEVICE_ARCH
      return std::sqrt(sum);
    }


    /// Compute Frobenius norm ... still needs specializing for complex
    template<typename T>
    SCOPE auto normf(const T* a, size_type n) {
      using accumulatorT = accumulator_type_t<T>;
      accumulatorT s = 0.0;
      for (size_type i = thread_id(); i < n; i += block_size()) {
        accumulatorT x = a[i];
        s += x*x;
      }
      accumulatorT sum = 0.0;
      detail::reduce_block(s, &sum, std::min(n, static_cast<size_type>(block_size())));
#ifdef HAVE_DEVICE_ARCH
      /* wait for all threads to contribute */
      SYNCTHREADS();
#endif // HAVE_DEVICE_ARCH
      return std::sqrt(sum);
    }


    template<typename T>
    SCOPE void print(const T& t) {
      foreach_idxs(t, [&](auto... idx){ printf("[%lu %lu %lu] %f\n", idx..., t(idx...)); });
      SYNCTHREADS();
    }

    template<typename T>
    SCOPE void print(const T& t, const char* loc, const char *name) {
      if constexpr (T::ndim() == 3) {
        foreach_idxs(t, [&](auto... idx){ printf("%s: %s[%lu %lu %lu] %p %e\n", loc, name, idx..., &t(idx...), t(idx...)); });
      } else if constexpr (T::ndim() == 2) {
        foreach_idxs(t, [&](auto... idx){ printf("%s: %s[%lu %lu] %p %e\n", loc, name, idx..., &t(idx...), t(idx...)); });
      }
      SYNCTHREADS();
    }

} // namespace mra

#endif // MRA_FUNCTIONS_H
