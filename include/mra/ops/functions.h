#ifndef MRA_FUNCTIONS_H
#define MRA_FUNCTIONS_H

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

#include <algorithm>

namespace mra {


    /// In given box return the truncation tolerance for given threshold
    template <typename T, Dimension NDIM>
    SCOPE T truncate_tol(const Key<NDIM>& key, const T thresh) {
        return thresh; // nothing clever for now
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
    SCOPE void distancesq(const Coordinate<T,1>& p, const TensorView<T,1>& q, T* rsq, size_type N) {
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
    SCOPE void distancesq(const Coordinate<T,2>& p, const TensorView<T,2>& q, T* rsq, size_type N) {
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
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,2>& q, T* rsq, size_type N) {
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
    SCOPE void distance(const Coordinate<T,1>& p, const TensorView<T,1>& q, T* rsq, size_type N) {
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
    SCOPE void distance(const Coordinate<T,2>& p, const TensorView<T,2>& q, T* rsq, size_type N) {
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
    SCOPE void distance(const Coordinate<T,3>& p, const TensorView<T,2>& q, T* rsq, size_type N) {
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
      /**
       * Reduce the contributions of each calling thread in a block into a single value.
       * On the host, we simply copy the result into the output value.
       * This requires block_size() elements in shared memory.
       * The block size can be controlled explicitly in case not all threads
       * contribute values.
       */
      template <typename T>
      SCOPE void reduce_block(const T input, T* output, size_type blocksize = block_size()) {
#ifdef HAVE_DEVICE_ARCH
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

        if (tid == 0) {
            *output = sdata[0];
        }
        SYNCTHREADS();
#else  // HAVE_DEVICE_ARCH
        *output = input;
#endif // HAVE_DEVICE_ARCH
      }
    }

    template <typename T, Dimension NDIM, typename accumulatorT>
    SCOPE void sumabssq(const TensorView<T, NDIM>& a, accumulatorT* sum) {
      accumulatorT s = 0.0;
      /* every thread computes a partial sum */
      foreach_idx(a, [&](size_type i) mutable {
        accumulatorT x = a[i];
        s += x*x;
      });
      detail::reduce_block(s, sum, std::min(a.size(), static_cast<size_type>(block_size())));
    }


    /// Compute Frobenius norm ... still needs specializing for complex
    template <typename T, Dimension NDIM, typename accumulatorT = std::decay_t<T>>
    SCOPE accumulatorT normf(const TensorView<T, NDIM>& a) {
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
