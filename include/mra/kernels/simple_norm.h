#ifndef MRA_KERNELS_SIMPLE_NORM_H
#define MRA_KERNELS_SIMPLE_NORM_H

#include "platform.h"
#include "types.h"
#include "tensorview.h"
#include "key.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void simple_norm_kernel_impl(
      const TensorView<T, NDIM>& n,
      T* result_norm)
    {
      const bool is_t0 = (0 == thread_id());
      T norm = normf(n);
      if (is_t0) {
        *result_norm = norm;
      }
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void simple_norm_kernel(
      Key<NDIM> key,
      const TensorView<T, NDIM+1> node,
      TensorView<T, 1> result_norms,
      size_type N)
    {
      const bool is_t0 = (0 == thread_id());
      SHARED TensorView<T, NDIM> n;
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (is_t0) {
          n = node(blockid);
        }
        SYNCTHREADS();
        simple_norm_kernel_impl<T, NDIM>(n, &result_norms[blockid]);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_simple_norm_kernel(
    Key<NDIM> key,
    const TensorView<T, NDIM+1>& in,
    size_type N,
    TensorView<T, 1> result_norms)
  {
    /* simple norm calculation can use as many threads as are available */
    CALL_KERNEL(detail::simple_norm_kernel, N, MAX_THREADS_PER_BLOCK, 0, ttg::device::current_stream(),
        (key, in, result_norms, N));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_SIMPLE_NORM_H