#ifndef MRA_KERNELS_NORM_H
#define MRA_KERNELS_NORM_H

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void norm_kernel_impl(
      const TensorView<T, NDIM>& n,
      T* result_norm,
      const std::array<T, Key<NDIM>::num_children()>& child_norms,
      size_type blockid,
      size_type K)
    {
      const bool is_t0 = (0 == thread_id());
      T norm = normf(n);
      if (is_t0) {
        /* thread 0 adds the child norms and publishes the result */
        for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
          norm += child_norms[i];
        }
        *result_norm = norm;
      }
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void norm_kernel(
      const TensorView<T, NDIM+1> node,
      T* result_norms,
      std::array<const T*, Key<NDIM>::num_children()> child_norms,
      size_type N,
      size_type K,
      const Key<NDIM>& key)
    {
      const bool is_t0 = (0 == thread_id());
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      SHARED TensorView<T, NDIM> n;
      SHARED std::array<T, Key<NDIM>::num_children()> block_child_norms;
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (is_t0) {
          n = node(blockid);
          for (size_type i = 0; i < Key<NDIM>::num_children(); ++i) {
            block_child_norms[i] = (child_norms[i] != nullptr) ? child_norms[i][blockid] : T(0.0);
          }
        }
        SYNCTHREADS();
        norm_kernel_impl<T, NDIM>(n, &result_norms[blockid], block_child_norms, blockid, K);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_norm_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    const TensorView<T, NDIM+1>& in,
    TensorView<T, 1>& result_norms,
    std::array<const T*, Key<NDIM>::num_children()>& child_norms,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::norm_kernel, N, thread_dims, 0, stream,
        (in, result_norms.data(), child_norms, N, K, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_NORM_H
