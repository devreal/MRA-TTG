#ifndef MRA_KERNELS_NORM_H
#define MRA_KERNELS_NORM_H

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void norm_kernel_impl(
      const concepts::TensorView<NDIM> auto& n,
      T* result_norm,
      const std::array<T, Key<NDIM>::num_children()>& child_norms,
      size_type blockid,
      size_type K,
      bool is_root)
    {
      const bool is_t0 = (0 == thread_id());
      /* norms are passed up the tree squared (a sum over all nodes of the
       * compressed tree) and only the root takes the square root */
      SHARED T node_normsq;
      sumabssq(n, &node_normsq);
      /* wait for all threads to contribute */
      SYNCTHREADS();
      if (is_t0) {
        /* thread 0 adds the children's squared norms and publishes the result */
        T normsq = node_normsq;
        for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
          normsq += child_norms[i];
        }
        *result_norm = is_root ? std::sqrt(normsq) : normsq;
      }
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void norm_kernel(
      const concepts::TensorView<NDIM+1> auto node,
      T* result_norms,
      std::array<const T*, Key<NDIM>::num_children()> child_norms,
      size_type N,
      size_type n_nonzero,
      size_type K,
      const Key<NDIM> key)
    {
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      SHARED DenseTensorView<const T, NDIM> n;
      SHARED std::array<T, Key<NDIM>::num_children()> block_child_norms;
      SHARED size_type blockid;
      SHARED bool is_zero;
      // One block per function, including functions that are zero at this node:
      // their children can still carry coefficients, whose norms must be passed up.
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        if (is_team_lead()) {
          blockid = pos;
          is_zero = node.is_zero(blockid);
          if (!is_zero) n = node(blockid);
          for (size_type i = 0; i < Key<NDIM>::num_children(); ++i) {
            block_child_norms[i] = (child_norms[i] != nullptr) ? child_norms[i][blockid] : T(0.0);
          }
        }
        SYNCTHREADS();
        if (is_zero) {
          if (is_team_lead()) {
            T normsq = 0.0;
            for (int i = 0; i < Key<NDIM>::num_children(); ++i) normsq += block_child_norms[i];
            result_norms[blockid] = (key.level() == 0) ? std::sqrt(normsq) : normsq;
          }
        } else {
          norm_kernel_impl<T, NDIM>(n, &result_norms[blockid], block_child_norms, blockid, K, key.level() == 0);
        }
        SYNCTHREADS(); // shared state is rewritten in the next iteration
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_norm_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type n_nonzero,
    size_type K,
    const concepts::TensorView<NDIM+1> auto& in,
    concepts::TensorView<1> auto& result_norms,
    std::array<const T*, Key<NDIM>::num_children()>& child_norms,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::norm_kernel, N, thread_dims, 0, stream,
        (in, result_norms.data(), child_norms, N, n_nonzero, K, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_NORM_H
