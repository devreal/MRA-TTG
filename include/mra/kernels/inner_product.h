#ifndef MRA_KERNELS_INNER_PRODUCT_H
#define MRA_KERNELS_INNER_PRODUCT_H

#include <array>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/ops/functions.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"

/**
 * Kernels for inner products between sets of functions in compressed form.
 *
 * Each kernel processes one node of the tree: it computes the node's own
 * contribution and adds the partial results already reduced over the node's
 * children, so that the result at the root is the inner product over the
 * whole tree. Children that do not exist pass a nullptr.
 *
 * In compressed (non-standard=false) form only the root carries scaling
 * coefficients, so the inner product of two functions is the sum over all
 * nodes of the dot product of their coefficients.
 */

namespace mra {
  namespace detail {

    /**
     * result[i] = <a_i|b_i> at this node + sum over children.
     * One block per function.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void inner_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a_view,
      const concepts::TensorView<NDIM+1> auto b_view,
      T* result,
      std::array<const T*, Key<NDIM>::num_children()> child_results,
      size_type N)
    {
      SHARED DenseTensorView<const T, NDIM> a, b;
      SHARED bool have_both;
      SHARED T node_sum;
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        if (is_team_lead()) {
          have_both = a_view.is_nonzero(pos) && b_view.is_nonzero(pos);
          if (have_both) {
            a = a_view(pos);
            b = b_view(pos);
          }
        }
        SYNCTHREADS();
        if (have_both) {
          sumprod(a, b, &node_sum);
        }
        SYNCTHREADS();
        if (is_team_lead()) {
          T r = have_both ? node_sum : T(0);
          for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
            if (child_results[c] != nullptr) r += child_results[c][pos];
          }
          result[pos] = r;
        }
        SYNCTHREADS(); // shared state is rewritten in the next iteration
      }
    }

    /**
     * result[i*M+j] = <a_i|b_j> at this node + sum over children.
     * One block per row i.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void matrix_inner_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a_view,
      const concepts::TensorView<NDIM+1> auto b_view,
      T* result,
      std::array<const T*, Key<NDIM>::num_children()> child_results,
      size_type N,
      size_type M)
    {
      SHARED DenseTensorView<const T, NDIM> a, b;
      SHARED bool a_nonzero, b_nonzero;
      SHARED T node_sum;
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        T* row = result + pos*M;
        if (is_team_lead()) {
          a_nonzero = a_view.is_nonzero(pos);
          if (a_nonzero) a = a_view(pos);
        }
        /* start the row from the sum over the children */
        for (size_type j = thread_id(); j < M; j += block_size()) {
          T r = 0;
          for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
            if (child_results[c] != nullptr) r += child_results[c][pos*M + j];
          }
          row[j] = r;
        }
        SYNCTHREADS();
        if (a_nonzero) {
          for (size_type j = 0; j < M; ++j) {
            if (is_team_lead()) {
              b_nonzero = b_view.is_nonzero(j);
              if (b_nonzero) b = b_view(j);
            }
            SYNCTHREADS();
            if (b_nonzero) {
              sumprod(a, b, &node_sum);
              SYNCTHREADS();
              if (is_team_lead()) row[j] += node_sum;
            }
            SYNCTHREADS();
          }
        }
        SYNCTHREADS(); // shared state is rewritten in the next iteration
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_inner_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a,
    const concepts::TensorView<NDIM+1> auto& b,
    T* result,
    const std::array<const T*, Key<NDIM>::num_children()>& child_results,
    size_type N,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::inner_kernel, N, thread_dims, 0, stream,
                (key, a, b, result, child_results, N));
    checkSubmit();
  }

  template <typename T, Dimension NDIM>
  void submit_matrix_inner_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a,
    const concepts::TensorView<NDIM+1> auto& b,
    T* result,
    const std::array<const T*, Key<NDIM>::num_children()>& child_results,
    size_type N,
    size_type M,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::matrix_inner_kernel, N, thread_dims, 0, stream,
                (key, a, b, result, child_results, N, M));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_INNER_PRODUCT_H
