#ifndef MRA_KERNELS_TRANSFORM_FUNCTIONS_H
#define MRA_KERNELS_TRANSFORM_FUNCTIONS_H

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/ops/functions.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"

/**
 * Linear combination of a set of functions at one node of the tree:
 *   out_j = sum_i U(i,j) in_i
 * (cf. madness::transform). U is N x M, row-major. Being linear, this applies
 * to the coefficients of compressed and reconstructed nodes alike.
 */

namespace mra {
  namespace detail {

    /**
     * One block per non-zero output function. Input functions that are zero at
     * this node, or whose coefficient U(i,j) is zero, are skipped.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void transform_functions_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto in_view,
      concepts::TensorView<NDIM+1> auto out_view,
      const T* U,
      size_type N,
      size_type M,
      size_type n_nonzero)
    {
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> out;
      SHARED size_type j;
      SHARED bool skip;
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          // out_view has exactly n_nonzero non-zero entries
          j = find_nth_nonzero(M, pos, out_view);
          out = out_view(j);
        }
        SYNCTHREADS();
        foreach_idx(out, [&](size_type k) { out[k] = T(0); });
        for (size_type i = 0; i < N; ++i) {
          if (is_team_lead()) {
            skip = in_view.is_zero(i) || U[i*M + j] == T(0);
            if (!skip) in = in_view(i);
          }
          SYNCTHREADS();
          if (!skip) {
            const T u = U[i*M + j];
            /* every thread updates the same elements in each iteration */
            foreach_idx(out, [&](size_type k) { out[k] += u*in[k]; });
          }
          SYNCTHREADS();
        }
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_transform_functions_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& in,
    concepts::TensorView<NDIM+1> auto& out,
    const T* U,
    size_type N,
    size_type M,
    size_type n_nonzero,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::transform_functions_kernel, n_nonzero, thread_dims, 0, stream,
                (key, in, out, U, N, M, n_nonzero));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_TRANSFORM_FUNCTIONS_H
