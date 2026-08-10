#ifndef MRA_KERNELS_SIMPLE_NORM_H
#define MRA_KERNELS_SIMPLE_NORM_H

#include "mra/misc/platform.h"
#include "mra/misc/types.h"
#include "mra/misc/key.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {
    template <Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void simple_norm_kernel(
      Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto node,
      concepts::TensorView<1> auto result_norms,
      size_type N,
      size_type n_nonzero)
    {
      using T = typename std::remove_reference_t<decltype(node)>::value_type;
      SHARED DenseTensorView<const T, NDIM> n;
      SHARED size_type blockid;
      // Zero-function entries of result_norms are pre-filled with 0.0 by the
      // caller (see e.g. mra/tasks/convolution.h) since no block touches
      // them here.
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          // node has exactly n_nonzero non-zero entries, so this always
          // finds a valid function id -- see submit_simple_norm_kernel.
          blockid = find_nth_nonzero(N, pos, node);
          n = node(blockid);
        }
        SYNCTHREADS();
        T norm = normf(n);
        if (is_team_lead()) {
          result_norms[blockid] = norm;
        }
      }
    }
  } // namespace detail


  template <Dimension NDIM>
  void submit_simple_norm_kernel(
    Key<NDIM> key,
    const concepts::TensorView<NDIM+1> auto&& in,
    size_type N,
    size_type n_nonzero,
    concepts::TensorView<1> auto&& result_norms)
  {
    if (n_nonzero == 0) return; // nothing to do; a 0-block launch trips "invalid argument" on some CUDA configs
    /* simple norm calculation can use as many threads as are available */
    CALL_KERNEL(detail::simple_norm_kernel, n_nonzero, MAX_THREADS_PER_BLOCK, 0, ttg::device::current_stream(),
        (key, in, result_norms, N, n_nonzero));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_SIMPLE_NORM_H