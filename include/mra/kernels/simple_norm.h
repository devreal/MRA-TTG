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
      size_type N)
    {
      using T = typename std::remove_reference_t<decltype(node)>::value_type;
      SHARED DenseTensorView<const T, NDIM> n;
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        if (node.is_zero(pos)) {
          result_norms[pos] = T(0);
          continue; // skip zero-function entries
        }
        if (is_team_lead()) {
          n = node(pos);
        }
        SYNCTHREADS();
        T norm = normf(n);
        if (is_team_lead()) {
          result_norms[pos] = norm;
        }
      }
    }
  } // namespace detail


  template <Dimension NDIM>
  void submit_simple_norm_kernel(
    Key<NDIM> key,
    const concepts::TensorView<NDIM+1> auto&& in,
    size_type N,
    concepts::TensorView<1> auto&& result_norms)
  {
    /* simple norm calculation can use as many threads as are available */
    CALL_KERNEL(detail::simple_norm_kernel, N, MAX_THREADS_PER_BLOCK, 0, ttg::device::current_stream(),
        (key, in, result_norms, N));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_SIMPLE_NORM_H