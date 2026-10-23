#ifndef MRA_KERNELS_CONCAT_H
#define MRA_KERNELS_CONCAT_H

#include <type_traits>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

/**
 * Concatenate the coefficients of two nodes of sets of functions into one node.
 */

namespace mra {

  namespace detail {

    /**
     * out_j = a_j for j < NA, b_{j-NA} otherwise, for every function j that is
     * non-zero in out. One block per non-zero output function.
     */
    template <Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void concat_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a,
      const concepts::TensorView<NDIM+1> auto b,
      concepts::TensorView<NDIM+1> auto out_view,
      size_type NA,
      size_type N,
      size_type n_nonzero)
    {
      using T = std::remove_const_t<typename std::remove_reference_t<decltype(out_view)>::value_type>;
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> out;
      SHARED size_type j;
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          j = find_nth_nonzero(N, pos, out_view);
          in = (j < NA) ? a(j) : b(j - NA);
          out = out_view(j);
        }
        SYNCTHREADS();
        out = in;
        SYNCTHREADS(); // shared state is rewritten in the next iteration
      }
    }
  } // namespace detail

  template <Dimension NDIM>
  void submit_concat_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a,
    const concepts::TensorView<NDIM+1> auto& b,
    concepts::TensorView<NDIM+1> auto& out,
    size_type NA,
    size_type N,
    size_type n_nonzero,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::concat_kernel, n_nonzero, thread_dims, 0, stream, (key, a, b, out, NA, N, n_nonzero));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_CONCAT_H
