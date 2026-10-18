#ifndef MRA_KERNELS_SQUARE_H
#define MRA_KERNELS_SQUARE_H

#include <cmath>

#include "mra/misc/key.h"
#include "mra/misc/domain.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"

/**
 * Pointwise square of the scaling coefficients of a node, for every function of
 * a set (cf. madness::FunctionImpl::square_inplace, without autorefinement).
 */

namespace mra {

  template<Dimension NDIM>
  SCOPE size_type square_tmp_size(size_type K) {
    return 2*size_type(std::pow(K, NDIM));         // value cube, workspace
  }

  namespace detail {

    /**
     * out_j = in_j^2 at node `key`: values at the quadrature points of the node,
     * squared, back to coefficients. One block per non-zero output function.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void square_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto in_view,
      concepts::TensorView<NDIM+1> auto out_view,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<1> auto quad_x,
      size_type N,
      size_type n_nonzero,
      size_type K,
      T* tmp)
    {
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> out, cube;
      SHARED T* workspace;
      SHARED size_type j;
      const size_type K2NDIM = std::pow(K, NDIM);
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          j = find_nth_nonzero(N, pos, out_view);
          T* block_tmp = &tmp[pos*square_tmp_size<NDIM>(K)];
          cube = DenseTensorView<T, NDIM>(&block_tmp[0], K, K, K);
          workspace = &block_tmp[K2NDIM];
          in = in_view(j);
          out = out_view(j);
        }
        SYNCTHREADS();
        fcube_for_mul<T, NDIM>(D, key, key, in, cube, phibar, phi, quad_x, K, workspace);
        foreach_idx(cube, [&](size_type i) { cube[i] *= cube[i]; });
        SYNCTHREADS();
        transform(cube, phibar, out, workspace);
        const T scale = std::pow(T(0.5), T(0.5*NDIM*key.level())) * std::sqrt(D.template get_volume<T>());
        out *= scale;
        SYNCTHREADS(); // shared state is rewritten in the next iteration
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_square_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& in,
    concepts::TensorView<NDIM+1> auto& out,
    const concepts::TensorView<2> auto& phibar,
    const concepts::TensorView<2> auto& phi,
    const concepts::TensorView<1> auto& quad_x,
    size_type N,
    size_type n_nonzero,
    size_type K,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CALL_KERNEL(detail::square_kernel, n_nonzero, thread_dims, smem_size, stream,
      (D, key, in, out, phibar, phi, quad_x, N, n_nonzero, K, tmp));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_SQUARE_H
