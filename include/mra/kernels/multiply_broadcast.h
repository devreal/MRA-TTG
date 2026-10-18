#ifndef MRA_KERNELS_MULTIPLY_BROADCAST_H
#define MRA_KERNELS_MULTIPLY_BROADCAST_H

#include <array>
#include <cmath>

#include "mra/misc/key.h"
#include "mra/misc/domain.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"

/**
 * Kernels for multiplying every function of a set by one function, on
 * reconstructed (scaling-function) coefficients. Both factors are given at the
 * level of the node: where a factor's tree ends above, its leaf coefficients
 * are refined to the level of the node first (refine_kernel), which is exact.
 */

namespace mra {

  template<Dimension NDIM>
  SCOPE size_type multiply_broadcast_tmp_size(size_type K) {
    return 3*size_type(std::pow(K, NDIM));         // two value cubes, workspace
  }

  template<Dimension NDIM>
  SCOPE size_type refine_tmp_size(size_type K) {
    return 3*size_type(std::pow(2*K, NDIM));       // padded parent, children, workspace
  }

  namespace detail {

    /**
     * out_j = a * b_j at node `key`, where a and b_j are given by their scaling
     * coefficients at that node: a from a_leaf if it holds them, else from a_inh,
     * likewise b_j. As madness::FunctionImpl::do_mul: values at the quadrature
     * points of the node, product, back to coefficients.
     * One block per non-zero output function.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void multiply_broadcast_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a_leaf,
      const concepts::TensorView<NDIM+1> auto a_inh,
      const concepts::TensorView<NDIM+1> auto b_leaf,
      const concepts::TensorView<NDIM+1> auto b_inh,
      concepts::TensorView<NDIM+1> auto out_view,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<1> auto quad_x,
      size_type N,
      size_type n_nonzero,
      size_type K,
      T* tmp)
    {
      SHARED DenseTensorView<const T, NDIM> a, b;
      SHARED DenseTensorView<T, NDIM> out, acube, bcube;
      SHARED T* workspace;
      SHARED size_type j;
      const size_type K2NDIM = std::pow(K, NDIM);
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          j = find_nth_nonzero(N, pos, out_view);
          T* block_tmp = &tmp[pos*multiply_broadcast_tmp_size<NDIM>(K)];
          acube = DenseTensorView<T, NDIM>(&block_tmp[0], K, K, K);
          bcube = DenseTensorView<T, NDIM>(&block_tmp[K2NDIM], K, K, K);
          workspace = &block_tmp[2*K2NDIM];
          a = a_leaf.is_nonzero(0) ? a_leaf(0) : a_inh(0);
          b = b_leaf.is_nonzero(j) ? b_leaf(j) : b_inh(j);
          out = out_view(j);
        }
        SYNCTHREADS();
        /* values at the quadrature points of the node (the same-level case of fcube_for_mul) */
        fcube_for_mul<T, NDIM>(D, key, key, a, acube, phibar, phi, quad_x, K, workspace);
        fcube_for_mul<T, NDIM>(D, key, key, b, bcube, phibar, phi, quad_x, K, workspace);
        foreach_idx(acube, [&](size_type i) { acube[i] *= bcube[i]; });
        SYNCTHREADS();
        transform(acube, phibar, out, workspace);
        const T scale = std::pow(T(0.5), T(0.5*NDIM*key.level())) * std::sqrt(D.template get_volume<T>());
        out *= scale;
        SYNCTHREADS(); // shared state is rewritten in the next iteration
      }
    }

    /**
     * Refine scaling coefficients at node `key` to its children (exact, the
     * two-scale relation with zero wavelet coefficients): out[c]_j for every
     * function j that is non-zero in out[0], taken from in_leaf if it holds
     * coefficients for j, else from in_inh. One block per refined function.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void refine_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto in_leaf,
      const concepts::TensorView<NDIM+1> auto in_inh,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto out_views,
      const concepts::TensorView<2> auto hg,
      size_type N,
      size_type n_nonzero,
      size_type K,
      T* tmp)
    {
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> padded, s;
      SHARED T* workspace;
      SHARED size_type j;
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          j = find_nth_nonzero(N, pos, out_views[0]);
          T* block_tmp = &tmp[pos*refine_tmp_size<NDIM>(K)];
          padded = DenseTensorView<T, NDIM>(&block_tmp[0], 2*K, 2*K, 2*K);
          s = DenseTensorView<T, NDIM>(&block_tmp[TWOK2NDIM], 2*K, 2*K, 2*K);
          workspace = &block_tmp[2*TWOK2NDIM];
          in = in_leaf.is_nonzero(j) ? in_leaf(j) : in_inh(j);
        }
        SYNCTHREADS();
        padded = 0.0;
        SYNCTHREADS();
        padded(get_child_slice<NDIM>(key, K, 0)) = in;
        SYNCTHREADS();
        transform(padded, hg, s, workspace);
        for (int c = 0; c < Key<NDIM>::num_children(); ++c) {
          auto child = out_views[c](j);
          child = s(get_child_slice<NDIM>(key, K, c));
        }
        SYNCTHREADS(); // shared state is rewritten in the next iteration
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_multiply_broadcast_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a_leaf,
    const concepts::TensorView<NDIM+1> auto& a_inh,
    const concepts::TensorView<NDIM+1> auto& b_leaf,
    const concepts::TensorView<NDIM+1> auto& b_inh,
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
    CALL_KERNEL(detail::multiply_broadcast_kernel, n_nonzero, thread_dims, smem_size, stream,
      (D, key, a_leaf, a_inh, b_leaf, b_inh, out, phibar, phi, quad_x, N, n_nonzero, K, tmp));
    checkSubmit();
  }

  template <typename T, Dimension NDIM>
  void submit_refine_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& in_leaf,
    const concepts::TensorView<NDIM+1> auto& in_inh,
    const concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& out_views,
    const concepts::TensorView<2> auto& hg,
    size_type N,
    size_type n_nonzero,
    size_type K,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CALL_KERNEL(detail::refine_kernel, n_nonzero, thread_dims, smem_size, stream,
      (key, in_leaf, in_inh, out_views, hg, N, n_nonzero, K, tmp));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_BROADCAST_H
