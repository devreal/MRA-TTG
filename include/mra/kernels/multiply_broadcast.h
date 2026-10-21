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
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"

/**
 * Kernels for multiplying every function of a set by one function, on
 * reconstructed (scaling-function) coefficients. Where a factor's tree ends
 * above the node, its values at the node's quadrature points are computed from
 * the leaf coefficients of the ancestor directly (fcube_for_mul), which is exact.
 */

namespace mra {

  template<Dimension NDIM>
  SCOPE size_type multiply_broadcast_tmp_size(size_type K) {
    return 3*size_type(std::pow(K, NDIM));         // two value cubes, workspace
  }

  namespace detail {

    /**
     * out_j = a * b_j at node `key`, where a and b_j are given by their scaling
     * coefficients: a from a_leaf (at key) if it holds them, else from a_inh (at
     * inh_keys[0], an ancestor of key), likewise b_j (inh_keys[1+j]). As
     * madness::FunctionImpl::do_mul: values at the quadrature points of the node,
     * product, back to coefficients.
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
      const Key<NDIM>* inh_keys,
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
      SHARED Key<NDIM> a_key, b_key;
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
          /* the leaf's own coefficients belong to key, inherited ones to their origin */
          const bool a_here = a_leaf.is_nonzero(0), b_here = b_leaf.is_nonzero(j);
          a = a_here ? a_leaf(0) : a_inh(0);
          b = b_here ? b_leaf(j) : b_inh(j);
          a_key = a_here ? key : inh_keys[0];
          b_key = b_here ? key : inh_keys[1 + j];
          out = out_view(j);
        }
        SYNCTHREADS();
        /* values at the quadrature points of the node, from coefficients at the node or at
         * an ancestor; a factor without coefficients (a leaf that is zero) is zero */
        if (a.data() == nullptr) acube = T(0);
        else fcube_for_mul<T, NDIM>(D, key, a_key, a, acube, phibar, phi, quad_x, K, workspace);
        if (b.data() == nullptr) bcube = T(0);
        else fcube_for_mul<T, NDIM>(D, key, b_key, b, bcube, phibar, phi, quad_x, K, workspace);
        foreach_idx(acube, [&](size_type i) { acube[i] *= bcube[i]; });
        SYNCTHREADS();
        transform(acube, phibar, out, workspace);
        const T scale = std::pow(T(0.5), T(0.5*NDIM*key.level())) * std::sqrt(D.template get_volume<T>());
        out *= scale;
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
    const Key<NDIM>* inh_keys,
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
    auto smem_size = std::max(static_cast<size_type>(K*K*NDIM*sizeof(T)), // phi of fcube_for_mul from an ancestor
                              mTxmq_shmem_size<T>(2*K));
    CALL_KERNEL(detail::multiply_broadcast_kernel, n_nonzero, thread_dims, smem_size, stream,
      (D, key, a_leaf, a_inh, b_leaf, b_inh, inh_keys, out, phibar, phi, quad_x, N, n_nonzero, K, tmp));
    checkSubmit();
  }


} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_BROADCAST_H
