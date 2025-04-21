#ifndef MRA_KERNELS_MULTIPLY_H
#define MRA_KERNELS_MULTIPLY_H

#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type multiply_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    return 7*K2NDIM; // workspace, r1, and r2, lcnodeA, rcnodeA, lcnodeB, rcnodeB,
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& keyA,
      const Key<NDIM>& keyB,
      const TensorView<T, NDIM>& nodeA,
      const TensorView<T, NDIM>& nodeB,
      TensorView<T, NDIM>& lcnodeA,
      TensorView<T, NDIM>& rcnodeA,
      TensorView<T, NDIM>& lcnodeB,
      TensorView<T, NDIM>& rcnodeB,
      TensorView<T, NDIM>& nodeR,
      TensorView<T, NDIM>& r1,
      TensorView<T, NDIM>& r2,
      T* workspace,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phiT,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 1>& quad_x,
      size_type K)
    {
      if (keyA.level() > keyB.level()){
        fcube_for_mul(D, keyA, keyB.right_child(), nodeB, rcnodeB, phibar, phi, quad_x, K, workspace);
        T scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyB.right_child().level())));
        rcnodeB *= scale;
        fcube_for_mul(D, keyA, keyB.left_child(), nodeB, lcnodeB, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyB.left_child().level())));
        lcnodeB *= scale;
        fcube_for_mul(D, keyA, keyA.right_child(), nodeA, rcnodeA, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyA.right_child().level())));
        rcnodeA *= scale;
        fcube_for_mul(D, keyA, keyA.left_child(), nodeA, lcnodeA, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyA.left_child().child.level())));
        lcnodeA *= scale;
      }
      else if (keyB.level() <= keyB.level()){
        fcube_for_mul(D, keyB, keyA.right_child(), nodeA, rcnodeA, phibar, phi, quad_x, K, workspace);
        T scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyB.right_child().level())));
        rcnodeA *= scale;
        fcube_for_mul(D, keyB, keyA.left_child(), nodeA, lcnodeA, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyB.left_child().level())));
        lcnodeA *= scale;
        fcube_for_mul(D, keyB, keyB.right_child(), nodeB, rcnodeB, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyA.right_child().level())));
        rcnodeB *= scale;
        fcube_for_mul(D, keyB, keyB.left_child(), nodeB, lcnodeB, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyA.left_child().child.level())));
        lcnodeB *= scale;

        // fcube_for_mul() returns function values evaluated at quadrature points
      }
      foreach_idx(nodeA, [&](size_type i) {
        lcnodeA[i] = lcnodeA[i] * lcnodeB[i];
        rcnodeA[i] = rcnodeA[i] * rcnodeB[i];
    });

      // convert back to coeffs
      transform(lcnodeA, phibar, r1, workspace);
      transform(rcnodeA, phibar, r2, workspace);

      // compress the result to nodeR
    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    multiply_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM>& keyA,
      const Key<NDIM>& keyB,
      const TensorView<T, NDIM+1> nodeA_view,
      const TensorView<T, NDIM+1> nodeB_view,
      TensorView<T, NDIM+1> nodeR_view,
      T* tmp,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2> phiT,
      const TensorView<T, 2> phibar,
      const TensorView<T, 1>& quad_x,
      size_type N,
      size_type K)
    {
      SHARED TensorView<T, NDIM> nodeA, nodeB, nodeR, lcnodeA, rcnodeA, lcnodeB, rcnodeB, r1, r2;
      SHARED T* workspace;
      size_type blockId = blockIdx.x;
      if (is_team_lead()){
        const size_type K2NDIM = std::pow(K, NDIM);
        T* block_tmp = &tmp[blockId*multiply_tmp_size<NDIM>(K)];
        r1        = TensorView<T, NDIM>(&block_tmp[       0], K);
        r2        = TensorView<T, NDIM>(&block_tmp[  K2NDIM], K);
        lcnodeA   = TensorView<T, NDIM>(&block_tmp[2*K2NDIM], K);
        rcnodeA   = TensorView<T, NDIM>(&block_tmp[3*K2NDIM], K);
        lcnodeB   = TensorView<T, NDIM>(&block_tmp[4*K2NDIM], K);
        rcnodeB   = TensorView<T, NDIM>(&block_tmp[5*K2NDIM], K);
        workspace = &block_tmp[6*K2NDIM];
      }

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x){
        if (is_team_lead()) {
          nodeA = nodeA_view(fnid);
          nodeB = nodeB_view(fnid);
          nodeR = nodeR_view(fnid);
        }
        SYNCTHREADS();
        multiply_kernel_impl<T, NDIM>(D, keyA, keyB, nodeA, nodeB, lcnodeA,
          rcnodeA, lcnodeB, rcnodeB, nodeR, r1, r2, workspace, phi, phiT,
          phibar, quad_x, K);
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_multiply_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& keyA,
    const Key<NDIM>& keyB,
    const TensorView<T, NDIM+1>& funcA,
    const TensorView<T, NDIM+1>& funcB,
    TensorView<T, NDIM+1>& funcR,
    const TensorView<T, 2>& phi,
    const TensorView<T, 2>& phiT,
    const TensorView<T, 2>& phibar,
    const TensorView<T, 1>& quad_x,
    size_type N,
    size_type K,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::multiply_kernel, N, thread_dims, 0, stream,
      (D, keyA, keyB, funcA, funcB, funcR, tmp, phi, phiT, phibar,
        quad_x, N, K));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_H
