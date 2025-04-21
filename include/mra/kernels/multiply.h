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
    return 25*K2NDIM; // workspace, r1*8, cnodeA*8, cnodeB*8,
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& keyA,
      const Key<NDIM>& keyB,
      const TensorView<T, NDIM>& nodeA,
      const TensorView<T, NDIM>& nodeB,
      TensorView<T, NDIM+1>& cnodeA,
      TensorView<T, NDIM+1>& cnodeB,
      TensorView<T, NDIM>& nodeR,
      TensorView<T, NDIM+1>& r1,
      T* workspace,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phiT,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 1>& quad_x,
      size_type K)
    {
      if (keyA.level() > keyB.level()){
        T scale;
        for (int i=0; i< keyA.num_children(); ++i){
          fcube_for_mul(D, keyA.child_at(i), keyB.child_at(i), nodeB, cnodeB[i], phibar, phi, quad_x, K, workspace);
          scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyB.right_child().level())));
          cnodeB[i] *= scale;

          fcube_for_mul(D, keyA.child_at[i], keyA.child_at(i), nodeA, cnodeA[i], phibar, phi, quad_x, K, workspace);
          scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyA.right_child().level())));
          cnodeA[i] *= scale;
        }
      }
      else if (keyB.level() <= keyB.level()){
        T scale;
        for (int i=0; i< keyA.num_children(); ++i){
          fcube_for_mul(D, keyB.child_at(i), keyA.child_at(i), nodeA, cnodeA[i], phibar, phi, quad_x, K, workspace);
          scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyB.right_child().level())));
          cnodeB[i] *= scale;

          fcube_for_mul(D, keyB.child_at[i], keyB.child_at(i), nodeB, cnodeB[i], phibar, phi, quad_x, K, workspace);
          scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*keyA.right_child().level())));
          cnodeA[i] *= scale;
        }
      }

      // fcube_for_mul() returns function values evaluated at quadrature points
      for (int i=0; i< keyA.num_children(); ++i)
        foreach_idx(cnodeA[i], [&](size_type j) {
          cnodeA[i][j] = cnodeA[i][j] * cnodeB[i][j];
      });

      // convert back to coeffs
      foreach_idx(nodeA, [&](size_type i) {
        transform(cnodeA[i], phibar, r1[i], workspace);
    });

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
      SHARED TensorView<T, NDIM> nodeA, nodeB, nodeR, cnodesA, cnodesB, r1;
      SHARED T* workspace;
      size_type blockId = blockIdx.x;
      if (is_team_lead()){
        const size_type K2NDIM = std::pow(K, NDIM);
        T* block_tmp = &tmp[blockId*multiply_tmp_size<NDIM>(K)];
        r1        = TensorView<T, NDIM+1>(&block_tmp[       0], K);
        cnodesA   = TensorView<T, NDIM+1>(&block_tmp[8*K2NDIM], 8, K, K, K);
        cnodesB   = TensorView<T, NDIM+1>(&block_tmp[16*K2NDIM], 8, K, K, K);
        workspace = &block_tmp[24*K2NDIM];
      }

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x){
        if (is_team_lead()) {
          nodeA = nodeA_view(fnid);
          nodeB = nodeB_view(fnid);
          nodeR = nodeR_view(fnid);
        }
        SYNCTHREADS();
        multiply_kernel_impl<T, NDIM>(D, keyA, keyB, nodeA, nodeB, cnodesA, cnodesB,
           nodeR, r1, workspace, phi, phiT, phibar, quad_x, K);
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
