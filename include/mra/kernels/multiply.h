#ifndef MRA_KERNELS_MULTIPLY_H
#define MRA_KERNELS_MULTIPLY_H

#include "mra/misc/maxk.h"
#include "mra/misc/key.h"
#include "mra/misc/domain.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"
#include "mra/tensor/child_slice.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type multiply_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    const size_type TWOK2NDIM = std::pow(2*K, NDIM);
    return 32*K2NDIM + 3 * TWOK2NDIM; // workspace, r1*8, cnodeA*8, cnodeB*8,
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& keyA,
      const Key<NDIM>& keyB,
      const TensorView<T, NDIM>& nodeA,
      const TensorView<T, NDIM>& nodeB,
      TensorView<T, NDIM+1>& cnodesA,
      TensorView<T, NDIM+1>& cnodesB,
      TensorView<T, NDIM>& cnodeR,
      TensorView<T, NDIM>& cnodeD,
      TensorView<T, NDIM>& nodeR,
      TensorView<T, NDIM+1>& r1,
      T* workspace,
      const TensorView<T, 2>& hgT,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phiT,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 1>& quad_x,
      size_type K)
    {
      Key<NDIM> target;
      if (keyA.level()>keyB.level()) target = keyA;
      else target = keyB;
      T scale;

      for (int i=0; i< keyA.num_children(); ++i){
        auto child = target.child_at(i);
        auto cnodeA = cnodesA(i);
        auto cnodeB = cnodesB(i);
        fcube_for_mul(D, child, keyB, nodeB, cnodeB, phibar, phi, quad_x, K, workspace);
        fcube_for_mul(D, child, keyA, nodeA, cnodeA, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*child.level())));
        cnodeB *= scale;
        cnodeA *= scale;
        cnodesA(i) = cnodeA;
        cnodesB(i) = cnodeB;
      }

      // fcube_for_mul() returns function values evaluated at quadrature points
      foreach_idx(cnodesA, [&](size_type i) {
        cnodesA[i] = cnodesA[i] * cnodesB[i];
      });

      // convert back to coeffs
      for (int i=0; i< keyA.num_children(); ++i){
        auto cnodeA = cnodesA(i);
        auto r = r1(i);
        transform(cnodeA, phibar, r, workspace);
      }

      // compress the result(r1 which is NDIM+1 tensorview) and store scaling functions to nodeR
      for (int i = 0; i<target.num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(target, K, i);
        const TensorView<T, NDIM>& in = r1(i);
        cnodeR(child_slice) = in;
      }

      transform<NDIM>(cnodeR, hgT, cnodeD, workspace);
      if (keyA.level() > 0) {
        auto child_slice = get_child_slice<NDIM>(target, K, 0);
        nodeR = cnodeD(child_slice);
      }
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
      const TensorView<T, 2>& hgT,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2> phiT,
      const TensorView<T, 2> phibar,
      const TensorView<T, 1>& quad_x,
      size_type N,
      size_type K)
    {
      SHARED TensorView<T, NDIM> nodeA, nodeB, nodeR, cnodesR, cnodesD;
      SHARED TensorView<T, NDIM+1> cnodesA, cnodesB, r1;
      SHARED T* workspace;
      size_type blockId = blockIdx.x;
      if (is_team_lead()){
        const size_type K2NDIM = std::pow(K, NDIM);
        const size_type TWO2NDIM = std::pow(2, NDIM);
        const size_type TWOK2NDIM = std::pow(2*K, NDIM);
        T* block_tmp = &tmp[blockId*multiply_tmp_size<NDIM>(K)];
        r1        = TensorView<T, NDIM+1>(&block_tmp[        0], TWO2NDIM, K, K, K);
        cnodesA   = TensorView<T, NDIM+1>(&block_tmp[ 8*K2NDIM], TWO2NDIM, K, K, K);
        cnodesB   = TensorView<T, NDIM+1>(&block_tmp[16*K2NDIM], TWO2NDIM, K, K, K);
        cnodesR   = TensorView<T, NDIM>(&block_tmp  [24*K2NDIM], 2*K, 2*K, 2*K);
        cnodesD   = TensorView<T, NDIM>(&block_tmp  [32*K2NDIM + TWOK2NDIM], 2*K, 2*K, 2*K);
        workspace = &block_tmp[32*K2NDIM + 2*TWOK2NDIM];
      }

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x){
        if (is_team_lead()) {
          nodeA = nodeA_view(fnid);
          nodeB = nodeB_view(fnid);
          nodeR = nodeR_view(fnid);
        }
        SYNCTHREADS();
        multiply_kernel_impl<T, NDIM>(D, keyA, keyB, nodeA, nodeB, cnodesA, cnodesB,
           cnodesR, cnodesD, nodeR, r1, workspace, hgT, phi, phiT, phibar, quad_x, K);
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
    const TensorView<T, 2>& hgT,
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
      (D, keyA, keyB, funcA, funcB, funcR, tmp, hgT, phi, phiT, phibar,
        quad_x, N, K));
    checkSubmit();
  }

  /* explicit instanatiation */
  extern template
  void submit_multiply_kernel<double, 3>(
    const Domain<3>& D,
    const Key<3>& keyA,
    const Key<3>& keyB,
    const TensorView<double, 3+1>& funcA,
    const TensorView<double, 3+1>& funcB,
    TensorView<double, 3+1>& funcR,
    const TensorView<double, 2>& hgT,
    const TensorView<double, 2>& phi,
    const TensorView<double, 2>& phiT,
    const TensorView<double, 2>& phibar,
    const TensorView<double, 1>& quad_x,
    size_type N,
    size_type K,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_H
