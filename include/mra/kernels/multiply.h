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
    return 4*K2NDIM; // workspace, r, r1, and r2
  }

  namespace detail {

    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_kernel_impl(
      const Domain<NDIM>& D,
      const TensorView<T, NDIM>& nodeA,
      const TensorView<T, NDIM>& nodeB,
      TensorView<T, NDIM>& nodeR,
      TensorView<T, NDIM>& r,
      TensorView<T, NDIM>& r1,
      TensorView<T, NDIM>& r2,
      T* workspace,
      const TensorView<T, 2>& phiT,
      const TensorView<T, 2>& phibar,
      Key<NDIM> key,
      size_type K)
    {
      // convert coeffs to function values
      transform(nodeA, phiT, r1, workspace);
      transform(nodeB, phiT, r2, workspace);
      const T scale = std::pow(T(2), T(0.5 * NDIM * key.level())) / std::sqrt(D.template get_volume<T>());

      foreach_idx(nodeA, [&](size_type i) {
          r[i] = scale * r1[i] * r2[i];
      });

      // convert back to coeffs
      transform(r, phibar, nodeR, workspace);
    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    multiply_kernel(
      const Domain<NDIM>& D,
      const TensorView<T, NDIM+1> nodeA_view,
      const TensorView<T, NDIM+1> nodeB_view,
      TensorView<T, NDIM+1> nodeR_view,
      T* tmp,
      const TensorView<T, 2> phiT,
      const TensorView<T, 2> phibar,
      Key<NDIM> key,
      size_type N,
      size_type K)
    {
      SHARED TensorView<T, NDIM> nodeA, nodeB, nodeR, r1, r2, r;
      SHARED T* workspace;
      size_type blockId = blockIdx.x;
      if (is_team_lead()){
        const size_type K2NDIM = std::pow(K, NDIM);
        T* block_tmp = &tmp[blockId*multiply_tmp_size<NDIM>(K)];
        r         = TensorView<T, NDIM>(&block_tmp[       0], K);
        r1        = TensorView<T, NDIM>(&block_tmp[  K2NDIM], K);
        r2        = TensorView<T, NDIM>(&block_tmp[2*K2NDIM], K);
        workspace = &block_tmp[3*K2NDIM];
      }

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x){
        if (is_team_lead()) {
          nodeA = nodeA_view(fnid);
          nodeB = nodeB_view(fnid);
          nodeR = nodeR_view(fnid);
        }
        SYNCTHREADS();
        multiply_kernel_impl<T, NDIM>(D, nodeA, nodeB, nodeR, r, r1, r2, workspace,
                                      phiT, phibar, key, K);
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_multiply_kernel(
    const Domain<NDIM>& D,
    const TensorView<T, NDIM+1>& funcA,
    const TensorView<T, NDIM+1>& funcB,
    TensorView<T, NDIM+1>& funcR,
    const TensorView<T, 2>& phiT,
    const TensorView<T, 2>& phibar,
    size_type N,
    size_type K,
    const Key<NDIM>& key,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::multiply_kernel, N, thread_dims, 0, stream,
      (D, funcA, funcB, funcR, tmp,
        phiT, phibar, key, N, K));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_H
