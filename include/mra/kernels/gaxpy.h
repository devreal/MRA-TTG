#ifndef MRA_KERNELS_GAXPY_H
#define MRA_KERNELS_GAXPY_H

#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void gaxpy_kernel_impl(
      const concepts::TensorView<NDIM> auto& nodeA,
      const concepts::TensorView<NDIM> auto& nodeB,
      concepts::TensorView<NDIM> auto& nodeR,
      const T scalarA,
      const T scalarB)
    {
      foreach_idx(nodeR, [&](size_type i) {
        nodeR[i] = scalarA*nodeA[i] + scalarB*nodeB[i];
      });
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gaxpy_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto nodeA_view,
      const concepts::TensorView<NDIM+1> auto nodeB_view,
      concepts::TensorView<NDIM+1> auto nodeR_view,
      const T scalarA,
      const T scalarB,
      size_type N)
    {
      SHARED DenseTensorView<T, NDIM> nodeA, nodeB, nodeR;
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (is_team_lead()) {
          nodeA = nodeA_view(blockid);
          nodeB = nodeB_view(blockid);
          nodeR = nodeR_view(blockid);
        }
        SYNCTHREADS();
        gaxpy_kernel_impl<T, NDIM>(nodeA, nodeB, nodeR, scalarA, scalarB);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_gaxpy_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& funcA,
    const concepts::TensorView<NDIM+1> auto& funcB,
    concepts::TensorView<NDIM+1> auto& funcR,
    const T scalarA,
    const T scalarB,
    size_type N,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::gaxpy_kernel, N, thread_dims, 0, stream,
      (key, funcA, funcB, funcR, scalarA, scalarB, N));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_GAXPY_H
