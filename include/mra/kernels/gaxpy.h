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
      const TensorView<T, NDIM>& nodeA,
      const TensorView<T, NDIM>& nodeB,
      TensorView<T, NDIM>& nodeR,
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
      const TensorView<T, NDIM+1> nodeA_view,
      const TensorView<T, NDIM+1> nodeB_view,
      TensorView<T, NDIM+1> nodeR_view,
      const T scalarA,
      const T scalarB,
      size_type N,
      const Key<NDIM> key)
    {
      SHARED TensorView<T, NDIM> nodeA, nodeB, nodeR;
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
    const TensorView<T, NDIM+1>& funcA,
    const TensorView<T, NDIM+1>& funcB,
    TensorView<T, NDIM+1>& funcR,
    const T scalarA,
    const T scalarB,
    size_type N,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::gaxpy_kernel, N, thread_dims, 0, stream,
      (funcA, funcB, funcR, scalarA, scalarB, N, key));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_GAXPY_H
