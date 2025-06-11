#ifndef MRA_KERNELS_TRANSFORM_H
#define MRA_KERNELS_TRANSFORM_H

#include <cstdlib>
#include "mra/ops/mxm.h"
#include "mra/ops/inner.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/cycledim.h"
#include "mra/tensor/tensorview.h"

//#define MRA_CUDA_ENABLE_SHARED_TRANSFORM

namespace mra {


#if defined(MRA_CUDA_ENABLE_SHARED_TRANSFORM) && defined(MRA_ENABLE_CUDA) && defined(MRA_HAVE_CUBLASDX)
  template <Dimension NDIM, typename T>
  SCOPE bool transform_shared(
    const TensorView<T, NDIM>& t,
    const TensorView<T, 2>& c,
    TensorView<T, NDIM>& result,
    T* workspace)
  {
    if ((2*t.size() + c.size() + result.size()) > mTxmq_shmem_size<T>(c.dim(0))) {
      return false; // cannot put everything in shared memory
    }

    extern SHARED __align__(16) T smem[];
    T *pc = &smem[0];
    T *t0 = &smem[c.size() ];
    T *t1 = &smem[c.size() + t.size()];
    mra::foreach_idx(c, [&](size_type i) {
      pc[i] = c[i];
    });
    mra::foreach_idx(t, [&](size_type i) {
      t0[i] = t[i];
    });
    //T *t0=workspace, *t1=result.data();
    const size_type dimj = c.dim(1);
    size_type dimi = 1;
    for (size_type n=1; n<t.ndim(); ++n) dimi *= dimj;
    for (size_type n=0; n<t.ndim(); ++n) {
      mTxmq(dimi, dimj, dimj, t1, t0, pc, true);
      std::swap(t0,t1);
    }
    mra::foreach_idx(result, [&](size_type i) {
      result[i] = t0[i];
    });
    /* no need to synchronize here, mTxmq synchronizes */
    return true;
  }
#else // defined(MRA_ENABLE_CUDA)
  template <Dimension NDIM, typename T>
  SCOPE bool transform_shared(
    const TensorView<T, NDIM>& t,
    const TensorView<T, 2>& c,
    TensorView<T, NDIM>& result,
    T* workspace) {
      return false;
  }
#endif // defined(MRA_ENABLE_CUDA)

  template <Dimension NDIM, typename T>
  SCOPE void transform(
    const TensorView<T, NDIM>& t,
    const TensorView<T, 2>& c,
    TensorView<T, NDIM>& result,
    T* workspace) {
    if (transform_shared(t, c, result, workspace)) return;
    const T* pc = c.data();
    T *t0=workspace, *t1=result.data();
    if (t.ndim() & 0x1) std::swap(t0,t1);
    const size_type dimj = c.dim(1);
    size_type dimi = 1;
    for (size_type n=1; n<t.ndim(); ++n) dimi *= dimj;
    mTxmq(dimi, dimj, dimj, t0, t.data(), pc);
    for (size_type n=1; n<t.ndim(); ++n) {
      mTxmq(dimi, dimj, dimj, t1, t0, pc);
      std::swap(t0,t1);
    }
    /* no need to synchronize here, mTxmq synchronizes */
  }

  template <typename T, Dimension NDIM>
  SCOPE void transform_dir(
    const TensorView<T, NDIM>& node,
    const TensorView<T, 2>& op,
    TensorView<T, NDIM>& tmp_result,
    TensorView<T, NDIM>& result,
    size_type axis) {
      if (axis == 0){
        detail::inner(op, node, result, 0, axis);
      }
      else if (axis == node.ndim()-1){
        detail::inner(node, op, result, axis, 0);
      }
      else {
        tmp_result = 0.0; // reset tmp_result
        detail::inner(node, op, tmp_result, axis, 0);
        detail::cycledim(tmp_result, result, 1, axis, -1); // copy to make contiguous
      }
    }

  template <typename T, Dimension NDIM, std::size_t ARRDIM = NDIM>
  SCOPE void general_transform(
    const TensorView<T, NDIM>& t,
    const std::array<TensorView<T, 2>, ARRDIM>& c,
    TensorView<T, NDIM>& result,
    TensorView<T, NDIM>& result_tmp)
    {
      if constexpr (NDIM % 2) {
        // make sure result and result_tmp
        // end up pointing to the same memory
        std::swap(result, result_tmp);
      }
      result = t; // prime result
      for (size_type i = 0; i < NDIM; ++i){
        // inner accumulates but we're passing
        // TODO: make accumulation optional?
        result_tmp = 0;
        detail::inner(result, c[i], result_tmp, 0, 0);
        std::swap(result, result_tmp);
      }
    }

} // namespace mra

#endif // MRA_TRANSFORM_H
