#ifndef MRA_KERNELS_TRANSFORM_H
#define MRA_KERNELS_TRANSFORM_H

#if !defined(MRA_JIT_COMPILE)
#include <cstdlib>
#endif // !defined(MRA_JIT_COMPILE)
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
    const concepts::TensorView<NDIM> auto& t,
    const concepts::TensorView<2> auto& c,
    concepts::TensorView<NDIM> auto& result,
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
  template <typename T>
  SCOPE bool transform_shared(
    const concepts::TensorView auto& t,
    const concepts::TensorView<2> auto& c,
    concepts::TensorView auto& result,
    T* workspace) {
      return false;
  }
#endif // defined(MRA_ENABLE_CUDA)

  template <typename T>
  SCOPE void transform(
    const concepts::TensorView auto& t,
    const concepts::TensorView<2> auto& c,
    concepts::TensorView auto& result,
    T* workspace)
  {
    static_assert(std::decay_t<decltype(t)>::ndim() == std::decay_t<decltype(result)>::ndim(),
                  "Input and output tensor views must have the same number of dimensions.");
    if (transform_shared(t, c, result, workspace)) return;
    const auto* pc = c.data();
    T* t0=workspace, *t1=result.data();
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

  SCOPE void transform_dir(
    const concepts::TensorView auto& node,
    const concepts::TensorView<2> auto& op,
    concepts::TensorView auto& tmp,
    concepts::TensorView auto& result,
    size_type axis)
  {
      if (axis == 0){
        result = 0.0; // start from 0
        detail::inner(op, node, result, 0, axis);
      }
      else if (axis == node.ndim()-1){
        result = 0.0; // start from 0
        detail::inner(node, op, result, axis, 0);
      }
      else {
        tmp = 0.0; // start from 0
        //std::cout << "transform_dir axis " << axis << " node " << node << std::endl;
        detail::inner(node, op, tmp, axis, 0);
        //std::cout << "transform_dir axis " << axis << " inner_result " << tmp << std::endl;
        detail::cycledim(tmp, result, 1, axis, -1);
        //std::cout << "transform_dir axis " << axis << " result " << result << std::endl;
      }
    }

  SCOPE void general_transform(
    const concepts::TensorView auto& t,
    const concepts::TensorViewArray<2> auto& c,
    concepts::TensorView auto& result_in,
    concepts::TensorView auto& result_tmp_in)
    {
      /* create our own tensor views pointing to the input
       * data so we don't have to modify the input views */
      using result_view_type = std::decay_t<decltype(result_in)>;
      constexpr const mra::Dimension ndim = result_view_type::ndim();
      SHARED result_view_type result, result_tmp;
      if (is_team_lead()) {
        result = result_view_type(result_in.data(), result_in.dims());
        result_tmp = result_view_type(result_tmp_in.data(), result_tmp_in.dims());
      }
      SYNCTHREADS();
      if constexpr (ndim % 2) {
        // make sure result and result_tmp
        // end up pointing to the same memory
        if (is_team_lead()) {
          std::swap(result, result_tmp);
        }
        SYNCTHREADS();
      }
      result = t; // prime result
      for (size_type i = 0; i < ndim; ++i){
        // inner accumulates but we're passing
        // TODO: make accumulation optional?
        result_tmp = 0;
        detail::inner(result, c[i], result_tmp, 0, 0);
        if (is_team_lead()) {
          std::swap(result, result_tmp);
        }
        SYNCTHREADS();
      }
    }

} // namespace mra

#endif // MRA_TRANSFORM_H
