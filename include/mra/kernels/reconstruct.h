#ifndef MRA_KERNELS_RECONSTRUCT_H
#define MRA_KERNELS_RECONSTRUCT_H

#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/kernels/transform.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type reconstruct_tmp_size(size_type K) {
    const size_type TWOK2NDIM = std::pow(2*K,NDIM);
    return 3*TWOK2NDIM; // s, tmp_node & workspace
  }

  namespace detail {

    /**
     * kernel for reconstruct
     */

    template<typename T, Dimension NDIM>
    DEVSCOPE void reconstruct_kernel_impl(
      Key<NDIM> key,
      size_type K,
      const concepts::TensorView<NDIM> auto& node,
      const concepts::TensorView<2> auto& hg,
      const concepts::TensorView<NDIM> auto& from_parent,
      concepts::TensorView<NDIM> auto& s,
      concepts::TensorView<NDIM> auto& tmp_node,
      T* workspace,
      concepts::TensorViewArray<NDIM, Key<NDIM>::num_children()> auto& r_arr)
    {
      s = 0.0;
      tmp_node = node;
      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      if (key.level() != 0) tmp_node(child_slice) = from_parent;

      //unfilter<T,K,NDIM>(node.get().coeffs, s);
      transform(tmp_node, hg, s, workspace);

      /* extract all r from s
      * NOTE: we could do this on 1<<NDIM blocks but the benefits would likely be small */
      for (size_type i = 0; i < key.num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(key, K, i);
        /* tmp layout: 2K^NDIM for s, K^NDIM for workspace, [K^NDIM]* for r fields */
        auto& r = r_arr[i];
        r = s(child_slice);
      }
    }


    template<typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    reconstruct_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      const concepts::TensorView<NDIM+1> auto node_view,
      T* tmp_ptr,
      const concepts::TensorView<2> auto hg,
      const concepts::TensorView<NDIM+1> auto from_parent_view,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto r_arr)
    {
      const bool is_t0 = (0 == thread_id());

      /* pick the r's for this function */
      SHARED std::array<decltype(r_arr[0](0)), Key<NDIM>::num_children()> block_r_arr;
      SHARED DenseTensorView<T, NDIM> s, tmp_node;
      SHARED T* workspace;
      SHARED DenseTensorView<const T, NDIM> node;
      SHARED DenseTensorView<const T, NDIM> from_parent;

      size_type blockId = blockIdx.x;
      T* block_tmp_ptr = &tmp_ptr[blockId*reconstruct_tmp_size<NDIM>(K)];
      if (is_t0) {
        const size_type TWOK2NDIM = std::pow(2*K,NDIM);
        s           = DenseTensorView<T, NDIM>(&block_tmp_ptr[0], 2*K);
        tmp_node    = DenseTensorView<T, NDIM>(&block_tmp_ptr[1*TWOK2NDIM], 2*K);
        workspace   = &block_tmp_ptr[2*TWOK2NDIM];
      }

      assert(node_view.is_any_nonzero() || from_parent.is_any_nonzero() && "why did we even get here?!");

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x){
        if (node.is_zero(fnid) && from_parent.is_zero(fnid)) {
          /* no work to do */
          continue;
        }
        if (is_t0) {
          node = node_view(fnid);
          from_parent = from_parent_view(fnid);
          for (size_type i = 0; i < Key<NDIM>::num_children(); ++i) {
            block_r_arr[i] = r_arr[i](fnid);
          }
        }
        SYNCTHREADS();
        reconstruct_kernel_impl(key, K, node, hg, from_parent, s, tmp_node, workspace, block_r_arr);
      }
    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_reconstruct_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    const concepts::TensorView<NDIM+1> auto& node,
    const concepts::TensorView<2> auto& hg,
    const concepts::TensorView<NDIM+1> auto& from_parent,
    const concepts::TensorViewArray<NDIM+1, mra::Key<NDIM>::num_children()> auto& r_arr,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CONFIGURE_KERNEL((detail::reconstruct_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::reconstruct_kernel, N, thread_dims, smem_size, stream,
      (key, N, K, node, tmp, hg, from_parent, r_arr));
    checkSubmit();
  }


  /* explicit declaration */
  extern template
  void submit_reconstruct_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    const SparseTensorView<double, 3+1>& node,
    const SparseTensorView<double, 2>& hg,
    const SparseTensorView<double, 3+1>& from_parent,
    const std::array<SparseTensorView<double, 3+1>, mra::Key<3>::num_children()>& r_arr,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_RECONSTRUCT_H
