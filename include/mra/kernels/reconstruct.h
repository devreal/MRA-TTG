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
      const TensorView<T, NDIM>& node,
      const TensorView<T, 2>& hg,
      const TensorView<T, NDIM>& from_parent,
      TensorView<T, NDIM>& s,
      TensorView<T, NDIM>& tmp_node,
      T* workspace,
      std::array<TensorView<T, NDIM>, Key<NDIM>::num_children()>& r_arr)
    {
      s = 0.0;
      tmp_node = node;
      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      if (key.level() != 0) tmp_node(child_slice) = from_parent;

      //unfilter<T,K,NDIM>(node.get().coeffs, s);
      transform<NDIM>(tmp_node, hg, s, workspace);

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
      TensorView<T, NDIM+1> node_view,
      T* tmp_ptr,
      const TensorView<T, 2> hg,
      const TensorView<T, NDIM+1> from_parent_view,
      std::array<TensorView<T, NDIM+1>, Key<NDIM>::num_children()> r_arr)
    {
      const bool is_t0 = (0 == thread_id());

      /* pick the r's for this function */
      SHARED std::array<TensorView<T, NDIM>, Key<NDIM>::num_children()> block_r_arr;
      SHARED TensorView<T, NDIM> s, tmp_node;
      SHARED T* workspace;
      SHARED TensorView<T, NDIM> node, from_parent;

      size_type blockId = blockIdx.x;
      T* block_tmp_ptr = &tmp_ptr[blockId*reconstruct_tmp_size<NDIM>(K)];
      if (is_t0) {
        const size_type TWOK2NDIM = std::pow(2*K,NDIM);
        s           = TensorView<T, NDIM>(&block_tmp_ptr[0], 2*K);
        tmp_node    = TensorView<T, NDIM>(&block_tmp_ptr[1*TWOK2NDIM], 2*K);
        workspace   = &block_tmp_ptr[2*TWOK2NDIM];
      }

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x){
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
    TensorView<T, NDIM+1>& node,
    const TensorView<T, 2>& hg,
    const TensorView<T, NDIM+1>& from_parent,
    const std::array<TensorView<T, NDIM+1>, mra::Key<NDIM>::num_children()>& r_arr,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::reconstruct_kernel, N, thread_dims, 0, stream,
      (key, N, K, node, tmp, hg, from_parent, r_arr));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_RECONSTRUCT_H
