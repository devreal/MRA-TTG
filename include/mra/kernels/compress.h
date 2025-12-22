#ifndef MRA_KERNELS_COMPRESS_H
#define MRA_KERNELS_COMPRESS_H

#include <array>
#include "mra/kernels.h"
#include "mra/kernels/transform.h"
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"

/**
 * Compress kernels
 */

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type compress_tmp_size(size_type K) {
    const size_type TWOK2NDIM = std::pow(2*K,NDIM);
    return (2*TWOK2NDIM); // s & workspace
  }

  namespace detail {

    template<typename T, Dimension NDIM, concepts::tensor_view InViewsT>
    DEVSCOPE void compress_kernel_impl(
      Key<NDIM> key,
      size_type K,
      concepts::tensor_view auto& p,
      concepts::tensor_view auto& d,
      const concepts::tensor_view_2d auto& hgT,
      concepts::tensor_view auto& s,
      T* workspace,
      T* d_sumsq,
      const std::array<InViewsT, Key<NDIM>::num_children()>& in_views)
    {

      for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(key, K, i);
        const InViewsT& in = in_views[i];
        s(child_slice) = in;
      }


      transform(s, hgT, d, workspace);


      if (key.level() > 0) {
        auto child_slice = get_child_slice<NDIM>(key, K, 0);
        p = d(child_slice);
        d(child_slice) = 0.0;
      }

      sumabssq(d, d_sumsq);
    }

    template<typename T, Dimension NDIM, concepts::tensor_view InViewsT>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void compress_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      concepts::tensor_view auto p_in,
      concepts::tensor_view auto result_in,
      const concepts::tensor_view_2d auto hgT,
      T* tmp,
      T* d_sumsq,
      const std::array<InViewsT, Key<NDIM>::num_children()> in_views)
    {
      const bool is_t0 = (0 == thread_id());
      const size_type K2NDIM    = std::pow(  K,NDIM);
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      using tensorview_t = decltype(p_in(0));
      SHARED std::array<decltype(in_views[0](0)), Key<NDIM>::num_children()> block_in_views;
      SHARED T* workspace;
      SHARED tensorview_t s, p, d;
      int blockId = blockIdx.x;
      T* block_tmp = &tmp[blockId*compress_tmp_size<NDIM>(K)];

      if (is_t0) {
        s = tensorview_t(&block_tmp[0], 2*K);
        workspace = &block_tmp[TWOK2NDIM];
      }

      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x) {
        /* no need to sync threads here */
        if (is_t0) {
          for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
            block_in_views[i] = in_views[i](fnid);
          }
          p = p_in(fnid);
          d = result_in(fnid);
        }
        SYNCTHREADS();

        compress_kernel_impl(key, K, p, d, hgT, s, workspace,
                             &d_sumsq[fnid], block_in_views);
      }
    }
  } // namespace detail

  template<typename T, Dimension NDIM, concepts::tensor_view InViewsT>
  void submit_compress_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    concepts::tensor_view auto& p_view,
    concepts::tensor_view auto& result_view,
    const concepts::tensor_view_2d auto& hgT_view,
    T* tmp,
    T* d_sumsq,
    const std::array<InViewsT, Key<NDIM>::num_children()>& in_views,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CONFIGURE_KERNEL((detail::compress_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::compress_kernel, N, thread_dims, smem_size, stream,
      (key, N, K, p_view, result_view, hgT_view, tmp, d_sumsq, in_views));
  }


/* explicit instantiation */
extern template
void submit_compress_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    TensorView<double, 3+1>& p_view,
    TensorView<double, 3+1>& result_view,
    const TensorView<double, 2>& hgT_view,
    double* tmp,
    double* d_sumsq,
    const std::array<TensorView<double, 3+1>, Key<3>::num_children()>& in_views,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_COMPRESS_H
