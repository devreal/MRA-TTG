#ifndef MRA_KERNELS_COMPRESS_H
#define MRA_KERNELS_COMPRESS_H

#include <array>

#include "mra/ops/functions.h"
#include "mra/kernels/transform.h"
#include "mra/ops/functions.h"
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

    template<typename T, Dimension NDIM>
    DEVSCOPE void compress_kernel_impl(
      Key<NDIM> key,
      size_type K,
      bool is_ns,
      concepts::TensorView<NDIM> auto& p,
      concepts::TensorView<NDIM> auto& d,
      const concepts::TensorView<2> auto& hgT,
      concepts::TensorView<NDIM> auto& s,
      T* workspace,
      T* d_sumsq,
      const concepts::TensorViewArray<NDIM, Key<NDIM>::num_children()> auto& in_views)
    {

      for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(key, K, i);
        const auto& in = in_views[i];
        s(child_slice) = in;
      }

      transform(s, hgT, d, workspace);

      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      if (!p.empty()) {
        p = d(child_slice);
      }

      if (key.level() > 0 && !is_ns) d(child_slice) = 0.0;

      sumabssq(d, d_sumsq);
    }

    template<typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void compress_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      bool is_ns,
      const concepts::TensorView<NDIM+1> auto node_in,
      concepts::TensorView<NDIM+1> auto p_in,
      concepts::TensorView<NDIM+1> auto result_in,
      const concepts::TensorView<2> auto hgT,
      T* tmp,
      T* d_sumsq,
      const concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto in_views)
    {
      const size_type K2NDIM    = std::pow(  K,NDIM);
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      SHARED std::array<decltype(in_views[0](0)), Key<NDIM>::num_children()> block_in_views;
      SHARED T* workspace;
      SHARED DenseTensorView<const T, NDIM> node;
      SHARED DenseTensorView<T, NDIM> s, p, d;
      int blockId = blockIdx.x;
      T* block_tmp = &tmp[blockId*compress_tmp_size<NDIM>(K)];

      if (is_team_lead()) {
        s = DenseTensorView<T, NDIM>(&block_tmp[0], 2*K);
        workspace = &block_tmp[TWOK2NDIM];
      }
      for (size_type fnid = blockId; fnid < N; fnid += gridDim.x) {
        if (result_in.is_zero(fnid) && p_in.is_zero(fnid)) {
          std::cout << "COMPRESS " << key << " skipping fnid " << fnid << " because result and p are zero" << std::endl;
          continue; // output is zero so skip computation and leave it zero
        }
        if (is_team_lead()) {
          for (int i = 0; i < Key<NDIM>::num_children(); ++i) {
            if (in_views[i].is_zero(fnid)) {
              block_in_views[i] = DenseTensorView<const T, NDIM>(); // dummy view since compress_kernel_impl expects a non-const view for all children
            } else {
              block_in_views[i] = in_views[i](fnid);
            }
          }
          p = p_in(fnid);
          if (!result_in.is_zero(fnid)) {
            d = result_in(fnid);
          }
          node = node_in(fnid);
        }
        SYNCTHREADS();
        if (result_in.is_zero(fnid) && !p_in.is_zero(fnid)) {
          p = node; // pass through the input to the output
          std::cout << "COMPRESS " << key << " pass through fnid " << fnid << " because result is zero but p is not zero" << std::endl;
          continue; // output is zero so skip computation and leave it zero
        }
        assert(!result_in.is_zero(fnid) && !p_in.is_zero(fnid) && "expected result_in and p_in to be non-zero!");
        compress_kernel_impl(key, K, is_ns, p, d, hgT, s, workspace,
                             &d_sumsq[fnid], block_in_views);
      }
    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_compress_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    bool is_ns,
    const concepts::TensorView<NDIM+1> auto& in_view,
    concepts::TensorView<NDIM+1> auto& p_view,
    concepts::TensorView<NDIM+1> auto& result_view,
    const concepts::TensorView<2> auto& hgT_view,
    T* tmp,
    T* d_sumsq,
    const concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& in_views,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CONFIGURE_KERNEL((detail::compress_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::compress_kernel, N, thread_dims, smem_size, stream,
      (key, N, K, is_ns, in_view, p_view, result_view, hgT_view, tmp, d_sumsq, in_views));
    checkSubmit();
  }


/* explicit instantiation */
extern template
void submit_compress_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    bool is_ns,
    const SparseTensorView<const double, 3+1>& in_view,
    SparseTensorView<double, 3+1>& p_view,
    SparseTensorView<double, 3+1>& result_view,
    const SparseTensorView<double, 2>& hgT_view,
    double* tmp,
    double* d_sumsq,
    const std::array<SparseTensorView<double, 3+1>, Key<3>::num_children()>& in_views,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_COMPRESS_H
