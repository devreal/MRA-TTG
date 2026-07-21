#ifndef MRA_KERNELS_COMPRESS_H
#define MRA_KERNELS_COMPRESS_H

#include <array>
#include <tuple>

#include "mra/ops/functions.h"
#include "mra/kernels/transform.h"
#include "mra/ops/functions.h"
#include "mra/misc/device_batch_pool.h"
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

    /**
     * Processes one function of one node: the per-block body shared by both
     * the unbatched compress_kernel below and compress_kernel_batched further
     * down -- there is exactly one copy of this logic to maintain instead of
     * two near-identical grid-stride loops. tmp/d_sumsq are this member's own
     * base pointers (i.e. already offset to this node, not indexed by a global
     * block id) -- the per-fnid offset into tmp is computed here, from fnid.
     */
    template<typename T, Dimension NDIM>
    DEVSCOPE void compress_process_one(
      Key<NDIM> key,
      size_type K,
      bool is_ns,
      const concepts::TensorView<NDIM+1> auto& node_in,
      concepts::TensorView<NDIM+1> auto& p_in,
      concepts::TensorView<NDIM+1> auto& result_in,
      const concepts::TensorView<2> auto& hgT,
      T* tmp,
      T* d_sumsq,
      const concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& in_views,
      size_type fnid)
    {
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      SHARED std::array<decltype(in_views[0](0)), Key<NDIM>::num_children()> block_in_views;
      SHARED T* workspace;
      SHARED DenseTensorView<const T, NDIM> node;
      SHARED DenseTensorView<T, NDIM> s, p, d;

      if (result_in.is_zero(fnid) && p_in.is_zero(fnid)) {
        //std::cout << "COMPRESS " << key << " skipping fnid " << fnid << " because result and p are zero" << std::endl;
        return; // output is zero so skip computation and leave it zero
      }
      if (is_team_lead()) {
        T* block_tmp = &tmp[fnid*compress_tmp_size<NDIM>(K)];
        s = DenseTensorView<T, NDIM>(&block_tmp[0], 2*K);
        workspace = &block_tmp[TWOK2NDIM];
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
        d_sumsq[fnid] = 0.0;
        // std::cout << "COMPRESS " << key << " pass through fnid " << fnid << " because result is zero but p is not zero" << std::endl;
        return; // output is zero so skip computation and leave it zero
      }
      assert(!result_in.is_zero(fnid) && !p_in.is_zero(fnid) && "expected result_in and p_in to be non-zero!");
      compress_kernel_impl(key, K, is_ns, p, d, hgT, s, workspace,
                           &d_sumsq[fnid], block_in_views);
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
      for (size_type fnid = blockIdx.x; fnid < N; fnid += gridDim.x) {
        compress_process_one<T, NDIM>(key, K, is_ns, node_in, p_in, result_in, hgT,
                                      tmp, d_sumsq, in_views, fnid);
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
    //CONFIGURE_KERNEL((detail::compress_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::compress_kernel, N, thread_dims, smem_size, stream,
      (key, N, K, is_ns, in_view, p_view, result_view, hgT_view, tmp, d_sumsq, in_views));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the compress kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/compress.h. Batching is unrestricted
   * (any level, any position): unlike convolution's per-displacement operator
   * data, compress's only "operator data" is hgT -- a single two-scale filter
   * matrix from FunctionData that is bitwise identical for every node
   * regardless of level or position (see mra/misc/functiondata.h). K/is_ns/hgT
   * are therefore true batch-wide kernel parameters, same as K/fac for
   * convolution; only the per-node views/scratch/child-views and `key` (needed
   * for compress_kernel_impl's `key.level() > 0` check) travel per member.
   */
  namespace detail {

    /**
     * Per-member argument bundle for compress_kernel_batched. key is carried
     * (unlike convolution's batched path, which could drop it to a dummy
     * value) because compress_kernel_impl reads key.level() to decide whether
     * to zero the scaling-coefficient block -- that really does vary member
     * to member once level is no longer a matching constraint.
     */
    template <typename T, Dimension NDIM>
    using CompressBatchArg = std::tuple<
      Key<NDIM>,                                                          // key
      SparseTensorView<T, NDIM+1>,                                        // node_in ("in": this node's own coeffs)
      SparseTensorView<T, NDIM+1>,                                        // p_in (parent-injection output)
      SparseTensorView<T, NDIM+1>,                                        // result_in (wavelet/d output)
      T*,                                                                 // tmp: this member's own scratch base
      T*,                                                                 // d_sumsq: this member's own scratch base
      std::array<SparseTensorView<T, NDIM+1>, Key<NDIM>::num_children()>, // in_views (the 8 children)
      size_type                                                          // n: number of functions this member contributes
    >;

    /* Named indices into CompressBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct CompressBatchArgIdx {
      static constexpr std::size_t key       = 0;
      static constexpr std::size_t node_in   = 1;
      static constexpr std::size_t p_in      = 2;
      static constexpr std::size_t result_in = 3;
      static constexpr std::size_t tmp       = 4;
      static constexpr std::size_t d_sumsq   = 5;
      static constexpr std::size_t in_views  = 6;
      static constexpr std::size_t n         = 7;
    };

    /**
     * One combined launch covering `num_members` independent nodes sharing
     * only (K, is_ns, hgT). Grid is 3D: blockIdx.y selects the batch member
     * (gridDim.y == num_members), blockIdx.x the function index within that
     * member (gridDim.x == the largest N_m across the whole batch); members
     * with fewer than gridDim.x functions simply have their higher-x blocks
     * do nothing (the grid-stride loop exits immediately once fnid >= n).
     * This makes compress_kernel_batched a thin wrapper: unpack one member's
     * args and hand off to the exact same per-(node, function) body
     * compress_kernel itself uses (compress_process_one, defined above with
     * compress_kernel_impl).
     */
    template<typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void compress_kernel_batched(
      CompressBatchArg<T, NDIM>* args,   // device ptr, size == gridDim.y
      size_type K,
      bool is_ns,
      const concepts::TensorView<2> auto hgT)
    {
      using idx = CompressBatchArgIdx;

      const size_type member = blockIdx.y;
      auto& arg = args[member];
      const size_type n = std::get<idx::n>(arg);

      for (size_type fnid = blockIdx.x; fnid < n; fnid += gridDim.x) {
        compress_process_one<T, NDIM>(std::get<idx::key>(arg), K, is_ns,
                                      std::get<idx::node_in>(arg), std::get<idx::p_in>(arg),
                                      std::get<idx::result_in>(arg), hgT,
                                      std::get<idx::tmp>(arg), std::get<idx::d_sumsq>(arg),
                                      std::get<idx::in_views>(arg), fnid);
      }
    }

  } // namespace detail

  /**
   * Batched counterpart of submit_compress_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.host_args (by the
   * caller, via detail::submit_compress_batch_leader below), sharing only
   * (K, is_ns, hgT) across the whole batch. Grid is (max_n, num_members, 1)
   * -- see compress_kernel_batched's comment for why.
   */
  template<typename T, Dimension NDIM>
  void submit_compress_kernel_batched(
    detail::BatchPool<detail::CompressBatchArg<T, NDIM>>& pool,
    typename detail::BatchPool<detail::CompressBatchArg<T, NDIM>>::slot_t& slot,
    size_type K,
    bool is_ns,
    const concepts::TensorView<2> auto& hgT,
    ttg::device::Stream stream)
  {
    using idx = detail::CompressBatchArgIdx;
    using arg_t = detail::CompressBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.host_args.size());
    size_type max_n = 0;
    for (const auto& arg : slot.host_args) {
      max_n = std::max(max_n, std::get<idx::n>(arg));
    }

#if defined(MRA_ENABLE_CUDA)
    detail::check_cuda_rt(cudaMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    detail::check_hip_rt(hipMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    Dim3 grid_dims(max_n, num_members, 1);

    CALL_KERNEL((detail::compress_kernel_batched<T, NDIM>), grid_dims, thread_dims, smem_size, stream,
                (slot.dev_args, K, is_ns, hgT));
    checkSubmit();

    pool.mark_submitted(slot, stream);
  }

  namespace detail {

    /**
     * Shared by do_compress in mra/tasks/compress.h: given the batch_view
     * returned by its own `co_await ttg::device::coop<Key<NDIM>>(...)` (which
     * must stay inline in the coroutine -- only the ordinary, non-suspending
     * code below is worth sharing), marshal every member into the current
     * device's pool and submit one combined kernel launch if this task is the
     * batch's leader.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_compress_batch_leader(
      BatchView& batch,
      BatchPoolRegistry<CompressBatchArg<T, NDIM>>& registry,
      size_type K,
      bool is_ns,
      const concepts::TensorView<2> auto& hgT)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      auto& pool = registry.get(ttg::device::current_device());
      auto& slot = pool.acquire(nb);
      slot.host_args.clear();
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_key       = batch[m].template get<0>();
        auto& m_node_in   = batch[m].template get<1>();
        auto& m_p_in      = batch[m].template get<2>();
        auto& m_result_in = batch[m].template get<3>();
        auto& m_tmp       = batch[m].template get<4>();
        auto& m_d_sumsq   = batch[m].template get<5>();
        auto& m_in_views  = batch[m].template get<6>();
        slot.host_args.emplace_back(m_key, m_node_in, m_p_in, m_result_in,
                                    m_tmp.current_device_ptr(), m_d_sumsq.current_device_ptr(),
                                    m_in_views, static_cast<size_type>(m_result_in.dim(0)));
      }
      submit_compress_kernel_batched<T, NDIM>(pool, slot, K, is_ns, hgT, ttg::device::current_stream());
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
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
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra

#endif // MRA_KERNELS_COMPRESS_H
