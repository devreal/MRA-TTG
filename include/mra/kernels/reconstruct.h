#ifndef MRA_KERNELS_RECONSTRUCT_H
#define MRA_KERNELS_RECONSTRUCT_H

#include <tuple>

#include "mra/misc/device_batch_pool.h"
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/kernels/transform.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"
#include "mra/ops/functions.h"

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
      bool accumulate_NS,
      const concepts::TensorView<NDIM> auto& node,
      const concepts::TensorView<2> auto& hg,
      const concepts::TensorView<NDIM> auto& from_parent,
      concepts::TensorView<NDIM> auto& s,
      concepts::TensorView<NDIM> auto& tmp_node,
      T* workspace,
      concepts::TensorViewArray<NDIM, Key<NDIM>::num_children()> auto& r_arr,
      concepts::TensorView<NDIM> auto& result)
    {
      s = 0.0;
      tmp_node = node;
      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      // TODO: MADNESS seems to have the parent node already in place on reconstruct. What is going on here?
      if (key.level() > 0) {
        if (accumulate_NS) {
          tmp_node(child_slice) += from_parent;
        } else {
          tmp_node(child_slice) = from_parent;
        }
      }
      //if (accumulate_NS && key.level() != 0) tmp_node(child_slice) += from_parent;
      //std::cout << "MRA-RECONSTRUCT tmp_node " << key << "\n" << tmp_node << std::endl;

      //unfilter<T,K,NDIM>(node.get().coeffs, s);
      transform(tmp_node, hg, s, workspace);

      //std::cout << "MRA-RECONSTRUCT " << key << " node norm " << normf(node)
      //          << " from_parent norm " << normf(from_parent) << " s norm " << normf(s) << std::endl;
      //std::cout << "MRA-RECONSTRUCT S " << key << "\n" << s << std::endl;
      /* extract all r from s
      * NOTE: we could do this on 1<<NDIM blocks but the benefits would likely be small */
      for (size_type i = 0; i < key.num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(key, K, i);
        /* tmp layout: 2K^NDIM for s, K^NDIM for workspace, [K^NDIM]* for r fields */
        auto& r = r_arr[i];
        if (r.empty()) {
          /* child is zero so skip */
          continue;
        }
        r = s(child_slice);
      }

      // extract the result from the input node
      if (!result.empty()) {
        result = from_parent;
      }
    }


    /**
     * Processes one function of one node: the per-block body shared by both
     * the unbatched reconstruct_kernel below and reconstruct_kernel_batched
     * further down -- there is exactly one copy of this logic to maintain
     * instead of two near-identical grid-stride loops. tmp_ptr is this
     * member's own base pointer (i.e. already offset to this node, not
     * indexed by a global block id) -- the per-fnid offset into it is
     * computed here, from fnid.
     */
    template<typename T, Dimension NDIM>
    DEVSCOPE void reconstruct_process_one(
      Key<NDIM> key,
      size_type K,
      bool accumulate_NS,
      const concepts::TensorView<NDIM+1> auto& node_view,
      T* tmp_ptr,
      const concepts::TensorView<2> auto& hg,
      const concepts::TensorView<NDIM+1> auto& from_parent_view,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& r_arr,
      concepts::TensorView<NDIM+1> auto& result_view,
      size_type fnid)
    {
      const bool is_t0 = (0 == thread_id());

      /* pick the r's for this function */
      SHARED std::array<decltype(r_arr[0](0)), Key<NDIM>::num_children()> block_r_arr;
      SHARED DenseTensorView<T, NDIM> s, tmp_node;
      SHARED T* workspace;
      SHARED DenseTensorView<const T, NDIM> node;
      SHARED DenseTensorView<const T, NDIM> from_parent;
      SHARED DenseTensorView<T, NDIM> result;

      if (node_view.is_zero(fnid) && from_parent_view.is_zero(fnid)) {
        /* no work to do */
        return;
      }
      if (is_t0) {
        T* block_tmp_ptr = &tmp_ptr[fnid*reconstruct_tmp_size<NDIM>(K)];
        const size_type TWOK2NDIM = std::pow(2*K,NDIM);
        s           = DenseTensorView<T, NDIM>(&block_tmp_ptr[0], 2*K);
        tmp_node    = DenseTensorView<T, NDIM>(&block_tmp_ptr[1*TWOK2NDIM], 2*K);
        workspace   = &block_tmp_ptr[2*TWOK2NDIM];
        //assert(node_view.is_any_nonzero() || from_parent_view.is_any_nonzero() && "why did we even get here?!");

        node = node_view(fnid);
        from_parent = from_parent_view(fnid);
        for (size_type i = 0; i < Key<NDIM>::num_children(); ++i) {
          if (r_arr[i].is_zero(fnid)) {
            block_r_arr[i] = DenseTensorView<T, NDIM>(); // dummy view since reconstruct_kernel_impl expects a non-const view for all children
          } else {
            block_r_arr[i] = r_arr[i](fnid);
          }
        }
        if (!result_view.is_zero(fnid)) {
          result = result_view(fnid);
        } else {
          result = DenseTensorView<T, NDIM>(); // dummy view since reconstruct_kernel_impl
        }
      }
      SYNCTHREADS();
      reconstruct_kernel_impl(key, K, accumulate_NS, node, hg, from_parent, s, tmp_node, workspace, block_r_arr, result);
    }

    template<typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    reconstruct_kernel(
      Key<NDIM> key,
      size_type N,
      size_type K,
      bool accumulate_NS,
      const concepts::TensorView<NDIM+1> auto node_view,
      T* tmp_ptr,
      const concepts::TensorView<2> auto hg,
      const concepts::TensorView<NDIM+1> auto from_parent_view,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto r_arr,
      concepts::TensorView<NDIM+1> auto result_view)
    {
      for (size_type fnid = blockIdx.x; fnid < N; fnid += gridDim.x){
        reconstruct_process_one<T, NDIM>(key, K, accumulate_NS, node_view, tmp_ptr, hg,
                                         from_parent_view, r_arr, result_view, fnid);
      }
    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_reconstruct_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    bool accumulate_NS,
    const concepts::TensorView<NDIM+1> auto& node,
    const concepts::TensorView<2> auto& hg,
    const concepts::TensorView<NDIM+1> auto& from_parent,
    const concepts::TensorViewArray<NDIM+1, mra::Key<NDIM>::num_children()> auto& r_arr,
    concepts::TensorView<NDIM+1> auto& result,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    //CONFIGURE_KERNEL((detail::reconstruct_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::reconstruct_kernel, N, thread_dims, smem_size, stream,
      (key, N, K, accumulate_NS, node, tmp, hg, from_parent, r_arr, result));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the reconstruct kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/reconstruct.h. Batching is
   * unrestricted (any level, any position): reconstruct's only "operator
   * data" is hg -- a single two-scale filter matrix from FunctionData that is
   * bitwise identical for every node regardless of level or position (see
   * mra/misc/functiondata.h). K/accumulate_NS/hg are therefore true batch-wide
   * kernel parameters, same as K/is_ns/hgT for compress; only the per-node
   * views/scratch/child-views and `key` (needed for reconstruct_kernel_impl's
   * `key.level() > 0` check) travel per member.
   */
  namespace detail {

    /**
     * Per-member argument bundle for reconstruct_kernel_batched. key is
     * carried because reconstruct_kernel_impl reads key.level() to decide
     * whether to inject from_parent at all -- that really does vary member to
     * member once level is no longer a matching constraint.
     */
    template <typename T, Dimension NDIM>
    using ReconstructBatchArg = std::tuple<
      Key<NDIM>,                                                          // key
      SparseTensorView<T, NDIM+1>,                                        // node_view (compressed node coeffs)
      T*,                                                                 // tmp: this member's own scratch base
      SparseTensorView<T, NDIM+1>,                                        // from_parent_view
      std::array<SparseTensorView<T, NDIM+1>, Key<NDIM>::num_children()>, // r_arr (the 8 children)
      SparseTensorView<T, NDIM+1>,                                        // result_view (leaf output)
      size_type                                                          // n: number of functions this member contributes
    >;

    /* Named indices into ReconstructBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct ReconstructBatchArgIdx {
      static constexpr std::size_t key             = 0;
      static constexpr std::size_t node_view       = 1;
      static constexpr std::size_t tmp             = 2;
      static constexpr std::size_t from_parent_view = 3;
      static constexpr std::size_t r_arr           = 4;
      static constexpr std::size_t result_view     = 5;
      static constexpr std::size_t n               = 6;
    };

    /**
     * One combined launch covering `num_members` independent nodes sharing
     * only (K, accumulate_NS, hg). Grid is 3D: blockIdx.y selects the batch
     * member (gridDim.y == num_members), blockIdx.x the function index within
     * that member (gridDim.x == the largest N_m across the whole batch);
     * members with fewer than gridDim.x functions simply have their higher-x
     * blocks do nothing (the grid-stride loop exits immediately once
     * fnid >= n). This makes reconstruct_kernel_batched a thin wrapper: unpack
     * one member's args and hand off to the exact same per-(node, function)
     * body reconstruct_kernel itself uses (reconstruct_process_one, defined
     * above with reconstruct_kernel_impl).
     */
    template<typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void reconstruct_kernel_batched(
      ReconstructBatchArg<T, NDIM>* args,   // device ptr, size == gridDim.y
      size_type K,
      bool accumulate_NS,
      const concepts::TensorView<2> auto hg)
    {
      using idx = ReconstructBatchArgIdx;

      const size_type member = blockIdx.y;
      auto& arg = args[member];
      const size_type n = std::get<idx::n>(arg);

      for (size_type fnid = blockIdx.x; fnid < n; fnid += gridDim.x) {
        reconstruct_process_one<T, NDIM>(std::get<idx::key>(arg), K, accumulate_NS,
                                         std::get<idx::node_view>(arg), std::get<idx::tmp>(arg), hg,
                                         std::get<idx::from_parent_view>(arg), std::get<idx::r_arr>(arg),
                                         std::get<idx::result_view>(arg), fnid);
      }
    }

  } // namespace detail

  /**
   * Batched counterpart of submit_reconstruct_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.host_args (by the
   * caller, via detail::submit_reconstruct_batch_leader below), sharing only
   * (K, accumulate_NS, hg) across the whole batch. Grid is
   * (max_n, num_members, 1) -- see reconstruct_kernel_batched's comment for why.
   */
  template<typename T, Dimension NDIM>
  void submit_reconstruct_kernel_batched(
    detail::BatchPool<detail::ReconstructBatchArg<T, NDIM>>& pool,
    typename detail::BatchPool<detail::ReconstructBatchArg<T, NDIM>>::slot_t& slot,
    size_type K,
    bool accumulate_NS,
    const concepts::TensorView<2> auto& hg,
    ttg::device::Stream stream)
  {
    using idx = detail::ReconstructBatchArgIdx;
    using arg_t = detail::ReconstructBatchArg<T, NDIM>;
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

    CALL_KERNEL((detail::reconstruct_kernel_batched<T, NDIM>), grid_dims, thread_dims, smem_size, stream,
                (slot.dev_args, K, accumulate_NS, hg));
    checkSubmit();

    pool.mark_submitted(slot, stream);
  }

  namespace detail {

    /**
     * Shared by do_reconstruct in mra/tasks/reconstruct.h: given the
     * batch_view returned by its own `co_await ttg::device::coop<Key<NDIM>>(...)`
     * (which must stay inline in the coroutine -- only the ordinary,
     * non-suspending code below is worth sharing), marshal every member into
     * the current device's pool and submit one combined kernel launch if this
     * task is the batch's leader.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_reconstruct_batch_leader(
      BatchView& batch,
      BatchPoolRegistry<ReconstructBatchArg<T, NDIM>>& registry,
      size_type K,
      bool accumulate_NS,
      const concepts::TensorView<2> auto& hg)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      auto& pool = registry.get(ttg::device::current_device());
      auto& slot = pool.acquire(registry.get_max_batch_size());
      slot.host_args.clear();
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_key             = batch[m].template get<0>();
        auto& m_node_view       = batch[m].template get<1>();
        auto& m_tmp             = batch[m].template get<2>();
        auto& m_from_parent_view = batch[m].template get<3>();
        auto& m_r_arr           = batch[m].template get<4>();
        auto& m_result_view     = batch[m].template get<5>();
        slot.host_args.emplace_back(m_key, m_node_view, m_tmp.current_device_ptr(),
                                    m_from_parent_view, m_r_arr, m_result_view,
                                    static_cast<size_type>(m_node_view.dim(0)));
      }
      submit_reconstruct_kernel_batched<T, NDIM>(pool, slot, K, accumulate_NS, hg, ttg::device::current_stream());
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* explicit declaration */
  extern template
  void submit_reconstruct_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    bool accumulate_NS,
    const SparseTensorView<double, 3+1>& node,
    const SparseTensorView<double, 2>& hg,
    const SparseTensorView<double, 3+1>& from_parent,
    const std::array<SparseTensorView<double, 3+1>, mra::Key<3>::num_children()>& r_arr,
    SparseTensorView<double, 3+1>& result,
    double* tmp,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra

#endif // MRA_KERNELS_RECONSTRUCT_H
