#ifndef MRA_KERNELS_COMPRESS_H
#define MRA_KERNELS_COMPRESS_H

#include <array>
#include <tuple>

#include "mra/ops/functions.h"
#include "mra/kernels/transform.h"
#include "mra/ops/functions.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsitymanager.h"
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

    template<typename T, Dimension NDIM, typename PT, typename DT, typename HgtT, typename ST, typename InViewsT>
    DEVSCOPE void compress_kernel_impl(
      Key<NDIM> key,
      size_type K,
      bool is_ns,
      PT& p,
      DT& d,
      const HgtT& hgT,
      ST& s,
      T* workspace,
      T* d_sumsq,
      const InViewsT& in_views)
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
     * Does the actual per-fnid setup work (finding fnid, rebinding
     * s/node/p/d/block_in_views to this function's slice of the batch), on
     * behalf of the team lead only. Deliberately factored into its own
     * noinline function -- called from inside `if (is_team_lead()) { ... }`
     * in compress_process_one below -- rather than inlined there directly:
     * see reconstruct_process_one_leader's comment in
     * mra/kernels/reconstruct.h for the full story (nvcc was observed to
     * miscompile the surrounding if(is_team_lead()){...} SYNCTHREADS()
     * pattern when this body was inlined, corrupting SHARED state -- here,
     * `p` reading back wrong after the barrier -- for every thread including
     * the one that did the writing; a real function-call boundary sidesteps
     * whatever register-liveness/name-hiding confusion the inlined form was
     * hitting).
     *
     * Every parameter that would naturally be an abbreviated
     * `concepts::TensorView<...> auto&` is instead spelled out as its own
     * named template parameter (NodeInT, PInT, ...), deduced from the call
     * site exactly as an abbreviated-auto parameter would be -- confirmed
     * load-bearing for the analogous reconstruct_process_one_leader, not
     * stylistic: this shape (explicit template parameters T, NDIM mixed with
     * many abbreviated-auto parameters) is what crashes nvcc 12.9/13.0's
     * front end outright with "internal error: assertion failed ... in
     * check_name_hiding_by_template_parameters".
     */
    template<typename T, Dimension NDIM, typename NodeInT, typename PInT, typename ResultInT, typename InViewsT,
             typename ST, typename BlockInViewsT, typename NodeT, typename PT, typename DT>
#if defined(__CUDACC__) || defined(__HIPCC__)
    __noinline__
#endif
    DEVSCOPE void compress_process_one_leader(
      const NodeInT& node_in,
      PInT& p_in,
      ResultInT& result_in,
      const InViewsT& in_views,
      T* tmp,
      size_type K,
      size_type N,
      size_type tmp_pos,
      ST& s,
      T*& workspace,
      BlockInViewsT& block_in_views,
      NodeT& node,
      PT& p,
      DT& d,
      size_type& fnid)
    {
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      // p_in has exactly n_nonzero non-zero entries (that's how tmp_pos's
      // range was sized), so this always finds a valid function id.
      fnid = find_nth_nonzero(N, tmp_pos, p_in);
      T* block_tmp = &tmp[tmp_pos*compress_tmp_size<NDIM>(K)];
      s = DenseTensorView<T, NDIM>(&block_tmp[0], DynamicDimensions<NDIM>(2*K, 2*K, 2*K));
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

    /**
     * Processes one function of one node: the per-block body shared by both
     * the unbatched compress_kernel below and compress_kernel_batched further
     * down -- there is exactly one copy of this logic to maintain instead of
     * two near-identical grid-stride loops. tmp/d_sumsq are this member's own
     * base pointers (i.e. already offset to this node, not indexed by a global
     * block id) -- the per-tmp_pos offset into tmp is computed here. `fnid`
     * (the real, sparse function index) is found by the team lead alone, via
     * an on-device scan of p_in's own sparsity (already device-resident, no
     * separate host-built index list/transfer needed), and shared with the
     * rest of the block via a SHARED variable.
     */
    template<typename T, Dimension NDIM, typename NodeInT, typename PInT, typename ResultInT, typename HgtT, typename InViewsT>
    DEVSCOPE void compress_process_one(
      Key<NDIM> key,
      size_type K,
      bool is_ns,
      const NodeInT& node_in,
      PInT& p_in,
      ResultInT& result_in,
      const HgtT& hgT,
      T* tmp,
      T* d_sumsq,
      const InViewsT& in_views,
      size_type N,
      size_type tmp_pos)
    {
      SHARED std::array<decltype(in_views[0](0)), Key<NDIM>::num_children()> block_in_views;
      SHARED T* workspace;
      SHARED DenseTensorView<const T, NDIM> node;
      SHARED DenseTensorView<T, NDIM> s, p, d;
      SHARED size_type fnid;

      if (is_team_lead()) {
        compress_process_one_leader<T, NDIM>(node_in, p_in, result_in, in_views, tmp, K, N, tmp_pos,
                                              s, workspace, block_in_views, node, p, d, fnid);
      }
      SYNCTHREADS();
      assert(!p.empty());
      if (result_in.is_zero(fnid) && !p_in.is_zero(fnid)) {
        p = node; // pass through the input to the output
        d_sumsq[tmp_pos] = 0.0;
        // std::cout << "COMPRESS " << key << " pass through fnid " << fnid << " because result is zero but p is not zero" << std::endl;
        return; // output is zero so skip computation and leave it zero
      }
      assert(!result_in.is_zero(fnid) && !p_in.is_zero(fnid) && "expected result_in and p_in to be non-zero!");
      compress_kernel_impl(key, K, is_ns, p, d, hgT, s, workspace,
                           &d_sumsq[tmp_pos], block_in_views);
    }

#if defined(MRA_CHECK_NORMS)
    /**
     * Debug-only, launched as its OWN kernel (grid of 1 block) immediately
     * before compress_kernel below, on the same stream -- NOT inlined into
     * compress_kernel itself. Putting this check in block 0 of
     * compress_kernel would race against every other block's call into
     * compress_process_one/find_nth_nonzero in that SAME launch: CUDA gives
     * no ordering guarantee between concurrently-running blocks, so if some
     * other block hits find_nth_nonzero's assert first, the whole context
     * aborts immediately and this check's THROWF (even if it also would have
     * fired) never gets to print -- device printf buffers are not flushed on
     * an abort. A separate, prior kernel on the same stream sidesteps this
     * entirely: stream ordering guarantees it runs (and, if it traps,
     * finishes aborting) to completion before compress_kernel ever starts.
     *
     * n_nonzero was computed host-side (mra/tasks/compress.h's `sparsity`,
     * = nonzero_if_any(result, in)) and sizes compress_kernel's grid/tmp
     * buffer; the non-batched path scatters that same sparsity into p_in's
     * device bitfield via SparsityManager::populate_device_sparsity
     * (mra/tensor/sparsitymanager.h) rather than the from-scratch scatter
     * kernel the batched path uses -- cross-check the two still agree, and
     * that result_in (documented as a subset of p_in's coverage) doesn't
     * need data outside it.
     */
    template<typename T, Dimension NDIM, typename PInT, typename ResultInT>
    GLOBALSCOPE void compress_verify_sparsity_kernel(
      Key<NDIM> key,
      size_type N,
      size_type n_nonzero,
      PInT p_in,
      ResultInT result_in)
    {
      if (is_team_lead()) {
        const size_type actual = count_union_nonzero(N, p_in);
        if (actual != n_nonzero) {
          THROWF("compress_kernel: n_nonzero mismatch at level %d: host=%llu device=%llu (N=%llu)\n",
                 (int)key.level(), (unsigned long long)n_nonzero, (unsigned long long)actual, (unsigned long long)N);
        }
        const size_type bad_result = find_nonzero_not_in_union(N, result_in, p_in);
        if (bad_result != N) {
          THROWF("compress_kernel: result_in non-zero at fnid=%llu (level %d) outside p_in's "
                 "coverage -- that position is never visited by this launch\n",
                 (unsigned long long)bad_result, (int)key.level());
        }
      }
    }
#endif // MRA_CHECK_NORMS

    template<typename T, Dimension NDIM, typename NodeInT, typename PInT, typename ResultInT, typename HgtT, typename InViewsT>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void compress_kernel(
      Key<NDIM> key,
      size_type N,
      size_type n_nonzero,
      size_type K,
      bool is_ns,
      const NodeInT node_in,
      PInT p_in,
      ResultInT result_in,
      const HgtT hgT,
      T* tmp,
      T* d_sumsq,
      const InViewsT in_views)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        compress_process_one<T, NDIM>(key, K, is_ns, node_in, p_in, result_in, hgT,
                                      tmp, d_sumsq, in_views, N, pos);
      }
    }
  } // namespace detail

  template<typename T, Dimension NDIM, typename InViewT, typename PViewT, typename ResultViewT, typename HgtViewT, typename InViewsT>
  void submit_compress_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type n_nonzero,
    size_type K,
    bool is_ns,
    const InViewT& in_view,
    PViewT& p_view,
    ResultViewT& result_view,
    const HgtViewT& hgT_view,
    T* tmp,
    T* d_sumsq,
    const InViewsT& in_views,
    ttg::device::Stream stream)
  {
#if defined(MRA_CHECK_NORMS)
    // Separate, prior kernel on the same stream -- see
    // compress_verify_sparsity_kernel's comment for why this must not be
    // inlined into compress_kernel itself (a same-launch race against
    // find_nth_nonzero's assert in other blocks would let this check's own
    // diagnostic go unprinted).
    CALL_KERNEL((detail::compress_verify_sparsity_kernel<T, NDIM>), 1, 32, 0, stream,
                (key, N, n_nonzero, p_view, result_view));
    checkSubmit();
#endif // MRA_CHECK_NORMS

    Dim3 thread_dims = max_thread_dims(2*K);

    auto smem_size = mTxmq_shmem_size<T>(2*K);
    //CONFIGURE_KERNEL((detail::compress_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::compress_kernel, n_nonzero, thread_dims, smem_size, stream,
      (key, N, n_nonzero, K, is_ns, in_view, p_view, result_view, hgT_view, tmp, d_sumsq, in_views));
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
      T*,                                                                 // tmp: this member's own scratch base, sized to this member's own n_nonzero, indexed by local_pos
      T*,                                                                 // d_sumsq: this member's own scratch base, same sizing/indexing as tmp
      std::array<SparseTensorView<T, NDIM+1>, Key<NDIM>::num_children()>, // in_views (the 8 children)
      size_type,                                                          // n: this member's structural function count N -- used both to size the sparsity-byte span below AND as the scan bound for find_nth_nonzero
      size_type                                                          // sparsity_offset: base of this member's [p_in bytes][result_in bytes] span in the aggregated sparsity staging buffer (see compress_scatter_sparsity_kernel)
    >;

    /* Named indices into CompressBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct CompressBatchArgIdx {
      static constexpr std::size_t key             = 0;
      static constexpr std::size_t node_in         = 1;
      static constexpr std::size_t p_in            = 2;
      static constexpr std::size_t result_in       = 3;
      static constexpr std::size_t tmp             = 4;
      static constexpr std::size_t d_sumsq         = 5;
      static constexpr std::size_t in_views        = 6;
      static constexpr std::size_t n               = 7;
      static constexpr std::size_t sparsity_offset = 8;
    };

    /**
     * One combined launch covering every non-zero function across all
     * members of the batch, flattened into a single 1D grid of size
     * total_nonzero (the sum, across members, of each member's own
     * n_nonzero) -- no padding blocks for members with fewer functions than
     * others, no blocks wasted on functions already known to be zero.
     * member_offsets (size num_members+1) names, for a given global grid
     * position, which member a block belongs to and that member's own
     * compact local position (find_member_for_pos, an O(num_members) scan --
     * cheap since num_members is small); compress_process_one's team lead
     * then turns that local position into a real function id via an
     * on-device scan of that member's own p_in sparsity (find_nth_nonzero).
     * So no per-function index list needs to travel with the batch at all --
     * only the tiny per-member offsets array. This makes
     * compress_kernel_batched a thin wrapper: look up one work item and hand
     * off to the exact same per-(node, function) body compress_kernel itself
     * uses (compress_process_one, defined above with compress_kernel_impl).
     */
    template<typename T, Dimension NDIM, typename HgtT>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void compress_kernel_batched(
      CompressBatchArg<T, NDIM>* args,        // device ptr, size == num_members
      const size_type* member_offsets,        // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      size_type K,
      bool is_ns,
      const HgtT hgT)
    {
      using idx = CompressBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;

      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        const size_type member_N = std::get<idx::n>(arg);

        compress_process_one<T, NDIM>(std::get<idx::key>(arg), K, is_ns,
                                      std::get<idx::node_in>(arg), std::get<idx::p_in>(arg),
                                      std::get<idx::result_in>(arg), hgT,
                                      std::get<idx::tmp>(arg), std::get<idx::d_sumsq>(arg),
                                      std::get<idx::in_views>(arg), member_N, local_pos);
      }
    }

    /**
     * Scatters pre-aggregated per-member sparsity bytes into each member's own
     * p_in/result_in tensors' inline bitfields. The bytes were computed
     * host-side (from each member's real p/result Tensors, via
     * detail::sparsity_to_bytes in submit_compress_batch_leader below),
     * assembled into one contiguous pinned buffer -- p_in's n bytes followed
     * by result_in's n bytes, per member -- and copied to `sparsity` with a
     * single H2D transfer, replacing what would otherwise be one
     * SparsityManager/MockTensor allocation + copy per member per tensor.
     * Launched on the same stream immediately before compress_kernel_batched,
     * so stream ordering alone guarantees the bytes are in place first.
     */
    template <typename T, Dimension NDIM>
    GLOBALSCOPE void compress_scatter_sparsity_kernel(
      CompressBatchArg<T, NDIM>* args,        // device ptr, size == gridDim.x
      const SparsityState* sparsity)          // device ptr, aggregated batch-wide staging buffer
    {
      using idx = CompressBatchArgIdx;

      const size_type member = blockIdx.x;
      auto& arg = args[member];
      auto& p_in = std::get<idx::p_in>(arg);
      auto& result_in = std::get<idx::result_in>(arg);
      const size_type n = std::get<idx::n>(arg);
      const size_type base = std::get<idx::sparsity_offset>(arg);

      for (size_type i = threadIdx.x; i < n; i += blockDim.x) {
        p_in.set_state(i, sparsity[base + i]);
        result_in.set_state(i, sparsity[base + n + i]);
      }
    }

  } // namespace detail

  /**
   * Batched counterpart of submit_compress_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.host_args (by the
   * caller, via detail::submit_compress_batch_leader below), sharing only
   * (K, is_ns, hgT) across the whole batch. Grid is 1D over total_nonzero
   * -- see compress_kernel_batched's comment for why. `sparsity_pool`/
   * `sparsity_slot` carry the batch-wide aggregated sparsity bytes assembled
   * by submit_compress_batch_leader; see compress_scatter_sparsity_kernel.
   * `offset_pool`/`offset_slot` carry the small per-member offsets array
   * (size num_members+1), also assembled by submit_compress_batch_leader.
   */
  template<typename T, Dimension NDIM, typename HgtT>
  void submit_compress_kernel_batched(
    detail::BatchPool<detail::CompressBatchArg<T, NDIM>>& pool,
    typename detail::BatchPool<detail::CompressBatchArg<T, NDIM>>::slot_t& slot,
    detail::BatchPool<detail::SparsityState>& sparsity_pool,
    typename detail::BatchPool<detail::SparsityState>::slot_t& sparsity_slot,
    detail::BatchPool<size_type>& offset_pool,
    typename detail::BatchPool<size_type>::slot_t& offset_slot,
    size_type total_nonzero,
    size_type K,
    bool is_ns,
    const HgtT& hgT,
    ttg::device::Stream stream)
  {
    using idx = detail::CompressBatchArgIdx;
    using arg_t = detail::CompressBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.host_args.size());

#if defined(MRA_ENABLE_CUDA)
    detail::check_cuda_rt(cudaMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    detail::check_cuda_rt(cudaMemcpyAsync(sparsity_slot.dev_args, sparsity_slot.host_args.data(),
                                          sparsity_slot.host_args.size()*sizeof(detail::SparsityState),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    detail::check_cuda_rt(cudaMemcpyAsync(offset_slot.dev_args, offset_slot.host_args.data(),
                                          offset_slot.host_args.size()*sizeof(size_type),
                                          cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    detail::check_hip_rt(hipMemcpyAsync(slot.dev_args, slot.host_args.data(), num_members*sizeof(arg_t),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    detail::check_hip_rt(hipMemcpyAsync(sparsity_slot.dev_args, sparsity_slot.host_args.data(),
                                        sparsity_slot.host_args.size()*sizeof(detail::SparsityState),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    detail::check_hip_rt(hipMemcpyAsync(offset_slot.dev_args, offset_slot.host_args.data(),
                                        offset_slot.host_args.size()*sizeof(size_type),
                                        hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif

    // Scatter each member's aggregated sparsity bytes into its own p_in/
    // result_in tensors' inline bitfields; same stream as the main kernel
    // below, so stream ordering guarantees it completes first.
    CALL_KERNEL((detail::compress_scatter_sparsity_kernel<T, NDIM>), num_members, 32, 0, stream,
                (slot.dev_args, sparsity_slot.dev_args));
    checkSubmit();

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    CALL_KERNEL((detail::compress_kernel_batched<T, NDIM>), total_nonzero, thread_dims, smem_size, stream,
                (slot.dev_args, offset_slot.dev_args, num_members, total_nonzero, K, is_ns, hgT));
    checkSubmit();

    pool.mark_submitted(slot, stream);
    sparsity_pool.mark_submitted(sparsity_slot, stream);
    offset_pool.mark_submitted(offset_slot, stream);
  }

  namespace detail {

    /**
     * Shared by do_compress in mra/tasks/compress.h: given the batch_view
     * returned by its own `co_await ttg::device::coop<Key<NDIM>>(...)` (which
     * must stay inline in the coroutine -- only the ordinary, non-suspending
     * code below is worth sharing), marshal every member into the current
     * device's pool and submit one combined kernel launch if this task is the
     * batch's leader.
     *
     * Sparsity: each member also passes its own real p/result Tensors
     * (get<7>()/get<8>(), not just their views) through coop(), so this
     * leader can read their RangeSparsityBase-backed sparsity directly (no
     * per-member SparsityManager/MockTensor allocation) and assemble every
     * member's [p bytes][result bytes] span into one pinned staging buffer
     * (from the same process-wide sparsity pool used by SparsityManager, see
     * sparsitymanager.h), copied to the device in a single transfer by
     * submit_compress_kernel_batched instead of one small H2D copy per
     * member per tensor. This is independent of the flattening below: it
     * scatters bytes across the tensor's own full structural N range (needed
     * so is_zero()/is_nonzero() reads on p_in/result_in stay correct for
     * every real function id), not just the compacted non-zero subset.
     *
     * Flattening: each member also passes its own n_nonzero (get<9>())
     * through coop() -- already computed independently of batching,
     * per-member, in mra/tasks/compress.h. The leader turns those into a
     * tiny (num_members+1)-entry offsets array (a running sum of
     * n_nonzero), so the combined kernel can launch exactly total_nonzero
     * blocks and each one can find its member with an O(num_members) scan
     * (find_member_for_pos) instead of indexing a per-function list --
     * see compress_kernel_batched.
     *
     * `total_functions` is the whole FunctionSet's total function count
     * (fixed for this operation's entire run, unlike any single member's own
     * structural N, which varies with key.batch()) -- used only to size the
     * sparsity-byte staging pool's first allocation to a fixed upper bound
     * (max_batch_size * 2 * total_functions), so, like the *BatchArg and
     * offsets pools above, it never needs to grow after that.
     */
    template <typename T, Dimension NDIM, typename BatchView, typename HgtT>
    void submit_compress_batch_leader(
      BatchView& batch,
      BatchPoolRegistry<CompressBatchArg<T, NDIM>>& registry,
      size_type K,
      bool is_ns,
      const HgtT& hgT,
      size_type total_functions)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      auto& pool = registry.get(ttg::device::current_device());
      auto& slot = pool.acquire(registry.get_max_batch_size()); // allocate space for full batch
      slot.host_args.clear();

      // Offsets slot: always acquired at max_batch_size+1 capacity (not
      // nb+1), so its device buffer is allocated once, on first use, and
      // never resized after that -- num_members is always <= max_batch_size,
      // which is kept small (O(100)) precisely so this is cheap.
      auto& offset_pool = member_offset_pool_registry().get(ttg::device::current_device());
      auto& offset_slot = offset_pool.acquire(registry.get_max_batch_size() + 1);
      offset_slot.host_args.resize(nb + 1);
      offset_slot.host_args[0] = 0;

      // Sparsity-byte slot: acquired at a fixed upper bound (every member
      // contributes at most 2*total_functions bytes, and there are at most
      // max_batch_size members), not the exact total_sparsity_bytes needed
      // this launch -- so, like the pools above, its device buffer is
      // allocated once and never resized, even though the exact byte count
      // varies launch to launch (different nodes can have different
      // structural N). The logical (used) size is still set to the exact
      // total_sparsity_bytes below, so only the bytes actually needed get
      // memcpy'd/scattered.
      const size_type max_sparsity_bytes = 2 * static_cast<size_type>(registry.get_max_batch_size()) * total_functions;
      size_type total_sparsity_bytes = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        total_sparsity_bytes += 2 * static_cast<size_type>(batch[m].template get<3>().dim(0));
      }
      auto& sparsity_pool = sparsity_pool_registry().get(ttg::device::current_device());
      auto& sparsity_slot = sparsity_pool.acquire(max_sparsity_bytes);
      sparsity_slot.host_args.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      auto key = batch[0].template get<0>();
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_key       = batch[m].template get<0>();
        auto& m_node_in   = batch[m].template get<1>();
        auto& m_p_in      = batch[m].template get<2>();
        auto& m_result_in = batch[m].template get<3>();
        auto& m_tmp       = batch[m].template get<4>();
        auto& m_d_sumsq   = batch[m].template get<5>();
        auto& m_in_views  = batch[m].template get<6>();
        auto& m_p_tensor      = batch[m].template get<7>(); // real p tensor, for its RangeSparsityBase sparsity
        auto& m_result_tensor = batch[m].template get<8>(); // real result (d) tensor
        const size_type m_n_nonzero = batch[m].template get<9>();
        const size_type n = static_cast<size_type>(m_result_in.dim(0)); // structural N

        sparsity_to_bytes(m_p_tensor.sparsity(), &sparsity_slot.host_args[sparsity_offset], n);
        sparsity_to_bytes(m_result_tensor.sparsity(), &sparsity_slot.host_args[sparsity_offset + n], n);

        slot.host_args.emplace_back(m_key, m_node_in, m_p_in, m_result_in,
                                    m_tmp.current_device_ptr(), m_d_sumsq.current_device_ptr(),
                                    m_in_views, n, sparsity_offset);
        sparsity_offset += 2 * n;

        offset_slot.host_args[m + 1] = offset_slot.host_args[m] + m_n_nonzero;
      }
      const size_type total_nonzero = offset_slot.host_args[nb];
      submit_compress_kernel_batched<T, NDIM>(pool, slot, sparsity_pool, sparsity_slot,
                                              offset_pool, offset_slot, total_nonzero,
                                              K, is_ns, hgT, ttg::device::current_stream());
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
/* explicit instantiation */
extern template
void submit_compress_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type n_nonzero,
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
