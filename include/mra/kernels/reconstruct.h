#ifndef MRA_KERNELS_RECONSTRUCT_H
#define MRA_KERNELS_RECONSTRUCT_H

#include <tuple>
#include <utility>
#include <mutex>
#include <sstream>
#include <vector>
#include <iostream>

#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsitymanager.h"
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
     * Does the actual per-fnid setup work (finding fnid, rebinding
     * s/tmp_node/node/from_parent/block_r_arr/result to this function's
     * slice of the batch), on behalf of the team lead only. Deliberately
     * factored into its own function -- called from inside
     * `if (is_t0) { ... }` in reconstruct_process_one below -- rather than
     * inlined there directly: nvcc was observed to miscompile the
     * surrounding if(is_t0){...} SYNCTHREADS() pattern when this body was
     * inlined, corrupting SHARED state (and even plain register-local
     * values like is_t0 itself) by the time execution reaches the code
     * after the barrier, for every thread including the one that did the
     * writing. A real function-call boundary discards all of this
     * function's own local/register state on return, leaving only the
     * genuinely-SHARED outputs (passed by reference) to cross the barrier
     * -- which sidesteps whatever register-liveness bug the inlined form
     * was hitting.
     *
     * Every parameter that would naturally be an abbreviated
     * `concepts::TensorView<...> auto&` is instead spelled out as its own
     * named template parameter (NodeViewT, FPViewT, ...), deduced from the
     * call site exactly as an abbreviated-auto parameter would be --
     * confirmed load-bearing, not stylistic: compiling this file with a
     * different nvcc release (12.9) crashes the front end outright with
     * "internal error: assertion failed ... in
     * check_name_hiding_by_template_parameters" pointing at this
     * function's closing brace, when it mixes explicit template parameters
     * (T, NDIM) with this many abbreviated-auto parameters. 13.3 doesn't
     * crash on this shape, but the surrounding investigation (an
     * overload-resolution ranking bug that made two independent,
     * syntactically-correct SFINAE exclusions get ignored; a plain local
     * bool reading back wrong for the writing thread across a
     * __syncthreads() barrier with racecheck reporting 0 hazards; and,
     * under -G, a hard jump to PC 0 at this call boundary) points at the
     * same underlying template-parameter-resolution confusion, just
     * miscompiling silently instead of crashing. Named template parameters
     * sidestep it entirely.
     */
    template<typename T, Dimension NDIM, typename NodeViewT, typename FPViewT, typename RArrT, typename ResultViewT,
             typename ST, typename TmpNodeT, typename NodeT, typename FromParentT, typename BlockRArrT, typename ResultT>
    DEVSCOPE void reconstruct_process_one_leader(
      const NodeViewT& node_view,
      T* tmp_ptr,
      const FPViewT& from_parent_view,
      RArrT& r_arr,
      ResultViewT& result_view,
      size_type N,
      size_type K,
      size_type tmp_pos,
      ST& s,
      TmpNodeT& tmp_node,
      T*& workspace,
      NodeT& node,
      FromParentT& from_parent,
      BlockRArrT& block_r_arr,
      ResultT& result,
      size_type& fnid)
    {
      // node_view/from_parent_view/result_view/r_arr[0..7] together have
      // exactly n_nonzero positions where at least one is non-zero, so
      // this always finds a valid function id -- see
      // submit_reconstruct_kernel. result_view/r_arr must be included: a
      // leaf/inner position can have an exactly-zero node/from_parent
      // *value* yet still need its output slot visited (see
      // mra/tasks/reconstruct.h's work_sparsity comment).
      fnid = find_nth_nonzero_any_with_result(N, tmp_pos, node_view, from_parent_view, result_view, r_arr,
                                              std::make_index_sequence<Key<NDIM>::num_children()>{});

      T* block_tmp_ptr = &tmp_ptr[tmp_pos*reconstruct_tmp_size<NDIM>(K)];
      const size_type TWOK2NDIM = std::pow(2*K,NDIM);
      s           = DenseTensorView<T, NDIM>(&block_tmp_ptr[0], 2*K);
      tmp_node    = DenseTensorView<T, NDIM>(&block_tmp_ptr[1*TWOK2NDIM], 2*K);
      workspace   = &block_tmp_ptr[2*TWOK2NDIM];

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
      size_type N,
      size_type tmp_pos)
    {
      const bool is_t0 = (0 == thread_id());

      /* pick the r's for this function */
      SHARED std::array<decltype(r_arr[0](0)), Key<NDIM>::num_children()> block_r_arr;
      SHARED DenseTensorView<T, NDIM> s, tmp_node;
      SHARED T* workspace;
      SHARED DenseTensorView<const T, NDIM> node;
      SHARED DenseTensorView<const T, NDIM> from_parent;
      SHARED DenseTensorView<T, NDIM> result;
      SHARED size_type fnid;

      if (is_t0) {
        reconstruct_process_one_leader<T, NDIM>(node_view, tmp_ptr, from_parent_view, r_arr, result_view, N, K, tmp_pos,
                                                s, tmp_node, workspace, node, from_parent, block_r_arr, result, fnid);
      }
      SYNCTHREADS();
      reconstruct_kernel_impl(key, K, accumulate_NS, node, hg, from_parent, s, tmp_node, workspace, block_r_arr, result);
    }

#if defined(MRA_CHECK_NORMS)
    /**
     * Debug-only, launched as its OWN kernel (grid of 1 block) immediately
     * before reconstruct_kernel below, on the same stream -- NOT inlined
     * into reconstruct_kernel itself. See
     * compress_verify_sparsity_kernel's comment (mra/kernels/compress.h)
     * for why: a check in block 0 of reconstruct_kernel would race against
     * every other block's call into find_nth_nonzero_any in that SAME
     * launch, and CUDA gives no ordering guarantee between concurrently
     * running blocks -- if another block hits that assert first, the whole
     * context aborts immediately and this check's own THROWF (even if it
     * also would have fired) never gets to print. A separate, prior kernel
     * on the same stream sidesteps this entirely.
     *
     * n_nonzero was computed host-side (mra/tasks/reconstruct.h's
     * work_sparsity, now a union over node/from_parent *and* result/r_arr's
     * host sparsity -- result/r_arr's own criteria, from_parent.is_leaf and
     * "from_parent is Inner" respectively, are independent of node/from_parent's
     * *value* sparsity, so they must be folded into the same union used here);
     * nothing scatters that result into node_view/from_parent_view/result_view/
     * r_arr's own device bitfields, so the two are only consistent if whatever
     * populated those bitfields agrees with work_sparsity.
     */
    template<typename T, Dimension NDIM>
    GLOBALSCOPE void reconstruct_verify_sparsity_kernel_single(
      Key<NDIM> key,
      size_type N,
      size_type n_nonzero,
      const concepts::TensorView<NDIM+1> auto node_view,
      const concepts::TensorView<NDIM+1> auto from_parent_view,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto r_arr,
      concepts::TensorView<NDIM+1> auto result_view)
    {
      if (is_team_lead()) {
        const size_type actual = count_union_nonzero_with_result(N, node_view, from_parent_view, result_view, r_arr,
                                                                  std::make_index_sequence<Key<NDIM>::num_children()>{});
        if (actual != n_nonzero) {
          // DEBUG: see reconstruct_verify_sparsity_kernel's (batched)
          // matching breakdown for why this is here.
          printf("RECONSTRUCT-VERIFY-BREAKDOWN key=(%d,[%lld,%lld,%lld]) n_nonzero=%llu actual=%llu N=%llu\n",
                 (int)key.level(), (long long)key.translation()[0], (long long)key.translation()[1],
                 (long long)key.translation()[2], (unsigned long long)n_nonzero, (unsigned long long)actual,
                 (unsigned long long)N);
          for (size_type i = 0; i < N; ++i) {
            printf("  i=%llu node=%d from_parent=%d result=%d r_arr=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                   (unsigned long long)i,
                   (int)node_view.is_nonzero(i), (int)from_parent_view.is_nonzero(i),
                   (int)result_view.is_nonzero(i),
                   (int)r_arr[0].is_nonzero(i), (int)r_arr[1].is_nonzero(i),
                   (int)r_arr[2].is_nonzero(i), (int)r_arr[3].is_nonzero(i),
                   (int)r_arr[4].is_nonzero(i), (int)r_arr[5].is_nonzero(i),
                   (int)r_arr[6].is_nonzero(i), (int)r_arr[7].is_nonzero(i));
          }
          THROWF("reconstruct_kernel: n_nonzero mismatch at level %d: host=%llu device=%llu (N=%llu)\n",
                 (int)key.level(), (unsigned long long)n_nonzero, (unsigned long long)actual, (unsigned long long)N);
        }
      }
    }
#endif // MRA_CHECK_NORMS

    template<typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    reconstruct_kernel(
      Key<NDIM> key,
      size_type N,
      size_type n_nonzero,
      size_type K,
      bool accumulate_NS,
      const concepts::TensorView<NDIM+1> auto node_view,
      T* tmp_ptr,
      const concepts::TensorView<2> auto hg,
      const concepts::TensorView<NDIM+1> auto from_parent_view,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto r_arr,
      concepts::TensorView<NDIM+1> auto result_view)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x){
        reconstruct_process_one<T, NDIM>(key, K, accumulate_NS, node_view, tmp_ptr, hg,
                                         from_parent_view, r_arr, result_view, N, pos);
      }
    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_reconstruct_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type n_nonzero,
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
#if defined(MRA_CHECK_NORMS)
    // Separate, prior kernel on the same stream -- see
    // reconstruct_verify_sparsity_kernel_single's comment for why this must
    // not be inlined into reconstruct_kernel itself (a same-launch race
    // against find_nth_nonzero_any's assert in other blocks would let this
    // check's own diagnostic go unprinted).
    CALL_KERNEL((detail::reconstruct_verify_sparsity_kernel_single<T, NDIM>), 1, 32, 0, stream,
                (key, N, n_nonzero, node, from_parent, r_arr, result));
    checkSubmit();
#endif // MRA_CHECK_NORMS

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    //CONFIGURE_KERNEL((detail::reconstruct_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::reconstruct_kernel, n_nonzero, thread_dims, smem_size, stream,
      (key, N, n_nonzero, K, accumulate_NS, node, tmp, hg, from_parent, r_arr, result));
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
      // No sparsity_offset field anymore: sparsity scatter is done as part
      // of the H2D transfer itself now (see
      // submit_reconstruct_batch_leader/submit_grouped_copy's comments), not
      // by a device kernel reading an offset out of this struct.
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
     * One combined launch covering every non-zero (node OR from_parent)
     * function position across all members of the batch, flattened into a
     * single 1D grid of size total_nonzero -- no padding blocks for members
     * with fewer functions than others, no blocks wasted on positions
     * already known to be zero in both node and from_parent. member_offsets
     * (size num_members+1) names, for a given global grid position, which
     * member a block belongs to and that member's own compact local
     * position (find_member_for_pos, an O(num_members) scan -- cheap since
     * num_members is small); reconstruct_process_one's team lead then turns
     * that local position into a real function id via an on-device union
     * scan of that member's own node_view/from_parent_view sparsity
     * (find_nth_nonzero_any). This makes reconstruct_kernel_batched a thin
     * wrapper: look up one work item and hand off to the exact same
     * per-(node, function) body reconstruct_kernel itself uses
     * (reconstruct_process_one, defined above with reconstruct_kernel_impl).
     */
    template<typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void reconstruct_kernel_batched(
      ReconstructBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets,        // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      size_type K,
      bool accumulate_NS,
      const concepts::TensorView<2> auto hg)
    {
      using idx = ReconstructBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;

      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        const size_type member_N = std::get<idx::n>(arg);

        reconstruct_process_one<T, NDIM>(std::get<idx::key>(arg), K, accumulate_NS,
                                         std::get<idx::node_view>(arg), std::get<idx::tmp>(arg), hg,
                                         std::get<idx::from_parent_view>(arg), std::get<idx::r_arr>(arg),
                                         std::get<idx::result_view>(arg), member_N, local_pos);
      }
    }

#if defined(MRA_CHECK_NORMS)
    /**
     * Debug-only: cross-checks, for every member, that the flattened launch
     * grid's slice for that member (member_offsets[m+1] - member_offsets[m],
     * a running sum of each member's own host-computed n_nonzero -- see
     * submit_reconstruct_batch_leader) agrees with a fresh on-device union
     * scan of that same member's node_view/from_parent_view/result_view/
     * r_arr[0..7] -- result/r_arr's own criteria (from_parent.is_leaf, and
     * "from_parent is Inner" per child) are independent of node/from_parent's
     * *value* sparsity, so they must be part of this union too (see
     * mra/tasks/reconstruct.h's work_sparsity comment). Nothing scatters the
     * host union result into these views' own bitfields, so the two can
     * silently drift apart. One block per member. Launched (gated by
     * MRA_CHECK_NORMS) immediately before reconstruct_kernel_batched in
     * submit_reconstruct_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    GLOBALSCOPE void reconstruct_verify_sparsity_kernel(
      ReconstructBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets)        // device ptr, size == num_members+1
    {
      using idx = ReconstructBatchArgIdx;

      const size_type member = blockIdx.x;
      if (is_team_lead()) {
        auto& arg = args[member];
        const size_type member_N = std::get<idx::n>(arg);
        const size_type expected = member_offsets[member + 1] - member_offsets[member];
        auto& node_view = std::get<idx::node_view>(arg);
        auto& from_parent_view = std::get<idx::from_parent_view>(arg);
        auto& r_arr = std::get<idx::r_arr>(arg);
        auto& result_view = std::get<idx::result_view>(arg);
        const size_type actual = count_union_nonzero_with_result(member_N, node_view, from_parent_view, result_view, r_arr,
                                                                  std::make_index_sequence<Key<NDIM>::num_children()>{});
        if (actual != expected) {
          // DEBUG: dump every view's per-position bit before trapping, so
          // the exact source of the discrepancy (which view, which fnid) is
          // visible instead of just the aggregate counts.
          auto member_key = std::get<idx::key>(arg);
          printf("RECONSTRUCT-VERIFY-BREAKDOWN member=%llu key=(%d,[%lld,%lld,%lld]) expected=%llu actual=%llu N=%llu\n",
                 (unsigned long long)member, (int)member_key.level(),
                 (long long)member_key.translation()[0], (long long)member_key.translation()[1],
                 (long long)member_key.translation()[2],
                 (unsigned long long)expected, (unsigned long long)actual, (unsigned long long)member_N);
          for (size_type i = 0; i < member_N; ++i) {
            printf("  i=%llu node=%d from_parent=%d result=%d r_arr=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                   (unsigned long long)i,
                   (int)node_view.is_nonzero(i), (int)from_parent_view.is_nonzero(i),
                   (int)result_view.is_nonzero(i),
                   (int)r_arr[0].is_nonzero(i), (int)r_arr[1].is_nonzero(i),
                   (int)r_arr[2].is_nonzero(i), (int)r_arr[3].is_nonzero(i),
                   (int)r_arr[4].is_nonzero(i), (int)r_arr[5].is_nonzero(i),
                   (int)r_arr[6].is_nonzero(i), (int)r_arr[7].is_nonzero(i));
          }
          THROWF("reconstruct_kernel_batched: n_nonzero mismatch for batch member %llu: "
                 "host=%llu device=%llu (N=%llu)\n",
                 (unsigned long long)member, (unsigned long long)expected,
                 (unsigned long long)actual, (unsigned long long)member_N);
        }
      }
    }
#endif // MRA_CHECK_NORMS

  } // namespace detail

  /**
   * Batched counterpart of submit_reconstruct_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.args (by the caller,
   * via detail::submit_reconstruct_batch_leader below), sharing only (K,
   * accumulate_NS, hg) across the whole batch. Grid is 1D over total_nonzero
   * -- see reconstruct_kernel_batched's comment for why. slot.offsets carries
   * the small per-member offsets array (size num_members+1), assembled by
   * submit_reconstruct_batch_leader alongside slot.args and slot.extra_dsts/
   * extra_srcs/extra_sizes (the per-member r_arr[0..7]/result scatter
   * destinations -- see submit_grouped_copy's comment) in the same
   * GroupedBatchPool slot -- one event for the whole launch instead of
   * separate ones per region.
   */
  template<typename T, Dimension NDIM>
  void submit_reconstruct_kernel_batched(
    detail::GroupedBatchPool<detail::ReconstructBatchArg<T, NDIM>>& pool,
    typename detail::GroupedBatchPool<detail::ReconstructBatchArg<T, NDIM>>::slot_t& slot,
    size_type total_nonzero,
    size_type K,
    bool accumulate_NS,
    const concepts::TensorView<2> auto& hg,
    ttg::device::Stream stream)
  {
    using idx = detail::ReconstructBatchArgIdx;
    using arg_t = detail::ReconstructBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.args.size());

    // Single combined H2D transfer for args/offsets, plus every member's
    // r_arr[0..7]/result scatter destination copy -- see
    // submit_grouped_copy's comment for the CUDA-batched-memcpy vs. fallback
    // split, and for why no separate scatter kernel is needed anymore.
    detail::submit_grouped_copy<arg_t>(slot, num_members, slot.offsets.size(), pool.device_id, stream);

#if defined(MRA_CHECK_NORMS)
    // Debug-only: verify the flattened grid's per-member slice (derived from
    // each member's host-computed n_nonzero) still matches a fresh on-device
    // union scan of that member's node_view/from_parent_view/result_view/
    // r_arr -- see reconstruct_verify_sparsity_kernel's comment for why these
    // can drift. Must run AFTER submit_grouped_copy above: result_view/r_arr
    // are freshly-allocated output tensors whose device-side sparsity
    // bitfield is only ever populated by that transfer (unlike node_view/
    // from_parent_view, already correctly set by their producing tasks) --
    // checking them before that transfer reads whatever stale/uninitialized
    // bytes happened to occupy that device allocation previously.
    CALL_KERNEL((detail::reconstruct_verify_sparsity_kernel<T, NDIM>), num_members, 32, 0, stream,
                (slot.dev_args, slot.dev_offsets));
    checkSubmit();
#endif // MRA_CHECK_NORMS

    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);

    CALL_KERNEL((detail::reconstruct_kernel_batched<T, NDIM>), total_nonzero, thread_dims, smem_size, stream,
                (slot.dev_args, slot.dev_offsets, num_members, total_nonzero, K, accumulate_NS, hg));
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
     *
     * Sparsity: each member also passes its own real r_arr (the array of 8
     * FunctionsReconstructedNode children, not just their views) and result
     * node through coop() (get<6>()/get<7>()), so this leader can read their
     * RangeSparsityBase-backed sparsity directly (no per-member
     * SparsityManager/MockTensor allocation), stage every member's
     * [r_arr[0] bytes]...[r_arr[7] bytes][result bytes] span into the slot's
     * own pinned sparsity region (part of the same GroupedBatchPool slot as
     * slot.args/slot.offsets, not the separate process-wide pool
     * SparsityManager uses for non-batched pushes -- see sparsitymanager.h),
     * and queue nine direct copies per member -- from each staged slice
     * straight to that tensor's own device buffer, which coincides exactly
     * with where its inline sparsity bitfield starts (see
     * submit_grouped_copy's comment) -- as entries in slot.extra_dsts/
     * extra_srcs/extra_sizes. No scatter kernel needed:
     * submit_reconstruct_kernel_batched's single submit_grouped_copy call
     * carries these straight to their final destinations alongside
     * slot.args/slot.offsets.
     *
     * Flattening: each member also passes its own n_nonzero (get<8>())
     * through coop() -- already computed independently of batching,
     * per-member, in mra/tasks/reconstruct.h. The leader turns those into a
     * tiny (num_members+1)-entry offsets array (a running sum of
     * n_nonzero), so the combined kernel can launch exactly total_nonzero
     * blocks and each one can find its member with an O(num_members) scan
     * (find_member_for_pos) instead of indexing a per-function list -- see
     * reconstruct_kernel_batched.
     *
     * `total_functions` is the whole FunctionSet's total function count
     * (fixed for this operation's entire run, unlike any single member's
     * own structural N) -- used only to size the sparsity-byte staging
     * pool's first allocation to a fixed upper bound
     * (max_batch_size * (num_children+1) * total_functions), so it never
     * needs to grow after that.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_reconstruct_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<ReconstructBatchArg<T, NDIM>>& registry,
      size_type K,
      bool accumulate_NS,
      const concepts::TensorView<2> auto& hg,
      size_type total_functions)
    {
      if (!batch.is_leader()) return;

      constexpr size_type num_children = Key<NDIM>::num_children();

      const std::size_t nb = batch.size();
      const auto device = ttg::device::current_device();
      auto& pool = registry.get(device);

      // Args/offsets regions sized to a fixed upper bound (max_batch_size,
      // +1 for offsets), and the sparsity staging region to
      // (num_children+1)*max_batch_size*total_functions bytes (every member
      // contributes at most (num_children+1)*total_functions bytes) -- not
      // the exact sizes needed this launch -- so the slot's device buffers
      // are each allocated once, on first use, and never resized after that.
      const size_type max_sparsity_bytes =
          (num_children + 1) * static_cast<size_type>(registry.get_max_batch_size()) * total_functions;
      auto& slot = pool.acquire(registry.get_max_batch_size(), registry.get_max_batch_size() + 1, max_sparsity_bytes);
      slot.args.clear();
      slot.offsets.resize(nb + 1);
      slot.offsets[0] = 0;
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();
      slot.extra_dsts.reserve((num_children + 1) * nb);  // r_arr[0..7] + result destination per member
      slot.extra_srcs.reserve((num_children + 1) * nb);
      slot.extra_sizes.reserve((num_children + 1) * nb);

      // Logical (used) sparsity-byte size for this launch varies (different
      // nodes can have different structural N), unlike the slot's capacity
      // above.
      size_type total_sparsity_bytes = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        total_sparsity_bytes += (num_children + 1) * static_cast<size_type>(batch[m].template get<1>().dim(0));
      }
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_key             = batch[m].template get<0>();
        auto& m_node_view       = batch[m].template get<1>();
        auto& m_tmp             = batch[m].template get<2>();
        auto& m_from_parent_view = batch[m].template get<3>();
        auto& m_r_arr           = batch[m].template get<4>();
        auto& m_result_view     = batch[m].template get<5>();
        auto& m_r_arr_tensor    = batch[m].template get<6>(); // real array of 8 r Tensors, for their sparsity
        auto& m_result_tensor   = batch[m].template get<7>(); // real result Tensor
        const size_type m_n_nonzero = batch[m].template get<8>();
        const size_type n = static_cast<size_type>(m_node_view.dim(0)); // structural N

#if defined(MRA_CHECK_NORMS)
        // DEBUG: recompute the true device-side union (node_view OR
        // from_parent_view, exactly what find_nth_nonzero_any scans) right
        // here at batch-assembly/kernel-launch time, and compare against
        // m_n_nonzero (the host-computed union this member's tmp buffer and
        // grid were sized from). Guarded against null storage (e.g. root's
        // empty from_parent).
        if (m_node_view.storage() != nullptr && m_from_parent_view.storage() != nullptr) {
          static std::mutex dbg_mtx_leader;
          std::lock_guard<std::mutex> lg(dbg_mtx_leader);
          cudaDeviceSynchronize();
          std::vector<unsigned char> node_bytes(n), fp_bytes(n);
          cudaMemcpy(node_bytes.data(), m_node_view.storage(), n, cudaMemcpyDeviceToHost);
          cudaMemcpy(fp_bytes.data(), m_from_parent_view.storage(), n, cudaMemcpyDeviceToHost);
          size_type union_nz = 0;
          for (size_type i = 0; i < n; ++i) {
            if ((node_bytes[i] & 1) || (fp_bytes[i] & 1)) ++union_nz;
          }
          if (union_nz != m_n_nonzero) {
            std::ostringstream oss;
            oss << "RECONSTRUCT-LEADER MISMATCH key=" << m_key << " n=" << n
                << " m_n_nonzero=" << m_n_nonzero << " union_nz=" << union_nz
                << " node_bytes=[";
            for (size_type i = 0; i < n; ++i) oss << (unsigned)node_bytes[i] << (i+1<n?",":"");
            oss << "] fp_bytes=[";
            for (size_type i = 0; i < n; ++i) oss << (unsigned)fp_bytes[i] << (i+1<n?",":"");
            oss << "]\n";
            std::cout << oss.str() << std::flush;
          }
        }
#endif // MRA_CHECK_NORMS

        for (size_type c = 0; c < num_children; ++c) {
          sparsity_to_bytes(m_r_arr_tensor[c].coeffs().sparsity(),
                            reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset + c*n]), n);
          slot.extra_dsts.push_back(const_cast<T*>(m_r_arr_tensor[c].coeffs().buffer().device_ptr_on(device)));
          slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset + c*n]);
          slot.extra_sizes.push_back(n);
        }
        sparsity_to_bytes(m_result_tensor.coeffs().sparsity(),
                          reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset + num_children*n]), n);
        slot.extra_dsts.push_back(const_cast<T*>(m_result_tensor.coeffs().buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset + num_children*n]);
        slot.extra_sizes.push_back(n);

        slot.args.emplace_back(m_key, m_node_view, m_tmp.current_device_ptr(),
                               m_from_parent_view, m_r_arr, m_result_view,
                               n);
        sparsity_offset += (num_children + 1) * n;

        slot.offsets[m + 1] = slot.offsets[m] + m_n_nonzero;
      }
      const size_type total_nonzero = slot.offsets[nb];
      submit_reconstruct_kernel_batched<T, NDIM>(pool, slot, total_nonzero, K, accumulate_NS, hg,
                                                  ttg::device::current_stream());
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* explicit declaration */
  extern template
  void submit_reconstruct_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type n_nonzero,
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
