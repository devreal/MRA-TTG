#ifndef MRA_TASKS_COMPRESS_H
#define MRA_TASKS_COMPRESS_H

#include <ttg.h>
#include <mutex>
#include <sstream>
#include <vector>
#include <iostream>
#include "mra/kernels.h"
#include "mra/tasks/common.h"
#include "mra/misc/batch_group.h"
#include "mra/misc/batch_size.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/options.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/functionset.h"
#include "mra/ops/functions.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/functionnorm.h"
#include "mra/functors/gaussian.h"
#include "mra/functors/functionfunctor.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra
{
namespace detail {
  /**
   * Bottom-up compression, shared by make_compress and make_compress_from_ns.
   *  - FromNS == false: `in` is a reconstructed tree, whose leaves are filtered upward.
   *  - FromNS == true: `in` is a tree in non-standard form [s|d], e.g., the result of a
   *    convolution. Contributions of coarser levels lie in V_n, which is orthogonal to W_n,
   *    so the standard-form wavelets of a node only depend on the node and its subtree:
   *    each node filters its children's scaling coefficients and adds its own [s|d] block,
   *    whose scaling part travels up with the children's. This yields the same tree as
   *    make_reconstruct(accumulate_NS = true) followed by make_compress, without the
   *    top-down pass.
   */
  template <bool FromNS, typename T, mra::Dimension NDIM, typename FunctionSetT,
            typename ProcMap, typename DeviceMap>
  auto make_compress_impl(
    const std::shared_ptr<FunctionSetT>& fns,
    const std::size_t K,
    const bool is_ns,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, std::conditional_t<FromNS, mra::FunctionsCompressedNode<T, NDIM>,
                                                         mra::FunctionsReconstructedNode<T, NDIM>>>& in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
    const std::string name,
    ProcMap&& procmap,
    DeviceMap&& devicemap,
    const bool enable_truncate,
    const T thresh,
    const int truncate_mode,
    const T cell_min_width,
    const BatchGroup& batch_group)
  {
    static_assert(NDIM == 3); // TODO: worth fixing?
    /* the node type of the input tree */
    using in_node_type = std::conditional_t<FromNS, mra::FunctionsCompressedNode<T, NDIM>,
                                                    mra::FunctionsReconstructedNode<T, NDIM>>;
    /**
     * Companion value type used to carry, for each function, whether a
     * node's own wavelet coefficients survived truncation -- flows up
     * alongside `p` so a parent's own truncation decision can see whether
     * any child still has live wavelet mass below it. Always wired (like
     * `is_ns` below, cost is small and independent of whether truncation is
     * actually enabled) so there is a single implementation to maintain;
     * `enable_truncate` only gates whether coefficients actually get
     * dropped.
     *
     * This is a plain host-side mra::KeptInfo (std::vector-backed), NOT a
     * device-resident Tensor: each child's KeptInfo is only ever read on the
     * host here, and do_compress below packs all 8 of them into a single
     * device buffer itself for the kernel's benefit instead of shipping
     * each child's vector to the device as its own small transfer (see
     * child_kept_packed below). truncate.h's own kept_tensor_type (a
     * DenseTensor<T,1>) is a separate, unrelated alias of the same name
     * used in its own, still-device-based scheme.
     */
    using kept_info_type = mra::KeptInfo<T>;

    // Batching is controlled process-wide via mra::set_batch_size(), not per
    // call here -- see mra/misc/batch_size.h. Read once, at graph-construction
    // time: this bakes the decision into do_compress/its matcher below, so
    // calling set_batch_size() again after this graph exists has no effect on it.
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_compress_batching = mra::batching_enabled();
    // Total function count across the whole FunctionSet -- fixed for this
    // operation's entire run (unlike a single node's N, which varies with
    // key.batch()). Used to size the batch leader's sparsity-byte staging
    // buffer to a fixed upper bound (max_batch_size * 2 * total_functions)
    // so it never needs to grow after its first allocation.
    const size_type total_functions = fns->num_functions();

#ifndef MRA_ENABLE_HOST
    // compress's only "operator data" is hgT, a single two-scale filter matrix
    // from FunctionData that never varies by level or position (see
    // mra/kernels/compress.h's batching-support comment) -- so unlike
    // convolution, batching here has no level/position restriction to begin
    // with; it's unrestricted from the start.
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::CompressBatchArg<T, NDIM>>> compress_pool;
    if (enable_compress_batching) {
      // shared by every instance of batch_group, whose batches may mix their
      // tasks: the leader's K, is_ns and hgT (which only depends on K) apply
      // to the whole batch
      using registry_t = detail::GroupedBatchPoolRegistry<detail::CompressBatchArg<T, NDIM>>;
      compress_pool = batch_group.pool<registry_t>([]{
        return std::make_shared<registry_t>(ttg::device::num_devices(), mra::get_batch_size());
      }, K, is_ns);
    }
#else
    // GroupedBatchPoolRegistry only exists on device builds; this placeholder
    // only exists so the (shared host/device) do_compress lambda below can
    // unconditionally list compress_pool in its capture list -- it is never
    // accessed on host builds.
    std::nullptr_t compress_pool = nullptr;
#endif // MRA_ENABLE_HOST

    // from non-standard form every node goes to compress, so the input feeds it directly
    ttg::Edge<mra::Key<NDIM>, in_node_type> filter_in = FromNS ? in : ttg::Edge<mra::Key<NDIM>, in_node_type>(name + "-filter_in");

    /**
     * A filter that only sends internal nodes to the compress task. Leaf nodes are sent up via the do_send_leafs_up task.
     */
    auto filter_fn = [fns, K, name](const mra::Key<NDIM>& key,
                                    const mra::FunctionsReconstructedNode<T, NDIM>& in) -> TASKTYPE {
      if (!in.is_all_leaf_or_invalid()) {
        /* otherwise send to the compress task */
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, in);
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, in);
#endif // MRA_ENABLE_HOST
      }
    };

    constexpr const std::size_t num_children = mra::Key<NDIM>::num_children();
    // "kept" in/out terminals sit at num_children..2*num_children-1, right
    // after the p terminals at 0..num_children-1 (see out_terminal_id below).
    // creates the right number of edges for nodes ("p") to flow from send_leafs_up/compress up to the parent's compress
    // send_leafs_up will select the right input for compress
    auto p_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>{})...);
      }(std::make_index_sequence<num_children>{});
    // companion "kept" edges: whether a child's own wavelet coefficients survived truncation
    // (see kept_info_type above); flow alongside p_edges, one per child.
    auto kept_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, kept_info_type>{})...);
      }(std::make_index_sequence<num_children>{});
    // do_compress's full set of "from below" inputs: p0..p7, kept0..kept7, then the self/prior edge
    auto send_to_compress_edges = std::tuple_cat(p_edges, kept_edges, std::make_tuple(filter_in));
    // output edges for the send_leafs_up / send_leafs_kept_up tasks, one set per child
    auto send_leaves_up_edges = p_edges;
    auto send_leaves_kept_up_edges = kept_edges;
    /* append out edge to set of edges: p0..p7 (up to parent), kept0..kept7 (up to parent), out */
    auto compress_out_edges = std::tuple_cat(p_edges, kept_edges, std::make_tuple(out));
    /* use the tuple variant to handle variable number of inputs while suppressing the output tuple */
    auto do_compress =
      [&, fns, K, is_ns, name, enable_compress_batching, compress_pool, total_functions,
       enable_truncate, thresh, truncate_mode, cell_min_width](
                          const mra::Key<NDIM>& key,
                          //const std::tuple<const FunctionsReconstructedNodeTypes&...>& input_frns
                          const mra::FunctionsReconstructedNode<T,NDIM> &in0,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in1,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in2,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in3,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in4,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in5,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in6,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in7,
                          const kept_info_type &kept0, const kept_info_type &kept1,
                          const kept_info_type &kept2, const kept_info_type &kept3,
                          const kept_info_type &kept4, const kept_info_type &kept5,
                          const kept_info_type &kept6, const kept_info_type &kept7,
                          const in_node_type &in // the node from the prior op
                          ) -> TASKTYPE {
      //const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
      //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
        size_type N = fns->num_functions(key);
        constexpr const auto num_children = mra::Key<NDIM>::num_children();
        constexpr const auto out_terminal_id = 2*num_children;
        mra::FunctionsCompressedNode<T,NDIM> result(key, N); // The eventual result
        // create empty, may be reset if needed
        mra::FunctionsReconstructedNode<T, NDIM> p(key, N);
        // this node's own "kept" contribution to send up to our parent;
        // stays default-constructed (empty, i.e. "nothing here") unless we
        // actually allocate a result below (mirrors do_send_leafs_kept_up's
        // leaf placeholder in tasks/common.h)
        kept_info_type kept;

        std::array<const kept_info_type*, num_children> kept_children =
              {&kept0, &kept1, &kept2, &kept3, &kept4, &kept5, &kept6, &kept7};

        /**
         * Optional in-line truncation bookkeeping: figure out, per function,
         * whether each child still has any live wavelet mass below it. Used
         * two ways below: (1) any_child_kept feeds the norm-threshold drop
         * decision further down (gated by enable_truncate), and (2) after
         * each apply_leaf_info() call below, mark_truncated_children_leaf()
         * additionally flags a child as leaf whenever it reports "not kept"
         * -- exactly what make_truncate's own dispatch/merge tasks do (see
         * tasks/truncate.h) -- purely additively (only ever flips a child
         * leaf-or-not from false to true), so it must run *after*
         * apply_leaf_info, not before, or apply_leaf_info's own (structural,
         * reconstructed-tree-based) decision would clobber it. This always
         * runs (cheap, and correct regardless of enable_truncate: a child
         * that reports "not kept" genuinely has no wavelet coefficients
         * there, whether because it's a true tree leaf or because truncation
         * dropped it); only the norm-threshold drop decision itself is
         * gated by enable_truncate.
         *
         * child_kept() is a plain, always-resolved host read: kept0..kept7
         * are mra::KeptInfo values (plain std::vector-backed, see
         * tasks/common.h), never device buffers, so unlike the old
         * DenseTensor-based scheme there is no "transfer still in flight"
         * ambiguity to guess around here -- an empty KeptInfo is a true leaf
         * (do_send_leafs_kept_up's placeholder) and a definite "not kept";
         * any non-empty one is immediately and fully readable.
         */
        auto child_kept = [&](size_type i, size_type c) -> bool {
          const auto& ck = *kept_children[c];
          return !ck.empty() && ck[i] != T(0);
        };
        auto mark_truncated_children_leaf = [&]() {
          for (size_type i = 0; i < N; ++i) {
            for (size_type c = 0; c < num_children; ++c) {
              if (!child_kept(i, c)) result.set_child_leaf(i, c, true);
            }
          }
        };


        /**
         * Buffer to pack child information into. Defined here so it
         * is destroyed at the end of the task. Small and only used by the
         * unbatched path, so plain pageable host memory rather than a pinned
         * DeviceAllocator allocation.
         */
        ttg::Buffer<unsigned char> child_kept_packed;
        ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch;
        ttg::Buffer<T, DeviceAllocator<T>> d_sumsq;

        /* check if all inputs are empty */
        bool all_empty = in.empty() && in0.empty() && in1.empty() && in2.empty() && in3.empty() &&
                         in4.empty() && in5.empty() && in6.empty() && in7.empty();

        if (all_empty) {
          // Collect child leaf info
          mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
          mark_truncated_children_leaf();
          //mra::apply_leaf_info(p, in, in0, in1, in2, in3, in4, in5, in6, in7);
          // p.set_all_leaf(LeafStatus::Invalid);
          //std::cout << name << " " << key << " all empty, all children leafs " << result.is_all_child_leaf() << " ["
          //           << in.is_all_leaf() << ", " << in0.is_all_leaf() << ", " << in1.is_all_leaf() << ", "
          //           << in2.is_all_leaf() << ", " << in3.is_all_leaf() << ", "
          //           << in4.is_all_leaf() << ", " << in5.is_all_leaf() << ", "
          //           << in6.is_all_leaf() << ", " << in7.is_all_leaf() << "] "
          //           << std::endl;
        } else {

          /* some inputs are on the device so submit a kernel */

          SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero); // start with all zero, we'll set the non-zero ones as we go
          /**
           * We only produce a result if at least one of the children is non-zero.
           */
          sparsity.nonzero_if_any(in0, in1, in2, in3, in4, in5, in6, in7);
          // the node's own non-standard block contributes to the result as well
          if constexpr (FromNS) sparsity.nonzero_if_any(in);
          //std::cout << name << " " << key << " sparsity: " << sparsity << std::endl;

          // allocate the result
          result.allocate(sparsity, K, ttg::scope::Allocate);
          result.set_ns(is_ns);

          // Collect child leaf info
          mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
          mark_truncated_children_leaf();
          //std::cout << name << " " << key << " result after apply_leaf_info " << result << " " << std::endl;

          /**
           * Allocate the reconstructed node to send up to the parent.
           */
          sparsity.set_all_zero(); // reset
          sparsity.nonzero_if_any(result, in); // set based on result and our input (to catch any new leafs we pick up from the input)
          // from non-standard form, p stays inner: the parent marks this node as a child leaf
          // of every function without coefficients here (see mark_truncated_children_leaf)
          if constexpr (!FromNS) mra::apply_leaf_info(p, in);

          // NOTE: we don't care about leaf info for p since it is a temporary only
          //mra::apply_leaf_info(p, result); // set leaf info on p since that's what we send up
          p.allocate(sparsity, K, ttg::scope::Allocate);
          if (sparsity.is_any_nonzero()) {
            assert(!p.empty());
          }
          //std::cout << name << " " << key << ", p before apply_leaf_info " << p.sparsity() << " " << std::endl;
          //mra::apply_leaf_info(p, in, in0, in1, in2, in3, in4, in5, in6, in7);
          //std::cout << name << " " << key << ", p after apply_leaf_info " << p.sparsity() << " " << std::endl;

          FunctionNorms<T, NDIM> norms(name, in, in0, in1, in2, in3, in4, in5, in6, in7, result);

          // p's sparsity is nonzero_if_any(result, in) (set a few lines
          // above), i.e. a superset of result's own nonzero set: p_in
          // being zero at some fnid implies result_in is zero there too. So
          // p's own non-zero count is exactly the number of function ids
          // compress_process_one does any work for; the actual ids
          // themselves are found on-device (find_nth_nonzero, scanning
          // p_in's own sparsity), not built into a host-side list here.
          const size_type n_nonzero = sparsity.count_nonzero();

          // When batching, submit_compress_kernel_batched's own occupancy-
          // capped scratch pool (GroupedBatchPool::ensure_scratch) supplies
          // every member's real per-block transform workspace -- no per-task
          // buffer is needed at all in that case, so tmp_size is 0 and
          // tmp_scratch is never allocated. The unbatched fallback (host, or
          // batching disabled at runtime) still needs the full size, since
          // it's the only thing submit_compress_kernel uses as scratch.
          // (d_sumsq below is NOT pooled the same way: where this node may
          // truncate, the host reads every compact position back after the
          // kernel runs, so it stays sized to this task's own n_nonzero.)
#ifndef MRA_ENABLE_HOST
          const bool use_batching = enable_compress_batching;
#else
          const bool use_batching = false;
#endif
          const std::size_t tmp_size = use_batching ? 0 : compress_tmp_size<NDIM>(K)*n_nonzero;
          if (tmp_size > 0) tmp_scratch.reset(tmp_size, ttg::scope::Allocate);
          const auto& hgT = functiondata.get_hgT();
          // The kernel decides truncation on-device; the host only needs the
          // sums back to mirror that decision into result's host-side
          // sparsity, so they are only transferred where this node may
          // truncate and are device-only scratch otherwise.
          const bool may_truncate = enable_truncate && key.level() > 1;
          /* stores sumsq for each non-zero function; indexed by compact position, not fnid */
          d_sumsq.reset(n_nonzero, may_truncate ? ttg::scope::SyncOut : ttg::scope::Allocate);

          // Computed up front (depends only on key/thresh/cell_min_width/
          // truncate_mode, all known before the kernel runs) so the kernel
          // itself can apply the truncation decision on-device -- see
          // compress_process_one in mra/kernels/compress.h. 0 when
          // truncation is disabled, which makes that kernel's sumsq
          // comparison always false (sumsq is never negative).
          const T trunc_tol = enable_truncate ? mra::truncate_tol(key, thresh, cell_min_width, truncate_mode) : T{};

          auto& d = result.coeffs();

          // Pack every child's KeptInfo into a single [num_children x N]
          // byte buffer (child-major: child c's row is
          // child_kept_packed[c*N .. c*N+N), 1 = kept) so only ONE
          // host->device transfer happens below, instead of up to 8 small
          // ones. A structurally-absent child (kept_children[c]->empty(),
          // i.e. a true leaf) leaves its row zeroed, i.e. "not kept".
          // The kernel only reads it where it may truncate (see
          // compress_process_one), so it is neither built nor moved to the
          // device otherwise, and the kernel gets a nullptr instead.
          // When batching, the batch leader packs these same bytes into its
          // slot's grouped H2D transfer instead (see
          // submit_compress_batch_leader), so no per-task buffer is needed.
          const bool pack_child_kept = may_truncate && !use_batching;
          if (pack_child_kept) {
            child_kept_packed.reset(N * num_children, ttg::scope::SyncIn);
            unsigned char* packed = child_kept_packed.host_ptr();
            for (size_type c = 0; c < num_children; ++c) {
              for (size_type i = 0; i < N; ++i) packed[c*N + i] = child_kept(i, c) ? 1 : 0;
            }
          }

#ifndef MRA_ENABLE_HOST
          auto input = ttg::device::Input(p.coeffs().buffer(), d.buffer(), hgT.buffer(), d_sumsq);
          if (tmp_size > 0) input.add(tmp_scratch);
          if (pack_child_kept) input.add(child_kept_packed);
          auto select_in = [&](const auto& in) {
            if (!in.empty()) {
              input.add(in.coeffs().buffer());
            }
          };
          select_in(in);
          select_in(in0); select_in(in1);
          select_in(in2); select_in(in3);
          select_in(in4); select_in(in5);
          select_in(in6); select_in(in7);
          input.add(norms.buffer());

          co_await ttg::device::select(input);
#endif

          /* some constness checks for the API */
          static_assert(std::is_const_v<std::remove_reference_t<decltype(in0)>>);
          static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs())>>);
          static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs().buffer())>>);
          static_assert(std::is_const_v<std::remove_reference_t<std::remove_reference_t<decltype(*in0.coeffs().buffer().current_device_ptr())>>>);

          /* assemble input array and submit kernel */
          //auto input_ptrs = std::apply([](auto... ins){ return std::array{(ins.coeffs.buffer().current_device_ptr())...}; });
          auto input_views = std::array{in0.coeffs().current_view(), in1.coeffs().current_view(), in2.coeffs().current_view(), in3.coeffs().current_view(),
                                        in4.coeffs().current_view(), in5.coeffs().current_view(), in6.coeffs().current_view(), in7.coeffs().current_view()};

          // the input node is either this node's reconstructed coefficients or its non-standard block
          auto in_view = FromNS ? decltype(in.coeffs().current_view()){} : in.coeffs().current_view();
          auto ns_view = FromNS ? in.coeffs().current_view() : decltype(in.coeffs().current_view()){};

          auto coeffs_view = p.coeffs().current_view();
          auto rcoeffs_view = d.current_view();
          auto hgT_view = hgT.current_view();

          // Single packed [num_children x N] "kept" device pointer, for the
          // unbatched kernel's in-kernel truncation decision -- valid right
          // after select() above stages child_kept_packed onto the device;
          // nullptr where the kernel cannot truncate.
          const unsigned char* child_kept_ptr = pack_child_kept ? child_kept_packed.current_device_ptr() : nullptr;

#ifndef MRA_ENABLE_HOST
          if (enable_compress_batching) {
            // key travels through coop() since compress_kernel_impl reads
            // key.level() -- see the batching-support comment in
            // kernels/compress.h. in_view/coeffs_view/rcoeffs_view/input_views
            // keep the exact roles submit_compress_kernel uses below.
            // p.coeffs()/d (the real tensors, not just their views) travel
            // through too, so the batch leader can read their sparsity and
            // aggregate every member's bytes into one pinned buffer + one
            // H2D copy instead of each member pushing its own via
            // SparsityManager here. n_nonzero (this member's own, computed
            // above independent of batching) travels through as well, so the
            // leader can flatten every member's own non-zero work items into
            // one combined 1D launch (see submit_compress_batch_leader); no
            // per-function index list needs to travel with it, since each
            // member's real function ids are found on-device. The children's
            // KeptInfo travel as host pointers (into this frame, which stays
            // alive while suspended in coop()); the leader packs them into
            // its grouped H2D transfer.
            const auto* kept_children_ptr = may_truncate ? &kept_children : nullptr;
            auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, in_view, coeffs_view, rcoeffs_view,
                                                                    d_sumsq, input_views,
                                                                    p.coeffs(), d, n_nonzero,
                                                                    kept_children_ptr, trunc_tol, ns_view);
            // followers: the leader's batched launch already wrote our slice of p/result/d_sumsq.
            detail::submit_compress_batch_leader<T, NDIM>(batch, *compress_pool, K, is_ns, hgT_view, total_functions,
                                                          enable_truncate);
          } else
#endif // MRA_ENABLE_HOST
          {
            auto sparseman = make_sparsity_manager(d, p);
            sparseman.populate_device_sparsity();
            submit_compress_kernel(key, N, n_nonzero, K, is_ns,
                                  in_view, coeffs_view, rcoeffs_view, hgT_view,
                                  tmp_scratch.current_device_ptr(), d_sumsq.current_device_ptr(), input_views,
                                  ns_view, child_kept_ptr, trunc_tol,
                                  ttg::device::current_stream());
          }
          norms.compute();
          /* wait for kernel and transfer sums back, only if we need them */
#ifndef MRA_ENABLE_HOST
          if (may_truncate) {
            co_await ttg::device::wait(d_sumsq, norms.buffer());
          }
#if defined(MRA_CHECK_NORMS)
          else {
            co_await ttg::device::wait(norms.buffer());
          }
#endif // MRA_CHECK_NORMS
#endif // MRA_ENABLE_HOST
          norms.verify();

#if defined(MRA_CHECK_NORMS)
          // DEBUG: right after the compress kernel (+ its sparsity scatter,
          // if batched) has completed, compare result's (d's) host-tracked
          // sparsity against what's actually on the device -- this is the
          // earliest point in the pipeline where "node" (as later seen by
          // reconstruct) gets its sparsity established.
          if (!result.empty() && rcoeffs_view.storage() != nullptr) {
            static std::mutex dbg_mtx_compress;
            std::lock_guard<std::mutex> lg(dbg_mtx_compress);
            cudaDeviceSynchronize();
            std::vector<unsigned char> devbytes(N);
            cudaMemcpy(devbytes.data(), rcoeffs_view.storage(), N, cudaMemcpyDeviceToHost);
            size_type dev_nz = 0;
            for (size_type i = 0; i < N; ++i) if (devbytes[i] & 1) ++dev_nz;
            size_type host_nz = result.coeffs().sparsity().count_nonzero();
            if (dev_nz != host_nz) {
              std::ostringstream oss;
              oss << "COMPRESS-POST-KERNEL MISMATCH key=" << key << " N=" << N
                  << " rcoeffs_view.storage()=" << (void*)rcoeffs_view.storage()
                  << " host_nz=" << host_nz << " dev_nz=" << dev_nz
                  << " dev_bytes=[";
              for (size_type i = 0; i < N; ++i) oss << (unsigned)devbytes[i] << (i+1<N?",":"");
              oss << "]\n";
              std::cout << oss.str() << std::flush;
            }
          }
#endif // MRA_CHECK_NORMS

          // Mirror compress_process_one's in-kernel truncation decision
          // (mra/kernels/compress.h), which already dropped this node's
          // device-resident wavelet coefficients for function i under the
          // identical condition (same d_sumsq, trunc_tol and child kept bits),
          // purely to keep this host-side `result`'s sparsity in sync with
          // the device. d_sumsq is compacted (indexed by compact position,
          // not fnid), so walk it by re-enumerating `sparsity`'s non-zero ids
          // (the same set find_nth_nonzero scanned on-device).
          if (may_truncate) {
            const auto* d_sumsq_arr = d_sumsq.host_ptr();
            size_type pos = 0;
            for (auto it = sparsity.begin_nonzero(); it != sparsity.end_nonzero(); ++it, ++pos) {
              const size_type i = *it;
              if (result.is_zero(i) || d_sumsq_arr[pos] >= trunc_tol*trunc_tol) continue;
              bool any_child_kept = false;
              for (size_type c = 0; c < num_children && !any_child_kept; ++c) any_child_kept = child_kept(i, c);
              if (!any_child_kept) result.set_zero(i);
            }
          }

          // This node's own contribution to send up to the parent: for every
          // function, did we end up keeping any wavelet coefficients here?
          // (Independent of enable_truncate: relays result's real sparsity
          // either way, since a parent always needs to know a child's true
          // nonzero set to make its own decision once truncation is enabled
          // higher up the tree.)
          kept = kept_info_type(N);
          for (size_type i = 0; i < N; ++i) {
            kept[i] = result.is_zero(i) ? T(0) : T(1);
          }
        }


        //std::cout << name << " " << key << " result " << result << " p " << p << std::endl;

        // Recur up
        if (key.level() > 0) {
          /* The parent marks this node as a child leaf of every function it does not
           * keep (see mark_truncated_children_leaf), so a node kept for no function --
           * truncated away, or without coefficients below it -- is not part of the
           * output tree and is not sent, as in make_truncate. Its descendants are not
           * kept either (a function is only dropped where no child keeps it). */
          bool any_kept = false;
          for (size_type i = 0; i < kept.size() && !any_kept; ++i) any_kept = (kept[i] != T(0));
          // will not return
#ifndef MRA_ENABLE_HOST
          auto sends = ttg::device::forward(
            // select to which child of our parent we send
            select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress"),
            // and forward our own "kept" contribution the same way, offset
            // to the parent's kept0..kept7 terminal group
            select_send_up<num_children>(key, std::move(kept), std::make_index_sequence<num_children>{}, "compress-kept"));
          // Send result to output tree
          if (any_kept) sends.push_back(ttg::device::send<out_terminal_id>(key, std::move(result)));
          co_await std::move(sends);
#else
            select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress");
            select_send_up<num_children>(key, std::move(kept), std::make_index_sequence<num_children>{}, "compress-kept");
            if (any_kept) ttg::send<out_terminal_id>(key, std::move(result));
#endif
        } else {
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::forward(
            // Send result to output tree
            ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
          ttg::send<out_terminal_id>(key, std::move(result));
#endif
        }
    };

    // set maps if provided
    auto set_maps = [&](auto& tt) {
      if constexpr (!std::is_same_v<std::decay_t<ProcMap>, ttg::Void>) tt->set_keymap(procmap);
      if constexpr (!std::is_same_v<std::decay_t<DeviceMap>, ttg::Void>) tt->set_devicemap(devicemap);
    };
    std::vector<std::unique_ptr<ttg::TTBase>> ops;
    auto add_op = [&](auto&& tt) {
      set_maps(tt);
      ops.push_back(std::move(tt));
    };

    auto compress_tt = ttg::make_tt<Space>(std::move(do_compress), send_to_compress_edges, compress_out_edges, "compress");
    set_maps(compress_tt);
    if constexpr (!FromNS) {
      add_op(ttg::make_tt<Space>(&do_send_leafs_up<T,NDIM>, edges(in), send_leaves_up_edges, "send_leaves_up"));
      add_op(ttg::make_tt<Space>(&do_send_leafs_kept_up<T,NDIM>, edges(in), send_leaves_kept_up_edges, "send_leaves_kept_up"));
      add_op(ttg::make_tt<Space>(std::move(filter_fn), ttg::edges(in), ttg::edges(filter_in), "filter"));
    } else {
      /**
       * Every node of the non-standard tree goes to compress. A child without a node
       * (a leaf for all functions) contributes nothing: the node itself sends empty
       * placeholders for it. This is a device task (without data) so that the node is not
       * pulled to the host, see do_send_leafs_up.
       */
      auto send_leaf_children_fn = [fns](const mra::Key<NDIM>& key, const in_node_type& node) -> TASKTYPE {
        const size_type N = fns->num_functions(key);
        constexpr auto seq = std::make_index_sequence<mra::Key<NDIM>::num_children()>{};
#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
#endif // MRA_ENABLE_HOST
        for (const mra::Key<NDIM>& child : mra::KeyChildren<NDIM>(key)) {
          if (!node.is_all_child_leaf(child)) continue;
          mra::FunctionsReconstructedNode<T, NDIM> p(child, N);
          p.set_all_leaf(LeafStatus::Leaf);
          mra::KeptInfo<T> kept; // empty placeholder, see do_send_leafs_kept_up
#ifndef MRA_ENABLE_HOST
          sends.push_back(select_send_up(child, std::move(p), seq, "send_leaf_children_up"));
          sends.push_back(select_send_up<mra::Key<NDIM>::num_children()>(child, std::move(kept), seq, "send_leaf_children_kept_up"));
#else
          select_send_up(child, std::move(p), seq, "send_leaf_children_up");
          select_send_up<mra::Key<NDIM>::num_children()>(child, std::move(kept), seq, "send_leaf_children_kept_up");
#endif // MRA_ENABLE_HOST
        }
#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      };
      auto send_leaf_children_edges = std::tuple_cat(p_edges, kept_edges);
      add_op(ttg::make_tt<Space>(std::move(send_leaf_children_fn), ttg::edges(in), send_leaf_children_edges,
                                 "send_leaf_children_up"));
    }

#ifndef MRA_ENABLE_HOST
    if (enable_compress_batching) {
      // Unrestricted matcher: any two do_compress tasks may batch together,
      // regardless of level or position, up to max_batch_size -- safe because
      // compress's only shared operator data (hgT) never varies by level or
      // position to begin with (see kernels/compress.h's batching-support
      // comment), unlike convolution which needed a similar relaxation.
      compress_tt->set_batch_matcher(
          [](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
          max_batch_size);
      batch_group.add(compress_tt);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(compress_tt->template in<2*num_children>());
    auto outs = std::make_tuple(compress_tt->template out<2*num_children>());
    ops.push_back(std::move(compress_tt));

    return make_ttg(std::move(ops), ins, outs, name);
  }
} // namespace detail

/// Make a composite operator that implements compression for a single function
  template <typename T, mra::Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  static auto make_compress(
    const std::shared_ptr<FunctionSetT>& fns,
    const std::size_t K,
    const bool is_ns,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>& in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
    const std::string name = "compress",
    ProcMap&& procmap = {},
    DeviceMap&& devicemap = {},
    const bool enable_truncate = false,
    const T thresh = T{},
    const int truncate_mode = 0,
    const T cell_min_width = T{},
    const BatchGroup& batch_group = {})
  {
    return detail::make_compress_impl<false>(fns, K, is_ns, functiondata, in, out, name,
                                             std::forward<ProcMap>(procmap), std::forward<DeviceMap>(devicemap),
                                             enable_truncate, thresh, truncate_mode, cell_min_width, batch_group);
  }

  /**
   * Compresses a tree in non-standard form (e.g., the result of make_convolution) into
   * standard form in a single bottom-up pass, producing the same tree as
   * make_reconstruct(accumulate_NS = true) followed by make_compress. Truncation and
   * batch groups work as in make_compress.
   */
  template <typename T, mra::Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  static auto make_compress_from_ns(
    const std::shared_ptr<FunctionSetT>& fns,
    const std::size_t K,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
    const std::string name = "compress_from_ns",
    ProcMap&& procmap = {},
    DeviceMap&& devicemap = {},
    const bool enable_truncate = false,
    const T thresh = T{},
    const int truncate_mode = 0,
    const T cell_min_width = T{},
    const BatchGroup& batch_group = {})
  {
    return detail::make_compress_impl<true>(fns, K, false, functiondata, in, out, name,
                                            std::forward<ProcMap>(procmap), std::forward<DeviceMap>(devicemap),
                                            enable_truncate, thresh, truncate_mode, cell_min_width, batch_group);
  }

}

#endif // MRA_TASKS_COMPRESS_H
