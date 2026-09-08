#ifndef MRA_TASKS_TRUNCATE_H
#define MRA_TASKS_TRUNCATE_H

#include <ttg.h>
#include "mra/kernels.h"
#include "mra/misc/batch_size.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/functionset.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/functionnode.h"
#include "mra/tasks/common.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Truncate a compressed tree: drop a node's wavelet coefficients for a given
   * function whenever none of its children still hold coefficients for that
   * function and the node's own norm falls below the truncation tolerance
   * (mirrors MADNESS's FunctionImpl::truncate/truncate_op). Nodes at level 0/1
   * are never truncated. Dropping a node also marks its children as leafs for
   * that function, so a subsequent reconstruct does not recurse into the
   * (now known to be negligible) subtree below.
   *
   * `in` must carry every key of the compressed tree (e.g. compress's output);
   * `out` receives the same set of keys with truncated coefficients/leaf info.
   *
   * NOTE: this tasks operates in-place, i.e., it modifies the `in` nodes sparsity
   *       (both device and host) and child info. This may change in the future.
   *
   * TODO: Conditionally compact the resuling node if the truncation drops a large
   *       fraction of its coefficients, to avoid wasting memory on a mostly-zero node.
   *
   */
  template <typename T, mra::Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_truncate(
    const std::shared_ptr<FunctionSetT>& fns,
    size_type K,
    T thresh,
    int truncate_mode,
    T cell_min_width,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
    const std::string& name = "truncate",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    static_assert(NDIM == 3); // TODO: worth fixing?
    using kept_tensor_type = mra::DenseTensor<T, 1>;
    static constexpr const auto num_children = mra::Key<NDIM>::num_children();

    // Batching is controlled process-wide via mra::set_batch_size(), not per
    // call here -- see mra/misc/batch_size.h and mra/tasks/compress.h (same
    // pattern). truncate has no operator data at all -- tol itself is
    // per-member (it depends on each member's own key.level()) -- so, like
    // simple_norm, batching here is unrestricted from the start.
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_truncate_batching = mra::batching_enabled();

#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::TruncateBatchArg<T, NDIM>>> truncate_pool;
    if (enable_truncate_batching) {
      truncate_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::TruncateBatchArg<T, NDIM>>>(ttg::device::num_devices(), mra::get_batch_size());
    }
#else
    // GroupedBatchPoolRegistry only exists on device builds; this placeholder
    // only exists so the (shared host/device) truncate_fn lambda below can
    // unconditionally list truncate_pool in its capture list -- it is never
    // accessed on host builds.
    std::nullptr_t truncate_pool = nullptr;
#endif // MRA_ENABLE_HOST

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> node_e;
    ttg::Edge<mra::Key<NDIM>, kept_tensor_type> kept_e0, kept_e1, kept_e2, kept_e3,
                                                kept_e4, kept_e5, kept_e6, kept_e7;

    /**
     * Forwards every incoming node to the merge task below, and preemptively
     * feeds an empty "kept" placeholder for children that do not exist as
     * separate nodes in `in` (i.e. that were already leafs for all functions),
     * so the merge task at the parent does not wait forever on them.
     */
    auto dispatch_fn = [fns, name](const mra::Key<NDIM>& key,
                                   const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
      size_type N = fns->num_functions(key);
      //std::cout << "DISPATCH " << key << " empty=" << in.empty() << std::endl;
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(ttg::device::send<num_children>(key, in));
#else  // MRA_ENABLE_HOST
      ttg::send<num_children>(key, in);
#endif // MRA_ENABLE_HOST
      for (auto child : mra::children(key)) {
        bool is_all_leaf = true;
        const auto childidx = child.childindex();
        for (size_type i = 0; i < N && is_all_leaf; ++i) {
          is_all_leaf &= in.is_child_leaf(i, childidx);
        }
        if (is_all_leaf) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(select_send_up(child, kept_tensor_type(), std::make_index_sequence<num_children>{}, "truncate-dispatch"));
#else  // MRA_ENABLE_HOST
          select_send_up(child, kept_tensor_type(), std::make_index_sequence<num_children>{}, "truncate-dispatch");
#endif // MRA_ENABLE_HOST
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn),
                                          ttg::edges(in),
                                          ttg::edges(kept_e0, kept_e1, kept_e2, kept_e3,
                                                      kept_e4, kept_e5, kept_e6, kept_e7, node_e),
                                          name + "-dispatch");

    /**
     * Merges the "kept" signal from each child with this node's own norm to
     * decide, per function, whether this node's coefficients can be dropped.
     * Always forwards the (possibly truncated) node to `out`, and forwards
     * its own post-decision "kept" signal up to the parent (unless at the root).
     */
    auto truncate_fn = [fns, K, thresh, truncate_mode, cell_min_width, name,
                        enable_truncate_batching, truncate_pool](
                            const mra::Key<NDIM>& key,
                            const kept_tensor_type& child0, const kept_tensor_type& child1,
                            const kept_tensor_type& child2, const kept_tensor_type& child3,
                            const kept_tensor_type& child4, const kept_tensor_type& child5,
                            const kept_tensor_type& child6, const kept_tensor_type& child7,
                            mra::FunctionsCompressedNode<T, NDIM>&& node) -> TASKTYPE {
      assert(!node.empty());
      constexpr const size_type out_terminal = num_children;
      const size_type N = fns->num_functions(key);
      //std::cout << "MERGE " << key << " empty=" << node.empty() << std::endl;
      /* only functions where node is non-zero get a thread-block launched, so
       * pre-fill with 0 -- matches the keep=false the kernel used to compute
       * for a zero node directly (this buffer is read back by real function id
       * below and forwarded to the parent); scope::SyncIn (rather than
       * TempScope/Allocate) ensures this host fill reaches the device copy the
       * kernel reads/writes. */
      kept_tensor_type kept(N, ttg::scope::Allocate);
      std::array<const kept_tensor_type*, num_children> children =
            {&child0, &child1, &child2, &child3, &child4, &child5, &child6, &child7};

      const T tol = mra::truncate_tol(key, thresh, cell_min_width, truncate_mode);

#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(node.coeffs().buffer(), kept.buffer(),
                                      child0.buffer(), child1.buffer(), child2.buffer(), child3.buffer(),
                                      child4.buffer(), child5.buffer(), child6.buffer(), child7.buffer());
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

      std::array<const T*, num_children> child_kept;
      for (std::size_t c = 0; c < num_children; ++c) {
        child_kept[c] = children[c]->buffer().current_device_ptr();
      }

      auto node_view = node.coeffs().current_view();
      auto kept_view = kept.current_view();

#ifndef MRA_ENABLE_HOST
      if (enable_truncate_batching) {
        // truncate has no operator data at all -- not even tol is shared
        // across members (it depends on each member's own key.level()) --
        // so batching here is unrestricted. No upfront nonzero detection
        // either: the leader launches N blocks per member, same as the
        // unbatched path just below.
        auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, node_view, kept_view, child_kept, N, tol);
        detail::submit_truncate_batch_leader<T, NDIM>(batch, *truncate_pool);
      } else
#endif // MRA_ENABLE_HOST
      {
        submit_truncate_kernel(key, node_view, kept_view, child_kept, N, tol,
                              ttg::device::current_stream());
      }

#ifndef MRA_ENABLE_HOST
      // make sure the children are available on the host
      co_await ttg::device::wait(kept.buffer(), child0.buffer(), child1.buffer(), child2.buffer(), child3.buffer(),
                                  child4.buffer(), child5.buffer(), child6.buffer(), child7.buffer());
#endif // MRA_ENABLE_HOST

      std::array child_kept_host = {child0.view_on(ttg::device::Device::host()),
                                    child1.view_on(ttg::device::Device::host()),
                                    child2.view_on(ttg::device::Device::host()),
                                    child3.view_on(ttg::device::Device::host()),
                                    child4.view_on(ttg::device::Device::host()),
                                    child5.view_on(ttg::device::Device::host()),
                                    child6.view_on(ttg::device::Device::host()),
                                    child7.view_on(ttg::device::Device::host())};


      // set up the node's child info
      for (size_type i = 0; i < N; ++i) {
        for (size_type c = 0; c < num_children; ++c) {
          // if the child is not present we mark it as leaf
          if (child_kept_host[c].empty() || child_kept_host[c][i] == T(0.0)) {
            node.set_child_leaf(i, c, true);
          }
        }
      }


      // patch the (host-only) child-leaf metadata for every function we are not keeping,
      // so reconstruct stops recursing into the now-empty subtree below this node
      auto kept_host_view = kept.view_on(ttg::device::Device::host());
      bool keep = false;
      for (size_type i = 0; i < N; ++i) {
        std::cout << "TRUNCATE " << key << " fnid=" << i
                  << " kept=" << kept_host_view[i] << std::endl;
        if (kept_host_view[i] == T(0.0)) {
          node.set_zero(i);
        } else {
          keep = true;
        }
      }

#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      if (keep) {
        sends.push_back(ttg::device::send<num_children>(key, std::move(node)));
      }
      if (key.level() > 0) {
        sends.push_back(select_send_up(key, std::move(kept), std::make_index_sequence<num_children>{}, name.c_str()));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::send<num_children>(key, std::move(node));
      if (key.level() > 0) {
        select_send_up(key, std::move(kept), std::make_index_sequence<num_children>{}, name.c_str());
      }
#endif // MRA_ENABLE_HOST
    };

    auto truncate_tt = ttg::make_tt<Space>(std::move(truncate_fn),
                                          ttg::edges(kept_e0, kept_e1, kept_e2, kept_e3,
                                                      kept_e4, kept_e5, kept_e6, kept_e7, node_e),
                                          ttg::edges(kept_e0, kept_e1, kept_e2, kept_e3,
                                                      kept_e4, kept_e5, kept_e6, kept_e7, out),
                                          name);

    // set maps if provided
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      dispatch_tt->set_keymap(procmap);
      truncate_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      dispatch_tt->set_devicemap(devicemap);
      truncate_tt->set_devicemap(devicemap);
    }

#ifndef MRA_ENABLE_HOST
    if (enable_truncate_batching) {
      // Unrestricted matcher: any two truncate_tt tasks may batch together,
      // regardless of level or position -- see the batching-support comment
      // in mra/kernels/truncate.h for why.
      truncate_tt->set_batch_matcher(
          [](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
          max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(dispatch_tt->template in<0>());
    auto outs = std::make_tuple(truncate_tt->template out<8>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(dispatch_tt);
    ops[1] = std::move(truncate_tt);

    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_TRUNCATE_H
