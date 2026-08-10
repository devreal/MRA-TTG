#ifndef MRA_TASKS_RECONSTRUCT_H
#define MRA_TASKS_RECONSTRUCT_H

#include <ttg.h>
#include <mutex>
#include <sstream>
#include <vector>
#include <iostream>
#include "mra/kernels.h"
#include "mra/misc/batch_size.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/options.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/functionset.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/functionnorm.h"
#include "mra/functors/gaussian.h"
#include "mra/functors/functionfunctor.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra{
  template <typename T, mra::Dimension NDIM, typename FunctionSetT, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_reconstruct(
    const std::shared_ptr<FunctionSetT>& fns,
    const std::size_t K,
    bool accumulate_NS,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name = "reconstruct",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    // Batching is controlled process-wide via mra::set_batch_size(), not per
    // call here -- see mra/misc/batch_size.h. Read once, at graph-construction
    // time: this bakes the decision into do_reconstruct/its matcher below, so
    // calling set_batch_size() again after this graph exists has no effect on it.
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_reconstruct_batching = mra::batching_enabled();
    // Total function count across the whole FunctionSet -- fixed for this
    // operation's entire run (unlike a single node's N, which varies with
    // key.batch()). Used to size the batch leader's sparsity-byte staging
    // buffer to a fixed upper bound so it never needs to grow after its
    // first allocation.
    const size_type total_functions = fns->num_functions();

#ifndef MRA_ENABLE_HOST
    // reconstruct's only "operator data" is hg, a single two-scale filter
    // matrix from FunctionData that never varies by level or position (see
    // mra/kernels/reconstruct.h's batching-support comment) -- so, like
    // compress, batching here is unrestricted from the start.
    std::shared_ptr<detail::BatchPoolRegistry<detail::ReconstructBatchArg<T, NDIM>>> reconstruct_pool;
    if (enable_reconstruct_batching) {
      reconstruct_pool = std::make_shared<detail::BatchPoolRegistry<detail::ReconstructBatchArg<T, NDIM>>>(ttg::device::num_devices(), mra::get_batch_size());
    }
#else
    // BatchPoolRegistry only exists on device builds; this placeholder only
    // exists so the (shared host/device) do_reconstruct lambda below can
    // unconditionally list reconstruct_pool in its capture list -- it is
    // never accessed on host builds.
    std::nullptr_t reconstruct_pool = nullptr;
#endif // MRA_ENABLE_HOST

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> S("S");  // passes scaling functions down
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> P("Primer"); // primer for root

    auto primer = [&, fns, name](const mra::Key<NDIM>& key,
                                 const mra::FunctionsCompressedNode<T, NDIM>& node) -> TASKTYPE {
      //std::cout << name << " primer " << key << std::endl;
      if (key.level() == 0) {
        /* root node: need to send an empty node as the parent to do_reconstruct */
        size_type N = fns->num_functions(key);
        auto r_empty = mra::FunctionsReconstructedNode<T,NDIM>(key, N);
        r_empty.set_all_leaf(LeafStatus::Inner);
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, std::move(r_empty));
#else
        ttg::send<0>(key, std::move(r_empty));
#endif
      }
    };

    auto p = ttg::make_tt<Space>(std::move(primer), ttg::edges(in), edges(P), "primer");

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) p->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) p->set_devicemap(devicemap);

    auto do_reconstruct = [&, fns, K, accumulate_NS, name, enable_reconstruct_batching, reconstruct_pool, total_functions](const mra::Key<NDIM>& key,
                                            const mra::FunctionsCompressedNode<T, NDIM>& node,
                                            const mra::FunctionsReconstructedNode<T, NDIM>& from_parent) -> TASKTYPE {
      size_type N = fns->num_functions(key);
      const auto& hg = functiondata.get_hg();
      mra::KeyChildren<NDIM> children(key);

      //std::cout << name << " " << key << " node " << node << " from_parent " << from_parent << std::endl;

      //std::cout << name << " " << key << " node norm " << normf(node.coeffs().current_view()) << " from_parent norm " << normf(from_parent.coeffs().current_view())  << std::endl;
#ifndef MRA_ENABLE_HOST
      // forward() returns a vector that we can push into
      auto sends = ttg::device::forward();
      auto do_send = [&]<std::size_t I, typename S>(auto& child, S&& node) {
            sends.push_back(ttg::device::send<I>(child, std::forward<S>(node)));
      };
#else
      auto do_send = []<std::size_t I, typename S>(auto& child, S&& node) {
        ttg::send<I>(child, std::forward<S>(node));
      };
#endif // MRA_ENABLE_HOST

      // array of child nodes
      std::array<mra::FunctionsReconstructedNode<T,NDIM>, mra::Key<NDIM>::num_children()> r_arr;
      std::array<SparsityInfo, mra::Key<NDIM>::num_children()> child_sparsity_arr;
      for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        auto& r = r_arr[it.index()];
        r = mra::FunctionsReconstructedNode<T,NDIM>(child, N);

        child_sparsity_arr[it.index()] = SparsityInfo(N, SparsityInfo::InitType::AllZero); // start with all zero, we'll set the non-zero ones as we go
        // collect leaf information
        for (std::size_t i = 0; i < N; ++i) {
          //std::cout << name << " " << key << " child " << child << " function " << i
          //          << " from_parent is_invalid " << from_parent.is_invalid(i)
          //          << " from_parent is_leaf " << from_parent.is_leaf(i)
          //          << " node is_child_leaf " << node.is_child_leaf(i, *it)
          //          << std::endl;
          if (from_parent.is_invalid(i) || from_parent.is_leaf(i)) {
            //std::cout << name << " " << key << " child " << child << " function " << i
            //          << " from_parent is_invalid " << from_parent.is_invalid(i)
            //          << " or leaf " << from_parent.is_leaf(i) << " so child is invalid" << std::endl;
            r.set_leaf(i, LeafStatus::Invalid); // parent is invalid, so the child must be too
          //} else if (node.is_zero(i)) {
          //  std::cout << name << " " << key << " child " << child << " function " << i
          //            << " node is zero so child is invalid" << std::endl;
          //  // parent is a leaf and the compressed node is zero, so the child must be a invalid
          //  r.set_leaf(i, LeafStatus::Invalid); // node is zero, so the child must be a leaf (but not necessarily invalid)
          } else if (node.is_child_leaf(i, *it)) {
            //std::cout << name << " " << key << " child " << child << " function " << i
            //          << " node is child leaf so child is leaf" << std::endl;
            // parent is not a leaf/invalid, the compressed node has coefficients, and its child is empty, so we are the leaf
            r.set_leaf(i, LeafStatus::Leaf);
            child_sparsity_arr[it.index()].set_nonzero(i);
          } else {
            //std::cout << name << " " << key << " child " << child << " function " << i
            //          << " child is inner" << std::endl;
            // parent is not a leaf/invalid and the compressed node is not empty, so we are the inner node
            r.set_leaf(i, LeafStatus::Inner);
            child_sparsity_arr[it.index()].set_nonzero(i);
          }
        }
        /**
         * Sanity check: if this is the last node in reconstructed form then the child of the compressed node must be empty.
         * TOOD: Is this sanity check valid? It appears that convolution may produce empty compressed nodes with non-empty children...
         */
        if (r.is_all_leaf_or_invalid()) {
          //assert(node.is_child_empty(*it) &&
          //       "if a reconstructed child is all leaf or invalid, the corresponding compressed node child should be empty");
        }
      }

#if 0
      if (node.empty() && from_parent.empty()) {
        //std::cout << "reconstruct " << key << " node and parent empty " << std::endl;
        /* both the node and the parent are empty so we can shortcut with empty results */
        for (auto it=children.begin(); it!=children.end(); ++it) {
          const mra::Key<NDIM> child= *it;
          auto& r = r_arr[it.index()];
          if (r.is_all_leaf_or_invalid()) {
            do_send.template operator()<1>(child, std::move(r));
          } else {
            do_send.template operator()<0>(child, std::move(r));
          }
        }
#ifndef MRA_ENABLE_HOST
        // won't return
        co_await std::move(sends);
        assert(0);
#else  // MRA_ENABLE_HOST
        return; // we're done
#endif // MRA_ENABLE_HOST
      }
#endif // 0

      /**
       * once we are here we know we need to invoke the reconstruct kernel
       * TODO: skip the kernel if the node and from_parent are both empty. We will just pass down empty child nodes in that case.
       */

      /**
       * The result that contains the coefficients of leaf nodes only.
       * We cannot forward the full child nodes because they might contain coefficients
       * for non-leaf nodes that we don't want to leave the reconstruct operation.
       */
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero); // start with all zero, we'll set the non-zero ones as we go
      for (std::size_t i = 0; i < N; ++i) {
        if (from_parent.is_leaf(i)) {
          // leafs are nonzero
          sparsity.set_nonzero(i);
        }
      }
      mra::FunctionsReconstructedNode<T,NDIM> result(key, sparsity, K, ttg::scope::Allocate);
      mra::apply_leaf_info(result, from_parent);

      /* populate the vector of r's
      * TODO: TTG/PaRSEC supports only a limited number of inputs so for higher dimensions
      *       we may have to consolidate the r's into a single buffer and pick them apart afterwards.
      *       That will require the ability to ref-count 'parent buffers'. */
      for (int i = 0; i < key.num_children(); ++i) {
        r_arr[i].allocate(child_sparsity_arr[i], K, ttg::scope::Allocate);
      }

      // Work sparsity for this node: the kernel must visit fnid whenever ANY
      // of node/from_parent (value sparsity: does this function have
      // non-zero coefficients here) OR result/r_arr[0..7] (status-based
      // sparsity: result is non-zero iff from_parent.is_leaf(i), each
      // r_arr[c] is non-zero iff from_parent is Inner for that child) is
      // non-zero. result/r_arr's criteria are independent of node/from_parent's
      // *value* sparsity -- a leaf or inner position can have an exactly-zero
      // coefficient and still need its output slot populated -- so folding
      // only node/from_parent in here (as this used to) leaves some of
      // result/r_arr's allocated slots outside the launch's own coverage,
      // hence never visited/written (see reconstruct_verify_sparsity_kernel's
      // "outside union" check in mra/kernels/reconstruct.h). Must run after
      // result/r_arr are allocated (above), since it needs their sparsity.
      SparsityInfo work_sparsity(N, SparsityInfo::InitType::AllZero);
      [&]<std::size_t... Is>(std::index_sequence<Is...>){
        work_sparsity.nonzero_if_any(node, from_parent, result, r_arr[Is]...);
      }(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
      const size_type n_nonzero = work_sparsity.count_nonzero();
      const std::size_t tmp_size = reconstruct_tmp_size<NDIM>(K)*n_nonzero;
      ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch(tmp_size, TempScope);

#ifdef MRA_CHECK_NORMS
      // DEBUG: host-side breakdown of the exact same views the device-side
      // verify kernel checks, keyed the same way (key + translation), so a
      // device-side mismatch can be cross-referenced against the TRUE host
      // bits for the same key -- tells us whether a residual discrepancy is
      // a real host/device coherence gap (host bits differ from what the
      // device reports for node/from_parent) or a remaining logic gap in
      // this widened union (host bits agree with what SHOULD be there, but
      // n_nonzero/the device scan still doesn't match).
      {
        static std::mutex dbg_mtx_host_breakdown;
        std::lock_guard<std::mutex> lg(dbg_mtx_host_breakdown);
        std::ostringstream oss;
        oss << "RECONSTRUCT-HOST-BREAKDOWN key=(" << key.level() << ",["
            << key.translation()[0] << "," << key.translation()[1] << "," << key.translation()[2]
            << "]) n_nonzero=" << n_nonzero << " N=" << N << "\n";
        for (size_type i = 0; i < N; ++i) {
          oss << "  i=" << i
              << " node=" << (int)node.sparsity().is_nonzero(i)
              << " from_parent=" << (int)from_parent.sparsity().is_nonzero(i)
              << " result=" << (int)result.sparsity().is_nonzero(i)
              << " r_arr=[";
          for (std::size_t c = 0; c < mra::Key<NDIM>::num_children(); ++c) {
            oss << (int)r_arr[c].sparsity().is_nonzero(i) << (c+1<mra::Key<NDIM>::num_children() ? "," : "");
          }
          oss << "]\n";
        }
        std::cout << oss.str() << std::flush;
      }
#endif // MRA_CHECK_NORMS

      // compute norms
      auto norms = [&]<std::size_t... Is>(std::index_sequence<Is...>){
        return FunctionNorms(name, node, from_parent, r_arr[Is]...);
      }(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});

#ifndef MRA_ENABLE_HOST
      // pick apart the array of r's into individual buffers for the kernel
      auto inputs = [&]<std::size_t... Is>(std::index_sequence<Is...>){
        return ttg::device::Input(hg.buffer(), tmp_scratch,
                                  (r_arr[Is].coeffs().buffer())...);
      }(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
      if (!node.empty()) inputs.add(node.coeffs().buffer());
      if (!result.empty()) inputs.add(result.coeffs().buffer());
      if (!from_parent.empty()) inputs.add(from_parent.coeffs().buffer());
      if (!norms.buffer().empty()) inputs.add(norms.buffer());
      /* select a device */
      co_await ttg::device::select(inputs);
#endif


      // pick apart the std::array
      auto r_ptrs = [&]<std::size_t... Is>(std::index_sequence<Is...>){
                        return std::array{(r_arr[Is].coeffs().current_view())...};
                      }(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
      auto node_view = node.coeffs().current_view();
      auto hg_view = hg.current_view();
      auto from_parent_view = from_parent.coeffs().current_view();
      auto result_view = result.coeffs().current_view();
      // DEBUG: compare node's host-tracked sparsity (RangeSparsityBase,
      // set at allocation/producer time) against node_view's device-side
      // inline bitmap, right after select() and before coop() ever runs --
      // i.e. before batching could have any opportunity to touch it.
#if defined(MRA_CHECK_NORMS)
      if (!node.empty() && node_view.storage() != nullptr) {
        static std::mutex dbg_mtx_pre;
        std::lock_guard<std::mutex> lg(dbg_mtx_pre);
        cudaDeviceSynchronize();
        std::vector<unsigned char> devbytes(N);
        cudaMemcpy(devbytes.data(), node_view.storage(), N, cudaMemcpyDeviceToHost);
        size_type dev_nz = 0;
        for (size_type i = 0; i < N; ++i) if (devbytes[i] & 1) ++dev_nz;
        size_type host_nz = node.sparsity().count_nonzero();
        if (dev_nz != host_nz) {
          std::ostringstream oss;
          oss << "RECONSTRUCT-PRE-COOP MISMATCH key=" << key << " N=" << N
              << " node_view.storage()=" << (void*)node_view.storage()
              << " host_nz=" << host_nz << " dev_nz=" << dev_nz
              << " dev_bytes=[";
          for (size_type i = 0; i < N; ++i) oss << (unsigned)devbytes[i] << (i+1<N?",":"");
          oss << "]\n";
          std::cout << oss.str() << std::flush;
        }
      }
#endif // MRA_CHECK_NORMS
#ifndef MRA_ENABLE_HOST
      if (enable_reconstruct_batching) {
        // key travels through coop() since reconstruct_kernel_impl reads
        // key.level() -- see the batching-support comment in
        // kernels/reconstruct.h. node_view/from_parent_view/r_ptrs/result_view
        // keep the exact roles submit_reconstruct_kernel uses below.
        // r_arr/result (the real tensors, not just their views) travel
        // through too, so the batch leader can read their sparsity and
        // aggregate every member's bytes into one pinned buffer + one H2D
        // copy instead of each member pushing its own via SparsityManager here.
        // n_nonzero (this member's own, computed above independent of
        // batching) travels through as well, so the leader can flatten
        // every member's own non-zero work items into one combined 1D
        // launch (see submit_reconstruct_batch_leader).
        auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, node_view, tmp_scratch,
                                                                from_parent_view, r_ptrs, result_view,
                                                                r_arr, result, n_nonzero);
        // followers: the leader's batched launch already wrote our slice of r_arr/result.
        detail::submit_reconstruct_batch_leader<T, NDIM>(batch, *reconstruct_pool, K, accumulate_NS, hg_view, total_functions);
      } else
#endif // MRA_ENABLE_HOST
      {
        auto sparseman = make_sparsity_manager(r_arr, result);
        sparseman.populate_device_sparsity();
        submit_reconstruct_kernel(key, N, n_nonzero, K, accumulate_NS, node_view, hg_view, from_parent_view,
                                  r_ptrs, result_view, tmp_scratch.current_device_ptr(),
                                  ttg::device::current_stream());
      }

#ifdef MRA_CHECK_NORMS
      norms.compute();
#ifndef MRA_ENABLE_HOST
    /* wait for norms to come back and verify */
      co_await ttg::device::wait(norms.buffer());
#endif // MRA_ENABLE_HOST
      norms.verify();
#endif // MRA_CHECK_NORMS

      // send result to the output
      //std::cout << name << " " << key << " result " << result << std::endl;
      do_send.template operator()<1>(key, std::move(result));

      /**
       * For each child, either recurse down (if there not all nodes are leaf/invalid) or send to the output (if they are all leaf/invalid).
       * Note that we have to wait until the end to send to the output because we need to have the full child node ready to determine if it's all leaf/invalid or not.
       */
      for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        mra::FunctionsReconstructedNode<T,NDIM>& r = r_arr[it.index()];
        r.key() = child;
        //std::cout << name << " " << key << " child " << r << std::endl;
        if (r.is_all_leaf_or_invalid()) {
          // if the child is all leaf or invalid then we can send it directly to the output
          do_send.template operator()<1>(child, std::move(r));
        } else {
          // recurse down
          do_send.template operator()<0>(child, std::move(r));
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };


    auto s = ttg::make_tt<Space>(std::move(do_reconstruct),
                                 ttg::edges(in, ttg::fuse(S, P)), // inputs
                                 ttg::edges(S, out),              // outputs
                                 name, {"input", "s/p"}, {"s", "output"});

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) s->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) s->set_devicemap(devicemap);

#ifndef MRA_ENABLE_HOST
    if (enable_reconstruct_batching) {
      s->set_batch_matcher(
          [](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
          max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    /* assemble the Reconstruct TTG */
    auto ins = std::make_tuple(s->template in<0>(), s->template in<0>());
    auto outs = std::make_tuple(s->template out<0>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(s);
    ops[1] = std::move(p);

    return ttg::make_ttg(std::move(ops), std::move(ins), std::move(outs), std::string(name));
  }
} // namespace mra

#endif // MRA_TASKS_RECONSTRUCT_H
