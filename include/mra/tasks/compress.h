#ifndef MRA_TASKS_COMPRESS_H
#define MRA_TASKS_COMPRESS_H

#include <ttg.h>
#include "mra/kernels.h"
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
    DeviceMap&& devicemap = {})
  {
    static_assert(NDIM == 3); // TODO: worth fixing?

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
    std::shared_ptr<detail::BatchPoolRegistry<detail::CompressBatchArg<T, NDIM>>> compress_pool;
    if (enable_compress_batching) {
      compress_pool = std::make_shared<detail::BatchPoolRegistry<detail::CompressBatchArg<T, NDIM>>>(ttg::device::num_devices(), mra::get_batch_size());
    }
#else
    // BatchPoolRegistry only exists on device builds; this placeholder only
    // exists so the (shared host/device) do_compress lambda below can
    // unconditionally list compress_pool in its capture list -- it is never
    // accessed on host builds.
    std::nullptr_t compress_pool = nullptr;
#endif // MRA_ENABLE_HOST

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> filter_in(name + "-filter_in");

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
    // creates the right number of edges for nodes to flow from send_leafs_up to compress
    // send_leafs_up will select the right input for compress
    auto send_to_compress_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>{})..., filter_in);
      }(std::make_index_sequence<num_children>{});
    // output edges for the send_leafs_up tasks, one for each child
    auto send_leaves_up_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return ttg::edges((std::get<Is>(send_to_compress_edges))...);
    }(std::make_index_sequence<num_children>{});
    /* append out edge to set of edges */
    auto compress_out_edges = std::tuple_cat(send_leaves_up_edges, std::make_tuple(out));
    /* use the tuple variant to handle variable number of inputs while suppressing the output tuple */
    auto do_compress =
      [&, fns, K, is_ns, name, enable_compress_batching, compress_pool, total_functions](
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
                          const mra::FunctionsReconstructedNode<T,NDIM> &in // the node from the prior op
                          ) -> TASKTYPE {
      //const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
      //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
        size_type N = fns->num_functions(key);
        constexpr const auto num_children = mra::Key<NDIM>::num_children();
        constexpr const auto out_terminal_id = num_children;
        mra::FunctionsCompressedNode<T,NDIM> result(key, N); // The eventual result
        // create empty, may be reset if needed
        mra::FunctionsReconstructedNode<T, NDIM> p(key, N);

        //std::cout << name << " in " << in << std::endl;
        //std::cout << name << " in0 " << in0 << std::endl;
        //std::cout << name << " in1 " << in1 << std::endl;
        //std::cout << name << " in2 " << in2 << std::endl;
        //std::cout << name << " in3 " << in3 << std::endl;
        //std::cout << name << " in4 " << in4 << std::endl;
        //std::cout << name << " in5 " << in5 << std::endl;
        //std::cout << name << " in6 " << in6 << std::endl;
        //std::cout << name << " in7 " << in7 << std::endl;


        /* check if all inputs are empty */
        bool all_empty = in.empty() && in0.empty() && in1.empty() && in2.empty() && in3.empty() &&
                         in4.empty() && in5.empty() && in6.empty() && in7.empty();

        if (all_empty) {
          // Collect child leaf info
          mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
          //mra::apply_leaf_info(p, in, in0, in1, in2, in3, in4, in5, in6, in7);
          /* all data is still on the host so the coefficients are zero */
          for (std::size_t i = 0; i < N; ++i) {
            p.sum(i) = 0.0;
          }
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
          //std::cout << name << " " << key << " sparsity: " << sparsity << std::endl;

          // allocate the result
          result.allocate(sparsity, K, ttg::scope::Allocate);
          result.set_ns(is_ns);

          // Collect child leaf info
          mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
          //std::cout << name << " " << key << " result after apply_leaf_info " << result << " " << std::endl;

          /**
           * Allocate the reconstructed node to send up to the parent.
           */
          sparsity.set_all_zero(); // reset
          sparsity.nonzero_if_any(result, in); // set based on result and our input (to catch any new leafs we pick up from the input)
          mra::apply_leaf_info(p, in);

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

          const std::size_t tmp_size = compress_tmp_size<NDIM>(K)*n_nonzero;
          ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch(tmp_size, TempScope);
          const auto& hgT = functiondata.get_hgT();
          /* stores sumsq for each non-zero function; indexed by compact position, not fnid */
          auto d_sumsq = ttg::Buffer<T, DeviceAllocator<T>>(n_nonzero, TempScope);

          auto& d = result.coeffs();

#ifndef MRA_ENABLE_HOST
          auto input = ttg::device::Input(p.coeffs().buffer(), d.buffer(), hgT.buffer(),
                                          tmp_scratch, d_sumsq);
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

          auto in_view = in.coeffs().current_view();

          auto coeffs_view = p.coeffs().current_view();
          auto rcoeffs_view = d.current_view();
          auto hgT_view = hgT.current_view();

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
            // member's real function ids are found on-device.
            auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, in_view, coeffs_view, rcoeffs_view,
                                                                    tmp_scratch, d_sumsq, input_views,
                                                                    p.coeffs(), d, n_nonzero);
            // followers: the leader's batched launch already wrote our slice of p/result/d_sumsq.
            detail::submit_compress_batch_leader<T, NDIM>(batch, *compress_pool, K, is_ns, hgT_view, total_functions);
          } else
#endif // MRA_ENABLE_HOST
          {
            auto sparseman = make_sparsity_manager(d, p);
            sparseman.populate_device_sparsity();
            submit_compress_kernel(key, N, n_nonzero, K, is_ns,
                                  in_view, coeffs_view, rcoeffs_view, hgT_view,
                                  tmp_scratch.current_device_ptr(), d_sumsq.current_device_ptr(), input_views,
                                  ttg::device::current_stream());
          }
          norms.compute();
          /* wait for kernel and transfer sums back */
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::wait(d_sumsq, norms.buffer());
#endif
          norms.verify();

          // Explicitly zero every function's sum first, rather than relying
          // on FunctionsReconstructedNode's default-constructed 0.0 --
          // mirrors the prior kernel's explicit d_sumsq[fnid] = 0.0 for zero
          // functions. The loop below then overwrites the non-zero ones.
          for (size_type i = 0; i < N; ++i) {
            p.sum(i) = T(0);
          }

          // d_sumsq is compacted (indexed by compact position, not fnid) --
          // walk it by re-enumerating `sparsity`'s own non-zero ids (host-side,
          // O(#ranges + n_nonzero), same set find_nth_nonzero scanned
          // on-device).
          auto* d_sumsq_arr = d_sumsq.host_ptr();
          size_type pos = 0;
          for (auto it = sparsity.begin_nonzero(); it != sparsity.end_nonzero(); ++it, ++pos) {
            const size_type i = *it;
            auto sumsqs = std::array{in0.sum(i), in1.sum(i), in2.sum(i), in3.sum(i),
                                    in4.sum(i), in5.sum(i), in6.sum(i), in7.sum(i)};
            auto child_sumsq = std::reduce(sumsqs.begin(), sumsqs.end());
            p.sum(i) = d_sumsq_arr[pos] + child_sumsq; // result sumsq is last element in sumsqs
            //std::cout << name << " " << key << " fn " << i << "/" << N << " d_sumsq " << d_sumsq_arr[pos]
            //          << " child_sumsq " << child_sumsq << " sum " << p.sum(i) << std::endl;
          }
        }


        //std::cout << name << " " << key << " result " << result << " p " << p << std::endl;

        // Recur up
        if (key.level() > 0) {
          // will not return
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::forward(
            // select to which child of our parent we send
            select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress"),
            // Send result to output tree
            ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
            select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress");
            ttg::send<out_terminal_id>(key, std::move(result));
#endif
        } else {
          bool all_correct = true;
          for (std::size_t i = 0; i < N; ++i) {
            if (std::abs(p.sum(i) - 1.0) > 1e-12) {
              all_correct = false;
              std::cout << name << ": at root of compressed tree " << key.batch() << " fn " << i << ": total normsq is " << p.sum(i) << std::endl;
            }
          }
          if (all_correct) {
            std::cout << name << ": at root of compressed tree " << key.batch() << ": all norms are 1.0 with 1e-12 tolerance" << std::endl;
          }
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::forward(
            // Send result to output tree
            ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
          ttg::send<out_terminal_id>(key, std::move(result));
#endif
        }
    };

    auto ttt = std::make_tuple(ttg::make_tt<Space>(&do_send_leafs_up<T,NDIM>, edges(in), send_leaves_up_edges, "send_leaves_up"),
                               ttg::make_tt<Space>(std::move(do_compress), send_to_compress_edges, compress_out_edges, "compress"),
                               ttg::make_tt<Space>(std::move(filter_fn), ttg::edges(in), ttg::edges(filter_in), "filter"));

      // set maps if provided
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      std::get<0>(ttt)->set_keymap(procmap);
      std::get<1>(ttt)->set_keymap(procmap);
      std::get<2>(ttt)->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      std::get<0>(ttt)->set_devicemap(devicemap);
      std::get<1>(ttt)->set_devicemap(devicemap);
      std::get<2>(ttt)->set_devicemap(devicemap);
    }

#ifndef MRA_ENABLE_HOST
    if (enable_compress_batching) {
      // Unrestricted matcher: any two do_compress tasks may batch together,
      // regardless of level or position, up to max_batch_size -- safe because
      // compress's only shared operator data (hgT) never varies by level or
      // position to begin with (see kernels/compress.h's batching-support
      // comment), unlike convolution which needed a similar relaxation.
      std::get<1>(ttt)->set_batch_matcher(
          [](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
          max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(std::get<2>(ttt)->template in<0>());
    auto outs = std::make_tuple(std::get<1>(ttt)->template out<8>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(3);
    ops[0] = std::move(std::get<0>(ttt));
    ops[1] = std::move(std::get<1>(ttt));
    ops[2] = std::move(std::get<2>(ttt));

    return make_ttg(std::move(ops), ins, outs, name);
  }

}

#endif // MRA_TASKS_COMPRESS_H
