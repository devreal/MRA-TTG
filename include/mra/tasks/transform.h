#ifndef MRA_TASKS_TRANSFORM_H
#define MRA_TASKS_TRANSFORM_H

#include <array>
#include <memory>
#include <ttg.h>
#include <utility>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/batch_size.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/functionnode.h"
#include "mra/kernels/transform_functions.h"
#include "mra/tasks/common.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Linear combination of a set of functions in compressed form,
   *   out_j = sum_i U(i,j) in_i,   i < N, j < M
   * (cf. madness::transform). U is an N x M row-major matrix. Its host copy is
   * read to determine the structure of the result, so it must not be modified
   * while the graph executes.
   *
   * The tree of out_j is the union of the trees of the in_i with U(i,j) != 0.
   * To keep the output consistent (a node is sent if and only if its parent
   * marks it as a child of at least one function), every task passes the set
   * of output functions present below each child down to it. Children that
   * receive input nodes always get such a mask, even if no output function is
   * present there, so that every input node is consumed.
   *
   * The input must be compressed, not non-standard. No screening is applied;
   * truncate the result if needed.
   */
  template <typename T, Dimension NDIM, typename FunctionSetIn, typename FunctionSetOut,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_transform(const std::shared_ptr<FunctionSetIn>& fns_in,
                      const std::shared_ptr<FunctionSetOut>& fns_out,
                      size_type K,
                      const std::shared_ptr<mra::DenseTensor<T, 2>> U,
                      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
                      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
                      const std::string& name = "transform",
                      ProcMap procmap = {},
                      DeviceMap devicemap = {})
  {
    using node_type = mra::FunctionsCompressedNode<T, NDIM>;
    /* per output function: LeafStatus::Inner if present at this key, LeafStatus::Invalid if not */
    using mask_type = mra::DenseTensor<LeafStatus, 1>;
    static constexpr const int num_children = mra::Key<NDIM>::num_children();

    if (U->dim(0) != fns_in->num_functions() || U->dim(1) != fns_out->num_functions()) {
      throw std::runtime_error(name + ": U must be (number of input functions) x (number of output functions)");
    }

    /* batching is controlled process-wide (mra/misc/batch_size.h), read once here; U is
     * the same for every node, so any two tasks may be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type total_functions = fns_out->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::TransformFunctionsBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::TransformFunctionsBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    ttg::Edge<mra::Key<NDIM>, node_type> node_e;
    ttg::Edge<mra::Key<NDIM>, mask_type> mask_e;

    /**
     * Forward the input nodes and seed the root with all output functions present.
     */
    auto dispatch_fn = [fns_out](const mra::Key<NDIM>& key, const node_type& node) -> TASKTYPE {
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(ttg::device::send<0>(key, node));
#else  // MRA_ENABLE_HOST
      ttg::send<0>(key, node);
#endif // MRA_ENABLE_HOST
      if (key.level() == 0) {
        const size_type M = fns_out->num_functions(key);
        mask_type mask(M);
        std::fill(mask.buffer().host_ptr(), mask.buffer().host_ptr() + M, LeafStatus::Inner);
#ifndef MRA_ENABLE_HOST
        sends.push_back(ttg::device::send<1>(key, std::move(mask)));
#else  // MRA_ENABLE_HOST
        ttg::send<1>(key, std::move(mask));
#endif // MRA_ENABLE_HOST
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn),
                                           ttg::edges(in), ttg::edges(node_e, mask_e),
                                           name + "-dispatch");

    auto transform_fn = [fns_in, fns_out, K, U, name, enable_batching, batch_pool, total_functions](
                                                      const mra::Key<NDIM>& key,
                                                      const node_type& in_node,
                                                      const mask_type& mask) -> TASKTYPE {
      if (in_node.is_ns()) {
        throw std::runtime_error(name + ": input must be compressed, not non-standard");
      }
      const size_type N = fns_in->num_functions(key);
      const size_type M = fns_out->num_functions(key);
      const T* Uh = U->buffer().host_ptr();
      const LeafStatus* mask_h = mask.buffer().host_ptr();

      /* output functions with a non-zero node here, and with children below here */
      SparsityInfo sparsity(M, SparsityInfo::InitType::AllZero);
      std::vector<std::array<bool, num_children>> has_child(M);
      bool any_present = false;
      for (size_type j = 0; j < M; ++j) {
        has_child[j].fill(false);
        if (mask_h[j] != LeafStatus::Inner) continue;
        any_present = true;
        for (size_type i = 0; i < N; ++i) {
          if (Uh[i*M + j] == T(0)) continue;
          if (!in_node.is_zero(i)) sparsity.set_nonzero(j);
          for (int c = 0; c < num_children; ++c) {
            has_child[j][c] = has_child[j][c] || !in_node.is_child_leaf(i, c);
          }
        }
      }

#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
#endif // MRA_ENABLE_HOST

      /* every child that receives input nodes gets a mask */
      for (auto child : children(key)) {
        if (in_node.is_all_child_leaf(child)) continue;
        mask_type child_mask(M);
        LeafStatus* cm = child_mask.buffer().host_ptr();
        for (size_type j = 0; j < M; ++j) {
          cm[j] = has_child[j][child.childindex()] ? LeafStatus::Inner : LeafStatus::Invalid;
        }
#ifndef MRA_ENABLE_HOST
        sends.push_back(ttg::device::send<1>(child, std::move(child_mask)));
#else  // MRA_ENABLE_HOST
        ttg::send<1>(child, std::move(child_mask));
#endif // MRA_ENABLE_HOST
      }

      if (any_present) {
        const size_type n_nonzero = sparsity.count_nonzero();
        node_type out = (n_nonzero > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate)
                                        : node_type(key, M);
        for (size_type j = 0; j < M; ++j) {
          for (int c = 0; c < num_children; ++c) {
            out.set_child_leaf(j, c, !has_child[j][c]);
          }
        }
        if (n_nonzero > 0) {
#ifndef MRA_ENABLE_HOST
          auto input = ttg::device::Input(out.coeffs().buffer(), std::as_const(*U).buffer());
          if (!in_node.empty()) input.add(in_node.coeffs().buffer());
          co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
          auto in_view = in_node.coeffs().current_view();
          auto out_view = out.coeffs().current_view();
          const T* U_ptr = std::as_const(*U).buffer().current_device_ptr();
#ifndef MRA_ENABLE_HOST
          if (enable_batching) {
            /* the output's sparsity is pushed by the leader along with the batch */
            auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(in_view, out_view, N, M, out.coeffs(), n_nonzero);
            detail::submit_transform_functions_batch_leader<T, NDIM>(batch, *batch_pool, U_ptr, K, total_functions);
          } else
#endif // MRA_ENABLE_HOST
          {
            auto sparseman = make_sparsity_manager(out);
            sparseman.populate_device_sparsity();
            submit_transform_functions_kernel<T, NDIM>(key, in_view, out_view, U_ptr, N, M, n_nonzero, K,
                                                       ttg::device::current_stream());
          }
        }
#ifndef MRA_ENABLE_HOST
        sends.push_back(ttg::device::send<0>(key, std::move(out)));
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, std::move(out));
#endif // MRA_ENABLE_HOST
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto transform_tt = ttg::make_tt<Space>(std::move(transform_fn),
                                            ttg::edges(node_e, mask_e),
                                            ttg::edges(out, mask_e),
                                            name, {"node", "mask"}, {"out", "mask"});

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      dispatch_tt->set_keymap(procmap);
      transform_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      dispatch_tt->set_devicemap(devicemap);
      transform_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      transform_tt->set_batch_matcher([](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
                                      max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(dispatch_tt->template in<0>());
    auto outs = std::make_tuple(transform_tt->template out<0>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(dispatch_tt);
    ops[1] = std::move(transform_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_TRANSFORM_H
