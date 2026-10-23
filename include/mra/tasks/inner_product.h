#ifndef MRA_TASKS_INNER_PRODUCT_H
#define MRA_TASKS_INNER_PRODUCT_H

#include <array>
#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/batch_size.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/functionnode.h"
#include "mra/kernels/inner_product.h"
#include "mra/tasks/common.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

/**
 * Inner products between two sets of functions in compressed form:
 *  - make_inner:        r[i]   = <a_i|b_i>   (both sets have N functions)
 *  - make_matrix_inner: r[i,j] = <a_i|b_j>   (N x M)
 *
 * The result is sent on `result` at the root key (level 0) once the
 * reduction over the tree has completed.
 *
 * The two trees are walked together. Where only one of them has children, the
 * other side is fed empty nodes (as in make_gaxpy) so that every incoming node
 * is consumed. Each node's contribution is computed by a reduction task that
 * also sums the partial results of its children; children that exist in
 * neither tree get an empty placeholder.
 *
 * The inputs must be compressed (not non-standard) trees, as produced by
 * make_compress(..., false, ...) or make_truncate: only the root holds scaling
 * coefficients, so the inner product is the sum over all nodes of the
 * coefficient dot products.
 */

namespace mra {

  namespace detail {

    /* Matrix: whether the result is <a_i|b_j> (N x M) rather than <a_i|b_i> (N) */
    template <typename T, Dimension NDIM, bool Matrix, typename ResultT, typename FnsA, typename FnsB,
              typename MakeResult, typename Submit, typename ProcMap, typename DeviceMap>
    auto make_inner_product_impl(
      const std::shared_ptr<FnsA>& fnsA,
      const std::shared_ptr<FnsB>& fnsB,
      size_type K,
      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> inA,
      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> inB,
      ttg::Edge<mra::Key<NDIM>, ResultT> result,
      MakeResult&& make_result,
      Submit&& submit,
      const std::string& name,
      ProcMap procmap,
      DeviceMap devicemap)
    {
      using node_type = mra::FunctionsCompressedNode<T, NDIM>;
      static constexpr const int num_children = mra::Key<NDIM>::num_children();
      constexpr const int NODE_A = num_children;     // output terminals of the pair task
      constexpr const int NODE_B = num_children + 1;
      constexpr const int BALANCE_A = num_children + 2;
      constexpr const int BALANCE_B = num_children + 3;

      ttg::Edge<mra::Key<NDIM>, node_type> SA, SB;   // empty nodes to balance the trees
      ttg::Edge<mra::Key<NDIM>, node_type> nodeA_e, nodeB_e;
      ttg::Edge<mra::Key<NDIM>, ResultT> r0, r1, r2, r3, r4, r5, r6, r7;

      /* batching is controlled process-wide (mra/misc/batch_size.h), read once here;
       * there is no operator data, so any two reduction tasks may be batched */
      const std::size_t max_batch_size = mra::get_batch_size();
      const bool enable_batching = mra::batching_enabled();
#ifndef MRA_ENABLE_HOST
      std::shared_ptr<GroupedBatchPoolRegistry<InnerBatchArg<T, NDIM>>> batch_pool;
      if (enable_batching) {
        batch_pool = std::make_shared<GroupedBatchPoolRegistry<InnerBatchArg<T, NDIM>>>(
                       ttg::device::num_devices(), max_batch_size);
      }
#else  // MRA_ENABLE_HOST
      std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

      /**
       * Pair the nodes of both trees: forward them to the reduction task, balance
       * the trees and send empty placeholders for children that exist in neither.
       */
      auto pair_fn = [fnsA, fnsB, name](const mra::Key<NDIM>& key,
                                        const node_type& a,
                                        const node_type& b) -> TASKTYPE {
#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward(ttg::device::send<NODE_A>(key, a),
                                          ttg::device::send<NODE_B>(key, b));
#else  // MRA_ENABLE_HOST
        ttg::send<NODE_A>(key, a);
        ttg::send<NODE_B>(key, b);
#endif // MRA_ENABLE_HOST
        const size_type NA = fnsA->num_functions(key);
        const size_type NB = fnsB->num_functions(key);
        std::vector<mra::Key<NDIM>> balance_a, balance_b;
        for (auto child : children(key)) {
          const bool a_leaf = a.is_all_child_leaf(child);
          const bool b_leaf = b.is_all_child_leaf(child);
          if (a_leaf && b_leaf) {
#ifndef MRA_ENABLE_HOST
            sends.push_back(select_send_up(child, ResultT(), std::make_index_sequence<num_children>{}, name.c_str()));
#else  // MRA_ENABLE_HOST
            select_send_up(child, ResultT(), std::make_index_sequence<num_children>{}, name.c_str());
#endif // MRA_ENABLE_HOST
          } else if (a_leaf) {
            balance_a.push_back(child);
          } else if (b_leaf) {
            balance_b.push_back(child);
          }
        }
        if (!balance_a.empty()) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::broadcast<BALANCE_A>(std::move(balance_a), node_type(key, NA)));
#else  // MRA_ENABLE_HOST
          ttg::broadcast<BALANCE_A>(std::move(balance_a), node_type(key, NA));
#endif // MRA_ENABLE_HOST
        }
        if (!balance_b.empty()) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::broadcast<BALANCE_B>(std::move(balance_b), node_type(key, NB)));
#else  // MRA_ENABLE_HOST
          ttg::broadcast<BALANCE_B>(std::move(balance_b), node_type(key, NB));
#endif // MRA_ENABLE_HOST
        }
#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      };

      auto pair_tt = ttg::make_tt<Space>(std::move(pair_fn),
                                         ttg::edges(ttg::fuse(SA, inA), ttg::fuse(SB, inB)),
                                         ttg::edges(r0, r1, r2, r3, r4, r5, r6, r7, nodeA_e, nodeB_e, SA, SB),
                                         name + "-pair");

      /**
       * Compute this node's contribution, add the children's partial results
       * and pass the sum on to the parent (or to `result` at the root).
       */
      auto reduce_fn = [fnsA, fnsB, K, make_result, submit, name, enable_batching, batch_pool](
                          const mra::Key<NDIM>& key,
                          const ResultT& c0, const ResultT& c1, const ResultT& c2, const ResultT& c3,
                          const ResultT& c4, const ResultT& c5, const ResultT& c6, const ResultT& c7,
                          const node_type& a, const node_type& b) -> TASKTYPE {
        if (a.is_ns() || b.is_ns()) {
          throw std::runtime_error(name + ": inputs must be compressed, not non-standard");
        }
        const size_type NA = fnsA->num_functions(key);
        const size_type NB = fnsB->num_functions(key);
        ResultT r = make_result(NA, NB);
        std::array<const ResultT*, num_children> children = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7};

#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(r.buffer());
        if (!a.empty()) input.add(a.coeffs().buffer());
        if (!b.empty()) input.add(b.coeffs().buffer());
        for (auto c : children) {
          if (!c->empty()) input.add(c->buffer());
        }
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

        std::array<const T*, num_children> child_results;
        for (int c = 0; c < num_children; ++c) {
          child_results[c] = children[c]->empty() ? nullptr : children[c]->buffer().current_device_ptr();
        }
        auto a_view = a.coeffs().current_view(), b_view = b.coeffs().current_view();
        T* r_ptr = r.buffer().current_device_ptr();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(a_view, b_view, r_ptr, child_results, NA, NB);
          submit_inner_batch_leader<T, NDIM, Matrix>(batch, *batch_pool, K);
        } else
#endif // MRA_ENABLE_HOST
        {
          submit(key, a_view, b_view, r_ptr, child_results, NA, NB, ttg::device::current_stream());
        }

#ifndef MRA_ENABLE_HOST
        if (key.level() == 0) {
          co_await ttg::device::send<num_children>(key, std::move(r));
        } else {
          co_await select_send_up(key, std::move(r), std::make_index_sequence<num_children>{}, name.c_str());
        }
#else  // MRA_ENABLE_HOST
        if (key.level() == 0) {
          ttg::send<num_children>(key, std::move(r));
        } else {
          select_send_up(key, std::move(r), std::make_index_sequence<num_children>{}, name.c_str());
        }
#endif // MRA_ENABLE_HOST
      };

      auto reduce_tt = ttg::make_tt<Space>(std::move(reduce_fn),
                                           ttg::edges(r0, r1, r2, r3, r4, r5, r6, r7, nodeA_e, nodeB_e),
                                           ttg::edges(r0, r1, r2, r3, r4, r5, r6, r7, result),
                                           name);

      if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
        pair_tt->set_keymap(procmap);
        reduce_tt->set_keymap(procmap);
      }
      if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
        pair_tt->set_devicemap(devicemap);
        reduce_tt->set_devicemap(devicemap);
      }
#ifndef MRA_ENABLE_HOST
      if (enable_batching) {
        reduce_tt->set_batch_matcher([](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
                                     max_batch_size);
      }
#endif // MRA_ENABLE_HOST

      auto ins = std::make_tuple(pair_tt->template in<0>(), pair_tt->template in<1>());
      auto outs = std::make_tuple(reduce_tt->template out<num_children>());
      std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
      ops[0] = std::move(pair_tt);
      ops[1] = std::move(reduce_tt);
      return make_ttg(std::move(ops), ins, outs, name);
    }
  } // namespace detail

  /**
   * r[i] = <a_i|b_i> for two sets of N functions in compressed form.
   * The result (size N) is sent on `result` at the root key.
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_inner(const std::shared_ptr<FunctionSetT>& fns,
                  size_type K,
                  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
                  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
                  ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> result,
                  const std::string& name = "inner",
                  ProcMap procmap = {},
                  DeviceMap devicemap = {})
  {
    using result_type = mra::DenseTensor<T, 1>;
    auto make_result = [](size_type NA, size_type NB) {
      if (NA != NB) throw std::runtime_error("inner: both sets need the same number of functions");
      // every entry is written by the kernel
      return result_type(NA, ttg::scope::Allocate);
    };
    auto submit = [K](const mra::Key<NDIM>& key, const auto& a, const auto& b, T* r,
                      const std::array<const T*, mra::Key<NDIM>::num_children()>& children,
                      size_type NA, size_type NB, ttg::device::Stream stream) {
      submit_inner_kernel<T, NDIM>(key, a, b, r, children, NA, K, stream);
    };
    return detail::make_inner_product_impl<T, NDIM, false, result_type>(
      fns, fns, K, a, b, result, std::move(make_result), std::move(submit), name, procmap, devicemap);
  }

  /**
   * r[i,j] = <a_i|b_j> for a set of N and a set of M functions in compressed form.
   * The result (N x M, row-major) is sent on `result` at the root key.
   * For the overlap matrix of a single set, pass the same edge as `a` and `b`.
   */
  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_matrix_inner(const std::shared_ptr<FunctionSetA>& fnsA,
                         const std::shared_ptr<FunctionSetB>& fnsB,
                         size_type K,
                         ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
                         ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
                         ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>> result,
                         const std::string& name = "matrix_inner",
                         ProcMap procmap = {},
                         DeviceMap devicemap = {})
  {
    using result_type = mra::DenseTensor<T, 2>;
    auto make_result = [](size_type NA, size_type NB) {
      // every entry is written by the kernel
      return result_type(std::array<size_type, 2>{NA, NB}, ttg::scope::Allocate);
    };
    auto submit = [K](const mra::Key<NDIM>& key, const auto& a, const auto& b, T* r,
                      const std::array<const T*, mra::Key<NDIM>::num_children()>& children,
                      size_type NA, size_type NB, ttg::device::Stream stream) {
      submit_matrix_inner_kernel<T, NDIM>(key, a, b, r, children, NA, NB, K, stream);
    };
    return detail::make_inner_product_impl<T, NDIM, true, result_type>(
      fnsA, fnsB, K, a, b, result, std::move(make_result), std::move(submit), name, procmap, devicemap);
  }

} // namespace mra

#endif // MRA_TASKS_INNER_PRODUCT_H
