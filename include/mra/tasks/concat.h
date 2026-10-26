#ifndef MRA_TASKS_CONCAT_H
#define MRA_TASKS_CONCAT_H

#include <vector>
#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/batch_size.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/tensor/functionnode.h"
#include "mra/kernels/concat.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Concatenate two sets of functions, in compressed form: the result holds the
   * NA functions of `a` followed by the NB functions of `b`. Its tree is the union
   * of both trees; where one set's tree has ended, that set's functions are absent
   * (zero), and the parent sends an empty node in place of the missing one, as
   * make_gaxpy does. The coefficients are copied: the result never shares nodes
   * with its inputs, which may therefore be fed with keep (see FunctionVector).
   *
   * Use it to grow a set, e.g., a history of iterates for an extrapolation, that
   * later operations (make_transform, make_matrix_inner) treat as one set.
   */
  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_concat(
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
    const std::string& name = "concat",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    using node_type = mra::FunctionsCompressedNode<T, NDIM>;
    constexpr const int OUT = 0, BALANCE_A = 1, BALANCE_B = 2;
    ttg::Edge<mra::Key<NDIM>, node_type> SA(name + "-balance_a"), SB(name + "-balance_b");

    /* batching is controlled process-wide (mra/misc/batch_size.h), read once here;
     * concatenation has no operator data, so any two tasks may be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type total_functions = fnsA->num_functions() + fnsB->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::ConcatBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::ConcatBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    auto concat_fn = [fnsA, fnsB, K, enable_batching, batch_pool, total_functions](const mra::Key<NDIM>& key, const node_type& na,
                                     const node_type& nb) -> TASKTYPE {
      const size_type NA = fnsA->num_functions(key), NB = fnsB->num_functions(key), N = NA + NB;

      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      if (!na.empty()) for (size_type j = 0; j < NA; ++j) if (na.is_nonzero(j)) sparsity.set_nonzero(j);
      if (!nb.empty()) for (size_type j = 0; j < NB; ++j) if (nb.is_nonzero(j)) sparsity.set_nonzero(NA + j);
      const size_type n_nonzero = sparsity.count_nonzero();

      node_type result = (n_nonzero > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate) : node_type(key, N);
      for (size_type j = 0; j < N; ++j) {
        for (auto child : children(key)) {
          const bool leaf = (j < NA) ? na.is_child_leaf(j, child) : nb.is_child_leaf(j - NA, child);
          result.set_child_leaf(j, child, leaf);
        }
      }
      result.set_ns(na.is_ns() || nb.is_ns());

      if (n_nonzero > 0) {
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(result.coeffs().buffer());
        if (!na.empty()) input.add(na.coeffs().buffer());
        if (!nb.empty()) input.add(nb.coeffs().buffer());
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto a_view = na.coeffs().current_view(), b_view = nb.coeffs().current_view();
        auto out_view = result.coeffs().current_view();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the result's sparsity is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(a_view, b_view, out_view, NA, N,
                                                                  result.coeffs(), n_nonzero);
          detail::submit_concat_batch_leader<T, NDIM>(batch, *batch_pool, K, total_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          auto sparseman = make_sparsity_manager(result);
          sparseman.populate_device_sparsity();
          submit_concat_kernel<NDIM>(key, a_view, b_view, out_view, NA, N, n_nonzero, K,
                                     ttg::device::current_stream());
        }
      }

      /* where only one set continues, the other gets empty nodes */
      std::vector<mra::Key<NDIM>> balance_a, balance_b;
      for (auto child : children(key)) {
        const bool a_leaf = na.is_all_child_leaf(child), b_leaf = nb.is_all_child_leaf(child);
        if (a_leaf && !b_leaf) balance_a.push_back(child);
        if (b_leaf && !a_leaf) balance_b.push_back(child);
      }
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(ttg::device::send<OUT>(key, std::move(result)));
      if (!balance_a.empty()) sends.push_back(ttg::device::broadcast<BALANCE_A>(std::move(balance_a), node_type(key, NA)));
      if (!balance_b.empty()) sends.push_back(ttg::device::broadcast<BALANCE_B>(std::move(balance_b), node_type(key, NB)));
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::send<OUT>(key, std::move(result));
      if (!balance_a.empty()) ttg::broadcast<BALANCE_A>(std::move(balance_a), node_type(key, NA));
      if (!balance_b.empty()) ttg::broadcast<BALANCE_B>(std::move(balance_b), node_type(key, NB));
#endif // MRA_ENABLE_HOST
    };

    auto tt = ttg::make_tt<Space>(std::move(concat_fn),
                                  ttg::edges(ttg::fuse(SA, a), ttg::fuse(SB, b)),
                                  ttg::edges(out, SA, SB), name, {"a", "b"}, {"out", "balance_a", "balance_b"});
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      tt->set_batch_matcher([](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; }, max_batch_size);
    }
#endif // MRA_ENABLE_HOST
    return tt;
  }

} // namespace mra

#endif // MRA_TASKS_CONCAT_H
