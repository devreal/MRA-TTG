#ifndef MRA_TASKS_MULTIPLY_BROADCAST_H
#define MRA_TASKS_MULTIPLY_BROADCAST_H

#include <array>
#include <vector>
#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/batch_size.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/functionnode.h"
#include "mra/kernels/merge.h"
#include "mra/kernels/multiply_broadcast.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Multiply every function of a set by one function, in reconstructed form:
   *   out_j = a * b_j     (a: one function, b: N functions)
   * e.g., a potential applied to a set of orbitals (cf. madness::mul(world, a, b)).
   *
   * The tree of out_j is the union of the trees of a and b_j: where one factor
   * has a leaf and the other has children, the leaf's coefficients are passed
   * down unchanged, and the product is formed at the leaves of the deeper tree
   * from the values of each factor at the quadrature points of the leaf, computed
   * directly from the coefficients where they originate (fcube_for_mul with an
   * ancestor), as madness::FunctionImpl::do_mul does. No screening is applied.
   *
   * Inputs and output are reconstructed trees with a node at every key of the
   * tree and leaf status per function (Leaf, Inner, or Invalid below a leaf), as
   * produced by make_reconstruct.
   *
   * Every key where either tree has a node gets a task, which receives four
   * inputs: the nodes of a and b (an empty node, sent by the parent, for a tree
   * that has ended), and the coefficients of a and of those b_j that have ended
   * above but are still needed (possibly none, sent by the parent or at the root
   * by a primer). A function is inherited where that node has a leaf or holds
   * coefficients for it; they belong to the node's origin_key. The parent passes
   * on the nodes it received wherever possible, so that nothing is refined or
   * copied on the way down.
   */
  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_multiply_broadcast(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name = "multiply_broadcast",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;

    /* batching is controlled process-wide (mra/misc/batch_size.h), read once here; the
     * operator data is the same for every node, so any two tasks may be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type total_functions = fnsB->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::MultiplyBroadcastBatchArg<T, NDIM>>> batch_pool;
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::MergeBatchArg<T, NDIM>>> merge_pool;
    if (enable_batching) {
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::MultiplyBroadcastBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
      merge_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::MergeBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr, merge_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST
    constexpr const int OUT = 0, BALANCE_A = 1, BALANCE_B = 2, INHERIT_A = 3, INHERIT_B = 4;

    ttg::Edge<mra::Key<NDIM>, node_type> SA(name + "-balance_a"), SB(name + "-balance_b");
    ttg::Edge<mra::Key<NDIM>, node_type> inh_a(name + "-inherit_a"), inh_b(name + "-inherit_b");

    /* a node that holds no coefficients for any of N functions, all Invalid */
    auto invalid_node = [](const mra::Key<NDIM>& key, size_type N) {
      node_type n(key, N);
      n.set_all_leaf(LeafStatus::Invalid);
      return n;
    };

    /* whether a node passed down by the parent provides coefficients of function j: the
     * parent passes on nodes it received, in which a leaf may hold zero coefficients */
    auto inherits = [](const node_type& n, size_type j) {
      return n.is_leaf(j) || (!n.empty() && n.is_nonzero(j));
    };

    /**
     * Nothing is inherited at the root.
     */
    auto primer_fn = [fnsB, invalid_node](const mra::Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      if (key.level() == 0) {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(ttg::device::send<0>(key, invalid_node(key, 1)),
                                      ttg::device::send<1>(key, invalid_node(key, fnsB->num_functions(key))));
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, invalid_node(key, 1));
        ttg::send<1>(key, invalid_node(key, fnsB->num_functions(key)));
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(a), ttg::edges(inh_a, inh_b),
                                         name + "-primer");

    auto multiply_fn = [&db, &functiondata, fnsA, fnsB, K, name, invalid_node, inherits,
                        enable_batching, batch_pool, merge_pool, total_functions](
                          const mra::Key<NDIM>& key,
                          const node_type& anode, const node_type& bnode,
                          const node_type& ainh, const node_type& binh) -> TASKTYPE {
      if (fnsA->num_functions(key) != 1) {
        throw std::runtime_error(name + ": the first factor must be a single function");
      }
      const size_type N = fnsB->num_functions(key);

      /* where does each factor stand at this node? */
      const bool a_inner = anode.has_children(0);
      const bool a_data = anode.is_leaf(0) || inherits(ainh, 0);
      std::vector<char> b_inner(N), b_data(N);
      bool b_tree_continues = false;
      for (size_type j = 0; j < N; ++j) {
        b_inner[j] = bnode.has_children(j);
        b_data[j] = bnode.is_leaf(j) || inherits(binh, j);
        b_tree_continues = b_tree_continues || b_inner[j];
      }
      const bool a_tree_continues = a_inner;

      /* status of the product: Leaf where both factors end here (or above),
       * Inner where either continues, Invalid where one of them is gone */
      SparsityInfo out_sparsity(N, SparsityInfo::InitType::AllZero);
      std::vector<LeafStatus> status(N, LeafStatus::Invalid);
      bool any_valid = false;
      for (size_type j = 0; j < N; ++j) {
        const bool a_here = a_inner || a_data, b_here = b_inner[j] || b_data[j];
        if (!a_here || !b_here) continue;
        any_valid = true;
        if (a_inner || b_inner[j]) {
          status[j] = LeafStatus::Inner;
        } else {
          status[j] = LeafStatus::Leaf;
          out_sparsity.set_nonzero(j);
        }
      }
      const size_type n_leaf = out_sparsity.count_nonzero();

      /* coefficients passed down: a where b continues, every b_j with coefficients where
       * a continues. a's come from either its node or what it inherited; the b_j's may
       * come from both, and only then are they merged into one node. */
      const bool pass_a = a_data && b_tree_continues;
      bool pass_b = false, b_from_node = false, b_from_inh = false;
      if (a_tree_continues) {
        for (size_type j = 0; j < N; ++j) {
          if (!b_data[j]) continue;
          pass_b = true;
          if (bnode.is_leaf(j)) b_from_node = true;
          else                  b_from_inh = true;
        }
      }
      const bool merge_b = b_from_node && b_from_inh;
      const bool any_children = a_tree_continues || b_tree_continues;

      node_type merged;
      size_type n_merged = 0;
      if (merge_b) {
        SparsityInfo merged_sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type j = 0; j < N; ++j) if (b_data[j]) merged_sparsity.set_nonzero(j);
        n_merged = merged_sparsity.count_nonzero();
        merged = node_type(key, merged_sparsity, K, ttg::scope::Allocate);
        for (size_type j = 0; j < N; ++j) {
          merged.set_leaf(j, b_data[j] ? LeafStatus::Leaf : LeafStatus::Invalid);
          if (b_data[j] && !bnode.is_leaf(j)) merged.set_origin(j, binh.origin(j) + (key.level() - binh.key().level()));
        }
      }

      node_type result = (n_leaf > 0) ? node_type(key, out_sparsity, K, ttg::scope::Allocate)
                                      : node_type(key, N);
      for (size_type j = 0; j < N; ++j) result.set_leaf(j, status[j]);

      /* scratch and source keys must outlive the kernels: destroying a buffer releases its
       * device memory right away, while the kernels launched below are still running */
      ttg::Buffer<T, DeviceAllocator<T>> mul_tmp;
      ttg::Buffer<mra::Key<NDIM>, DeviceAllocator<mra::Key<NDIM>>> inh_keys;
      if (n_leaf > 0 || merge_b) {
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        mul_tmp = ttg::Buffer<T, DeviceAllocator<T>>(multiply_broadcast_tmp_size<NDIM>(K)*std::max<size_type>(n_leaf, 1), TempScope);
        /* the nodes the inherited coefficients belong to: a, then b_0..b_N-1 */
        inh_keys = ttg::Buffer<mra::Key<NDIM>, DeviceAllocator<mra::Key<NDIM>>>(1 + N);
        mra::Key<NDIM>* ik = inh_keys.host_ptr();
        ik[0] = ainh.origin_key(0);
        for (size_type j = 0; j < N; ++j) ik[1 + j] = binh.origin_key(j);
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, phibar.buffer(), phi.buffer(), quad_x.buffer(), mul_tmp, inh_keys);
        if (!result.empty()) input.add(result.coeffs().buffer());
        if (!merged.empty()) input.add(merged.coeffs().buffer());
        for (const node_type* n : {&anode, &bnode, &ainh, &binh}) {
          if (!n->empty()) input.add(n->coeffs().buffer());
        }
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto a_leaf_view = anode.coeffs().current_view(), a_inh_view = ainh.coeffs().current_view();
        auto b_leaf_view = bnode.coeffs().current_view(), b_inh_view = binh.coeffs().current_view();
        auto out_view = result.coeffs().current_view();
        std::remove_const_t<decltype(merged.coeffs().current_view())> merged_view;
        if (merge_b) merged_view = merged.coeffs().current_view();
        const mra::Key<NDIM>* inh_keys_ptr = inh_keys.current_device_ptr();
        T* tmp_ptr = mul_tmp.current_device_ptr();
        /* merging: never both, binh holds coefficients only of functions whose tree ended above */
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* a task can join one batch only, so the leader also merges for every member that
           * needs it; the sparsity of the outputs is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, a_leaf_view, a_inh_view, b_leaf_view, b_inh_view,
                                                                  inh_keys_ptr, out_view, N, tmp_ptr,
                                                                  result.coeffs(), n_leaf,
                                                                  merged_view, merged.coeffs(), n_merged);
          if (batch.is_leader()) {
            detail::submit_merge_batch<T, NDIM>(*merge_pool, batch.size(), [&](std::size_t m) {
              auto& b = batch[m];
              return std::forward_as_tuple(b.template get<3>(), b.template get<4>(), b.template get<11>(),
                                           b.template get<12>(), b.template get<7>(), b.template get<13>());
            }, K, total_functions);
          }
          detail::submit_multiply_broadcast_batch_leader<T, NDIM>(batch, *batch_pool, D, phibar.current_view(),
                                                                  phi.current_view(), quad_x.current_view(),
                                                                  K, total_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          if (merge_b) {
            auto sparseman = make_sparsity_manager(merged);
            sparseman.populate_device_sparsity();
            submit_merge_kernel<NDIM>(key, b_leaf_view, b_inh_view, merged_view, N, n_merged, K,
                                      ttg::device::current_stream());
          }
          if (n_leaf > 0) {
            auto sparseman = make_sparsity_manager(result);
            sparseman.populate_device_sparsity();
            submit_multiply_broadcast_kernel<T, NDIM>(D, key, a_leaf_view, a_inh_view, b_leaf_view, b_inh_view,
                                                      inh_keys_ptr, out_view,
                                                      phibar.current_view(), phi.current_view(), quad_x.current_view(),
                                                      N, n_leaf, K, tmp_ptr, ttg::device::current_stream());
          }
        }
      }

      std::vector<mra::Key<NDIM>> child_keys;
      for (auto child : children(key)) child_keys.push_back(child);
      /* a node of the product exists here if any function is present (always at the root) */
      const bool send_out = any_valid || key.level() == 0;
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      if (send_out) sends.push_back(ttg::device::send<OUT>(key, std::move(result)));
      if (any_children) {
        if (!a_tree_continues) sends.push_back(ttg::device::broadcast<BALANCE_A>(child_keys, invalid_node(key, 1)));
        if (!b_tree_continues) sends.push_back(ttg::device::broadcast<BALANCE_B>(child_keys, invalid_node(key, N)));
        if (!pass_a)               sends.push_back(ttg::device::broadcast<INHERIT_A>(child_keys, invalid_node(key, 1)));
        else if (anode.is_leaf(0)) sends.push_back(ttg::device::broadcast<INHERIT_A>(child_keys, anode));
        else                       sends.push_back(ttg::device::broadcast<INHERIT_A>(child_keys, ainh));
        if (!pass_b)          sends.push_back(ttg::device::broadcast<INHERIT_B>(child_keys, invalid_node(key, N)));
        else if (merge_b)     sends.push_back(ttg::device::broadcast<INHERIT_B>(child_keys, std::move(merged)));
        else if (b_from_node) sends.push_back(ttg::device::broadcast<INHERIT_B>(child_keys, bnode));
        else                  sends.push_back(ttg::device::broadcast<INHERIT_B>(child_keys, binh));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      if (send_out) ttg::send<OUT>(key, std::move(result));
      if (any_children) {
        if (!a_tree_continues) ttg::broadcast<BALANCE_A>(child_keys, invalid_node(key, 1));
        if (!b_tree_continues) ttg::broadcast<BALANCE_B>(child_keys, invalid_node(key, N));
        if (!pass_a)               ttg::broadcast<INHERIT_A>(child_keys, invalid_node(key, 1));
        else if (anode.is_leaf(0)) ttg::broadcast<INHERIT_A>(child_keys, anode);
        else                       ttg::broadcast<INHERIT_A>(child_keys, ainh);
        if (!pass_b)          ttg::broadcast<INHERIT_B>(child_keys, invalid_node(key, N));
        else if (merge_b)     ttg::broadcast<INHERIT_B>(child_keys, std::move(merged));
        else if (b_from_node) ttg::broadcast<INHERIT_B>(child_keys, bnode);
        else                  ttg::broadcast<INHERIT_B>(child_keys, binh);
      }
#endif // MRA_ENABLE_HOST
    };

    auto multiply_tt = ttg::make_tt<Space>(std::move(multiply_fn),
                                           ttg::edges(ttg::fuse(SA, a), ttg::fuse(SB, b), inh_a, inh_b),
                                           ttg::edges(out, SA, SB, inh_a, inh_b),
                                           name, {"a", "b", "inherit_a", "inherit_b"},
                                           {"out", "balance_a", "balance_b", "inherit_a", "inherit_b"});

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      primer_tt->set_keymap(procmap);
      multiply_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      primer_tt->set_devicemap(devicemap);
      multiply_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      multiply_tt->set_batch_matcher([](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
                                     max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(multiply_tt->template in<0>(), multiply_tt->template in<1>());
    auto outs = std::make_tuple(multiply_tt->template out<OUT>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(primer_tt);
    ops[1] = std::move(multiply_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_MULTIPLY_BROADCAST_H
