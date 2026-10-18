#ifndef MRA_TASKS_MULTIPLY_BROADCAST_H
#define MRA_TASKS_MULTIPLY_BROADCAST_H

#include <array>
#include <vector>
#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/functiondata.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/functionnode.h"
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
   * has a leaf and the other has children, the leaf's coefficients are refined
   * to the children (exact) and passed down, and the product is formed at the
   * leaves of the deeper tree, as madness::FunctionImpl::do_mul does. No
   * screening is applied.
   *
   * Inputs and output are reconstructed trees with a node at every key of the
   * tree and leaf status per function (Leaf, Inner, or Invalid below a leaf), as
   * produced by make_reconstruct.
   *
   * Every key where either tree has a node gets a task, which receives four
   * inputs: the nodes of a and b (an empty node, sent by the parent, for a tree
   * that has ended), and the refined coefficients of a and of those b_j that
   * have ended above but are still needed (possibly empty, sent by the parent or
   * at the root by a primer).
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
    static constexpr const int num_children = mra::Key<NDIM>::num_children();
    constexpr const int OUT = 0, BALANCE_A = 1, BALANCE_B = 2, INHERIT_A = 3, INHERIT_B = 4;

    ttg::Edge<mra::Key<NDIM>, node_type> SA(name + "-balance_a"), SB(name + "-balance_b");
    ttg::Edge<mra::Key<NDIM>, node_type> inh_a(name + "-inherit_a"), inh_b(name + "-inherit_b");

    /* a node that holds no coefficients for any of N functions, all Invalid */
    auto invalid_node = [](const mra::Key<NDIM>& key, size_type N) {
      node_type n(key, N);
      n.set_all_leaf(LeafStatus::Invalid);
      return n;
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

    auto multiply_fn = [&db, &functiondata, fnsA, fnsB, K, name, invalid_node](
                          const mra::Key<NDIM>& key,
                          const node_type& anode, const node_type& bnode,
                          const node_type& ainh, const node_type& binh) -> TASKTYPE {
      if (fnsA->num_functions(key) != 1) {
        throw std::runtime_error(name + ": the first factor must be a single function");
      }
      const size_type N = fnsB->num_functions(key);

      /* where does each factor stand at this node? */
      const bool a_inner = anode.has_children(0);
      const bool a_data = anode.is_leaf(0) || (!ainh.empty() && ainh.is_nonzero(0));
      std::vector<char> b_inner(N), b_data(N);
      bool b_tree_continues = false;
      for (size_type j = 0; j < N; ++j) {
        b_inner[j] = bnode.has_children(j);
        b_data[j] = bnode.is_leaf(j) || (!binh.empty() && binh.is_nonzero(j));
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

      /* coefficients passed down: a where b continues, b_j where a continues */
      const bool refine_a = a_data && b_tree_continues;
      SparsityInfo refine_b_sparsity(N, SparsityInfo::InitType::AllZero);
      if (a_tree_continues) {
        for (size_type j = 0; j < N; ++j) if (b_data[j]) refine_b_sparsity.set_nonzero(j);
      }
      const size_type n_refine_b = refine_b_sparsity.count_nonzero();
      const bool any_children = a_tree_continues || b_tree_continues;

      std::array<node_type, num_children> a_children, b_children;
      for (auto child : children(key)) {
        const int c = child.childindex();
        a_children[c] = invalid_node(child, 1);
        b_children[c] = invalid_node(child, N);
        if (refine_a) {
          SparsityInfo s(1, SparsityInfo::InitType::AllZero);
          s.set_nonzero(0);
          a_children[c].allocate(s, K, ttg::scope::Allocate);
        }
        if (n_refine_b > 0) b_children[c].allocate(refine_b_sparsity, K, ttg::scope::Allocate);
      }

      node_type result = (n_leaf > 0) ? node_type(key, out_sparsity, K, ttg::scope::Allocate)
                                      : node_type(key, N);
      for (size_type j = 0; j < N; ++j) result.set_leaf(j, status[j]);

      if (n_leaf > 0 || refine_a || n_refine_b > 0) {
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        const auto& hg = functiondata.get_hg();
        ttg::Buffer<T, DeviceAllocator<T>> mul_tmp(multiply_broadcast_tmp_size<NDIM>(K)*std::max<size_type>(n_leaf, 1), TempScope);
        ttg::Buffer<T, DeviceAllocator<T>> refine_tmp(refine_tmp_size<NDIM>(K)*std::max<size_type>(std::max<size_type>(n_refine_b, refine_a), 1), TempScope);
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, phibar.buffer(), phi.buffer(), quad_x.buffer(), hg.buffer(),
                                        mul_tmp, refine_tmp);
        if (!result.empty()) input.add(result.coeffs().buffer());
        for (const node_type* n : {&anode, &bnode, &ainh, &binh}) {
          if (!n->empty()) input.add(n->coeffs().buffer());
        }
        for (int c = 0; c < num_children; ++c) {
          if (!a_children[c].empty()) input.add(a_children[c].coeffs().buffer());
          if (!b_children[c].empty()) input.add(b_children[c].coeffs().buffer());
        }
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        if (n_leaf > 0) {
          auto out_view = result.coeffs().current_view();
          auto sparseman = make_sparsity_manager(result);
          sparseman.populate_device_sparsity();
          submit_multiply_broadcast_kernel<T, NDIM>(D, key, anode.coeffs().current_view(), ainh.coeffs().current_view(),
                                                    bnode.coeffs().current_view(), binh.coeffs().current_view(), out_view,
                                                    phibar.current_view(), phi.current_view(), quad_x.current_view(),
                                                    N, n_leaf, K, mul_tmp.current_device_ptr(),
                                                    ttg::device::current_stream());
        }
        auto refine = [&](const node_type& leaf, const node_type& inh, std::array<node_type, num_children>& kids,
                          size_type Nf, size_type n) {
          auto views = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::array{(kids[Is].coeffs().current_view())...};
          }(std::make_index_sequence<num_children>{});
          for (auto& kid : kids) {
            auto sparseman = make_sparsity_manager(kid);
            sparseman.populate_device_sparsity();
          }
          submit_refine_kernel<T, NDIM>(key, leaf.coeffs().current_view(), inh.coeffs().current_view(), views,
                                        hg.current_view(), Nf, n, K, refine_tmp.current_device_ptr(),
                                        ttg::device::current_stream());
        };
        if (refine_a) refine(anode, ainh, a_children, 1, 1);
        if (n_refine_b > 0) refine(bnode, binh, b_children, N, n_refine_b);
      }

#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      auto do_send = [&]<std::size_t I, typename V>(const mra::Key<NDIM>& k, V&& v) {
        sends.push_back(ttg::device::send<I>(k, std::forward<V>(v)));
      };
#else  // MRA_ENABLE_HOST
      auto do_send = []<std::size_t I, typename V>(const mra::Key<NDIM>& k, V&& v) {
        ttg::send<I>(k, std::forward<V>(v));
      };
#endif // MRA_ENABLE_HOST
      /* a node of the product exists here if any function is present (always at the root) */
      if (any_valid || key.level() == 0) do_send.template operator()<OUT>(key, std::move(result));
      if (any_children) {
        for (auto child : children(key)) {
          const int c = child.childindex();
          if (!a_tree_continues) do_send.template operator()<BALANCE_A>(child, invalid_node(child, 1));
          if (!b_tree_continues) do_send.template operator()<BALANCE_B>(child, invalid_node(child, N));
          do_send.template operator()<INHERIT_A>(child, std::move(a_children[c]));
          do_send.template operator()<INHERIT_B>(child, std::move(b_children[c]));
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
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

    auto ins = std::make_tuple(multiply_tt->template in<0>(), multiply_tt->template in<1>());
    auto outs = std::make_tuple(multiply_tt->template out<OUT>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(primer_tt);
    ops[1] = std::move(multiply_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_MULTIPLY_BROADCAST_H
