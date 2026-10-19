#ifndef MRA_TASKS_DERIVATIVE_H
#define MRA_TASKS_DERIVATIVE_H

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
#include "mra/kernels/derivative.h"
#include "mra/kernels/merge.h"
#include "mra/kernels/multiply_broadcast.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Derivative along one axis of every function of a set, in reconstructed form,
   * as madness::Derivative computes it: at each leaf of f_j, from the coefficients
   * of f_j at that node and at its two neighbors along the axis. A neighbor where
   * the tree of f_j is coarser contributes its leaf coefficients refined to the
   * level of the node (exact). Where the tree of f_j is finer at a neighbor, the
   * leaf is refined (exact) and the derivative is formed at its children, and
   * further down as needed. The tree of each derivative is thus the tree of f_j
   * refined where needed; the trees of the functions of the set may differ.
   *
   * At the boundaries of the domain the neighbor is absent; the boundary condition is
   * that of the derivative operators in functiondata (see FunctionData).
   *
   * Input and output are reconstructed trees with a node at every key of the tree
   * and leaf status per function, as produced by make_reconstruct; inner nodes of
   * the output carry no coefficients.
   *
   * Every node of the (refined) tree goes through three tasks:
   *  - assemble: combines the input node, if the key is in the input tree, with the
   *    coefficients refined from above (for functions whose tree ended above), and
   *    sends the combined node to its two neighbors along the axis;
   *  - compute: decides per function whether to compute here or refine, forms the
   *    derivatives, and tells its neighbors whether it has children;
   *  - spawn: starts the children, if any. A child whose neighbor across the
   *    parent's boundary does not exist (the neighbor of the parent has no children)
   *    receives that neighbor's coefficients from the parent instead.
   * Each node knows which of its neighbors exist from its parent, so that every
   * message has a receiver. Nodes are sent directly to every task that reads them
   * rather than forwarded from one reader to the next.
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_derivative(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetT>& fns,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    const Dimension axis,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name = "derivative",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;
    using key_type = mra::Key<NDIM>;
    using node_edge = ttg::Edge<key_type, node_type>;
    using flag_edge = ttg::Edge<key_type, int>;
    static constexpr const int num_children = key_type::num_children();
    /* which neighbors along the axis exist */
    constexpr const int LEFT_EXISTS = 1, RIGHT_EXISTS = 2;

    if (axis >= NDIM) throw std::runtime_error(name + ": invalid axis");

    node_edge in_sub(name + "-in_sub"), inh(name + "-inherit");
    node_edge b_center(name + "-center"), b_left(name + "-left"), b_right(name + "-right");
    node_edge s_center(name + "-s_center"), s_left(name + "-s_left"), s_right(name + "-s_right");
    flag_edge a_flags(name + "-a_flags"), b_flags(name + "-b_flags"), s_flags(name + "-s_flags");
    flag_edge s_ldec(name + "-left_has_children"), s_rdec(name + "-right_has_children");

    /* a node that holds no coefficients for any of N functions, all Invalid */
    auto invalid_node = [](const key_type& key, size_type N) {
      node_type n(key, N);
      n.set_all_leaf(LeafStatus::Invalid);
      return n;
    };

    /* status of each function of the derivative at a node, given the combined nodes
     * of the node and of its neighbors: Inner where the tree continues or the
     * function is refined here, Leaf where the derivative is formed here */
    auto plan = [axis](const key_type& key, const node_type& center, const node_type& left,
                       const node_type& right, size_type N) {
      const bool lb = key.is_left_boundary(axis), rb = key.is_right_boundary(axis);
      std::vector<LeafStatus> status(N, LeafStatus::Invalid);
      for (size_type j = 0; j < N; ++j) {
        const LeafStatus s = center.leaf_status(j);
        if (s == LeafStatus::Leaf) {
          const bool finer = (!lb && left.leaf_status(j) == LeafStatus::Inner)
                          || (!rb && right.leaf_status(j) == LeafStatus::Inner);
          status[j] = finer ? LeafStatus::Inner : LeafStatus::Leaf;
        } else {
          status[j] = s;
        }
      }
      return status;
    };
    auto has_children = [](const std::vector<LeafStatus>& status) {
      for (auto s : status) if (s == LeafStatus::Inner) return true;
      return false;
    };

#ifndef MRA_ENABLE_HOST
#define MRA_DERIV_SENDS                                                              \
      auto sends = ttg::device::forward();                                           \
      auto do_send = [&]<std::size_t I, typename V>(const key_type& k, V&& v) {      \
        sends.push_back(ttg::device::send<I>(k, std::forward<V>(v)));                \
      };
#define MRA_DERIV_FLUSH co_await std::move(sends);
#else  // MRA_ENABLE_HOST
#define MRA_DERIV_SENDS                                                              \
      auto do_send = []<std::size_t I, typename V>(const key_type& k, V&& v) {       \
        ttg::send<I>(k, std::forward<V>(v));                                          \
      };
#define MRA_DERIV_FLUSH
#endif // MRA_ENABLE_HOST

    /**
     * Root: nothing is inherited, no neighbors.
     */
    auto primer_fn = [fns, invalid_node](const key_type& key, const node_type& node) -> TASKTYPE {
      if (key.level() == 0) {
        const size_type N = fns->num_functions(key);
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(ttg::device::send<0>(key, invalid_node(key, N)),
                                      ttg::device::send<1>(key, 0),
                                      ttg::device::send<2>(key, invalid_node(key, N)),
                                      ttg::device::send<3>(key, invalid_node(key, N)),
                                      ttg::device::send<4>(key, invalid_node(key, N)),
                                      ttg::device::send<5>(key, invalid_node(key, N)));
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, invalid_node(key, N));
        ttg::send<1>(key, 0);
        ttg::send<2>(key, invalid_node(key, N));
        ttg::send<3>(key, invalid_node(key, N));
        ttg::send<4>(key, invalid_node(key, N));
        ttg::send<5>(key, invalid_node(key, N));
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(in),
                                         ttg::edges(inh, a_flags, b_left, b_right, s_left, s_right),
                                         name + "-primer");

    /**
     * Assemble: per function, the input node where the function has a leaf or children
     * here, else the coefficients refined from above (Leaf: to be differentiated here
     * or below, Invalid: only needed by the neighbors).
     */
    auto assemble_fn = [fns, K, axis, invalid_node](const key_type& key, const node_type& input,
                                                    const node_type& inherited, const int& flags) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      std::vector<LeafStatus> status(N);
      for (size_type j = 0; j < N; ++j) {
        const LeafStatus s = input.leaf_status(j);
        if (s == LeafStatus::Inner) {
          status[j] = LeafStatus::Inner;
        } else if (s == LeafStatus::Leaf) {
          status[j] = LeafStatus::Leaf;
          if (!input.empty() && input.is_nonzero(j)) sparsity.set_nonzero(j);
        } else {
          status[j] = inherited.leaf_status(j);
          if (!inherited.empty() && inherited.is_nonzero(j)) sparsity.set_nonzero(j);
        }
      }
      const size_type n_nonzero = sparsity.count_nonzero();
      node_type combined = (n_nonzero > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate) : node_type(key, N);
      for (size_type j = 0; j < N; ++j) combined.set_leaf(j, status[j]);
      if (n_nonzero > 0) {
#ifndef MRA_ENABLE_HOST
        auto inputs = ttg::device::Input(combined.coeffs().buffer());
        if (!input.empty()) inputs.add(input.coeffs().buffer());
        if (!inherited.empty()) inputs.add(inherited.coeffs().buffer());
        co_await ttg::device::select(inputs);
#endif // MRA_ENABLE_HOST
        auto out_view = combined.coeffs().current_view();
        auto sparseman = make_sparsity_manager(combined);
        sparseman.populate_device_sparsity();
        submit_merge_kernel<NDIM>(key, input.coeffs().current_view(), inherited.coeffs().current_view(), out_view,
                                  N, n_nonzero, K, ttg::device::current_stream());
      }
      /* to the compute (0-3) and spawn (4-7) tasks: 0/4 center, 1/5 left input of the right
       * neighbor, 2/6 right input of the left neighbor, 3/7 flags. The node is sent directly
       * to every task that reads it rather than forwarded by a reader. */
      std::vector<key_type> self = {key}, right_keys, left_keys;
      if (flags & RIGHT_EXISTS) right_keys.push_back(key.neighbor(axis, 1));
      if (flags & LEFT_EXISTS) left_keys.push_back(key.neighbor(axis, -1));
#ifndef MRA_ENABLE_HOST
      /* the device broadcast does not skip empty key lists */
      auto sends = ttg::device::forward(ttg::device::send<3>(key, flags), ttg::device::send<7>(key, flags));
      if (!right_keys.empty() && !left_keys.empty()) {
        sends.push_back(ttg::device::broadcast<0, 1, 2, 4, 5, 6>(
          std::make_tuple(self, right_keys, left_keys, self, right_keys, left_keys), std::move(combined)));
      } else if (!right_keys.empty()) {
        sends.push_back(ttg::device::broadcast<0, 1, 4, 5>(
          std::make_tuple(self, right_keys, self, right_keys), std::move(combined)));
      } else if (!left_keys.empty()) {
        sends.push_back(ttg::device::broadcast<0, 2, 4, 6>(
          std::make_tuple(self, left_keys, self, left_keys), std::move(combined)));
      } else {
        sends.push_back(ttg::device::broadcast<0, 4>(std::make_tuple(self, self), std::move(combined)));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::broadcast<0, 1, 2, 4, 5, 6>(std::make_tuple(self, right_keys, left_keys, self, right_keys, left_keys),
                                       std::move(combined));
      ttg::send<3>(key, flags);
      ttg::send<7>(key, flags);
#endif // MRA_ENABLE_HOST
    };
    auto assemble_tt = ttg::make_tt<Space>(std::move(assemble_fn),
                                           ttg::edges(ttg::fuse(in, in_sub), inh, a_flags),
                                           ttg::edges(b_center, b_left, b_right, b_flags,
                                                      s_center, s_left, s_right, s_flags),
                                           name + "-assemble", {"input", "inherited", "flags"},
                                           {"center", "left", "right", "flags",
                                            "spawn_center", "spawn_left", "spawn_right", "spawn_flags"});

    /**
     * Compute the derivative of the functions with a leaf here, and tell the neighbors
     * (or, where there is none, the own spawn task) whether this node has children.
     */
    auto compute_fn = [&db, &functiondata, fns, K, axis, plan, has_children](
                          const key_type& key, const node_type& center, const node_type& left,
                          const node_type& right, const int& flags) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      const bool lb = key.is_left_boundary(axis), rb = key.is_right_boundary(axis);
      std::vector<LeafStatus> status = plan(key, center, left, right, N);
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      for (size_type j = 0; j < N; ++j) {
        if (status[j] != LeafStatus::Leaf) continue;
        const bool nonzero = (!center.empty() && center.is_nonzero(j))
                          || (!lb && !left.empty() && left.is_nonzero(j))
                          || (!rb && !right.empty() && right.is_nonzero(j));
        if (nonzero) sparsity.set_nonzero(j);
      }
      const size_type n_nonzero = sparsity.count_nonzero();
      node_type result = (n_nonzero > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate) : node_type(key, N);
      for (size_type j = 0; j < N; ++j) result.set_leaf(j, status[j]);

      if (n_nonzero > 0) {
        const auto& operators = functiondata.get_operators();
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        ttg::Buffer<T, DeviceAllocator<T>> tmp(derivative_tmp_size<NDIM>(K)*n_nonzero, TempScope);
#ifndef MRA_ENABLE_HOST
        auto inputs = ttg::device::Input(db, result.coeffs().buffer(), operators.buffer(), phibar.buffer(),
                                         phi.buffer(), quad_x.buffer(), tmp);
        for (const node_type* n : {&center, &left, &right}) {
          if (!n->empty()) inputs.add(n->coeffs().buffer());
        }
        co_await ttg::device::select(inputs);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto result_view = result.coeffs().current_view();
        auto sparseman = make_sparsity_manager(result);
        sparseman.populate_device_sparsity();
        /* the neighbors' coefficients are given at the neighbors' keys */
        submit_derivative_kernel(D, key, key.neighbor(axis, -1), key, key.neighbor(axis, 1),
                                 left.coeffs().current_view(), center.coeffs().current_view(),
                                 right.coeffs().current_view(), operators.current_view(), result_view,
                                 phi.current_view(), phibar.current_view(), quad_x.current_view(),
                                 tmp.current_device_ptr(), N, n_nonzero, K, T(0), T(0), axis,
                                 FunctionData<T, NDIM>::BC_FREE, FunctionData<T, NDIM>::BC_FREE,
                                 ttg::device::current_stream());
      }

      const int dec = has_children(status) ? 1 : 0;
      MRA_DERIV_SENDS
      /* 0: output, 1: left decision of the right neighbor's spawn (or of the own spawn),
       * 2: right decision of the left neighbor's spawn (or own) */
      do_send.template operator()<0>(key, std::move(result));
      if (flags & RIGHT_EXISTS) do_send.template operator()<1>(key.neighbor(axis, 1), dec);
      else                      do_send.template operator()<2>(key, 0);
      if (flags & LEFT_EXISTS)  do_send.template operator()<2>(key.neighbor(axis, -1), dec);
      else                      do_send.template operator()<1>(key, 0);
      MRA_DERIV_FLUSH
    };
    auto compute_tt = ttg::make_tt<Space>(std::move(compute_fn),
                                          ttg::edges(b_center, b_left, b_right, b_flags),
                                          ttg::edges(out, s_ldec, s_rdec),
                                          name, {"center", "left", "right", "flags"},
                                          {"out", "left_has_children", "right_has_children"});

    /**
     * Start the children, if this node has any: pass down the refined coefficients of
     * every function whose tree ends here or above, the neighbors' refined
     * coefficients where a neighbor has no children, and which neighbors exist.
     */
    auto spawn_fn = [&functiondata, fns, K, axis, plan, has_children, invalid_node](
                        const key_type& key, const node_type& center, const node_type& left,
                        const node_type& right, const int& flags, const int& left_dec,
                        const int& right_dec) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      std::vector<LeafStatus> status = plan(key, center, left, right, N);
      if (!has_children(status)) {
#ifndef MRA_ENABLE_HOST
        co_return;
#else  // MRA_ENABLE_HOST
        return;
#endif // MRA_ENABLE_HOST
      }
      const bool lb = key.is_left_boundary(axis), rb = key.is_right_boundary(axis);
      /* children are in the input tree iff a function continues here in the input */
      bool children_in_input = false;
      for (size_type j = 0; j < N; ++j) children_in_input |= (center.leaf_status(j) == LeafStatus::Inner);
      /* substitutes for a neighbor without children */
      const bool sub_left = !lb && (left_dec == 0), sub_right = !rb && (right_dec == 0);

      auto data_sparsity = [N](const node_type& n) {
        SparsityInfo s(N, SparsityInfo::InitType::AllZero);
        if (!n.empty()) for (size_type j = 0; j < N; ++j) if (n.is_nonzero(j)) s.set_nonzero(j);
        return s;
      };
      auto make_children = [&](const key_type& parent, const node_type& node, bool own) {
        SparsityInfo s = data_sparsity(node);
        std::array<node_type, num_children> kids;
        for (auto child : children(parent)) {
          node_type kid(child, N);
          if (s.count_nonzero() > 0) kid.allocate(s, K, ttg::scope::Allocate);
          for (size_type j = 0; j < N; ++j) {
            LeafStatus ks = LeafStatus::Invalid;
            if (own) {
              if (center.leaf_status(j) == LeafStatus::Inner) ks = LeafStatus::Inner;
              else if (center.leaf_status(j) == LeafStatus::Leaf && status[j] == LeafStatus::Inner) ks = LeafStatus::Leaf;
            }
            kid.set_leaf(j, ks);
          }
          kids[child.childindex()] = std::move(kid);
        }
        return std::make_pair(std::move(kids), s.count_nonzero());
      };

      auto [kids, n_own] = make_children(key, center, true);
      std::array<node_type, num_children> lkids, rkids;
      size_type n_left = 0, n_right = 0;
      if (sub_left) std::tie(lkids, n_left) = make_children(key.neighbor(axis, -1), left, false);
      if (sub_right) std::tie(rkids, n_right) = make_children(key.neighbor(axis, 1), right, false);

      if (n_own > 0 || n_left > 0 || n_right > 0) {
        const auto& hg = functiondata.get_hg();
        ttg::Buffer<T, DeviceAllocator<T>> tmp_own(refine_tmp_size<NDIM>(K)*std::max<size_type>(n_own, 1), TempScope);
        ttg::Buffer<T, DeviceAllocator<T>> tmp_left(refine_tmp_size<NDIM>(K)*std::max<size_type>(n_left, 1), TempScope);
        ttg::Buffer<T, DeviceAllocator<T>> tmp_right(refine_tmp_size<NDIM>(K)*std::max<size_type>(n_right, 1), TempScope);
#ifndef MRA_ENABLE_HOST
        auto inputs = ttg::device::Input(hg.buffer(), tmp_own, tmp_left, tmp_right);
        for (const node_type* n : {&center, &left, &right}) {
          if (!n->empty()) inputs.add(n->coeffs().buffer());
        }
        for (auto* ks : {&kids, &lkids, &rkids}) {
          for (auto& k : *ks) if (!k.empty()) inputs.add(k.coeffs().buffer());
        }
        co_await ttg::device::select(inputs);
#endif // MRA_ENABLE_HOST
        auto refine = [&](const key_type& parent, const node_type& node, std::array<node_type, num_children>& ks,
                          size_type n, ttg::Buffer<T, DeviceAllocator<T>>& tmp) {
          if (n == 0) return;
          auto views = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::array{(ks[Is].coeffs().current_view())...};
          }(std::make_index_sequence<num_children>{});
          for (auto& k : ks) {
            auto sparseman = make_sparsity_manager(k);
            sparseman.populate_device_sparsity();
          }
          submit_refine_kernel<T, NDIM>(parent, node.coeffs().current_view(), node.coeffs().current_view(), views,
                                        hg.current_view(), N, n, K, tmp.current_device_ptr(),
                                        ttg::device::current_stream());
        };
        refine(key, center, kids, n_own, tmp_own);
        if (sub_left) refine(key.neighbor(axis, -1), left, lkids, n_left, tmp_left);
        if (sub_right) refine(key.neighbor(axis, 1), right, rkids, n_right, tmp_right);
      }

      MRA_DERIV_SENDS
#ifndef MRA_ENABLE_HOST
      auto do_send2 = [&]<std::size_t I, std::size_t J>(const key_type& k, node_type&& v) {
        sends.push_back(ttg::device::broadcast<I, J>(std::make_tuple(k, k), std::move(v)));
      };
#else  // MRA_ENABLE_HOST
      auto do_send2 = []<std::size_t I, std::size_t J>(const key_type& k, node_type&& v) {
        ttg::broadcast<I, J>(std::make_tuple(k, k), std::move(v));
      };
#endif // MRA_ENABLE_HOST
      /* 0: inherited, 1: flags, 2: input substitute, 3/5: left input of compute/spawn,
       * 4/6: right input of compute/spawn */
      for (auto child : children(key)) {
        const int c = child.childindex();
        const bool is_left = child.is_left_child(axis);
        int child_flags = 0;
        /* the sibling exists; across the boundary the neighbor's child exists iff it has children */
        if (is_left) {
          child_flags |= RIGHT_EXISTS;
          if (!lb && left_dec) child_flags |= LEFT_EXISTS;
          if (lb) do_send2.template operator()<3, 5>(child, invalid_node(child, N));
          else if (sub_left) do_send2.template operator()<3, 5>(child, std::move(lkids[child.neighbor(axis, -1).childindex()]));
        } else {
          child_flags |= LEFT_EXISTS;
          if (!rb && right_dec) child_flags |= RIGHT_EXISTS;
          if (rb) do_send2.template operator()<4, 6>(child, invalid_node(child, N));
          else if (sub_right) do_send2.template operator()<4, 6>(child, std::move(rkids[child.neighbor(axis, 1).childindex()]));
        }
        if (!children_in_input) do_send.template operator()<2>(child, invalid_node(child, N));
        do_send.template operator()<1>(child, child_flags);
        do_send.template operator()<0>(child, std::move(kids[c]));
      }
      MRA_DERIV_FLUSH
    };
    auto spawn_tt = ttg::make_tt<Space>(std::move(spawn_fn),
                                        ttg::edges(s_center, s_left, s_right, s_flags, s_ldec, s_rdec),
                                        ttg::edges(inh, a_flags, in_sub, b_left, b_right, s_left, s_right),
                                        name + "-spawn",
                                        {"center", "left", "right", "flags", "left_has_children", "right_has_children"},
                                        {"inherited", "flags", "input", "left", "right", "spawn_left", "spawn_right"});
#undef MRA_DERIV_SENDS
#undef MRA_DERIV_FLUSH

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      primer_tt->set_keymap(procmap);
      assemble_tt->set_keymap(procmap);
      compute_tt->set_keymap(procmap);
      spawn_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      primer_tt->set_devicemap(devicemap);
      assemble_tt->set_devicemap(devicemap);
      compute_tt->set_devicemap(devicemap);
      spawn_tt->set_devicemap(devicemap);
    }

    auto ins = std::make_tuple(assemble_tt->template in<0>());
    auto outs = std::make_tuple(compute_tt->template out<0>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(4);
    ops[0] = std::move(primer_tt);
    ops[1] = std::move(assemble_tt);
    ops[2] = std::move(compute_tt);
    ops[3] = std::move(spawn_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_DERIVATIVE_H
