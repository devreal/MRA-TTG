#ifndef MRA_TASKS_DERIVATIVE_H
#define MRA_TASKS_DERIVATIVE_H

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
#include "mra/kernels/derivative.h"
#include "mra/kernels/merge.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Derivative along one axis of every function of a set, in reconstructed form,
   * as madness::Derivative computes it: at each leaf of f_j, from the coefficients
   * of f_j at that node and at its two neighbors along the axis. A neighbor where
   * the tree of f_j is coarser contributes its leaf coefficients projected to the
   * level of the node (exact). Where the tree of f_j is finer at a neighbor, the
   * leaf is refined and the derivative is formed at its children, and further down
   * as needed. The tree of each derivative is thus the tree of f_j refined where
   * needed; the trees of the functions of the set may differ.
   *
   * Coefficients are never refined level by level: a node passes its own
   * coefficients down unchanged and records per function at which ancestor they
   * originate (FunctionsReconstructedNode::origin). They are projected once, from
   * their origin to the node where a derivative is formed (cf. madness's
   * parent_to_child).
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
   *    coefficients of the parent (for functions whose tree ended above), and sends
   *    the combined node to its two neighbors along the axis;
   *  - compute: decides per function whether to compute here or refine, forms the
   *    derivatives, and tells its neighbors whether it has children;
   *  - spawn: starts the children, if any. A child whose neighbor across the
   *    parent's boundary does not exist (the neighbor of the parent has no children)
   *    receives the parent's neighbor instead. Spawn does no device work: it only
   *    passes on nodes it received and nodes without coefficients.
   * Each node knows which of its neighbors exist from its parent, so that every
   * message has a receiver: the parent records whether its neighbors have children
   * in the flags of the node it sends to all children, and every child derives from
   * them which of its own neighbors exist; the combined node carries the result to
   * compute and spawn. Nodes are sent directly to every task that reads them rather
   * than forwarded from one reader to the next, except where a task passes on a node
   * it received unchanged.
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
    /* node flags: in the combined node, which neighbors along the axis exist; in the
     * node a parent sends to its children, whether the parent's neighbors have children */
    constexpr const std::uint8_t LEFT_EXISTS = 1, RIGHT_EXISTS = 2;
    constexpr const std::uint8_t LEFT_HAS_CHILDREN = 1, RIGHT_HAS_CHILDREN = 2;

    if (axis >= NDIM) throw std::runtime_error(name + ": invalid axis");

    /* batching of the compute kernel is controlled process-wide (mra/misc/batch_size.h),
     * read once here; the operator data is the same for every node, so any two tasks may
     * be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type total_functions = fns->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::DerivativeBatchArg<T, NDIM>>> batch_pool;
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::MergeBatchArg<T, NDIM>>> merge_pool;
    if (enable_batching) {
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::DerivativeBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
      merge_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::MergeBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr, merge_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    /* inh: the status of each function at the children, without coefficients;
     * parent: the parent's combined node, holding the coefficients they inherit */
    node_edge in_sub(name + "-in_sub"), inh(name + "-inherit"), parent_e(name + "-parent");
    node_edge b_center(name + "-center"), b_left(name + "-left"), b_right(name + "-right");
    node_edge s_center(name + "-s_center"), s_left(name + "-s_left"), s_right(name + "-s_right");
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
    /* which neighbors of a node exist, given whether the neighbors of its parent have
     * children: the sibling always exists, the neighbor across the parent's boundary
     * iff the parent's neighbor on that side has children (the root has none) */
    auto neighbor_flags = [axis](const key_type& key, std::uint8_t parent_flags) -> std::uint8_t {
      if (key.level() == 0) return 0;
      if (key.is_left_child(axis)) {
        const bool left = !key.is_left_boundary(axis) && (parent_flags & LEFT_HAS_CHILDREN);
        return RIGHT_EXISTS | (left ? LEFT_EXISTS : 0);
      } else {
        const bool right = !key.is_right_boundary(axis) && (parent_flags & RIGHT_HAS_CHILDREN);
        return LEFT_EXISTS | (right ? RIGHT_EXISTS : 0);
      }
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
     * Root: nothing is inherited, no neighbors. One node without coefficients, all
     * Invalid, stands in for all of them.
     */
    auto primer_fn = [fns, invalid_node](const key_type& key, const node_type& node) -> TASKTYPE {
      if (key.level() == 0) {
        const size_type N = fns->num_functions(key);
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::broadcast<0, 1, 2, 3, 4, 5>(std::make_tuple(key, key, key, key, key, key),
                                                          invalid_node(key, N));
#else  // MRA_ENABLE_HOST
        ttg::broadcast<0, 1, 2, 3, 4, 5>(std::make_tuple(key, key, key, key, key, key), invalid_node(key, N));
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(in),
                                         ttg::edges(inh, parent_e, b_left, b_right, s_left, s_right),
                                         name + "-primer");

    /**
     * Assemble: per function, the input node where the function has a leaf or children
     * here, else the parent's coefficients, with the status the parent decided (Leaf:
     * to be differentiated here or below, Invalid: only needed by the neighbors).
     * Coefficients taken from the parent keep their origin, one level further up.
     */
    auto assemble_fn = [fns, K, axis, neighbor_flags, enable_batching, merge_pool, total_functions](
                                                const key_type& key, const node_type& input,
                                                const node_type& inherited, const node_type& parent) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      const std::uint8_t flags = neighbor_flags(key, inherited.flags());
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      std::vector<LeafStatus> status(N);
      std::vector<Level> origin(N, 0);
      for (size_type j = 0; j < N; ++j) {
        const LeafStatus s = input.leaf_status(j);
        if (s == LeafStatus::Inner) {
          status[j] = LeafStatus::Inner;
        } else if (s == LeafStatus::Leaf) {
          status[j] = LeafStatus::Leaf;
          if (!input.empty() && input.is_nonzero(j)) sparsity.set_nonzero(j);
        } else {
          status[j] = inherited.leaf_status(j);
          if (!parent.empty() && parent.is_nonzero(j)) {
            sparsity.set_nonzero(j);
            origin[j] = parent.origin(j) + (key.level() - parent.key().level());
          }
        }
      }
      const size_type n_nonzero = sparsity.count_nonzero();
      node_type combined = (n_nonzero > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate) : node_type(key, N);
      for (size_type j = 0; j < N; ++j) {
        combined.set_leaf(j, status[j]);
        combined.set_origin(j, origin[j]);
      }
      combined.set_flags(flags);
      if (n_nonzero > 0) {
#ifndef MRA_ENABLE_HOST
        auto inputs = ttg::device::Input(combined.coeffs().buffer());
        if (!input.empty()) inputs.add(input.coeffs().buffer());
        if (!parent.empty()) inputs.add(parent.coeffs().buffer());
        co_await ttg::device::select(inputs);
#endif // MRA_ENABLE_HOST
        auto out_view = combined.coeffs().current_view();
        auto input_view = input.coeffs().current_view(), parent_view = parent.coeffs().current_view();
        /* never both: the parent holds coefficients only of functions whose input tree ended above */
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the sparsity of the combined node is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<key_type>(input_view, parent_view, out_view, combined.coeffs(),
                                                            N, n_nonzero);
          if (batch.is_leader()) {
            detail::submit_merge_batch<T, NDIM>(*merge_pool, batch.size(), [&](std::size_t m) {
              auto& b = batch[m];
              return std::forward_as_tuple(b.template get<0>(), b.template get<1>(), b.template get<2>(),
                                           b.template get<3>(), b.template get<4>(), b.template get<5>());
            }, K, total_functions);
          }
        } else
#endif // MRA_ENABLE_HOST
        {
          auto sparseman = make_sparsity_manager(combined);
          sparseman.populate_device_sparsity();
          submit_merge_kernel<NDIM>(key, input_view, parent_view, out_view, N, n_nonzero, K,
                                    ttg::device::current_stream());
        }
      }
      /* to the compute (0-2) and spawn (3-5) tasks: 0/3 center, 1/4 left input of the right
       * neighbor, 2/5 right input of the left neighbor. The node is sent directly to every
       * task that reads it rather than forwarded by a reader. */
      std::vector<key_type> self = {key}, right_keys, left_keys;
      if (flags & RIGHT_EXISTS) right_keys.push_back(key.neighbor(axis, 1));
      if (flags & LEFT_EXISTS) left_keys.push_back(key.neighbor(axis, -1));
#ifndef MRA_ENABLE_HOST
      /* the device broadcast does not skip empty key lists */
      if (!right_keys.empty() && !left_keys.empty()) {
        co_await ttg::device::broadcast<0, 1, 2, 3, 4, 5>(
          std::make_tuple(self, right_keys, left_keys, self, right_keys, left_keys), std::move(combined));
      } else if (!right_keys.empty()) {
        co_await ttg::device::broadcast<0, 1, 3, 4>(
          std::make_tuple(self, right_keys, self, right_keys), std::move(combined));
      } else if (!left_keys.empty()) {
        co_await ttg::device::broadcast<0, 2, 3, 5>(
          std::make_tuple(self, left_keys, self, left_keys), std::move(combined));
      } else {
        co_await ttg::device::broadcast<0, 3>(std::make_tuple(self, self), std::move(combined));
      }
#else  // MRA_ENABLE_HOST
      ttg::broadcast<0, 1, 2, 3, 4, 5>(std::make_tuple(self, right_keys, left_keys, self, right_keys, left_keys),
                                       std::move(combined));
#endif // MRA_ENABLE_HOST
    };
    auto assemble_tt = ttg::make_tt<Space>(std::move(assemble_fn),
                                           ttg::edges(ttg::fuse(in, in_sub), inh, parent_e),
                                           ttg::edges(b_center, b_left, b_right, s_center, s_left, s_right),
                                           name + "-assemble", {"input", "inherited", "parent"},
                                           {"center", "left", "right", "spawn_center", "spawn_left", "spawn_right"});

    /**
     * Compute the derivative of the functions with a leaf here, and tell the neighbors
     * (or, where there is none, the own spawn task) whether this node has children.
     */
    auto compute_fn = [&db, &functiondata, fns, K, axis, plan, has_children,
                       enable_batching, batch_pool, total_functions](
                          const key_type& key, const node_type& center, const node_type& left,
                          const node_type& right) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      const std::uint8_t flags = center.flags();
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

      /* scratch and source keys must outlive the kernel: destroying a buffer releases its
       * device memory right away, while the kernel launched below is still running */
      ttg::Buffer<T, DeviceAllocator<T>> tmp;
      ttg::Buffer<key_type, DeviceAllocator<key_type>> src_keys;
      if (n_nonzero > 0) {
        const auto& operators = functiondata.get_operators();
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        /* batched, the scratch is the batch slot's, see submit_derivative_batch_leader */
#ifndef MRA_ENABLE_HOST
        const std::size_t tmp_size = enable_batching ? 0 : derivative_tmp_size<NDIM>(K)*n_nonzero;
#else
        const std::size_t tmp_size = derivative_tmp_size<NDIM>(K)*n_nonzero;
#endif
        if (tmp_size > 0) tmp = ttg::Buffer<T, DeviceAllocator<T>>(tmp_size, TempScope);
        /* per function, the nodes the coefficients of the left, center, and right nodes
         * belong to; the kernel projects them to the neighbors and to this node */
        src_keys = ttg::Buffer<key_type, DeviceAllocator<key_type>>(3*N);
        key_type* sk = src_keys.host_ptr();
        for (size_type j = 0; j < N; ++j) {
          sk[3*j + 0] = left.origin_key(j);
          sk[3*j + 1] = center.origin_key(j);
          sk[3*j + 2] = right.origin_key(j);
        }
#ifndef MRA_ENABLE_HOST
        auto inputs = ttg::device::Input(db, result.coeffs().buffer(), operators.buffer(), phibar.buffer(),
                                         phi.buffer(), quad_x.buffer(), src_keys);
        if (tmp_size > 0) inputs.add(tmp);
        for (const node_type* n : {&center, &left, &right}) {
          if (!n->empty()) inputs.add(n->coeffs().buffer());
        }
        co_await ttg::device::select(inputs);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto result_view = result.coeffs().current_view();
        auto left_view = left.coeffs().current_view(), center_view = center.coeffs().current_view();
        auto right_view = right.coeffs().current_view();
        const key_type* src_keys_ptr = src_keys.current_device_ptr();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the result's sparsity is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<key_type>(key, src_keys_ptr, left_view, center_view, right_view,
                                                            result_view, N, result.coeffs(), n_nonzero);
          detail::submit_derivative_batch_leader<T, NDIM>(batch, *batch_pool, D, operators.current_view(),
                                                          phi.current_view(), phibar.current_view(),
                                                          quad_x.current_view(), K, T(0), T(0), axis,
                                                          FunctionData<T, NDIM>::BC_FREE,
                                                          FunctionData<T, NDIM>::BC_FREE, total_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          auto sparseman = make_sparsity_manager(result);
          sparseman.populate_device_sparsity();
          submit_derivative_kernel(D, key, src_keys_ptr, left_view, center_view, right_view,
                                   operators.current_view(), result_view,
                                   phi.current_view(), phibar.current_view(), quad_x.current_view(),
                                   tmp.current_device_ptr(), N, n_nonzero, K, T(0), T(0), axis,
                                   FunctionData<T, NDIM>::BC_FREE, FunctionData<T, NDIM>::BC_FREE,
                                   ttg::device::current_stream());
        }
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
                                          ttg::edges(b_center, b_left, b_right),
                                          ttg::edges(out, s_ldec, s_rdec),
                                          name, {"center", "left", "right"},
                                          {"out", "left_has_children", "right_has_children"});

    /**
     * Start the children, if this node has any. Every child receives the status of each
     * function from above together with whether this node's neighbors have children (in
     * a node without coefficients), and this node, whose coefficients it inherits. A
     * child whose neighbor across this node's boundary does not exist receives this
     * node's neighbor in its place. All of it is shared by the children, and the nodes
     * passed on are the ones this task received, so no data is copied or computed here.
     */
    auto spawn_fn = [fns, axis, plan, has_children, invalid_node](
                        const key_type& key, const node_type& center, const node_type& left,
                        const node_type& right, const int& left_dec, const int& right_dec) -> TASKTYPE {
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

      /* Inner: the input continues (the children take the input), Leaf: refined here,
       * Invalid: done here, coefficients only needed by the neighbors' children */
      node_type inherited(key, N);
      for (size_type j = 0; j < N; ++j) {
        LeafStatus ks = LeafStatus::Invalid;
        if (center.leaf_status(j) == LeafStatus::Inner) ks = LeafStatus::Inner;
        else if (center.leaf_status(j) == LeafStatus::Leaf && status[j] == LeafStatus::Inner) ks = LeafStatus::Leaf;
        inherited.set_leaf(j, ks);
      }
      inherited.set_flags((left_dec ? LEFT_HAS_CHILDREN : 0) | (right_dec ? RIGHT_HAS_CHILDREN : 0));

      std::vector<key_type> all_children, left_children, right_children;
      for (auto child : children(key)) {
        all_children.push_back(child);
        (child.is_left_child(axis) ? left_children : right_children).push_back(child);
      }

      /* 0: inherited, 1: parent, 2: input substitute, 3/5: left input of compute/spawn,
       * 4/6: right input of compute/spawn. A node at the boundary has no neighbor beyond
       * it; the children take an invalid node in its place. */
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(
        ttg::device::broadcast<0>(all_children, std::move(inherited)),
        ttg::device::broadcast<1>(all_children, center));
      if (!children_in_input) {
        sends.push_back(ttg::device::broadcast<2>(all_children, invalid_node(key, N)));
      }
      if (lb) {
        sends.push_back(ttg::device::broadcast<3, 5>(std::make_tuple(left_children, left_children), invalid_node(key, N)));
      } else if (sub_left) {
        sends.push_back(ttg::device::broadcast<3, 5>(std::make_tuple(left_children, left_children), left));
      }
      if (rb) {
        sends.push_back(ttg::device::broadcast<4, 6>(std::make_tuple(right_children, right_children), invalid_node(key, N)));
      } else if (sub_right) {
        sends.push_back(ttg::device::broadcast<4, 6>(std::make_tuple(right_children, right_children), right));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::broadcast<0>(all_children, std::move(inherited));
      ttg::broadcast<1>(all_children, center);
      if (!children_in_input) ttg::broadcast<2>(all_children, invalid_node(key, N));
      if (lb) ttg::broadcast<3, 5>(std::make_tuple(left_children, left_children), invalid_node(key, N));
      else if (sub_left) ttg::broadcast<3, 5>(std::make_tuple(left_children, left_children), left);
      if (rb) ttg::broadcast<4, 6>(std::make_tuple(right_children, right_children), invalid_node(key, N));
      else if (sub_right) ttg::broadcast<4, 6>(std::make_tuple(right_children, right_children), right);
#endif // MRA_ENABLE_HOST
    };
    auto spawn_tt = ttg::make_tt<Space>(std::move(spawn_fn),
                                        ttg::edges(s_center, s_left, s_right, s_ldec, s_rdec),
                                        ttg::edges(inh, parent_e, in_sub, b_left, b_right, s_left, s_right),
                                        name + "-spawn",
                                        {"center", "left", "right", "left_has_children", "right_has_children"},
                                        {"inherited", "parent", "input", "left", "right", "spawn_left", "spawn_right"});
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
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      compute_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
      assemble_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
    }
#endif // MRA_ENABLE_HOST

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
