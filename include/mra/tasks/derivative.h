#ifndef MRA_TASKS_DERIVATIVE_H
#define MRA_TASKS_DERIVATIVE_H

#include <algorithm>
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
   * Coefficients are never refined level by level: a node passes coefficients down
   * unchanged and records per function at which ancestor they originate
   * (FunctionsReconstructedNode::origin). They are projected once, from their origin
   * to the node where a derivative is formed (cf. madness's parent_to_child).
   *
   * At the boundaries of the domain the neighbor is absent; the boundary condition is
   * that of the derivative operators in functiondata (see FunctionData).
   *
   * Input and output are reconstructed trees with a node at every key of the tree
   * and leaf status per function, as produced by make_reconstruct; inner nodes of
   * the output carry no coefficients.
   *
   * A node of the (refined) tree is described by three nodes: the status of each
   * function and which neighbors exist (meta, without coefficients), the input node
   * (own, empty where the key is not in the input tree), and the coefficients
   * inherited from the ancestors (inh) of functions whose input tree ended above.
   * A function has coefficients in at most one of own and inh, and the derivative
   * kernel reads them from whichever holds them, so own and inh are never merged at
   * a node. The inherited node is passed down unchanged where possible: the children
   * of a node inherit its inh if the node holds no coefficients of its own, its own
   * node if it inherited none, and only if it has both are they merged.
   *
   * Every node of the (refined) tree goes through three tasks:
   *  - assemble: forms the meta node from the input node and the status of each
   *    function passed down by the parent, and sends it with the own and inherited
   *    nodes to its two neighbors along the axis;
   *  - compute: decides per function whether to compute here or refine, forms the
   *    derivatives, and tells its neighbors whether it has children;
   *  - spawn: starts the children, if any. A child whose neighbor across the
   *    parent's boundary does not exist (the neighbor of the parent has no children)
   *    receives the parent's neighbor instead. The only device work of spawn is the
   *    merge of the own and inherited coefficients where the children need both.
   * Each node knows which of its neighbors exist from its parent, so that every
   * message has a receiver: the parent records whether its neighbors have children
   * in the flags of the node it sends to all children, and every child derives from
   * them which of its own neighbors exist; the meta node carries the result to
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
    /* node flags: in the meta node, which neighbors along the axis exist; in the
     * node a parent sends to its children, whether the parent's neighbors have children */
    constexpr const std::uint8_t LEFT_EXISTS = 1, RIGHT_EXISTS = 2;
    constexpr const std::uint8_t LEFT_HAS_CHILDREN = 1, RIGHT_HAS_CHILDREN = 2;

    if (axis >= NDIM) throw std::runtime_error(name + ": invalid axis");

    /* batching of the compute and merge kernels is controlled process-wide
     * (mra/misc/batch_size.h), read once here; the operator data is the same for every
     * node, so any two tasks may be batched */
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

    /* down: the status of each function at the children, without coefficients;
     * down_c: the coefficients they inherit */
    node_edge in_sub(name + "-in_sub"), down(name + "-inherit"), down_c(name + "-inherit_coeffs");
    /* a node as read by compute (b_) and spawn (s_): meta, own, and inherited node of
     * the node itself (center), of the left and of the right neighbor */
    auto triple = [&name](const std::string& what) {
      return std::array<node_edge, 3>{node_edge(name + "-" + what + "_meta"), node_edge(name + "-" + what + "_own"),
                                      node_edge(name + "-" + what + "_inh")};
    };
    auto b_center = triple("center"), b_left = triple("left"), b_right = triple("right");
    auto s_center = triple("s_center"), s_left = triple("s_left"), s_right = triple("s_right");
    flag_edge s_ldec(name + "-left_has_children"), s_rdec(name + "-right_has_children");

    /* a node that holds no coefficients for any of N functions, all Invalid */
    auto invalid_node = [](const key_type& key, size_type N) {
      node_type n(key, N);
      n.set_all_leaf(LeafStatus::Invalid);
      return n;
    };
    auto has_coeffs = [](const node_type& n, size_type j) { return !n.empty() && n.is_nonzero(j); };

    /* status of each function of the derivative at a node, given the meta nodes of
     * the node and of its neighbors: Inner where the tree continues or the function
     * is refined here, Leaf where the derivative is formed here */
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
        /* the same key to each of the 14 outputs */
        auto to_all = [&]<std::size_t... I>(std::index_sequence<I...>) {
#ifndef MRA_ENABLE_HOST
          return ttg::device::broadcast<I...>(std::make_tuple(((void)I, key)...), invalid_node(key, N));
#else  // MRA_ENABLE_HOST
          ttg::broadcast<I...>(std::make_tuple(((void)I, key)...), invalid_node(key, N));
#endif // MRA_ENABLE_HOST
        };
#ifndef MRA_ENABLE_HOST
        co_await to_all(std::make_index_sequence<14>{});
#else  // MRA_ENABLE_HOST
        to_all(std::make_index_sequence<14>{});
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(in),
                                         ttg::edges(down, down_c,
                                                    b_left[0], b_left[1], b_left[2], b_right[0], b_right[1], b_right[2],
                                                    s_left[0], s_left[1], s_left[2], s_right[0], s_right[1], s_right[2]),
                                         name + "-primer");

    /**
     * Assemble: the status of each function, that of the input node where the function
     * has a leaf or children here, else the one the parent decided (Leaf: to be
     * differentiated here or below, Invalid: only needed by the neighbors). The input
     * node and the inherited coefficients are passed on unchanged.
     */
    auto assemble_fn = [fns, axis, neighbor_flags](const key_type& key, const node_type& input,
                                             const node_type& inherited, const node_type& inh) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      const std::uint8_t flags = neighbor_flags(key, inherited.flags());
      node_type meta(key, N);
      for (size_type j = 0; j < N; ++j) {
        const LeafStatus s = input.leaf_status(j);
        meta.set_leaf(j, (s == LeafStatus::Inner || s == LeafStatus::Leaf) ? s : inherited.leaf_status(j));
      }
      meta.set_flags(flags);
      /* to the compute (receivers 0-2) and spawn (3-5) tasks: 0/3 center, 1/4 left input
       * of the right neighbor, 2/5 right input of the left neighbor; receiver R takes
       * meta, own, and inherited node on outputs 3R, 3R+1, 3R+2. The nodes are sent
       * directly to every task that reads them rather than forwarded by a reader. */
      std::vector<key_type> self = {key}, right_keys, left_keys;
      if (flags & RIGHT_EXISTS) right_keys.push_back(key.neighbor(axis, 1));
      if (flags & LEFT_EXISTS) left_keys.push_back(key.neighbor(axis, -1));
      auto to = [&]<std::size_t... R>(std::index_sequence<R...>, const auto&... keys) {
#ifndef MRA_ENABLE_HOST
        return ttg::device::forward(
          ttg::device::broadcast<(3*R + 0)...>(std::make_tuple(keys...), std::move(meta)),
          ttg::device::broadcast<(3*R + 1)...>(std::make_tuple(keys...), input),
          ttg::device::broadcast<(3*R + 2)...>(std::make_tuple(keys...), inh));
#else  // MRA_ENABLE_HOST
        ttg::broadcast<(3*R + 0)...>(std::make_tuple(keys...), std::move(meta));
        ttg::broadcast<(3*R + 1)...>(std::make_tuple(keys...), input);
        ttg::broadcast<(3*R + 2)...>(std::make_tuple(keys...), inh);
#endif // MRA_ENABLE_HOST
      };
#ifndef MRA_ENABLE_HOST
      /* the device broadcast does not skip empty key lists */
      if (!right_keys.empty() && !left_keys.empty()) {
        co_await to(std::index_sequence<0, 1, 2, 3, 4, 5>{}, self, right_keys, left_keys, self, right_keys, left_keys);
      } else if (!right_keys.empty()) {
        co_await to(std::index_sequence<0, 1, 3, 4>{}, self, right_keys, self, right_keys);
      } else if (!left_keys.empty()) {
        co_await to(std::index_sequence<0, 2, 3, 5>{}, self, left_keys, self, left_keys);
      } else {
        co_await to(std::index_sequence<0, 3>{}, self, self);
      }
#else  // MRA_ENABLE_HOST
      to(std::index_sequence<0, 1, 2, 3, 4, 5>{}, self, right_keys, left_keys, self, right_keys, left_keys);
#endif // MRA_ENABLE_HOST
    };
    auto assemble_tt = ttg::make_tt<Space>(std::move(assemble_fn),
                                           ttg::edges(ttg::fuse(in, in_sub), down, down_c),
                                           ttg::edges(b_center[0], b_center[1], b_center[2],
                                                      b_left[0], b_left[1], b_left[2],
                                                      b_right[0], b_right[1], b_right[2],
                                                      s_center[0], s_center[1], s_center[2],
                                                      s_left[0], s_left[1], s_left[2],
                                                      s_right[0], s_right[1], s_right[2]),
                                           name + "-assemble", {"input", "inherited", "inherited_coeffs"},
                                           {"center_meta", "center_own", "center_inh",
                                            "left_meta", "left_own", "left_inh",
                                            "right_meta", "right_own", "right_inh",
                                            "spawn_center_meta", "spawn_center_own", "spawn_center_inh",
                                            "spawn_left_meta", "spawn_left_own", "spawn_left_inh",
                                            "spawn_right_meta", "spawn_right_own", "spawn_right_inh"});

    /**
     * Compute the derivative of the functions with a leaf here, and tell the neighbors
     * (or, where there is none, the own spawn task) whether this node has children.
     */
    auto compute_fn = [&db, &functiondata, fns, K, axis, plan, has_children, has_coeffs,
                       enable_batching, batch_pool, total_functions](
                          const key_type& key,
                          const node_type& center, const node_type& center_own, const node_type& center_inh,
                          const node_type& left, const node_type& left_own, const node_type& left_inh,
                          const node_type& right, const node_type& right_own, const node_type& right_inh) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      const std::uint8_t flags = center.flags();
      const bool lb = key.is_left_boundary(axis), rb = key.is_right_boundary(axis);
      std::vector<LeafStatus> status = plan(key, center, left, right, N);
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      for (size_type j = 0; j < N; ++j) {
        if (status[j] != LeafStatus::Leaf) continue;
        const bool nonzero = has_coeffs(center_own, j) || has_coeffs(center_inh, j)
                          || (!lb && (has_coeffs(left_own, j) || has_coeffs(left_inh, j)))
                          || (!rb && (has_coeffs(right_own, j) || has_coeffs(right_inh, j)));
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
        tmp = ttg::Buffer<T, DeviceAllocator<T>>(derivative_tmp_size<NDIM>(K)*n_nonzero, TempScope);
        /* per function, the nodes the coefficients at the left neighbor, here, and at the
         * right neighbor belong to; the kernel projects them to the neighbors and to this
         * node. The kernel takes them from the own node, else from the inherited node. */
        src_keys = ttg::Buffer<key_type, DeviceAllocator<key_type>>(3*N);
        key_type* sk = src_keys.host_ptr();
        auto src_key = [&](const node_type& meta, const node_type& own, const node_type& inh, size_type j) {
          if (has_coeffs(own, j)) return own.origin_key(j);
          if (has_coeffs(inh, j)) return inh.origin_key(j);
          return meta.key();
        };
        for (size_type j = 0; j < N; ++j) {
          sk[3*j + 0] = src_key(left, left_own, left_inh, j);
          sk[3*j + 1] = src_key(center, center_own, center_inh, j);
          sk[3*j + 2] = src_key(right, right_own, right_inh, j);
        }
#ifndef MRA_ENABLE_HOST
        auto inputs = ttg::device::Input(db, result.coeffs().buffer(), operators.buffer(), phibar.buffer(),
                                         phi.buffer(), quad_x.buffer(), tmp, src_keys);
        /* siblings share their inherited node, so one may be passed more than once */
        std::array<const void*, 6> added{};
        std::size_t n_added = 0;
        for (const node_type* n : {&center_own, &center_inh, &left_own, &left_inh, &right_own, &right_inh}) {
          if (n->empty()) continue;
          const void* b = &n->coeffs().buffer();
          if (std::find(added.begin(), added.begin() + n_added, b) != added.begin() + n_added) continue;
          added[n_added++] = b;
          inputs.add(n->coeffs().buffer());
        }
        co_await ttg::device::select(inputs);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto result_view = result.coeffs().current_view();
        auto left_view = left_own.coeffs().current_view(), left_inh_view = left_inh.coeffs().current_view();
        auto center_view = center_own.coeffs().current_view(), center_inh_view = center_inh.coeffs().current_view();
        auto right_view = right_own.coeffs().current_view(), right_inh_view = right_inh.coeffs().current_view();
        const key_type* src_keys_ptr = src_keys.current_device_ptr();
        T* tmp_ptr = tmp.current_device_ptr();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the result's sparsity is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<key_type>(key, src_keys_ptr, left_view, left_inh_view,
                                                            center_view, center_inh_view, right_view, right_inh_view,
                                                            result_view, tmp_ptr, N, result.coeffs(), n_nonzero);
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
          submit_derivative_kernel(D, key, src_keys_ptr, left_view, left_inh_view, center_view, center_inh_view,
                                   right_view, right_inh_view, operators.current_view(), result_view,
                                   phi.current_view(), phibar.current_view(), quad_x.current_view(),
                                   tmp_ptr, N, n_nonzero, K, T(0), T(0), axis,
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
                                          ttg::edges(b_center[0], b_center[1], b_center[2],
                                                     b_left[0], b_left[1], b_left[2],
                                                     b_right[0], b_right[1], b_right[2]),
                                          ttg::edges(out, s_ldec, s_rdec),
                                          name, {"center_meta", "center_own", "center_inh",
                                                 "left_meta", "left_own", "left_inh",
                                                 "right_meta", "right_own", "right_inh"},
                                          {"out", "left_has_children", "right_has_children"});

    /**
     * Start the children, if this node has any. Every child receives the status of each
     * function from above together with whether this node's neighbors have children (in
     * a node without coefficients), and the coefficients it inherits: this node's
     * inherited node if this node holds no coefficients of its own, its own (input) node
     * if it inherited none, else the two merged. A child whose neighbor across this
     * node's boundary does not exist receives this node's neighbor in its place. All of
     * it is shared by the children; apart from the merge, the nodes passed on are the
     * ones this task received, so no data is copied or computed here.
     */
    auto spawn_fn = [fns, K, axis, plan, has_children, has_coeffs, invalid_node,
                     enable_batching, merge_pool, total_functions](
                        const key_type& key,
                        const node_type& center, const node_type& center_own, const node_type& center_inh,
                        const node_type& left, const node_type& left_own, const node_type& left_inh,
                        const node_type& right, const node_type& right_own, const node_type& right_inh,
                        const int& left_dec, const int& right_dec) -> TASKTYPE {
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

      /* the coefficients the children inherit: merged only if this node has both */
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      bool any_own = false, any_inh = false;
      for (size_type j = 0; j < N; ++j) {
        const bool own = has_coeffs(center_own, j), inh = has_coeffs(center_inh, j);
        any_own |= own;
        any_inh |= inh;
        if (own || inh) sparsity.set_nonzero(j);
      }
      const bool merge = any_own && any_inh;
      node_type merged;
      if (merge) {
        const size_type n_nonzero = sparsity.count_nonzero();
        merged = node_type(key, sparsity, K, ttg::scope::Allocate);
        for (size_type j = 0; j < N; ++j) {
          if (has_coeffs(center_own, j)) merged.set_origin(j, center_own.origin(j));
          else if (has_coeffs(center_inh, j))
            merged.set_origin(j, center_inh.origin(j) + (key.level() - center_inh.key().level()));
        }
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::select(merged.coeffs().buffer(), center_own.coeffs().buffer(),
                                     center_inh.coeffs().buffer());
#endif // MRA_ENABLE_HOST
        auto out_view = merged.coeffs().current_view();
        auto own_view = center_own.coeffs().current_view(), inh_view = center_inh.coeffs().current_view();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the sparsity of the merged node is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<key_type>(inh_view, own_view, out_view, merged.coeffs(),
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
          auto sparseman = make_sparsity_manager(merged);
          sparseman.populate_device_sparsity();
          submit_merge_kernel<NDIM>(key, inh_view, own_view, out_view, N, n_nonzero, K,
                                    ttg::device::current_stream());
        }
      }

      std::vector<key_type> all_children, left_children, right_children;
      for (auto child : children(key)) {
        all_children.push_back(child);
        (child.is_left_child(axis) ? left_children : right_children).push_back(child);
      }

      /* 0: inherited status, 1: inherited coefficients, 2: input substitute, 3-5/9-11:
       * left meta, own, and inherited node of compute/spawn, 6-8/12-14: right of
       * compute/spawn. A node at the boundary has no neighbor beyond it; the children
       * take an invalid node in its place. */
      auto lr = std::make_tuple(left_children, left_children), rr = std::make_tuple(right_children, right_children);
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(ttg::device::broadcast<0>(all_children, std::move(inherited)));
      if (merge)        sends.push_back(ttg::device::broadcast<1>(all_children, std::move(merged)));
      else if (any_own) sends.push_back(ttg::device::broadcast<1>(all_children, center_own));
      else              sends.push_back(ttg::device::broadcast<1>(all_children, center_inh));
      if (!children_in_input) {
        sends.push_back(ttg::device::broadcast<2>(all_children, invalid_node(key, N)));
      }
      if (lb) {
        sends.push_back(ttg::device::broadcast<3, 4, 5, 9, 10, 11>(
          std::tuple_cat(lr, lr, lr), invalid_node(key, N)));
      } else if (sub_left) {
        sends.push_back(ttg::device::broadcast<3, 9>(lr, left));
        sends.push_back(ttg::device::broadcast<4, 10>(lr, left_own));
        sends.push_back(ttg::device::broadcast<5, 11>(lr, left_inh));
      }
      if (rb) {
        sends.push_back(ttg::device::broadcast<6, 7, 8, 12, 13, 14>(
          std::tuple_cat(rr, rr, rr), invalid_node(key, N)));
      } else if (sub_right) {
        sends.push_back(ttg::device::broadcast<6, 12>(rr, right));
        sends.push_back(ttg::device::broadcast<7, 13>(rr, right_own));
        sends.push_back(ttg::device::broadcast<8, 14>(rr, right_inh));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::broadcast<0>(all_children, std::move(inherited));
      if (merge)        ttg::broadcast<1>(all_children, std::move(merged));
      else if (any_own) ttg::broadcast<1>(all_children, center_own);
      else              ttg::broadcast<1>(all_children, center_inh);
      if (!children_in_input) ttg::broadcast<2>(all_children, invalid_node(key, N));
      if (lb) {
        ttg::broadcast<3, 4, 5, 9, 10, 11>(std::tuple_cat(lr, lr, lr), invalid_node(key, N));
      } else if (sub_left) {
        ttg::broadcast<3, 9>(lr, left);
        ttg::broadcast<4, 10>(lr, left_own);
        ttg::broadcast<5, 11>(lr, left_inh);
      }
      if (rb) {
        ttg::broadcast<6, 7, 8, 12, 13, 14>(std::tuple_cat(rr, rr, rr), invalid_node(key, N));
      } else if (sub_right) {
        ttg::broadcast<6, 12>(rr, right);
        ttg::broadcast<7, 13>(rr, right_own);
        ttg::broadcast<8, 14>(rr, right_inh);
      }
#endif // MRA_ENABLE_HOST
    };
    auto spawn_tt = ttg::make_tt<Space>(std::move(spawn_fn),
                                        ttg::edges(s_center[0], s_center[1], s_center[2],
                                                   s_left[0], s_left[1], s_left[2],
                                                   s_right[0], s_right[1], s_right[2], s_ldec, s_rdec),
                                        ttg::edges(down, down_c, in_sub,
                                                   b_left[0], b_left[1], b_left[2], b_right[0], b_right[1], b_right[2],
                                                   s_left[0], s_left[1], s_left[2], s_right[0], s_right[1], s_right[2]),
                                        name + "-spawn",
                                        {"center_meta", "center_own", "center_inh",
                                         "left_meta", "left_own", "left_inh",
                                         "right_meta", "right_own", "right_inh",
                                         "left_has_children", "right_has_children"},
                                        {"inherited", "inherited_coeffs", "input",
                                         "left_meta", "left_own", "left_inh", "right_meta", "right_own", "right_inh",
                                         "spawn_left_meta", "spawn_left_own", "spawn_left_inh",
                                         "spawn_right_meta", "spawn_right_own", "spawn_right_inh"});
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
      spawn_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
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
