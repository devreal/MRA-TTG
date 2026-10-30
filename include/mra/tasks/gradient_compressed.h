#ifndef MRA_TASKS_GRADIENT_COMPRESSED_H
#define MRA_TASKS_GRADIENT_COMPRESSED_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/batch_size.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/tensor/functionnode.h"
#include "mra/kernels/derivative.h"
#include "mra/kernels/reconstruct.h"
#include "mra/kernels/gradient_compressed.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>
#include <ttg/serialization/std/vector.h>

namespace mra {

  /**
   * Per function, one bit per axis; see make_gradient_compressed. A plain host value
   * (like KeptInfo), never shipped to the device.
   */
  struct GradientTags : public ttg::TTValue<GradientTags> {
  private:
    std::vector<std::uint8_t> m_tags;

  public:
    GradientTags() = default;
    GradientTags(const GradientTags&) = delete;
    GradientTags& operator=(const GradientTags&) = delete;
    GradientTags(GradientTags&&) = default;
    GradientTags& operator=(GradientTags&&) = default;

    explicit GradientTags(size_type N, std::uint8_t value = 0) : m_tags(N, value) { }

    size_type size() const { return static_cast<size_type>(m_tags.size()); }
    std::uint8_t& operator[](size_type i) { return m_tags[i]; }
    std::uint8_t operator[](size_type i) const { return m_tags[i]; }
    bool any() const { return std::any_of(m_tags.begin(), m_tags.end(), [](std::uint8_t t) { return t != 0; }); }

    template <typename Archive>
    void serialize(Archive& ar) { ar & m_tags; }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int) { serialize(ar); }
  };

  /**
   * The derivatives along all three axes of every function of a set given in
   * compressed form, without reconstructing it first; out[axis] receives the
   * derivative along axis in reconstructed form, the same trees and leaf status as
   * make_derivative (from the reconstructed input) produces for that axis.
   *
   * One descent over the union of the trees of the three derivatives. Every node of
   * it holds, for every function, scaling coefficients (the scaling node S): those at
   * the node, unfiltered from [s, d] of the parent, where the input continues at the
   * parent; else the parent's, passed on unchanged and recorded as belonging to an
   * ancestor (FunctionsReconstructedNode::origin), so that, as in make_derivative,
   * they are projected only once, by the derivative kernel, where a derivative is
   * formed from them. The status of a function in S is that of the input tree: Inner
   * where it has children, Leaf at its leaf, Invalid below it.
   *
   * Every node goes through four tasks, of which only compute does device work:
   *  - compute (device): from S of the node and of its six face neighbors decides per
   *    axis and function whether the derivative is formed here (Leaf) or below (Inner:
   *    the input continues here, or is finer at a neighbor along the axis), forms the
   *    derivatives, and, if the node has children, forms S of the children from S and
   *    the compressed node in the same task;
   *  - notify: tells the neighbors whether this node has children and sends the
   *    children without a compressed node an empty one. These small messages are sent
   *    here rather than by compute because a device task holds a data copy of every
   *    value it sends until it completes, and PaRSEC limits their number per task
   *    (MAX_PARAM_COUNT, including the inputs);
   *  - spawn: once the neighbors have said whether they have children, tells
   *    every child which of its neighbors exist (its flags); without children, sends S
   *    to the children of the neighbors that have some, as the nearest node at the
   *    position of their missing neighbor;
   *  - forward: sends S of the node, which the parent's compute sent once, to
   *    the node's compute and spawn and to the compute tasks of its existing neighbors
   *    (the siblings, and those across the parent's boundary that its flags name), and
   *    an invalid node in place of a neighbor beyond the boundary of the domain. It
   *    only sends its input, so nothing is copied.
   *
   * Which neighbors exist follows make_derivative: a sibling always does, a neighbor
   * across the parent's boundary iff the parent's neighbor has children (flag bit
   * 2*axis for the left neighbor, 2*axis+1 for the right). Where a neighbor does not
   * exist its place is taken by the nearest existing node at its position, a coarser
   * one: the neighbor of the parent sends its own S if it has no children, and where
   * that neighbor of the parent does not exist either the parent passes on the one it
   * received. The derivative kernel projects coefficients from the node they belong
   * to, so a coarser neighbor needs no refinement.
   *
   * The membership of a function in the tree of each axis travels as GradientTags from
   * compute to the children's compute (bit a: Inner along axis a here, so the tree of
   * axis a includes the children).
   *
   * Inputs must be standard compressed form with a compressed node at every key where
   * some function of the set has children, as produced by make_compress. At the
   * boundaries of the domain the boundary condition is free, as in make_derivative.
   *
   * The compute task uses up to 28 device buffers (S, six neighbors, three results,
   * eight children, the compressed node, operator data and scratch) and holds up to 23
   * data copies (10 inputs, 3 results, 8 children, the tags and the summary), so PaRSEC
   * must be built with PARSEC_MAX_PARAM_COUNT of at least 28.
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_gradient_compressed(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetT>& fns,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
    std::array<ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>, NDIM> out,
    const std::string& name = "gradient_compressed",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    static_assert(NDIM == 3, "make_gradient_compressed: the terminal layout is written for three axes");
#if !defined(MRA_ENABLE_HOST) && defined(MAX_PARAM_COUNT)
    static_assert(MAX_PARAM_COUNT >= 28, "make_gradient_compressed needs PaRSEC built with PARSEC_MAX_PARAM_COUNT >= 28");
#endif
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;
    using cnode_type = mra::FunctionsCompressedNode<T, NDIM>;
    using key_type = mra::Key<NDIM>;
    using node_edge = ttg::Edge<key_type, node_type>;
    using tags_edge = ttg::Edge<key_type, GradientTags>;
    using flag_edge = ttg::Edge<key_type, int>;
    constexpr size_type num_children = key_type::num_children();
    constexpr std::size_t NB = 2*NDIM; // face neighbors

    /* the neighbor along axis a on side s (-1 left, +1 right) */
    auto nb_index = [](Dimension a, int s) -> std::size_t { return 2*a + (s > 0 ? 1 : 0); };
    auto at_boundary = [](const key_type& key, Dimension a, int s) {
      return s < 0 ? key.is_left_boundary(a) : key.is_right_boundary(a);
    };
    /* the children of key on side s of axis a */
    auto children_on_side = [](const key_type& key, Dimension a, int s) {
      std::vector<key_type> r;
      for (size_type c = 0; c < num_children; ++c) {
        if (((c >> a) & 1) == (s > 0 ? 1u : 0u)) r.push_back(key.child_at(c));
      }
      return r;
    };
    auto invalid_node = [](const key_type& key, size_type N) {
      node_type n(key, N);
      n.set_all_leaf(LeafStatus::Invalid);
      return n;
    };
    auto has_coeffs = [](const node_type& n, size_type j) { return !n.empty() && n.is_nonzero(j); };

    /* S into forward and from there to compute and spawn; the flags of a node into
     * forward and compute; the tags into compute; the six neighbors of compute; the
     * placeholder for a child without a compressed node; whether a node has children,
     * to its spawn and to the spawn of its neighbors */
    node_edge s_forward(name + "-s_forward"), s_compute(name + "-s_compute"), s_spawn(name + "-s_spawn");
    flag_edge flags_forward(name + "-flags_forward"), flags_compute(name + "-flags_compute");
    tags_edge tags_compute(name + "-tags");
    flag_edge children_spawn(name + "-has_children");
    std::array<node_edge, NB> nb_e;
    std::array<flag_edge, NB> dec_e;
    for (std::size_t i = 0; i < NB; ++i) {
      nb_e[i] = node_edge(name + "-neighbor" + std::to_string(i));
      dec_e[i] = flag_edge(name + "-neighbor_has_children" + std::to_string(i));
    }
    ttg::Edge<key_type, cnode_type> cnode_balance(name + "-cnode_balance");
    /* compute's summary for its notify task, see NOTIFY_* */
    flag_edge notify_e(name + "-notify");

    /* batching of the compute task is controlled process-wide (mra/misc/batch_size.h),
     * read once here; the operator data is the same for every node, so any two tasks
     * may be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type max_functions = fns->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::GradientDerivativeBatchArg<T, NDIM>>> deriv_pool;
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::GradientSpawnBatchArg<T, NDIM>>> spawn_pool;
    if (enable_batching) {
      deriv_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::GradientDerivativeBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
      spawn_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::GradientSpawnBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t deriv_pool = nullptr, spawn_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

#ifndef MRA_ENABLE_HOST
#define MRA_GRAD_SENDS                                                               \
      auto sends = ttg::device::forward();                                           \
      auto do_send = [&]<std::size_t I, typename V>(const key_type& k, V&& v) {      \
        sends.push_back(ttg::device::send<I>(k, std::forward<V>(v)));                \
      };
#define MRA_GRAD_FLUSH co_await std::move(sends);
#else  // MRA_ENABLE_HOST
#define MRA_GRAD_SENDS                                                               \
      auto do_send = []<std::size_t I, typename V>(const key_type& k, V&& v) {       \
        ttg::send<I>(k, std::forward<V>(v));                                          \
      };
#define MRA_GRAD_FLUSH
#endif // MRA_ENABLE_HOST
    /* send to output terminal Base + t, t in [0, NB) known at run time */
#define MRA_GRAD_SEND_NB(Base, t, k, v)                                              \
      switch (t) {                                                                   \
        case 0: do_send.template operator()<(Base) + 0>(k, v); break;                \
        case 1: do_send.template operator()<(Base) + 1>(k, v); break;                \
        case 2: do_send.template operator()<(Base) + 2>(k, v); break;                \
        case 3: do_send.template operator()<(Base) + 3>(k, v); break;                \
        case 4: do_send.template operator()<(Base) + 4>(k, v); break;                \
        case 5: do_send.template operator()<(Base) + 5>(k, v); break;                \
        default: throw std::logic_error("gradient_compressed: bad neighbor index");  \
      }

    /**
     * Root: S without coefficients (compute takes s from the compressed root), every
     * function continues along every axis, no neighbors.
     */
    auto primer_fn = [fns](const key_type& key, const cnode_type&) -> TASKTYPE {
      if (key.level() == 0) {
        const size_type N = fns->num_functions(key);
        node_type s(key, N);
        s.set_all_leaf(LeafStatus::Inner);
        MRA_GRAD_SENDS
        do_send.template operator()<0>(key, std::move(s));
        do_send.template operator()<1>(key, 0);
        do_send.template operator()<2>(key, GradientTags(N, (1u << NDIM) - 1));
        MRA_GRAD_FLUSH
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(in),
                                         ttg::edges(s_forward, flags_forward, tags_compute), name + "-primer");

    /**
     * Forward: S of the node to every task that reads it.
     */
    constexpr const int F_S_COMPUTE = 0, F_FLAGS_COMPUTE = 1, F_S_SPAWN = 2, F_NB = 3;
    auto forward_fn = [fns, nb_index, at_boundary, invalid_node](const key_type& key, const node_type& S,
                                                                  const int& flags) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      MRA_GRAD_SENDS
      do_send.template operator()<F_S_COMPUTE>(key, S);
      do_send.template operator()<F_FLAGS_COMPUTE>(key, flags);
      do_send.template operator()<F_S_SPAWN>(key, S);
      for (Dimension a = 0; a < NDIM; ++a) {
        for (int s : {-1, 1}) {
          if (at_boundary(key, a, s)) {
            /* beyond the boundary of the domain: our own compute gets an invalid node */
            MRA_GRAD_SEND_NB(F_NB, nb_index(a, s), key, invalid_node(key, N))
          } else if ((flags >> nb_index(a, s)) & 1) {
            /* an existing neighbor sees us on its opposite side */
            MRA_GRAD_SEND_NB(F_NB, nb_index(a, -s), key.neighbor(a, s), S)
          }
        }
      }
      MRA_GRAD_FLUSH
    };
    auto forward_tt = ttg::make_tt<Space>(std::move(forward_fn), ttg::edges(s_forward, flags_forward),
                                          ttg::edges(s_compute, flags_compute, s_spawn,
                                                     nb_e[0], nb_e[1], nb_e[2], nb_e[3], nb_e[4], nb_e[5]),
                                          name + "-forward", {"s", "flags"},
                                          {"s_compute", "flags_compute", "s_spawn",
                                           "nb0", "nb1", "nb2", "nb3", "nb4", "nb5"});

    /**
     * Compute: per axis and function the status of the derivative here, the derivatives
     * formed here, S of the children, and the decisions for the neighbors' spawn.
     */
    constexpr const int C_NOTIFY = NDIM, C_NB = C_NOTIFY + 1, C_S_CHILD = C_NB + NB, C_TAGS = C_S_CHILD + 1;
    /* bits of the summary compute sends to its notify task: the neighbor flags, whether the
     * node has children, and which children have no compressed node */
    constexpr const int NOTIFY_HAS_CHILDREN = 1 << NB, NOTIFY_PLACEHOLDER_SHIFT = NB + 1;
    auto compute_fn = [&db, &functiondata, fns, K, nb_index, at_boundary, children_on_side, has_coeffs,
                       enable_batching, deriv_pool, spawn_pool, max_functions](
                          const key_type& key, const node_type& S, const int& flags, const GradientTags& tags,
                          const cnode_type& cnode,
                          const node_type& nb0, const node_type& nb1, const node_type& nb2,
                          const node_type& nb3, const node_type& nb4, const node_type& nb5) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      if (cnode.is_ns()) throw std::runtime_error("gradient_compressed: input must be in standard compressed form");
      const std::array<const node_type*, NB> nb = {&nb0, &nb1, &nb2, &nb3, &nb4, &nb5};
      auto neighbor = [&](Dimension a, int s) -> const node_type& { return *nb[nb_index(a, s)]; };
      auto exists = [&](Dimension a, int s) { return (flags >> nb_index(a, s)) & 1; };

      /* Inner where the input continues here or is finer at a neighbor along the axis */
      GradientTags inner(N);
      std::array<std::vector<LeafStatus>, NDIM> status;
      std::array<bool, NDIM> in_tree{};
      for (Dimension a = 0; a < NDIM; ++a) status[a].assign(N, LeafStatus::Invalid);
      for (size_type j = 0; j < N; ++j) {
        for (Dimension a = 0; a < NDIM; ++a) {
          if (!((tags[j] >> a) & 1)) continue;
          in_tree[a] = true;
          bool is_inner = (S.leaf_status(j) == LeafStatus::Inner);
          for (int s : {-1, 1}) {
            is_inner = is_inner || (!at_boundary(key, a, s) && neighbor(a, s).leaf_status(j) == LeafStatus::Inner);
          }
          status[a][j] = is_inner ? LeafStatus::Inner : LeafStatus::Leaf;
          if (is_inner) inner[j] |= std::uint8_t(1u << a);
        }
      }
      const bool has_children = inner.any();

      /* the derivatives formed here */
      std::array<node_type, NDIM> result;
      std::array<size_type, NDIM> n_nonzero{};
      size_type total_nonzero = 0;
      for (Dimension a = 0; a < NDIM; ++a) {
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type j = 0; j < N; ++j) {
          if (status[a][j] != LeafStatus::Leaf) continue;
          bool nonzero = has_coeffs(S, j);
          for (int s : {-1, 1}) nonzero = nonzero || (!at_boundary(key, a, s) && has_coeffs(neighbor(a, s), j));
          if (nonzero) sparsity.set_nonzero(j);
        }
        n_nonzero[a] = sparsity.count_nonzero();
        total_nonzero += n_nonzero[a];
        result[a] = (n_nonzero[a] > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate) : node_type(key, N);
        for (size_type j = 0; j < N; ++j) result[a].set_leaf(j, status[a][j]);
      }

      /* S of the children: unfiltered where the input continues here, else passed on and
       * belonging one level further up than at this node */
      std::array<node_type, num_children> child_s;
      if (has_children) {
        for (size_type c = 0; c < num_children; ++c) {
          child_s[c] = node_type(key.child_at(c), SparsityInfo(N, SparsityInfo::InitType::AllNonZero), K,
                                 ttg::scope::Allocate);
          for (size_type j = 0; j < N; ++j) {
            if (S.leaf_status(j) == LeafStatus::Inner) {
              child_s[c].set_leaf(j, cnode.is_child_leaf(j, c) ? LeafStatus::Leaf : LeafStatus::Inner);
            } else {
              child_s[c].set_leaf(j, LeafStatus::Invalid);
              child_s[c].set_origin(j, S.origin(j) + 1);
            }
          }
        }
      }

      /* scratch, source keys and flags must outlive the kernels: destroying a buffer releases
       * its device memory right away, while the kernels launched below are still running.
       * One scratch buffer: the derivatives first, then the children. */
      const size_type deriv_tmp = derivative_tmp_size<NDIM>(K)*total_nonzero;
      const size_type spawn_tmp = has_children ? reconstruct_tmp_size<NDIM>(K)*N : 0;
      ttg::Buffer<T, DeviceAllocator<T>> tmp;
      ttg::Buffer<key_type, DeviceAllocator<key_type>> src_keys;
      ttg::Buffer<std::uint8_t, DeviceAllocator<std::uint8_t>> unfilter;
      if (total_nonzero > 0 || has_children) {
        const auto& operators = functiondata.get_operators();
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        const auto& hg = functiondata.get_hg();
        tmp = ttg::Buffer<T, DeviceAllocator<T>>(deriv_tmp + spawn_tmp, ttg::scope::Allocate);
        /* per axis and function: the nodes the coefficients at the left neighbor, here, and
         * at the right neighbor belong to; an ancestor where the input ended above, and a
         * coarser node (or its ancestor) where a neighbor does not exist */
        src_keys = ttg::Buffer<key_type, DeviceAllocator<key_type>>(3*NDIM*N);
        key_type* sk = src_keys.host_ptr();
        for (Dimension a = 0; a < NDIM; ++a) {
          for (size_type j = 0; j < N; ++j) {
            sk[3*(a*N + j) + 0] = neighbor(a, -1).origin_key(j);
            sk[3*(a*N + j) + 1] = S.origin_key(j);
            sk[3*(a*N + j) + 2] = neighbor(a, 1).origin_key(j);
          }
        }
        unfilter = ttg::Buffer<std::uint8_t, DeviceAllocator<std::uint8_t>>(N);
        for (size_type j = 0; j < N; ++j) unfilter.host_ptr()[j] = (S.leaf_status(j) == LeafStatus::Inner);
#ifndef MRA_ENABLE_HOST
        auto inputs = ttg::device::Input(db, operators.buffer(), phibar.buffer(), phi.buffer(), quad_x.buffer(),
                                         hg.buffer(), tmp, src_keys, unfilter);
        std::array<const void*, NB + 1> added{};
        std::size_t n_added = 0;
        auto add_once = [&](const node_type& n) {
          if (n.empty()) return;
          const void* b = &n.coeffs().buffer();
          if (std::find(added.begin(), added.begin() + n_added, b) != added.begin() + n_added) return;
          added[n_added++] = b;
          inputs.add(n.coeffs().buffer());
        };
        add_once(S);
        for (auto* n : nb) add_once(*n);
        for (auto& r : result) if (!r.empty()) inputs.add(r.coeffs().buffer());
        for (auto& c : child_s) if (!c.empty()) inputs.add(c.coeffs().buffer());
        if (!cnode.empty()) inputs.add(cnode.coeffs().buffer());
        co_await ttg::device::select(inputs);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto center_view = S.coeffs().current_view();
        T* tmp_ptr = tmp.current_device_ptr();
        const key_type* src_keys_ptr = src_keys.current_device_ptr();
        const std::uint8_t* unfilter_ptr = unfilter.current_device_ptr();
        auto cnode_view = cnode.coeffs().current_view();
        auto nb_views = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::array{(nb[Is]->coeffs().current_view())...};
        }(std::make_index_sequence<NB>{});
        auto result_views = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::array{(result[Is].coeffs().current_view())...};
        }(std::make_index_sequence<NDIM>{});
        auto child_views = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::array{(child_s[Is].coeffs().current_view())...};
        }(std::make_index_sequence<num_children>{});
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the leader issues the launches of every member (a task joins one batch only)
           * and pushes the sparsity of the outputs along with the batch */
          bool spawn_children = has_children;
          auto batch = co_await ttg::device::coop<key_type>(key, center_view, nb_views, result_views, result,
                                                            n_nonzero, src_keys_ptr, tmp_ptr, spawn_children,
                                                            cnode_view, child_views, child_s, unfilter_ptr, N);
          detail::submit_gradient_compute_batch_leader<T, NDIM>(batch, *deriv_pool, *spawn_pool, D,
                                                                operators.current_view(), phi.current_view(),
                                                                phibar.current_view(), quad_x.current_view(),
                                                                hg.current_view(), K,
                                                                FunctionData<T, NDIM>::BC_FREE,
                                                                FunctionData<T, NDIM>::BC_FREE, max_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          /* all axes in one launch */
          for (Dimension a = 0; a < NDIM; ++a) {
            if (n_nonzero[a] == 0) continue;
            auto sparseman = make_sparsity_manager(result[a]);
            sparseman.populate_device_sparsity();
          }
          submit_gradient_derivative_kernel<T, NDIM>(D, key, src_keys_ptr, nb_views, center_view, result_views,
                                            operators.current_view(), phi.current_view(), phibar.current_view(),
                                            quad_x.current_view(), tmp_ptr, N, n_nonzero, K,
                                            FunctionData<T, NDIM>::BC_FREE, FunctionData<T, NDIM>::BC_FREE,
                                            ttg::device::current_stream());
          if (has_children) {
            auto sparseman = make_sparsity_manager(child_s);
            sparseman.populate_device_sparsity();
            submit_gradient_spawn_kernel<T, NDIM>(key, N, K, cnode_view, center_view, hg.current_view(), child_views,
                                                  unfilter_ptr, tmp_ptr + deriv_tmp, ttg::device::current_stream());
          }
        }
      }

      MRA_GRAD_SENDS
      /* a node of the derivative along an axis exists where any function's tree includes it */
      if (in_tree[0]) do_send.template operator()<0>(key, std::move(result[0]));
      if (in_tree[1]) do_send.template operator()<1>(key, std::move(result[1]));
      if (in_tree[2]) do_send.template operator()<2>(key, std::move(result[2]));
      /* the small messages go through the notify task: a device task holds a data copy of
       * every value it sends until it completes, and PaRSEC limits their number per task
       * (MAX_PARAM_COUNT, including the inputs); this keeps compute at 23 */
      int summary = flags | (has_children ? NOTIFY_HAS_CHILDREN : 0);
      if (has_children) {
        for (size_type c = 0; c < num_children; ++c) {
          /* a child without a compressed node, where the input has no children there */
          bool any_inner = false;
          for (size_type j = 0; j < N && !any_inner; ++j) any_inner = (child_s[c].leaf_status(j) == LeafStatus::Inner);
          if (!any_inner) summary |= 1 << (NOTIFY_PLACEHOLDER_SHIFT + c);
        }
      }
      do_send.template operator()<C_NOTIFY>(key, summary);
      for (Dimension a = 0; a < NDIM; ++a) {
        for (int s : {-1, 1}) {
          /* no neighbor here: our children take the coarser one we got in its place */
          if (has_children && !exists(a, s) && !at_boundary(key, a, s)) {
            for (const auto& child : children_on_side(key, a, s)) {
              MRA_GRAD_SEND_NB(C_NB, nb_index(a, s), child, neighbor(a, s))
            }
          }
        }
      }
      if (has_children) {
        std::vector<key_type> child_keys;
        for (size_type c = 0; c < num_children; ++c) child_keys.push_back(key.child_at(c));
        for (size_type c = 0; c < num_children; ++c) {
          do_send.template operator()<C_S_CHILD>(child_keys[c], std::move(child_s[c]));
        }
#ifndef MRA_ENABLE_HOST
        sends.push_back(ttg::device::broadcast<C_TAGS>(child_keys, std::move(inner)));
#else  // MRA_ENABLE_HOST
        ttg::broadcast<C_TAGS>(child_keys, std::move(inner));
#endif // MRA_ENABLE_HOST
      }
      MRA_GRAD_FLUSH
    };
    auto compute_tt = ttg::make_tt<Space>(std::move(compute_fn),
                                          ttg::edges(s_compute, flags_compute, tags_compute,
                                                     ttg::fuse(in, cnode_balance),
                                                     nb_e[0], nb_e[1], nb_e[2], nb_e[3], nb_e[4], nb_e[5]),
                                          ttg::edges(out[0], out[1], out[2], notify_e,
                                                     nb_e[0], nb_e[1], nb_e[2], nb_e[3], nb_e[4], nb_e[5],
                                                     s_forward, tags_compute),
                                          name, {"s", "flags", "tags", "cnode",
                                                 "nb0", "nb1", "nb2", "nb3", "nb4", "nb5"},
                                          {"out0", "out1", "out2", "notify",
                                           "nb0", "nb1", "nb2", "nb3", "nb4", "nb5",
                                           "s_child", "tags_child"});

    /**
     * Notify (no device work): from compute's summary, tells the neighbors' spawn (or, without a
     * neighbor, our own) whether this node has children, tells our own spawn, and sends
     * the children without a compressed node an empty one.
     */
    constexpr const int N_CHILDREN = 0, N_DEC = 1, N_CNODE = N_DEC + NB;
    auto notify_fn = [fns, nb_index](const key_type& key, const int& summary) -> TASKTYPE {
      const bool has_children = summary & NOTIFY_HAS_CHILDREN;
      const int dec = has_children ? 1 : 0;
      MRA_GRAD_SENDS
      do_send.template operator()<N_CHILDREN>(key, dec);
      for (Dimension a = 0; a < NDIM; ++a) {
        for (int s : {-1, 1}) {
          /* the neighbor's spawn hears it on the opposite side; without a neighbor, our own */
          if ((summary >> nb_index(a, s)) & 1) {
            MRA_GRAD_SEND_NB(N_DEC, nb_index(a, -s), key.neighbor(a, s), dec)
          } else {
            MRA_GRAD_SEND_NB(N_DEC, nb_index(a, s), key, 0)
          }
        }
      }
      for (size_type c = 0; c < num_children; ++c) {
        if (!((summary >> (NOTIFY_PLACEHOLDER_SHIFT + c)) & 1)) continue;
        const key_type child = key.child_at(c);
        do_send.template operator()<N_CNODE>(child, cnode_type(child, fns->num_functions(child)));
      }
      MRA_GRAD_FLUSH
    };
    auto notify_tt = ttg::make_tt<Space>(std::move(notify_fn), ttg::edges(notify_e),
                                         ttg::edges(children_spawn,
                                                    dec_e[0], dec_e[1], dec_e[2], dec_e[3], dec_e[4], dec_e[5],
                                                    cnode_balance),
                                         name + "-notify", {"summary"},
                                         {"has_children", "dec0", "dec1", "dec2", "dec3", "dec4", "dec5", "cnode"});

    /**
     * Spawn (no device work): the flags of the children, once the neighbors have decided; or,
     * without children, this node's S to the children of the neighbors that have some.
     */
    constexpr const int S_FLAGS = 0, S_NB = 1;
    auto spawn_fn = [nb_index, at_boundary, children_on_side](
                        const key_type& key, const node_type& S, const int& has_children,
                        const int& dec0, const int& dec1, const int& dec2,
                        const int& dec3, const int& dec4, const int& dec5) -> TASKTYPE {
      const std::array<int, NB> decs = {dec0, dec1, dec2, dec3, dec4, dec5};
      auto neighbor_has_children = [&](Dimension a, int s) { return decs[nb_index(a, s)] != 0; };
      MRA_GRAD_SENDS
      if (has_children) {
        /* the siblings exist, and across our boundary the children of our neighbors that
         * have some */
        for (size_type c = 0; c < num_children; ++c) {
          int flags = 0;
          for (Dimension a = 0; a < NDIM; ++a) {
            const int across = ((c >> a) & 1) ? 1 : -1; // the side that faces away from the siblings
            flags |= 1 << nb_index(a, -across);
            if (!at_boundary(key, a, across) && neighbor_has_children(a, across)) flags |= 1 << nb_index(a, across);
          }
          do_send.template operator()<S_FLAGS>(key.child_at(c), flags);
        }
      } else {
        for (Dimension a = 0; a < NDIM; ++a) {
          for (int s : {-1, 1}) {
            if (!neighbor_has_children(a, s)) continue;
            for (const auto& own_side : children_on_side(key, a, s)) {
              MRA_GRAD_SEND_NB(S_NB, nb_index(a, -s), own_side.neighbor(a, s), S)
            }
          }
        }
      }
      MRA_GRAD_FLUSH
    };
    auto spawn_tt = ttg::make_tt<Space>(std::move(spawn_fn),
                                        ttg::edges(s_spawn, children_spawn,
                                                   dec_e[0], dec_e[1], dec_e[2], dec_e[3], dec_e[4], dec_e[5]),
                                        ttg::edges(flags_forward,
                                                   nb_e[0], nb_e[1], nb_e[2], nb_e[3], nb_e[4], nb_e[5]),
                                        name + "-spawn",
                                        {"s", "has_children", "dec0", "dec1", "dec2", "dec3", "dec4", "dec5"},
                                        {"flags", "nb0", "nb1", "nb2", "nb3", "nb4", "nb5"});
#undef MRA_GRAD_SENDS
#undef MRA_GRAD_FLUSH
#undef MRA_GRAD_SEND_NB

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      primer_tt->set_keymap(procmap);
      forward_tt->set_keymap(procmap);
      compute_tt->set_keymap(procmap);
      notify_tt->set_keymap(procmap);
      spawn_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      compute_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      compute_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(compute_tt->template in<3>());
    auto outs = std::make_tuple(compute_tt->template out<0>(), compute_tt->template out<1>(),
                                compute_tt->template out<2>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(5);
    ops[0] = std::move(primer_tt);
    ops[1] = std::move(forward_tt);
    ops[2] = std::move(compute_tt);
    ops[3] = std::move(notify_tt);
    ops[4] = std::move(spawn_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_GRADIENT_COMPRESSED_H
