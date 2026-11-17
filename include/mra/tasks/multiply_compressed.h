#ifndef MRA_TASKS_MULTIPLY_COMPRESSED_H
#define MRA_TASKS_MULTIPLY_COMPRESSED_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <optional>
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
#include "mra/tensor/combinednode.h"
#include "mra/kernels/multiply_compressed.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

/**
 * Products of functions given in compressed form, without reconstructing them
 * first (Algorithm B of "Compressed-form multiplication in MADNESS").
 *
 * make_reconstruct followed by make_multiply/make_square visits every node of
 * the trees three times: reconstruct unfilters [s, d] into the children and
 * sends every reconstructed node, then the product task multiplies at the
 * leaves. Here a single descent over the compressed trees does both: the task
 * at a node unfilters [s, d] of each factor and, in the same kernel, multiplies
 * at every child where the product ends; only children where the product
 * continues get a task, which receives the scaling coefficients of both
 * factors. The scaling coefficients of the inputs are never stored, and the
 * inputs are only read.
 *
 * The output is the product in reconstructed form, a node at every key of its
 * tree with the same leaf status as make_multiply_broadcast and make_square
 * produce; feed it into make_compress for a compressed product.
 *
 * The tree of the product is the union of the trees of the factors: where one
 * factor ends and the other continues, the scaling coefficients of the one that
 * ended are unfiltered with d = 0, which represents the same polynomial exactly.
 * No screening is applied, so the result is today's leaf product (tol = 0).
 *
 * Inputs must be standard compressed form (not non-standard) with a compressed
 * node at every key where some function of the set has children, as produced
 * by make_compress.
 */

namespace mra {

  namespace detail {

    /** one node the task at a key writes: sparsity and leaf status of each function; no status, no node */
    struct MulCompressedNodeInfo {
      SparsityInfo sparsity;
      std::vector<LeafStatus> status;
      bool empty() const { return status.empty(); }
    };

    /**
     * What the task at a key produces, as host metadata only: the product node of the
     * key, for every child either its product node (no task there) or the scaling
     * coefficients for its task, and the work items of the kernel.
     */
    template <Dimension NDIM>
    struct MulCompressedLayout {
      static constexpr size_type num_children = Key<NDIM>::num_children();
      MulCompressedNodeInfo result;
      std::array<MulCompressedNodeInfo, num_children> out;
      std::array<bool, num_children> task{};      // the child has a task
      std::array<bool, num_children> a_node_at{}; // the first factor has a compressed node at the child
      std::array<bool, num_children> b_node_at{};
      std::vector<MulCompressedWork<NDIM>> work;
    };

    /**
     * MulCompressedLayout with its nodes allocated.
     */
    template <typename T, Dimension NDIM>
    struct MulCompressedPlan {
      using node_type = FunctionsReconstructedNode<T, NDIM>;
      static constexpr size_type num_children = Key<NDIM>::num_children();
      node_type result;
      std::array<node_type, num_children> out;
      std::array<bool, num_children> task{};      // the child has a task
      std::array<bool, num_children> a_node_at{}; // the first factor has a compressed node at the child
      std::array<bool, num_children> b_node_at{};
      std::vector<MulCompressedWork<NDIM>> work;
    };

    /**
     * s_status(m) is, for functions [0, NA) of the first factor and (unless b_node is
     * null, a square) [NA, NA+N) of the second, the status of the function in its own
     * tree at the key: Inner where it has children, Leaf where its tree ended here or
     * above, Invalid where it is not needed. The scaling coefficients passed down to
     * the key (s_in of the kernel) are laid out the same way. NA is 1 (one function
     * times N) or N (elementwise).
     */
    template <typename T, Dimension NDIM, typename StatusFn>
    MulCompressedLayout<NDIM> plan_multiply_compressed_layout(
      size_type NA,
      size_type N,
      const FunctionsCompressedNode<T, NDIM>& a_node,
      const FunctionsCompressedNode<T, NDIM>* b_node,
      StatusFn&& s_status)
    {
      constexpr size_type num_children = Key<NDIM>::num_children();
      const bool square = (b_node == nullptr);
      MulCompressedLayout<NDIM> plan;

      auto ia_of = [&](size_type j) { return (NA == 1) ? size_type(0) : j; };
      auto a_status = [&](size_type i) { return s_status(i); };
      auto b_status = [&](size_type j) { return square ? s_status(j) : s_status(NA + j); };
      /* a function continues below child c if it has children here and the node says so */
      auto a_inner_at = [&](size_type i, size_type c) {
        return a_status(i) == LeafStatus::Inner && !a_node.is_child_leaf(i, c);
      };
      auto b_inner_at = [&](size_type j, size_type c) {
        return square ? a_inner_at(j, c) : (b_status(j) == LeafStatus::Inner && !b_node->is_child_leaf(j, c));
      };

      /* status of the product here: Inner where either factor continues */
      std::vector<LeafStatus> status(N, LeafStatus::Invalid);
      std::vector<char> a_needed(NA, 0);
      SparsityInfo result_sparsity(N, SparsityInfo::InitType::AllZero);
      bool any_inner = false;
      for (size_type j = 0; j < N; ++j) {
        const size_type ia = ia_of(j);
        if (a_status(ia) == LeafStatus::Invalid || b_status(j) == LeafStatus::Invalid) continue;
        if (a_status(ia) == LeafStatus::Inner || b_status(j) == LeafStatus::Inner) {
          status[j] = LeafStatus::Inner;
          a_needed[ia] = 1;
          any_inner = true;
        } else {
          status[j] = LeafStatus::Leaf;
          result_sparsity.set_nonzero(j);
        }
      }
      plan.result.sparsity = std::move(result_sparsity);
      plan.result.status = status;

      if (any_inner) {
        /* a factor sends a compressed node to a child iff some function continues there,
         * which also makes it a node of the product with a task */
        for (size_type c = 0; c < num_children; ++c) {
          for (size_type i = 0; i < NA && !plan.a_node_at[c]; ++i) plan.a_node_at[c] = a_inner_at(i, c);
          if (!square) {
            for (size_type j = 0; j < N && !plan.b_node_at[c]; ++j) plan.b_node_at[c] = b_inner_at(j, c);
          }
          plan.task[c] = plan.a_node_at[c] || plan.b_node_at[c];
        }

        for (size_type c = 0; c < num_children; ++c) {
          auto& out = plan.out[c];
          if (!plan.task[c]) {
            /* the product ends at the child for every function that continues here */
            out.sparsity = SparsityInfo(N, SparsityInfo::InitType::AllZero);
            out.status.assign(N, LeafStatus::Invalid);
            for (size_type j = 0; j < N; ++j) {
              if (status[j] != LeafStatus::Inner) continue;
              out.sparsity.set_nonzero(j);
              out.status[j] = LeafStatus::Leaf;
            }
          } else {
            /* scaling coefficients of both factors for the task of the child */
            const size_type M = square ? N : NA + N;
            std::vector<LeafStatus> child_status(M, LeafStatus::Invalid);
            SparsityInfo sparsity(M, SparsityInfo::InitType::AllZero);
            for (size_type i = 0; i < NA; ++i) {
              if (!a_needed[i]) continue;
              child_status[i] = a_inner_at(i, c) ? LeafStatus::Inner : LeafStatus::Leaf;
              sparsity.set_nonzero(i);
            }
            if (!square) {
              for (size_type j = 0; j < N; ++j) {
                if (status[j] != LeafStatus::Inner) continue;
                child_status[NA + j] = b_inner_at(j, c) ? LeafStatus::Inner : LeafStatus::Leaf;
                sparsity.set_nonzero(NA + j);
              }
            }
            out.sparsity = std::move(sparsity);
            out.status = std::move(child_status);
          }
        }
      }

      /* one work item per function that is a leaf here (multiply) or continues (unfilter) */
      std::vector<char> a_written(NA, 0);
      for (size_type j = 0; j < N; ++j) {
        if (status[j] == LeafStatus::Invalid) continue;
        const size_type ia = ia_of(j);
        MulCompressedWork<NDIM> w{};
        w.j = j;
        w.ia = ia;
        if (status[j] == LeafStatus::Leaf) {
          w.type = 0;
        } else {
          w.type = 1;
          w.a_inner = (a_status(ia) == LeafStatus::Inner);
          w.b_inner = (b_status(j) == LeafStatus::Inner);
          w.write_a = !a_written[ia];
          a_written[ia] = 1;
          for (size_type c = 0; c < num_children; ++c) w.child[c] = plan.task[c] ? MULC_PASS : MULC_LEAF;
        }
        plan.work.push_back(w);
      }
      return plan;
    }

    /** a reconstructed node at `key` with the sparsity and leaf statuses of `info` */
    template <typename T, Dimension NDIM>
    FunctionsReconstructedNode<T, NDIM> make_multiply_compressed_node(
      const Key<NDIM>& key, const MulCompressedNodeInfo& info, size_type K)
    {
      FunctionsReconstructedNode<T, NDIM> node(key, info.sparsity, K, ttg::scope::Allocate);
      for (size_type m = 0; m < info.status.size(); ++m) node.set_leaf(m, info.status[m]);
      return node;
    }

    /**
     * The plan of the task at `key`, its nodes allocated. s_in holds the statuses
     * of plan_multiply_compressed_layout. The product node of `key` is allocated even
     * without leaves, as make_reconstruct does, so its sparsity can be pushed.
     */
    template <typename T, Dimension NDIM>
    MulCompressedPlan<T, NDIM> plan_multiply_compressed(
      const Key<NDIM>& key,
      size_type K,
      size_type NA,
      size_type N,
      const FunctionsCompressedNode<T, NDIM>& a_node,
      const FunctionsCompressedNode<T, NDIM>* b_node,
      const FunctionsReconstructedNode<T, NDIM>& s_in)
    {
      auto layout = plan_multiply_compressed_layout<T, NDIM>(NA, N, a_node, b_node,
                                                             [&](size_type m) { return s_in.leaf_status(m); });
      MulCompressedPlan<T, NDIM> plan;
      plan.result = make_multiply_compressed_node<T, NDIM>(key, layout.result, K);
      for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
        if (!layout.out[c].empty()) {
          plan.out[c] = make_multiply_compressed_node<T, NDIM>(key.child_at(c), layout.out[c], K);
        }
      }
      plan.task = layout.task;
      plan.a_node_at = layout.a_node_at;
      plan.b_node_at = layout.b_node_at;
      plan.work = std::move(layout.work);
      return plan;
    }

    /** the scaling coefficients at the root: none, every function continues */
    template <typename T, Dimension NDIM>
    FunctionsReconstructedNode<T, NDIM> multiply_compressed_root_s(const Key<NDIM>& key, size_type M) {
      FunctionsReconstructedNode<T, NDIM> s(key, M);
      s.set_all_leaf(LeafStatus::Inner);
      return s;
    }

    inline std::atomic<int>& multiply_compressed_levels_state() {
      static std::atomic<int> value{[] {
        const char* env = std::getenv("MRA_TTG_MULTIPLY_LEVELS");
        return (env != nullptr && std::atoi(env) == 2) ? 2 : 1;
      }()};
      return value;
    }

  } // namespace detail

  /**
   * Number of tree levels each task of make_multiply_compressed handles, 1 or 2 (see
   * make_multiply_compressed_2level). Read once when a graph is built, like the batch
   * size; the default is MRA_TTG_MULTIPLY_LEVELS, or 1 if unset.
   */
  inline void set_multiply_compressed_levels(int levels) {
    detail::multiply_compressed_levels_state().store(levels == 2 ? 2 : 1, std::memory_order_relaxed);
  }

  inline int get_multiply_compressed_levels() {
    return detail::multiply_compressed_levels_state().load(std::memory_order_relaxed);
  }

  /**
   * make_multiply_compressed with tasks only at even levels: the task at a key unfilters
   * both factors at the key into its children (step 1) and, for every child where the
   * product continues, again into the grandchildren (step 2), multiplying wherever the
   * product ends. Half the levels on the critical path, and about 8x fewer kernel tasks.
   *
   * - The routers send each compressed node to the task that reads it: a node at an even
   *   level to the task at its key, a node at an odd level to the task at its parent as
   *   the node of its child. A task needs all 18 nodes, so the router of a node at an even
   *   level sends empty nodes for the children where the factor has none, and where a
   *   factor has no node at a grandchild with a task, the balance task sends empty nodes
   *   for the grandchild and its children.
   * - The scaling coefficients at the grandchildren with a task travel in one
   *   CombinedReconstructedNode, broadcast to all of them; the task at a key reads its slot.
   * - The product nodes at the children and the leaves at the grandchildren are written
   *   to a second CombinedReconstructedNode, the packed output, broadcast to one emit task
   *   per child, which copies them out into nodes and sends them on. The product node at
   *   the key is sent directly. The scaling coefficients at the children between the two
   *   steps are in a third, which stays with the task.
   *
   * Both steps run multiply_compressed_kernel, one launch for step 1 and one per child
   * for step 2, on the same stream. Batched, a batch runs two launches of
   * multiply_compressed_kernel_batched: step 1 of every task, then step 2 of every
   * (task, child). NDIM=3 only: the task takes the 8 children of each factor as inputs.
   */
  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_multiply_compressed_2level(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name = "multiply_compressed",
    ProcMap procmap = {},
    DeviceMap devicemap = {});

  /**
   * make_multiply_compressed_2level without unpacking: `out` receives only the product
   * nodes at even levels, and `packed` the packed output of each task, keyed by the task's
   * key, for make_compress_combined.
   */
  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_multiply_compressed_combined(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    ttg::Edge<mra::Key<NDIM>, mra::CombinedReconstructedNode<T, NDIM>> packed,
    const std::string& name = "multiply_compressed",
    ProcMap procmap = {},
    DeviceMap devicemap = {});

  namespace detail {

  /** both of the above; without packed_out, the emit task unpacks */
  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap, typename DeviceMap>
  auto make_multiply_compressed_2level_impl(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    std::optional<ttg::Edge<mra::Key<NDIM>, mra::CombinedReconstructedNode<T, NDIM>>> packed_out,
    const std::string& name,
    ProcMap procmap,
    DeviceMap devicemap)
  {
    static_assert(NDIM == 3, "make_multiply_compressed_2level: NDIM=3 only");
    using key_type = mra::Key<NDIM>;
    using cnode_type = mra::FunctionsCompressedNode<T, NDIM>;
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;
    using combined_type = mra::CombinedReconstructedNode<T, NDIM>;
    using view_type = typename combined_type::view_type;
    using work_type = detail::MulCompressedWork<NDIM>;
    using launch_type = detail::MulCompressedLaunch<T, NDIM>;
    /* static: the lambdas below use them without capturing them */
    static constexpr size_type num_children = key_type::num_children();
    static constexpr size_type num_terminals = num_children + 1; // per factor: the node and its children
    constexpr const int OUT = 0, PACKED = 1, S = 2, BALANCE = 3;

    /* per factor, to terminal 0 the node at the task's key, to terminal 1+c the node at child c */
    std::array<ttg::Edge<key_type, cnode_type>, num_terminals> route_a, route_b, balance_a, balance_b;
    ttg::Edge<key_type, combined_type> s_e(name + "-s"), primer_e(name + "-primer");
    const bool combined_out = packed_out.has_value();
    ttg::Edge<key_type, combined_type> packed_e = combined_out ? *packed_out
                                                               : ttg::Edge<key_type, combined_type>(name + "-packed");
    /* per grandchild (index num_children*c + g): bit 0 a has no node there, bit 1 b has none */
    ttg::Edge<key_type, std::vector<std::uint8_t>> balance_e(name + "-balance");

    /* batching as in make_multiply_compressed, sharing its pools */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::MulCompressedBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = detail::shared_pool_registry<detail::MulCompressedBatchArg<T, NDIM>>(max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    auto check_sizes = [name](size_type NA, size_type N) {
      if (NA != N && NA != 1) {
        throw std::runtime_error(name + ": the first factor must have one function or as many as the second");
      }
    };

    auto make_router =[&](const auto& fns, ttg::Edge<key_type, cnode_type> in,
                           std::array<ttg::Edge<key_type, cnode_type>, num_terminals>& outs,
                           const std::string& router_name) {
      auto route_fn = [fns](const key_type& key, const cnode_type& node) -> TASKTYPE {
#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_to = [&]<std::size_t... I>(std::index_sequence<I...>, std::size_t terminal, const key_type& to,
                                             auto&& value) {
          ((I == terminal ? sends.push_back(ttg::device::send<I>(to, std::forward<decltype(value)>(value))) : void()), ...);
        };
#else  // MRA_ENABLE_HOST
        auto send_to = [&]<std::size_t... I>(std::index_sequence<I...>, std::size_t terminal, const key_type& to,
                                             auto&& value) {
          ((I == terminal ? ttg::send<I>(to, std::forward<decltype(value)>(value)) : void()), ...);
        };
#endif // MRA_ENABLE_HOST
        constexpr auto terminals = std::make_index_sequence<num_terminals>{};
        if (key.level() % 2 == 1) {
          send_to(terminals, 1 + key.childindex(), key.parent(), node);
        } else {
          send_to(terminals, 0, key, node);
          for (size_type c = 0; c < num_children; ++c) {
            bool has_node = false;
            for (size_type i = 0; i < node.count() && !has_node; ++i) has_node = !node.is_child_leaf(i, c);
            if (!has_node) {
              const key_type child = key.child_at(c);
              send_to(terminals, 1 + c, key, cnode_type(child, fns->num_functions(child)));
            }
          }
        }
#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      };
      return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return ttg::make_tt<Space>(std::move(route_fn), ttg::edges(in), ttg::edges(outs[I]...), router_name);
      }(std::make_index_sequence<num_terminals>{});
    };
    auto route_a_tt = make_router(fnsA, a, route_a, name + "-route_a");
    auto route_b_tt = make_router(fnsB, b, route_b, name + "-route_b");

    auto primer_fn = [fnsA, fnsB, K, check_sizes](const key_type& key, const cnode_type&) -> TASKTYPE {
      if (key.level() == 0) {
        const size_type NA = fnsA->num_functions(key), N = fnsB->num_functions(key);
        check_sizes(NA, N);
        /* the scaling coefficients at the root: none, every function continues */
        combined_type s(key, K);
        s.add_slot(key, SparsityInfo(NA + N, SparsityInfo::InitType::AllZero),
                   std::vector<LeafStatus>(NA + N, LeafStatus::Inner));
        s.allocate(ttg::scope::SyncIn);
        s.populate_device_sparsity(ttg::device::Device::host());
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, std::move(s));
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, std::move(s));
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(a), ttg::edges(primer_e), name + "-primer");

    auto multiply_fn = [&db, &functiondata, fnsA, fnsB, K, name, check_sizes, enable_batching, batch_pool,
                        combined_out](
                          const key_type& key,
                          const cnode_type& anode, const cnode_type& a0, const cnode_type& a1, const cnode_type& a2,
                          const cnode_type& a3, const cnode_type& a4, const cnode_type& a5, const cnode_type& a6,
                          const cnode_type& a7,
                          const cnode_type& bnode, const cnode_type& b0, const cnode_type& b1, const cnode_type& b2,
                          const cnode_type& b3, const cnode_type& b4, const cnode_type& b5, const cnode_type& b6,
                          const cnode_type& b7,
                          const combined_type& s_in) -> TASKTYPE {
      const size_type NA = fnsA->num_functions(key), N = fnsB->num_functions(key);
      check_sizes(NA, N);
      const std::array<const cnode_type*, num_children> a_child = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7};
      const std::array<const cnode_type*, num_children> b_child = {&b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7};
      if (anode.is_ns() || bnode.is_ns()) {
        throw std::runtime_error(name + ": inputs must be in standard compressed form");
      }
      const int s_slot = s_in.find_slot(key);
      if (s_slot < 0) throw std::logic_error(name + ": no scaling coefficients for the key");

      /* step 1 at the key, step 2 at every child with a task */
      auto layout = detail::plan_multiply_compressed_layout<T, NDIM>(
                      NA, N, anode, &bnode, [&](size_type m) { return s_in.slot_status(s_slot, m); });
      std::array<detail::MulCompressedLayout<NDIM>, num_children> child_layout;
      for (size_type c = 0; c < num_children; ++c) {
        if (!layout.task[c]) continue;
        child_layout[c] = detail::plan_multiply_compressed_layout<T, NDIM>(
                            NA, N, *a_child[c], b_child[c], [&](size_type m) { return layout.out[c].status[m]; });
      }

      /* the outputs: product nodes in packed, the scaling coefficients for the
       * grandchildren's tasks in down; the scaling coefficients at the children between
       * the steps in scratch, which only this task uses */
      node_type result = detail::make_multiply_compressed_node<T, NDIM>(key, layout.result, K);
      combined_type packed(key, K), down(key, K), scratch(key, K);
      std::array<int, num_children> out_slot, child_result_slot;  // in scratch (a child with a task) or packed
      std::array<std::array<int, num_children>, num_children> grand_slot; // in packed or down
      std::array<std::array<bool, num_children>, num_children> grand_down{};
      out_slot.fill(-1);
      child_result_slot.fill(-1);
      for (auto& g : grand_slot) g.fill(-1);
      for (size_type c = 0; c < num_children; ++c) {
        if (layout.out[c].empty()) continue;
        const key_type child = key.child_at(c);
        out_slot[c] = (layout.task[c] ? scratch : packed).add_slot(child, layout.out[c].sparsity,
                                                                   layout.out[c].status);
        if (!layout.task[c]) continue;
        const auto& cl = child_layout[c];
        child_result_slot[c] = packed.add_slot(child, cl.result.sparsity, cl.result.status);
        for (size_type g = 0; g < num_children; ++g) {
          if (cl.out[g].empty()) continue;
          if (cl.task[g]) {
            grand_slot[c][g] = down.add_slot(child.child_at(g), cl.out[g].sparsity, cl.out[g].status);
            grand_down[c][g] = true;
          } else {
            grand_slot[c][g] = packed.add_slot(child.child_at(g), cl.out[g].sparsity, cl.out[g].status);
          }
        }
      }
      if (packed.num_slots() > 0) packed.allocate();
      if (down.num_slots() > 0) down.allocate();
      if (scratch.num_slots() > 0) scratch.allocate();

      /* the work items of step 1, then those of step 2 per child */
      const size_type n_items = layout.work.size();
      std::vector<work_type> all_work = layout.work;
      std::array<size_type, num_children> work_offset{};
      size_type max_child_items = 0, sum_child_items = 0;
      for (size_type c = 0; c < num_children; ++c) {
        if (!layout.task[c]) continue;
        work_offset[c] = all_work.size();
        all_work.insert(all_work.end(), child_layout[c].work.begin(), child_layout[c].work.end());
        max_child_items = std::max<size_type>(max_child_items, child_layout[c].work.size());
        sum_child_items += child_layout[c].work.size();
      }

      /* scratch and work items must outlive the kernels. Unbatched, the launches run one
       * after the other on the task's stream and share tmp; batched, the launches of step 2
       * run in one kernel and each child has its own part of tmp. Batched, the leader
       * stages the work items (as in make_multiply_compressed): no work buffer. */
#ifndef MRA_ENABLE_HOST
      const bool batched = enable_batching;
#else  // MRA_ENABLE_HOST
      constexpr bool batched = false;
#endif // MRA_ENABLE_HOST
      const size_type tmp_items = std::max(n_items, batched ? sum_child_items : max_child_items);
      ttg::Buffer<T, DeviceAllocator<T>> tmp;
      ttg::Buffer<work_type, DeviceAllocator<work_type>> work;
      if (n_items > 0) {
        const auto& hg = functiondata.get_hg();
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        tmp = ttg::Buffer<T, DeviceAllocator<T>>(multiply_compressed_tmp_size<NDIM>(K)*tmp_items, ttg::scope::Allocate);
        if (!batched) {
          work = ttg::Buffer<work_type, DeviceAllocator<work_type>>(all_work.size());
          std::copy(all_work.begin(), all_work.end(), work.host_ptr());
        }
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, hg.buffer(), phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp);
        if (!batched) input.add(work);
        for (const cnode_type* n : {&anode, &bnode}) if (!n->empty()) input.add(n->coeffs().buffer());
        for (size_type c = 0; c < num_children; ++c) {
          for (const cnode_type* n : {a_child[c], b_child[c]}) if (!n->empty()) input.add(n->coeffs().buffer());
        }
        input.add(s_in.buffer());
        input.add(result.coeffs().buffer());
        if (!packed.empty()) input.add(packed.buffer());
        if (!down.empty()) input.add(down.buffer());
        if (!scratch.empty()) input.add(scratch.buffer());
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        node_type no_node; // the view of a child without output
        const view_type none = no_node.coeffs().current_view();
        auto views = [&]<std::size_t... I>(std::index_sequence<I...>, auto&& view_at) {
          return std::array<view_type, num_children>{view_at(I)...};
        };
        constexpr auto children = std::make_index_sequence<num_children>{};
        T* tmp_ptr = tmp.current_device_ptr();
        const size_type tmp_size = multiply_compressed_tmp_size<NDIM>(K);

        /* step 1 at the key, then step 2 at every child with work items */
        std::vector<launch_type> launches;
        launches.push_back(launch_type{key, anode.coeffs().current_view(), bnode.coeffs().current_view(),
                                       s_in.slot_view(s_slot), result.coeffs().current_view(),
                                       views(children, [&](size_type c) {
                                         if (out_slot[c] < 0) return none;
                                         return layout.task[c] ? scratch.slot_view(out_slot[c])
                                                               : packed.slot_view(out_slot[c]);
                                       }),
                                       0, n_items, tmp_ptr});
        size_type child_tmp = 0;
        for (size_type c = 0; c < num_children; ++c) {
          const size_type n_child_items = layout.task[c] ? child_layout[c].work.size() : 0;
          if (n_child_items == 0) continue;
          launches.push_back(launch_type{key.child_at(c), a_child[c]->coeffs().current_view(),
                                         b_child[c]->coeffs().current_view(), scratch.slot_view(out_slot[c]),
                                         packed.slot_view(child_result_slot[c]),
                                         views(children, [&](size_type g) {
                                           const int s = grand_slot[c][g];
                                           if (s < 0) return none;
                                           return grand_down[c][g] ? down.slot_view(s) : packed.slot_view(s);
                                         }),
                                         work_offset[c], n_child_items, tmp_ptr + child_tmp*tmp_size});
          if (batched) child_tmp += n_child_items;
        }

#ifndef MRA_ENABLE_HOST
        if (batched) {
          /* the sparsity of the outputs is pushed by the leader along with the batch */
          const auto device = ttg::device::current_device();
          detail::MulCompressed2BatchMember<T, NDIM> member{std::move(launches), all_work.data(), NA, {}};
          const size_type n_result = result.coeffs().dim(0);
          std::vector<detail::SparsityState> result_sparsity(n_result);
          detail::sparsity_to_bytes(result.coeffs().sparsity(), result_sparsity.data(), n_result);
          member.sparsity.push_back({const_cast<T*>(result.coeffs().buffer().device_ptr_on(device)),
                                     result_sparsity.data(), n_result});
          for (combined_type* node : {&packed, &down, &scratch}) {
            if (node->empty()) continue;
            T* base = const_cast<T*>(node->buffer().device_ptr_on(device));
            for (size_type s = 0; s < node->num_slots(); ++s) {
              member.sparsity.push_back({base + node->slot_offset(s), node->slot_sparsity_data(s),
                                         node->slot_count(s)});
            }
          }
          auto batch = co_await ttg::device::coop<key_type>(key, member);
          detail::submit_multiply_compressed_2level_batch_leader<T, NDIM>(batch, *batch_pool, D, hg.current_view(),
                                                                          phibar.current_view(), phi.current_view(),
                                                                          quad_x.current_view(), K);
        } else
#endif // MRA_ENABLE_HOST
        {
          {
            auto sparseman = make_sparsity_manager(result);
            sparseman.populate_device_sparsity();
          }
          if (!packed.empty()) packed.populate_device_sparsity();
          if (!down.empty()) down.populate_device_sparsity();
          if (!scratch.empty()) scratch.populate_device_sparsity();
          const work_type* work_ptr = work.current_device_ptr();
          for (auto& l : launches) {
            submit_multiply_compressed_kernel<T, NDIM>(D, l.key, l.a_node, l.b_node, l.s_in, l.result, l.out_arr,
                                                       hg.current_view(), phibar.current_view(), phi.current_view(),
                                                       quad_x.current_view(), work_ptr + l.work, NA, false,
                                                       l.n_items, K, l.tmp, ttg::device::current_stream());
          }
        }
      }

      /* the balance task sends empty nodes where a factor has none at a grandchild with a task */
      std::vector<std::uint8_t> balance;
      std::vector<key_type> emit_keys, down_keys;
      if (combined_out) {
        if (packed.num_slots() > 0) emit_keys.push_back(key); // packed goes on whole, keyed by the task
      }
      for (size_type c = 0; c < num_children; ++c) {
        if (!combined_out && out_slot[c] >= 0) emit_keys.push_back(key.child_at(c));
        for (size_type g = 0; g < num_children; ++g) {
          if (!grand_down[c][g]) continue;
          down_keys.push_back(key.child_at(c).child_at(g));
          const std::uint8_t flags = (child_layout[c].a_node_at[g] ? 0 : 1) | (child_layout[c].b_node_at[g] ? 0 : 2);
          if (flags == 0) continue;
          if (balance.empty()) balance.resize(num_children * num_children, 0);
          balance[num_children * c + g] = flags;
        }
      }
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      sends.push_back(ttg::device::send<OUT>(key, std::move(result)));
      if (!emit_keys.empty()) sends.push_back(ttg::device::broadcast<PACKED>(std::move(emit_keys), std::move(packed)));
      if (!down_keys.empty()) sends.push_back(ttg::device::broadcast<S>(std::move(down_keys), std::move(down)));
      if (!balance.empty()) sends.push_back(ttg::device::send<BALANCE>(key, std::move(balance)));
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::send<OUT>(key, std::move(result));
      if (!emit_keys.empty()) ttg::broadcast<PACKED>(std::move(emit_keys), std::move(packed));
      if (!down_keys.empty()) ttg::broadcast<S>(std::move(down_keys), std::move(down));
      if (!balance.empty()) ttg::send<BALANCE>(key, std::move(balance));
#endif // MRA_ENABLE_HOST
    };

    /**
     * Balance (no device work): one empty node per factor, sent to the grandchildren where
     * the factor has none, as their node and as the node of each of their children.
     */
    auto balance_fn = [fnsA, fnsB](const key_type& key, const std::vector<std::uint8_t>& flags) -> TASKTYPE {
      std::vector<key_type> a_keys, b_keys;
      for (size_type i = 0; i < flags.size(); ++i) {
        const key_type grandchild = key.child_at(i / num_children).child_at(i % num_children);
        if (flags[i] & 1) a_keys.push_back(grandchild);
        if (flags[i] & 2) b_keys.push_back(grandchild);
      }
      auto to_all = [&]<std::size_t First, std::size_t... I>(std::integral_constant<std::size_t, First>,
                                                              std::index_sequence<I...>,
                                                              const std::vector<key_type>& keys, cnode_type&& empty) {
#ifndef MRA_ENABLE_HOST
        return ttg::device::broadcast<(First + I)...>(std::make_tuple(((void)I, keys)...), std::move(empty));
#else  // MRA_ENABLE_HOST
        ttg::broadcast<(First + I)...>(std::make_tuple(((void)I, keys)...), std::move(empty));
#endif // MRA_ENABLE_HOST
      };
      constexpr auto terminals = std::make_index_sequence<num_terminals>{};
      using a_first = std::integral_constant<std::size_t, 0>;
      using b_first = std::integral_constant<std::size_t, num_terminals>;
#ifndef MRA_ENABLE_HOST
      /* the device broadcast does not skip empty key lists */
      auto sends = ttg::device::forward();
      if (!a_keys.empty()) sends.push_back(to_all(a_first{}, terminals, a_keys, cnode_type(key, fnsA->num_functions(key))));
      if (!b_keys.empty()) sends.push_back(to_all(b_first{}, terminals, b_keys, cnode_type(key, fnsB->num_functions(key))));
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      if (!a_keys.empty()) to_all(a_first{}, terminals, a_keys, cnode_type(key, fnsA->num_functions(key)));
      if (!b_keys.empty()) to_all(b_first{}, terminals, b_keys, cnode_type(key, fnsB->num_functions(key)));
#endif // MRA_ENABLE_HOST
    };
    auto balance_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(balance_fn), ttg::edges(balance_e),
                                 ttg::edges(balance_a[I]..., balance_b[I]...), name + "-balance");
    }(std::make_index_sequence<num_terminals>{});

    auto multiply_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(multiply_fn),
                                 ttg::edges(ttg::fuse(route_a[I], balance_a[I])...,
                                            ttg::fuse(route_b[I], balance_b[I])...,
                                            ttg::fuse(s_e, primer_e)),
                                 ttg::edges(out, packed_e, s_e, balance_e), name);
    }(std::make_index_sequence<num_terminals>{});

    /**
     * Emit, at a child of the task's key: copy the product nodes at the child and at its
     * children out of the packed output into nodes, and send them on.
     */
    auto emit_fn = [](const key_type& key, const combined_type& packed) -> TASKTYPE {
      std::vector<size_type> slots;
      for (size_type s = 0; s < packed.num_slots(); ++s) {
        const key_type& slot_key = packed.slot_key(s);
        if (slot_key == key || (slot_key.level() == key.level() + 1 && slot_key.parent() == key)) slots.push_back(s);
      }
      std::vector<node_type> nodes;
      nodes.reserve(slots.size());
      for (size_type s : slots) nodes.push_back(packed.make_node(s));
#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(packed.buffer());
      for (auto& node : nodes) input.add(node.coeffs().buffer());
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
      for (size_type i = 0; i < slots.size(); ++i) packed.copy_slot_to(slots[i], nodes[i]);
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      for (auto& node : nodes) {
        const key_type node_key = node.key();
        sends.push_back(ttg::device::send<0>(node_key, std::move(node)));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      for (auto& node : nodes) {
        const key_type node_key = node.key();
        ttg::send<0>(node_key, std::move(node));
      }
#endif // MRA_ENABLE_HOST
    };
    /* combined output: packed goes on to the user, no emit task */
    std::unique_ptr<ttg::TTBase> emit_op;
    if (!combined_out) {
      auto emit_tt = ttg::make_tt<Space>(std::move(emit_fn), ttg::edges(packed_e), ttg::edges(out), name + "-emit");
      if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
        emit_tt->set_keymap([procmap](const key_type& key) { return procmap(key.parent()); });
      }
      if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
        emit_tt->set_devicemap([devicemap](const key_type& key) { return devicemap(key.parent()); });
      }
      emit_op = std::move(emit_tt);
    }

    /* a two-level subtree stays together: routers and emits (at odd levels, above) go where
     * their multiply task goes */
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      auto by_task = [procmap](const key_type& key) { return procmap(key.level() % 2 == 0 ? key : key.parent()); };
      route_a_tt->set_keymap(by_task);
      route_b_tt->set_keymap(by_task);
      primer_tt->set_keymap(procmap);
      multiply_tt->set_keymap(procmap);
      balance_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      auto by_task = [devicemap](const key_type& key) { return devicemap(key.level() % 2 == 0 ? key : key.parent()); };
      route_a_tt->set_devicemap(by_task);
      route_b_tt->set_devicemap(by_task);
      primer_tt->set_devicemap(devicemap);
      multiply_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      multiply_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(route_a_tt->template in<0>(), route_b_tt->template in<0>());
    auto outs = std::make_tuple(multiply_tt->template out<OUT>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops;
    ops.push_back(std::move(primer_tt));
    ops.push_back(std::move(route_a_tt));
    ops.push_back(std::move(route_b_tt));
    ops.push_back(std::move(multiply_tt));
    ops.push_back(std::move(balance_tt));
    if (emit_op) ops.push_back(std::move(emit_op));
    return make_ttg(std::move(ops), ins, outs, name);
  }

  } // namespace detail

  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap, typename DeviceMap>
  auto make_multiply_compressed_2level(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name,
    ProcMap procmap,
    DeviceMap devicemap)
  {
    return detail::make_multiply_compressed_2level_impl<T, NDIM>(db, fnsA, fnsB, functiondata, K, a, b, out,
                                                                 std::nullopt, name, procmap, devicemap);
  }

  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap, typename DeviceMap>
  auto make_multiply_compressed_combined(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    ttg::Edge<mra::Key<NDIM>, mra::CombinedReconstructedNode<T, NDIM>> packed,
    const std::string& name,
    ProcMap procmap,
    DeviceMap devicemap)
  {
    return detail::make_multiply_compressed_2level_impl<T, NDIM>(db, fnsA, fnsB, functiondata, K, a, b, out,
                                                                 packed, name, procmap, devicemap);
  }

  /**
   * Product of two sets of functions in compressed form, out_j = a_ia * b_j, with
   * ia = j (a and b have the same number of functions) or ia = 0 (a is a single
   * function, cf. madness::mul(world, a, v)). The output is the product in
   * reconstructed form; see the comment at the top of this file.
   */
  template <typename T, Dimension NDIM, typename FunctionSetA, typename FunctionSetB,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_multiply_compressed(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetA>& fnsA,
    const std::shared_ptr<FunctionSetB>& fnsB,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> a,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> b,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name = "multiply_compressed",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    if constexpr (NDIM == 3) {
      if (get_multiply_compressed_levels() == 2) {
        return make_multiply_compressed_2level<T, NDIM>(db, fnsA, fnsB, functiondata, K, a, b, out, name,
                                                        procmap, devicemap);
      }
    }
    using cnode_type = mra::FunctionsCompressedNode<T, NDIM>;
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;
    constexpr size_type num_children = mra::Key<NDIM>::num_children();
    constexpr const int OUT = 0, S = 1, BALANCE = 2;

    ttg::Edge<mra::Key<NDIM>, node_type> s_e(name + "-s"), primer_e(name + "-primer");
    ttg::Edge<mra::Key<NDIM>, cnode_type> balance_a(name + "-balance_a"), balance_b(name + "-balance_b");
    /* which children get an empty compressed node of a (bits 0..7) and of b (bits 8..15) */
    ttg::Edge<mra::Key<NDIM>, int> balance_e(name + "-balance");

    /* batching is controlled process-wide (mra/misc/batch_size.h), read once here; the
     * operator data is the same for every node, so any two tasks may be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type max_functions = fnsA->num_functions() + fnsB->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::MulCompressedBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = detail::shared_pool_registry<detail::MulCompressedBatchArg<T, NDIM>>(max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    auto check_sizes = [name](size_type NA, size_type N) {
      if (NA != N && NA != 1) {
        throw std::runtime_error(name + ": the first factor must have one function or as many as the second");
      }
    };

    auto primer_fn = [fnsA, fnsB, check_sizes](const mra::Key<NDIM>& key, const cnode_type&) -> TASKTYPE {
      if (key.level() == 0) {
        const size_type NA = fnsA->num_functions(key), N = fnsB->num_functions(key);
        check_sizes(NA, N);
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, detail::multiply_compressed_root_s<T, NDIM>(key, NA + N));
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, detail::multiply_compressed_root_s<T, NDIM>(key, NA + N));
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(a), ttg::edges(primer_e), name + "-primer");

    auto multiply_fn = [&db, &functiondata, fnsA, fnsB, K, name, check_sizes,
                        enable_batching, batch_pool, max_functions](
                          const mra::Key<NDIM>& key, const cnode_type& anode, const cnode_type& bnode,
                          const node_type& s_in) -> TASKTYPE {
      const size_type NA = fnsA->num_functions(key), N = fnsB->num_functions(key);
      check_sizes(NA, N);
      if (anode.is_ns() || bnode.is_ns()) {
        throw std::runtime_error(name + ": inputs must be in standard compressed form");
      }
      auto plan = detail::plan_multiply_compressed<T, NDIM>(key, K, NA, N, anode, &bnode, s_in);
      const size_type n_items = plan.work.size();

      /* scratch and work items must outlive the kernel: destroying a buffer releases its
       * device memory right away, while the kernel launched below is still running */
      ttg::Buffer<T, DeviceAllocator<T>> tmp;
      ttg::Buffer<detail::MulCompressedWork<NDIM>, DeviceAllocator<detail::MulCompressedWork<NDIM>>> work;
      if (n_items > 0) {
        const auto& hg = functiondata.get_hg();
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        tmp = ttg::Buffer<T, DeviceAllocator<T>>(multiply_compressed_tmp_size<NDIM>(K)*n_items, ttg::scope::Allocate);
        /* batched, the leader stages the work items of every member with the batch's
         * grouped copy (see submit_multiply_compressed_batch_leader): no buffer, and one
         * flow less to stage in, per task */
#ifndef MRA_ENABLE_HOST
        const bool stage_work = enable_batching;
#else  // MRA_ENABLE_HOST
        constexpr bool stage_work = false;
#endif // MRA_ENABLE_HOST
        if (!stage_work) {
          work = ttg::Buffer<detail::MulCompressedWork<NDIM>, DeviceAllocator<detail::MulCompressedWork<NDIM>>>(n_items);
          std::copy(plan.work.begin(), plan.work.end(), work.host_ptr());
        }
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, hg.buffer(), phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp);
        if (!stage_work) input.add(work);
        for (const cnode_type* n : {&anode, &bnode}) if (!n->empty()) input.add(n->coeffs().buffer());
        if (!s_in.empty()) input.add(s_in.coeffs().buffer());
        if (!plan.result.empty()) input.add(plan.result.coeffs().buffer());
        for (auto& o : plan.out) if (!o.empty()) input.add(o.coeffs().buffer());
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto out_views = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::array{(plan.out[Is].coeffs().current_view())...};
        }(std::make_index_sequence<num_children>{});
        auto result_view = plan.result.coeffs().current_view();
        auto a_view = anode.coeffs().current_view(), b_view = bnode.coeffs().current_view();
        auto s_view = s_in.coeffs().current_view();
        /* batched: the host plan, which lives in this frame while suspended in coop() */
        const detail::MulCompressedWork<NDIM>* work_ptr = stage_work ? plan.work.data() : work.current_device_ptr();
        T* tmp_ptr = tmp.current_device_ptr();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the sparsity of the outputs is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, a_view, b_view, s_view, result_view, out_views,
                                                                  work_ptr, NA, tmp_ptr, plan.result, plan.out,
                                                                  n_items);
          detail::submit_multiply_compressed_batch_leader<T, NDIM>(batch, *batch_pool, D, hg.current_view(),
                                                                   phibar.current_view(), phi.current_view(),
                                                                   quad_x.current_view(), false, K, max_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          if (plan.out[0].count() > 0) {
            auto sparseman = make_sparsity_manager(plan.out, plan.result);
            sparseman.populate_device_sparsity();
          } else {
            auto sparseman = make_sparsity_manager(plan.result);
            sparseman.populate_device_sparsity();
          }
          submit_multiply_compressed_kernel<T, NDIM>(D, key, a_view, b_view, s_view, result_view, out_views,
                                                     hg.current_view(), phibar.current_view(), phi.current_view(),
                                                     quad_x.current_view(), work_ptr, NA, false,
                                                     n_items, K, tmp_ptr, ttg::device::current_stream());
        }
      }

      /* the empty compressed nodes for children where a tree has none are sent by the balance
       * task: a device task holds a data copy of every value it sends until it completes, and
       * PaRSEC limits their number per task (MAX_PARAM_COUNT, including the inputs); this
       * keeps the task at 13 (3 inputs, the result, 8 children and the mask) */
      int balance = 0;
      for (size_type c = 0; c < num_children; ++c) {
        if (plan.out[c].count() == 0 || !plan.task[c]) continue;
        if (!plan.a_node_at[c]) balance |= 1 << c;
        if (!plan.b_node_at[c]) balance |= 1 << (num_children + c);
      }
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      sends.push_back(ttg::device::send<OUT>(key, std::move(plan.result)));
      for (size_type c = 0; c < num_children; ++c) {
        const mra::Key<NDIM> child = key.child_at(c);
        if (plan.out[c].count() == 0) continue; // nothing continues here
        if (plan.task[c]) sends.push_back(ttg::device::send<S>(child, std::move(plan.out[c])));
        else              sends.push_back(ttg::device::send<OUT>(child, std::move(plan.out[c])));
      }
      if (balance != 0) sends.push_back(ttg::device::send<BALANCE>(key, balance));
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::send<OUT>(key, std::move(plan.result));
      for (size_type c = 0; c < num_children; ++c) {
        const mra::Key<NDIM> child = key.child_at(c);
        if (plan.out[c].count() == 0) continue; // nothing continues here
        if (plan.task[c]) ttg::send<S>(child, std::move(plan.out[c]));
        else              ttg::send<OUT>(child, std::move(plan.out[c]));
      }
      if (balance != 0) ttg::send<BALANCE>(key, balance);
#endif // MRA_ENABLE_HOST
    };

    /**
     * Balance (no device work): an empty compressed node for every child where a or b has none
     * but the product continues, as compute's mask says.
     */
    auto balance_fn = [fnsA, fnsB](const mra::Key<NDIM>& key, const int& balance) -> TASKTYPE {
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
#endif // MRA_ENABLE_HOST
      for (size_type c = 0; c < num_children; ++c) {
        const mra::Key<NDIM> child = key.child_at(c);
        if ((balance >> c) & 1) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::send<0>(child, cnode_type(child, fnsA->num_functions(child))));
#else  // MRA_ENABLE_HOST
          ttg::send<0>(child, cnode_type(child, fnsA->num_functions(child)));
#endif // MRA_ENABLE_HOST
        }
        if ((balance >> (num_children + c)) & 1) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::send<1>(child, cnode_type(child, fnsB->num_functions(child))));
#else  // MRA_ENABLE_HOST
          ttg::send<1>(child, cnode_type(child, fnsB->num_functions(child)));
#endif // MRA_ENABLE_HOST
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };
    auto balance_tt = ttg::make_tt<Space>(std::move(balance_fn), ttg::edges(balance_e),
                                          ttg::edges(balance_a, balance_b), name + "-balance",
                                          {"mask"}, {"balance_a", "balance_b"});

    auto multiply_tt = ttg::make_tt<Space>(std::move(multiply_fn),
                                           ttg::edges(ttg::fuse(balance_a, a), ttg::fuse(balance_b, b),
                                                      ttg::fuse(s_e, primer_e)),
                                           ttg::edges(out, s_e, balance_e),
                                           name, {"a", "b", "s"}, {"out", "s", "balance"});

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      primer_tt->set_keymap(procmap);
      multiply_tt->set_keymap(procmap);
      balance_tt->set_keymap(procmap);
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
    std::vector<std::unique_ptr<ttg::TTBase>> ops(3);
    ops[0] = std::move(primer_tt);
    ops[1] = std::move(multiply_tt);
    ops[2] = std::move(balance_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }

  /**
   * Square of every function of a set in compressed form, out_j = f_j^2; the
   * compressed counterpart of make_square. The output is the product in
   * reconstructed form; see the comment at the top of this file.
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_square_compressed(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetT>& fns,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name = "square_compressed",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    using cnode_type = mra::FunctionsCompressedNode<T, NDIM>;
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;
    constexpr size_type num_children = mra::Key<NDIM>::num_children();
    constexpr const int OUT = 0, S = 1;

    ttg::Edge<mra::Key<NDIM>, node_type> s_e(name + "-s"), primer_e(name + "-primer");

    /* batching, as in make_multiply_compressed */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type max_functions = fns->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::MulCompressedBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = detail::shared_pool_registry<detail::MulCompressedBatchArg<T, NDIM>>(max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    auto primer_fn = [fns](const mra::Key<NDIM>& key, const cnode_type&) -> TASKTYPE {
      if (key.level() == 0) {
        const size_type N = fns->num_functions(key);
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, detail::multiply_compressed_root_s<T, NDIM>(key, N));
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, detail::multiply_compressed_root_s<T, NDIM>(key, N));
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(in), ttg::edges(primer_e), name + "-primer");

    /* a function's tree is the product's tree, so every child that continues has a
     * compressed node and no balancing is needed */
    auto square_fn = [&db, &functiondata, fns, K, name, enable_batching, batch_pool, max_functions](
                        const mra::Key<NDIM>& key, const cnode_type& node, const node_type& s_in) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      if (node.is_ns()) {
        throw std::runtime_error(name + ": input must be in standard compressed form");
      }
      auto plan = detail::plan_multiply_compressed<T, NDIM>(key, K, N, N, node, nullptr, s_in);
      const size_type n_items = plan.work.size();

      ttg::Buffer<T, DeviceAllocator<T>> tmp;
      ttg::Buffer<detail::MulCompressedWork<NDIM>, DeviceAllocator<detail::MulCompressedWork<NDIM>>> work;
      if (n_items > 0) {
        const auto& hg = functiondata.get_hg();
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        tmp = ttg::Buffer<T, DeviceAllocator<T>>(multiply_compressed_tmp_size<NDIM>(K)*n_items, ttg::scope::Allocate);
        /* batched, the leader stages the work items of every member with the batch's
         * grouped copy (see submit_multiply_compressed_batch_leader): no buffer, and one
         * flow less to stage in, per task */
#ifndef MRA_ENABLE_HOST
        const bool stage_work = enable_batching;
#else  // MRA_ENABLE_HOST
        constexpr bool stage_work = false;
#endif // MRA_ENABLE_HOST
        if (!stage_work) {
          work = ttg::Buffer<detail::MulCompressedWork<NDIM>, DeviceAllocator<detail::MulCompressedWork<NDIM>>>(n_items);
          std::copy(plan.work.begin(), plan.work.end(), work.host_ptr());
        }
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, hg.buffer(), phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp);
        if (!stage_work) input.add(work);
        if (!node.empty()) input.add(node.coeffs().buffer());
        if (!s_in.empty()) input.add(s_in.coeffs().buffer());
        if (!plan.result.empty()) input.add(plan.result.coeffs().buffer());
        for (auto& o : plan.out) if (!o.empty()) input.add(o.coeffs().buffer());
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto out_views = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::array{(plan.out[Is].coeffs().current_view())...};
        }(std::make_index_sequence<num_children>{});
        auto node_view = node.coeffs().current_view();
        auto result_view = plan.result.coeffs().current_view();
        auto s_view = s_in.coeffs().current_view();
        /* batched: the host plan, which lives in this frame while suspended in coop() */
        const detail::MulCompressedWork<NDIM>* work_ptr = stage_work ? plan.work.data() : work.current_device_ptr();
        T* tmp_ptr = tmp.current_device_ptr();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the sparsity of the outputs is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, node_view, node_view, s_view, result_view,
                                                                  out_views, work_ptr, N, tmp_ptr, plan.result,
                                                                  plan.out, n_items);
          detail::submit_multiply_compressed_batch_leader<T, NDIM>(batch, *batch_pool, D, hg.current_view(),
                                                                   phibar.current_view(), phi.current_view(),
                                                                   quad_x.current_view(), true, K, max_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          if (plan.out[0].count() > 0) {
            auto sparseman = make_sparsity_manager(plan.out, plan.result);
            sparseman.populate_device_sparsity();
          } else {
            auto sparseman = make_sparsity_manager(plan.result);
            sparseman.populate_device_sparsity();
          }
          submit_multiply_compressed_kernel<T, NDIM>(D, key, node_view, node_view, s_view, result_view, out_views,
                                                     hg.current_view(), phibar.current_view(), phi.current_view(),
                                                     quad_x.current_view(), work_ptr, N, true,
                                                     n_items, K, tmp_ptr, ttg::device::current_stream());
        }
      }

#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      sends.push_back(ttg::device::send<OUT>(key, std::move(plan.result)));
      for (size_type c = 0; c < num_children; ++c) {
        if (plan.out[c].count() == 0) continue; // nothing continues here
        if (plan.task[c]) sends.push_back(ttg::device::send<S>(key.child_at(c), std::move(plan.out[c])));
        else              sends.push_back(ttg::device::send<OUT>(key.child_at(c), std::move(plan.out[c])));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::send<OUT>(key, std::move(plan.result));
      for (size_type c = 0; c < num_children; ++c) {
        if (plan.out[c].count() == 0) continue; // nothing continues here
        if (plan.task[c]) ttg::send<S>(key.child_at(c), std::move(plan.out[c]));
        else              ttg::send<OUT>(key.child_at(c), std::move(plan.out[c]));
      }
#endif // MRA_ENABLE_HOST
    };

    auto square_tt = ttg::make_tt<Space>(std::move(square_fn),
                                         ttg::edges(in, ttg::fuse(s_e, primer_e)),
                                         ttg::edges(out, s_e),
                                         name, {"in", "s"}, {"out", "s"});

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      primer_tt->set_keymap(procmap);
      square_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      primer_tt->set_devicemap(devicemap);
      square_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      square_tt->set_batch_matcher([](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
                                   max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(square_tt->template in<0>());
    auto outs = std::make_tuple(square_tt->template out<OUT>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(primer_tt);
    ops[1] = std::move(square_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_MULTIPLY_COMPRESSED_H
