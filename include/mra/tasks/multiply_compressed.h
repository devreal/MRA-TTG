#ifndef MRA_TASKS_MULTIPLY_COMPRESSED_H
#define MRA_TASKS_MULTIPLY_COMPRESSED_H

#include <algorithm>
#include <array>
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

    /**
     * What the task at `key` produces: the product node of `key`, for every child
     * either its product node (no task there) or the scaling coefficients for its
     * task, and the work items of the kernel.
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
     * s_in carries, for functions [0, NA) of the first factor and (unless b_node is
     * null, a square) [NA, NA+N) of the second, the scaling coefficients at `key`
     * and the status of the function in its own tree: Inner where it has children,
     * Leaf where its tree ended here or above, Invalid where it is not needed.
     * NA is 1 (one function times N) or N (elementwise).
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
      using node_type = FunctionsReconstructedNode<T, NDIM>;
      constexpr size_type num_children = Key<NDIM>::num_children();
      const bool square = (b_node == nullptr);
      MulCompressedPlan<T, NDIM> plan;

      auto ia_of = [&](size_type j) { return (NA == 1) ? size_type(0) : j; };
      auto a_status = [&](size_type i) { return s_in.leaf_status(i); };
      auto b_status = [&](size_type j) { return square ? s_in.leaf_status(j) : s_in.leaf_status(NA + j); };
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
      /* allocated even without leaves, as make_reconstruct does, so its sparsity can be pushed */
      plan.result = node_type(key, result_sparsity, K, ttg::scope::Allocate);
      for (size_type j = 0; j < N; ++j) plan.result.set_leaf(j, status[j]);

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
          const Key<NDIM> child = key.child_at(c);
          auto& out = plan.out[c];
          if (!plan.task[c]) {
            /* the product ends at the child for every function that continues here */
            SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
            for (size_type j = 0; j < N; ++j) if (status[j] == LeafStatus::Inner) sparsity.set_nonzero(j);
            out = node_type(child, sparsity, K, ttg::scope::Allocate);
            for (size_type j = 0; j < N; ++j) {
              out.set_leaf(j, status[j] == LeafStatus::Inner ? LeafStatus::Leaf : LeafStatus::Invalid);
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
            out = node_type(child, sparsity, K, ttg::scope::Allocate);
            for (size_type m = 0; m < M; ++m) out.set_leaf(m, child_status[m]);
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

    /** the scaling coefficients at the root: none, every function continues */
    template <typename T, Dimension NDIM>
    FunctionsReconstructedNode<T, NDIM> multiply_compressed_root_s(const Key<NDIM>& key, size_type M) {
      FunctionsReconstructedNode<T, NDIM> s(key, M);
      s.set_all_leaf(LeafStatus::Inner);
      return s;
    }

  } // namespace detail

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
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::MulCompressedBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
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
        work = ttg::Buffer<detail::MulCompressedWork<NDIM>, DeviceAllocator<detail::MulCompressedWork<NDIM>>>(n_items);
        std::copy(plan.work.begin(), plan.work.end(), work.host_ptr());
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, hg.buffer(), phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp, work);
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
        const detail::MulCompressedWork<NDIM>* work_ptr = work.current_device_ptr();
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
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::MulCompressedBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
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
        work = ttg::Buffer<detail::MulCompressedWork<NDIM>, DeviceAllocator<detail::MulCompressedWork<NDIM>>>(n_items);
        std::copy(plan.work.begin(), plan.work.end(), work.host_ptr());
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, hg.buffer(), phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp, work);
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
        const detail::MulCompressedWork<NDIM>* work_ptr = work.current_device_ptr();
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
