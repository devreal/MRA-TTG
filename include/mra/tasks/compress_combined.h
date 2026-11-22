#ifndef MRA_TASKS_COMPRESS_COMBINED_H
#define MRA_TASKS_COMPRESS_COMBINED_H

#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <ttg.h>
#include "mra/tasks/compress.h"
#include "mra/tensor/combinednode.h"

/**
 * make_compress for the output of make_multiply_compressed_combined, which leaves the
 * product nodes at odd levels and the leaves at even levels in the packed output of
 * each multiply task (a CombinedReconstructedNode keyed by the task's key) instead of
 * unpacking them into nodes. The compress tasks read those nodes from their slots.
 *
 * - The router takes the product nodes at even levels (the `in` edge): a node where some
 *   function continues goes to the compress task at its key, a leaf goes up to its
 *   parent's compress task, as make_compress does.
 * - The dispatch task takes the packed output of the multiply task at an even key K and
 *   forwards it to the compress task at K and to the compress tasks at the children of K
 *   where the product continues (whose own node is a slot). A child read from a slot
 *   (a leaf for every function) still needs a value on its parent's input terminals: the
 *   dispatch task sends an empty node and an empty "kept" there.
 * - Two compress tasks: at even levels, with the node from the router and the packed
 *   output of the task at the key; at odd levels, with the packed output of the task at
 *   the parent, which holds the task's own node. Both read a child from packed if it has
 *   a slot there without a function that continues, and from their input terminal
 *   otherwise. They run the kernels of make_compress, batched with each other.
 *
 * Compress tasks at odd levels are mapped like the multiply task at their parent, so the
 * packed output stays where it was written. NDIM=3 only, as make_compress.
 */

namespace mra {

  namespace detail {

    /** whether some function of slot s continues below the slot's key */
    template <typename T, Dimension NDIM>
    bool slot_any_inner(const CombinedReconstructedNode<T, NDIM>& node, size_type s) {
      for (size_type m = 0; m < node.slot_count(s); ++m) {
        if (node.slot_status(s, m) == LeafStatus::Inner) return true;
      }
      return false;
    }

    inline std::atomic<int>& compress_levels_state() {
      static std::atomic<int> value{[] {
        const char* env = std::getenv("MRA_TTG_COMPRESS_LEVELS");
        return (env != nullptr && std::atoi(env) == 2) ? 2 : 1;
      }()};
      return value;
    }

    /**
     * With two levels, whether the even-level compress task at the parent of `child` (whose
     * multiply task wrote `packed`) compresses `child` itself: the product continues at the
     * child, but ends at all of its children, which are slots of packed. Then no task a level
     * below sends anything to the child's compress, which needs packed only.
     */
    template <typename T, Dimension NDIM>
    bool compress_folds_child(const CombinedReconstructedNode<T, NDIM>& packed, const Key<NDIM>& child) {
      const int s = packed.find_slot(child);
      if (s < 0 || !slot_any_inner(packed, s)) return false;
      for (size_type g = 0; g < Key<NDIM>::num_children(); ++g) {
        const int sg = packed.find_slot(child.child_at(g));
        if (sg < 0 || slot_any_inner(packed, sg)) return false;
      }
      return true;
    }

    /** one reconstructed input of the combined compress: a node, or a slot of a combined node */
    template <typename T, Dimension NDIM>
    struct CompressCombinedSource {
      using node_type = FunctionsReconstructedNode<T, NDIM>;
      using combined_type = CombinedReconstructedNode<T, NDIM>;
      using view_type = typename combined_type::view_type;

      const node_type* node = nullptr;
      const combined_type* combined = nullptr;
      size_type slot = 0;

      static CompressCombinedSource of_node(const node_type& n) { return {&n, nullptr, 0}; }
      static CompressCombinedSource of_slot(const combined_type& c, size_type s) { return {nullptr, &c, s}; }

      bool is_slot() const { return combined != nullptr; }

      bool is_nonzero(size_type i) const {
        if (node) return node->sparsity().is_nonzero(i);
        return combined->slot_sparsity_data(slot)[i] & SparsityState::NONZERO;
      }

      /* no data: an empty node, or a slot without nonzero functions */
      bool empty() const {
        if (node) return node->empty();
        for (size_type i = 0; i < combined->slot_count(slot); ++i) if (is_nonzero(i)) return false;
        return true;
      }

      LeafStatus leaf_status(size_type i) const {
        return node ? node->leaf_status(i) : combined->slot_status(slot, i);
      }

      bool is_leaf_or_invalid(size_type i) const {
        const LeafStatus s = leaf_status(i);
        return s == LeafStatus::Leaf || s == LeafStatus::Invalid;
      }

      view_type view() const {
        if (node) return node->coeffs().current_view();
        return combined->slot_view(slot);
      }
    };

    /** what both compress tasks of make_compress_combined share */
    template <typename T, Dimension NDIM, typename FunctionSetT>
    struct CompressCombinedContext {
      std::shared_ptr<FunctionSetT> fns;
      size_type K;
      bool is_ns;
      const FunctionData<T, NDIM>* functiondata;
      std::string name;
      bool enable_batching;
#ifndef MRA_ENABLE_HOST
      std::shared_ptr<GroupedBatchPoolRegistry<CompressBatchArg<T, NDIM>>> compress_pool;
#else  // MRA_ENABLE_HOST
      std::nullptr_t compress_pool = nullptr;
#endif // MRA_ENABLE_HOST
      size_type total_functions;
      bool enable_truncate;
      T thresh;
      int truncate_mode;
      T cell_min_width;
    };

    /**
     * The compress task at `key` (do_compress of make_compress, from a reconstructed tree):
     * own is the node at the key, packed the packed output read by the task, p_in/kept_in
     * the task's input terminals for the children; a child with a slot in packed without
     * a function that continues is read from there (its terminals hold placeholders).
     * The outputs are p and kept to the parent's terminals 0..7 and 8..15, and the
     * compressed node to terminal 16.
     */
    template <typename T, Dimension NDIM, typename Ctx>
    TASKTYPE compress_combined_task(
      const Ctx& ctx,
      Key<NDIM> key,
      CompressCombinedSource<T, NDIM> own,
      const CombinedReconstructedNode<T, NDIM>& packed,
      std::array<const FunctionsReconstructedNode<T, NDIM>*, Key<NDIM>::num_children()> p_in,
      std::array<const KeptInfo<T>*, Key<NDIM>::num_children()> kept_children)
    {
      using source_type = CompressCombinedSource<T, NDIM>;
      using view_type = typename source_type::view_type;
      using kept_info_type = KeptInfo<T>;
      constexpr const auto num_children = Key<NDIM>::num_children();
      constexpr const auto out_terminal_id = 2*num_children;
      const size_type N = ctx.fns->num_functions(key);
      const size_type K = ctx.K;
      mra::FunctionsCompressedNode<T, NDIM> result(key, N);
      mra::FunctionsReconstructedNode<T, NDIM> p(key, N);
      kept_info_type kept;

      /* the children: from packed where it holds a leaf, else from the terminals */
      std::array<source_type, num_children> child;
      bool uses_packed = own.is_slot();
      for (size_type c = 0; c < num_children; ++c) {
        const int s = packed.empty() ? -1 : packed.find_slot(key.child_at(c));
        if (s >= 0 && !slot_any_inner(packed, s)) {
          child[c] = source_type::of_slot(packed, s);
          uses_packed = true;
        } else {
          child[c] = source_type::of_node(*p_in[c]);
        }
      }

      /* truncation bookkeeping as in do_compress; a child read from packed has an empty
       * "kept" (its placeholder): a leaf */
      auto child_kept = [&](size_type i, size_type c) -> bool {
        const auto& ck = *kept_children[c];
        return !ck.empty() && ck[i] != T(0);
      };
      auto mark_truncated_children_leaf = [&]() {
        for (size_type i = 0; i < N; ++i) {
          for (size_type c = 0; c < num_children; ++c) {
            if (!child_kept(i, c)) result.set_child_leaf(i, c, true);
          }
        }
      };
      auto apply_child_leaf_info = [&]() {
        for (size_type i = 0; i < N; ++i) {
          for (size_type c = 0; c < num_children; ++c) result.set_child_leaf(i, c, child[c].is_leaf_or_invalid(i));
        }
      };

      ttg::Buffer<unsigned char> child_kept_packed;
      ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch;
      ttg::Buffer<T, DeviceAllocator<T>> d_sumsq;

      bool all_empty = own.empty();
      for (size_type c = 0; c < num_children; ++c) all_empty = all_empty && child[c].empty();

      if (all_empty) {
        apply_child_leaf_info();
        mark_truncated_children_leaf();
      } else {
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type i = 0; i < N; ++i) {
          for (size_type c = 0; c < num_children; ++c) {
            if (child[c].is_nonzero(i)) { sparsity.set_nonzero(i); break; }
          }
        }
        result.allocate(sparsity, K, ttg::scope::Allocate);
        result.set_ns(ctx.is_ns);
        apply_child_leaf_info();
        mark_truncated_children_leaf();

        /* p: nonzero where the result or the node itself is, with the node's leaf status */
        sparsity.set_all_zero();
        for (size_type i = 0; i < N; ++i) {
          if (result.sparsity().is_nonzero(i) || own.is_nonzero(i)) sparsity.set_nonzero(i);
        }
        for (size_type i = 0; i < N; ++i) p.set_leaf(i, own.leaf_status(i));
        p.allocate(sparsity, K, ttg::scope::Allocate);

        FunctionNorms<T, NDIM> norms(ctx.name, p, result);
        const size_type n_nonzero = sparsity.count_nonzero();

#ifndef MRA_ENABLE_HOST
        const bool use_batching = ctx.enable_batching;
#else
        const bool use_batching = false;
#endif
        const std::size_t tmp_size = use_batching ? 0 : compress_tmp_size<NDIM>(K)*n_nonzero;
        if (tmp_size > 0) tmp_scratch.reset(tmp_size, ttg::scope::Allocate);
        const auto& hgT = ctx.functiondata->get_hgT();
        const bool may_truncate = ctx.enable_truncate && key.level() > 1;
        /* see do_compress (compress.h): with results_via_slot(), d_sumsq is host memory the
         * batch's results are copied to, and only where the host reads them */
#ifndef MRA_ENABLE_HOST
        const bool slot_results = use_batching && detail::results_via_slot();
#else
        const bool slot_results = false;
#endif
        if (!slot_results) {
          d_sumsq.reset(n_nonzero, may_truncate ? ttg::scope::SyncOut : ttg::scope::Allocate);
        } else if (may_truncate) {
          d_sumsq.reset(n_nonzero, ttg::scope::SyncOut);
        }
        const T trunc_tol = ctx.enable_truncate ? mra::truncate_tol(key, ctx.thresh, ctx.cell_min_width,
                                                                    ctx.truncate_mode)
                                                : T{};
        auto& d = result.coeffs();

        const bool pack_child_kept = may_truncate && !use_batching;
        if (pack_child_kept) {
          child_kept_packed.reset(N * num_children, ttg::scope::SyncIn);
          unsigned char* kept_bytes = child_kept_packed.host_ptr();
          for (size_type c = 0; c < num_children; ++c) {
            for (size_type i = 0; i < N; ++i) kept_bytes[c*N + i] = child_kept(i, c) ? 1 : 0;
          }
        }

#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(p.coeffs().buffer(), d.buffer(), hgT.buffer());
        if (!slot_results) input.add(d_sumsq);
        if (tmp_size > 0) input.add(tmp_scratch);
        if (pack_child_kept) input.add(child_kept_packed);
        if (own.node != nullptr && !own.node->empty()) input.add(own.node->coeffs().buffer());
        for (size_type c = 0; c < num_children; ++c) {
          if (!child[c].is_slot() && !child[c].node->empty()) input.add(child[c].node->coeffs().buffer());
        }
        if (uses_packed) input.add(packed.buffer());
        input.add(norms.buffer());
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

        auto input_views = [&]<std::size_t... I>(std::index_sequence<I...>) {
          return std::array<view_type, num_children>{child[I].view()...};
        }(std::make_index_sequence<num_children>{});
        view_type in_view = own.view();
        view_type ns_view{};
        auto coeffs_view = p.coeffs().current_view();
        auto rcoeffs_view = d.current_view();
        auto hgT_view = hgT.current_view();
        const unsigned char* child_kept_ptr = pack_child_kept ? child_kept_packed.current_device_ptr() : nullptr;

#ifndef MRA_ENABLE_HOST
        if (ctx.enable_batching) {
          /* the same batch arguments as do_compress, so its leader serves both */
          const auto* kept_children_ptr = may_truncate ? &kept_children : nullptr;
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, in_view, coeffs_view, rcoeffs_view,
                                                                  d_sumsq, input_views,
                                                                  p.coeffs(), d, n_nonzero,
                                                                  kept_children_ptr, trunc_tol, ns_view);
          detail::submit_compress_batch_leader<T, NDIM>(batch, *ctx.compress_pool, K, ctx.is_ns, hgT_view,
                                                        ctx.total_functions, ctx.enable_truncate);
        } else
#endif // MRA_ENABLE_HOST
        {
          auto sparseman = make_sparsity_manager(d, p);
          sparseman.populate_device_sparsity();
          submit_compress_kernel(key, N, n_nonzero, K, ctx.is_ns,
                                 in_view, coeffs_view, rcoeffs_view, hgT_view,
                                 tmp_scratch.current_device_ptr(), d_sumsq.current_device_ptr(), input_views,
                                 ns_view, child_kept_ptr, trunc_tol,
                                 ttg::device::current_stream());
        }
        norms.compute();
#ifndef MRA_ENABLE_HOST
        if (may_truncate) {
          if (slot_results) {
#if defined(MRA_CHECK_NORMS)
            co_await ttg::device::wait(norms.buffer());
#else
            co_await ttg::device::wait();
#endif
          } else {
            co_await ttg::device::wait(d_sumsq, norms.buffer());
          }
        }
#if defined(MRA_CHECK_NORMS)
        else {
          co_await ttg::device::wait(norms.buffer());
        }
#endif // MRA_CHECK_NORMS
#endif // MRA_ENABLE_HOST
        norms.verify();

        /* mirror the kernel's truncation decision on the host, as do_compress */
        if (may_truncate) {
          const auto* d_sumsq_arr = d_sumsq.host_ptr();
          size_type pos = 0;
          for (auto it = sparsity.begin_nonzero(); it != sparsity.end_nonzero(); ++it, ++pos) {
            const size_type i = *it;
            if (result.is_zero(i) || d_sumsq_arr[pos] >= trunc_tol*trunc_tol) continue;
            bool any_child_kept = false;
            for (size_type c = 0; c < num_children && !any_child_kept; ++c) any_child_kept = child_kept(i, c);
            if (!any_child_kept) result.set_zero(i);
          }
        }

        kept = kept_info_type(N);
        for (size_type i = 0; i < N; ++i) kept[i] = result.is_zero(i) ? T(0) : T(1);
      }

      if (key.level() > 0) {
        /* a node kept for no function is not part of the output tree, see do_compress */
        bool any_kept = false;
        for (size_type i = 0; i < kept.size() && !any_kept; ++i) any_kept = (kept[i] != T(0));
        constexpr auto children = std::make_index_sequence<num_children>{};
#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward(
          select_send_up(key, std::move(p), children, "compress_combined"),
          select_send_up<num_children>(key, std::move(kept), children, "compress_combined-kept"));
        if (any_kept) sends.push_back(ttg::device::send<out_terminal_id>(key, std::move(result)));
        co_await std::move(sends);
#else
        select_send_up(key, std::move(p), children, "compress_combined");
        select_send_up<num_children>(key, std::move(kept), children, "compress_combined-kept");
        if (any_kept) ttg::send<out_terminal_id>(key, std::move(result));
#endif
      } else {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
        ttg::send<out_terminal_id>(key, std::move(result));
#endif
      }
    }


#ifndef MRA_ENABLE_HOST
    /**
     * The compress task at an even key with two levels (batched only): as
     * compress_combined_task, but it also compresses the children that
     * compress_folds_child() names, from their children in packed, in a first launch (step
     * 1). Their scaling coefficients go to a scratch combined node that the launch at the key
     * (step 2) reads, and step 1 writes their "kept" bits into the child_kept bytes of step
     * 2, so both run in one batch. The terminals of a folded child hold placeholders.
     */
    template <typename T, Dimension NDIM, typename Ctx>
    TASKTYPE compress_combined_2level_task(
      const Ctx& ctx,
      Key<NDIM> key,
      const FunctionsReconstructedNode<T, NDIM>& own_node,
      const CombinedReconstructedNode<T, NDIM>& packed,
      std::array<const FunctionsReconstructedNode<T, NDIM>*, Key<NDIM>::num_children()> p_in,
      std::array<const KeptInfo<T>*, Key<NDIM>::num_children()> kept_children)
    {
      using key_type = Key<NDIM>;
      using source_type = CompressCombinedSource<T, NDIM>;
      using view_type = typename source_type::view_type;
      using kept_info_type = KeptInfo<T>;
      using cnode_type = FunctionsCompressedNode<T, NDIM>;
      using rnode_type = FunctionsReconstructedNode<T, NDIM>;
      using combined_type = CombinedReconstructedNode<T, NDIM>;
      using launch_type = Compress2Launch<T, NDIM>;
      constexpr const auto num_children = key_type::num_children();
      constexpr const auto out_terminal_id = 2*num_children;
      const size_type N = ctx.fns->num_functions(key);
      const size_type K = ctx.K;
      /* a folded child is at an odd level >= 1, so it may truncate exactly when the key does */
      const bool may_truncate = ctx.enable_truncate && key.level() > 1;
      const bool slot_results = detail::results_via_slot();
      const source_type own = source_type::of_node(own_node);
      auto tol_of = [&](const key_type& k) {
        return ctx.enable_truncate ? mra::truncate_tol(k, ctx.thresh, ctx.cell_min_width, ctx.truncate_mode) : T{};
      };

      /* the folded children: compressed from their children's slots in packed */
      std::array<bool, num_children> folded{};
      std::array<int, num_children> pslot;   // slot of the scaling coefficients in pscratch, -1 if none
      pslot.fill(-1);
      std::array<std::array<source_type, num_children>, num_children> gsrc;
      std::array<cnode_type, num_children> cres;
      std::array<SparsityInfo, num_children> csp; // the work items of a folded child
      std::array<kept_info_type, num_children> ckept;
      combined_type pscratch(key, K);
      for (size_type c = 0; c < num_children; ++c) {
        const key_type ck = key.child_at(c);
        if (!compress_folds_child(packed, ck)) continue;
        folded[c] = true;
        const source_type cown = source_type::of_slot(packed, packed.find_slot(ck));
        bool all_empty = cown.empty();
        for (size_type g = 0; g < num_children; ++g) {
          gsrc[c][g] = source_type::of_slot(packed, packed.find_slot(ck.child_at(g)));
          all_empty = all_empty && gsrc[c][g].empty();
        }
        cres[c] = cnode_type(ck, N);
        if (all_empty) continue; // nothing kept: neither sent on nor read
        SparsityInfo rs(N, SparsityInfo::InitType::AllZero);
        for (size_type i = 0; i < N; ++i) {
          for (size_type g = 0; g < num_children; ++g) {
            if (gsrc[c][g].is_nonzero(i)) { rs.set_nonzero(i); break; }
          }
        }
        cres[c].allocate(rs, K, ttg::scope::Allocate);
        cres[c].set_ns(ctx.is_ns);
        /* its children end everywhere and none is kept: all are leaves (see compress_combined_task) */
        for (size_type i = 0; i < N; ++i) {
          for (size_type g = 0; g < num_children; ++g) cres[c].set_child_leaf(i, g, true);
        }
        SparsityInfo ps(N, SparsityInfo::InitType::AllZero);
        std::vector<LeafStatus> status(N);
        for (size_type i = 0; i < N; ++i) {
          if (cres[c].sparsity().is_nonzero(i) || cown.is_nonzero(i)) ps.set_nonzero(i);
          status[i] = cown.leaf_status(i);
        }
        pslot[c] = pscratch.add_slot(ck, ps, std::move(status));
        csp[c] = std::move(ps);
      }
      if (pscratch.num_slots() > 0) pscratch.allocate();

      /* the children of the key: from packed where it holds a leaf, the scratch where folded,
       * else from the terminals (also a folded child without coefficients: a placeholder) */
      std::array<source_type, num_children> child;
      for (size_type c = 0; c < num_children; ++c) {
        const int s = packed.empty() ? -1 : packed.find_slot(key.child_at(c));
        if (!folded[c] && s >= 0 && !slot_any_inner(packed, s)) {
          child[c] = source_type::of_slot(packed, s);
        } else if (pslot[c] >= 0) {
          child[c] = source_type::of_slot(pscratch, pslot[c]);
        } else {
          child[c] = source_type::of_node(*p_in[c]);
        }
      }
      /* kept bits of the children, those of folded ones known after the kernels */
      auto child_kept = [&](size_type i, size_type c) -> bool {
        const auto& ck = folded[c] ? ckept[c] : *kept_children[c];
        return !ck.empty() && ck[i] != T(0);
      };

      cnode_type result(key, N);
      rnode_type p(key, N);
      kept_info_type kept;
      bool all_empty = own.empty();
      for (size_type c = 0; c < num_children; ++c) all_empty = all_empty && child[c].empty();
      auto finish_child_leaf_info = [&]() {
        for (size_type i = 0; i < N; ++i) {
          for (size_type c = 0; c < num_children; ++c) {
            result.set_child_leaf(i, c, child[c].is_leaf_or_invalid(i) || !child_kept(i, c));
          }
        }
      };

      if (all_empty) {
        /* then every folded child is empty too: nothing to compute */
        finish_child_leaf_info();
      } else {
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type i = 0; i < N; ++i) {
          for (size_type c = 0; c < num_children; ++c) {
            if (child[c].is_nonzero(i)) { sparsity.set_nonzero(i); break; }
          }
        }
        result.allocate(sparsity, K, ttg::scope::Allocate);
        result.set_ns(ctx.is_ns);
        sparsity.set_all_zero();
        for (size_type i = 0; i < N; ++i) {
          if (result.sparsity().is_nonzero(i) || own.is_nonzero(i)) sparsity.set_nonzero(i);
        }
        for (size_type i = 0; i < N; ++i) p.set_leaf(i, own.leaf_status(i));
        p.allocate(sparsity, K, ttg::scope::Allocate);
        const size_type n_nonzero = sparsity.count_nonzero();

        /* the sums of squares of all launches, the key's first */
        std::array<size_type, num_children> csumsq{};
        size_type total_sumsq = n_nonzero;
        for (size_type c = 0; c < num_children; ++c) {
          if (pslot[c] < 0) continue;
          csumsq[c] = total_sumsq;
          total_sumsq += csp[c].count_nonzero();
        }
        ttg::Buffer<T, DeviceAllocator<T>> d_sumsq;
        if (!slot_results) {
          d_sumsq.reset(total_sumsq, may_truncate ? ttg::scope::SyncOut : ttg::scope::Allocate);
        } else if (may_truncate) {
          d_sumsq.reset(total_sumsq, ttg::scope::SyncOut);
        }
        const auto& hgT = ctx.functiondata->get_hgT();
        auto& d = result.coeffs();

        auto input = ttg::device::Input(p.coeffs().buffer(), d.buffer(), hgT.buffer());
        if (!d_sumsq.empty()) input.add(d_sumsq);
        if (!own_node.empty()) input.add(own_node.coeffs().buffer());
        for (size_type c = 0; c < num_children; ++c) {
          if (pslot[c] >= 0) input.add(cres[c].coeffs().buffer());
          else if (!child[c].is_slot() && !child[c].node->empty()) input.add(child[c].node->coeffs().buffer());
        }
        if (!packed.empty()) input.add(packed.buffer());
        if (!pscratch.empty()) input.add(pscratch.buffer());
        co_await ttg::device::select(input);

        const auto device = ttg::device::current_device();
        auto sumsq_at = [&](size_type offset, launch_type& l) {
          if (d_sumsq.empty()) return;
          if (slot_results) l.d_sumsq_host = d_sumsq.host_ptr() + offset;
          else l.d_sumsq_dev = d_sumsq.current_device_ptr() + offset;
        };
        Compress2BatchMember<T, NDIM> member;
        std::vector<std::vector<SparsityState>> bytes;
        bytes.reserve(2 + num_children);
        auto push_tensor = [&](auto& tensor) {
          auto& b = bytes.emplace_back(N);
          sparsity_to_bytes(tensor.sparsity(), b.data(), N);
          member.sparsity.push_back({const_cast<T*>(tensor.buffer().device_ptr_on(device)), b.data(), N});
        };
        auto views = [&]<std::size_t... I>(std::index_sequence<I...>, auto&& view_at) {
          return std::array<view_type, num_children>{view_at(I)...};
        };
        constexpr auto children = std::make_index_sequence<num_children>{};
        /* step 1 */
        for (size_type c = 0; c < num_children; ++c) {
          if (pslot[c] < 0) continue;
          const key_type ck = key.child_at(c);
          launch_type l{ck, packed.slot_view(packed.find_slot(ck)), pscratch.slot_view(pslot[c]),
                        cres[c].coeffs().current_view(),
                        views(children, [&](size_type g) { return gsrc[c][g].view(); }),
                        N, csp[c].count_nonzero()};
          sumsq_at(csumsq[c], l);
          l.trunc_tol = tol_of(ck);
          if (may_truncate) {
            l.child_kept.assign(num_children * N, 0);
            l.kept_out_row = static_cast<int>(c);
          }
          member.step1.push_back(std::move(l));
          push_tensor(cres[c].coeffs());
          member.sparsity.push_back({const_cast<T*>(pscratch.buffer().device_ptr_on(device)) + pscratch.slot_offset(pslot[c]),
                                     pscratch.slot_sparsity_data(pslot[c]), N});
        }
        /* step 2, at the key */
        {
          /* views are constructed, then moved (which rebinds them) */
          member.step2 = launch_type{key, own.view(), p.coeffs().current_view(), d.current_view(),
                                     views(children, [&](size_type c) { return child[c].view(); }), N, n_nonzero};
          launch_type& l = member.step2;
          sumsq_at(0, l);
          l.trunc_tol = tol_of(key);
          if (may_truncate) {
            l.child_kept.assign(num_children * N, 0);
            for (size_type c = 0; c < num_children; ++c) {
              if (folded[c]) continue; // written by step 1
              for (size_type i = 0; i < N; ++i) l.child_kept[c*N + i] = child_kept(i, c) ? 1 : 0;
            }
          }
          push_tensor(p.coeffs());
          push_tensor(d);
        }
        auto batch = co_await ttg::device::coop<key_type>(key, member);
        detail::submit_compress_2level_batch_leader<T, NDIM>(batch, *ctx.compress_pool, K, ctx.is_ns,
                                                             hgT.current_view());
        if (may_truncate) {
          if (slot_results) co_await ttg::device::wait();
          else co_await ttg::device::wait(d_sumsq);
        }

        /* mirror the kernels' truncation decisions on the host, as compress_combined_task:
         * the folded children first (none of their children is kept), then the key */
        auto mirror = [&](cnode_type& res, const SparsityInfo& sp, size_type offset, auto&& any_child_kept) {
          if (!may_truncate) return;
          const T* sumsq = d_sumsq.host_ptr() + offset;
          const T tol = tol_of(res.key());
          size_type pos = 0;
          for (auto it = sp.begin_nonzero(); it != sp.end_nonzero(); ++it, ++pos) {
            const size_type i = *it;
            if (res.is_zero(i) || sumsq[pos] >= tol*tol) continue;
            if (!any_child_kept(i)) res.set_zero(i);
          }
        };
        for (size_type c = 0; c < num_children; ++c) {
          if (pslot[c] < 0) continue;
          mirror(cres[c], csp[c], csumsq[c], [](size_type) { return false; });
          ckept[c] = kept_info_type(N);
          for (size_type i = 0; i < N; ++i) ckept[c][i] = cres[c].is_zero(i) ? T(0) : T(1);
        }
        finish_child_leaf_info();
        mirror(result, sparsity, 0, [&](size_type i) {
          for (size_type c = 0; c < num_children; ++c) if (child_kept(i, c)) return true;
          return false;
        });
        kept = kept_info_type(N);
        for (size_type i = 0; i < N; ++i) kept[i] = result.is_zero(i) ? T(0) : T(1);
      }

      /* the folded children's compressed nodes, where kept for some function */
      auto sends = ttg::device::forward();
      for (size_type c = 0; c < num_children; ++c) {
        bool any = false;
        for (size_type i = 0; i < ckept[c].size() && !any; ++i) any = (ckept[c][i] != T(0));
        if (any) sends.push_back(ttg::device::send<out_terminal_id>(key.child_at(c), std::move(cres[c])));
      }
      if (key.level() > 0) {
        bool any_kept = false;
        for (size_type i = 0; i < kept.size() && !any_kept; ++i) any_kept = (kept[i] != T(0));
        constexpr auto seq = std::make_index_sequence<num_children>{};
        sends.push_back(select_send_up(key, std::move(p), seq, "compress_combined"));
        sends.push_back(select_send_up<num_children>(key, std::move(kept), seq, "compress_combined-kept"));
        if (any_kept) sends.push_back(ttg::device::send<out_terminal_id>(key, std::move(result)));
      } else {
        sends.push_back(ttg::device::send<out_terminal_id>(key, std::move(result)));
      }
      co_await std::move(sends);
    }
#endif // MRA_ENABLE_HOST
  } // namespace detail

  /**
   * Number of tree levels an even-level task of make_compress_combined compresses, 1 or 2
   * (see compress_combined_2level_task; batched only); the default is
   * MRA_TTG_COMPRESS_LEVELS, or 1 if unset.
   */
  inline void set_compress_levels(int levels) {
    detail::compress_levels_state().store(levels == 2 ? 2 : 1, std::memory_order_relaxed);
  }

  inline int get_compress_levels() {
    return detail::compress_levels_state().load(std::memory_order_relaxed);
  }

  /**
   * Compression of the product of make_multiply_compressed_combined: `in` its product
   * nodes at even levels, `packed` its packed outputs. Parameters and output as
   * make_compress; pass the maps the multiply was built with. The batches share the
   * batch pool of batch_group with every make_compress and make_compress_combined of it,
   * but a batch only holds tasks of one TT (the even or the odd compress tasks).
   */
  template <typename T, mra::Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_compress_combined(
    const std::shared_ptr<FunctionSetT>& fns,
    const std::size_t K,
    const bool is_ns,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>& in,
    ttg::Edge<mra::Key<NDIM>, mra::CombinedReconstructedNode<T, NDIM>>& packed,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
    const std::string name = "compress_combined",
    ProcMap procmap = {},
    DeviceMap devicemap = {},
    const bool enable_truncate = false,
    const T thresh = T{},
    const int truncate_mode = 0,
    const T cell_min_width = T{},
    const BatchGroup& batch_group = {})
  {
    static_assert(NDIM == 3, "make_compress_combined: NDIM=3 only");
    using key_type = mra::Key<NDIM>;
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;
    using combined_type = mra::CombinedReconstructedNode<T, NDIM>;
    using kept_info_type = mra::KeptInfo<T>;
    using source_type = detail::CompressCombinedSource<T, NDIM>;
    /* static: the lambdas below use it without capturing it */
    static constexpr size_type num_children = key_type::num_children();
    constexpr auto children = std::make_index_sequence<num_children>{};

    detail::CompressCombinedContext<T, NDIM, FunctionSetT> ctx;
    ctx.fns = fns;
    ctx.K = K;
    ctx.is_ns = is_ns;
    ctx.functiondata = &functiondata;
    ctx.name = name;
    ctx.enable_batching = mra::batching_enabled();
    ctx.total_functions = fns->num_functions();
    ctx.enable_truncate = enable_truncate;
    ctx.thresh = thresh;
    ctx.truncate_mode = truncate_mode;
    ctx.cell_min_width = cell_min_width;
    const std::size_t max_batch_size = mra::get_batch_size();
#ifndef MRA_ENABLE_HOST
    /* two levels: an even-level task also compresses the children that need nothing from
     * the tasks below (compress_folds_child), which then have no odd-level task */
    const bool two_level = ctx.enable_batching && get_compress_levels() == 2;
#else  // MRA_ENABLE_HOST
    constexpr bool two_level = false;
#endif // MRA_ENABLE_HOST
#ifndef MRA_ENABLE_HOST
    if (ctx.enable_batching) {
      using registry_t = detail::GroupedBatchPoolRegistry<detail::CompressBatchArg<T, NDIM>>;
      ctx.compress_pool = batch_group.pool<registry_t>([]{
        return detail::shared_pool_registry<detail::CompressBatchArg<T, NDIM>>(mra::get_batch_size());
      }, K, is_ns);
    }
#endif // MRA_ENABLE_HOST

    /* the compress tasks' terminals for their children: p and kept, at even and odd levels;
     * produced by the compress tasks a level below, the router and the dispatch task */
    std::array<ttg::Edge<key_type, node_type>, num_children> even_p, odd_p;
    std::array<ttg::Edge<key_type, kept_info_type>, num_children> even_kept, odd_kept;
    ttg::Edge<key_type, node_type> own_e(name + "-own");
    ttg::Edge<key_type, combined_type> even_packed_e(name + "-packed_even"), odd_packed_e(name + "-packed_odd");

    /**
     * Router, at an even key: a node where some function continues goes to the compress
     * task at its key, a leaf goes up with an empty "kept" (do_send_leafs_up).
     */
    auto route_fn = [](const key_type& key, const node_type& node) -> TASKTYPE {
      if (node.any_have_children()) {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<2*num_children>(key, node);
#else
        ttg::send<2*num_children>(key, node);
#endif
      } else if (key.level() > 0) {
        constexpr auto seq = std::make_index_sequence<num_children>{};
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(select_send_up(key, node, seq, "compress_combined-route"),
                                      select_send_up<num_children>(key, kept_info_type{}, seq,
                                                                   "compress_combined-route-kept"));
#else
        select_send_up(key, node, seq, "compress_combined-route");
        select_send_up<num_children>(key, kept_info_type{}, seq, "compress_combined-route-kept");
#endif
      }
    };
    auto route_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(route_fn), ttg::edges(in), ttg::edges(odd_p[I]..., odd_kept[I]..., own_e),
                                 name + "-route");
    }(children);

    /**
     * Dispatch, at the key K of a multiply task: packed goes to the compress task at K and
     * to those at the children of K where the product continues; for each slot that holds
     * a leaf, an empty node and an empty "kept" go to the terminals of its parent (K or a
     * child of K), which reads the leaf from packed.
     */
    auto dispatch_fn = [fns, two_level](const key_type& key, const combined_type& packed) -> TASKTYPE {
      constexpr std::size_t EVEN_P = 0, EVEN_KEPT = num_children, ODD_P = 2*num_children,
                            ODD_KEPT = 3*num_children, EVEN_PACKED = 4*num_children, ODD_PACKED = 4*num_children + 1;
      /* the parents to send placeholders to, by terminal: [0, num_children) the even
       * terminals (a leaf at a child of K), [num_children, 2*num_children) the odd ones (a
       * leaf at a grandchild); the same lists serve the p and the kept terminals */
      std::array<std::vector<key_type>, 2*num_children> parents;
      std::vector<key_type> odd_keys;
      bool any_leaf = false;
      /* with two levels, the children the compress task at K compresses itself: no odd-level
       * task, and placeholders on K's terminals of the child instead of on the child's */
      std::array<bool, num_children> folded{};
      if (two_level) {
        for (size_type c = 0; c < num_children; ++c) folded[c] = detail::compress_folds_child(packed, key.child_at(c));
      }
      for (size_type s = 0; s < packed.num_slots(); ++s) {
        const key_type& slot_key = packed.slot_key(s);
        const bool is_child = (slot_key.level() == key.level() + 1);
        if (is_child && folded[slot_key.childindex()]) {
          parents[slot_key.childindex()].push_back(key);
          any_leaf = true;
          continue;
        }
        if (!is_child && folded[slot_key.parent().childindex()]) continue; // read by the task at K
        if (detail::slot_any_inner(packed, s)) {
          if (is_child) odd_keys.push_back(slot_key); // the slot is the node of an odd-level compress task
          continue;
        }
        parents[(is_child ? 0 : num_children) + slot_key.childindex()].push_back(slot_key.parent());
        any_leaf = true;
      }
      /* the compress task reads a leaf child from packed, never its placeholder: one
       * placeholder node and one empty "kept" serve every receiver, each a single data
       * copy (a copy per send overflows the task's copies, MAX_PARAM_COUNT) */
      auto to_all = [&]<std::size_t EvenFirst, std::size_t OddFirst, std::size_t... I, typename Value>(
                        std::integral_constant<std::size_t, EvenFirst>, std::integral_constant<std::size_t, OddFirst>,
                        std::index_sequence<I...>, Value&& value) {
#ifndef MRA_ENABLE_HOST
        return ttg::device::broadcast<(EvenFirst + I)..., (OddFirst + I)...>(
                 std::make_tuple(parents[I]..., parents[num_children + I]...), std::forward<Value>(value));
#else  // MRA_ENABLE_HOST
        ttg::broadcast<(EvenFirst + I)..., (OddFirst + I)...>(
          std::make_tuple(parents[I]..., parents[num_children + I]...), std::forward<Value>(value));
#endif // MRA_ENABLE_HOST
      };
      constexpr auto seq = std::make_index_sequence<num_children>{};
      using even_p = std::integral_constant<std::size_t, EVEN_P>;
      using odd_p = std::integral_constant<std::size_t, ODD_P>;
      using even_kept = std::integral_constant<std::size_t, EVEN_KEPT>;
      using odd_kept = std::integral_constant<std::size_t, ODD_KEPT>;
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      if (any_leaf) {
        sends.push_back(to_all(even_p{}, odd_p{}, seq, node_type(key, fns->num_functions(key))));
        sends.push_back(to_all(even_kept{}, odd_kept{}, seq, kept_info_type{}));
      }
      /* one data copy for all receivers */
      if (odd_keys.empty()) {
        sends.push_back(ttg::device::send<EVEN_PACKED>(key, packed));
      } else {
        sends.push_back(ttg::device::broadcast<EVEN_PACKED, ODD_PACKED>(
                          std::make_tuple(std::vector<key_type>{key}, std::move(odd_keys)), packed));
      }
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      if (any_leaf) {
        to_all(even_p{}, odd_p{}, seq, node_type(key, fns->num_functions(key)));
        to_all(even_kept{}, odd_kept{}, seq, kept_info_type{});
      }
      if (odd_keys.empty()) {
        ttg::send<EVEN_PACKED>(key, packed);
      } else {
        ttg::broadcast<EVEN_PACKED, ODD_PACKED>(std::make_tuple(std::vector<key_type>{key}, std::move(odd_keys)),
                                                packed);
      }
#endif // MRA_ENABLE_HOST
    };
    auto dispatch_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(dispatch_fn), ttg::edges(packed),
                                 ttg::edges(even_p[I]..., even_kept[I]..., odd_p[I]..., odd_kept[I]...,
                                            even_packed_e, odd_packed_e),
                                 name + "-dispatch");
    }(children);

    /* the compress tasks: inputs p0..p7, kept0..kept7, then (even) the node and packed or
     * (odd) packed; outputs p and kept to the parent, and the compressed node */
    auto even_fn = [ctx, two_level](const key_type& key,
                         const node_type& p0, const node_type& p1, const node_type& p2, const node_type& p3,
                         const node_type& p4, const node_type& p5, const node_type& p6, const node_type& p7,
                         const kept_info_type& k0, const kept_info_type& k1, const kept_info_type& k2,
                         const kept_info_type& k3, const kept_info_type& k4, const kept_info_type& k5,
                         const kept_info_type& k6, const kept_info_type& k7,
                         const node_type& own, const combined_type& packed) -> TASKTYPE {
#ifndef MRA_ENABLE_HOST
      if (two_level) {
        return detail::compress_combined_2level_task<T, NDIM>(ctx, key, own, packed,
                                                              {&p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7},
                                                              {&k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7});
      }
#endif // MRA_ENABLE_HOST
      return detail::compress_combined_task<T, NDIM>(ctx, key, source_type::of_node(own), packed,
                                                     {&p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7},
                                                     {&k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7});
    };
    auto odd_fn = [ctx](const key_type& key,
                        const node_type& p0, const node_type& p1, const node_type& p2, const node_type& p3,
                        const node_type& p4, const node_type& p5, const node_type& p6, const node_type& p7,
                        const kept_info_type& k0, const kept_info_type& k1, const kept_info_type& k2,
                        const kept_info_type& k3, const kept_info_type& k4, const kept_info_type& k5,
                        const kept_info_type& k6, const kept_info_type& k7,
                        const combined_type& packed) -> TASKTYPE {
      const int s = packed.find_slot(key);
      if (s < 0) throw std::logic_error(ctx.name + ": no node for the key in the packed output");
      return detail::compress_combined_task<T, NDIM>(ctx, key, source_type::of_slot(packed, s), packed,
                                                     {&p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7},
                                                     {&k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7});
    };
    auto even_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(even_fn), ttg::edges(even_p[I]..., even_kept[I]..., own_e, even_packed_e),
                                 ttg::edges(odd_p[I]..., odd_kept[I]..., out), name + "-even");
    }(children);
    auto odd_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(odd_fn), ttg::edges(odd_p[I]..., odd_kept[I]..., odd_packed_e),
                                 ttg::edges(even_p[I]..., even_kept[I]..., out), name + "-odd");
    }(children);

    /* everything at an odd key goes where the multiply task at its parent went */
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      route_tt->set_keymap(procmap);
      dispatch_tt->set_keymap(procmap);
      even_tt->set_keymap(procmap);
      odd_tt->set_keymap([procmap](const key_type& key) { return procmap(key.parent()); });
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      route_tt->set_devicemap(devicemap);
      dispatch_tt->set_devicemap(devicemap);
      even_tt->set_devicemap(devicemap);
      odd_tt->set_devicemap([devicemap](const key_type& key) { return devicemap(key.parent()); });
    }
#ifndef MRA_ENABLE_HOST
    if (ctx.enable_batching) {
      even_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
      odd_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
      /* even_tt and odd_tt (and make_compress's TT) differ in type, so they cannot join
       * one ttg::device::batch_group: each batches its own tasks, all share the pool */
    }
#else  // MRA_ENABLE_HOST
    (void)max_batch_size;
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(route_tt->template in<0>(), dispatch_tt->template in<0>());
    auto outs = std::make_tuple(even_tt->template out<2*num_children>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops;
    ops.push_back(std::move(route_tt));
    ops.push_back(std::move(dispatch_tt));
    ops.push_back(std::move(even_tt));
    ops.push_back(std::move(odd_tt));
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_COMPRESS_COMBINED_H
