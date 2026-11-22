#ifndef MRA_TASKS_NONSTANDARD_2LEVEL_H
#define MRA_TASKS_NONSTANDARD_2LEVEL_H

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <ttg.h>

#include "mra/kernels.h"
#include "mra/misc/batch_size.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/two_level.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/combinednode.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  namespace detail {
    inline std::atomic<int>& nonstandard_levels_state() {
      static std::atomic<int> value{[] {
        const char* env = std::getenv("MRA_TTG_NONSTANDARD_LEVELS");
        return (env != nullptr && std::atoi(env) == 2) ? 2 : 1;
      }()};
      return value;
    }
  } // namespace detail

  /**
   * Number of tree levels a task of the non-standard conversion handles, 1
   * (make_nonstandard) or 2 (make_nonstandard_2level); the default is
   * MRA_TTG_NONSTANDARD_LEVELS, or 1 if unset.
   */
  inline void set_nonstandard_levels(int levels) {
    detail::nonstandard_levels_state().store(levels == 2 ? 2 : 1, std::memory_order_relaxed);
  }

  inline int get_nonstandard_levels() {
    return detail::nonstandard_levels_state().load(std::memory_order_relaxed);
  }

  /**
   * make_nonstandard with tasks that handle two levels: the task at a key unfilters [s|d] at
   * the key into the scaling coefficients of its children (step 1) and, for every child
   * where some function continues, [s|d] there into those of the grandchildren (step 2),
   * emitting the non-standard blocks at the key and at those children. Same output as
   * make_nonstandard; NDIM=3 only.
   *
   * Where the tasks are depends on the maps (TwoLevelPlan): a key and its children are
   * handled by one task only if they map to the same process and device, so a task never
   * pulls nodes across processes or devices. Elsewhere (above the level where the maps
   * distribute the tree, and along their cuts) a task handles its key only, as
   * make_nonstandard does, and its children are the keys of the next tasks.
   *
   * - The router sends each compressed node to the task that handles it: to the task at
   *   its key (terminal 0), or to the task at its parent as the node of its child (terminal
   *   1 + child index). A task takes nodes for all 8 children, so the router of a node with
   *   a task sends empty nodes for the children without a node, and for all 8 children if
   *   the task handles one level.
   * - The scaling coefficients at the keys of the next tasks travel in one
   *   CombinedReconstructedNode (`down`), broadcast to all of them; the task at a key reads
   *   its slot. Those at the children between the two steps are in a second one (`scratch`),
   *   which stays with the task. The root's come from the primer.
   *
   * Both steps run the reconstruct kernel: unbatched, one launch for step 1 and one per
   * child for step 2 on the task's stream; batched, a batch runs two launches of
   * reconstruct_kernel_batched, step 1 of every task, then step 2 of every (task, child).
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_nonstandard_2level(
    const std::shared_ptr<FunctionSetT>& fns,
    const size_type K,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
    const std::string& name = "nonstandard",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    static_assert(NDIM == 3, "make_nonstandard_2level: NDIM=3 only");
    using key_type = mra::Key<NDIM>;
    using cnode_type = mra::FunctionsCompressedNode<T, NDIM>;
    using combined_type = mra::CombinedReconstructedNode<T, NDIM>;
    using view_type = SparseTensorView<T, NDIM+1>;
    using plan_type = TwoLevelPlan<NDIM, ProcMap, DeviceMap>;
    using Role = typename plan_type::Role;
    using launch_type = detail::Reconstruct2Launch<T, NDIM>;
    /* static: the lambdas below use them without capturing them */
    static constexpr size_type num_children = key_type::num_children();
    static constexpr size_type num_terminals = num_children + 1; // the node and its children
    constexpr const int OUT = 0, S = 1;

    const plan_type plan{procmap, devicemap};

    std::array<ttg::Edge<key_type, cnode_type>, num_terminals> route_e;
    ttg::Edge<key_type, combined_type> s_e(name + "-s"), primer_e(name + "-primer");

    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::ReconstructBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = detail::shared_pool_registry<detail::ReconstructBatchArg<T, NDIM>>(max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    /**
     * Router: a node goes to the task that handles it. The router of a node with a task
     * also completes the task's child terminals with empty nodes.
     */
    auto route_fn = [fns, plan](const key_type& key, const cnode_type& node) -> TASKTYPE {
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
      const Role role = plan.role(key);
      if (role == Role::Covered) {
        send_to(terminals, 1 + key.childindex(), key.parent(), node);
      } else {
        send_to(terminals, 0, key, node);
        for (size_type c = 0; c < num_children; ++c) {
          bool has_node = false;
          if (role == Role::Two) {
            for (size_type i = 0; i < node.count() && !has_node; ++i) has_node = !node.is_child_leaf(i, c);
          }
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
    auto route_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(route_fn), ttg::edges(in), ttg::edges(route_e[I]...), name + "-route");
    }(std::make_index_sequence<num_terminals>{});

    /** the scaling coefficients at the root: none, every function continues */
    auto primer_fn = [fns, K](const key_type& key, const cnode_type&) -> TASKTYPE {
      if (key.level() == 0) {
        const size_type N = fns->num_functions(key);
        combined_type s(key, K);
        s.add_slot(key, SparsityInfo(N, SparsityInfo::InitType::AllZero), std::vector<LeafStatus>(N, LeafStatus::Inner));
        s.allocate(ttg::scope::SyncIn);
        s.populate_device_sparsity(ttg::device::Device::host());
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, std::move(s));
#else  // MRA_ENABLE_HOST
        ttg::send<0>(key, std::move(s));
#endif // MRA_ENABLE_HOST
      }
    };
    auto primer_tt = ttg::make_tt<Space>(std::move(primer_fn), ttg::edges(in), ttg::edges(primer_e), name + "-primer");

    auto task_fn = [&functiondata, fns, K, name, plan, enable_batching, batch_pool](
                     const key_type& key, const cnode_type& node,
                     const cnode_type& n0, const cnode_type& n1, const cnode_type& n2, const cnode_type& n3,
                     const cnode_type& n4, const cnode_type& n5, const cnode_type& n6, const cnode_type& n7,
                     const combined_type& s_in) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      const std::array<const cnode_type*, num_children> child_node = {&n0, &n1, &n2, &n3, &n4, &n5, &n6, &n7};
      const bool two = plan.role(key) == Role::Two;
      const int s_slot = s_in.find_slot(key);
      if (s_slot < 0) throw std::logic_error(name + ": no scaling coefficients for the key");
      const detail::SparsityState* s_sparsity = s_in.slot_sparsity_data(s_slot);
      auto any_inner = [](const std::vector<LeafStatus>& st) {
        for (auto x : st) if (x == LeafStatus::Inner) return true;
        return false;
      };
      /* the status of the functions at the children of a node, from theirs at the node */
      auto child_status = [N](const std::vector<LeafStatus>& st, const cnode_type& n, size_type c) {
        std::vector<LeafStatus> res(N);
        for (size_type i = 0; i < N; ++i) {
          res[i] = (st[i] != LeafStatus::Inner) ? LeafStatus::Invalid
                 : n.is_child_leaf(i, c)         ? LeafStatus::Leaf
                                                 : LeafStatus::Inner;
        }
        return res;
      };
      /* the scaling coefficients of the functions that continue, which are not leaves */
      auto inner_sparsity = [N](const std::vector<LeafStatus>& st) {
        SparsityInfo sp(N, SparsityInfo::InitType::AllZero);
        for (size_type i = 0; i < N; ++i) if (st[i] == LeafStatus::Inner) sp.set_nonzero(i);
        return sp;
      };
      /* the non-standard block at a node: the functions internal there that have coefficients */
      auto make_ns = [N, K](const key_type& k, const std::vector<LeafStatus>& st, const cnode_type& n,
                            auto&& from_parent_nonzero,
                            const std::array<std::vector<LeafStatus>, num_children>& cst) {
        cnode_type ns(k, N);
        SparsityInfo sp(N, SparsityInfo::InitType::AllZero);
        for (size_type i = 0; i < N; ++i) {
          if (st[i] == LeafStatus::Inner && (!n.is_zero(i) || from_parent_nonzero(i))) sp.set_nonzero(i);
        }
        ns.allocate(sp, K, ttg::scope::Allocate);
        ns.set_ns(true);
        for (size_type i = 0; i < N; ++i) {
          for (size_type c = 0; c < num_children; ++c) {
            ns.set_child_leaf(i, c, ns.is_zero(i) || cst[c][i] != LeafStatus::Inner);
          }
        }
        return ns;
      };

      /* step 1 at the key */
      const std::vector<LeafStatus> st0 = s_in.slot_statuses(s_slot);
      std::array<std::vector<LeafStatus>, num_children> cst;
      for (size_type c = 0; c < num_children; ++c) cst[c] = child_status(st0, node, c);
      cnode_type ns0 = make_ns(key, st0, node,
                               [&](size_type i) { return bool(s_sparsity[i] & detail::SparsityState::NONZERO); }, cst);

      /* step 2 at the children where a function continues (two levels only) */
      std::array<bool, num_children> step2{};
      std::array<std::array<std::vector<LeafStatus>, num_children>, num_children> gst;
      std::array<cnode_type, num_children> ns_child;
      for (size_type c = 0; c < num_children; ++c) {
        if (!two || !any_inner(cst[c])) continue;
        step2[c] = true;
        for (size_type g = 0; g < num_children; ++g) gst[c][g] = child_status(cst[c], *child_node[c], g);
        ns_child[c] = make_ns(key.child_at(c), cst[c], *child_node[c],
                              [&](size_type i) { return cst[c][i] == LeafStatus::Inner; }, gst[c]);
      }

      /* the scaling coefficients written: those for the next tasks in down, those at the
       * children between the steps in scratch */
      combined_type down(key, K), scratch(key, K);
      std::array<int, num_children> r_slot;  // of child c, in scratch (two levels) or down
      std::array<std::array<int, num_children>, num_children> g_slot; // of grandchild g of child c, in down
      r_slot.fill(-1);
      for (auto& g : g_slot) g.fill(-1);
      for (size_type c = 0; c < num_children; ++c) {
        if (!any_inner(cst[c])) continue;
        r_slot[c] = (two ? scratch : down).add_slot(key.child_at(c), inner_sparsity(cst[c]), cst[c]);
        if (!step2[c]) continue;
        for (size_type g = 0; g < num_children; ++g) {
          if (!any_inner(gst[c][g])) continue;
          g_slot[c][g] = down.add_slot(key.child_at(c).child_at(g), inner_sparsity(gst[c][g]), gst[c][g]);
        }
      }
      if (down.num_slots() > 0) down.allocate();
      if (scratch.num_slots() > 0) scratch.allocate();

      /* the work items of each launch: the union of the sparsity of its inputs and outputs */
      auto count_work = [N](auto&& is_nonzero) {
        size_type n = 0;
        for (size_type i = 0; i < N; ++i) if (is_nonzero(i)) ++n;
        return n;
      };
      const size_type n_work0 = count_work([&](size_type i) {
        if (!node.is_zero(i) || (s_sparsity[i] & detail::SparsityState::NONZERO)) return true;
        for (size_type c = 0; c < num_children; ++c) {
          if (r_slot[c] >= 0 && cst[c][i] == LeafStatus::Inner) return true;
        }
        return false;
      });
      std::array<size_type, num_children> n_work{};
      size_type max_work = n_work0;
      for (size_type c = 0; c < num_children; ++c) {
        if (!step2[c]) continue;
        n_work[c] = count_work([&](size_type i) {
          if (!child_node[c]->is_zero(i) || cst[c][i] == LeafStatus::Inner) return true;
          for (size_type g = 0; g < num_children; ++g) {
            if (g_slot[c][g] >= 0 && gst[c][g][i] == LeafStatus::Inner) return true;
          }
          return false;
        });
        max_work = std::max(max_work, n_work[c]);
      }

#ifndef MRA_ENABLE_HOST
      const bool batched = enable_batching;
#else  // MRA_ENABLE_HOST
      constexpr bool batched = false;
#endif // MRA_ENABLE_HOST
      const auto& hg = functiondata.get_hg();
      /* unbatched, the launches run one after the other on the task's stream and share tmp */
      ttg::Buffer<T, DeviceAllocator<T>> tmp;
      if (!batched && max_work > 0) {
        tmp = ttg::Buffer<T, DeviceAllocator<T>>(reconstruct_tmp_size<NDIM>(K)*max_work, ttg::scope::Allocate);
      }

#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(hg.buffer());
      if (!tmp.empty()) input.add(tmp);
      if (!node.empty()) input.add(node.coeffs().buffer());
      for (size_type c = 0; c < num_children; ++c) {
        if (step2[c] && !child_node[c]->empty()) input.add(child_node[c]->coeffs().buffer());
      }
      input.add(s_in.buffer());
      if (!ns0.empty()) input.add(ns0.coeffs().buffer());
      for (size_type c = 0; c < num_children; ++c) {
        if (step2[c] && !ns_child[c].empty()) input.add(ns_child[c].coeffs().buffer());
      }
      if (!down.empty()) input.add(down.buffer());
      if (!scratch.empty()) input.add(scratch.buffer());
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

      /* views are constructed, never assigned: assigning a view copies the values */
      const view_type none{};
      auto views = [&]<std::size_t... I>(std::index_sequence<I...>, auto&& view_at) {
        return std::array<view_type, num_children>{view_at(I)...};
      };
      constexpr auto children = std::make_index_sequence<num_children>{};
      std::vector<launch_type> launches;
      launches.push_back(launch_type{key, node.coeffs().current_view(), s_in.slot_view(s_slot),
                                     views(children, [&](size_type c) {
                                       if (r_slot[c] < 0) return none;
                                       return two ? scratch.slot_view(r_slot[c]) : down.slot_view(r_slot[c]);
                                     }),
                                     ns0.coeffs().current_view(), N, n_work0});
      for (size_type c = 0; c < num_children; ++c) {
        if (!step2[c]) continue;
        launches.push_back(launch_type{key.child_at(c), child_node[c]->coeffs().current_view(),
                                       scratch.slot_view(r_slot[c]),
                                       views(children, [&](size_type g) {
                                         if (g_slot[c][g] < 0) return none;
                                         return down.slot_view(g_slot[c][g]);
                                       }),
                                       ns_child[c].coeffs().current_view(), N, n_work[c]});
      }

#ifndef MRA_ENABLE_HOST
      if (batched) {
        /* the sparsity of the outputs is pushed by the leader along with the batch */
        const auto device = ttg::device::current_device();
        detail::Reconstruct2BatchMember<T, NDIM> member{std::move(launches), {}};
        std::vector<std::vector<detail::SparsityState>> ns_bytes;
        ns_bytes.reserve(num_terminals);
        auto push_ns = [&](cnode_type& ns) {
          if (ns.empty()) return;
          auto& bytes = ns_bytes.emplace_back(N);
          detail::sparsity_to_bytes(ns.coeffs().sparsity(), bytes.data(), N);
          member.sparsity.push_back({const_cast<T*>(ns.coeffs().buffer().device_ptr_on(device)), bytes.data(), N});
        };
        push_ns(ns0);
        for (size_type c = 0; c < num_children; ++c) if (step2[c]) push_ns(ns_child[c]);
        for (combined_type* cn : {&down, &scratch}) {
          if (cn->empty()) continue;
          T* base = const_cast<T*>(cn->buffer().device_ptr_on(device));
          for (size_type s = 0; s < cn->num_slots(); ++s) {
            member.sparsity.push_back({base + cn->slot_offset(s), cn->slot_sparsity_data(s), cn->slot_count(s)});
          }
        }
        auto batch = co_await ttg::device::coop<key_type>(key, member);
        detail::submit_nonstandard_2level_batch_leader<T, NDIM>(batch, *batch_pool, K, hg.current_view());
      } else
#endif // MRA_ENABLE_HOST
      {
        {
          auto sparseman = make_sparsity_manager(ns0);
          sparseman.populate_device_sparsity();
        }
        for (size_type c = 0; c < num_children; ++c) {
          if (!step2[c]) continue;
          auto sparseman = make_sparsity_manager(ns_child[c]);
          sparseman.populate_device_sparsity();
        }
        if (!down.empty()) down.populate_device_sparsity();
        if (!scratch.empty()) scratch.populate_device_sparsity();
        view_type result_view{};
        for (auto& l : launches) {
          if (l.n_nonzero == 0) continue;
          submit_reconstruct_kernel<T, NDIM>(l.key, l.n, l.n_nonzero, K, false, l.node, hg.current_view(), l.from_parent,
                                             l.r_arr, result_view, l.ns, tmp.current_device_ptr(),
                                             ttg::device::current_stream());
        }
      }

      /* the non-standard blocks, and the scaling coefficients for the next tasks */
      std::vector<key_type> down_keys;
      for (size_type s = 0; s < down.num_slots(); ++s) down_keys.push_back(down.slot_key(s));
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      sends.push_back(ttg::device::send<OUT>(key, std::move(ns0)));
      for (size_type c = 0; c < num_children; ++c) {
        if (step2[c]) sends.push_back(ttg::device::send<OUT>(key.child_at(c), std::move(ns_child[c])));
      }
      if (!down_keys.empty()) sends.push_back(ttg::device::broadcast<S>(std::move(down_keys), std::move(down)));
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::send<OUT>(key, std::move(ns0));
      for (size_type c = 0; c < num_children; ++c) {
        if (step2[c]) ttg::send<OUT>(key.child_at(c), std::move(ns_child[c]));
      }
      if (!down_keys.empty()) ttg::broadcast<S>(std::move(down_keys), std::move(down));
#endif // MRA_ENABLE_HOST
    };

    auto task_tt = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ttg::make_tt<Space>(std::move(task_fn), ttg::edges(route_e[I]..., ttg::fuse(s_e, primer_e)),
                                 ttg::edges(out, s_e), name);
    }(std::make_index_sequence<num_terminals>{});

    /* routers go where the task that handles their node goes */
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      route_tt->set_keymap([plan](const key_type& key) { return plan.procmap(plan.task_of(key)); });
      primer_tt->set_keymap(procmap);
      task_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      route_tt->set_devicemap([plan](const key_type& key) { return plan.devicemap(plan.task_of(key)); });
      primer_tt->set_devicemap(devicemap);
      task_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      task_tt->set_batch_matcher([](const key_type&, const key_type&) { return true; }, max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(route_tt->template in<0>());
    auto outs = std::make_tuple(task_tt->template out<OUT>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops;
    ops.push_back(std::move(primer_tt));
    ops.push_back(std::move(route_tt));
    ops.push_back(std::move(task_tt));
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_NONSTANDARD_2LEVEL_H
