#ifndef MRA_TASKS_VMRA_H
#define MRA_TASKS_VMRA_H

#include <ttg.h>
#include <madness/mra/mra.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "mra/misc/batching.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/sparsityinfo.h"

/**
 * Tasks to load functions from MADNESS vmra into MRA FunctionNodes and feed them into TTG.
 */

namespace mra::vmra {

namespace detail {
  /**
   * Which children of a node are interior, per function: bit c of mask[fnid] is set if
   * child c (Key::childindex) is interior for function fnid. Held by the owner of the node.
   */
  struct ChildInterior {
    std::vector<std::uint32_t> mask;

    void set(std::size_t N, int child, const std::vector<std::uint8_t>& interior) {
      mask.resize(N, 0);
      for (std::size_t fnid = 0; fnid < N; ++fnid) {
        if (interior[fnid]) mask[fnid] |= std::uint32_t(1) << child;
      }
    }

    template <typename Archive>
    void serialize(Archive& ar) { ar & mask; }
  };
} // namespace detail

/**
 * Load a vector of MADNESS functions, aggregating nodes with the same key into a single
 * FunctionNode (Reconstructed or Compressed) and sending each node on the output edge.
 *
 * NodeT must be FunctionsReconstructedNode<T,NDIM> or FunctionsCompressedNode<T,NDIM>.
 * All functions must share the same process map (same owner per key) and the same K.
 *
 * The functions are loaded in the batches of `layout` (by default one batch of all): the node
 * of batch b at a key holds the functions layout->offset(b) ... offset(b) + batch_size(b) - 1.
 *
 * Three TTG tasks are created:
 *  - dispatch:  triggered by `control` at the root key of batch 0 (as make_start sends it; the
 *               roots of other batches are ignored), starts one enumerate task per batch on
 *               every process.
 *  - enumerate: enumerates the locally owned MADNESS nodes of the functions of its batch and
 *               sends their keys to the load task.
 *  - load:      for each key, builds the aggregated NodeT of the key's batch from the local
 *               MADNESS data.
 *
 * Compressed nodes mark which of their children are leafs. Children may live on another
 * process, so before the graph executes every process tells the owners of the parents of
 * its nodes which are interior (a MADNESS fence: collective, outside of graph execution).
 * The load task then reads only local data: blocking on a remote MADNESS lookup inside a
 * TTG task would execute other tasks nested in it.
 */
template<typename T, Dimension NDIM, typename NodeT>
auto make_vmra_load(const std::vector<madness::Function<T, (std::size_t)NDIM>>& vmra,
                    ttg::Edge<mra::Key<NDIM>, void>& control,
                    ttg::Edge<mra::Key<NDIM>, NodeT>& out,
                    const std::string& name = "vmra_load",
                    std::shared_ptr<const BatchLayout> layout = nullptr) {

  if (vmra.empty()) throw std::invalid_argument("make_vmra_load: empty function vector");
  if (!layout) layout = std::make_shared<const BatchLayout>(BatchLayout::even(vmra.size(), 1));
  if (layout->num_functions() != vmra.size())
    throw std::invalid_argument("make_vmra_load: the layout must have as many functions as vmra");

  /**
   * Sanity checks: support compressed, nonstandard, and reconstructed MADNESS trees.
   */
  bool ignore_leaf = false;
  if constexpr (std::is_same_v<NodeT, FunctionsCompressedNode<T, NDIM>>) {
    if (!((vmra.front().get_impl()->get_tree_state() == madness::TreeState::compressed) ||
          (vmra.front().get_impl()->get_tree_state() == madness::TreeState::nonstandard))) {
      throw std::invalid_argument("make_vmra_load: NodeT is FunctionsCompressedNode but vmra is not compressed");
    }
    ignore_leaf = true; // compressed nodes do not store leaf information
  } else {
    if (vmra.front().get_impl()->get_tree_state() != madness::TreeState::reconstructed) {
      throw std::invalid_argument("make_vmra_load: NodeT is FunctionsReconstructedNode but vmra is not reconstructed");
    }
  }

  ttg::Edge<int, void> dispatch_to_enumerate("dispatch_to_enumerate");
  ttg::Edge<mra::Key<NDIM>, void> dispatch_to_load("dispatch_to_load");

  /* All functions are assumed to share the same process map; use the first impl as keymap. */
  auto impl = vmra.front().get_impl();
  auto keymap = [impl](const mra::Key<NDIM>& key) {
    return impl->get_coeffs().owner(key.to_madness_key());
  };

  /* For compressed nodes: which children of each node are interior, held by the node's owner */
  using ChildMap = madness::WorldContainer<madness::Key<NDIM>, detail::ChildInterior>;
  std::shared_ptr<ChildMap> child_interior;
  if constexpr (std::is_same_v<NodeT, FunctionsCompressedNode<T, NDIM>>) {
    const size_type N = vmra.size();
    const size_type K = impl->get_k();
    madness::World& world = vmra.front().world();
    child_interior = std::make_shared<ChildMap>(world, impl->get_pmap());
    /* as in the load task: interior if it has children or holds wavelet coefficients */
    std::map<madness::Key<NDIM>, std::vector<std::uint8_t>> local;
    for (size_type fnid = 0; fnid < N; ++fnid) {
      const auto& coeffs = vmra[fnid].get_impl()->get_coeffs();
      for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
        const auto& mad_node = it->second;
        const bool interior = mad_node.has_children() ||
                              (mad_node.has_coeff() && mad_node.coeff().dim(0) == long(2*K));
        if (it->first.level() == 0 || !interior) continue;
        auto& flags = local[it->first];
        flags.resize(N, 0);
        flags[fnid] = 1;
      }
    }
    for (const auto& [mad_key, flags] : local) {
      std::array<Translation, NDIM> l;
      for (Dimension d = 0; d < NDIM; ++d) l[d] = mad_key.translation()[d];
      const int child = mra::Key<NDIM>(0, static_cast<Level>(mad_key.level()), l).childindex();
      child_interior->send(mad_key.parent(), &detail::ChildInterior::set, std::size_t(N), child, flags);
    }
    world.gop.fence();
  }

  /* Dispatch task: every process enumerates its own MADNESS nodes, one enumerate task
   * per (batch, process). */
  auto dispatch_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(
    [layout](const mra::Key<NDIM>& key) {
      if (key.batch() != 0) return;
      const int nproc = ttg::default_execution_context().size();
      for (Batch b = 0; b < layout->num_batches(); ++b) {
        for (int p = 0; p < nproc; ++p) ttg::sendk<0>(int(b*nproc + p));
      }
    }, ttg::edges(control), ttg::edges(dispatch_to_enumerate),
    name + "-dispatch", {"control"}, {"enumerate"});

  /* Enumerate task: iterate the locally owned MADNESS nodes of the functions of a batch and
   * forward each unique MRA key to the load task. */
  auto enumerate_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(
    [&vmra, ignore_leaf, layout](const int& id) {
      const Batch batch = id / ttg::default_execution_context().size();
      std::set<mra::Key<NDIM>> key_set;
      for (size_type l = 0; l < layout->batch_size(batch); ++l) {
        const auto& coeffs = vmra[layout->global_index(batch, l)].get_impl()->get_coeffs();
        for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
          std::array<Translation, NDIM> l;
          for (Dimension d = 0; d < NDIM; ++d) l[d] = it->first.translation()[d];
          key_set.insert(mra::Key<NDIM>(batch, static_cast<Level>(it->first.level()), l));
        }
      }
      for (const auto& k : key_set) {
        ttg::sendk<0>(k);
      }
    }, ttg::edges(dispatch_to_enumerate), ttg::edges(dispatch_to_load),
    name + "-enumerate", {"enumerate"}, {"dispatch"});
  enumerate_tt->set_keymap([](const int& id) {
    return id % ttg::default_execution_context().size();
  });

  /* Load task: for a given key, aggregate the coefficients of the functions of its batch into
   * one NodeT (function fnid of the node is function layout->global_index(batch, fnid)). */
  auto do_load_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(
    [&vmra, child_interior, layout](const mra::Key<NDIM>& key) {
      const Batch batch = key.batch();
      const size_type N = layout->batch_size(batch);
      auto fn = [&](size_type fnid) -> const auto& { return vmra[layout->global_index(batch, fnid)]; };
      const size_type K = vmra.front().get_impl()->get_k();
      const auto mad_key = key.to_madness_key();
      NodeT result(key, N);
      bool do_send = false;

      if constexpr (std::is_same_v<NodeT, FunctionsCompressedNode<T, NDIM>>) {
        /**
         * A node is interior for a function if it has children or holds wavelet coefficients
         * (2K per dimension), as in MADNESS's reconstruct_op: transform and truncate can leave
         * interior nodes without coefficients and leafs with them. Only interior nodes are sent;
         * MRA marks the leafs in their parent. The tree is taken from this rather than from the
         * coefficients, so that an interior node without coefficients is sent as a zero node
         * instead of being marked a leaf while its children are sent.
         */
        auto has_wavelets = [K](const auto& mad_node) {
          return mad_node.has_coeff() && mad_node.coeff().dim(0) == long(2*K);
        };
        /* which children are interior, from the pass before execution (local to this node) */
        std::vector<std::uint32_t> child_mask(N, 0);
        {
          auto acc = child_interior->find(mad_key);
          if (acc.get() != child_interior->end()) {
            const auto& mask = acc.get()->second.mask;   // of all functions
            for (size_type fnid = 0; fnid < N; ++fnid) child_mask[fnid] = mask[layout->global_index(batch, fnid)];
          }
        }

        std::vector<bool> interior(N, false);
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type fnid = 0; fnid < N; ++fnid) {
          const auto& coeffs = fn(fnid).get_impl()->get_coeffs();
          auto accessor = coeffs.find(mad_key);
          if (accessor.get() == coeffs.end()) continue;
          const auto& mad_node = accessor.get()->second;
          interior[fnid] = mad_node.has_children() || has_wavelets(mad_node);
          if (has_wavelets(mad_node)) sparsity.set_nonzero(fnid);
          do_send |= interior[fnid];
        }

        if (do_send) {
          result.allocate(sparsity, K);
          if (vmra.front().get_impl()->get_tree_state() == madness::TreeState::nonstandard) {
            result.set_ns(true);
          }
          for (size_type fnid = 0; fnid < N; ++fnid) {
            if (!interior[fnid]) continue;   // all children are leafs (the default)
            const auto& coeffs = fn(fnid).get_impl()->get_coeffs();
            if (sparsity.is_nonzero(fnid)) {
              const auto& mad_coeff = coeffs.find(mad_key).get()->second.coeff();
              /* Both MADNESS Tensor and MRA TensorView use contiguous row-major storage. */
              auto mra_subview = result.coeffs_view(fnid);
              std::copy_n(mad_coeff.ptr(), mad_coeff.size(), mra_subview.data());
            }
            /* MRA encodes leaf information in the *parent* node via set_child_leaf. */
            for (auto child : children(key)) {
              result.set_child_leaf(fnid, child, !(child_mask[fnid] & (std::uint32_t(1) << child.childindex())));
            }
          }
        }
      } else {
        /* Always send reconstructed nodes. */
        do_send = true;

        /* Determine sparsity: which functions have a leaf at this key. */
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type fnid = 0; fnid < N; ++fnid) {
          const auto& coeffs = fn(fnid).get_impl()->get_coeffs();
          auto accessor = coeffs.find(mad_key);
          if (accessor.get() != coeffs.end() && accessor.get()->second.is_leaf()) {
            sparsity.set_nonzero(fnid);
          }
        }

        if (sparsity.is_any_nonzero()) {
          result.allocate(sparsity, K);
          for (size_type fnid = 0; fnid < N; ++fnid) {
            if (sparsity.is_zero(fnid)) continue;
            const auto& mad_coeff = fn(fnid).get_impl()->get_coeffs().find(mad_key).get()->second.coeff();
            /* Both MADNESS Tensor and MRA TensorView use contiguous row-major storage. */
            auto mra_subview = result.coeffs_view(fnid);
            std::copy_n(mad_coeff.ptr(), mad_coeff.size(), mra_subview.data());
          }
        }
        /* The status of every function, also where no function has a leaf here and
         * the node carries no coefficients (statuses default to Inner): a function
         * without a node at this key ended above (Invalid). */
        for (size_type fnid = 0; fnid < N; ++fnid) {
          const auto& coeffs = fn(fnid).get_impl()->get_coeffs();
          auto accessor = coeffs.find(mad_key);
          if (accessor.get() == coeffs.end()) {
            result.set_leaf(fnid, LeafStatus::Invalid);
          } else if (accessor.get()->second.is_leaf()) {
            result.set_leaf(fnid, LeafStatus::Leaf);
          } else {
            result.set_leaf(fnid, LeafStatus::Inner);
          }
        }
      }
      if (do_send) {
        ttg::send<0>(key, std::move(result));
      }
    }, ttg::edges(dispatch_to_load), ttg::edges(out),
    name, {"dispatch"}, {"output"});

  do_load_tt->set_keymap(keymap);

  auto ins = std::make_tuple(dispatch_tt->template in<0>());
  auto outs = std::make_tuple(do_load_tt->template out<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops(3);
  ops[0] = std::move(dispatch_tt);
  ops[1] = std::move(enumerate_tt);
  ops[2] = std::move(do_load_tt);

  return make_ttg(std::move(ops), ins, outs, name);
}

namespace detail {
  /**
   * Helper to map a node type to a MADNESS TreeState.
   */
  template<typename NodeT>
  struct madfunc_state : std::conditional_t<std::is_same_v<NodeT, FunctionsCompressedNode<typename NodeT::value_type, NodeT::ndim()>>,
                     std::integral_constant<madness::TreeState, madness::TreeState::compressed>,
                     std::integral_constant<madness::TreeState, madness::TreeState::reconstructed>> {};

  template<typename NodeT>
  constexpr const madness::TreeState madfunc_state_v = madfunc_state<NodeT>::value;
} // namespace detail

/**
 * Store MRA FunctionNodes back into a vector of MADNESS functions.
 *
 * For each incoming node, the coefficients for each non-zero function are extracted
 * and stored into the corresponding MADNESS function's distributed coefficient table.
 *
 * For reconstructed nodes the has_children flag is derived from the MRA leaf status.
 * For compressed nodes has_children is always set to true; MADNESS leaf-level zero nodes
 * (marked via is_child_leaf in the parent) are not stored in MRA and must be separately
 * inserted into MADNESS if a fully populated tree is required.
 *
 * \param vmra Vector of MADNESS functions to store into.
 * \param in Input edge carrying MRA FunctionNodes.
 * \param add_leaves If true, add empty leaves for compressed nodes. MRA does not store
 *                   them but MADNESS sometimes does.
 * \param tree_state The MADNESS TreeState to set for each function. Defaults to the state
 *                   derived from the node type.
 * \param name Name of the TTG task.
 * \param layout The batches of the nodes (by default one batch of all functions): function
 *               fnid of a node of batch b is stored into vmra[layout->global_index(b, fnid)].
 */
template<typename T, Dimension NDIM, typename NodeT>
auto make_vmra_store(std::vector<madness::Function<T, (std::size_t)NDIM>>& vmra,
                     ttg::Edge<mra::Key<NDIM>, NodeT>& in,
                     bool add_leaves = false,
                     madness::TreeState tree_state = detail::madfunc_state_v<NodeT>,
                     const std::string& name = "vmra_store",
                     std::shared_ptr<const BatchLayout> layout = nullptr) {

  if (vmra.empty()) throw std::invalid_argument("make_vmra_store: empty function vector");
  if (!layout) layout = std::make_shared<const BatchLayout>(BatchLayout::even(vmra.size(), 1));
  if (layout->num_functions() != vmra.size())
    throw std::invalid_argument("make_vmra_store: the layout must have as many functions as vmra");

  auto impl = vmra.front().get_impl();
  auto keymap = [impl](const mra::Key<NDIM>& key) {
    return impl->get_coeffs().owner(key.to_madness_key());
  };

  for (auto& fn : vmra) {
    fn.get_impl()->set_tree_state(tree_state);
  }

  auto store_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(
    [&vmra, add_leaves, layout](const mra::Key<NDIM>& key, const NodeT& node) {
      const auto mad_key = key.to_madness_key();
      const size_type N   = layout->batch_size(key.batch());
      const size_type K   = vmra.front().get_impl()->get_k();

      /* Coefficient tensor side-length: K for reconstructed, 2K for compressed. */
      const size_type dim_size = std::is_same_v<NodeT, FunctionsCompressedNode<T, NDIM>>
                                 ? 2 * K : K;
      const std::vector<long> dims(NDIM, static_cast<long>(dim_size));

      using coeffT = madness::GenTensor<T>;
      using nodeT  = madness::FunctionNode<T, NDIM>;

      for (size_type fnid = 0; fnid < N; ++fnid) {

        // TODO: not if we can skip empty nodes
        //if (node.is_zero(fnid)) continue;

        auto fn_impl = vmra[layout->global_index(key.batch(), fnid)].get_impl();

        /* Copy MRA coefficients (contiguous row-major) into a MADNESS Tensor. */
        madness::Tensor<T> mad_tensor;
        bool has_children;
        if constexpr (std::is_same_v<NodeT, FunctionsReconstructedNode<T, NDIM>>) {
          has_children = !node.is_leaf(fnid);
          if (node.is_leaf(fnid)) {
            assert(!node.is_zero(fnid));
            mad_tensor = madness::Tensor<T>(dims);
            auto mra_subview = node.coeffs_view(fnid);
            std::copy_n(mra_subview.data(), mad_tensor.size(), mad_tensor.ptr());
          } else if (node.is_invalid(fnid)) {
            continue; // do not store invalid nodes in MADNESS
          }

        } else {
          /* In compressed form, wavelet coefficients live on interior nodes: a node is
           * interior for this function if it holds coefficients or has children. A
           * zero node without children is a leaf of this function, which MRA does not
           * store; its parent records it via is_child_leaf (and adds it as an empty
           * leaf with add_leaves). Writing it here as well would race with that. */
          has_children = !node.is_zero(fnid) || !node.is_all_child_leaf(fnid);
          if (!has_children && key.level() > 0) continue;   // except the root of a zero function
          if (!node.is_zero(fnid)) {
            mad_tensor = madness::Tensor<T>(dims);
            auto mra_subview = node.coeffs_view(fnid);
            std::copy_n(mra_subview.data(), mad_tensor.size(), mad_tensor.ptr());
          }
        }

        fn_impl->get_coeffs().replace(
            mad_key,
            nodeT(coeffT(mad_tensor, fn_impl->get_tensor_args()), has_children));

        /**
         * MADNESS stores the empty leaf nodes in compressed form, but MRA does not.
         * Instantiate them explicitly.
         */
        if constexpr (std::is_same_v<NodeT, FunctionsCompressedNode<T, NDIM>>) {
          if (add_leaves && has_children) {
            for (auto child : children(key)) {
              if (node.is_child_leaf(fnid, child)) {
                const auto mad_child_key = child.to_madness_key();
                fn_impl->get_coeffs().replace(
                    mad_child_key,
                    nodeT(coeffT(), false));
              }
            }
          }
        }
      }
    }, ttg::edges(in), ttg::edges(), name, {"input"}, {});

  store_tt->set_keymap(keymap);
  return store_tt;
}

} // namespace mra::vmra

#endif // MRA_TASKS_VMRA_H
