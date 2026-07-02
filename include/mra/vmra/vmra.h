#ifndef MRA_TASKS_VMRA_H
#define MRA_TASKS_VMRA_H

#include <ttg.h>
#include <madness/mra/mra.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <set>
#include <stdexcept>
#include <vector>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/sparsityinfo.h"

/**
 * Tasks to load functions from MADNESS vmra into MRA FunctionNodes and feed them into TTG.
 */

namespace mra::vmra {

/**
 * Load a vector of MADNESS functions, aggregating nodes with the same key into a single
 * FunctionNode (Reconstructed or Compressed) and sending each node on the output edge.
 *
 * NodeT must be FunctionsReconstructedNode<T,NDIM> or FunctionsCompressedNode<T,NDIM>.
 * All functions must share the same process map (same owner per key) and the same K.
 *
 * Two TTG tasks are created:
 *  - dispatch: triggered by `control`, enumerates all locally owned MADNESS nodes across
 *              all functions and sends their keys to the load task.
 *  - load:     for each key, builds the aggregated NodeT from the local MADNESS data.
 *
 * For distributed operation the caller must trigger `control` once per MPI rank so that
 * each rank enumerates its own local MADNESS nodes.
 */
template<typename T, Dimension NDIM, typename NodeT>
auto make_vmra_load(const std::vector<madness::Function<T, (std::size_t)NDIM>>& vmra,
                    ttg::Edge<mra::Key<NDIM>, void>& control,
                    ttg::Edge<mra::Key<NDIM>, NodeT>& out,
                    const std::string& name = "vmra_load") {

  if (vmra.empty()) throw std::invalid_argument("make_vmra_load: empty function vector");

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

  ttg::Edge<mra::Key<NDIM>, void> dispatch_to_load("dispatch_to_load");

  /* All functions are assumed to share the same process map; use the first impl as keymap. */
  auto impl = vmra.front().get_impl();
  auto keymap = [impl](const mra::Key<NDIM>& key) {
    return impl->get_coeffs().owner(key.to_madness_key());
  };

  /* Dispatch task: iterate all locally owned MADNESS nodes across all functions and
   * forward each unique MRA key to the load task. */
  auto dispatch_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(
    [&vmra, ignore_leaf](const mra::Key<NDIM>& key) {
      const Batch batch = key.batch();
      std::set<mra::Key<NDIM>> key_set;
      for (const auto& fn : vmra) {
        const auto& coeffs = fn.get_impl()->get_coeffs();
        for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
          std::array<Translation, NDIM> l;
          for (Dimension d = 0; d < NDIM; ++d) l[d] = it->first.translation()[d];
          key_set.insert(mra::Key<NDIM>(batch, static_cast<Level>(it->first.level()), l));
        }
      }
      for (const auto& k : key_set) {
        ttg::sendk<0>(k);
      }
    }, ttg::edges(control), ttg::edges(dispatch_to_load),
    name + "-dispatch", {"control"}, {"dispatch"});

  /* Load task: for a given key, aggregate coefficients from all functions into one NodeT. */
  auto do_load_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(
    [&vmra](const mra::Key<NDIM>& key) {
      const Batch batch = key.batch();
      const size_type N = vmra.size();
      const size_type K = vmra.front().get_impl()->get_k();
      /**
       * Always send reconstructed nodes. We don't send compressed nodes if they are leafs.
       * MADNESS stores them explicitly, MRA does not.
       */
      bool do_send = std::is_same_v<NodeT, FunctionsReconstructedNode<T, NDIM>>;
      const auto mad_key = key.to_madness_key();

      /* Determine sparsity: which functions have a non-empty node at this key. */
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      for (size_type fnid = 0; fnid < N; ++fnid) {
        const auto& coeffs = vmra[fnid].get_impl()->get_coeffs();
        auto accessor = coeffs.find(mad_key);
        if (accessor.get() != coeffs.end()) {
          if (accessor.get()->second.coeff().size() > 0) {
            sparsity.set_nonzero(fnid);
          }
          if constexpr (std::is_same_v<NodeT, FunctionsCompressedNode<T, NDIM>>) {
            if (!accessor.get()->second.is_leaf()) {
              do_send = true;
            }
          }
        }
      }

      NodeT result(key, sparsity, K);

      for (size_type fnid = 0; fnid < N; ++fnid) {
        const auto& coeffs = vmra[fnid].get_impl()->get_coeffs();
        auto accessor = coeffs.find(mad_key);

        if (sparsity.is_zero(fnid)) {
          /**
           * Handle leaf information for reconstructed nodes.
           */
          if constexpr (std::is_same_v<NodeT, FunctionsReconstructedNode<T, NDIM>>) {
            if (accessor.get() == coeffs.end()) {
              result.set_leaf(fnid, LeafStatus::Invalid);
            } else if (accessor.get()->second.is_leaf()) {
              result.set_leaf(fnid, LeafStatus::Leaf);
            } else {
              result.set_leaf(fnid, LeafStatus::Inner);
            }
          }
          continue;
        }

        assert(accessor.get() != coeffs.end());
        const auto& mad_node = accessor.get()->second;
        const auto& mad_coeff = mad_node.coeff();

        /* Both MADNESS Tensor and MRA TensorView use contiguous row-major storage. */
        auto mra_subview = result.coeffs_view(fnid);
        std::copy_n(mad_coeff.ptr(), mad_coeff.size(), mra_subview.data());

        if constexpr (std::is_same_v<NodeT, FunctionsCompressedNode<T, NDIM>>) {
          /* MRA encodes leaf information in the *parent* node via set_child_leaf.
           * Iterate the 2^NDIM children of the current key and check whether each
           * child is a MADNESS leaf node. */
          if (vmra.front().get_impl()->get_tree_state() == madness::TreeState::nonstandard) {
            result.set_ns(true);
          }
          result.set_all_child_leaf(fnid, false);
          for (auto child : children(key)) {
            const madness::Key<NDIM> mad_child_key = child.to_madness_key();
            auto child_acc = coeffs.find(mad_child_key);
            if (child_acc.get() != coeffs.end() && child_acc.get()->second.is_leaf()) {
              std::cout << "LOAD " << key << " setting child " << child << " leaf for fnid " << fnid << std::endl;
              result.set_child_leaf(fnid, child, true);
              assert(child_acc.get()->second.coeff().size() == 0);
            }
          }
        } else if constexpr (std::is_same_v<NodeT, FunctionsReconstructedNode<T, NDIM>>) {
          /**
           * Reconstructed nodes encode leaf information in the node.
           */
          if (mad_node.is_leaf()) {
            result.set_leaf(fnid);
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
  std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
  ops[0] = std::move(dispatch_tt);
  ops[1] = std::move(do_load_tt);

  return make_ttg(std::move(ops), ins, outs, name);
}


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
 * TODO: do we store the empty leaf nodes in MADNESS compressed form?
 */
template<typename T, Dimension NDIM, typename NodeT>
auto make_vmra_store(std::vector<madness::Function<T, (std::size_t)NDIM>>& vmra,
                     ttg::Edge<mra::Key<NDIM>, NodeT>& in,
                     const std::string& name = "vmra_store") {

  if (vmra.empty()) throw std::invalid_argument("make_vmra_store: empty function vector");

  auto impl = vmra.front().get_impl();
  auto keymap = [impl](const mra::Key<NDIM>& key) {
    return impl->get_coeffs().owner(key.to_madness_key());
  };

  auto store_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(
    [&vmra](const mra::Key<NDIM>& key, const NodeT& node) {
      const auto mad_key = key.to_madness_key();
      const size_type N   = vmra.size();
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

        auto fn_impl = vmra[fnid].get_impl();

        /* Copy MRA coefficients (contiguous row-major) into a MADNESS Tensor. */
        madness::Tensor<T> mad_tensor(dims);
        if (!node.is_zero(fnid)) {
          auto mra_subview = node.coeffs_view(fnid);
          std::copy_n(mra_subview.data(), mad_tensor.size(), mad_tensor.ptr());
        }

        bool has_children;
        if constexpr (std::is_same_v<NodeT, FunctionsReconstructedNode<T, NDIM>>) {
          has_children = !node.is_leaf(fnid);
        } else {
          /* In MRA compressed form, leaf-level zero nodes are not stored; the
           * parent records them via is_child_leaf.  MADNESS requires has_children
           * to reflect the actual tree structure, so we conservatively set it true
           * for every stored compressed node. */
          has_children = true;
        }

        fn_impl->get_coeffs().replace(
            mad_key,
            nodeT(coeffT(mad_tensor, fn_impl->get_tensor_args()), has_children));
      }
    }, ttg::edges(in), ttg::edges(), name, {"input"}, {});

  store_tt->set_keymap(keymap);
  return store_tt;
}

} // namespace mra::vmra

#endif // MRA_TASKS_VMRA_H
