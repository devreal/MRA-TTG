#ifndef HAVE_COMPARE_MAD_MRA_H
#define HAVE_COMPARE_MAD_MRA_H

#include <mra/misc/key.h>
#include <mra/tensor/functionnode.h>
#include <map>
#include <iostream>
#include <string>
#include <cmath>
#include <type_traits>
#include <array>
#include <stdexcept>

namespace mra {

  namespace detail {
    std::string madfunc_state(auto& madfunc) {
      std::string state;
      if (madfunc.is_compressed()) state += "compressed ";
      if (madfunc.is_nonstandard()) state += "nonstandard ";
      if (madfunc.is_reconstructed()) state += "reconstructed ";
      if (madfunc.is_redundant()) state += "redundant ";
      return state;
    }
  } // namespace detail

  template<typename T, typename NodeT, Dimension NDIM>
  inline void compare_mra_madness(const auto& madfunc, const std::map<Key<NDIM>, NodeT>& mramap, std::string name, T precision = 1e-15)
  {
    bool check = true;
    bool all_zero = true;
    Batch batch = mramap.begin() != mramap.end() ? mramap.begin()->first.batch() : 0; // assume all keys in MRA map have the same batch as MADNESS key
    int num_functions = 0;
    if (mramap.begin() != mramap.end()) {
      batch = mramap.begin()->first.batch();
      num_functions = mramap.begin()->second.count();
    }


    auto compare_single_mad_func = [&](auto& madfunc, auto& mramap, size_type func_idx = 0, bool verbose = false) {
      std::cout << name << "Function " << func_idx << " MRA: " << mramap.size() << " nodes; MAD: " << madfunc.min_nodes() << " nodes" << std::endl;

      /**
       * Sanity check for the right format
       */
      if constexpr (std::is_same_v<NodeT, mra::FunctionsCompressedNode<T, NDIM>>) {
        if (!madfunc.is_compressed() && !madfunc.is_nonstandard()) {
          std::cout << name << ": MADNESS function expected as compressed or nonstandard but found "
                    << detail::madfunc_state(madfunc) << std::endl;
          throw std::runtime_error(name + ": expected MADNESS function to be either compressed or nonstandard");
        }
      } else if constexpr (std::is_same_v<NodeT, mra::FunctionsReconstructedNode<T, NDIM>>) {
        if (!madfunc.is_reconstructed()) {
          std::cout << name << ": MADNESS function expected as reconstructed but found "
                    << detail::madfunc_state(madfunc) << std::endl;
          throw std::runtime_error(name + ": expected MADNESS function to be reconstructed");
        }
      }

      /**
       * Compare each MADNESS node to MRA node, and check that all MRA nodes are in MADNESS map.
       * We allow missing leaf nodes in MRA since MADNESS stores them but MRA does not, but only if they are zero in MADNESS.
       */
      const auto &coeffs = madfunc.get_impl()->get_coeffs();
      for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
        std::array<Translation,NDIM> l;
        for (int i=0; i<NDIM; ++i){
          l[i] = it->first.translation()[i];
        }
        auto mad_coeff = it->second;
        Key<NDIM> key = Key<NDIM>(batch, it->first.level(), l);
        const auto& mra_coeff = mramap.find(key);
        const auto& mad_norm = mad_coeff.coeff().svd_normf();
        if (mra_coeff != mramap.end()) {
          const auto& mra_node = mra_coeff->second;
          auto mra_coeffs = mra_node.coeffs().current_view()(func_idx);
          auto mra_norm = mra_node.is_zero(func_idx) ? 0.0 : mra::normf(mra_coeffs);
          T absdiff = std::abs(mad_norm - mra_norm);
          if (mra_norm != 0.0) {
            all_zero = false;
          }
          if (absdiff > precision) {
            check = false;
            std::cout << "" << name << ": " << it->first << " with norm " << mad_norm
                      << " DOES NOT MATCH MRA norm " << mra_norm << " (absdiff: " << absdiff << ")" << std::endl;
            if (verbose) {
              if (mad_coeff.coeff().size() == mra_coeffs.size()) {
                for (int i = 0; i < mad_coeff.coeff().dim(0); ++i) {
                  for (int j = 0; j < mad_coeff.coeff().dim(1); ++j) {
                    for (int k = 0; k < mad_coeff.coeff().dim(2); ++k) {
                      if (std::abs(mad_coeff.coeff()(i, j, k) - mra_coeffs(i, j, k)) > precision) {
                        std::cout << "    DIFF at coeff (" << i << ", " << j << ", " << k << "): MAD " << mad_coeff.coeff()(i, j, k)
                                  << " vs MRA " << mra_coeffs(i, j, k) << " DIFF " << mad_coeff.coeff()(i, j, k) - mra_coeffs(i, j, k) << std::endl;
                      }
                    }
                  }
                }
              } else if (mra_norm != mad_norm) {
                std::cout << "    MADNESS coeff size: " << mad_coeff.coeff().size() << " with norm " << mad_norm << " vs MRA coeff size: " << mra_coeffs.size() << " with norm " << mra_norm << std::endl;
                throw std::runtime_error(name + ": mismatch in coefficient sizes between MADNESS and MRA");
              }
            }
          } else {
            std::cout << name << ": " << it->first << " with norm " << mad_norm
                      << " matches MRA norm " << mra_norm << std::endl;
          }
        } else {
          // check whether the missing node is a leaf node; MADNESS stores them, MRA does not.
          bool mra_is_child_leaf = false;
          if constexpr(std::is_same_v<NodeT, mra::FunctionsCompressedNode<T, NDIM>>) {
            auto parent_coeff = mramap.find(key.parent());
            if (parent_coeff != mramap.end() && parent_coeff->second.is_child_leaf(func_idx, key)) {
              mra_is_child_leaf = true; // for compressed nodes, we don't want to check leaf nodes since they won't be in the MRA map
            }
          }
          if (!(mad_norm == 0.0 && mad_coeff.is_leaf() && mra_is_child_leaf)) {
            std::cout << name << ": missing node in MRA: " << it->first << " with norm " << mad_norm << std::endl;
            check = false;
          }
          //throw std::runtime_error(name + ": mismatch in tree nodes between MADNESS and MRA");
        }
      }
      // check if all MRA keys are in the madness map
      for (auto it = mramap.begin(); it != mramap.end(); ++it) {
        madness::Vector<Translation, 3UL> l(it->first.translation());
        auto mad_key = madness::Key<NDIM>(it->first.level(), l);
        auto mad_coeff = coeffs.find(mad_key);
        auto mra_coeffs = it->second.coeffs().current_view()(func_idx);
        if (mad_coeff.get() == coeffs.end()) {
          if (mra::normf(mra_coeffs) > precision) check = false;
          std::cout << name << ": missing node in MADNESS: " << it->first << " norm "
                    << mra::normf(mra_coeffs) << std::endl;
        }
      }
    };

    /**
     * Support both vector
     */
    if constexpr (std::is_same_v<std::decay_t<decltype(madfunc)>, std::vector<madness::Function<T, NDIM>>>) {
      if (num_functions != madfunc.size()) {
        std::cout << name << ": number of functions in MRA map (" << num_functions << ") does not match number of MADNESS functions (" << madfunc.size() << ")" << std::endl;
        throw std::runtime_error(name + ": mismatch in number of functions between MADNESS and MRA");
      }
      for (size_type func_idx = 0; func_idx < num_functions; ++func_idx) {
        compare_single_mad_func(madfunc[func_idx], mramap, func_idx);
      }
    } else {
      compare_single_mad_func(madfunc, mramap);
    }

    /**
     * Summary
     */
    if (all_zero) {
      std::cout << name << ": all existing nodes are zero in MRA, something is weird" << std::endl;
    } else if (check) {
      std::cout << name << ": all nodes match between MADNESS and MRA" << std::endl;
    } else {
      std::cout << name << ": some nodes match between MADNESS and MRA, but not all" << std::endl;
      throw std::runtime_error(name + ": mismatch in norms between MADNESS and MRA");
    }
  }


  template<typename T, std::size_t NDIM>
  inline void compare_mra_madness(const std::vector<madness::Function<T, NDIM>>& madfunc1,
                                  const std::vector<madness::Function<T, NDIM>>& madfunc2,
                                  const std::string name, T precision = 1e-15)
  {
    if (madfunc1.size() != madfunc2.size()) {
      std::cout << name << ": number of functions in MADNESS vector 1 (" << madfunc1.size() << ") does not match number of functions in MADNESS vector 2 (" << madfunc2.size() << ")" << std::endl;
      throw std::runtime_error(name + ": mismatch in number of functions between MADNESS vectors");
    }
    for (std::size_t i = 0; i < madfunc1.size(); ++i) {
      auto tree1_size = madfunc1[i].get_impl()->tree_size();
      auto tree2_size = madfunc2[i].get_impl()->tree_size();
      if (tree1_size != tree2_size) {
        std::cout << name << ": MADNESS function " << i << " in vector 1 has "
        << tree1_size << " nodes but in vector 2 has "
        << tree2_size << " nodes" << std::endl;
        throw std::runtime_error(name + ": mismatch in tree size between MADNESS vectors");
      }
    }
    bool check = true;
    for (std::size_t i = 0; i < madfunc1.size(); ++i) {
      // TODO: check that both trees are in the same state
      if (madfunc1[i].get_impl()->get_tree_state() != madfunc2[i].get_impl()->get_tree_state()) {
        std::cout << name << ": MADNESS function " << i << " in vector 1 is in state "
                  << detail::madfunc_state(madfunc1[i]) << " but in vector 2 is in state "
                  << detail::madfunc_state(madfunc2[i]) << std::endl;
        check = false;
        continue;
      }
      for (auto& node1 : madfunc1[i].get_impl()->get_coeffs()) {
        auto node2 = madfunc2[i].get_impl()->get_coeffs().find(node1.first);
        if (node2.get() == madfunc2[i].get_impl()->get_coeffs().end()) {
          std::cout << name << ": node " << node1.first << " in MADNESS vector 1 function " << i << " (norm " << node1.second.coeff().normf() << ") not found in MADNESS vector 2" << std::endl;
          check = false;
          continue;
        }

        if (node2.get()->second.has_children() != node1.second.has_children()) {
          std::cout << name << ": node " << node1.first << " in MADNESS vector 1 function " << i
                    << " has children " << node1.second.has_children() << " but in vector 2 has children "
                    << node2.get()->second.has_children() << std::endl;
          check = false;
          /* non-fatal error */
        }

        auto norm1 = node1.second.coeff().normf();
        auto norm2 = node2.get()->second.coeff().normf();
        if (std::abs(norm1 - norm2) > precision) {
          std::cout << name << ": node " << node1.first << " in MADNESS function " << i
                    << " has norm " << norm1 << " in vector 1 but norm " << norm2 << " in vector 2" << std::endl;
          check = false;
          continue;
        }
      }
    }
    if (!check) {
      throw std::runtime_error(name + ": mismatch in norms between MADNESS vectors");
    } else {
      std::cout << name << ": all nodes match between MADNESS vectors" << std::endl;
    }
  }

} // namespace mra

#endif // HAVE_COMPARE_MAD_MRA_H