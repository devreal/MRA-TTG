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
  void compare_mra_madness(const auto& madfunc, const std::map<Key<NDIM>, NodeT>& mramap, std::string name, T precision = 1e-15)
  {
    bool check = true;
    bool all_zero = true;
    const auto &coeffs = madfunc.get_impl()->get_coeffs();
    Batch batch = mramap.begin() != mramap.end() ? mramap.begin()->first.batch() : 0; // assume all keys in MRA map have the same batch as MADNESS key
    std::cout << name << " MRA: " << mramap.size() << " nodes; MAD: " << madfunc.min_nodes() << " nodes" << std::endl;
#if 0
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
#endif // 0
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
        auto mra_norm = mra::normf(mra_coeff->second.coeffs().current_view());
        T absdiff = std::abs(mad_norm - mra_norm);
        if (mra_norm != 0.0) {
          all_zero = false;
        }
        if (absdiff > precision) {
          check = false;
          std::cout << "" << name << ": " << it->first << " with norm " << mad_norm
                    << " DOES NOT MATCH MRA norm " << mra_norm << " (absdiff: " << absdiff << ")" << std::endl;
          auto mra_view = mra_coeff->second.coeffs().current_view();
          if (mad_coeff.coeff().size() == mra_view.size()) {
            for (int i = 0; i < mad_coeff.coeff().dim(0); ++i) {
              for (int j = 0; j < mad_coeff.coeff().dim(1); ++j) {
                for (int k = 0; k < mad_coeff.coeff().dim(2); ++k) {
                  if (std::abs(mad_coeff.coeff()(i, j, k) - mra_view(0, i, j, k)) > precision) {
                    std::cout << "    DIFF at coeff (" << i << ", " << j << ", " << k << "): MAD " << mad_coeff.coeff()(i, j, k)
                              << " vs MRA " << mra_view(0, i, j, k) << " DIFF " << mad_coeff.coeff()(i, j, k) - mra_view(0, i, j, k) << std::endl;
                  }
                }
              }
            }
          }
        } else {
          std::cout << name << ": " << it->first << " with norm " << mad_norm
                    << " matches MRA norm " << mra_norm << std::endl;
        }
      } else {
        // check whether the missing node is a leaf node; MADNESS stores them, MRA does not.
        bool mra_is_all_child_leafs = false;
        if constexpr(std::is_same_v<NodeT, mra::FunctionsCompressedNode<T, NDIM>>) {
          auto parent_coeff = mramap.find(key.parent());
          if (parent_coeff != mramap.end() && parent_coeff->second.is_child_leaf(0, key.childindex())) {
            mra_is_all_child_leafs = true; // for compressed nodes, we don't want to check leaf nodes since they won't be in the MRA map
          }
        }
        if (!(mad_norm == 0.0 && mad_coeff.is_leaf() && mra_is_all_child_leafs)) {
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
      if (mad_coeff.get() == coeffs.end()) {
      if (mra::normf(it->second.coeffs().current_view()) > precision) check = false;
        std::cout << name << ": missing node in MADNESS: " << it->first << " norm "
                  << mra::normf(it->second.coeffs().current_view()) << std::endl;
      }
    }
    if (all_zero) {
      std::cout << name << ": all existing nodes are zero in MRA, something is weird" << std::endl;
    } else if (check) {
      std::cout << name << ": all nodes match between MADNESS and MRA" << std::endl;
    } else {
      std::cout << name << ": some nodes match between MADNESS and MRA, but not all" << std::endl;
      throw std::runtime_error(name + ": mismatch in norms between MADNESS and MRA");
    }
  }

} // namespace mra

#endif // HAVE_COMPARE_MAD_MRA_H