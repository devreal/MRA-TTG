#ifndef MRA_CONV_FACTORY_H
#define MRA_CONV_FACTORY_H

#include <cmath>
#include <memory>
#include <vector>
#include <madness/mra/mra.h>

namespace mra {

  using real_convolution_t = madness::SeparatedConvolution<double, 3>;

  /**
   * Returns a MADNESS convolution operator for a Gaussian with given exponent and K,
   * with N convolution operators.
   */
  template<typename T>
  inline auto make_mad_convolution(T expnt, int K, int N, madness::World &world) {

    std::vector< std::shared_ptr< madness::Convolution1D<double> > > ops(N);
    for (int i = 0; i < N; ++i) {
      expnt *= 0.9; // slightly different exponent for each operator
      double coeff = std::pow(2.0*expnt/std::numbers::pi, 0.25*3);
      ops[i].reset(new madness::GaussianConvolution1D<double>(K, coeff, expnt, 0, madness::LatticeRange()));
    }

    return std::make_shared<real_convolution_t>(world, ops, K);
  }

} // namespace mra

#endif // MRA_CONV_FACTORY_H