/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680
*/

#include <numbers>

#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>

#include "mra/misc/options.h"

using namespace madness;

static const double L = 12.0;      // box size
static const double thresh = 1e-6; // precision

static double expnt = 1500.0;
static double fac = std::pow(2.0*expnt/std::numbers::pi, 0.25*3);
static double coeff = fac;

using real_factory_t = madness::FunctionFactory<double, 3>;
using real_function_t = madness::Function<double, 3>;
using real_convolution_t = madness::SeparatedConvolution<double, 3>;

// A class that behaves like a function to compute a Gaussian of given origin
// and exponent
class Gaussian : public FunctionFunctorInterface<double, 3> {
public:
  const coord_3d center;
  const double exponent;
  const double coefficient;
  std::vector<coord_3d> specialpt;

  Gaussian(const coord_3d &center, double exponent, double coefficient)
      : center(center), exponent(exponent), coefficient(coefficient),
        specialpt(1) {
    specialpt[0][0] = center[0];
    specialpt[0][1] = center[1];
    specialpt[0][2] = center[2];
  }

  // MADNESS will call this interface
  double operator()(const coord_3d &x) const {
    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
      double xx = center[i] - x[i];
      sum += xx * xx;
    };
    return coefficient * exp(-exponent * sum);
  }

  // By default, adaptive projection into the spectral element basis
  // starts uniformly distributed at the initial level.  However, if
  // a function is "spiky" it may be necessary to project at a finer
  // level but doing this uniformly is expensive.  This method
  // enables us to tell MADNESS about points/areas needing deep
  // refinement (the default is no special points).
  std::vector<coord_3d> special_points() const final { return specialpt; }
};

// Makes a new square-normalized Gaussian functor with random origin and
// exponent
real_functor_3d random_gaussian(int seed) {
  const double expntmin = 1500;
  const double expntmax = 3000;
  const real_tensor &cell = FunctionDefaults<3>::get_cell();
  coord_3d origin;
  default_random_generator.setstate(seed);
  for (int i = 0; i < 3; i++) {
    origin[i] = RandomValue<double>() * (cell(i, 1) - cell(i, 0)) + cell(i, 0);
  }
  double lo = log(expntmin);
  double hi = log(expntmax);
  double expnt = exp(RandomValue<double>() * (hi - lo) + lo);
  if (seed != 0) print("expnt", expnt, origin);
  double coeff = pow(2.0 * expnt / constants::pi, 0.75);
  return real_functor_3d(new Gaussian(origin, expnt, coeff));
}

// Makes a vector of new square-normalized Gaussian functions with random origin
// and exponent
std::vector<real_function_3d> random_gaussians(size_t n, World &world, int seed) {
  std::vector<real_function_3d> result(n);
  for (size_t i = 0; i < n; i++) {
    result[i] = FunctionFactory<double, 3>(world).functor(random_gaussian(seed));
  }
  return result;
}

// Makes a new square-normalized Gaussian functor with fixed origin and
// exponent
real_functor_3d fixed_gaussian() {
  const real_tensor &cell = FunctionDefaults<3>::get_cell();
  coord_3d origin;
  for (int i = 0; i < 3; i++) {
    origin[i] = 0.0;
  }
  return real_functor_3d(new Gaussian(origin, expnt, coeff));
}

// Makes a vector of new square-normalized Gaussian functions with random origin
// and exponent
std::vector<real_function_3d> fixed_gaussians(size_t n, World &world) {
  std::vector<real_function_3d> result(n);
  for (size_t i = 0; i < n; i++) {
    result[i] = FunctionFactory<double, 3>(world).functor(fixed_gaussian());
  }
  return result;
}

void test(World &world, int N, int K, int nrep, int seed, int initial_level) {

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  default_random_generator.setstate(
      99); // Ensure all processes have the same state

  std::vector< std::shared_ptr< madness::Convolution1D<double> > > ops(N);
  for (int i = 0; i < N; ++i) {
    ops[i].reset(new madness::GaussianConvolution1D<double>(K, coeff, expnt, 0, false));
  }

  // f.make_nonstandard(false, true);
  // f.compress();

  real_convolution_t op(world, ops, K);

  //real_function_t opf = op(f);

  // Create a vector of random Gaussian functions
  for (int i = 0; i < nrep; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    std::vector<real_function_3d> a;
    if (seed == 0) {
      a = fixed_gaussians(N, world);
    } else {
      a = random_gaussians(N, world, seed);
    }
    make_nonstandard(world, a, true);
    auto b = apply(world, op, a);
    auto norms = norm2s(world, b);
    end = std::chrono::high_resolution_clock::now();
    if (world.rank() == 0) {
      std::cout << "MAD Execution Time (milliseconds) : "
                << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000
                << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);

  auto opt = mra::OptionParser(argc, argv);
  int N = opt.parse("-N", 1);
  int K = opt.parse("-K", 10);
  int nrep = opt.parse("-n", 3);
  int seed = opt.parse("-s", 5551212);
  int initial_level = opt.parse("-i", 2); // initial level for the Gaussian functions
  int log_precision = opt.parse("-p", 8); // default: 1e-8

  thresh = std::pow(10, -log_precision);

  startup(world, argc, argv);

  FunctionDefaults<3>::set_k(K);
  FunctionDefaults<3>::set_thresh(thresh);
  FunctionDefaults<3>::set_truncate_mode(1);
  FunctionDefaults<3>::set_cubic_cell(-L / 2, L / 2);
  FunctionDefaults<3>::set_initial_level(initial_level);

  if (world.rank() == 0)
    FunctionDefaults<3>::print();

  test(world, N, K, nrep, seed, initial_level);

  finalize();
  return 0;
}
