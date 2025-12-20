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


#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>

#include "mra/misc/options.h"

using namespace madness;

typedef madness::Vector<double,3> coordT;
typedef madness::Function<double,3> functionT;
typedef madness::FunctionFactory<double,3> factoryT;
typedef madness::Tensor<double> tensorT;

static const double L = 12.0;      // box size
static const double thresh = 1e-6; // precision
static double expnt = 1000.0;

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
  const double expnt = 1500.0;
  const real_tensor &cell = FunctionDefaults<3>::get_cell();
  coord_3d origin;
  for (int i = 0; i < 3; i++) {
    origin[i] = 0.0;
  }
  double coeff = pow(2.0 * expnt / constants::pi, 0.75);
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

template <typename T>
static T xbdy_dirichlet(const coordT &pt) {
  return (std::exp(-1*expnt*pt[0]*pt[0]) * std::exp(-1*expnt*pt[1]*pt[1]) * std::exp(-1*expnt*pt[2]*pt[2]));
}


template<typename T>
void test_derivative(World &world, std::size_t N, int K, int axis_a, int axis_b, T thresh,
                     int max_level, int initial_level, int nrep, int seed) {

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  FunctionDefaults<3>::set_k(K);
  FunctionDefaults<3>::set_thresh(thresh);
  FunctionDefaults<3>::set_truncate_mode(1);
  FunctionDefaults<3>::set_cubic_cell(-L / 2, L / 2);
  FunctionDefaults<3>::set_initial_level(initial_level);

  default_random_generator.setstate(
      seed); // Ensure all processes have the same state

  if (world.rank() == 0)
    FunctionDefaults<3>::print();


  functionT xleft_d = factoryT(world).f(xbdy_dirichlet) ;
  functionT xright_d = factoryT(world).f(xbdy_dirichlet) ;

  madness::BoundaryConditions<3> bc;
  bc(0,0) = madness::BCType::BC_FREE;
  bc(0,1) = madness::BCType::BC_FREE;
  bc(1,0) = madness::BCType::BC_FREE;
  bc(1,1) = madness::BCType::BC_FREE;
  bc(2,0) = madness::BCType::BC_FREE;
  bc(2,1) = madness::BCType::BC_FREE;

  // Create a vector of random Gaussian functions
  for (int i = 0; i < nrep; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    std::vector<real_function_3d> a;
    if (seed == 0) {
      a = fixed_gaussians(N, world);
    } else {
      a = random_gaussians(N, world, seed);
    }

    std::vector<real_function_3d> dudx1 = std::move(a);
    for (int ax = axis_a;
         (axis_a < axis_b) ? (ax <= axis_b) : (ax >= axis_b);
         ax = (axis_a < axis_b) ? ax + 1 : ax - 1) {
      madness::Derivative<T, 3> dx1(world, ax, bc, xleft_d, xright_d, K);
      dudx1 = apply(world, dx1, dudx1);
      //dudx1.truncate();
    }

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

  /* options */
  auto opt = mra::OptionParser(argc, argv);
  std::size_t N = opt.parse("-N", 1);
  std::size_t K = opt.parse("-K", 8);
  expnt = opt.parse("-e", 100.0); // default: 100.0
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int axis_a  = opt.parse("-a1", 0); // from axis 0
  int axis_b  = opt.parse("-a2", 2); // to axis 2
  int log_precision = opt.parse("-p", 6); // default: 1e-6
  int max_level = opt.parse("-l", -1);
  int domain = opt.parse("-d", 6);
  int initial_level = opt.parse("-i", 2); // initial level for the Gaussian functions
  int nrep = opt.parse("-n", 1); // number of repetitions
  int seed = opt.parse("-s", 0); // random seed

  if (world.rank() == 0) {
    std::cout << "Running MADNESS derivative benchmark with parameters: "
              << "N = " << N << ", K = " << K
              << ", expnt = " << expnt
              << ", axis = " << axis_a << "-" << axis_b
              << ", log_precision = " << log_precision
              << ", max_level = " << max_level
              << ", initial_level = " << initial_level
              << std::endl;
  }

  startup(world, argc, argv);

  test_derivative<double>(world, N, K, axis_a, axis_b, std::pow(10, -log_precision), max_level, initial_level,
                             nrep, seed);
  finalize();
  return 0;
}
