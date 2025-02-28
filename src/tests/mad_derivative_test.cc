#include <ttg.h>
#include "mra/mra.h"
#include <any>
#include <madness/mra/mra.h>
#include <madness/world/world.h>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

typedef madness::Vector<double,3> coordT;
typedef madness::Function<double,3> functionT;
typedef madness::FunctionFactory<double,3> factoryT;
typedef madness::Tensor<double> tensorT;

template <typename T>
static T u_exact(const coordT &pt) {
  return (std::exp(-pt[0]*pt[0]) * std::exp(-pt[1]*pt[1]) * std::exp(-pt[2]*pt[2]));
}

template <typename T>
static T xleft_dirichlet(const coordT &pt) {
  return T(0);
}

template <typename T, mra::Dimension NDIM>
auto compute_madness(madness::World& world) {
  size_type k = 10;
  functionT u = factoryT(world).f(u_exact );
  functionT  xleft_d = factoryT(world).f( xleft_dirichlet) ;
  functionT xright_d = factoryT(world).f(xleft_dirichlet) ;

  madness::FunctionDefaults<3>::set_cubic_cell( -6, 6);

  madness::BoundaryConditions<3> bc;
  // bc(0,0) = madness::BoundaryType::BC_DIRICHLET;
  // bc(0,1) = madness::BoundaryType::BC_FREE;
  // bc(1,0) = madness::BoundaryType::BC_FREE;
  // bc(1,1) = madness::BoundaryType::BC_FREE;
  // bc(2,0) = madness::BoundaryType::BC_FREE;
  // bc(2,1) = madness::BoundaryType::BC_FREE;

  madness::Derivative<T,NDIM> dx1(world, 0, bc, xleft_d, xright_d, k);
  functionT dudx1 = dx1(u);

  return dudx1;
}

template<typename T, mra::Dimension NDIM>
void test_derivative(std::size_t N, std::size_t K, Dimension axis, T precision, int max_level, int argc, char** argv) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6,6);
  T g1 = 0;
  T g2 = 0;

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, derivative_result;

  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  T expnt = 1.0;

  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> map;

  for (int i = 0; i < N; ++i) {
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  // auto start_d = make_start(project_d_control);
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, precision, project_control, project_result);
  // C(P)
  auto compress = make_compress(N, K, functiondata, project_result, compress_result, "compress-cp");
  // // R(C(P))
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result, "reconstruct-rcp");
  // D(R(C(P)))
  auto derivative = make_derivative(N, K, reconstruct_result, derivative_result, functiondata, db, g1, g2, axis,
                                    FunctionData<T, NDIM>::BC_DIRICHLET, FunctionData<T, NDIM>::BC_DIRICHLET, "derivative");
  auto extract = make_extract(derivative_result, map);

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
      // std::cout << "Is everything connected? " << connected << std::endl;
      // std::cout << "==== begin dot ====\n";
      // std::cout << Dot()(start.get()) << std::endl;
      // std::cout << "====  end dot  ====\n";

      beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke(mra::Key<NDIM>(0, {0}));
  }
  ttg::execute();
  ttg::fence();

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world,argc,argv);
  // call madness function and compare the vector with the map defined above (iterate as in pr_writecoeff)
  compute_madness<T, NDIM>(world);

}

int main(int argc, char **argv) {

  /* options */
  auto opt = mra::OptionParser(argc, argv);
  size_type N = opt.parse("-N", 1);
  size_type K = opt.parse("-K", 10);
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int axis    = opt.parse("-a", 0);
  int log_precision = opt.parse("-p", 4); // default: 1e-4
  int max_level = opt.parse("-l", -1);
  int domain = opt.parse("-d", 6);

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  /* initialize MADNESS PaRSEC backend with the same PaRSEC context */
#if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);

  test_derivative<double, 3>(N, K, axis, std::pow(10, -log_precision), max_level, argc, argv);

  madness::finalize();
  ttg::finalize();
}
