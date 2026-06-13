#include <ttg.h>
#include "mra/mra.h"
#include "compare_mad_mra.h"
#include <any>
#include <numbers>
#include <madness/mra/mra.h>
#include <madness/world/world.h>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

typedef madness::Vector<double,3> coordT;
typedef madness::Function<double,3> functionT;
typedef madness::FunctionFactory<double,3> factoryT;
typedef madness::Tensor<double> tensorT;

static const int init_lev = 2;
static double expnt = 1000.0;

template <typename T>
static T u(const coordT &pt) {
  auto fac = std::pow(T(2.0*expnt/std::numbers::pi),T(0.25*3)); // normalization factor
  return fac*(std::exp(-1*expnt*pt[0]*pt[0]) * std::exp(-1*expnt*pt[1]*pt[1]) * std::exp(-1*expnt*pt[2]*pt[2]));
}

template <typename T>
auto compute_conv_madness(madness::World& world, size_type k, T thresh, int domain, int init_lev) {

  madness::FunctionDefaults<3>::set_cubic_cell( -domain, domain );
  madness::FunctionDefaults<3>::set_k(k);
  madness::FunctionDefaults<3>::set_refine(true);
  madness::FunctionDefaults<3>::set_autorefine(true);
  madness::FunctionDefaults<3>::set_thresh(thresh);
  madness::FunctionDefaults<3>::set_initial_level(init_lev);

  functionT f = factoryT(world).f(u);
  f.set_autorefine(true);
  // functionT opf = op(f);
  return f;

}

template <typename T, mra::Dimension NDIM>
void compute_conv_mra(size_type N, size_type K, int num_batches, T precision, int domain, int max_level,
                      T verification_precision, int argc, char** argv) {

  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-domain,domain);
  bool is_ns = false;

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> project_result, reconstruct_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T,NDIM>> compress_result;

  // std::map<mra::Key<NDIM>, mra::FunctionsCompressedNode<T,NDIM>> cmap;
  std::map<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> rmap;

  auto pmap = make_procmap<NDIM>(N, num_batches);
  auto dmap = make_devicemap<NDIM>(pmap);

  // define N Gaussians
  auto gaussians = make_functionset<mra::Gaussian<T, NDIM>>(pmap.batch_manager());
  auto gaussians_view = gaussians->current_view(); // host view
  // define N Gaussians
  for (int i = 0; i < gaussians->num_functions(); ++i) {
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = 0.0;
    }
    gaussians_view[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r, init_lev);
  }

  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);

  auto start = make_start(gaussians, project_control);
  auto project = make_project(db, gaussians, K, max_level, functiondata, precision, project_control, project_result);
  auto compress = make_compress(gaussians, K, false, functiondata, project_result, compress_result, "compress");
  auto reconstruct = make_reconstruct(gaussians, K, false, functiondata, compress_result, reconstruct_result, "reconstruct");
  auto extract = make_extract(reconstruct_result, rmap);

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
      beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke();
  }
  ttg::execute();
  ttg::fence();

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world, argc, argv);
  {
    auto mad_f = compute_conv_madness<T>(world, K, precision, domain, init_lev);
    compare_mra_madness<T, NDIM>(mad_f, rmap, "projection", T(1e-12));
  }
  world.gop.fence();
}

int main(int argc, char** argv) {

  auto opt = mra::OptionParser(argc, argv);
  int num_batches = opt.parse("-b", 1);
  size_type N = opt.parse("-N", 1);
  size_type K = opt.parse("-K", 8);
  expnt = opt.parse("-e", expnt); // default: 1500
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int log_precision = opt.parse("-p", 6); // default: 1e-6
  int max_level = opt.parse("-l", -1);
  int domain = opt.parse("-d", 6);
  int verification_log_precision = opt.parse("-v", 12); // default: 1e-12

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  #if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);

  compute_conv_mra<double, 3>(N, K, num_batches, std::pow(10, -log_precision), domain, max_level,
                              std::pow(10, -verification_log_precision), argc, argv);

  madness::finalize();
  ttg::finalize();
}
