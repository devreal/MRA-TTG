#include <ttg.h>
#include "mra/mra.h"
#include <any>
#include <numbers>
#include <madness/mra/mra.h>
#include <madness/world/world.h>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

static const int init_lev = 2;
static double expnt = 1000.0;

template<typename T, mra::Dimension NDIM>
void test_derivative(std::size_t N, size_type K, Dimension axis, T precision,
                     int max_level, int initial_level,
                     T verification_precision, int num_batches) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6,6);
  T g1 = 0;
  T g2 = 0;


  auto pmap = make_procmap<NDIM>(N, num_batches);
  auto dmap = make_devicemap<NDIM>(pmap);

  std::array<Slice,NDIM> slices = {Slice(0, K-1), Slice(0, K-1), Slice(0, 2*K-1)};

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result,
                                                                      reconstruct_result,
                                                                      derivative_result;

  auto gaussians = make_functionset<Gaussian<T, NDIM>>(pmap.batch_manager());
  auto gaussians_view = gaussians->current_view(); // host view

  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> umap;
  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> cmap;

  for (int i = 0; i < gaussians->num_functions(); ++i) {
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = 0.0;
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians_view[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r, init_lev);
  }

  // put it into a buffer
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(gaussians, project_control);
  // auto start_d = make_start(project_d_control);
  auto project = make_project(db, gaussians, K, max_level, functiondata, precision, project_control, project_result, "project", pmap, dmap);
  // C(P)
  auto compress = make_compress(gaussians, K, functiondata, project_result, compress_result, "compress", pmap, dmap);
  // // R(C(P))
  auto reconstruct = make_reconstruct(gaussians, K, functiondata, compress_result, reconstruct_result, "reconstruct", pmap, dmap);
  // D(R(C(P)))
  //auto extract_u = make_extract(reconstruct_result, umap);
  auto derivative = make_derivative(gaussians, K, reconstruct_result, derivative_result, functiondata, db, g1, g2, axis,
                                    FunctionData<T, NDIM>::BC_DIRICHLET, FunctionData<T, NDIM>::BC_DIRICHLET, "derivative", pmap, dmap);

  auto sink = ttg::make_tt([](const Key<3>& key, const FunctionsReconstructedNode<T, NDIM>& node){
      //std::cout << "Received derivative for key " << key << " with norm " << node.norm() << " and sum " << node.sum() << std::endl;
    },
    ttg::edges(derivative_result), ttg::edges(), "sink");

  auto connected = make_graph_executable(start.get());
  assert(connected);

  /* start executing */
  ttg::execute();

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
    // std::cout << "Is everything connected? " << connected << std::endl;
    // std::cout << "==== begin dot ====\n";
    // std::cout << Dot()(start.get()) << std::endl;
    // std::cout << "====  end dot  ====\n";

    beg = std::chrono::high_resolution_clock::now();
  }

  if (ttg::default_execution_context().rank() == 0) {
    // This kicks off the entire computation
    start->invoke();
  }
  ttg::fence();

  if (ttg::default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000
              << std::endl;
  }

}

int main(int argc, char **argv) {

  /* options */
  auto opt = mra::OptionParser(argc, argv);
  size_type N = opt.parse("-N", 1);
  size_type K = opt.parse("-K", 8);
  expnt = opt.parse("-e", 100.0); // default: 100.0
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int axis    = opt.parse("-a", 0);
  int log_precision = opt.parse("-p", 6); // default: 1e-6
  int max_level = opt.parse("-l", -1);
  int domain = opt.parse("-d", 6);
  int verification_log_precision = opt.parse("-v", 12); // default: 1e-12
  int initial_level = opt.parse("-i", 2); // initial level for the Gaussian functions
  int num_batches = opt.parse("-b", 0); // batch size for the test, default is 0 (select automatically)
  int nrep = opt.parse("-n", 1); // number of repetitions

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  if (ttg::default_execution_context().rank() == 0) {
    std::cout << "Running MADNESS derivative test with parameters: "
              << "N = " << N << ", K = " << K
              << ", expnt = " << expnt
              << ", axis = " << axis
              << ", log_precision = " << log_precision
              << ", max_level = " << max_level
              << ", verification_log_precision = " << verification_log_precision
              << ", initial_level = " << initial_level
              << std::endl;
  }

  for (int i = 0; i < nrep; ++i) {
    test_derivative<double, 3>(N, K, axis, std::pow(10, -log_precision), max_level,
                              std::pow(10, -verification_log_precision), initial_level,
                              num_batches);
  }

  ttg::finalize();
}
