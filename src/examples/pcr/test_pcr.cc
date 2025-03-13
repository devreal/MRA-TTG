#include <ttg.h>
#include "mra.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;


template<typename T, mra::Dimension NDIM>
void test_pcr(std::size_t N, std::size_t K, int max_level) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6.0,6.0);

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  auto pmap = PartitionKeymap<NDIM>(); // process map
  auto dmap = PartitionKeymap<NDIM>(ttg::device::num_devices(), pmap.target_level()+1); // device map is one level below the process map

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_reconstruct_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  // T expnt = 1000.0;
  for (int i = 0; i < N; ++i) {
    T expnt = 1500 + 1500*drand48();
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
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, T(1e-6), project_control, project_result, "project", pmap, dmap);
  // C(P)
  auto compress = make_compress(N, K, functiondata, project_result, compress_result, "compress-cp", pmap, dmap);
  // // R(C(P))
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result, "reconstruct-rcp", pmap, dmap);
  // C(R(C(P)))
  auto compress_r = make_compress(N, K, functiondata, reconstruct_result, compress_reconstruct_result, "compress-crcp", pmap, dmap);

  // C(R(C(P))) - C(P)
  auto gaxpy = make_gaxpy(compress_reconstruct_result, compress_result, gaxpy_result, T(1.0), T(-1.0), N, K, "gaxpy", pmap, dmap);
  // | C(R(C(P))) - C(P) |
  auto norm  = make_norm(N, K, gaxpy_result, norm_result, "norm", pmap, dmap);
  // final check
  auto norm_check = ttg::make_tt([&](const mra::Key<NDIM>& key, const mra::Tensor<T, 1>& norms){
    // TODO: check for the norm within machine precision
    auto norms_arr = norms.buffer().current_device_ptr();
    for (size_type i = 0; i < N; ++i) {
      std::cout << "Final norm " << i << ": " << norms_arr[i] << std::endl;
    }
  }, ttg::edges(norm_result), ttg::edges(), "norm-check");

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
      // std::cout << "Is everything connected? " << connected << std::endl;
      // std::cout << "==== begin dot ====\n";
      // std::cout << ttg::Dot(true)(start.get()) << std::endl;
      // std::cout << "====  end dot  ====\n";

      beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke(mra::Key<NDIM>(0, {0}));
  }
  ttg::execute();
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
  size_type K = opt.parse("-K", 10);
  int max_level = opt.parse("-l", -1);
  int cores   = opt.parse("-c", -1); // -1: use all cores

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  test_pcr<double, 3>(N, K, max_level);

  ttg::finalize();
}