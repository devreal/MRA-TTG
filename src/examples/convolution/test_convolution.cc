#include <ttg.h>
#include "mra/mra.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

template<typename T, mra::Dimension NDIM>
void test_convolution(std::size_t N, std::size_t K, Dimension axis, T precision, int max_level, int d) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-d,d);
  T g1 = 0;
  T g2 = 0;
  bool is_ns = true;

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_convolution_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  T expnt = 1000.0;
  T factor = expnt;

  for (int i = 0; i < N; ++i) {
    // T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    std::cout << "GaussianDerivative " << i << " expnt " << expnt << std::endl;
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r);
    // gaussians_deriv[i] = mra::GaussianDerivative<T, NDIM>(D[0], expnt, r);
  }
  T coeff = 10.0; // coefficient for the Gaussian
  mra::Convolution<T, NDIM> conv(K, K, coeff, expnt, functiondata);
  mra::ConvolutionOperator<T, NDIM> op(K, K, conv);

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  // auto gauss_deriv_buffer = ttg::Buffer<mra::GaussianDerivative<T, NDIM>>(std::move(gaussians_deriv), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, precision, project_control, project_result);
  auto compress = make_compress(N, K, is_ns, functiondata, project_result, compress_result, "compress");
  auto convolve = make_convolution(N, K, compress_result, compress_convolution_result, op, "convolution");

  auto norm  = make_norm(N, K, compress_convolution_result, norm_result);
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
  int N = opt.parse("-N", 1);
  int K = opt.parse("-K", 10);
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int axis    = opt.parse("-a", 0);
  int log_precision = opt.parse("-p", 4); // default: 1e-4
  int max_level = opt.parse("-l", -1);
  int domain = opt.parse("-d", 6);

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();
  allocator_init(argc, argv);

  test_convolution<double, 3>(N, K, axis, std::pow(10, -log_precision), max_level, domain);

  allocator_fini();
  ttg::finalize();
}