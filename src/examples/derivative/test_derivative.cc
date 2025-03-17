#include <ttg.h>
#include "mra/mra.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

template<typename T, mra::Dimension NDIM>
void test_derivative(std::size_t N, std::size_t K, Dimension axis, T precision, int max_level, int d) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-d,d);
  T g1 = 0;
  T g2 = 0;

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control, project_d_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_derivative_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_d_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> derivative_result, project_d_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  auto gaussians_deriv = std::make_unique<mra::GaussianDerivative<T, NDIM>[]>(N);
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
    gaussians_deriv[i] = mra::GaussianDerivative<T, NDIM>(D[0], expnt, r);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  auto gauss_deriv_buffer = ttg::Buffer<mra::GaussianDerivative<T, NDIM>>(std::move(gaussians_deriv), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  // auto start_d = make_start(project_d_control);
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, precision, project_control, project_result);
  auto project_d = make_project(db, gauss_deriv_buffer, N, K, max_level, functiondata, precision, project_control, project_d_result);
  // C(P)
  auto compress = make_compress(N, K, functiondata, project_result, compress_result, "compress-cp");
  auto compress_d = make_compress(N, K, functiondata, project_d_result, compress_d_result, "compress-Dcp");
  // // R(C(P))
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result, "reconstruct-rcp");
  // D(R(C(P)))
  auto derivative = make_derivative(N, K, reconstruct_result, derivative_result, functiondata, db, g1, g2, axis,
                                    FunctionData<T, NDIM>::BC_DIRICHLET, FunctionData<T, NDIM>::BC_DIRICHLET, "derivative");

  // C(D(R(C(P))))
  auto compress_r = make_compress(N, K, functiondata, derivative_result, compress_derivative_result, "compress-deriv-crcp");

  // | C(D(R(C(P)))) - factor * C(P) |
  auto gaxpy_r = make_gaxpy(compress_derivative_result, compress_d_result, gaxpy_result, T(1.0), T(-1.0), N, K, "gaxpy");

  auto norm  = make_norm(N, K, gaxpy_result, norm_result);
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
      // std::cout << Dot()(start.get()) << std::endl;
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
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int axis    = opt.parse("-a", 0);
  int log_precision = opt.parse("-p", 4); // default: 1e-4
  int max_level = opt.parse("-l", -1);
  int domain = opt.parse("-d", 6);

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  test_derivative<double, 3>(N, K, axis, std::pow(10, -log_precision), max_level, domain);

  ttg::finalize();
}