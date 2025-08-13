#include <ttg.h>
#include "mra/mra.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

template<typename T, mra::Dimension NDIM>
void test(std::size_t N, std::size_t K, int max_level) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6.0,6.0);
  T g1 = 0;
  T g2 = 0;
  Dimension axis = 0;

  auto pmap = make_procmap<NDIM>(N, 1);
  auto dmap = make_devicemap<NDIM>(pmap);

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> derivative_result;

  // define N Gaussians
  auto gaussians = make_functionset<mra::Gaussian<T, NDIM>>(pmap.batch_manager());
  auto gaussians_view = gaussians->current_view(); // host view
  for (int i = 0; i < gaussians->num_local_functions(); ++i) {
    T expnt = 1500 + 1500*drand48();
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = T(-6.0) + T(12.0)*drand48();
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians_view[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r);
  }

  // put it into a buffer
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(gaussians, project_control);
  auto project = make_project(db, gaussians, K, max_level, functiondata, T(1e-6), project_control, project_result);
  auto compress = make_compress(gaussians, K, functiondata, project_result, compress_result);
  auto reconstruct = make_reconstruct(gaussians, K, functiondata, compress_result, reconstruct_result);
  auto gaxpy = make_gaxpy(T(1.0), T(-1.0), gaussians, K, compress_result, compress_result, gaxpy_result);
  auto multiply = make_multiply(gaussians, functiondata, db, K, reconstruct_result, reconstruct_result, multiply_result);
  auto derivative = make_derivative(gaussians, K, multiply_result, derivative_result, functiondata, db, g1, g2, axis,
                                    FunctionData<T, NDIM>::BC_DIRICHLET, FunctionData<T, NDIM>::BC_DIRICHLET, "derivative");
  auto printer =   make_printer(project_result,    "projected    ", false);
  auto printer2 =  make_printer(compress_result,   "compressed   ", false);
  auto printer3 =  make_printer(reconstruct_result,"reconstructed", false);
  auto printer4 = make_printer(gaxpy_result, "gaxpy", false);
  auto printer5 = make_printer(multiply_result, "multiply", false);
  auto printer6 = make_printer(derivative_result, "derivative", false);

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
      //std::cout << "Is everything connected? " << connected << std::endl;
      //std::cout << "==== begin dot ====\n";
      //std::cout << Dot()(start.get()) << std::endl;
      //std::cout << "====  end dot  ====\n";

      beg = std::chrono::high_resolution_clock::now();
  }
  ttg::execute();

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
  size_type K = opt.parse("-K", 10);
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int max_level = opt.parse("-l", -1);

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  test<double, 3>(N, K, max_level);

  ttg::finalize();
}
