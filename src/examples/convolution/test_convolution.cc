#include <ttg.h>
#include "mra/mra.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

using coord_t = madness::Vector<double, 3>;
using real_factory_t = madness::FunctionFactory<double, 3>;
using real_function_t = madness::Function<double, 3>;
using real_convolution_t = madness::SeparatedConvolution<double, 3>;

template<typename T, mra::Dimension NDIM>
void test_convolution(int nrep, std::size_t N, std::size_t K, Dimension axis, T expnt_arg, int seed, T precision, int max_level, int d, int initial_level, bool print_dot) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-d,d);
  bool is_ns = true;

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_convolution_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);
  T expnt = (seed > 0) ? (expnt_arg + expnt_arg*drand48()) : expnt_arg;

  for (int i = 0; i < N; ++i) {
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = (seed > 0) ? (T(-6.0) + T(6.0)*drand48()) : 0.0;
    }
    if (seed > 0) {
      std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    }
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r, initial_level);
  }

  if (seed == 0) {
    if (seed == 0) std::cout << N << " Gaussians with expnt " << expnt_arg << std::endl;
  }

  double coeff = std::pow(2.0*expnt/std::numbers::pi, 0.25*3);
  madness::World world(SafeMPI::COMM_WORLD);
  std::vector< std::shared_ptr< madness::Convolution1D<double> > > ops(1);
  ops[0].reset(new madness::GaussianConvolution1D<double>(K, coeff, expnt, 0, false));
  real_convolution_t mad_conv(world, ops, K);

  mra::GaussianConvolutionOperator<T, NDIM> op(mad_conv);

  std::vector<std::unique_ptr<ttg::TTBase>> tts;

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  // auto gauss_deriv_buffer = ttg::Buffer<mra::GaussianDerivative<T, NDIM>>(std::move(gaussians_deriv), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, precision, project_control, project_result);
  auto compress = make_compress(N, K, is_ns, functiondata, project_result, compress_result, "compress");

  auto convolution = make_convolution(N, K, compress_result, compress_convolution_result, op, precision, "convolution");

#if 0
  /**
   * This is purely for debugging: a thread that prints the pending tasks in each TT every second.
   * You can use this to see if the TTs are making progress or if they are stuck waiting for something.
   */
  tts.push_back(std::move(up_tt));
  tts.push_back(std::move(down_tt));
  tts.push_back(std::move(screener_tt));
  //tts.push_back(std::move(neighbor_dispatch_tt));
  //tts.push_back(std::move(rebalance_down_tt));
  tts.push_back(std::move(shell0_tt));
  tts.push_back(std::move(adjust_leaf_tt));
  tts.push_back(std::move(accumulate_dispatch_tt));
  tts.push_back(std::move(accumulate_tt));
  std::atomic<int> signal = 0;
  auto print_thread = std::thread([&](){
    while (signal.load() == 0) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "======================" << std::endl;
      for (auto& tt : tts) {
        std::cout << "TT " << tt->get_name() << " pending tasks: " << std::endl;
        tt->print_incomplete_tasks();
      }
      signal.store(0);
    }
  });
#endif // 0

  auto norm  = make_norm(N, K, compress_convolution_result, norm_result);
  // final check
  auto norm_check = ttg::make_tt([&](const mra::Key<NDIM>& key, const mra::Tensor<T, 1>& norms){
    // TODO: check for the norm within machine precision
    auto norms_arr = norms.buffer().current_device_ptr();
    for (size_type i = 0; i < N; ++i) {
      //std::cout << "Final norm " << i << ": " << norms_arr[i] << std::endl;
    }
  }, ttg::edges(norm_result), ttg::edges(), "norm-check");

  auto connected = make_graph_executable(start.get());
  assert(connected);

  if (print_dot && ttg::default_execution_context().rank() == 0) {
    std::cout << ttg::Dot(true)(start.get()) << std::endl;
  }

  for (int i = 0; i < nrep; ++i) {
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
}

int main(int argc, char **argv) {

  /* options */
  auto opt = mra::OptionParser(argc, argv);
  int N = opt.parse("-N", 1);
  int K = opt.parse("-K", 10);
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int axis    = opt.parse("-a", 0);
  int log_precision = opt.parse("-p", 8); // default: 1e-4
  int max_level = opt.parse("-l", -1);
  int initial_level = opt.parse("-i", 2);
  bool norand = opt.exists("-norand");
  int seed = opt.parse("-s", norand ? 0 : 5551212); // seed for random number generator, 0 for deterministic
  int domain = opt.parse("-d", 6);
  bool print_dot = opt.exists("-dot");
  int nrep = opt.parse("-n", 3);
  double expnt_arg = opt.parse("-e", 1000.0);

  mra::initialize(argc, argv, cores);

  test_convolution<double, 3>(nrep, N, K, axis, expnt_arg, seed, std::pow(10, -log_precision), max_level, domain, initial_level, print_dot);

  mra::finalize();
}