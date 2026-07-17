#include <ttg.h>
#include "mra/mra.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

#include "conv_factory.h"

using namespace mra;

using coord_t = madness::Vector<double, 3>;
using real_factory_t = madness::FunctionFactory<double, 3>;
using real_function_t = madness::Function<double, 3>;

template<typename T, mra::Dimension NDIM>
void test_convolution(int nrep, int N, int K, int Nop,
                      int num_batches, int seed,
                      int max_level, int initial_level,
                      T precision,
                      T root_radius, T expnt,
                      T domain_size, bool print_dot)
{
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-domain_size, domain_size);
  bool is_ns = true;

  auto pmap = make_procmap<NDIM>(N, num_batches);
  auto dmap = make_devicemap<NDIM>(pmap);

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_convolution_result;
  ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = make_functionset<mra::Gaussian<T, NDIM>>(pmap.batch_manager());
  auto gaussians_view = gaussians->current_view(); // host view
  // T expnt = 1000.0;
  for (int i = 0; i < gaussians->num_functions(); ++i) {
    expnt = (seed > 0) ? (expnt + 1500*drand48()) : expnt;
    mra::Coordinate<T,NDIM> r;
    if (seed > 0) {
      for (size_t d=0; d<NDIM; d++) {
        r[d] = T(-1*(root_radius)) + T(root_radius)*drand48();
      }
    }
    gaussians_view[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r, initial_level);
  }

  if (seed == 0) {
    if (seed == 0) std::cout << N << " Gaussians with expnt " << expnt << std::endl;
  }

  madness::World mad_world(SafeMPI::COMM_WORLD);
  auto mad_conv = mra::make_mad_convolution(expnt, K, Nop, mad_world);
  auto op = mra::GaussianConvolutionOperator<T, NDIM>{mad_conv};

  std::vector<std::unique_ptr<ttg::TTBase>> tts;

  // auto gauss_deriv_buffer = ttg::Buffer<mra::GaussianDerivative<T, NDIM>>(std::move(gaussians_deriv), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(gaussians, project_control);
  auto project = make_project(db, gaussians, K, max_level, functiondata, precision, 0, 1.0, project_control, project_result);
  auto compress = make_compress(gaussians, K, is_ns, functiondata, project_result, compress_result, "compress");

  auto convolution = make_convolution(gaussians, K, compress_result, compress_convolution_result, op, precision, 0, 1.0, "convolution");

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

  auto norm  = make_norm(gaussians, K, compress_convolution_result, norm_result);
  // final check
  auto norm_check = ttg::make_tt([&](const mra::Key<NDIM>& key, const mra::DenseTensor<T, 1>& norms){
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
        start->invoke();
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
  int Nop = opt.parse("-O", 1);
  int K = opt.parse("-K", 8);
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int log_precision = opt.parse("-p", 8); // default: 1e-4
  int max_level = opt.parse("-l", -1);
  int initial_level = opt.parse("-i", 2);
  bool norand = opt.exists("-norand");
  int num_batches = opt.parse("-b", 0); // batch size for the test, default is 0 (select automatically)
  int seed = opt.parse("-s", norand ? 0 : 5551212); // seed for random number generator, 0 for deterministic
  double root_radius = opt.parse("-r", 2.0); // radius of the root domain cube
  int domain = opt.parse("-d", 6);
  bool print_dot = opt.exists("-dot");
  int nrep = opt.parse("-n", 3);
  double expnt_arg = opt.parse("-e", 100.0);

  mra::initialize(argc, argv, cores);

  test_convolution<double, 3>(nrep, N, K, Nop, num_batches, seed,
                              max_level, initial_level, std::pow(10, -log_precision),
                              root_radius, expnt_arg, domain, print_dot);

  mra::finalize();
}