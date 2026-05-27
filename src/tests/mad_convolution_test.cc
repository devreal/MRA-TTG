#include <ttg.h>
#include "mra/mra.h"
#include <any>
#include <numbers>
#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/mra/operator.h>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

#include "compare_mad_mra.h"

using namespace mra;

static double Length = 6.0;
// static double width = 2*Length;
static double expnt = 1500.0;
static const int init_lev = 2;

using coord_t = madness::Vector<double, 3>;
using real_factory_t = madness::FunctionFactory<double, 3>;
using real_function_t = madness::Function<double, 3>;
using real_convolution_t = madness::SeparatedConvolution<double, 3>;

template <typename T>
static T u_exact(const coord_t &pt) {
  auto fac = std::pow(T(2.0*expnt/std::numbers::pi),T(0.25*3)); // normalization factor
  return fac*(std::exp(-1*expnt*pt[0]*pt[0]) * std::exp(-1*expnt*pt[1]*pt[1]) * std::exp(-1*expnt*pt[2]*pt[2]));
}

template <typename T, Dimension NDIM>
auto compute_conv_madness(real_convolution_t& mad_conv) {

  //madness::FunctionDefaults<3>::set_truncate_on_project(false);

  real_function_t f = real_factory_t(mad_conv.get_world()).f(u_exact);
  f.set_autorefine(true);


  std::cout << "MAD function has " << f.min_nodes() << " nodes before convolution" << std::endl;

  f.make_nonstandard(false, true);

  real_function_t opf = mad_conv(f);
  // std::cout << "Tree State of f: " << f.get_impl()->get_tree_state() << std::endl;
  return std::make_tuple(std::move(f), std::move(opf));
}


template<typename T, mra::Dimension NDIM>
void test_convolution(int num_batches, std::size_t N, size_type K, T precision, int max_level,
                     T verification_precision, real_convolution_t& mad_conv) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto functiondata2 = mra::FunctionData<T,NDIM>(2*K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-Length,Length);

  std::map<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> cmap, nsmap, convmap;
  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> projmap, rmap, rconvmap;

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result,
                                                                      reconstruct_result,
                                                                      reconstruct_conv_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result,
                                                                   compress_r_result,
                                                                   convolution_result;
  ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> norm_result;

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

  std::cout << N << " Gaussians with expnt " << expnt << std::endl;

  mra::GaussianConvolutionOperator<T, NDIM> op(mad_conv);

  // auto gauss_deriv_buffer = ttg::Buffer<mra::GaussianDerivative<T, NDIM>>(std::move(gaussians_deriv), N);
  auto db            = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start         = make_start(gaussians, project_control);
  auto project       = make_project(db, gaussians, K, max_level, functiondata, precision, project_control, project_result);
  auto extract_project = make_extract(project_result, projmap);
  auto compress      = make_compress(gaussians, K, false, functiondata, project_result, compress_result, "compress");
  auto extract_compress = make_extract(compress_result, cmap);
  auto reconstruct     = make_reconstruct(gaussians, K, false, functiondata, compress_result, reconstruct_result, "reconstruct");
  auto extract_reconstruct = make_extract(reconstruct_result, rmap);
  auto compress_r   = make_compress(gaussians, K, true, functiondata, reconstruct_result, compress_r_result, "compress_reconstruct");
  auto extract_ns = make_extract(compress_r_result, nsmap);
  auto convolve      = make_convolution(gaussians, K, compress_r_result, convolution_result, op, precision, "convolution");
  auto extract_conv  = make_extract(convolution_result, convmap);
  auto reconstruct_conv = make_reconstruct(gaussians, K, true, functiondata, convolution_result, reconstruct_conv_result, "reconstruct_convolution");
  auto extract_rconv  = make_extract(reconstruct_conv_result, rconvmap);
  auto connected     = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {

      // beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke();
  }
  ttg::execute();
  ttg::fence();

  {
    auto [madfunc, madconv] = compute_conv_madness<T, NDIM>(mad_conv);
    // std::cout << "Tree State of madfunc: " << madfunc.get_impl()->get_tree_state() << std::endl;
    // auto madkey = madness::Key<NDIM>(0, {0, 0, 0});
    // const auto &madcoeffs = madfunc.get_impl()->get_coeffs();
    // for (auto it = madcoeffs.begin(); it != madcoeffs.end(); ++it) {
    //   std::array<Translation,NDIM> l;
    //   if (it->first.level() == madkey.level()) {
    //     auto madcoeff = it->second;
    //     test_conv_node<T, NDIM>(world, madcoeff.coeff(), N, K, cmap, op, precision, init_lev);
    //   }
    // }
    // // auto madcoeff = madcoeffs.find(madkey);
    // auto coeff_itr = madcoeff.get();
    // const auto& coeffs = coeff_itr->second.coeff();
    // std::cout << "Coeffs: " << coeffs << std::endl;

    // test_conv_node<T, NDIM>(world, madcoeff, N, K, cmap, op, precision, init_lev);
    // compare_mra_madness<T, NDIM>(madfunc, rmap, "reconstruct_result", verification_precision);
    // madfunc.get_impl()->change_tree_state(madness::TreeState::nonstandard);
    //madness::Function<T,NDIM> fff=(madfunc);
    // fff.make_nonstandard(false, true);
    // fff.compress();
    //compare_mra_madness(madfunc, cmap, "compress_result", verification_precision);


    //madfunc.reconstruct();
    //compare_mra_madness(madfunc, rmap, "reconstruct_result", verification_precision);
    //madfunc.compress();
    //compare_mra_madness(madfunc, cmap, "compress_result", verification_precision);
    madfunc.make_nonstandard(false, true);
    compare_mra_madness(madfunc, nsmap, "nonstandard_result", verification_precision);
    compare_mra_madness(madconv, rconvmap, "conv_result", verification_precision);
  }
  mad_conv.get_world().gop.fence();
}

int main(int argc, char **argv) {

  /* options */
  auto opt = mra::OptionParser(argc, argv);
  size_type N = opt.parse("-N", 1);
  size_type K = opt.parse("-K", 8);
  expnt = opt.parse("-e", expnt); // default: 100.0
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int log_precision = opt.parse("-p", 6); // default: 1e-6
  int max_level = opt.parse("-l", -1);
  int num_batches = opt.parse("-b", 0); // batch size for the test, default is 0 (select automatically)
  Length = opt.parse("-d", Length);
  bool norand = opt.exists("-norand");
  /**
   * Adaptively set log precision based on the K the user selected.
   * NOTE: MRA/TTG does not use low-rank operator matrices and instead
   *       does full-rank every time. This leads to slight deviations
   *       below the user-selected threshold.
   */
  int verification_log_precision = opt.parse("-v", std::min(12, log_precision+2));
  bool trace = opt.exists("-trace");

  auto precision = std::pow(10, -log_precision);


  if (trace) {
    ttg::trace_on();
  }

  /* initializes TTG, MADNESS, and PaRSEC */
  mra::initialize(argc, argv, cores);

  // Setup MADNESS
  madness::FunctionDefaults<3>::set_cubic_cell( -Length, Length );
  madness::FunctionDefaults<3>::set_k(K);
  madness::FunctionDefaults<3>::set_refine(true);
  madness::FunctionDefaults<3>::set_autorefine(true);
  madness::FunctionDefaults<3>::set_thresh(precision);
  madness::FunctionDefaults<3>::set_initial_level(init_lev);

  double coeff = std::pow(2.0*expnt/std::numbers::pi, 0.25*3);
  madness::World world(SafeMPI::COMM_WORLD);
  std::vector< std::shared_ptr< madness::Convolution1D<double> > > ops(1);
  ops[0].reset(new madness::GaussianConvolution1D<double>(K, coeff, expnt, 0, madness::LatticeRange()));
  real_convolution_t mad_conv(world, ops, K);


  if (ttg::default_execution_context().rank() == 0) {
    std::cout << "Running MADNESS convolution test with parameters: "
              << "N = " << N << ", K = " << K
              << ", expnt = " << expnt
              << ", log_precision = " << -1*log_precision
              << ", max_level = " << max_level
              << ", verification_log_precision = " << -1*verification_log_precision
              << std::endl;
  }

  test_convolution<double, 3>(num_batches, N, K, precision, max_level,
                             std::pow(10, -verification_log_precision), mad_conv);

  mra::finalize();
}
