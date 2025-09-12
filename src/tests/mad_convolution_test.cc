#include <ttg.h>
#include "mra/mra.h"
#include <any>
#include <numbers>
#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/mra/operator.h>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

static double Length = 3.0;
static double width = 2*Length;
static double expnt = 1500.0;
static double coeff = width/sqrt(madness::constants::pi);
static const int init_lev = 2;

using coord_t = madness::Vector<double, 3>;
using real_factory_t = madness::FunctionFactory<double, 3>;
using real_function_t = madness::Function<double, 3>;
using real_convolution_t = madness::SeparatedConvolution<double, 3>;

// template <typename T, Dimension NDIM>
double g(const coord_t& r) {
  // fac (initially: std::pow(madness::constants::inv_sqrt_pi, coord_t::static_size);)
  // is modified to match the normalization in MRA
  static const double fac = std::pow(2.0*expnt/std::numbers::pi, 0.25*3);
  return fac*(std::exp(-1*expnt*r[0]*r[0]) * std::exp(-1*expnt*r[1]*r[1]) * std::exp(-1*expnt*r[2]*r[2]));
}

template <typename T, Dimension NDIM>
auto compute_conv_madness(madness::World& world, size_type k, T thresh, int init_lev) {

  madness::FunctionDefaults<3>::set_cubic_cell( -Length, Length );
  madness::FunctionDefaults<3>::set_k(k);
  madness::FunctionDefaults<3>::set_refine(true);
  madness::FunctionDefaults<3>::set_autorefine(true);
  madness::FunctionDefaults<3>::set_thresh(thresh);
  madness::FunctionDefaults<3>::set_initial_level(init_lev);

  std::vector< std::shared_ptr< madness::Convolution1D<double> > > ops(1);
  ops[0].reset(new madness::GaussianConvolution1D<double>(k, coeff, expnt, 0, false));
  real_convolution_t op(world, ops, k);

  real_function_t f = real_factory_t(world).f(g);

  real_function_t opf = op(f);
  return std::make_tuple(std::move(f), std::move(opf));
}

template<typename T, Dimension NDIM>
void compare_mra_madness(auto& madfunc, auto& mramap, std::string name, T precision = 1e-15)
{
  bool check = true;
  bool all_zero = true;
  const auto &coeffs = madfunc.get_impl()->get_coeffs();
  for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
    std::array<Translation,NDIM> l;
    for (int i=0; i<NDIM; ++i){
      l[i] = it->first.translation()[i];
    }
    auto mad_coeff = it->second;
    Key<NDIM> key = Key<NDIM>(it->first.level(), l);
    auto mra_coeff = mramap.find(key);
    auto mad_norm = mad_coeff.coeff().svd_normf();
    if (mra_coeff != mramap.end()) {
      auto mra_norm = mra::normf(mra_coeff->second.coeffs().current_view());
      T absdiff = std::abs(mad_norm - mra_norm);
      if (mra_norm != 0.0) {
        all_zero = false;
      }
      if (absdiff > precision) {
        check = false;
        std::cout << "" << name << ": " << it->first << " with norm " << mad_norm
                  << " DOES NOT MATCH MRA norm " << mra_norm << " (absdiff: " << absdiff << ")" << std::endl;
        //throw std::runtime_error(name + ": mismatch in norms between MADNESS and MRA");
      } else {
        //std::cout << name << ": " << it->first << " with norm " << mad_norm
        //          << " matches MRA norm " << mra_norm << std::endl;
      }
    } else {
      std::cout << name << ": missing node in MRA: " << it->first << " with norm " << mad_norm << std::endl;
      check = false;
      //throw std::runtime_error(name + ": mismatch in tree nodes between MADNESS and MRA");
    }
  }
  // check if all MRA keys are in the madness map
  for (auto it = mramap.begin(); it != mramap.end(); ++it) {
    madness::Vector<Translation, 3UL> l(it->first.translation());
    auto mad_key = madness::Key<NDIM>(it->first.level(), l);
    auto mad_coeff = coeffs.find(mad_key);
    if (mad_coeff.get() == coeffs.end()) {
      if (mra::normf(it->second.coeffs().current_view()) > precision) check = false;
      std::cout << name << ": missing node in MADNESS: " << it->first << " norm "
                << mra::normf(it->second.coeffs().current_view()) << std::endl;
    }
  }
  if (all_zero) {
    std::cout << name << ": all existing nodes are zero in MRA, something is weird" << std::endl;
  } else if (check) {
    std::cout << name << ": all nodes match between MADNESS and MRA" << std::endl;
  } else {
    std::cout << name << ": some nodes match between MADNESS and MRA, but not all" << std::endl;
    throw std::runtime_error(name + ": mismatch in norms between MADNESS and MRA");
  }
}

template<typename T, mra::Dimension NDIM>
void test_convolution(std::size_t N, size_type K, T precision, int max_level,
                     T verification_precision, int argc, char** argv) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-Length,Length);

  //std::map<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> compmap, convmap;
  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> rmap, convmap;

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result,
                                                                      reconstruct_result,
                                                                      reconstruct_conv_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result,
                                                                   compress_r_result,
                                                                   convolution_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);

  for (int i = 0; i < N; ++i) {
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = 0.0;
    }
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r, init_lev);
  }

  std::cout << N << " Gaussians with expnt " << expnt << std::endl;

  mra::Convolution<T, NDIM> conv(K, K, coeff, expnt, functiondata);
  mra::ConvolutionOperator<T, NDIM> op(K, K, conv);

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  // auto gauss_deriv_buffer = ttg::Buffer<mra::GaussianDerivative<T, NDIM>>(std::move(gaussians_deriv), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, precision, project_control, project_result);
  auto compress = make_compress(N, K, false, functiondata, project_result, compress_result, "compress");
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result, "reconstruct");
  auto compress_r = make_compress(N, K, true, functiondata, reconstruct_result, compress_r_result, "compress_r");
  auto extract_r = make_extract(reconstruct_result, rmap);
  auto convolve = make_convolution(N, K, compress_r_result, convolution_result, op, "convolution");
  auto reconstruct_c = make_reconstruct(N, K, functiondata, convolution_result, reconstruct_conv_result, "reconstruct_conv");
  auto extract_conv = make_extract(reconstruct_conv_result, convmap);

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

  /**
   * compare reconstructed and convolved results of MRA
   */
  for (auto& [key, _] : rmap) {
    auto cit = convmap.find(key);
    if (cit == convmap.end()) {
      std::cout << "Missing key in convolution result: " << key << std::endl;
      throw std::runtime_error("Missing key in convolution result");
    }
  }

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world,argc,argv);
  {
    auto [madfunc, madconv] = compute_conv_madness<T, NDIM>(world, K, precision, init_lev);
    compare_mra_madness<T, NDIM>(madfunc, rmap, "reconstruct_result", verification_precision);
    compare_mra_madness<T, NDIM>(madconv, convmap, "conv_result", verification_precision);
  }
  world.gop.fence();
}

int main(int argc, char **argv) {

  /* options */
  auto opt = mra::OptionParser(argc, argv);
  size_type N = opt.parse("-N", 1);
  size_type K = opt.parse("-K", 10);
  expnt = opt.parse("-e", expnt); // default: 100.0
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int log_precision = opt.parse("-p", 6); // default: 1e-6
  int max_level = opt.parse("-l", -1);
  Length = opt.parse("-d", Length);
  bool norand = opt.exists("-norand");
  int verification_log_precision = opt.parse("-v", 12); // default: 1e-12

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  if (ttg::default_execution_context().rank() == 0) {
    std::cout << "Running MADNESS convolution test with parameters: "
              << "N = " << N << ", K = " << K
              << ", expnt = " << expnt
              << ", log_precision = " << -1*log_precision
              << ", max_level = " << max_level
              << ", verification_log_precision = " << -1*verification_log_precision
              << std::endl;
  }

  /* initialize MADNESS PaRSEC backend with the same PaRSEC context */
#if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);

  test_convolution<double, 3>(N, K, std::pow(10, -log_precision), max_level,
                             std::pow(10, -verification_log_precision), argc, argv);

  madness::finalize();
  ttg::finalize();
}
