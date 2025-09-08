#include <ttg.h>
#include "mra/mra.h"
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

static const double Length = 4.0;
static const int init_lev = 2;
static double expnt = 1000.0;

template <typename T>
auto compute_conv_madness(madness::World& world, size_type k, T thresh, int init_lev) {
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
void test_convolution(std::size_t N, size_type K, Dimension axis, T precision, int max_level,
                     T verification_precision, int argc, char** argv) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-6,6);
  T g1 = 0;
  T g2 = 0;
  bool is_ns = false;

  std::array<Slice,NDIM> slices = {Slice(0, K-1), Slice(0, K-1), Slice(0, 2*K-1)};

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, derivative_result;

  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);

  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> umap;
  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> cmap;
  std::map<Key<NDIM>, FunctionsReconstructedNode<T, NDIM>> pmap; // results directly after project

  for (int i = 0; i < N; ++i) {
    mra::Coordinate<T,NDIM> r;
    for (size_t d=0; d<NDIM; d++) {
      r[d] = 0.0;
    }
    std::cout << "Gaussian " << i << " expnt " << expnt << std::endl;
    gaussians[i] = mra::Gaussian<T, NDIM>(D[0], expnt, r, init_lev);
  }

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(std::move(gaussians), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(project_control);
  // auto start_d = make_start(project_d_control);
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, precision, project_control, project_result);
  auto extract_project = make_extract(project_result, pmap);
  // C(P)
  auto compress = make_compress(N, K, is_ns, functiondata, project_result, compress_result, "compress");
  // // R(C(P))
  auto reconstruct = make_reconstruct(N, K, functiondata, compress_result, reconstruct_result, "reconstruct");
  // D(R(C(P)))
  auto extract_u = make_extract(reconstruct_result, umap);
  auto derivative = make_derivative(N, K, reconstruct_result, derivative_result, functiondata, db, g1, g2, axis,
                                    FunctionData<T, NDIM>::BC_DIRICHLET, FunctionData<T, NDIM>::BC_DIRICHLET, "derivative");
  auto extract = make_extract(derivative_result, cmap);

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

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world,argc,argv);
  {
    auto u_result = compute_u_madness<T>(world, K, precision, init_lev);
    compare_mra_madness<T, NDIM>(u_result, umap, "u_result", verification_precision);

    auto deriv_result = compute_udx_madness<T>(world, axis, K, precision, init_lev);
    compare_mra_madness<T, NDIM>(deriv_result, cmap, "deriv_result", verification_precision);
  }
  world.gop.fence();
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
              << std::endl;
  }

  /* initialize MADNESS PaRSEC backend with the same PaRSEC context */
#if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);

  test_convolution<double, 3>(N, K, axis, std::pow(10, -log_precision), max_level,
                             std::pow(10, -verification_log_precision), argc, argv);

  madness::finalize();
  ttg::finalize();
}
