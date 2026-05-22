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

static const int init_lev = 2;
static double expnt = 1000.0;

template <typename T>
static T u(const coordT &pt) {
  auto fac = std::pow(T(2.0*expnt/std::numbers::pi),T(0.25*3)); // normalization factor
  return fac*(std::exp(-1*expnt*pt[0]*pt[0]) * std::exp(-1*expnt*pt[1]*pt[1]) * std::exp(-1*expnt*pt[2]*pt[2]));
}

template<typename T, Dimension NDIM>
void compare_mra_madness(auto& madfunc, auto& mramap, std::string name, T precision = 1e-15) {
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

template <typename T>
auto compute_conv_madness(madness::World& world, size_type k, T thresh, int domain, int init_lev) {

  madness::FunctionDefaults<3>::set_cubic_cell( -domain, domain );
  madness::FunctionDefaults<3>::set_k(k);
  madness::FunctionDefaults<3>::set_refine(true);
  madness::FunctionDefaults<3>::set_autorefine(true);
  madness::FunctionDefaults<3>::set_thresh(thresh);
  madness::FunctionDefaults<3>::set_initial_level(init_lev);

  functionT f = factoryT(world).f(u);
  f.set_autorefine(true);
  // functionT opf = op(f);
  return f;

}

template <typename T, mra::Dimension NDIM>
void compute_conv_mra(size_type N, size_type K, T precision, int domain, int max_level,
                      T verification_precision, int argc, char** argv) {

  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-domain,domain);
  bool is_ns = false;

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> project_result, reconstruct_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T,NDIM>> compress_result;

  // std::map<mra::Key<NDIM>, mra::FunctionsCompressedNode<T,NDIM>> cmap;
  std::map<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> rmap;

  auto gaussians = std::make_unique<mra::Gaussian<T, NDIM>[]>(N);

  for (size_type n=0; n<N; ++n) {
    mra::Coordinate<T, NDIM> r;
    for (size_type d=0; d<NDIM; ++d) r[d] = 0.0;
    gaussians[n] = mra::Gaussian<T,NDIM>(D[0], expnt, r, init_lev);
  }

  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T,NDIM>>(std::move(gaussians), N);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);

  auto start = make_start(project_control);
  auto project = make_project(db, gauss_buffer, N, K, max_level, functiondata, precision, project_control, project_result);
  auto compress = make_compress(N, K, false, functiondata, project_result, compress_result, "compress");
  auto reconstruct = make_reconstruct(N, K, false, functiondata, compress_result, reconstruct_result, "reconstruct");
  auto extract = make_extract(reconstruct_result, rmap);

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

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world, argc, argv);
  {
    auto mad_f = compute_conv_madness<T>(world, K, precision, domain, init_lev);
    compare_mra_madness<T, NDIM>(mad_f, rmap, "projection", T(1e-12));
  }
  world.gop.fence();
}

int main(int argc, char** argv) {

  auto opt = mra::OptionParser(argc, argv);
  size_type N = opt.parse("-N", 1);
  size_type K = opt.parse("-K", 8);
  expnt = opt.parse("-e", expnt); // default: 1500
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int log_precision = opt.parse("-p", 6); // default: 1e-6
  int max_level = opt.parse("-l", -1);
  int domain = opt.parse("-d", 6);
  int verification_log_precision = opt.parse("-v", 12); // default: 1e-12

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();

  #if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);

  compute_conv_mra<double, 3>(N, K, std::pow(10, -log_precision), domain, max_level,
                              std::pow(10, -verification_log_precision), argc, argv);

  madness::finalize();
  ttg::finalize();
}
