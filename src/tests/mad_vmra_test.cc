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
#include "mra/vmra/vmra.h"


std::vector<ttg::TTBase*> all_tts;

void print_incomplete_tasks() {
  std::cout << "Incomplete tasks:" << std::endl;
  for (auto tt : all_tts) {
    tt->print_incomplete_tasks();
  }
}

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
static T u_exact(const coord_t &pt, T expnt) {
  auto fac = std::pow(T(2.0*expnt/std::numbers::pi),T(0.25*3)); // normalization factor
  return fac*(std::exp(-1*expnt*pt[0]*pt[0]) * std::exp(-1*expnt*pt[1]*pt[1]) * std::exp(-1*expnt*pt[2]*pt[2]));
}


template <typename T>
static T u1(const coord_t &pt) {
  return u_exact(pt, expnt);
}

template <typename T>
static T u2(const coord_t &pt) {
  return u_exact(pt, expnt/2);
}

template <typename T, Dimension NDIM>
auto compute_conv_madness(size_type N, std::vector<std::shared_ptr<real_convolution_t>>& mad_convs) {

  //madness::FunctionDefaults<3>::set_truncate_on_project(false);

  if (N > 2) {
    throw std::runtime_error("compute_conv_madness: only support N=1 or 2 for now");
  }

  madness::World& world = mad_convs[0]->get_world();

  std::vector<real_function_t> functions(N);
  functions[0] = real_factory_t(world).f(u1);
  if (N == 2) {
    functions[1] = real_factory_t(world).f(u2);
  }

  for (auto& f : functions) {
    f.set_autorefine(false);
    f.make_nonstandard(false, false);
  }
  // wait for everything to complete before starting convolution
  world.gop.fence();

  //real_function_t f = real_factory_t(mad_conv.get_world()).f(u_exact);
  //f.set_autorefine(true);


  std::cout << "MAD function has " << functions[0].min_nodes() << " nodes before convolution" << std::endl;

  //f.make_nonstandard(false, true);
  //madness::make_nonstandard(world, functions);

  std::vector<real_function_t> opf;
  if (mad_convs.size() == 1) {
    opf = (*mad_convs[0])(functions);
  } else {
    opf = madness::apply(world, mad_convs, functions);
  }
  // std::cout << "Tree State of f: " << f.get_impl()->get_tree_state() << std::endl;
  return std::make_tuple(std::move(functions), std::move(opf));
}


template<typename T, mra::Dimension NDIM>
void test_convolution(int num_batches, std::size_t N, size_type K, T precision, int max_level,
                     T verification_precision, std::vector<std::shared_ptr<real_convolution_t>>& mad_convs, bool print_dot) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto functiondata2 = mra::FunctionData<T,NDIM>(2*K);


  auto pmap = make_procmap<NDIM>(N, num_batches);
  auto dmap = make_devicemap<NDIM>(pmap);

  // define N Gaussians, don't instantiate
  auto gaussians = make_functionset<mra::Gaussian<T, NDIM>>(pmap.batch_manager());

  mra::GaussianConvolutionOperator<T, NDIM> op(mad_convs);

  // generate MADNESS comparison
  auto [madfunc, madconv] = compute_conv_madness<T, NDIM>(N, mad_convs);

  /**
   * Feed the MADNESS functions in reconstructed form into MRA/TTG and perform convolution.
   * Then compare the results.
   */
  {
    std::cout << "Testing MADNESS RECONSTRUCTED function trees" << std::endl;

    std::map<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> cmap;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> reconstruct_conv_result;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result,
                                                                    convolution_result;
    ttg::Edge<mra::Key<NDIM>, void> load_control;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> load_vmra;

    std::vector<real_function_t> madconv_mra(N);
    std::vector<real_function_t> madfunc_mra(N);
    for (size_type i = 0; i < N; ++i) {
      madconv_mra[i].set_impl(madfunc[i], false);
      madfunc_mra[i].set_impl(madfunc[i], false);
    }

    // put the MADNESS function into reconstructed form, then compress using MRA/TTG
    madness::reconstruct(mad_convs[0]->get_world(), madfunc);
    auto start            = make_start(gaussians, load_control);
    auto load_tt          = mra::vmra::make_vmra_load(madfunc, load_control, load_vmra, "load_vmra");
    auto compress         = make_compress(gaussians, K, true, functiondata, load_vmra, compress_result, "compress");
    auto extract          = mra::vmra::make_vmra_store(madfunc_mra, compress_result, madness::TreeState::nonstandard, "store_func");
    auto convolve         = make_convolution(gaussians, K, compress_result, convolution_result, op, precision, "convolution");
    auto reconstruct_conv = make_reconstruct(gaussians, K, true, functiondata, convolution_result, reconstruct_conv_result, "reconstruct_convolution");
    auto store_tt         = mra::vmra::make_vmra_store(madconv_mra, reconstruct_conv_result, madness::TreeState::reconstructed, "store_conv");
    auto connected        = make_graph_executable(start.get());
    assert(connected);

    std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
    if (ttg::default_execution_context().rank() == 0) {
      // beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke();
    }
    ttg::execute();
    ttg::fence();

    madness::make_nonstandard(mad_convs[0]->get_world(), madfunc);
    compare_mra_madness(madfunc, madfunc_mra, "madfunc_result", verification_precision, false);
    compare_mra_madness(madconv, madconv_mra, "madconv_result_from_reconstruct", verification_precision, true);
  }

  /**
   * Feed the MADNESS functions in comressed form into MRA/TTG and perform convolution.
   * Then compare the results.
   * TODO: the load implementation for compressed MADNESS function trees is not working yet.
   *       It seems that the child information is incorrect. That needs to be fixed before this test can be enabled.
   *       For now, we will just skip this test since we usually get the MADNESS functions in reconstructed form anyway.
   */
//#if 0
  {

    std::cout << "Testing MADNESS COMPRESSED function trees" << std::endl;

    ttg::Edge<mra::Key<NDIM>, void> load_control;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> load_vmra;

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> reconstruct_conv_result, reconstruct_result;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> convolution_result, compress_result;
    std::vector<real_function_t> madconv_mra(N);
    for (size_type i = 0; i < N; ++i) {
      madconv_mra[i].set_impl(madfunc[i], false);
    }

    // put the MADNESS function into reconstructed form, then compress using MRA/TTG
    //madness::make_nonstandard(mad_convs[0]->get_world(), madfunc);
    madness::compress(mad_convs[0]->get_world(), madfunc);
    auto start            = make_start(gaussians, load_control);
    all_tts.push_back(start.get());
    auto load_tt          = mra::vmra::make_vmra_load(madfunc, load_control, load_vmra, "load_vmra");
    all_tts.push_back(load_tt.get());
    // have to reconstruct and compress into nonstandard form
    auto reconstruct      = make_reconstruct(gaussians, K, false, functiondata, load_vmra, reconstruct_result, "reconstruct");
    all_tts.push_back(reconstruct.get());
    auto compress_ns      = make_compress(gaussians, K, true, functiondata, reconstruct_result, compress_result, "compress");
    all_tts.push_back(compress_ns.get());
    auto convolve         = make_convolution(gaussians, K, compress_result, convolution_result, op, precision, "convolution");
    all_tts.push_back(convolve.get());
    auto reconstruct_conv = make_reconstruct(gaussians, K, true, functiondata, convolution_result, reconstruct_conv_result, "reconstruct_convolution");
    all_tts.push_back(reconstruct_conv.get());
    auto store_tt         = mra::vmra::make_vmra_store(madconv_mra, reconstruct_conv_result, madness::TreeState::reconstructed, "store_vmra");
    all_tts.push_back(store_tt.get());
    auto connected        = make_graph_executable(start.get());
    assert(connected);

    std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
    if (ttg::default_execution_context().rank() == 0) {

      std::cout << "==== begin dot ====\n";
      std::cout << ttg::Dot(true)(start.get()) << std::endl;
      std::cout << "====  end dot  ====\n";

      // beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke();
    }
    ttg::execute();
    ttg::fence();

    compare_mra_madness(madconv, madconv_mra, "madconv_result_from_compressed", verification_precision, true);
  }
//#endif // 0
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
  //int num_batches = opt.parse("-b", 1); // batch size for the test, default is 0 (select automatically)
  int num_batches = 1; // for now the check only support num_batches=1, which means all functions are in the same batch. We will enable num_batches>1 later, which will require some changes in the test code to handle multiple batches and also changes in the MRA code to support convolution with functions in different batches (e.g., by doing batch-wise convolution and then merging results).
  int op_rank = opt.parse("-r", 1); // number of times to repeat the test for timing purposes
  int num_ops = opt.parse("-O", 1); // number of operators to use in the convolution, default is 1
  Length = opt.parse("-d", Length);
  bool norand = opt.exists("-norand");
  bool print_dot = opt.exists("-dot");
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
  std::vector< std::shared_ptr< madness::Convolution1D<double> > > ops_1d(op_rank);
  std::vector<std::shared_ptr<real_convolution_t>> mad_convs;
  for (int o = 0; o < num_ops; ++o) {
    for (int i = 0; i < op_rank; ++i) {
      ops_1d[i].reset(new madness::GaussianConvolution1D<double>(K, (2*(o+1)+1)/(i+1)*100, 1/(i+1)*100, 0, madness::LatticeRange()));
    }
    mad_convs.push_back(std::make_shared<real_convolution_t>(world, ops_1d, K));
  }


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
                             std::pow(10, -verification_log_precision), mad_convs, print_dot);

  mra::finalize();
}
