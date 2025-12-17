#include <ttg.h>
#include "mra/mra.h"
#include <any>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;


template<typename T, mra::Dimension NDIM>
void test_pcr(std::size_t N, std::size_t K,
              int num_batches, int max_level,
              int seed, int initial_level,
              T root_radius, T expnt,
              T domain_size, bool print_dot)
{
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-domain_size, domain_size);

  if (seed > 0) {
    srand48(seed);
  }

  auto pmap = make_procmap<NDIM>(N, num_batches);
  auto dmap = make_devicemap<NDIM>(pmap);

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> project_result, reconstruct_result, multiply_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_result, compress_reconstruct_result, gaxpy_result;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> norm_result;

  // define N Gaussians
  //std::cout << "Defining " << N << " Gaussians with initial level "
  //          << initial_level << " and seed " << seed
  //          << " in " << pmap.batch_manager()->num_batches() << " batches" << std::endl;
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
    std::cout << N << " Gaussians with expnt " << 1500
              << " in " << gaussians->num_batches() << " batches, "
              << gaussians->num_local_functions() << " of "
              << N << " functions are local in "
              << gaussians->num_local_batches() << " local batches" << std::endl;
  }

  // put it into a buffer
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);
  auto start = make_start(gaussians, project_control);
  auto project = make_project(db, gaussians, K, max_level, functiondata, T(1e-6), project_control, project_result, "project", pmap, dmap);
  // C(P)
  auto compress = make_compress(gaussians, K, functiondata, project_result, compress_result, "compress-cp", pmap, dmap);
  // // R(C(P))
  auto reconstruct = make_reconstruct(gaussians, K, functiondata, compress_result, reconstruct_result, "reconstruct-rcp", pmap, dmap);
  // C(R(C(P)))
  auto compress_r = make_compress(gaussians, K, functiondata, reconstruct_result, compress_reconstruct_result, "compress-crcp", pmap, dmap);

  // C(R(C(P))) - C(P)
  auto gaxpy = make_gaxpy(T(1.0), T(-1.0), gaussians, K, compress_reconstruct_result, compress_result, gaxpy_result, "gaxpy", pmap, dmap);
  // | C(R(C(P))) - C(P) |
  auto norm  = make_norm(gaussians, K, gaxpy_result, norm_result, "norm", pmap, dmap);
  // final check
  auto norm_check = ttg::make_tt([&](const mra::Key<NDIM>& key, const mra::Tensor<T, 1>& norms){
    // TODO: check for the norm within machine precision
    auto norms_view = norms.current_view();
    for (size_type i = 0; i < norms_view.size(); ++i) {
      if (std::abs(norms_view[i]) > 1e12) {
        std::cout << "Final norm " << i << " in batch " << key.batch() << " : " << norms_view[i] << std::endl;
      }
    }
  }, ttg::edges(norm_result), ttg::edges(), "norm-check");

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
    // std::cout << "Is everything connected? " << connected << std::endl;
    if (print_dot) {
      std::cout << "==== begin dot ====\n";
      std::cout << ttg::Dot(true)(start.get()) << std::endl;
      std::cout << "====  end dot  ====\n";
    }

    beg = std::chrono::high_resolution_clock::now();
  }
  ttg::execute();

  if (ttg::default_execution_context().rank() == 0) {
    // This kicks off the entire computation
    // NOTE: we need to do this after ttg::execute()
    //       so we have TTG properly set up.
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
  int nrep = opt.parse("-n", 3);
  bool norand = opt.exists("-norand");
  int max_level = opt.parse("-l", -1);
  int cores   = opt.parse("-c", -1); // -1: use all cores
  int initial_level = opt.parse("-i", 0); // initial level for the Gaussian functions, default is using a heuristic
  int seed    = opt.parse("-s", norand ? 0 : 5551212); // seed for random number generator, 0 for deterministic
  int num_batches = opt.parse("-b", 0); // batch size for the test, default is 0 (select automatically)
  double root_radius = opt.parse("-r", 2.0); // radius of the root domain cube
  double expnt = opt.parse("-e", 1500.0); // default: 1000.0
  double domain_size = opt.parse("-d", 6.0); // size of the domain cube [-d,d]
  bool print_dot = opt.exists("-dot");

  ttg::initialize(argc, argv, cores);
  mra::GLinitialize();
  allocator_init(argc, argv);

  /**
   * TODO: we currently cannot precreate a TTG and run it because make_reconstruct primes the
   * with the first key it receives. We need to find a way to do that automatically outside of make_project.
   */
  for (int i = 0; i < nrep; ++i) {
    test_pcr<double, 3>(N, K, num_batches, max_level, seed, initial_level, root_radius, expnt, domain_size, print_dot);
  }

  allocator_fini();
  ttg::finalize();
}
