/**
 * Tests of operations on sets of functions against MADNESS:
 *  - make_inner / make_matrix_inner
 *  - make_transform (with and without make_truncate)
 *  - FunctionVector (store, feed with and without keep)
 *  - make_multiply_broadcast
 *
 * The functions are Gaussians with different centers and exponents so that
 * their trees differ. Every MADNESS function written back by make_vmra_store
 * is also checked for a well-formed tree. Returns non-zero if any check fails.
 */

#include <ttg.h>
#include "mra/mra.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <madness/mra/mra.h>
#include <madness/world/world.h>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

#include "mra/vmra/vmra.h"

using T = double;
constexpr mra::Dimension NDIM = 3;
using coord_t = madness::Vector<double, 3>;
using real_function_t = madness::Function<double, 3>;
using vec_t = std::vector<real_function_t>;
using CompressedEdge = ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>;
using ReconstructedEdge = ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>;

static double Length = 6.0;
static int failures = 0;

static void check(bool ok, const std::string& what, double err, double tol) {
  if (ttg::default_execution_context().rank() == 0) {
    std::cout << (ok ? "PASSED " : "FAILED ") << what << ": error " << err << " (tolerance " << tol << ")" << std::endl;
  }
  if (!ok) ++failures;
}

static void check_close(const std::string& what, double err, double scale, double rel_tol = 1e-12) {
  const double tol = rel_tol*std::max(1.0, scale);
  check(err <= tol, what, err, tol);
}

/// exp(-a |r - c|^2)
struct GaussianFunctor : public madness::FunctionFunctorInterface<double, 3> {
  double expnt;
  coord_t center;
  GaussianFunctor(double expnt, coord_t center) : expnt(expnt), center(center) {}
  double operator()(const coord_t& r) const override {
    double rsq = 0.0;
    for (int d = 0; d < 3; ++d) rsq += (r[d] - center[d])*(r[d] - center[d]);
    return std::exp(-expnt*rsq);
  }
};

static real_function_t gaussian(madness::World& world, double expnt, coord_t center) {
  std::shared_ptr<madness::FunctionFunctorInterface<double, 3>> functor = std::make_shared<GaussianFunctor>(expnt, center);
  return madness::FunctionFactory<double, 3>(world).functor(functor);
}

static vec_t gaussians(madness::World& world, const std::vector<std::pair<double, coord_t>>& params) {
  vec_t v;
  for (auto& [expnt, center] : params) v.push_back(gaussian(world, expnt, center));
  return v;
}

static auto make_functions(mra::size_type N) {
  auto pmap = mra::make_procmap<NDIM>(N, 1);
  return mra::make_functionset<mra::Gaussian<T, NDIM>>(pmap.batch_manager());
}

/// the tasks of one graph, started at the root key of a single batch
struct Graph {
  ttg::Edge<mra::Key<NDIM>, void> control;
  std::vector<std::unique_ptr<ttg::TTBase>> tts;
  mra::size_type K;
  mra::FunctionData<T, NDIM> functiondata;

  explicit Graph(mra::size_type K) : K(K), functiondata(K) {}

  void add(auto&& tt) { tts.push_back(std::move(tt)); }

  void execute() {
    auto fns = make_functions(1);
    auto start = mra::make_start(fns, control);
    [[maybe_unused]] bool connected = make_graph_executable(start.get());
    assert(connected);
    if (ttg::default_execution_context().rank() == 0) start->invoke();
    ttg::execute();
    ttg::fence();
  }

  /// MADNESS functions as compressed nodes on `out` (loaded reconstructed, compressed in the graph)
  void load_compressed(vec_t& f, CompressedEdge& out, const std::string& name) {
    madness::reconstruct(f.front().world(), f);
    ReconstructedEdge reconstructed;
    add(mra::vmra::make_vmra_load(f, control, reconstructed, "load_" + name));
    add(mra::make_compress(make_functions(f.size()), K, false, functiondata, reconstructed, out, "compress_" + name));
  }
};

/// empty functions with the process map of `like`, for make_vmra_store
static vec_t empty_like(const real_function_t& like, std::size_t n) {
  vec_t v(n);
  for (auto& f : v) f.set_impl(like, false);
  return v;
}

static double max_diff_norm(madness::World& world, const vec_t& a, const vec_t& b) {
  auto d = madness::norm2s(world, madness::sub(world, a, b));
  return *std::max_element(d.begin(), d.end());
}

static double max_norm(madness::World& world, const vec_t& a) {
  auto n = madness::norm2s(world, a);
  return *std::max_element(n.begin(), n.end());
}

/**
 * MADNESS tree invariants: an interior node has all children, a leaf none,
 * and every node but the root has a parent. Only checked on a single process.
 */
static void check_well_formed(madness::World& world, const vec_t& v, const std::string& what) {
  if (world.size() != 1) return;
  long bad = 0;
  for (const auto& f : v) {
    const auto& coeffs = f.get_impl()->get_coeffs();
    if (coeffs.size() == 0) ++bad;   // not even a root
    for (auto& [key, node] : coeffs) {
      if (key.level() > 0 && coeffs.find(key.parent()).get() == coeffs.end()) ++bad;
      long present = 0;
      for (madness::KeyChildIterator<3> kit(key); kit; ++kit) {
        if (coeffs.find(kit.key()).get() != coeffs.end()) ++present;
      }
      if (node.has_children() ? present != 8 : present != 0) ++bad;
    }
  }
  check(bad == 0, what + " is a well-formed MADNESS tree", bad, 0);
}

static void test_inner(madness::World& world, mra::size_type K, vec_t& f, vec_t& g, vec_t& h) {
  const mra::size_type N = f.size(), M = g.size();

  /* <f_i|h_i> */
  {
    Graph G(K);
    CompressedEdge fe, he;
    ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> result_e;
    G.load_compressed(f, fe, "f");
    G.load_compressed(h, he, "h");
    G.add(mra::make_inner<T, NDIM>(make_functions(N), K, fe, he, result_e));
    madness::Tensor<double> result(N);
    G.add(ttg::make_tt([&](const mra::Key<NDIM>&, const mra::DenseTensor<T, 1>& r) {
      for (mra::size_type i = 0; i < N; ++i) result(i) = r.current_view().data()[i];
    }, ttg::edges(result_e), ttg::edges(), "sink"));
    G.execute();
    world.gop.sum(result.ptr(), result.size());
    auto ref = madness::inner(world, f, h);
    check_close("inner(f,h)", (result - ref).absmax(), ref.absmax());
  }

  /* <f_i|g_j>, and <f_i|f_j> with one edge into both inputs */
  for (bool same : {false, true}) {
    vec_t& b = same ? f : g;
    const mra::size_type Mb = same ? N : M;
    Graph G(K);
    CompressedEdge fe, ge;
    ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>> result_e;
    G.load_compressed(f, fe, "f");
    if (!same) G.load_compressed(g, ge, "g");
    G.add(mra::make_matrix_inner<T, NDIM>(make_functions(N), make_functions(Mb), K, fe, same ? fe : ge, result_e));
    madness::Tensor<double> result(N, Mb);
    G.add(ttg::make_tt([&](const mra::Key<NDIM>&, const mra::DenseTensor<T, 2>& r) {
      const T* data = r.current_view().data();
      for (mra::size_type i = 0; i < N; ++i)
        for (mra::size_type j = 0; j < Mb; ++j) result(i, j) = data[i*Mb + j];
    }, ttg::edges(result_e), ttg::edges(), "sink"));
    G.execute();
    world.gop.sum(result.ptr(), result.size());
    auto ref = madness::matrix_inner(world, f, b);
    check_close(same ? "matrix_inner(f,f)" : "matrix_inner(f,g)", (result - ref).absmax(), ref.absmax());
  }
}

/// U as the device matrix make_transform expects
static auto device_matrix(const madness::Tensor<double>& U) {
  const mra::size_type N = U.dim(0), M = U.dim(1);
  auto Ud = std::make_shared<mra::DenseTensor<T, 2>>(std::array<mra::size_type, 2>{N, M});
  T* p = Ud->buffer().host_ptr();
  for (mra::size_type i = 0; i < N; ++i)
    for (mra::size_type j = 0; j < M; ++j) p[i*M + j] = U(i, j);
  return Ud;
}

static void test_transform(madness::World& world, mra::size_type K, vec_t& f, double thresh) {
  const long N = f.size();
  /* dense square U */
  madness::Tensor<double> Udense(N, N);
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j) Udense(i, j) = 1.0/(1.0 + std::abs(i - j));
  /* sparse rectangular U: identity, a column mixing two functions, a zero column */
  madness::Tensor<double> Usparse(N, N + 2);
  for (long i = 0; i < N; ++i) Usparse(i, i) = 1.0;
  Usparse(0, N) = 0.5;
  Usparse(N - 1, N) = -0.5;

  for (const auto& [name, U] : {std::pair<std::string, madness::Tensor<double>>{"dense", Udense},
                                std::pair<std::string, madness::Tensor<double>>{"sparse", Usparse}}) {
    for (bool truncate : {false, true}) {
      const std::string what = "transform " + name + (truncate ? " + truncate" : "");
      const mra::size_type M = U.dim(1);
      Graph G(K);
      CompressedEdge in_e, out_e, trunc_e;
      G.load_compressed(f, in_e, "f");
      auto fns_out = make_functions(M);
      G.add(mra::make_transform<T, NDIM>(make_functions(N), fns_out, K, device_matrix(U), in_e, out_e));
      if (truncate) {
        G.add(mra::make_truncate(fns_out, K, thresh, madness::FunctionDefaults<3>::get_truncate_mode(),
                                 madness::FunctionDefaults<3>::get_cell_min_width(), out_e, trunc_e, "truncate"));
      }
      vec_t result = empty_like(f.front(), M);
      G.add(mra::vmra::make_vmra_store(result, truncate ? trunc_e : out_e, true /* add_leaves */));
      G.execute();

      vec_t ref = madness::transform(world, f, U, 0.0, true);
      if (truncate) madness::truncate(world, ref, thresh);
      madness::compress(world, ref);
      check_close(what, max_diff_norm(world, ref, result), max_norm(world, ref),
                  truncate ? 10*thresh : 1e-12);
      check_well_formed(world, result, what);
    }
  }
}

static void test_function_vector(madness::World& world, mra::size_type K, vec_t& f) {
  const mra::size_type N = f.size();
  auto fns = make_functions(N);
  mra::FunctionVector<T, NDIM> v;

  /* store */
  {
    Graph G(K);
    CompressedEdge e;
    G.load_compressed(f, e, "f");
    G.add(v.make_store(e, N));
    G.execute();
  }
  const bool stored = v.size() == N && (world.size() > 1 || !v.empty());
  check(stored, "FunctionVector store", stored ? 0 : 1, 0);

  /* feed with keep, twice: the vector is unchanged */
  auto ref_norms = madness::norm2s(world, f);
  for (int round = 0; round < 2; ++round) {
    const auto nodes_before = v.num_local_nodes();
    Graph G(K);
    CompressedEdge e;
    ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> norm_e;
    G.add(v.make_feed(G.control, e, true));
    G.add(mra::make_norm(fns, K, e, norm_e));
    std::vector<double> norms(N, 0.0);
    G.add(ttg::make_tt([&](const mra::Key<NDIM>&, const mra::DenseTensor<T, 1>& n) {
      for (mra::size_type i = 0; i < N; ++i) norms[i] = n.current_view().data()[i];
    }, ttg::edges(norm_e), ttg::edges(), "sink"));
    G.execute();
    world.gop.sum(norms.data(), norms.size());
    double err = 0.0;
    for (mra::size_type i = 0; i < N; ++i) err = std::max(err, std::abs(norms[i] - ref_norms[i]));
    check_close("FunctionVector feed with keep (round " + std::to_string(round) + "): norms", err, 1.0);
    check(v.num_local_nodes() == nodes_before, "FunctionVector feed with keep: nodes restored",
          double(v.num_local_nodes()) - double(nodes_before), 0);
  }

  /* feed without keep and store into the same vector: v <- transform(v, U) */
  madness::Tensor<double> U(static_cast<long>(N), static_cast<long>(N));
  for (long i = 0; i < long(N); ++i)
    for (long j = 0; j < long(N); ++j) U(i, j) = (i == j) ? 1.0 : 0.1*(i + 1);
  {
    Graph G(K);
    CompressedEdge in_e, out_e;
    G.add(v.make_feed(G.control, in_e, false));
    G.add(mra::make_transform<T, NDIM>(fns, fns, K, device_matrix(U), in_e, out_e));
    G.add(v.make_store(out_e, N));
    G.execute();
  }
  vec_t ref = madness::transform(world, f, U, 0.0, true);
  madness::compress(world, ref);

  /* overlap through a kept vector (one edge into both inputs plus the restore) */
  {
    Graph G(K);
    CompressedEdge e;
    ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>> result_e;
    G.add(v.make_feed(G.control, e, true));
    G.add(mra::make_matrix_inner<T, NDIM>(fns, fns, K, e, e, result_e));
    madness::Tensor<double> S(static_cast<long>(N), static_cast<long>(N));
    G.add(ttg::make_tt([&](const mra::Key<NDIM>&, const mra::DenseTensor<T, 2>& r) {
      const T* data = r.current_view().data();
      for (mra::size_type i = 0; i < N; ++i)
        for (mra::size_type j = 0; j < N; ++j) S(i, j) = data[i*N + j];
    }, ttg::edges(result_e), ttg::edges(), "sink"));
    G.execute();
    world.gop.sum(S.ptr(), S.size());
    auto Sref = madness::matrix_inner(world, ref, ref);
    check_close("FunctionVector in-place transform: overlap", (S - Sref).absmax(), Sref.absmax());
  }

  /* back to MADNESS */
  {
    Graph G(K);
    CompressedEdge e;
    vec_t result = empty_like(f.front(), N);
    G.add(v.make_feed(G.control, e, true));
    G.add(mra::vmra::make_vmra_store(result, e, true /* add_leaves */));
    G.execute();
    check_close("FunctionVector in-place transform: functions", max_diff_norm(world, ref, result), max_norm(world, ref));
    check_well_formed(world, result, "FunctionVector download");
  }
}

static void test_multiply_broadcast(madness::World& world, mra::size_type K, real_function_t& a, vec_t& b) {
  const mra::size_type N = b.size();
  auto D = std::make_unique<mra::Domain<NDIM>[]>(1);
  D[0].set_cube(-Length, Length);
  auto db = ttg::Buffer<mra::Domain<NDIM>>(std::move(D), 1);

  vec_t av = {a};
  madness::reconstruct(world, av);
  madness::reconstruct(world, b);
  Graph G(K);
  ReconstructedEdge a_e, b_e, out_e;
  G.add(mra::vmra::make_vmra_load(av, G.control, a_e, "load_a"));
  G.add(mra::vmra::make_vmra_load(b, G.control, b_e, "load_b"));
  G.add(mra::make_multiply_broadcast<T, NDIM>(db, make_functions(1), make_functions(N), G.functiondata, K, a_e, b_e, out_e));
  vec_t result = empty_like(b.front(), N);
  G.add(mra::vmra::make_vmra_store(result, out_e));
  G.execute();

  vec_t ref = madness::mul(world, a, b);
  check_close("multiply_broadcast", max_diff_norm(world, ref, result), max_norm(world, ref));
  check_well_formed(world, result, "multiply_broadcast");
}

int main(int argc, char** argv) {
  auto opt = mra::OptionParser(argc, argv);
  mra::size_type K = opt.parse("-K", 8);
  int log_precision = opt.parse("-p", 6);
  int cores = opt.parse("-c", -1);
  const double thresh = std::pow(10.0, -log_precision);

  mra::initialize(argc, argv, cores);

  madness::FunctionDefaults<3>::set_cubic_cell(-Length, Length);
  madness::FunctionDefaults<3>::set_k(K);
  madness::FunctionDefaults<3>::set_thresh(thresh);
  madness::FunctionDefaults<3>::set_refine(true);
  madness::FunctionDefaults<3>::set_autorefine(true);
  madness::FunctionDefaults<3>::set_initial_level(2);

  {
    madness::World world(SafeMPI::COMM_WORLD);
    const coord_t c0{0.0, 0.0, 0.0}, c1{0.7, -0.4, 0.2}, c2{-1.1, 0.5, 0.0}, c3{0.3, 1.2, -0.8};
    vec_t f = gaussians(world, {{40.0, c0}, {15.0, c1}, {80.0, c2}});
    vec_t g = gaussians(world, {{25.0, c1}, {60.0, c3}, {10.0, c0}, {120.0, c2}});
    vec_t h = gaussians(world, {{30.0, c3}, {90.0, c0}, {5.0, c2}});
    /* deeper than f near c3, shallower near c2 */
    real_function_t a = gaussian(world, 150.0, c3) + gaussian(world, 2.0, c0);

    test_inner(world, K, f, g, h);
    test_transform(world, K, f, thresh);
    test_function_vector(world, K, f);
    test_multiply_broadcast(world, K, a, f);

    if (ttg::default_execution_context().rank() == 0) {
      std::cout << (failures == 0 ? "All tests passed" : std::to_string(failures) + " test(s) failed") << std::endl;
    }
  }
  mra::finalize();
  return failures == 0 ? 0 : 1;
}
