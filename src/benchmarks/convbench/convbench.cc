/**
 * Benchmark for the convolution kernel (mra/kernels/convolution.h) using real
 * MADNESS separated convolution operators (Coulomb, BSH, Slater).
 *
 * This is a synthetic sweep: no function trees are built. For every
 * (operator, level, displacement) point we build the operator data exactly
 * like the convolution TTG does (GaussianConvolutionOperator::get_op), fill
 * one input node with N random functions of prescribed norm, and launch the
 * kernel for that node from many concurrent device tasks, mirroring
 * accumulate_tt in mra/tasks/convolution.h (shell0_tt for displacement 0,
 * where there is no pre-existing accumulator node). The batched path
 * (ttg::device::coop) is used whenever mra batching is enabled (-B > 1),
 * just like in the real task graph.
 *
 * The work the kernel performs depends on the operator's screening: a
 * function is skipped if cnorm*opnorm <= tol/fac, and each separated term mu
 * is only applied if its munorm exceeds 0.01*tol/(fac*cnorm*rank) (and its R
 * or S norm is non-zero). We replicate that screening on the host to report
 * how many terms survive at each point and to count the useful flops (the
 * mTxm contractions of the kept terms only).
 *
 * Displacements are canonical representatives (a >= b >= c >= 0) of each
 * symmetry class: the Gaussian expansions are symmetric in each coordinate
 * and identical across dimensions, so these cover all displacements.
 */

#include <ttg.h>
#include "mra/mra.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

using namespace mra;

namespace {

  constexpr Dimension NDIM = 3;
  using T = double;
  using mad_op_t = madness::SeparatedConvolution<T, NDIM>;
  using disp_t = std::array<Translation, NDIM>;

  std::vector<std::string> split(std::string_view s, char sep = ',') {
    std::vector<std::string> out;
    std::stringstream ss{std::string(s)};
    std::string item;
    while (std::getline(ss, item, sep)) {
      if (!item.empty()) out.push_back(item);
    }
    return out;
  }

  std::shared_ptr<mad_op_t> make_mad_op(madness::World& world, const std::string& name,
                                        double lo, double eps, double mu, double gamma) {
    if (name == "coulomb") {
      return std::shared_ptr<mad_op_t>(madness::CoulombOperatorPtr(world, lo, eps));
    } else if (name == "bsh") {
      return std::shared_ptr<mad_op_t>(madness::BSHOperatorPtr3D(world, mu, lo, eps));
    } else if (name == "slater") {
      return std::shared_ptr<mad_op_t>(madness::SlaterOperatorPtr(world, gamma, lo, eps));
    }
    throw std::runtime_error("convbench: unknown operator '" + name + "' (expected coulomb, bsh, or slater)");
  }

  /* Canonical displacements with max component <= R, ordered by distance. */
  std::vector<disp_t> make_displacements(int R) {
    std::vector<disp_t> disps;
    for (Translation a = 0; a <= R; ++a) {
      for (Translation b = 0; b <= a; ++b) {
        for (Translation c = 0; c <= b; ++c) {
          disps.push_back({a, b, c});
        }
      }
    }
    std::sort(disps.begin(), disps.end(), [](const disp_t& x, const disp_t& y) {
      auto dx = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
      auto dy = y[0]*y[0] + y[1]*y[1] + y[2]*y[2];
      return (dx != dy) ? dx < dy : x > y;
    });
    return disps;
  }

  std::string disp_str(const disp_t& d) {
    std::stringstream ss;
    ss << d[0] << ":" << d[1] << ":" << d[2];
    return ss.str();
  }

  /**
   * Host-side replica of the kernel's screening (convolution_kernel_impl,
   * apply_conv, muopxv_fast and accel::apply_conv_k) for one function norm.
   */
  struct TermStats {
    bool active = false; // passes cnorm*opnorm > tol/fac
    size_type kept_r = 0;
    size_type kept_s = 0;
  };

  template <typename NormsView>
  TermStats screen_terms(const NormsView& norms, T cnorm, T tol, T fac,
                         const std::array<bool, 2>& at) {
    TermStats stats;
    const T opnorm = norms(0, 0, 0, (size_type)NormId::Opnorm);
    if (!(cnorm * opnorm > tol / fac)) return stats;
    stats.active = true;
    const size_type rank = norms(0, 0, 0, (size_type)NormId::Rank);
    const T optol = 0.01 * (tol / fac / cnorm) / rank;
    for (size_type mu = 0; mu < rank; ++mu) {
      if (!(norms(0, mu, 0, (size_type)NormId::MUnorm) > optol)) continue;
      double rnorm = 1.0, snorm = 1.0;
      for (Dimension d = 0; d < NDIM; ++d) {
        rnorm *= norms(0, mu, d, (size_type)NormId::Rnorm);
        snorm *= norms(0, mu, d, (size_type)NormId::Snorm);
      }
      if (at[0] && rnorm > 1.e-20) ++stats.kept_r;
      if (at[1] && snorm > 0.0) ++stats.kept_s;
    }
    return stats;
  }

  /* Flops of the NDIM mTxm contractions of one term applied to one function
   * with dimension k per direction: NDIM * 2 * k^(NDIM+1). */
  double term_flops(size_type k) {
    return NDIM * 2.0 * std::pow(double(k), NDIM + 1);
  }

  /**
   * Everything the benchmark tasks read for the current sweep point. Owned by
   * the host loop and replaced between executions; tasks only read it.
   */
  struct PointState {
    Key<NDIM> key;          // destination node
    Key<NDIM> displacement; // key - source
    std::shared_ptr<const ConvolutionData<T, NDIM>> op_data;
    FunctionsCompressedNode<T, NDIM> in; // accumulator; empty for displacement 0 (shell0)
    FunctionsCompressedNode<T, NDIM> f;  // contribution the operator is applied to
    T tol;
    std::array<bool, 2> at;
    bool with_resnorms;
  };

  void fill_random(FunctionsCompressedNode<T, NDIM>& node, size_type N,
                   const std::vector<T>& fnorms, std::mt19937_64& rng) {
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    auto view = node.coeffs().current_view();
    for (size_type i = 0; i < N; ++i) {
      auto fi = view(i);
      T* ptr = fi.data();
      T sumsq = 0.0;
      for (size_type j = 0; j < fi.size(); ++j) {
        ptr[j] = dist(rng);
        sumsq += ptr[j]*ptr[j];
      }
      const T scale = fnorms[i] / std::sqrt(sumsq);
      for (size_type j = 0; j < fi.size(); ++j) ptr[j] *= scale;
    }
  }

  struct Config {
    std::vector<std::string> ops;
    size_type K;
    size_type N;
    int lmin, lmax, R;
    int ntasks, nreps;
    T thresh, lo, mu, gamma, L;
    T cnorm, cspread;
    bool all_points;
    bool with_resnorms;
    unsigned long seed;
  };

  void run_convbench(const Config& cfg) {
    const size_type K = cfg.K;
    const size_type N = cfg.N;
    const bool is_root = ttg::default_execution_context().rank() == 0;

    /* same shell volume as make_convolution: expected number of contributions per box */
    const double radius = 1.5 + 0.33 * std::max(0.0, 2 - std::log10(cfg.thresh) - K);
    const T fac = vol_nsphere(NDIM, radius);

    std::unique_ptr<PointState> state;

    const bool enable_batching = mra::batching_enabled();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::ConvolutionBatchArg<T, NDIM>>> conv_pool;
    if (enable_batching) {
      conv_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::ConvolutionBatchArg<T, NDIM>>>(
                    ttg::device::num_devices(), mra::get_batch_size());
    }
#else
    std::nullptr_t conv_pool = nullptr;
#endif // MRA_ENABLE_HOST

    ttg::Edge<int, void> e;

    auto start = ttg::make_tt([&](){
      for (int i = 0; i < cfg.ntasks; ++i) {
        ttg::sendk<0>(i);
      }
    }, ttg::edges(), ttg::edges(e), "start");

    /* Mirrors accumulate_tt (and shell0_tt if s.in is empty). */
    auto conv_tt = ttg::make_tt<Space>([&, K, N, fac, enable_batching, conv_pool](const int& id) -> TASKTYPE {
      const PointState& s = *state;
      const auto& in_node = s.in;
      const auto& f_node = s.f;

      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      if (in_node.empty()) {
        sparsity.nonzero_if_any(f_node);
      } else {
        sparsity.nonzero_if_any(in_node, f_node);
      }
      const size_type n_nonzero = sparsity.count_nonzero();

      FunctionsCompressedNode<T, NDIM> out(s.key, N);
      out.allocate(sparsity, K, ttg::scope::Allocate);

      DenseTensor<T, 1> resnorms;
      if (s.with_resnorms) {
        resnorms = DenseTensor<T, 1>(N, ttg::scope::Allocate);
      }
      T tol = s.tol;
      std::array<bool, 2> at = s.at;
      auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K) * n_nonzero, TempScope);
      const auto& op_data = s.op_data;

#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(f_node.coeffs().buffer(), out.coeffs().buffer(), tmp);
      if (!in_node.empty()) {
        input.add(in_node.coeffs().buffer());
      }
      input.add(op_data->norms.buffer());
      for (Dimension d = 0; d < NDIM; ++d) {
        input.add(op_data->data[d]->R.buffer());
        input.add(op_data->data[d]->S.buffer());
      }
      if (s.with_resnorms) {
        input.add(resnorms.buffer());
      }
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

      auto transr = std::array{op_data->data[0]->R.current_view(), op_data->data[1]->R.current_view(), op_data->data[2]->R.current_view()};
      auto transs = std::array{op_data->data[0]->S.current_view(), op_data->data[1]->S.current_view(), op_data->data[2]->S.current_view()};
      auto opnorms_view = op_data->norms.current_view();
      auto out_view = out.coeffs().current_view();
      auto f_view = f_node.coeffs().current_view();
      auto in_view = in_node.coeffs().current_view();
      auto resnorms_view = resnorms.current_view();

#ifndef MRA_ENABLE_HOST
      if (enable_batching) {
        auto batch = co_await ttg::device::coop<int>(in_view, f_view, out_view, resnorms_view, tmp,
                                                     transr, transs, opnorms_view, tol, at, out.coeffs(),
                                                     n_nonzero, resnorms);
        detail::submit_convolution_batch_leader<T, NDIM>(batch, *conv_pool, K, fac, N);
      } else
#endif // MRA_ENABLE_HOST
      {
        auto sparseman = make_sparsity_manager(out);
        sparseman.populate_device_sparsity();
        submit_convolution_kernel<T, NDIM>(s.key, s.displacement, K, N, n_nonzero, fac, tol, in_view,
                                           f_view, out_view, resnorms_view, transr, transs,
                                           opnorms_view, at,
                                           tmp.current_device_ptr(), ttg::device::current_stream());
      }

#ifndef MRA_ENABLE_HOST
      if (s.with_resnorms) {
        co_await ttg::device::wait(resnorms.buffer());
      }
#endif // MRA_ENABLE_HOST
    }, ttg::edges(e), ttg::edges(), "conv-bench");

#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      conv_tt->set_batch_matcher([](const int&, const int&) { return true; }, mra::get_batch_size());
    }
#endif // MRA_ENABLE_HOST

    auto connected = ttg::make_graph_executable(start.get());
    assert(connected);

    /* per-function norms: log-spaced over cspread decades below cnorm */
    std::vector<T> fnorms(N);
    for (size_type i = 0; i < N; ++i) {
      T frac = (N > 1) ? T(i) / T(N - 1) : T(0);
      fnorms[i] = cfg.cnorm * std::pow(10.0, -cfg.cspread * frac);
    }

    std::mt19937_64 rng(cfg.seed);
    madness::World& world = madness::World::get_default();
    const auto disps = make_displacements(cfg.R);

    if (is_root) {
      std::cout << "# convbench: K=" << K << " N=" << N << " thresh=" << cfg.thresh
                << " lo=" << cfg.lo << " L=" << cfg.L << " cnorm=" << cfg.cnorm
                << " cspread=" << cfg.cspread << " tasks=" << cfg.ntasks
                << " reps=" << cfg.nreps << " batch=" << mra::get_batch_size()
                << " fac=" << fac << " resnorms=" << cfg.with_resnorms << std::endl;
      std::cout << "op,K,N,level,disp,rank,opnorm,active_fns,avg_kept_r,avg_kept_s,"
                   "tasks,batch,t_min_us,t_mean_us,us_per_task,gflop_per_rep,gflops"
                << std::endl;
    }

    for (const auto& opname : cfg.ops) {
      auto mad_op = make_mad_op(world, opname, cfg.lo, cfg.thresh, cfg.mu, cfg.gamma);
      GaussianConvolutionOperator<T, NDIM> op(mad_op);
      if (is_root) {
        std::cout << "# operator " << opname << ": rank " << mad_op->get_rank() << std::endl;
      }

      for (int n = cfg.lmin; n <= cfg.lmax; ++n) {
        for (const auto& disp : disps) {
          /* non-periodic: displacements must stay inside the domain at this level */
          if (*std::max_element(disp.begin(), disp.end()) >= (Translation(1) << n)) continue;

          const bool shell0 = (disp == disp_t{0, 0, 0});
          const Level level = n;
          auto ps = std::make_unique<PointState>();
          ps->key = Key<NDIM>(0, level, disp);
          ps->displacement = ps->key - Key<NDIM>(0, level, {0, 0, 0});
          ps->op_data = op.get_op(level, ps->displacement);
          ps->tol = truncate_tol(ps->key, cfg.thresh, T(1.0), 0);
          ps->at = {true, n > 0};
          ps->with_resnorms = cfg.with_resnorms;

          /* screening statistics, identical to what the kernel will do */
          auto norms = ps->op_data->norms.view_on(ttg::device::Device::host());
          const size_type rank = norms(0, 0, 0, (size_type)NormId::Rank);
          const T opnorm = norms(0, 0, 0, (size_type)NormId::Opnorm);
          size_type active = 0, kept_r = 0, kept_s = 0;
          double flops_per_task = 0.0;
          for (size_type i = 0; i < N; ++i) {
            auto st = screen_terms(norms, fnorms[i], ps->tol, fac, ps->at);
            if (!st.active) continue;
            ++active;
            kept_r += st.kept_r;
            kept_s += st.kept_s;
            flops_per_task += st.kept_r * term_flops(2*K) + st.kept_s * term_flops(K);
          }
          const double avg_r = active ? double(kept_r) / active : 0.0;
          const double avg_s = active ? double(kept_s) / active : 0.0;

          if (active == 0 && !cfg.all_points) {
            /* the screener would never create an accumulate task for this pair */
            if (is_root) {
              std::cout << "# skip " << opname << " level " << n << " disp " << disp_str(disp)
                        << ": screened out (opnorm " << opnorm << ")" << std::endl;
            }
            continue;
          }

          /* all functions non-zero; current_view() in fill_random writes the
           * inline sparsity bytes into the host buffer before the first H2D */
          ps->f = FunctionsCompressedNode<T, NDIM>(ps->key, SparsityInfo(N, SparsityInfo::InitType::AllNonZero), K);
          fill_random(ps->f, N, fnorms, rng);
          if (!shell0) {
            ps->in = FunctionsCompressedNode<T, NDIM>(ps->key, SparsityInfo(N, SparsityInfo::InitType::AllNonZero), K);
            fill_random(ps->in, N, fnorms, rng);
          }
          state = std::move(ps);

          /* first iteration is warm-up: moves inputs and operator data to the device */
          std::vector<double> times;
          for (int r = 0; r < cfg.nreps + 1; ++r) {
            auto beg = std::chrono::high_resolution_clock::now();
            if (is_root) start->invoke();
            ttg::execute();
            ttg::fence();
            auto end = std::chrono::high_resolution_clock::now();
            if (r > 0) {
              times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count() * 1e-3);
            }
          }

          const double t_min = *std::min_element(times.begin(), times.end());
          double t_mean = 0.0;
          for (auto t : times) t_mean += t;
          t_mean /= times.size();
          const double gflop = flops_per_task * cfg.ntasks * 1e-9;

          if (is_root) {
            std::cout << opname << "," << K << "," << N << "," << n << "," << disp_str(disp) << ","
                      << rank << "," << std::scientific << std::setprecision(3) << opnorm << std::defaultfloat << ","
                      << active << "," << std::fixed << std::setprecision(2) << avg_r << "," << avg_s << ","
                      << cfg.ntasks << "," << mra::get_batch_size() << ","
                      << std::setprecision(1) << t_min << "," << t_mean << ","
                      << std::setprecision(3) << t_min / cfg.ntasks << ","
                      << gflop << "," << std::setprecision(2) << gflop / (t_min * 1e-6)
                      << std::defaultfloat << std::endl;
          }
        }
      }
    }
    state.reset();
  }

} // namespace

int main(int argc, char **argv) {

  auto opt = mra::OptionParser(argc, argv);
  if (opt.exists("-h") || opt.exists("--help")) {
    std::cout << "Usage: " << argv[0] << " [options]\n"
              << "  -op <list>     operators: coulomb,bsh,slater (default: coulomb,bsh)\n"
              << "  -K <k>         polynomial order (default: 8; K=8 uses the WMMA path on sm_80+)\n"
              << "  -N <n>         functions per node (default: 16)\n"
              << "  -p <p>         threshold 10^-p (default: 6)\n"
              << "  -lo <lo>       smallest length scale resolved by the operator (default: 1e-6)\n"
              << "  -mu <mu>       BSH exponent (default: 1.0)\n"
              << "  -gamma <g>     Slater exponent (default: 1.0)\n"
              << "  -L <L>         simulation cell is [-L,L]^3 (default: 20)\n"
              << "  -lmin/-lmax    level range (default: 1..8)\n"
              << "  -R <r>         max displacement component (default: 2)\n"
              << "  -cnorm <c>     norm of the largest input function (default: 1.0)\n"
              << "  -cspread <s>   function norms span s decades below cnorm (default: 0)\n"
              << "  -n <tasks>     kernel launches (one node each) per repetition (default: 256)\n"
              << "  -r <reps>      timed repetitions per point, after one warm-up (default: 3)\n"
              << "  -B <b>         max batch size, <=1 disables batching (default: mra default)\n"
              << "  -resnorms      also compute result norms (as for the last contribution)\n"
              << "  -all           also run points where every function is screened out\n"
              << "  -s <seed>      random seed (default: 5551212)\n"
              << "  -c <cores>     number of cores (default: all)\n";
    return 0;
  }

  Config cfg;
  auto ops_arg = opt.get("-op");
  cfg.ops = split(ops_arg.empty() ? std::string_view("coulomb,bsh") : ops_arg);
  cfg.K = opt.parse("-K", 8);
  cfg.N = opt.parse("-N", 16);
  cfg.thresh = std::pow(10.0, -opt.parse("-p", 6));
  cfg.lo = opt.parse("-lo", 1e-6);
  cfg.mu = opt.parse("-mu", 1.0);
  cfg.gamma = opt.parse("-gamma", 1.0);
  cfg.L = opt.parse("-L", 20.0);
  cfg.lmin = opt.parse("-lmin", 1);
  cfg.lmax = opt.parse("-lmax", 8);
  cfg.R = opt.parse("-R", 2);
  cfg.cnorm = opt.parse("-cnorm", 1.0);
  cfg.cspread = opt.parse("-cspread", 0.0);
  cfg.ntasks = opt.parse("-n", 256);
  cfg.nreps = opt.parse("-r", 3);
  cfg.with_resnorms = opt.exists("-resnorms");
  cfg.all_points = opt.exists("-all");
  cfg.seed = opt.parse("-s", 5551212L);
  int cores = opt.parse("-c", -1);
  if (opt.exists("-B")) {
    mra::set_batch_size(opt.parse("-B", 1));
  }

  mra::initialize(argc, argv, cores);

  /* MADNESS builds the operators relative to these defaults */
  madness::FunctionDefaults<3>::set_cubic_cell(-cfg.L, cfg.L);
  madness::FunctionDefaults<3>::set_k(cfg.K);
  madness::FunctionDefaults<3>::set_thresh(cfg.thresh);

  run_convbench(cfg);

  mra::finalize();
}
