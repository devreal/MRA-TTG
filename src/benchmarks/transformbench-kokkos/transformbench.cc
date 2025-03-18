
#include "mra/misc/platform.h"
#include "mra/kernels/transform.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/options.h"
#include "mra/misc/types.h"

#include <ttg.h>

using namespace mra; // lazy

template<typename T>
static
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
GLOBALSCOPE void transform_kernel(int N, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {

  SHARED TensorView<T, 3> a, c, w;
  SHARED TensorView<T, 2> b;
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    if (is_team_lead()) {
      a = A(i);
      b = B(i);
      c = C(i);
      w = workspace(i);
    }
    SYNCTHREADS();
    transform(a, b, c, w.data());
  }
}

template<typename T>
static void submit_transform_bench(int N, int K, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {
  Dim3 thread_dims = max_thread_dims(K);
  CALL_KERNEL(transform_kernel, std::min(N, M), thread_dims, 0, ttg::device::current_stream(), (A, B, C, workspace));
  checkSubmit();
}

int main(int argc, char **argv) {

  auto opt = mra::OptionParser(argc, argv);
  int ntasks = opt.parse("-n", 100);
  int N = opt.parse("-N", 10); // number of functions
  int K = opt.parse("-K", 10); // number of coefficients
  int M = opt.parse("-M", 128); // max number of blocks
  ttg::initialize(argc, argv);
  ttg::Edge<int, void> e; // control edge

  auto start = ttg::make_tt([&](){
    for (int i = 0; i < ntasks; i++) {
      ttg::sendk<0>(i);
    }
  }, ttg::edges(), ttg::edges(e));

  auto tt = ttg::make_tt<Space>([&](const int& key) -> TASKTYPE {
    auto a = Tensor<double, 3+1>({N, K, K, K}, ttg::scope::Allocate); // nblocks x size^3 elements
    auto b = Tensor<double, 2+1>({N, K, K}, ttg::scope::Allocate); // size^2 elements
    auto c = Tensor<double, 3+1>({N, K, K, K}, ttg::scope::Allocate); // size^3 elements
    auto workspace = Tensor<double, 3+1>({N, K, K, K}, ttg::scope::Allocate); // size^3 elements
#ifndef MRA_ENABLE_HOST
    co_await ttg::device::select(a.buffer(), b.buffer(), c.buffer(), workspace.buffer());
#endif // MRA_ENABLE_HOST
    submit_transform_bench(N, M, K, a.current_view(), b.current_view(), c.current_view(), workspace.current_view());
  }, ttg::edges(e), ttg::edges());

  auto connected = ttg::make_graph_executable(start.get());
  assert(connected);


  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  beg = std::chrono::high_resolution_clock::now();
  start->invoke(); // kick off
  ttg::execute();
  ttg::fence();
  end = std::chrono::high_resolution_clock::now();

  auto ms = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000;
  uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * N;
  std::cout << "Transform N = " << N << ";M = " << M << ";K = " << K << ";tasks = " << ntasks
            << ";Time (milliseconds) = "
            << ms
            << ";GFlop = " << flops*1e-9
            << ";Gflop/s = " << (1e-6 * flops) / ms
            << std::endl;
  ttg::finalize();
}