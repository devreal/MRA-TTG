
#include "mra/misc/platform.h"
#include "mra/kernels/transform.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/options.h"
#include "mra/misc/types.h"

#include <ttg.h>

using namespace mra; // lazy

#if __has_include(<cutlass/cutlass.h>)
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#define HAVE_CUTLASS 1
//using namespace cute;
#endif // __has_include(<cutlass/cutlass.h>)

/**
 * CUTLASS
 */

#ifdef HAVE_CUTLASS

constexpr int MAX_MN = 64;

template<int M, int N, int K, typename aT, typename bT, typename cT>
DEVSCOPE void mTxmq_cutlass_core(cT* c, aT* a, bT* b, int strideA = 1, int strideB = 1, int strideC = 1) {

  static_assert(M <= MAX_MN, "M must be less than or equal to MAX_MN");
  static_assert(N <= MAX_MN, "N must be less than or equal to MAX_MN");

  using ValueA = typename std::decay_t<aT>;
  using ValueB = typename std::decay_t<bT>;
  using ValueC = typename std::decay_t<cT>;

  using GemmShape = cutlass::gemm::GemmShape<M, N, K>;
  using WarpShape = cutlass::gemm::GemmShape<8, 4, 2>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>; // for Simt

  using Gemm = typename cutlass::gemm::threadblock::DefaultMma<
    ValueA, cutlass::layout::ColumnMajor, 1,
    ValueB, cutlass::layout::RowMajor, 1,
    ValueC /* accumulator */, cutlass::layout::RowMajor,
    cutlass::arch::OpClassSimt, // can be OpClassSimt, OpClassTensorOp, or OpClassWmmaTensorOp
    cutlass::arch::Sm80,        // can be Sm70, Sm80, or Sm90
    GemmShape,                  // max 64xK / Kx64
    WarpShape,                  // max 8x8xK
    InstructionShape,           // 8, 8, 4 is the only option for Sm80
    2 /* stages */,
    cutlass::arch::OpMultiplyAdd>;

  using MmaCore = typename Gemm::MmaCore;

  cutlass::TensorRef<ValueA, cutlass::layout::ColumnMajor> ref_A((ValueA*)a, strideA);
  cutlass::TensorRef<ValueB, cutlass::layout::RowMajor> ref_B((ValueB*)b, strideB);
  cutlass::TensorRef<ValueC, cutlass::layout::RowMajor> ref_C((ValueC*)c, strideC);

  cutlass::MatrixCoord tb_offset_A{0, 0}; // only one block so threadblock offset is 0
  cutlass::MatrixCoord tb_offset_B{0, 0};

  // Compute position within threadblock
  int tb_thread_id = mra::thread_id();

  cutlass::gemm::GemmCoord problem_size = {M, N, K};


  typename Gemm::IteratorA::Params params_A(ref_A.layout());
  typename Gemm::IteratorB::Params params_B(ref_B.layout());

  // Construct iterators to A and B operands
  typename Gemm::IteratorA iterator_A(params_A, ref_A.data(),
                                     {problem_size.m(), problem_size.k()},
                                     tb_thread_id, tb_offset_A);

  typename Gemm::IteratorB iterator_B(params_B, ref_B.data(),
                                     {problem_size.k(), problem_size.n()},
                                     tb_thread_id, tb_offset_B);

  // Define MmaPipeline Single Stage
  using MmaPipelineSingleStage =  cutlass::gemm::threadblock::MmaSingleStage<
      typename MmaCore::Shape, typename Gemm::IteratorA, typename MmaCore::SmemIteratorA,
      typename Gemm::IteratorB, typename MmaCore::SmemIteratorB, cT, cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy>;

  // Define MmaPipeline Two Stages
  using MmaPipelineTwoStages =  cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, typename Gemm::IteratorA, typename MmaCore::SmemIteratorA,
      typename Gemm::IteratorB, typename MmaCore::SmemIteratorB, cT, cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy>;

  // Define the threadblock-scoped pipelined matrix multiply (Select between Single vs. Two stages)
  using Mma = MmaPipelineTwoStages;

  SHARED typename Mma::SharedStorage shared_storage;

  //if (thread_id() == 0) printf("mTxmq_cutlass_core M = %d, N = %d, K = %d, shared mem %zu\n", M, N, K, sizeof(shared_storage));

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;

  // Construct thread-scoped matrix multiply
  Mma mma(shared_storage, tb_thread_id, warp_id, threadIdx.x);

  typename Mma::FragmentC accum;
  accum.clear();

  int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

  // Output results
  typename Mma::Operator::IteratorC iterator_C({(ValueC*)c, strideC}, lane_id);

  iterator_C.add_tile_offset(
      {(warp_id % Mma::WarpCount::kM), (warp_id / Mma::WarpCount::kM)});

  iterator_C.store(accum);
}


template<int M, int N, int K, typename aT, typename bT, typename cT>
DEVSCOPE void mTxmq_cutlass_block(cT* c, aT* a, bT* b) {
  if (M == K*K) {
    int m = 0;
    for (; m < M; m += MAX_MN) {
      if (m+MAX_MN <= M) {
        mTxmq_cutlass_core<MAX_MN, N, K>(c + m*N, a + m, b, M*M, N, N);
      }
    }
    if constexpr (0 < M%MAX_MN) {
      m -= MAX_MN;
      mTxmq_cutlass_core<M%MAX_MN, N, K>(c + m*N, a + m, b, M*M, N, N);
    }
  } else {
    // TODO: implement!
    static_assert(M == K*K, "M must be equal to K*K");
  }
}

template <typename aT, typename bT, typename cT>
DEVSCOPE void mTxmq_cutlass(long dimi, long dimj, long dimk,
                             cT* MADNESS_RESTRICT c, const aT* a, const bT* b) {
  int M = dimi;
  int N = dimj;
  int K = dimk;
  if (M == K*K) {
    // A is tall and skinny, B is square
    if (K == 6) {
      //mTxmq_cutlass_block<36, 6, 6>(c, a, b);
    } else if (K == 8) {
      //mTxmq_cutlass_block<64, 8, 8>(c, a, b);
    } else if (K == 10) {
      //mTxmq_cutlass_block<100, 10, 10>(c, a, b);
    } else if (K == 12) {
      //mTxmq_cutlass_block<12*12, 12, 12>(c, a, b);
    } else if (K == 16) {
      mTxmq_cutlass_block<16*16, 16, 16>(c, a, b);
    } else if (K == 20) {
      //mTxmq_cutlass_block<400, 20, 20>(c, a, b);
    } else {
      if (is_team_lead()) printf("mTxmq: Unsupport K = %d\n", K);
    }
  } else if (N == K*K) {
    // B is wide and narrow, A is square
    if (K == 6) {
      //mTxmq_cutlass_block<6, 36, 6>(c, a, b);
    } else if (K == 8) {
      //mTxmq_cutlass_block<8, 64, 8>(c, a, b);
    } else if (K == 10) {
      //mTxmq_cutlass_block<10, 100, 10>(c, a, b);
    } else if (K == 12) {
      //mTxmq_cutlass_block<12, 12*12, 12>(c, a, b);
    } else if (K == 16) {
      //mTxmq_cutlass_block<16, 16*16, 16>(c, a, b);
    } else if (K == 20) {
      //mTxmq_cutlass_block<20, 400, 20>(c, a, b);
    } else {
      if (is_team_lead()) printf("mTxmq: Unsupport K = %d\n", K);
    }
  } else {
      printf("mTxmq: Unknown configuration with M = %d, N = %d, K = %d\n", M, N, K);
  }
}

template<typename T>
static
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
GLOBALSCOPE void transform_cutlass_kernel(int N, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {

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

    const T* pc = b.data();
    T *t0=w.data(), *t1=c.data();
    if (a.ndim() & 0x1) std::swap(t0,t1);
    const size_type dimj = c.dim(1);
    size_type dimi = 1;
    for (size_type n=1; n<a.ndim(); ++n) dimi *= dimj;
    mTxmq_cutlass(dimi, dimj, dimj, t0, a.data(), pc);
    for (size_type n=1; n<a.ndim(); ++n) {
      mTxmq_cutlass(dimi, dimj, dimj, t1, t0, pc);
      std::swap(t0,t1);
    }
  }
}

template<typename T>
static void submit_transform_cutlass_bench(int N, int M, int K, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {
  Dim3 thread_dims = max_thread_dims(K);
  CALL_KERNEL(transform_cutlass_kernel, std::min(N, M), thread_dims, 0, ttg::device::current_stream(), (N, A, B, C, workspace));
  checkSubmit();
}

#else

template<typename T>
static void submit_transform_cutlass_bench(int N, int M, int K, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {
  std::cout << "No CUTLASS available" << std::endl;
}

#endif // HAVE_CUTLASS

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
static void submit_transform_bench(int N, int M, int K, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {
  Dim3 thread_dims = max_thread_dims(K);
  CALL_KERNEL(transform_kernel, std::min(N, M), thread_dims, mTxmq_shmem_size<T>(K), ttg::device::current_stream(), (N, A, B, C, workspace));
  checkSubmit();
}

enum BLASType {
  BLAS_CUTLASS,
  BLAS_CUBLASDX,
  BLAS_DEFAULT
};

void transform_bench(int nreps, int ntasks, mra::size_type N, mra::size_type M, mra::size_type K, BLASType blas_type) {

  ttg::Edge<int, void> e; // control edge

  auto start = ttg::make_tt([&](){
    for (int i = 0; i < ntasks; i++) {
      ttg::sendk<0>(i);
    }
  }, ttg::edges(), ttg::edges(e));

  auto tt = ttg::make_tt<Space>([&](const int& key) -> TASKTYPE {
    auto a = Tensor<double, 3+1>({N, K, K, K}, ttg::scope::Allocate); // N x K^3 elements
    auto b = Tensor<double, 2+1>({N, K, K}, ttg::scope::Allocate); // K^2 elements
    auto c = Tensor<double, 3+1>({N, K, K, K}, ttg::scope::Allocate); // K^3 elements
    auto workspace = Tensor<double, 3+1>({N, K, K, K}, ttg::scope::Allocate); // K^3 elements
#ifndef MRA_ENABLE_HOST
    co_await ttg::device::select(a.buffer(), b.buffer(), c.buffer(), workspace.buffer());
#endif // MRA_ENABLE_HOST
    if (blas_type == BLAS_CUTLASS) {
      submit_transform_cutlass_bench(N, M, K, a.current_view(), b.current_view(), c.current_view(), workspace.current_view());
    } else {
      submit_transform_bench(N, M, K, a.current_view(), b.current_view(), c.current_view(), workspace.current_view());
    }
  }, ttg::edges(e), ttg::edges());

  auto connected = ttg::make_graph_executable(start.get());
  assert(connected);


  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  for (int i = 0; i < nreps+1; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    start->invoke(); // kick off
    ttg::execute();
    ttg::fence();
    end = std::chrono::high_resolution_clock::now();

    /* skip warm-up */
    if (i > 0) {
      auto us = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
      uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * 2 /* multiply-add */ * N;
      std::cout << "Transform N = " << N << ";M = " << M << ";K = " << K << ";tasks = " << ntasks
                << ";Time (microseconds) = "
                << us
                << ";GFlop = " << flops*1e-9
                << ";Gflop/s = " << (1e-3 * flops) / us
                << std::endl;
    }
  }
}

int main(int argc, char **argv) {

  auto opt = mra::OptionParser(argc, argv);
  int nreps = opt.parse("-r", 1);
  int ntasks = opt.parse("-n", 100);
  int N = opt.parse("-N", 10); // number of functions
  int K = opt.parse("-K", 10); // number of coefficients
  int M = opt.parse("-M", 128); // max number of blocks
  BLASType blas_type = BLAS_DEFAULT;
  if (opt.exists("--cutlass")) {
    blas_type = BLAS_CUTLASS;
  }
  std::cout << "Running benchmark with " << nreps << " repetitions, " << ntasks << " tasks, "
            << N << " functions, " << K << " coefficients, " << M << " blocks"
            << std::endl;
  ttg::initialize(argc, argv);
  allocator_init(argc, argv);

  transform_bench(nreps, ntasks, N, M, K, blas_type);

  allocator_fini();
  ttg::finalize();
}
