
#include "mra/misc/platform.h"
#include "mra/kernels/transform.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/misc/options.h"
#include "mra/misc/types.h"

#include <ttg.h>

#if __has_include(<cutlass/cutlass.h>)
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
using namespace cutlass;
//using namespace cute;
#define HAVE_CUTLASS 1
#endif // __has_include(<cutlass/cutlass.h>)

using namespace mra; // lazy

#define HAVE_CUTLASS 1

#ifdef HAVE_CUTLASS

template <int M, int N, int K, typename aT, typename bT, typename cT>
struct mTxmq_cutlass {

  using ElementA = aT;
  using ElementB = bT;
  using ElementC = cT;
  using ElementAccumulator = cT;

  // The code section below describes matrix layout of input and output matrices. Row Major for
  // Matrix A, Column Major for Matrix B and Row Major for Matrix C
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);

  // CollectiveBuilder is only supported from SM90 onwards

#if CUDA_VERSION >= 9000
  using op = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      Shape<M,N,K>, Shape<_2,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
#else // CUDA_VERSION >= 9000
  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<M, N, K>;  // <- threadblock tile M = M, N = M, K = K

  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<K, K, K>;  // <- warp tile M = K, N = K, K = K


  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16


  using op = cutlass::gemm::device::Gemm<
    aT, cutlass::layout::ColumnMajor,
    bT, cutlass::layout::RowMajor,
    cT, cutlass::layout::RowMajor,
    cT /* accumulator */>;

  static void run(const aT* a, const bT* b, cT* c, int M, int N, int K) {
    typename op::Arguments args({M, N, K}, a, K, b, N, c, N, c, N);
    op::launch(args);
  }
#endif // CUDA_VERSION >= 9000
#if 0
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute>
#endif // 0
};

#if 0
template <typename aT, typename bT, typename cT>
__device__ void mTxmq_cutlass(
  /* problem size (static or dynamic) */
  auto bM, auto bN, auto bK,
  /* data */
  cT* c, const aT* a, const bT* b) {

  auto shape_MNK = make_shape(bM, bN, bK);

  // Define TN strides (mixed)
  auto dA = make_stride(bM, cute::Int<1>{});               // (dM, dK)
  auto dB = make_stride(bN, cute::Int<1>{});               // (dN, dK)
  auto dC = make_stride(cute::Int<1>{}, bN);               // (dM, dN)

  // global layout
  auto gA = make_layout(make_shape(bM, bK), cute::LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto gB = make_layout(make_shape(bN, bK), cute::LayoutRight{});   // (n,k) -> smem_idx; k-major
  auto gC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx; m-major

  // Define the smem layouts (static)
  auto slA = make_layout(make_shape(bM, bK), cute::LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto slB = make_layout(make_shape(bN, bK), cute::LayoutRight{});   // (n,k) -> smem_idx; k-major
  auto slC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx; m-major

  using ASmemLayout = decltype(slA);
  using BSmemLayout = decltype(slB);
  using CSmemLayout = decltype(slC);

  static_assert(cute::is_static<ASmemLayout>::value);
  static_assert(cute::is_static<BSmemLayout>::value);
  static_assert(cute::is_static<CSmemLayout>::value);

  // Shared memory buffers
  __shared__ aT smemA[cute::cosize_v<ASmemLayout>];
  __shared__ bT smemB[cute::cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), slA);  // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), slB);  // (BLK_N,BLK_K)

  // Define the thread layouts (static)
  constexpr const auto thread_dims = max_thread_dims(bK);
  auto tA = make_layout(make_shape(cute::Int<thread_dims.y>{}, cute::Int<thread_dims.x>{}));   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(cute::Int<thread_dims.y>{}, cute::Int<thread_dims.x>{}));   // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(cute::Int<thread_dims.y>{}, cute::Int<thread_dims.x>{}));   // (m,n) -> thr_idx

  // simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles
  cute::Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  cute::Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  cute::Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  cute::Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

  // Partition gC (M,N) by the tile of tC
  cute::Tensor tCgC = local_partition(gC, tC, threadIdx.x, cute::Step<cute::Int<1>,cute::Int<1>>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  cute::Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)

  // Represent the full tensors
  cute::Tensor mA = make_tensor(cute::make_gmem_ptr(a), select<0,2>(shape_MNK), dA); // (M,K)
  cute::Tensor mB = make_tensor(cute::make_gmem_ptr(b), select<1,2>(shape_MNK), dB); // (N,K)
  cute::Tensor mC = make_tensor(cute::make_gmem_ptr(c), select<0,1>(shape_MNK), dC); // (M,N)


  auto M_TILE_MAX = size<0>(tAgA);

  for (int m_tile = 0; m_tile < M_TILE_MAX; ++m_tile)
  {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(m_tile,cute::_,cute::_), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(cute::_,cute::_,cute::_), tBsB);           // B   (THR_N,THR_K) -> (THR_N,THR_K)

    // TUTORIAL: The above call to copy(tAgA(_,_,k_tile), tAsA) is equivalent to
    //   Tensor tAgAk = tAgA(_,_,k_tile);
    //   CUTE_UNROLL
    //   for (int i = 0; i < size(tAsA); ++i) {
    //     tAsA(i) = tAgAk(i);
    //   }

    arch::cp_async_fence();        // Label the end of (potential) cp.async instructions
    arch::cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
    //       }
    //     }
    //   }

    __syncthreads();         // Wait for all threads to read from smem
  }
}


template<typename T>
static
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
GLOBALSCOPE void transform_cutlass_kernel(int N, auto K, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {

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

    if (K == 20) {
      mTxmq_cutlass(cute::Int<20*20>{}, cute::Int<20>{}, cute::Int<20>{}, a.data(), b.data(), c.data());
    }

    const T* pc = c.data();
    T *t0=workspace, *t1=result.data();
    if (t.ndim() & 0x1) std::swap(t0,t1);
    mTxmq_cutlass(K, t0, t.data(), pc);
    for (size_type n=1; n<t.ndim(); ++n) {
      mTxmq_cutlass<K>(K, t1, t0, pc);
      std::swap(t0,t1);
    }
  }
}
#endif // 0

template<typename T>
static void submit_transform_cutlass_bench(int N, int M, int K, TensorView<T, 3+1> A, TensorView<T, 2+1> B, TensorView<T, 3+1> C, TensorView<T, 3+1> workspace) {
  Dim3 thread_dims = max_thread_dims(K);
  if (K == 10) {
    CALL_KERNEL(transform_cutlass_kernel, std::min(N, M), thread_dims, 0, ttg::device::current_stream(), (N, A, B, C, workspace));
  }
  checkSubmit();
}

void transform_cutlass_bench(int nreps, int ntasks, int N, int M, int K) {

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
    submit_transform_cutlass_bench(N, M, K, a.current_view(), b.current_view(), c.current_view(), workspace.current_view());
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
      auto ms = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000;
      uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * 2 /* multiply-add */ * N;
      std::cout << "Transform CUTLASS N = " << N << ";M = " << M << ";K = " << K << ";tasks = " << ntasks
                << ";Time (milliseconds) = "
                << ms
                << ";GFlop = " << flops*1e-9
                << ";Gflop/s = " << (1e-6 * flops) / ms
                << std::endl;
    }
  }
}
#else

void transform_cutlass_bench(int nreps, int ntasks, int N, int M, int K) {
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
  CALL_KERNEL(transform_kernel, std::min(N, M), thread_dims, 0, ttg::device::current_stream(), (N, A, B, C, workspace));
  checkSubmit();
}

void transform_bench(int nreps, int ntasks, int N, int M, int K) {

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
  for (int i = 0; i < nreps+1; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    start->invoke(); // kick off
    ttg::execute();
    ttg::fence();
    end = std::chrono::high_resolution_clock::now();

    /* skip warm-up */
    if (i > 0) {
      auto ms = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000;
      uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * 2 /* multiply-add */ * N;
      std::cout << "Transform N = " << N << ";M = " << M << ";K = " << K << ";tasks = " << ntasks
                << ";Time (milliseconds) = "
                << ms
                << ";GFlop = " << flops*1e-9
                << ";Gflop/s = " << (1e-6 * flops) / ms
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
  ttg::initialize(argc, argv);

  transform_bench(nreps, ntasks, N, M, K);

  ttg::finalize();
}