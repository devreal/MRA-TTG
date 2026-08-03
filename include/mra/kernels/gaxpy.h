#ifndef MRA_KERNELS_GAXPY_H
#define MRA_KERNELS_GAXPY_H

#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

#if defined(MRA_ENABLE_JIT) && defined(MRA_ENABLE_CUDA) && !defined(MRA_JIT_COMPILE)
#include "mra/jit/cache.h"
#include "mra/jit/embedded_headers.h"
#include "mra/jit/launch.h"
#include "mra/jit/type_name.h"
#endif

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void axpy_kernel_impl(
      const concepts::TensorView<NDIM> auto& nodeA,
      concepts::TensorView<NDIM> auto& nodeR,
      const T scalarA)
    {
      foreach_idx(nodeR, [&](size_type i) {
        nodeR[i] += scalarA*nodeA[i];
      });
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void gaxpy_kernel_impl(
      const concepts::TensorView<NDIM> auto& nodeA,
      const concepts::TensorView<NDIM> auto& nodeB,
      concepts::TensorView<NDIM> auto& nodeR,
      const T scalarA,
      const T scalarB)
    {
      foreach_idx(nodeR, [&](size_type i) {
        nodeR[i] = scalarA*nodeA[i] + scalarB*nodeB[i];
      });
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gaxpy_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto nodeA_view,
      const concepts::TensorView<NDIM+1> auto nodeB_view,
      concepts::TensorView<NDIM+1> auto nodeR_view,
      const T scalarA,
      const T scalarB,
      size_type N)
    {
      SHARED DenseTensorView<const T, NDIM> nodeA, nodeB;
      SHARED DenseTensorView<T, NDIM> nodeR;
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (nodeR_view.is_zero(blockid)) {
          /* no work to do */
          continue;
        }
        if (is_team_lead()) {
          nodeA = nodeA_view(blockid);
          nodeB = nodeB_view(blockid);
          nodeR = nodeR_view(blockid);
        }
        SYNCTHREADS();
        gaxpy_kernel_impl<T, NDIM>(nodeA, nodeB, nodeR, scalarA, scalarB);
      }
    }
  } // namespace detail


#if !defined(MRA_JIT_COMPILE)
  // Host-only AOT launcher -- irrelevant to a JIT compile pass (which only
  // ever needs the detail::gaxpy_kernel body above; a JIT build gets its own
  // submit_gaxpy_kernel_jit launcher instead, see the plan's Step 5). Guarded
  // out here purely because ttg::device::Stream isn't available under
  // MRA_JIT_COMPILE (types.h skips <ttg.h> for the JIT pass).
  template <typename T, Dimension NDIM>
  void submit_gaxpy_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& funcA,
    const concepts::TensorView<NDIM+1> auto& funcB,
    concepts::TensorView<NDIM+1> auto& funcR,
    const T scalarA,
    const T scalarB,
    size_type N,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::gaxpy_kernel, N, thread_dims, 0, stream,
      (key, funcA, funcB, funcR, scalarA, scalarB, N));
    checkSubmit();
  }
#endif // !MRA_JIT_COMPILE

#if defined(MRA_ENABLE_JIT) && defined(MRA_ENABLE_CUDA) && !defined(MRA_JIT_COMPILE)
  // JIT-compiled alternative to submit_gaxpy_kernel: same math, but compiled
  // on demand via NVRTC (see mra::jit::Compiler) instead of ahead-of-time.
  // First proof-of-concept consumer of the mra::jit infrastructure; the
  // real payoff (baking a runtime K into a compile-time constant to unlock
  // cuBLASDX/rocWMMA) lands once compress.h/transform.h grow their own
  // submit_*_kernel_jit, since gaxpy has no such GEMM dependency.
  template <typename T, Dimension NDIM>
  void submit_gaxpy_kernel_jit(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& funcA,
    const concepts::TensorView<NDIM+1> auto& funcB,
    concepts::TensorView<NDIM+1> auto& funcR,
    const T scalarA,
    const T scalarB,
    size_type N,
    size_type K,
    ttg::device::Stream stream)
  {
    using ViewA = std::decay_t<decltype(funcA)>;
    using ViewB = std::decay_t<decltype(funcB)>;
    using ViewR = std::decay_t<decltype(funcR)>;

    // Explicit, in-declaration-order template arguments for gaxpy_kernel's
    // 2 declared (T, NDIM) + 3 invented (nodeA_view/nodeB_view/nodeR_view)
    // template parameters -- see mra/jit/type_name.h and
    // spike/nvrtc/gaxpy_spike.cc for why this has to be fully explicit.
    const std::string name_expr =
      "mra::detail::gaxpy_kernel<" + jit::type_name_v<T>() + "," + std::to_string(NDIM) + "," +
      jit::type_name_v<ViewA>() + "," + jit::type_name_v<ViewB>() + "," + jit::type_name_v<ViewR>() + ">";

    jit::CompileOptions opts;
    // TODO: derive compute_major/compute_minor from the actual current
    // device (e.g. cuDeviceComputeCapability) instead of the struct
    // default; deferred until this is exercised on real hardware.
    const std::string cache_key = jit::make_cache_key("gaxpy_kernel", name_expr, opts);

    const jit::CompiledKernel& kernel = jit::Cache::instance().get_or_compile(
      cache_key, jit::Compiler{}, "gaxpy",
      "#include \"mra/kernels/gaxpy.h\"\n",
      jit::embedded_headers(), name_expr, opts);

    Dim3 thread_dims = max_thread_dims(2*K);
    jit::launch(kernel, Dim3(N, 1, 1), thread_dims, 0, stream,
                key, funcA, funcB, funcR, scalarA, scalarB, N);
  }
#endif // defined(MRA_ENABLE_JIT) && defined(MRA_ENABLE_CUDA) && !defined(MRA_JIT_COMPILE)

} // namespace mra

#endif // MRA_KERNELS_GAXPY_H
