#ifndef MRA_JIT_COMPILER_H
#define MRA_JIT_COMPILER_H

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(MRA_ENABLE_CUDA)
#include <cuda.h>
#elif defined(MRA_ENABLE_HIP)
#include <hip/hip_runtime.h>
#else
#error "mra/jit/compiler.h requires MRA_ENABLE_CUDA or MRA_ENABLE_HIP (JIT support targets CUDA first; see the plan's Step 5)"
#endif

namespace mra::jit {

#if defined(MRA_ENABLE_CUDA)
  using DeviceModule   = CUmodule;
  using DeviceFunction = CUfunction;
  using DeviceStream   = CUstream;
#elif defined(MRA_ENABLE_HIP)
  using DeviceModule   = hipModule_t;
  using DeviceFunction = hipFunction_t;
  using DeviceStream   = hipStream_t;
#endif

  /**
   * Thrown on any NVRTC/hiprtc or driver API failure. what() includes the
   * NVRTC compile log when the failure was a compile error.
   */
  class CompileError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
  };

  /**
   * Target architecture plus any kernel-specific extra compile options
   * (e.g. a future `-DMRA_JIT_K=<value>` to bake a compile-time K into a
   * cuBLASDX/rocWMMA specialization). The cache key (see cache.h) must
   * include compute_major/compute_minor: a PTX/cubin compiled for one
   * architecture reused on another is a correctness bug, not just a load
   * failure.
   */
  struct CompileOptions {
    int compute_major = 8;
    int compute_minor = 0;
    std::vector<std::string> extra_options;
  };

  /**
   * A ready-to-launch JIT-compiled kernel. Entries live for the lifetime of
   * the process (see cache.h) -- there is no unload/eviction path yet.
   */
  struct CompiledKernel {
    DeviceModule module = nullptr;
    DeviceFunction function = nullptr;
  };

  /**
   * Compiles one kernel header (already embedded via
   * mra_generate_jit_sources(), see mra::jit::embedded_headers() in
   * include/mra/jit/embedded_headers.h) through NVRTC and loads the result
   * via the CUDA driver API. This implements, for real, the approach
   * empirically validated in spike/nvrtc/ (see the *_spike.cc files): NVRTC
   * has no filesystem
   * access of its own (headers must be supplied as the `headers` map),
   * needs `-DMRA_JIT_COMPILE=1` and `-default-device` (several plain
   * constexpr helpers in the kernel headers lack __host__ __device__
   * annotations -- harmless in the AOT build, rejected outright by NVRTC's
   * stricter JIT mode otherwise), and needs libcu++'s `<cuda/std/...>`
   * headers reachable via `-I` (mra/jit/std_compat.h redirects the small
   * set of std:: facilities the kernel headers use to their cuda::std::
   * equivalents -- real system headers are not viable under NVRTC at all).
   */
  class Compiler {
  public:
    Compiler() = default;

    /**
     * `program_name`: a virtual filename for NVRTC diagnostics only (e.g.
     *   "gaxpy").
     * `source`: the top-level program text; typically just
     *   `#include "mra/kernels/<kernel>.h"\n`.
     * `headers`: the embedded header map (mra::jit::embedded_headers()).
     * `name_expression`: the fully-qualified template-id string for the
     *   exact instantiation wanted, e.g.
     *   "mra::detail::gaxpy_kernel<double,3,mra::SparseTensorView<double,4>,
     *    mra::SparseTensorView<double,4>,mra::SparseTensorView<double,4>>"
     *   -- every invented (abbreviated-auto) template parameter must be
     *   supplied explicitly, in the order the `auto` parameters appear in
     *   the kernel's declaration (see mra/jit/type_name.h for building
     *   this string from a caller's own template parameters).
     *
     * Throws CompileError on any NVRTC or driver API failure.
     */
    CompiledKernel compile(
      std::string_view program_name,
      std::string_view source,
      const std::unordered_map<std::string, std::string>& headers,
      std::string_view name_expression,
      const CompileOptions& opts) const;
  };

} // namespace mra::jit

#endif // MRA_JIT_COMPILER_H
