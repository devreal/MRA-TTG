#include "mra/jit/compiler.h"
#include "mra/jit/launch.h"

#include <mutex>

#if defined(MRA_ENABLE_CUDA)
#include <nvrtc.h>
#elif defined(MRA_ENABLE_HIP)
#include <hip/hiprtc.h>
#endif

// CMake bakes CUDAToolkit_INCLUDE_DIRS in here (see src/CMakeLists.txt) --
// NVRTC needs libcu++'s <cuda/std/...> reachable via -I, and there's no
// portable way to discover the toolkit's own include dir at runtime
// otherwise.
#ifndef MRA_CUDA_TOOLKIT_INCLUDE_DIR
#define MRA_CUDA_TOOLKIT_INCLUDE_DIR ""
#endif

namespace mra::jit {

namespace {

#if defined(MRA_ENABLE_CUDA)

  void check_nvrtc(nvrtcResult res, const char* what) {
    if (res != NVRTC_SUCCESS) {
      throw CompileError(std::string(what) + " failed: " + nvrtcGetErrorString(res));
    }
  }

  void check_cu(CUresult res, const char* what) {
    if (res != CUDA_SUCCESS) {
      const char* name = nullptr;
      cuGetErrorName(res, &name);
      throw CompileError(std::string(what) + " failed: " + (name ? name : "unknown CUDA driver error"));
    }
  }

  // cuInit() must be called exactly once per process before any other
  // driver API call; concurrent JIT compiles from multiple PaRSEC/TTG
  // worker threads are expected, hence std::call_once rather than a plain
  // bool flag.
  void ensure_cuda_initialized() {
    static std::once_flag once;
    static CUresult init_result = CUDA_ERROR_NOT_INITIALIZED;
    std::call_once(once, [] { init_result = cuInit(0); });
    check_cu(init_result, "cuInit");
  }

#endif // MRA_ENABLE_CUDA

} // namespace

CompiledKernel Compiler::compile(
    std::string_view program_name,
    std::string_view source,
    const std::unordered_map<std::string, std::string>& headers,
    std::string_view name_expression,
    const CompileOptions& opts) const
{
#if defined(MRA_ENABLE_CUDA)
  // nvrtcCreateProgram needs contiguous const char* arrays; keep the
  // backing std::strings alive (header_names/header_contents) for the
  // whole call.
  std::vector<std::string> header_names;
  std::vector<std::string> header_contents;
  header_names.reserve(headers.size());
  header_contents.reserve(headers.size());
  for (const auto& [name, content] : headers) {
    header_names.push_back(name);
    header_contents.push_back(content);
  }
  std::vector<const char*> header_name_ptrs;
  std::vector<const char*> header_content_ptrs;
  header_name_ptrs.reserve(headers.size());
  header_content_ptrs.reserve(headers.size());
  for (std::size_t i = 0; i < header_names.size(); ++i) {
    header_name_ptrs.push_back(header_names[i].c_str());
    header_content_ptrs.push_back(header_contents[i].c_str());
  }

  const std::string program_name_str(program_name);
  nvrtcProgram prog;
  check_nvrtc(nvrtcCreateProgram(&prog, source.data(), program_name_str.c_str(),
                                 static_cast<int>(header_names.size()),
                                 header_content_ptrs.data(), header_name_ptrs.data()),
              "nvrtcCreateProgram");

  const std::string name_expr(name_expression);
  check_nvrtc(nvrtcAddNameExpression(prog, name_expr.c_str()), "nvrtcAddNameExpression");

  const std::string arch_flag = "--gpu-architecture=compute_" +
      std::to_string(opts.compute_major) + std::to_string(opts.compute_minor);
  const std::string include_flag = std::string("-I") + MRA_CUDA_TOOLKIT_INCLUDE_DIR;

  std::vector<std::string> compile_opts_str = {
      "--std=c++20",
      "-DMRA_JIT_COMPILE=1",
      arch_flag,
      include_flag,
      // Several plain constexpr helpers in the kernel headers (e.g.
      // mTxm_shmem_size and friends in mxm.h) lack __host__ __device__
      // annotations -- harmless in the AOT build (host-only call sites),
      // rejected outright by NVRTC's JIT mode otherwise.
      "-default-device",
  };
  compile_opts_str.insert(compile_opts_str.end(), opts.extra_options.begin(), opts.extra_options.end());

  std::vector<const char*> compile_opts;
  compile_opts.reserve(compile_opts_str.size());
  for (const auto& o : compile_opts_str) compile_opts.push_back(o.c_str());

  const nvrtcResult compile_res =
      nvrtcCompileProgram(prog, static_cast<int>(compile_opts.size()), compile_opts.data());

  std::size_t log_size = 0;
  check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size), "nvrtcGetProgramLogSize");
  std::string log;
  if (log_size > 1) {
    log.resize(log_size);
    check_nvrtc(nvrtcGetProgramLog(prog, log.data()), "nvrtcGetProgramLog");
  }

  if (compile_res != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    throw CompileError("nvrtcCompileProgram failed for '" + program_name_str + "':\n" + log);
  }

  const char* lowered_name = nullptr;
  check_nvrtc(nvrtcGetLoweredName(prog, name_expr.c_str(), &lowered_name), "nvrtcGetLoweredName");
  // Copy before nvrtcDestroyProgram invalidates the pointer NVRTC owns.
  const std::string lowered_name_copy(lowered_name);

  std::size_t ptx_size = 0;
  check_nvrtc(nvrtcGetPTXSize(prog, &ptx_size), "nvrtcGetPTXSize");
  std::string ptx(ptx_size, '\0');
  check_nvrtc(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");

  nvrtcDestroyProgram(&prog);

  ensure_cuda_initialized();

  CompiledKernel result;
  check_cu(cuModuleLoadDataEx(&result.module, ptx.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");
  check_cu(cuModuleGetFunction(&result.function, result.module, lowered_name_copy.c_str()),
            "cuModuleGetFunction");
  return result;

#elif defined(MRA_ENABLE_HIP)
  (void)program_name; (void)source; (void)headers; (void)name_expression; (void)opts;
  throw CompileError("mra::jit::Compiler: HIP backend not implemented yet (CUDA-first per the plan)");
#else
  (void)program_name; (void)source; (void)headers; (void)name_expression; (void)opts;
  throw CompileError("mra::jit::Compiler: neither MRA_ENABLE_CUDA nor MRA_ENABLE_HIP defined");
#endif
}

void launch_impl(const CompiledKernel& kernel, Dim3 grid, Dim3 block,
                  unsigned int shared_mem_bytes, DeviceStream stream,
                  void** kernel_args) {
#if defined(MRA_ENABLE_CUDA)
  const CUresult res = cuLaunchKernel(kernel.function,
                                       grid.x, grid.y, grid.z,
                                       block.x, block.y, block.z,
                                       shared_mem_bytes, stream,
                                       kernel_args, nullptr);
  if (res != CUDA_SUCCESS) {
    const char* name = nullptr;
    cuGetErrorName(res, &name);
    throw CompileError(std::string("cuLaunchKernel failed: ") + (name ? name : "unknown CUDA driver error"));
  }
#elif defined(MRA_ENABLE_HIP)
  (void)kernel; (void)grid; (void)block; (void)shared_mem_bytes; (void)stream; (void)kernel_args;
  throw CompileError("mra::jit::launch: HIP backend not implemented yet (CUDA-first per the plan)");
#else
  (void)kernel; (void)grid; (void)block; (void)shared_mem_bytes; (void)stream; (void)kernel_args;
  throw CompileError("mra::jit::launch: neither MRA_ENABLE_CUDA nor MRA_ENABLE_HIP defined");
#endif
}

} // namespace mra::jit
