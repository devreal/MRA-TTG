#ifndef MRA_JIT_CACHE_H
#define MRA_JIT_CACHE_H

#include "mra/jit/compiler.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace mra::jit {

  /**
   * Builds the cache key for a compiled kernel. Must include the target
   * architecture -- a PTX/cubin compiled for one compute capability
   * silently reused on another would be a correctness bug (wrong code
   * loaded), not just a cuModuleLoadDataEx failure.
   */
  inline std::string make_cache_key(std::string_view kernel_name,
                                     std::string_view name_expression,
                                     const CompileOptions& opts) {
    return std::string(kernel_name) + "|" + std::string(name_expression) + "|sm_" +
           std::to_string(opts.compute_major) + std::to_string(opts.compute_minor);
  }

  /**
   * Process-wide, in-memory cache of compiled kernels. Compiling per launch
   * would be far too slow, so every (kernel, template args, architecture)
   * combination is compiled at most once per process and never evicted.
   * On-disk persistence (surviving across process launches) is explicitly
   * deferred -- see the plan's Step 5 risk notes: worth revisiting once the
   * real deployment shape (long-running vs. many short-lived MPI processes)
   * is clearer.
   *
   * Thread-safe: a single mutex serializes both cache lookups and the
   * (infrequent, one-time-per-key) compile itself, rather than trying to
   * fine-grained-lock per key -- NVRTC/the CUDA driver's compile path isn't
   * performance-critical the way kernel launches are, so this is a
   * deliberate simplicity-over-concurrency tradeoff.
   */
  class Cache {
  public:
    static Cache& instance() {
      static Cache cache;
      return cache;
    }

    /**
     * Returns the cached kernel for `key`, compiling it via `compiler` on
     * first use. The returned reference remains valid for the lifetime of
     * the process.
     */
    const CompiledKernel& get_or_compile(
        const std::string& key,
        const Compiler& compiler,
        std::string_view program_name,
        std::string_view source,
        const std::unordered_map<std::string, std::string>& headers,
        std::string_view name_expression,
        const CompileOptions& opts) {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_kernels.find(key);
      if (it != m_kernels.end()) {
        return it->second;
      }
      auto [inserted, _] = m_kernels.emplace(
          key, compiler.compile(program_name, source, headers, name_expression, opts));
      return inserted->second;
    }

  private:
    Cache() = default;

    std::mutex m_mutex;
    std::unordered_map<std::string, CompiledKernel> m_kernels;
  };

} // namespace mra::jit

#endif // MRA_JIT_CACHE_H
