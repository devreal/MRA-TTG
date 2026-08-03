#ifndef MRA_JIT_LAUNCH_H
#define MRA_JIT_LAUNCH_H

#include "mra/jit/compiler.h"
#include "mra/misc/platform.h"

#include <cstddef>

namespace mra::jit {

  /**
   * Implemented in src/jit/compiler.cc (needs the real cuLaunchKernel/
   * hipModuleLaunchKernel call) -- packs already-boxed argument pointers
   * into the launch call. Not meant to be called directly; use launch()
   * below, which builds `kernel_args` from real arguments via their
   * addresses.
   */
  void launch_impl(const CompiledKernel& kernel, Dim3 grid, Dim3 block,
                    unsigned int shared_mem_bytes, DeviceStream stream,
                    void** kernel_args);

  /**
   * Launches a JIT-compiled kernel, packing `args` into the void* array
   * cuLaunchKernel/hipModuleLaunchKernel expects. `args` must be passed in
   * the exact order (and of the exact types) the target __global__ function
   * declares them -- there is no compile-time check against the kernel
   * signature (unlike CALL_KERNEL's <<<>>> syntax for the AOT path), since
   * the JIT'd kernel is a runtime handle, not a compile-time symbol.
   */
  template <typename... Args>
  void launch(const CompiledKernel& kernel, Dim3 grid, Dim3 block,
              unsigned int shared_mem_bytes, DeviceStream stream, const Args&... args) {
    // cuLaunchKernel/hipModuleLaunchKernel only read through these pointers
    // to copy each argument's bytes into the kernel's parameter buffer, so
    // casting away constness here is safe.
    void* kernel_args[] = { const_cast<void*>(static_cast<const void*>(&args))... };
    launch_impl(kernel, grid, block, shared_mem_bytes, stream, kernel_args);
  }

} // namespace mra::jit

#endif // MRA_JIT_LAUNCH_H
