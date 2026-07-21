#ifndef MRA_MISC_BATCH_SIZE_H
#define MRA_MISC_BATCH_SIZE_H

#include <atomic>
#include <cstddef>

/**
 * Process-wide device-kernel batch size, shared by every batching-capable
 * kernel (convolution today; eventually compress, reconstruct, multiply,
 * derivative, norm, gaxpy, ...) instead of each one taking its own
 * enable/size parameters. A value <= 1 disables batching entirely.
 *
 * Set this once, before constructing the task graphs that should observe it
 * (e.g. before make_convolution(...)): each graph reads get_batch_size() once
 * at construction time and bakes the resulting batching decision into its
 * tasks (whether set_batch_matcher is installed at all, and which runtime
 * path each task takes), so changing the value after a graph is already
 * built has no effect on that graph -- this is deliberate, not an oversight:
 * a task already matched into a device-stream batch ring must unconditionally
 * reach its own coop() call, so the enable/disable decision cannot safely be
 * re-read per task invocation once matching has started.
 */

#ifdef MRA_ENABLE_HOST
// batching disabled on host builds
#define MRA_BATCH_SIZE_DEFAULT 1
#else
// batching enabled on device builds, default size 128 (arbitrary, but not too small)
#define MRA_BATCH_SIZE_DEFAULT 128
#endif

namespace mra {

  namespace detail {
    inline std::atomic<std::size_t>& batch_size_state() {
      static std::atomic<std::size_t> value{MRA_BATCH_SIZE_DEFAULT};
      return value;
    }
  } // namespace detail

  /** Sets the process-wide max batch size. A value <= 1 disables batching. */
  inline void set_batch_size(std::size_t n) {
    detail::batch_size_state().store(n, std::memory_order_relaxed);
  }

  /** Returns the current process-wide max batch size (default: 1, i.e. disabled). */
  inline std::size_t get_batch_size() {
    return detail::batch_size_state().load(std::memory_order_relaxed);
  }

  /** Convenience: true iff get_batch_size() > 1. */
  inline bool batching_enabled() {
    return get_batch_size() > 1;
  }

} // namespace mra

#endif // MRA_MISC_BATCH_SIZE_H
