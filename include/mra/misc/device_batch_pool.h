#ifndef MRA_MISC_DEVICE_BATCH_POOL_H
#define MRA_MISC_DEVICE_BATCH_POOL_H

/**
 * Generic per-device pool used to build ONE batched device-kernel launch from
 * a ttg::device::coop-collected batch (see mra/tasks/convolution.h for the
 * convolution use of this). Not specific to any one kernel: BatchPool<Arg> is
 * templated purely on the per-batch-member argument type `Arg` -- typically a
 * std::tuple of whatever views/pointers/indices a specific batched kernel
 * needs per member -- so any future batched kernel (compress, reconstruct,
 * multiply, derivative, norm, gaxpy, ...) can reuse this pool with its own
 * `Arg` tuple instead of hand-rolling another copy of this machinery.
 */

// NOTE: !defined(MRA_ENABLE_HOST) alone is not enough here -- under
// MRA_JIT_COMPILE none of MRA_ENABLE_HOST/CUDA/HIP are defined either, so
// this guard would otherwise let this entirely host-only batching pool
// (std::mutex/vector, ttg.h, cudaMalloc/hipMalloc) through into the JIT
// compile, which never needs it (only the host-side submit_*_kernel_batched
// wrappers in e.g. compress.h use it, never the __global__ kernel bodies).
#if !defined(MRA_ENABLE_HOST) && !defined(MRA_JIT_COMPILE)

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <ttg.h>  // for ttg::device::Stream/current_device/current_stream -- do not
                  // drop this and rely on an includer to have already pulled TTG in
                  // transitively; that's ordering-dependent (a real nvcc failure was
                  // hit here from exactly that: "name followed by '::' must be a
                  // class or namespace name" on ttg::device::Stream, because this
                  // header used to be included before whatever transitively provided it).
#include "mra/misc/allocator.h"
#include "mra/misc/platform.h"

namespace mra::detail {

#if defined(MRA_ENABLE_CUDA)
  inline void check_cuda_rt(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("BatchPool: ") + what + " failed: " + cudaGetErrorString(err));
    }
  }
#elif defined(MRA_ENABLE_HIP)
  inline void check_hip_rt(hipError_t err, const char* what) {
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("BatchPool: ") + what + " failed: " + hipGetErrorString(err));
    }
  }
#endif

  /**
   * Per-device pool of pinned host / device arrays of `Arg` used to build ONE
   * batched kernel launch. Memory is per-device, not per-stream: any slot may
   * be reused for any stream submitted on this pool's device. There is no
   * fixed slot count and no blocking wait -- acquire() scans for a slot whose
   * previous submission has completed (a non-blocking event query) and, if
   * none is free, grows the pool by one more slot rather than stalling the
   * caller. Host storage uses DeviceAllocator<Arg> (the same pinned-memory
   * allocator used for ttg::Buffer elsewhere in this codebase) instead of
   * hand-rolled cudaHostRegister calls.
   */
  template <typename Arg>
  struct BatchPool {
    struct slot_t {
      std::vector<Arg, DeviceAllocator<Arg>> host_args; // pinned host storage
      Arg* dev_args = nullptr;
      std::size_t dev_capacity = 0;
#if defined(MRA_ENABLE_CUDA)
      cudaEvent_t event;
#elif defined(MRA_ENABLE_HIP)
      hipEvent_t event;
#endif
      bool event_recorded = false; // false until this slot has been submitted at least once

      slot_t() {
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaEventCreate(&event), "cudaEventCreate");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipEventCreate(&event), "hipEventCreate");
#endif
      }

      slot_t(const slot_t&) = delete;
      slot_t& operator=(const slot_t&) = delete;

      ~slot_t() {
#if defined(MRA_ENABLE_CUDA)
        if (dev_args) cudaFree(dev_args);
        cudaEventDestroy(event);
#elif defined(MRA_ENABLE_HIP)
        if (dev_args) hipFree(dev_args);
        hipEventDestroy(event);
#endif
      }
    };

    // Named device_id, not device, purely so the ttg::device::Stream parameter
    // a few lines down doesn't read like it might be related to this member.
    explicit BatchPool(int device_id) : device_id(device_id) { }

    BatchPool(const BatchPool&) = delete;
    BatchPool& operator=(const BatchPool&) = delete;

    /* Never blocks: returns a slot with device-side capacity for at least
     * num_members entries, ready to be filled via slot_t::host_args. */
    slot_t& acquire(std::size_t num_members) {
      std::lock_guard<std::mutex> lock(mtx);
      for (auto& sp : slots) {
        if (!sp->event_recorded || event_ready(sp->event)) {
          ensure_capacity(*sp, num_members);
          return *sp;
        }
      }
      slots.push_back(std::make_unique<slot_t>());
      auto& s = *slots.back();
      ensure_capacity(s, num_members);
      return s;
    }

    /* Call right after the H2D copy + kernel launch have been issued on `stream`. */
    void mark_submitted(slot_t& s, ttg::device::Stream stream) {
#if defined(MRA_ENABLE_CUDA)
      check_cuda_rt(cudaEventRecord(s.event, stream), "cudaEventRecord");
#elif defined(MRA_ENABLE_HIP)
      check_hip_rt(hipEventRecord(s.event, stream), "hipEventRecord");
#endif
      s.event_recorded = true;
    }

    int device_id;

   private:
#if defined(MRA_ENABLE_CUDA)
    static bool event_ready(cudaEvent_t event) {
      cudaError_t err = cudaEventQuery(event);
      if (err == cudaSuccess) return true;
      if (err == cudaErrorNotReady) return false;
      check_cuda_rt(err, "cudaEventQuery");
      return false; // unreachable
    }
#elif defined(MRA_ENABLE_HIP)
    static bool event_ready(hipEvent_t event) {
      hipError_t err = hipEventQuery(event);
      if (err == hipSuccess) return true;
      if (err == hipErrorNotReady) return false;
      check_hip_rt(err, "hipEventQuery");
      return false; // unreachable
    }
#endif

    void ensure_capacity(slot_t& s, std::size_t num_members) {
      if (num_members > s.dev_capacity) {
        if (s.dev_args) {
#if defined(MRA_ENABLE_CUDA)
          check_cuda_rt(cudaFree(s.dev_args), "cudaFree");
#elif defined(MRA_ENABLE_HIP)
          check_hip_rt(hipFree(s.dev_args), "hipFree");
#endif
        }
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaMalloc(&s.dev_args, num_members*sizeof(Arg)), "cudaMalloc");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipMalloc(&s.dev_args, num_members*sizeof(Arg)), "hipMalloc");
#endif
        s.dev_capacity = num_members;
      }
      s.host_args.reserve(num_members);
    }

    std::mutex mtx;
    std::vector<std::unique_ptr<slot_t>> slots;
  };

  /**
   * Lazily constructs one BatchPool<Arg> per device, the first time that
   * device is actually used (rather than eagerly allocating memory on every
   * device up front). Construction happens from inside a device task, so
   * ttg::device::current_device() -- used as the index -- already reflects
   * the correct CUDA/HIP context; no explicit cudaSetDevice/hipSetDevice
   * bookkeeping is needed here.
   */
  template <typename Arg>
  struct BatchPoolRegistry {
    explicit BatchPoolRegistry(int num_devices, int max_batch_size)
    : entries(num_devices), max_batch_size(max_batch_size)
    { }

    BatchPool<Arg>& get(int device_id) {
      auto& e = entries[device_id];
      std::call_once(e.once, [&]{ e.pool = std::make_unique<BatchPool<Arg>>(device_id); });
      return *e.pool;
    }

    int get_max_batch_size() const { return max_batch_size; }

   private:
    struct entry_t {
      std::once_flag once;
      std::unique_ptr<BatchPool<Arg>> pool;
    };
    std::vector<entry_t> entries;
    int max_batch_size;
  };

} // namespace mra::detail

#endif // !MRA_ENABLE_HOST

#endif // MRA_MISC_DEVICE_BATCH_POOL_H
