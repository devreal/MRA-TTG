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

#ifndef MRA_ENABLE_HOST

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
#include "mra/misc/types.h"  // for size_type -- see the ttg.h note above; do not
                             // rely on an includer having pulled this in already

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
      // True from the moment acquire() hands this slot to a caller until
      // that caller's mark_submitted() call. Without this, two concurrent
      // acquire() calls (e.g. two different MockTensor pushes racing on the
      // *shared* SparsityState pool returned by sparsity_pool_registry() --
      // shared across every non-batched sparsity push, not per-kernel-type --
      // can both pass the "!event_recorded || event_ready(event)" check and
      // be handed the SAME never-yet-submitted slot, then race on filling/
      // resizing its host_args from two different threads.
      bool checked_out = false;

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
     * num_members entries, ready to be filled via slot_t::host_args. The
     * returned slot is marked checked_out (under this call's lock) so no
     * other acquire() call -- even one racing in from a different thread
     * before this caller reaches mark_submitted() -- can be handed the same
     * slot. Callers must eventually call mark_submitted() on the returned
     * slot to release it back to the pool. */
    slot_t& acquire(std::size_t num_members) {
      std::lock_guard<std::mutex> lock(mtx);
      for (auto& sp : slots) {
        if (!sp->checked_out && (!sp->event_recorded || event_ready(sp->event))) {
          ensure_capacity(*sp, num_members);
          sp->checked_out = true;
          return *sp;
        }
      }
      slots.push_back(std::make_unique<slot_t>());
      auto& s = *slots.back();
      ensure_capacity(s, num_members);
      s.checked_out = true;
      return s;
    }

    /* Call right after the H2D copy + kernel launch have been issued on
     * `stream`. Releases the slot (clears checked_out) so a future
     * acquire() may hand it out again once its device event completes. */
    void mark_submitted(slot_t& s, ttg::device::Stream stream) {
      std::lock_guard<std::mutex> lock(mtx);
#if defined(MRA_ENABLE_CUDA)
      check_cuda_rt(cudaEventRecord(s.event, stream), "cudaEventRecord");
#elif defined(MRA_ENABLE_HIP)
      check_hip_rt(hipEventRecord(s.event, stream), "hipEventRecord");
#endif
      s.event_recorded = true;
      s.checked_out = false;
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

  /**
   * Bundles the three co-allocated regions every batched-kernel leader
   * (compress/reconstruct/convolution) needs -- the per-member Arg array,
   * the (num_members+1)-entry member-offsets array, and the aggregated
   * sparsity-byte array -- into a SINGLE slot with a SINGLE CUDA/HIP event,
   * instead of three independently-tracked BatchPool<Arg> instances each
   * recording/querying their own event. This is what lets a batch leader
   * submission use one acquire()/mark_submitted() pair (one event to record
   * and later query) rather than three.
   *
   * The offsets region always uses size_type; the sparsity region is stored
   * as raw bytes (not detail::SparsityState) so this header does not need to
   * know about SparsityState (defined in mra/tensor/sparsity.h, which is not
   * included here -- see the ttg.h/size_type note at the top of this file
   * for why this header avoids that kind of include-order dependency);
   * callers reinterpret_cast dev_sparsity to SparsityState* at the point of
   * use, where that type is already visible.
   *
   * The three regions are transferred host->device with a single call (see
   * submit_grouped_copy below): on CUDA >= 12.8 that is one
   * cudaMemcpyBatchAsync describing all three copies; otherwise (older CUDA
   * toolkits, or HIP, which has no batched-copy API) it falls back to three
   * ordinary async copies issued back-to-back on the same stream -- either
   * way only ONE event is recorded for the whole slot, so the one-slot/
   * one-event bookkeeping win holds on every platform even where the copy
   * itself isn't literally batched.
   *
   * Each kernel family gets its own GroupedBatchPoolRegistry<XxxBatchArg<T,NDIM>>
   * (the Arg region's element type differs per family); unlike the old
   * per-region BatchPool<size_type>/BatchPool<SparsityState> pools, the
   * offsets/sparsity regions are no longer shared process-wide across
   * kernel families -- bundling them with the Arg region into one slot ties
   * their lifetime to that specific family's batches instead.
   */
  template <typename Arg>
  struct GroupedBatchPool {
    struct slot_t {
      std::vector<Arg, DeviceAllocator<Arg>> args;                       // pinned host storage
      std::vector<size_type, DeviceAllocator<size_type>> offsets;        // pinned host storage
      std::vector<unsigned char, DeviceAllocator<unsigned char>> sparsity; // pinned host storage (raw bytes)

      Arg* dev_args = nullptr;
      size_type* dev_offsets = nullptr;
      unsigned char* dev_sparsity = nullptr;
      std::size_t args_capacity = 0;
      std::size_t offsets_capacity = 0;
      std::size_t sparsity_capacity = 0;

#if defined(MRA_ENABLE_CUDA)
      cudaEvent_t event;
#elif defined(MRA_ENABLE_HIP)
      hipEvent_t event;
#endif
      bool event_recorded = false; // false until this slot has been submitted at least once
      // See BatchPool::slot_t::checked_out above for why this is needed.
      bool checked_out = false;

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
        if (dev_offsets) cudaFree(dev_offsets);
        if (dev_sparsity) cudaFree(dev_sparsity);
        cudaEventDestroy(event);
#elif defined(MRA_ENABLE_HIP)
        if (dev_args) hipFree(dev_args);
        if (dev_offsets) hipFree(dev_offsets);
        if (dev_sparsity) hipFree(dev_sparsity);
        hipEventDestroy(event);
#endif
      }
    };

    explicit GroupedBatchPool(int device_id) : device_id(device_id) { }

    GroupedBatchPool(const GroupedBatchPool&) = delete;
    GroupedBatchPool& operator=(const GroupedBatchPool&) = delete;

    /* Never blocks -- same reuse policy as BatchPool::acquire (one non-
     * blocking event query covering all three regions at once). Each region
     * is grown independently (only if its own requested capacity exceeds
     * what is already allocated), so growing one region never reallocates
     * the others. */
    slot_t& acquire(std::size_t num_args, std::size_t num_offsets, std::size_t num_sparsity) {
      std::lock_guard<std::mutex> lock(mtx);
      for (auto& sp : slots) {
        if (!sp->checked_out && (!sp->event_recorded || event_ready(sp->event))) {
          ensure_capacity(*sp, num_args, num_offsets, num_sparsity);
          sp->checked_out = true;
          return *sp;
        }
      }
      slots.push_back(std::make_unique<slot_t>());
      auto& s = *slots.back();
      ensure_capacity(s, num_args, num_offsets, num_sparsity);
      s.checked_out = true;
      return s;
    }

    /* Call right after the H2D copy + kernel launch(es) have been issued on
     * `stream`. Releases the slot (clears checked_out) so a future
     * acquire() may hand it out again once its device event completes. */
    void mark_submitted(slot_t& s, ttg::device::Stream stream) {
      std::lock_guard<std::mutex> lock(mtx);
#if defined(MRA_ENABLE_CUDA)
      check_cuda_rt(cudaEventRecord(s.event, stream), "cudaEventRecord");
#elif defined(MRA_ENABLE_HIP)
      check_hip_rt(hipEventRecord(s.event, stream), "hipEventRecord");
#endif
      s.event_recorded = true;
      s.checked_out = false;
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

    template <typename ElemT>
    static void grow(ElemT*& dev_ptr, std::size_t& capacity, std::size_t num_elems) {
      if (num_elems <= capacity) return;
      if (dev_ptr) {
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaFree(dev_ptr), "cudaFree");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipFree(dev_ptr), "hipFree");
#endif
      }
#if defined(MRA_ENABLE_CUDA)
      check_cuda_rt(cudaMalloc(&dev_ptr, num_elems*sizeof(ElemT)), "cudaMalloc");
#elif defined(MRA_ENABLE_HIP)
      check_hip_rt(hipMalloc(&dev_ptr, num_elems*sizeof(ElemT)), "hipMalloc");
#endif
      capacity = num_elems;
    }

    void ensure_capacity(slot_t& s, std::size_t num_args, std::size_t num_offsets, std::size_t num_sparsity) {
      grow(s.dev_args, s.args_capacity, num_args);
      grow(s.dev_offsets, s.offsets_capacity, num_offsets);
      grow(s.dev_sparsity, s.sparsity_capacity, num_sparsity);
      s.args.reserve(num_args);
      s.offsets.reserve(num_offsets);
      s.sparsity.reserve(num_sparsity);
    }

    std::mutex mtx;
    std::vector<std::unique_ptr<slot_t>> slots;
  };

  /**
   * Lazily constructs one GroupedBatchPool<Arg> per device, the first time
   * that device is actually used. Same lazy-construction rationale as
   * BatchPoolRegistry above.
   */
  template <typename Arg>
  struct GroupedBatchPoolRegistry {
    explicit GroupedBatchPoolRegistry(int num_devices, int max_batch_size)
    : entries(num_devices), max_batch_size(max_batch_size)
    { }

    GroupedBatchPool<Arg>& get(int device_id) {
      auto& e = entries[device_id];
      std::call_once(e.once, [&]{ e.pool = std::make_unique<GroupedBatchPool<Arg>>(device_id); });
      return *e.pool;
    }

    int get_max_batch_size() const { return max_batch_size; }

   private:
    struct entry_t {
      std::once_flag once;
      std::unique_ptr<GroupedBatchPool<Arg>> pool;
    };
    std::vector<entry_t> entries;
    int max_batch_size;
  };

  /**
   * Issues the host->device transfer for a GroupedBatchPool slot's three
   * regions (args/offsets/sparsity) as ONE call. num_args/num_offsets/
   * num_sparsity_bytes are the LOGICAL (used) sizes for this launch, which
   * may be smaller than the slot's allocated capacity (see
   * GroupedBatchPool::ensure_capacity).
   *
   * On CUDA 12.8+, this is a single cudaMemcpyBatchAsync call describing all
   * three copies (one shared cudaMemcpyAttributes entry, since all three are
   * plain pinned-host -> device copies with the same access-order/location
   * semantics as an ordinary cudaMemcpyAsync). On older CUDA toolkits, or on
   * HIP (no batched-copy API exists there), this falls back to three
   * ordinary async copies on the same stream; the caller still only records
   * one event for the slot afterwards (see GroupedBatchPool::mark_submitted),
   * so the slot/event count stays at one regardless of which path is taken.
   *
   * cudaMemcpyBatchAsync's signature is NOT stable across toolkits:
   * CUDA 12.8-12.x takes a trailing `size_t* failIdx` output parameter;
   * CUDA 13.0+ dropped it (and tightened dsts/srcs/sizes to const*). Both
   * versions number CUDART_VERSION as major*1000+minor*10+patch (12.8 ->
   * 12080, 13.0 -> 13000), so a single "CUDART_VERSION >= 12080" guard
   * would (wrongly) take the old 9-argument path on CUDA 13, where it fails
   * to compile ("too many arguments"). Band the check on 13000 to pick the
   * right call shape per toolkit.
   */
  template <typename Arg>
  void submit_grouped_copy(typename GroupedBatchPool<Arg>::slot_t& slot,
                            std::size_t num_args, std::size_t num_offsets, std::size_t num_sparsity_bytes,
                            int device_id, ttg::device::Stream stream)
  {
#if defined(MRA_ENABLE_CUDA) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    const void* dsts[3]   = { static_cast<void*>(slot.dev_args), static_cast<void*>(slot.dev_offsets),
                              static_cast<void*>(slot.dev_sparsity) };
    const void* srcs[3]   = { static_cast<void*>(slot.args.data()), static_cast<void*>(slot.offsets.data()),
                              static_cast<void*>(slot.sparsity.data()) };
    std::size_t sizes[3]  = { num_args*sizeof(Arg), num_offsets*sizeof(size_type), num_sparsity_bytes };
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint = cudaMemLocation{cudaMemLocationTypeHost, 0};
    attrs.dstLocHint = cudaMemLocation{cudaMemLocationTypeDevice, device_id};
    attrs.flags = cudaMemcpyFlagDefault;
    std::size_t attrsIdxs[1] = { 2 }; // one attribute set applies to entries [0,2]
#if CUDART_VERSION >= 13000
    check_cuda_rt(cudaMemcpyBatchAsync(dsts, srcs, sizes, 3, &attrs, attrsIdxs, 1, stream),
                  "cudaMemcpyBatchAsync");
#else
    std::size_t failIdx = 0;
    check_cuda_rt(cudaMemcpyBatchAsync(dsts, srcs, sizes, 3, &attrs, attrsIdxs, 1, &failIdx, stream),
                  "cudaMemcpyBatchAsync");
#endif
#elif defined(MRA_ENABLE_CUDA)
    // Toolkit predates cudaMemcpyBatchAsync (added in CUDA 12.8).
    check_cuda_rt(cudaMemcpyAsync(slot.dev_args, slot.args.data(), num_args*sizeof(Arg),
                                  cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    check_cuda_rt(cudaMemcpyAsync(slot.dev_offsets, slot.offsets.data(), num_offsets*sizeof(size_type),
                                  cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    check_cuda_rt(cudaMemcpyAsync(slot.dev_sparsity, slot.sparsity.data(), num_sparsity_bytes,
                                  cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    // No HIP equivalent of cudaMemcpyBatchAsync exists yet.
    check_hip_rt(hipMemcpyAsync(slot.dev_args, slot.args.data(), num_args*sizeof(Arg),
                                hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    check_hip_rt(hipMemcpyAsync(slot.dev_offsets, slot.offsets.data(), num_offsets*sizeof(size_type),
                                hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    check_hip_rt(hipMemcpyAsync(slot.dev_sparsity, slot.sparsity.data(), num_sparsity_bytes,
                                hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif
  }

  /**
   * Finds which member owns global position `pos` in a flattened batch
   * launch, given that batch's ascending member-offsets array (size
   * num_members+1, offsets[0] == 0, offsets[num_members] == total_nonzero),
   * and writes that member's local position (pos - offsets[member]) to
   * `local_pos_out`. O(num_members) linear scan -- fine since num_members is
   * small (bounded by max_batch_size, O(100)); call once per block (e.g. by
   * the team lead, sharing the result via a SHARED variable) rather than
   * once per thread.
   */
  SCOPE size_type find_member_for_pos(const size_type* offsets, size_type num_members,
                                       size_type pos, size_type* local_pos_out) {
    for (size_type m = 0; m < num_members; ++m) {
      if (pos < offsets[m + 1]) {
        *local_pos_out = pos - offsets[m];
        return m;
      }
    }
    assert(false && "find_member_for_pos: pos out of range of the batch's member offsets");
    return num_members;
  }

} // namespace mra::detail

#endif // !MRA_ENABLE_HOST

#endif // MRA_MISC_DEVICE_BATCH_POOL_H
