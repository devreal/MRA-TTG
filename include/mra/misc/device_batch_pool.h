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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
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
#include "ttg/parsec/device.h"  // ttg_parsec::detail::ttg_device_to_parsec_device/first_device_id --
                                // for translating this pool's plain device id into PaRSEC's own
                                // device-id space, the same translation Buffer::allocate_on uses.
#include <parsec/mca/device/device_gpu.h>
#include <parsec/utils/zone_malloc.h>
#include "mra/misc/allocator.h"
#include "mra/misc/platform.h"
#include "mra/misc/types.h"  // for size_type -- see the ttg.h note above; do not
                             // rely on an includer having pulled this in already

namespace mra::detail {

  /**
   * zone_malloc against the SAME per-GPU memory pool PaRSEC pre-reserves at
   * startup and that ttg::Buffer's own device-side allocation already draws
   * from (see ttg_parsec::Buffer::allocate_on in ttg/parsec/buffer.h) --
   * `device_id` is this file's usual plain (TTG-space) device id, not
   * PaRSEC's; translated here exactly as ttg_parsec::Buffer::allocate_on
   * does.
   */
  inline parsec_device_gpu_module_t* parsec_gpu_device(int device_id) {
    int parsec_id = device_id + ttg_parsec::detail::first_device_id;
    return (parsec_device_gpu_module_t*)parsec_mca_device_get(parsec_id);
  }

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
   * A single device allocation, requested once and never resized: the first
   * call to ensure() allocates it -- preferring PaRSEC's zone_malloc (the
   * same per-GPU pool ttg::Buffer's own device-side allocation draws from,
   * see the comment on parsec_gpu_device above), falling back to a plain
   * cudaMalloc/hipMalloc if the zone can't satisfy the request right now
   * (e.g. temporarily exhausted) -- and every later call just returns that
   * same buffer. A later call asking for MORE than the first one already
   * provided throws rather than freeing and reallocating: zone_malloc/
   * zone_free are plain host-side bookkeeping over the device's shared
   * memory pool, with no awareness of which GPU kernels (on this buffer's
   * own stream, or anyone else's) might still be running against a freed
   * range, so handing it straight back to the zone for immediate reuse
   * elsewhere has none of cudaFree's implicit synchronization safety net --
   * safe only once nothing could possibly still be using the memory, which
   * for every current owner (BatchPool::slot_t, GroupedBatchPool::slot_t)
   * is exactly when this object itself is destroyed (process-lifetime pool
   * teardown), never mid-run. The destructor frees via whichever allocator
   * actually provided the memory (zone_free or cudaFree/hipFree), so PaRSEC
   * isn't left thinking this range is still outstanding. Provides an
   * implicit conversion to ElemT* so existing call sites that pass this
   * object directly to a kernel launch or memcpy (expecting a raw pointer)
   * keep working unchanged.
   *
   * grow() is the one exception to "never resized": an owner that can
   * guarantee no device work still references the region (e.g. a
   * GroupedBatchPool slot between acquire(), which only hands out a slot
   * whose previous submission's event has completed, and its next
   * copy/launch) may reallocate it to a larger size.
   */
  template <typename ElemT>
  class FixedDeviceRegion {
   public:
    FixedDeviceRegion() = default;
    FixedDeviceRegion(const FixedDeviceRegion&) = delete;
    FixedDeviceRegion& operator=(const FixedDeviceRegion&) = delete;

    ~FixedDeviceRegion() {
      release();
    }

    ElemT* get() const { return ptr_; }
    operator ElemT*() const { return ptr_; }
    bool empty() const { return ptr_ == nullptr; }

    void ensure(int device_id, std::size_t num_elems) {
      if (ptr_ == nullptr) {
        allocate(device_id, num_elems);
      } else if (num_elems > capacity_) {
        throw std::runtime_error(
            "FixedDeviceRegion::ensure: requested size exceeds the fixed capacity allocated "
            "at first use -- this region is sized once from a known fixed upper bound and "
            "never regrown (see class comment for why growing it would be unsafe)");
      }
    }

    /* Reallocates (discarding the contents) if num_elems exceeds the current
     * capacity. Only valid while no device work references the region, see
     * the class comment. */
    void grow(int device_id, std::size_t num_elems) {
      if (ptr_ != nullptr && num_elems <= capacity_) return;
      release();
      allocate(device_id, num_elems);
    }

   private:
    void allocate(int device_id, std::size_t num_elems) {
      device_id_ = device_id;
      void* p = zone_malloc(parsec_gpu_device(device_id)->memory, num_elems*sizeof(ElemT));
      if (p != nullptr) {
        ptr_ = static_cast<ElemT*>(p);
        from_zone_ = true;
      } else {
        // Zone exhausted right now -- fall back to a plain allocation
        // rather than failing outright.
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaMalloc(&ptr_, num_elems*sizeof(ElemT)), "cudaMalloc (zone_malloc fallback)");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipMalloc(&ptr_, num_elems*sizeof(ElemT)), "hipMalloc (zone_malloc fallback)");
#endif
        from_zone_ = false;
      }
      capacity_ = num_elems;
    }

    void release() {
      if (!ptr_) return;
      if (from_zone_) {
        zone_free(parsec_gpu_device(device_id_)->memory, ptr_);
      } else {
#if defined(MRA_ENABLE_CUDA)
        cudaFree(ptr_);
#elif defined(MRA_ENABLE_HIP)
        hipFree(ptr_);
#endif
      }
      ptr_ = nullptr;
      capacity_ = 0;
    }

    ElemT* ptr_ = nullptr;
    std::size_t capacity_ = 0;
    int device_id_ = -1;
    bool from_zone_ = false;
  };

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
      FixedDeviceRegion<Arg> dev_args;
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
        check_cuda_rt(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipEventCreateWithFlags(&event, hipEventDisableTiming), "hipEventCreate");
#endif
      }

      slot_t(const slot_t&) = delete;
      slot_t& operator=(const slot_t&) = delete;

      ~slot_t() {
        // dev_args frees itself (FixedDeviceRegion's destructor).
#if defined(MRA_ENABLE_CUDA)
        // the device regions are freed after this body runs: wait for the last
        // copy or launch that used them
        if (event_recorded) cudaEventSynchronize(event);
        cudaEventDestroy(event);
#elif defined(MRA_ENABLE_HIP)
        if (event_recorded) hipEventSynchronize(event);
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
      s.dev_args.ensure(device_id, num_members);
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
      BatchPool<Arg>* p = e.pool.load(std::memory_order_acquire);
      if (p == nullptr) {
        std::lock_guard<std::mutex> lock(e.mtx);
        p = e.pool.load(std::memory_order_relaxed);
        if (p == nullptr) {
          p = new BatchPool<Arg>(device_id);
          e.pool.store(p, std::memory_order_release);
        }
      }
      return *p;
    }

    /**
     * Frees every device's pool. The pools' device memory comes from PaRSEC's
     * zone (see FixedDeviceRegion), so a registry that outlives the TTs using
     * it (a process-wide static, e.g. sparsity_pool_registry()) must release
     * it before PaRSEC tears its devices down: mra::finalize() runs this via
     * mra::detail::at_finalize. No task may use the registry concurrently; a
     * later get() creates the pool anew.
     */
    void release() {
      for (auto& e : entries) {
        std::lock_guard<std::mutex> lock(e.mtx);
        delete e.pool.exchange(nullptr, std::memory_order_acq_rel);
      }
    }

    int get_max_batch_size() const { return max_batch_size; }

   private:
    struct entry_t {
      std::mutex mtx;
      std::atomic<BatchPool<Arg>*> pool{nullptr};
      ~entry_t() { delete pool.load(std::memory_order_relaxed); }
    };
    std::vector<entry_t> entries;
    int max_batch_size;
  };

  /**
   * Bundles what every batched-kernel leader (compress/reconstruct/
   * convolution) needs into a SINGLE slot with a SINGLE CUDA/HIP event,
   * instead of several independently-tracked BatchPool<Arg> instances each
   * recording/querying their own event. This is what lets a batch leader
   * submission use one acquire()/mark_submitted() pair (one event to record
   * and later query) rather than several.
   *
   * Host side: the per-member Arg array, the (num_members+1)-entry
   * member-offsets array (always size_type) and a pinned staging region,
   * `sparsity` (raw bytes, not
   * detail::SparsityState, so this header does not need to know about
   * SparsityState -- defined in mra/tensor/sparsity.h, not included here,
   * see the ttg.h/size_type note at the top of this file for why), used to
   * gather each member's sparsity bytes from its host-side
   * RangeSparsityBase before they're scattered to their own final
   * destination tensors.
   *
   * `extra_dsts`/`extra_srcs`/`extra_sizes` name every one of those
   * point-to-point destination copies (rebuilt fresh every launch, unlike
   * the other regions -- see their own comment on slot_t). submit_grouped_copy
   * packs args, offsets, a descriptor table of the extra copies and their
   * source bytes into one pinned buffer (`packed`), transfers it with ONE
   * async copy into one device region (`dev_packed`), and scatters the extra
   * copies with one small kernel -- see submit_grouped_copy for why this
   * replaced cudaMemcpyBatchAsync. Only ONE event is recorded for the whole
   * slot.
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
      // Pinned host staging for each member's per-tensor sparsity bytes,
      // gathered from host-side RangeSparsityBase (via sparsity_to_bytes).
      // Source-only: submit_grouped_copy packs these bytes and scatters them
      // to each destination tensor's own device buffer.
      std::vector<unsigned char, DeviceAllocator<unsigned char>> sparsity;

      // Per-launch list of additional point-to-point H2D copies folded into
      // the same submit_grouped_copy call as args/offsets -- populated fresh
      // by the caller's submit_*_batch_leader before each submit_grouped_copy
      // (cleared and rebuilt every launch, not persisted/grown like
      // args/offsets/sparsity). Typically one entry per member per
      // destination tensor (scattering `sparsity` above into each member's
      // own, separately-allocated sparsity bitfield -- see
      // submit_grouped_copy), plus, for families that need it (e.g.
      // convolution's resnorms), zero-fill entries: a nullptr source
      // zero-fills extra_sizes[k] bytes at extra_dsts[k]. Any non-null
      // source must be host memory that stays valid until
      // submit_grouped_copy returns (it is packed into `packed` there).
      std::vector<void*> extra_dsts;
      std::vector<const void*> extra_srcs;
      std::vector<std::size_t> extra_sizes;

      // Pinned host staging for submit_grouped_copy's single H2D transfer:
      // [args | offsets | GroupedCopyDesc table | source bytes]. Only ever
      // grows (never cleared), so steady-state launches neither reallocate
      // nor re-zero it.
      std::vector<unsigned char, DeviceAllocator<unsigned char>> packed;
      // Device counterpart of `packed`. Grown in submit_grouped_copy while
      // the slot is checked out -- safe because acquire() only hands out a
      // slot once its previous submission has completed (see
      // FixedDeviceRegion::grow).
      FixedDeviceRegion<unsigned char> dev_packed;

      // Point into dev_packed; set by submit_grouped_copy, so they are only
      // valid for kernels launched after it.
      Arg* dev_args = nullptr;
      size_type* dev_offsets = nullptr;

      // Persistent, zone_malloc-backed scratch region shared by every launch
      // of this kernel family on this device -- NOT per-member. Paired with
      // kernel_grid_cap (platform.h): once the launch grid is capped at
      // actual device occupancy instead of one block per work item, each
      // kernel's own per-block scratch only needs occupancy-many slots
      // (indexed by blockIdx.x inside the kernel), not one per item -- see
      // e.g. submit_convolution_kernel_batched. Tied to this slot (not a
      // single pool-wide buffer) so the SAME non-blocking event-based reuse
      // that already protects dev_args/dev_offsets from two concurrent
      // launches aliasing the same memory also protects this region: a slot
      // is never handed out by acquire() until its previous submission's
      // event has completed.
      FixedDeviceRegion<unsigned char> dev_scratch;

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
        check_cuda_rt(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipEventCreateWithFlags(&event, hipEventDisableTiming), "hipEventCreate");
#endif
      }

      slot_t(const slot_t&) = delete;
      slot_t& operator=(const slot_t&) = delete;

      ~slot_t() {
        // dev_packed/dev_scratch each free themselves
        // (FixedDeviceRegion's destructor, via the matching deallocator for
        // however they were actually allocated).
#if defined(MRA_ENABLE_CUDA)
        // the device regions are freed after this body runs: wait for the last
        // copy or launch that used them
        if (event_recorded) cudaEventSynchronize(event);
        cudaEventDestroy(event);
#elif defined(MRA_ENABLE_HIP)
        if (event_recorded) hipEventSynchronize(event);
        hipEventDestroy(event);
#endif
      }
    };

    explicit GroupedBatchPool(int device_id) : device_id(device_id) { }

    // Delegates to FixedDeviceRegion::ensure (see its class comment) for this
    // slot's shared scratch region. Callers are expected to size num_elems
    // from kernel_grid_cap (platform.h) -- a fixed bound for a given
    // kernel/device, independent of any particular launch's batch size -- so
    // the first call already requests the one size every later call will
    // also need; exceeding it indicates that assumption broke (e.g. K
    // changed at runtime). No locking: the slot is already exclusively
    // checked_out to whoever is calling this (between their own
    // acquire()/mark_submitted() pair), same as slot.args/slot.offsets being
    // mutated without extra locking elsewhere.
    template <typename ElemT>
    ElemT* ensure_scratch(slot_t& s, std::size_t num_elems) {
      s.dev_scratch.ensure(device_id, num_elems * sizeof(ElemT));
      return reinterpret_cast<ElemT*>(s.dev_scratch.get());
    }

    GroupedBatchPool(const GroupedBatchPool&) = delete;
    GroupedBatchPool& operator=(const GroupedBatchPool&) = delete;

    /* Never blocks -- same reuse policy as BatchPool::acquire (one non-
     * blocking event query covering every region at once). Only reserves the
     * host-side vectors; the device region is sized by submit_grouped_copy
     * from what a launch actually packs. extra_dsts/extra_srcs/extra_sizes
     * are rebuilt fresh every launch by the caller, not sized here. */
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

    void ensure_capacity(slot_t& s, std::size_t num_args, std::size_t num_offsets, std::size_t num_sparsity) {
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
   * One extra copy of a GroupedBatchPool slot, as packed by
   * submit_grouped_copy: `size` bytes from offset `src_offset` of the packed
   * source bytes to `dst`, or a zero-fill of `size` bytes at `dst` if
   * src_offset == zero_fill.
   */
  struct GroupedCopyDesc {
    static constexpr std::uint32_t zero_fill = ~std::uint32_t(0);
    void* dst;
    std::uint32_t src_offset;
    std::uint32_t size;
  };

#if defined(__CUDACC__) || defined(__HIP__)
  /* One warp per descriptor, grid-stride over descriptors. A template only
   * so that every translation unit including this header may define it. */
  template <typename DescT>
  __global__ void grouped_copy_scatter_kernel(const DescT* __restrict__ descs, std::uint32_t num_descs,
                                              const unsigned char* __restrict__ src)
  {
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned warps_per_block = blockDim.x / warpSize;
    const unsigned num_warps = gridDim.x * warps_per_block;
    for (unsigned d = blockIdx.x * warps_per_block + threadIdx.x / warpSize; d < num_descs; d += num_warps) {
      const DescT desc = descs[d];
      unsigned char* dst = static_cast<unsigned char*>(desc.dst);
      if (desc.src_offset == DescT::zero_fill) {
        for (std::uint32_t i = lane; i < desc.size; i += warpSize) dst[i] = 0;
      } else {
        const unsigned char* s = src + desc.src_offset;
        for (std::uint32_t i = lane; i < desc.size; i += warpSize) dst[i] = s[i];
      }
    }
  }
#endif // __CUDACC__ || __HIP__

  /**
   * Issues the host->device transfer for a GroupedBatchPool slot's args and
   * offsets, PLUS every point-to-point copy queued in
   * slot.extra_dsts/extra_srcs/extra_sizes, and points slot.dev_args/
   * slot.dev_offsets at the transferred arrays. num_args/num_offsets are the
   * LOGICAL (used) sizes for this launch.
   *
   * The extra copies typically take slices of slot.sparsity (this batch's
   * pinned, host-side-aggregated sparsity bytes for one member's one
   * destination tensor) to that tensor's own device buffer --
   * slot.extra_dsts[k] is the destination tensor's buffer().device_ptr_on(device),
   * which coincides exactly with where its inline sparsity bitfield starts
   * (byte offset 0 -- see SparseArrayBase's layout). A nullptr source
   * zero-fills the destination instead (e.g. convolution's resnorms_view).
   *
   * Everything is packed on the host into slot.packed as
   * [args | offsets | GroupedCopyDesc table | source bytes], transferred with
   * ONE async copy into slot.dev_packed, and the extra copies are then done
   * by grouped_copy_scatter_kernel on the same stream, so kernels launched
   * afterwards see them complete. slot.sparsity is packed with a single
   * memcpy; any other source is appended individually.
   *
   * This used to be one cudaMemcpyBatchAsync describing every copy, but
   * that call's host-side cost grows by ~0.7us per copy (~200us for a
   * 138-member batch with two copies each, on an A100/EPYC node), which the
   * calling thread cannot overlap with anything. Packing costs ~6us
   * regardless of batch size, at the price of one extra small kernel on the
   * device.
   */
  template <typename Arg>
  void submit_grouped_copy(typename GroupedBatchPool<Arg>::slot_t& slot,
                            std::size_t num_args, std::size_t num_offsets,
                            int device_id, ttg::device::Stream stream)
  {
#if defined(__CUDACC__) || defined(__HIP__)
    auto align_up = [](std::size_t n) { return (n + 15) & ~std::size_t(15); };
    const std::size_t num_extra = slot.extra_dsts.size();
    const unsigned char* sparsity_begin = slot.sparsity.data();
    const unsigned char* sparsity_end = sparsity_begin + slot.sparsity.size();

    // Sources outside slot.sparsity are appended after it.
    std::size_t num_descs = 0;
    std::size_t num_src_bytes = slot.sparsity.size();
    for (std::size_t k = 0; k < num_extra; ++k) {
      if (slot.extra_sizes[k] == 0) continue;
      ++num_descs;
      auto src = static_cast<const unsigned char*>(slot.extra_srcs[k]);
      if (src != nullptr && !(src >= sparsity_begin && src + slot.extra_sizes[k] <= sparsity_end)) {
        num_src_bytes += slot.extra_sizes[k];
      }
    }

    const std::size_t args_bytes = num_args*sizeof(Arg);
    const std::size_t offsets_off = align_up(args_bytes);
    const std::size_t descs_off = align_up(offsets_off + num_offsets*sizeof(size_type));
    const std::size_t src_off = descs_off + num_descs*sizeof(GroupedCopyDesc);
    const std::size_t total = src_off + num_src_bytes;
    if (num_src_bytes > GroupedCopyDesc::zero_fill) {
      throw std::runtime_error("submit_grouped_copy: source bytes exceed 32-bit descriptor offsets");
    }

    if (slot.packed.size() < total) slot.packed.resize(total + total/4);
    slot.dev_packed.grow(device_id, slot.packed.size());

    unsigned char* h = slot.packed.data();
    std::memcpy(h, slot.args.data(), args_bytes);
    std::memcpy(h + offsets_off, slot.offsets.data(), num_offsets*sizeof(size_type));
    std::memcpy(h + src_off, sparsity_begin, slot.sparsity.size());
    auto descs = reinterpret_cast<GroupedCopyDesc*>(h + descs_off);
    std::size_t d = 0;
    std::size_t appended = slot.sparsity.size();
    for (std::size_t k = 0; k < num_extra; ++k) {
      const std::size_t size = slot.extra_sizes[k];
      if (size == 0) continue;
      auto src = static_cast<const unsigned char*>(slot.extra_srcs[k]);
      std::uint32_t offset;
      if (src == nullptr) {
        offset = GroupedCopyDesc::zero_fill;
      } else if (src >= sparsity_begin && src + size <= sparsity_end) {
        offset = static_cast<std::uint32_t>(src - sparsity_begin);
      } else {
        std::memcpy(h + src_off + appended, src, size);
        offset = static_cast<std::uint32_t>(appended);
        appended += size;
      }
      descs[d++] = GroupedCopyDesc{slot.extra_dsts[k], offset, static_cast<std::uint32_t>(size)};
    }

    unsigned char* dev = slot.dev_packed.get();
    slot.dev_args = reinterpret_cast<Arg*>(dev);
    slot.dev_offsets = reinterpret_cast<size_type*>(dev + offsets_off);
    if (total == 0) return;
#if defined(MRA_ENABLE_CUDA)
    check_cuda_rt(cudaMemcpyAsync(dev, h, total, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    check_hip_rt(hipMemcpyAsync(dev, h, total, hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif
    if (num_descs > 0) {
      // The kernel uses the device's actual warpSize; 32 only sizes the grid.
      constexpr unsigned threads = 128;
      constexpr unsigned warps_per_block = threads / 32;
      const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((num_descs + warps_per_block - 1) / warps_per_block, 256));
      CALL_KERNEL(grouped_copy_scatter_kernel<GroupedCopyDesc>, blocks, threads, 0, stream,
                  (reinterpret_cast<const GroupedCopyDesc*>(dev + descs_off),
                   static_cast<std::uint32_t>(num_descs), dev + src_off));
    }
#else
    static_assert(sizeof(Arg) == 0, "submit_grouped_copy must be instantiated in a CUDA/HIP translation unit");
#endif // __CUDACC__ || __HIP__
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
