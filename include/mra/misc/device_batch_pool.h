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
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cxxabi.h>
#include <typeinfo>
#include <map>
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
#include "mra/misc/init.h"  // at_finalize
#include "mra/misc/platform.h"
#include "mra/misc/types.h"  // for size_type -- see the ttg.h note above; do not
                             // rely on an includer having pulled this in already
#include "mra/tensor/sparsity.h"  // find_nth_pred_warp
#if defined(MRA_ENABLE_CUDA)
#include <nvtx3/nvToolsExt.h>
#endif

namespace mra::detail {

  /**
   * NVTX range around the submission of one batch (domain "MRA-TTG batches"), named after
   * the kernel family with the number of tasks in the batch as its payload. The range holds
   * the kernel's launch, so nsys shows it on the GPU too (NVTX GPU projection), where the
   * block count does not tell the batch size (e.g., the convolution's grid is capped).
   */
  class BatchRange {
  public:
    BatchRange(const char* name, std::size_t ntasks) {
#if defined(MRA_ENABLE_CUDA)
      static nvtxDomainHandle_t domain = nvtxDomainCreateA("MRA-TTG batches");
      domain_ = domain;
      nvtxEventAttributes_t attr = {};
      attr.version = NVTX_VERSION;
      attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
      attr.message.ascii = name;
      attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
      attr.payload.ullValue = ntasks;
      nvtxDomainRangePushEx(domain_, &attr);
#else
      (void)name; (void)ntasks;
#endif
    }
    ~BatchRange() {
#if defined(MRA_ENABLE_CUDA)
      nvtxDomainRangePop(domain_);
#endif
    }
    BatchRange(const BatchRange&) = delete;
    BatchRange& operator=(const BatchRange&) = delete;
  private:
#if defined(MRA_ENABLE_CUDA)
    nvtxDomainHandle_t domain_;
#endif
  };

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
   * Accounting of the device regions of the batch pools (FixedDeviceRegion), per pool kind
   * (kernel family and region) and device: bytes held now and at the peak, how many came from
   * PaRSEC's zone and how many from the cudaMalloc/hipMalloc fallback, and how many regions
   * (one per slot) exist. Printed at mra::finalize with MRA_TTG_REGION_STATS=1; each fallback
   * allocation and each failure is reported as it happens with MRA_TTG_REGION_STATS=2.
   */
  struct DeviceRegionStats {
    struct entry_t {
      std::string name;
      int device = -1;
      std::atomic<std::int64_t> bytes{0}, peak_bytes{0}, regions{0}, peak_regions{0};
      std::atomic<std::int64_t> zone_allocs{0}, malloc_allocs{0}, malloc_bytes{0}, failures{0};
      std::atomic<std::int64_t> grows{0};
    };

    static int level() {
      static const int l = [] {
        const char* v = std::getenv("MRA_TTG_REGION_STATS");
        return v ? std::atoi(v) : 0;
      }();
      return l;
    }

    static DeviceRegionStats& instance() {
      static DeviceRegionStats* s = [] {
        auto* p = new DeviceRegionStats();  /* leaked: used until the end of the process */
        at_finalize([p] { if (level() > 0) p->print("finalize"); });
        return p;
      }();
      return *s;
    }

    entry_t* entry(const std::string& name, int device) {
      std::lock_guard<std::mutex> lock(mtx);
      for (auto& e : entries) if (e->name == name && e->device == device) return e.get();
      entries.push_back(std::make_unique<entry_t>());
      entries.back()->name = name;
      entries.back()->device = device;
      return entries.back().get();
    }

    static void raise_max(std::atomic<std::int64_t>& m, std::int64_t v) {
      std::int64_t cur = m.load(std::memory_order_relaxed);
      while (v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) { }
    }

    void on_alloc(entry_t* e, std::size_t bytes, bool from_zone) {
      if (e) {
        raise_max(e->peak_bytes, e->bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes);
        raise_max(e->peak_regions, e->regions.fetch_add(1, std::memory_order_relaxed) + 1);
        (from_zone ? e->zone_allocs : e->malloc_allocs).fetch_add(1, std::memory_order_relaxed);
        if (!from_zone) e->malloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
      raise_max(peak_total, total.fetch_add(bytes, std::memory_order_relaxed) + bytes);
    }

    void on_free(entry_t* e, std::size_t bytes) {
      if (e) {
        e->bytes.fetch_sub(bytes, std::memory_order_relaxed);
        e->regions.fetch_sub(1, std::memory_order_relaxed);
      }
      total.fetch_sub(bytes, std::memory_order_relaxed);
    }

    void print(const char* when, const entry_t* highlight = nullptr) {
      std::lock_guard<std::mutex> lock(mtx);
      std::fprintf(stderr, "[%d] MRA-TTG device regions (%s): now %.1f MB, peak %.1f MB\n",
                   ttg::default_execution_context().rank(), when, total.load() / 1048576.0,
                   peak_total.load() / 1048576.0);
      std::fprintf(stderr, "[%d]   %-3s %10s %10s %7s %7s %7s %7s %10s %5s %6s  %s\n",
                   ttg::default_execution_context().rank(), "dev", "now MB", "peak MB",
                   "regions", "peak", "zone", "malloc", "malloc MB", "fail", "grows", "pool");
      for (auto& e : entries) {
        std::fprintf(stderr, "[%d] %c %-3d %10.1f %10.1f %7lld %7lld %7lld %7lld %10.1f %5lld %6lld  %s\n",
                     ttg::default_execution_context().rank(), e.get() == highlight ? '*' : ' ',
                     e->device,
                     e->bytes.load() / 1048576.0, e->peak_bytes.load() / 1048576.0,
                     (long long)e->regions.load(), (long long)e->peak_regions.load(),
                     (long long)e->zone_allocs.load(), (long long)e->malloc_allocs.load(),
                     e->malloc_bytes.load() / 1048576.0, (long long)e->failures.load(),
                     (long long)e->grows.load(), e->name.c_str());
      }
    }

    std::mutex mtx;
    std::vector<std::unique_ptr<entry_t>> entries;
    std::atomic<std::int64_t> total{0}, peak_total{0};
  };

  /* "MulCompressedBatchArg<double, 3ul>.scratch" for Arg = mra::detail::MulCompressedBatchArg<double, 3ul> */
  template <typename Arg>
  std::string region_stats_name(const char* region) {
    std::string n;
    int status = 0;
    char* d = abi::__cxa_demangle(typeid(Arg).name(), nullptr, nullptr, &status);
    n = (status == 0 && d) ? d : typeid(Arg).name();
    std::free(d);
    for (const char* ns : {"mra::detail::", "mra::", "std::"}) {
      for (auto pos = n.find(ns); pos != std::string::npos; pos = n.find(ns)) n.erase(pos, std::strlen(ns));
    }
    return n + "." + region;
  }

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
    /* the DeviceRegionStats entry this region is counted in */
    void set_stats(DeviceRegionStats::entry_t* e) { stats_ = e; }
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
      auto* gpu = parsec_gpu_device(device_id);
      const std::size_t bytes = num_elems*sizeof(ElemT);
      void* p = zone_malloc(gpu->memory, bytes);
      if (p != nullptr) {
        ptr_ = static_cast<ElemT*>(p);
        from_zone_ = true;
      } else {
        // Zone exhausted right now -- fall back to a plain allocation
        // rather than failing outright.
        auto& stats = DeviceRegionStats::instance();
        std::size_t free_b = 0, total_b = 0;
#if defined(MRA_ENABLE_CUDA)
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        cudaMemGetInfo(&free_b, &total_b);
#elif defined(MRA_ENABLE_HIP)
        hipError_t err = hipMalloc(&ptr_, bytes);
        hipMemGetInfo(&free_b, &total_b);
#endif
        const bool failed = (err != decltype(err){});
        if (failed && stats_) stats_->failures.fetch_add(1, std::memory_order_relaxed);
        if (DeviceRegionStats::level() > 1 || failed) {
          std::fprintf(stderr, "[%d] MRA-TTG device region %s: cudaMalloc of %.1f MB on device %d "
                       "(zone in use %.1f of %.1f MB, device free %.1f of %.1f MB) for %s\n",
                       ttg::default_execution_context().rank(), failed ? "FAILED" : "fallback",
                       bytes / 1048576.0, device_id,
                       zone_in_use(gpu->memory) / 1048576.0,
                       (double)gpu->mem_block_size * gpu->mem_nb_blocks / 1048576.0,
                       free_b / 1048576.0, total_b / 1048576.0,
                       stats_ ? stats_->name.c_str() : "?");
          if (failed) stats.print("allocation failure", stats_);
        }
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(err, "cudaMalloc (zone_malloc fallback)");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(err, "hipMalloc (zone_malloc fallback)");
#endif
        from_zone_ = false;
      }
      capacity_ = num_elems;
      DeviceRegionStats::instance().on_alloc(stats_, bytes, from_zone_);
    }

    void release() {
      if (!ptr_) return;
      DeviceRegionStats::instance().on_free(stats_, capacity_*sizeof(ElemT));
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
    DeviceRegionStats::entry_t* stats_ = nullptr;
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
    explicit BatchPool(int device_id)
    : device_id(device_id)
    , args_stats(DeviceRegionStats::instance().entry(region_stats_name<Arg>("args"), device_id))
    { }

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
      s.dev_args.set_stats(args_stats);
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
      /* acquire() only hands out a slot whose previous submission has completed,
       * so its device region may be regrown here (see FixedDeviceRegion::grow) */
      s.dev_args.grow(device_id, num_members);
      s.host_args.reserve(num_members);
    }

    std::mutex mtx;
    std::vector<std::unique_ptr<slot_t>> slots;
    DeviceRegionStats::entry_t* args_stats;
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
  /**
   * Whether a GroupedBatchPool slot is returned to its pool once the stream work of the
   * submitting device task completed, as the device scheduler observes it
   * (ttg_parsec::detail::defer_to_stream_completion: where ttg::device::wait() completes,
   * or the event PaRSEC records after the task's last submission fired), instead of
   * through an event recorded at submission and queried by acquire().
   * MRA_TTG_BATCH_SLOT_RELEASE=event selects the event.
   */
  inline bool release_slots_with_task() {
#if defined(TTG_HAVE_STREAM_COMPLETION_CALLBACKS)
    static const bool with_task = [] {
      const char* v = std::getenv("MRA_TTG_BATCH_SLOT_RELEASE");
      return v == nullptr || std::string(v) != "event";
    }();
    return with_task;
#else
    return false;
#endif
  }

  /**
   * Whether batched kernels write the small per-member results the host reads (sums of
   * squares, norms) into pinned memory of their slot instead of the members' device
   * buffers, see GroupedBatchPool::add_result: the members then wait for their stream
   * work (ttg::device::wait() without buffers) instead of transferring the results back.
   * Requires slots released with their task (release_slots_with_task()), whose release
   * copies the results to the members. MRA_TTG_RESULTS_VIA_SLOT=0 disables it.
   */
  inline bool results_via_slot() {
    static const bool via_slot = [] {
      const char* v = std::getenv("MRA_TTG_RESULTS_VIA_SLOT");
      return release_slots_with_task() && (v == nullptr || std::string(v) != "0");
    }();
    return via_slot;
  }

  /**
   * Whether submit_grouped_copy has its copy kernel read the packed arguments from the
   * pinned host buffer (mapped into the device's address space) instead of transferring
   * them with cudaMemcpyAsync first. MRA_TTG_GROUPED_COPY=memcpy selects the transfer.
   */
  inline bool grouped_copy_reads_host() {
    static const bool reads_host = [] {
      const char* v = std::getenv("MRA_TTG_GROUPED_COPY");
      return v == nullptr || std::string(v) != "memcpy";
    }();
    return reads_host;
  }

  template <typename Arg>
  struct GroupedBatchPool {
    struct slot_t {
      std::vector<Arg, DeviceAllocator<Arg>> args;                       // pinned host storage
      std::vector<size_type, DeviceAllocator<size_type>> offsets;        // pinned host storage
      // Pinned host staging for submit_grouped_copy's single H2D transfer:
      // [staged bytes (`sparsity`) | args | offsets | GroupedCopyDesc table |
      // appended source bytes]. Only ever grows (never cleared), so
      // steady-state launches neither reallocate nor re-zero it.
      std::vector<unsigned char, DeviceAllocator<unsigned char>> packed;

      /* The bytes a leader stages for a launch, at the start of `packed`, so they are
       * written once, where the transfer reads them (not copied there by
       * submit_grouped_copy). The vector operations the leaders use: resize() before
       * writing (it zero-fills the new bytes, as std::vector does, and may move the
       * storage), data(), operator[], size(), clear(). */
      class staged_bytes_t {
        std::vector<unsigned char, DeviceAllocator<unsigned char>>& m_buf;
        std::size_t m_size = 0;
      public:
        explicit staged_bytes_t(std::vector<unsigned char, DeviceAllocator<unsigned char>>& buf) : m_buf(buf) { }
        void reserve(std::size_t n) { if (m_buf.size() < n) m_buf.resize(n); }
        void resize(std::size_t n) {
          if (m_buf.size() < n) m_buf.resize(n + n/4);
          if (n > m_size) std::memset(m_buf.data() + m_size, 0, n - m_size);
          m_size = n;
        }
        void clear() { m_size = 0; }
        std::size_t size() const { return m_size; }
        bool empty() const { return m_size == 0; }
        unsigned char* data() { return m_buf.data(); }
        const unsigned char* data() const { return m_buf.data(); }
        unsigned char& operator[](std::size_t i) { return m_buf[i]; }
        const unsigned char& operator[](std::size_t i) const { return m_buf[i]; }
      };
      // Each member's per-tensor sparsity bytes (gathered from host-side
      // RangeSparsityBase via sparsity_to_bytes) and other host data a launch
      // reads: scattered to the destination tensors' device buffers by the
      // grouped copy, and readable by kernels at dev_src.
      staged_bytes_t sparsity{packed};

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

      // `packed` as the device addresses it (cudaHostGetDevicePointer), for
      // grouped_copy_reads_host(); looked up again whenever `packed` moves.
      const unsigned char* packed_host = nullptr;
      const unsigned char* packed_mapped = nullptr;

      // Device counterpart of `packed`. Grown in submit_grouped_copy while
      // the slot is checked out -- safe because acquire() only hands out a
      // slot once its previous submission has completed (see
      // FixedDeviceRegion::grow).
      FixedDeviceRegion<unsigned char> dev_packed;

      // Point into dev_packed; set by submit_grouped_copy, so they are only
      // valid for kernels launched after it.
      Arg* dev_args = nullptr;
      size_type* dev_offsets = nullptr;
      // Device copy of `sparsity` (byte i of `sparsity` is dev_src[i]), so a
      // kernel can also read bytes staged there directly instead of only
      // through extra_dsts scatters.
      const unsigned char* dev_src = nullptr;

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
      std::size_t dev_scratch_bytes = 0;  // for DeviceRegionStats

      // Pinned memory the kernels of a launch write the members' host-bound results to,
      // through `results_mapped`, and where they are copied from to the members' own host
      // memory once the launch completed (see begin_results/add_result).
      std::vector<unsigned char, DeviceAllocator<unsigned char>> results;
      unsigned char* results_host = nullptr;    // results.data() when results_mapped was looked up
      unsigned char* results_mapped = nullptr;  // results as the device addresses it
      std::size_t results_used = 0;
      struct result_copy_t {
        std::size_t offset;
        void* dst;
        std::size_t bytes;
      };
      std::vector<result_copy_t> result_copies;

#if defined(MRA_ENABLE_CUDA)
      cudaEvent_t event;
#elif defined(MRA_ENABLE_HIP)
      hipEvent_t event;
#endif
      bool event_recorded = false; // false until this slot has been submitted at least once
      // See BatchPool::slot_t::checked_out above for why this is needed. With
      // release_slots_with_task(), it stays set after mark_submitted until the
      // submitting task's stream work completed.
      bool checked_out = false;
      GroupedBatchPool* owner = nullptr;

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

    explicit GroupedBatchPool(int device_id)
    : device_id(device_id)
    , scratch_stats(DeviceRegionStats::instance().entry(region_stats_name<Arg>("scratch"), device_id))
    , packed_stats(DeviceRegionStats::instance().entry(region_stats_name<Arg>("packed"), device_id))
    { }

    // This slot's shared scratch region, of at least num_elems. Callers size
    // num_elems from kernel_grid_cap (platform.h) -- a bound for a given
    // kernel/device and K, independent of any particular launch's batch size
    // -- so it is allocated once per K; a pool shared by the graphs of a whole
    // run (shared_pool_registry) sees K grow (e.g. from one protocol to the
    // next) and then regrows it, which is safe because a checked-out slot's
    // previous submission has completed (see FixedDeviceRegion::grow).
    // No locking: the slot is already exclusively checked_out to whoever is
    // calling this (between their own acquire()/mark_submitted() pair), same
    // as slot.args/slot.offsets being mutated without extra locking elsewhere.
    template <typename ElemT>
    ElemT* ensure_scratch(slot_t& s, std::size_t num_elems) {
      if (s.dev_scratch.empty() || num_elems * sizeof(ElemT) > s.dev_scratch_bytes) {
        if (!s.dev_scratch.empty()) scratch_stats->grows.fetch_add(1, std::memory_order_relaxed);
        s.dev_scratch_bytes = num_elems * sizeof(ElemT);
      }
      s.dev_scratch.grow(device_id, num_elems * sizeof(ElemT));
      return reinterpret_cast<ElemT*>(s.dev_scratch.get());
    }

    GroupedBatchPool(const GroupedBatchPool&) = delete;
    GroupedBatchPool& operator=(const GroupedBatchPool&) = delete;

    /**
     * Prepares the slot's results region (see results_via_slot()) for at most max_bytes of
     * results added with add_result, before the launch is built. Grows it only while the slot
     * is checked out, before any pointer into it was handed out.
     */
    void begin_results(slot_t& s, std::size_t max_bytes) {
      s.results_used = 0;
      s.result_copies.clear();
      if (s.results.size() < max_bytes) {
        s.results.resize(max_bytes + max_bytes/4);
      }
      if (s.results.empty()) return;
      if (s.results_host != s.results.data()) {
        void* mapped = nullptr;
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaHostGetDevicePointer(&mapped, s.results.data(), 0), "cudaHostGetDevicePointer");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipHostGetDevicePointer(&mapped, s.results.data(), 0), "hipHostGetDevicePointer");
#endif
        s.results_host = s.results.data();
        s.results_mapped = static_cast<unsigned char*>(mapped);
      }
    }

    /**
     * Space for n values of a member's results, which the launch's kernels write through the
     * returned (device-addressable) pointer and which are copied to host_dst (host memory of
     * the member, n values; nullptr makes the space scratch of the launch) once the launch completed, before any member of the batch is
     * resumed (see mark_submitted). Zero-filled if zero_fill. Requires results_via_slot() and
     * begin_results with enough room.
     */
    template <typename ElemT>
    ElemT* add_result(slot_t& s, ElemT* host_dst, std::size_t n, bool zero_fill) {
      constexpr std::size_t align = 16;
      const std::size_t offset = (s.results_used + align - 1) / align * align;
      const std::size_t bytes = n * sizeof(ElemT);
      if (offset + bytes > s.results.size()) {
        throw std::logic_error("GroupedBatchPool::add_result: results region too small, see begin_results");
      }
      s.results_used = offset + bytes;
      if (zero_fill) std::memset(s.results.data() + offset, 0, bytes);
      /* without a destination, the region is scratch for the launch only */
      if (host_dst != nullptr && bytes > 0) s.result_copies.push_back({offset, host_dst, bytes});
      return reinterpret_cast<ElemT*>(s.results_mapped + offset);
    }

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
      s.owner = this;
      s.dev_scratch.set_stats(scratch_stats);
      s.dev_packed.set_stats(packed_stats);
      ensure_capacity(s, num_args, num_offsets, num_sparsity);
      s.checked_out = true;
      return s;
    }

    /* Call right after the H2D copy + kernel launch(es) have been issued on
     * `stream`. Releases the slot (clears checked_out) so a future
     * acquire() may hand it out again once its device event completes --
     * or, with release_slots_with_task(), once the device scheduler observed
     * that the work the calling task submitted to its stream (this slot's
     * copy and launches) completed: no event is recorded or queried here. */
    void mark_submitted(slot_t& s, ttg::device::Stream stream) {
#if defined(TTG_HAVE_STREAM_COMPLETION_CALLBACKS)
      if (release_slots_with_task() &&
          ttg_parsec::detail::defer_to_stream_completion(&GroupedBatchPool::release_slot, &s)) {
        return;
      }
#endif
      std::lock_guard<std::mutex> lock(mtx);
#if defined(MRA_ENABLE_CUDA)
      check_cuda_rt(cudaEventRecord(s.event, stream), "cudaEventRecord");
#elif defined(MRA_ENABLE_HIP)
      check_hip_rt(hipEventRecord(s.event, stream), "hipEventRecord");
#endif
      s.event_recorded = true;
      if (!s.result_copies.empty()) {
        /* no completion callback left for the results (rare): the members resume once their
         * stream work completed and read the results then, so they must be in place now */
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaEventSynchronize(s.event), "cudaEventSynchronize");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipEventSynchronize(s.event), "hipEventSynchronize");
#endif
        copy_results(s);
      }
      s.checked_out = false;
    }

    int device_id;

   private:
    /* defer_to_stream_completion callback: the slot's submission completed */
    static void copy_results(slot_t& s) {
      for (const auto& c : s.result_copies) std::memcpy(c.dst, s.results.data() + c.offset, c.bytes);
      s.result_copies.clear();
    }

    static void release_slot(void* p) {
      auto& s = *static_cast<slot_t*>(p);
      /* runs before any member of the batch resumes (see ttg_parsec::detail::defer_to_stream_completion) */
      copy_results(s);
      std::lock_guard<std::mutex> lock(s.owner->mtx);
      s.event_recorded = false;
      s.checked_out = false;
    }

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
    DeviceRegionStats::entry_t* scratch_stats;
    DeviceRegionStats::entry_t* packed_stats;
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
      GroupedBatchPool<Arg>* p = e.pool.load(std::memory_order_acquire);
      if (p == nullptr) {
        std::lock_guard<std::mutex> lock(e.mtx);
        p = e.pool.load(std::memory_order_relaxed);
        if (p == nullptr) {
          p = new GroupedBatchPool<Arg>(device_id);
          e.pool.store(p, std::memory_order_release);
        }
      }
      return *p;
    }

    /**
     * Frees every device's pool, as BatchPoolRegistry::release (for a registry
     * that outlives the TTs using it, see shared_pool_registry). No task may
     * use the registry concurrently; a later get() creates the pool anew.
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
      std::atomic<GroupedBatchPool<Arg>*> pool{nullptr};
      ~entry_t() { delete pool.load(std::memory_order_relaxed); }
    };
    std::vector<entry_t> entries;
    int max_batch_size;
  };

  /**
   * The process-wide GroupedBatchPoolRegistry of a kernel family (Arg) and
   * batch size, shared by every TT (and BatchGroup) of that family instead of
   * one registry per TT. The pools' slots -- each with its CUDA/HIP event,
   * device regions and pinned host staging -- thereby outlive the graph that
   * created them and are reused by the next one: a program that executes many
   * short graphs (e.g., one per tile of a computation) no longer creates and
   * destroys them (event synchronization and destruction, zone and pinned
   * frees) for each graph. Sharing is safe as the pools already are shared by
   * concurrent tasks: acquire() only hands out a slot that is not checked out
   * and whose previous submission has completed. The pools are freed in
   * mra::finalize(), while PaRSEC's devices still exist.
   */
  template <typename Arg>
  std::shared_ptr<GroupedBatchPoolRegistry<Arg>> shared_pool_registry(int max_batch_size) {
    using registry_t = GroupedBatchPoolRegistry<Arg>;
    static std::mutex mtx;
    static std::map<int, std::shared_ptr<registry_t>> registries;
    std::lock_guard<std::mutex> lock(mtx);
    auto& r = registries[max_batch_size];
    if (!r) {
      r = std::make_shared<registry_t>(ttg::device::num_devices(), max_batch_size);
      at_finalize([max_batch_size] {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = registries.find(max_batch_size);
        if (it == registries.end()) return;
        it->second->release();   // TTs that still refer to the registry find it empty
        registries.erase(it);
      });
    }
    return r;
  }

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

  /**
   * Performs the num_descs copies described by descs (in device memory), reading
   * from the device buffer src, on stream. Defined in src/kernels/batch_pool.cc so
   * that translation units not compiled by nvcc/hipcc can use submit_grouped_copy.
   */
  void launch_grouped_copy_scatter(const GroupedCopyDesc* descs, std::uint32_t num_descs,
                                   const unsigned char* src, ttg::device::Stream stream);

  /**
   * Copies `total` bytes from host (pinned memory mapped into the device's address space,
   * `host` its device address) to dev, and performs the num_descs copies described at
   * host + descs_off, reading their sources from host + src_off. One kernel on stream, no
   * transfer before it. Defined in src/kernels/batch_pool.cc.
   */
  void launch_grouped_copy_from_host(const unsigned char* host, unsigned char* dev, std::size_t total,
                                     std::size_t descs_off, std::uint32_t num_descs, std::size_t src_off,
                                     ttg::device::Stream stream);

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
    auto align_up = [](std::size_t n) { return (n + 15) & ~std::size_t(15); };
    const std::size_t num_extra = slot.extra_dsts.size();
    // The staged bytes are already in place at the start of `packed`. Sources within
    // them are named by their offset; `packed` may move when it grows below, so
    // they are told apart by address before that (compared as integers, the old
    // storage is never read).
    const std::size_t staged = slot.sparsity.size();
    const std::uintptr_t staged_begin = reinterpret_cast<std::uintptr_t>(slot.packed.data());
    const std::uintptr_t staged_end = staged_begin + staged;
    auto in_staged = [&](const void* p, std::size_t size) {
      const auto a = reinterpret_cast<std::uintptr_t>(p);
      return a >= staged_begin && a + size <= staged_end;
    };

    // Sources outside the staged bytes are appended after the descriptors.
    std::size_t num_descs = 0;
    std::size_t num_appended = 0;
    for (std::size_t k = 0; k < num_extra; ++k) {
      if (slot.extra_sizes[k] == 0) continue;
      ++num_descs;
      if (slot.extra_srcs[k] != nullptr && !in_staged(slot.extra_srcs[k], slot.extra_sizes[k])) {
        num_appended += slot.extra_sizes[k];
      }
    }

    const std::size_t args_bytes = num_args*sizeof(Arg);
    const std::size_t args_off = align_up(staged);
    const std::size_t offsets_off = align_up(args_off + args_bytes);
    const std::size_t descs_off = align_up(offsets_off + num_offsets*sizeof(size_type));
    const std::size_t appended_off = descs_off + num_descs*sizeof(GroupedCopyDesc);
    const std::size_t total = appended_off + num_appended;
    if (total >= GroupedCopyDesc::zero_fill) {
      throw std::runtime_error("submit_grouped_copy: source bytes exceed 32-bit descriptor offsets");
    }

    if (slot.packed.size() < total) slot.packed.resize(total + total/4); // keeps the staged bytes
    slot.dev_packed.grow(device_id, slot.packed.size());

    unsigned char* h = slot.packed.data();
    std::memcpy(h + args_off, slot.args.data(), args_bytes);
    std::memcpy(h + offsets_off, slot.offsets.data(), num_offsets*sizeof(size_type));
    auto descs = reinterpret_cast<GroupedCopyDesc*>(h + descs_off);
    std::size_t d = 0;
    std::size_t appended = appended_off;
    for (std::size_t k = 0; k < num_extra; ++k) {
      const std::size_t size = slot.extra_sizes[k];
      if (size == 0) continue;
      auto src = static_cast<const unsigned char*>(slot.extra_srcs[k]);
      std::uint32_t offset;
      if (src == nullptr) {
        offset = GroupedCopyDesc::zero_fill;
      } else if (in_staged(src, size)) {
        offset = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(src) - staged_begin);
      } else {
        std::memcpy(h + appended, src, size);
        offset = static_cast<std::uint32_t>(appended);
        appended += size;
      }
      descs[d++] = GroupedCopyDesc{slot.extra_dsts[k], offset, static_cast<std::uint32_t>(size)};
    }

    unsigned char* dev = slot.dev_packed.get();
    slot.dev_args = reinterpret_cast<Arg*>(dev + args_off);
    slot.dev_offsets = reinterpret_cast<size_type*>(dev + offsets_off);
    slot.dev_src = dev; // the staged bytes, and the base of the descriptors' offsets
    if (total == 0) return;
    if (grouped_copy_reads_host()) {
      /* one kernel reads everything from the pinned buffer: no cudaMemcpyAsync */
      if (slot.packed_host != h) {
        void* mapped = nullptr;
#if defined(MRA_ENABLE_CUDA)
        check_cuda_rt(cudaHostGetDevicePointer(&mapped, h, 0), "cudaHostGetDevicePointer");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipHostGetDevicePointer(&mapped, h, 0), "hipHostGetDevicePointer");
#endif
        slot.packed_host = h;
        slot.packed_mapped = static_cast<const unsigned char*>(mapped);
      }
      launch_grouped_copy_from_host(slot.packed_mapped, dev, total, descs_off,
                                    static_cast<std::uint32_t>(num_descs), 0, stream);
      return;
    }
#if defined(MRA_ENABLE_CUDA)
    check_cuda_rt(cudaMemcpyAsync(dev, h, total, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
#elif defined(MRA_ENABLE_HIP)
    check_hip_rt(hipMemcpyAsync(dev, h, total, hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
#endif
    if (num_descs > 0) {
      launch_grouped_copy_scatter(reinterpret_cast<const GroupedCopyDesc*>(dev + descs_off),
                                  static_cast<std::uint32_t>(num_descs), dev, stream);
    }
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

  /**
   * Warp-cooperative find_member_for_pos (find_nth_pred_warp, so all
   * existing lanes of the calling warp must call it, and all get the
   * result): the member is the first m with pos < offsets[m+1]. Lets every
   * warp look up its member itself instead of waiting on a team lead.
   */
  SCOPE size_type find_member_for_pos_warp(const size_type* offsets, size_type num_members,
                                            size_type pos, size_type* local_pos_out) {
    const size_type m = find_nth_pred_warp(num_members, 0, offsets,
                                           [pos](const size_type* off, size_type i) { return pos < off[i + 1]; });
    *local_pos_out = pos - offsets[m];
    return m;
  }

} // namespace mra::detail

#endif // !MRA_ENABLE_HOST

#endif // MRA_MISC_DEVICE_BATCH_POOL_H
