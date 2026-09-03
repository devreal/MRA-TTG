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
        check_cuda_rt(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
#elif defined(MRA_ENABLE_HIP)
        check_hip_rt(hipEventCreateWithFlags(&event, hipEventDisableTiming), "hipEventCreate");
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
   * Bundles what every batched-kernel leader (compress/reconstruct/
   * convolution) needs into a SINGLE slot with a SINGLE CUDA/HIP event,
   * instead of several independently-tracked BatchPool<Arg> instances each
   * recording/querying their own event. This is what lets a batch leader
   * submission use one acquire()/mark_submitted() pair (one event to record
   * and later query) rather than several.
   *
   * Two co-allocated device-backed regions: the per-member Arg array and the
   * (num_members+1)-entry member-offsets array (always size_type). Plus one
   * host-only pinned staging region, `sparsity` (raw bytes, not
   * detail::SparsityState, so this header does not need to know about
   * SparsityState -- defined in mra/tensor/sparsity.h, not included here,
   * see the ttg.h/size_type note at the top of this file for why), used to
   * gather each member's sparsity bytes from its host-side
   * RangeSparsityBase before they're copied straight to their own final
   * destination tensors (see submit_grouped_copy's comment) -- no device-side
   * counterpart of its own anymore, since those destinations already exist
   * (they're each member's own, separately-allocated tensor buffer).
   *
   * `extra_dsts`/`extra_srcs`/`extra_sizes` name every one of those
   * point-to-point destination copies (rebuilt fresh every launch, unlike
   * the other regions -- see their own comment on slot_t). All regions --
   * args, offsets, and every extra copy -- are transferred host->device with
   * a single call (see submit_grouped_copy below): on CUDA >= 12.8 that is
   * one cudaMemcpyBatchAsync describing every copy; otherwise (older CUDA
   * toolkits, or HIP, which has no batched-copy API) it falls back to that
   * many ordinary async copies issued back-to-back on the same stream --
   * either way only ONE event is recorded for the whole slot, so the
   * one-slot/one-event bookkeeping win holds on every platform even where
   * the copy itself isn't literally batched.
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
      // Source-only now -- see submit_grouped_copy's comment: bytes are
      // copied directly from here to each destination tensor's own device
      // buffer (no intermediate device-side sparsity buffer anymore).
      std::vector<unsigned char, DeviceAllocator<unsigned char>> sparsity;

      // Per-launch list of additional point-to-point H2D copies folded into
      // the same submit_grouped_copy call as args/offsets -- populated fresh
      // by the caller's submit_*_batch_leader before each submit_grouped_copy
      // (cleared and rebuilt every launch, not persisted/grown like
      // args/offsets/sparsity). Typically one entry per member per
      // destination tensor (scattering `sparsity` above into each member's
      // own, separately-allocated sparsity bitfield -- see
      // submit_grouped_copy's comment for why this replaced a dedicated
      // scatter kernel), plus, for families that need it (e.g. convolution's
      // resnorms), entries sourced from a shared zero buffer (see
      // zero_source_pool_registry) instead of `sparsity`.
      std::vector<void*> extra_dsts;
      std::vector<const void*> extra_srcs;
      std::vector<std::size_t> extra_sizes;

      Arg* dev_args = nullptr;
      size_type* dev_offsets = nullptr;
      std::size_t args_capacity = 0;
      std::size_t offsets_capacity = 0;

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
#if defined(MRA_ENABLE_CUDA)
        if (dev_args) cudaFree(dev_args);
        if (dev_offsets) cudaFree(dev_offsets);
        cudaEventDestroy(event);
#elif defined(MRA_ENABLE_HIP)
        if (dev_args) hipFree(dev_args);
        if (dev_offsets) hipFree(dev_offsets);
        hipEventDestroy(event);
#endif
      }
    };

    explicit GroupedBatchPool(int device_id) : device_id(device_id) { }

    GroupedBatchPool(const GroupedBatchPool&) = delete;
    GroupedBatchPool& operator=(const GroupedBatchPool&) = delete;

    /* Never blocks -- same reuse policy as BatchPool::acquire (one non-
     * blocking event query covering both regions at once). Each region is
     * grown independently (only if its own requested capacity exceeds what
     * is already allocated), so growing one region never reallocates the
     * other. num_sparsity only resizes the pinned host staging vector (no
     * device-side counterpart anymore -- see slot_t's comment); extra_dsts/
     * extra_srcs/extra_sizes are rebuilt fresh every launch by the caller,
     * not sized here. */
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
   * Lazily allocates, per device, a persistent pinned host buffer of zero
   * bytes -- a reusable memcpy SOURCE for zero-initializing arbitrary device
   * destinations from within a single batched submit_grouped_copy call (e.g.
   * convolution's resnorms_view, for positions its compute kernel won't
   * visit). The same source pointer can be reused across many different
   * destination entries in one cudaMemcpyBatchAsync call -- there is no rule
   * against repeating a source -- so ONE allocation, grown (and re-zeroed)
   * only when a caller asks for more bytes than it currently holds, suffices
   * regardless of how many destinations end up using it. Indexed per device
   * purely for consistency with BatchPoolRegistry/GroupedBatchPoolRegistry
   * above; the content itself (all zero) doesn't actually vary by device.
   */
  struct ZeroSourcePool {
    const void* get(std::size_t num_bytes) {
      std::lock_guard<std::mutex> lock(mtx);
      if (num_bytes > buf.size()) {
        buf.assign(num_bytes, static_cast<unsigned char>(0));
      }
      return buf.data();
    }

   private:
    std::mutex mtx;
    std::vector<unsigned char, DeviceAllocator<unsigned char>> buf; // pinned host storage, always zero
  };

  struct ZeroSourcePoolRegistry {
    explicit ZeroSourcePoolRegistry(int num_devices) : entries(num_devices) { }

    ZeroSourcePool& get(int device_id) {
      auto& e = entries[device_id];
      std::call_once(e.once, [&]{ e.pool = std::make_unique<ZeroSourcePool>(); });
      return *e.pool;
    }

   private:
    struct entry_t {
      std::once_flag once;
      std::unique_ptr<ZeroSourcePool> pool;
    };
    std::vector<entry_t> entries;
  };

  inline ZeroSourcePoolRegistry& zero_source_pool_registry() {
    static ZeroSourcePoolRegistry registry(ttg::device::num_devices());
    return registry;
  }

  /**
   * Issues the host->device transfer for a GroupedBatchPool slot's args and
   * offsets regions, PLUS every point-to-point copy queued in
   * slot.extra_dsts/extra_srcs/extra_sizes, as ONE call. num_args/
   * num_offsets are the LOGICAL (used) sizes for this launch, which may be
   * smaller than the slot's allocated capacity (see
   * GroupedBatchPool::ensure_capacity).
   *
   * The "extra" copies are what used to require a dedicated scatter kernel
   * per kernel family (compress_scatter_sparsity_kernel,
   * reconstruct_scatter_sparsity_kernel, convolution_scatter_sparsity_kernel):
   * each one copies straight from a slice of slot.sparsity (this batch's
   * pinned, host-side-aggregated sparsity bytes for one member's one
   * destination tensor) DIRECTLY to that tensor's own device buffer --
   * slot.extra_dsts[k] is the destination tensor's buffer().device_ptr_on(device),
   * which coincides exactly with where its inline sparsity bitfield starts
   * (byte offset 0 -- see SparseArrayBase's layout), so no on-device
   * scatter/set_state loop is needed at all. A family with an additional
   * duty (e.g. convolution zero-filling resnorms_view for positions its
   * compute kernel won't visit) can queue more entries sourced from
   * zero_source_pool_registry instead of slot.sparsity -- same mechanism,
   * no extra kernel either way.
   *
   * On CUDA 12.8+, this is a single cudaMemcpyBatchAsync call describing all
   * `2 + slot.extra_dsts.size()` copies (one shared cudaMemcpyAttributes
   * entry, since all of them are plain pinned-host -> device copies with the
   * same access-order/location semantics as an ordinary cudaMemcpyAsync). On
   * older CUDA toolkits, or on HIP (no batched-copy API exists there), this
   * falls back to that many ordinary async copies on the same stream; the
   * caller still only records one event for the slot afterwards (see
   * GroupedBatchPool::mark_submitted), so the slot/event count stays at one
   * regardless of which path is taken.
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
                            std::size_t num_args, std::size_t num_offsets,
                            int device_id, ttg::device::Stream stream)
  {
    const std::size_t num_extra = slot.extra_dsts.size();
    const std::size_t total = 2 + num_extra;
#if defined(MRA_ENABLE_CUDA) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    std::vector<const void*> dsts;
    std::vector<const void*> srcs;
    std::vector<std::size_t> sizes;
    dsts.reserve(total);
    srcs.reserve(total);
    sizes.reserve(total);
    dsts.push_back(static_cast<void*>(slot.dev_args));
    srcs.push_back(static_cast<void*>(slot.args.data()));
    sizes.push_back(num_args*sizeof(Arg));
    dsts.push_back(static_cast<void*>(slot.dev_offsets));
    srcs.push_back(static_cast<void*>(slot.offsets.data()));
    sizes.push_back(num_offsets*sizeof(size_type));
    for (std::size_t k = 0; k < num_extra; ++k) {
      dsts.push_back(slot.extra_dsts[k]);
      srcs.push_back(slot.extra_srcs[k]);
      sizes.push_back(slot.extra_sizes[k]);
    }
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint = cudaMemLocation{cudaMemLocationTypeHost, 0};
    attrs.dstLocHint = cudaMemLocation{cudaMemLocationTypeDevice, device_id};
    attrs.flags = cudaMemcpyFlagDefault;
    std::size_t attrsIdxs[1] = { 0 }; // one attribute set applies to all entries
#if CUDART_VERSION >= 13000
    check_cuda_rt(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), total, &attrs, attrsIdxs, 1, stream),
                  "cudaMemcpyBatchAsync");
#else
    std::size_t failIdx = 0;
    check_cuda_rt(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), total, &attrs, attrsIdxs, 1, &failIdx, stream),
                  "cudaMemcpyBatchAsync");
#endif
#elif defined(MRA_ENABLE_CUDA)
    // Toolkit predates cudaMemcpyBatchAsync (added in CUDA 12.8).
    check_cuda_rt(cudaMemcpyAsync(slot.dev_args, slot.args.data(), num_args*sizeof(Arg),
                                  cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    check_cuda_rt(cudaMemcpyAsync(slot.dev_offsets, slot.offsets.data(), num_offsets*sizeof(size_type),
                                  cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    for (std::size_t k = 0; k < num_extra; ++k) {
      check_cuda_rt(cudaMemcpyAsync(slot.extra_dsts[k], slot.extra_srcs[k], slot.extra_sizes[k],
                                    cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    }
#elif defined(MRA_ENABLE_HIP)
    // No HIP equivalent of cudaMemcpyBatchAsync exists yet.
    check_hip_rt(hipMemcpyAsync(slot.dev_args, slot.args.data(), num_args*sizeof(Arg),
                                hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    check_hip_rt(hipMemcpyAsync(slot.dev_offsets, slot.offsets.data(), num_offsets*sizeof(size_type),
                                hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    for (std::size_t k = 0; k < num_extra; ++k) {
      check_hip_rt(hipMemcpyAsync(slot.extra_dsts[k], slot.extra_srcs[k], slot.extra_sizes[k],
                                  hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    }
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
