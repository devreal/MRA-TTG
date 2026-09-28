#ifndef CONV_MAD_H
#define CONV_MAD_H

#include <atomic>
#include <memory>
#include <array>
#include <utility>

#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/world/worldhashmap.h>
#include <madness/mra/operator.h>
#include <madness/mra/convolution1d.h>
#include "mra/misc/types.h"
#include "mra/misc/key.h"
#ifndef MRA_ENABLE_HOST
#include "mra/kernels/generate_op1d.h"
#endif // !MRA_ENABLE_HOST

namespace mra {

  enum class NormId {
    Rnorm = 0,
    Snorm,
    Rnormf,
    Snormf,
    NSnormf,
    Fac,
    MUnorm,
    Opnorm, // overall operator norm
    Rank,   // stored rank of the operator
    Count
  };

  /// Max |translation| (in each dimension) covered by the level-wide 1D
  /// operator generation table -- see GaussianConvolutionOperator's header
  /// comment and detail::LevelOp1DTable. A translation outside this range is
  /// a hard error (GaussianConvolutionOperator::try_get_op() throws), not a
  /// fallback -- screener_tt/accumulate_tt only ever walk MADNESS's own
  /// shell-ordered, quickly-decaying displacement list, so this is not
  /// expected to trigger in practice.
  inline constexpr Translation MRA_OP1D_MAX_DISPLACEMENT = 4;
  inline constexpr size_type MRA_OP1D_NUM_DISPLACEMENTS = 2 * MRA_OP1D_MAX_DISPLACEMENT + 1;

  template <typename T>
  struct ConvolutionData1D {

    // 4D: count x rank x [R|S] x 2D operator matrix
    // count should either be 1 or the number of functions to which the operators are applied
    using tensor_type = DenseTensor<T, 4>;

    /**
     * We store R and S in separate tensors because they have different dimensions (2K and K).
     */
    tensor_type R, S;

    ConvolutionData1D() : R(), S(){}
    ConvolutionData1D(size_type count, size_type rank, size_type K)
    : R(std::array{count, rank, 2*K, 2*K}, ttg::scope::SyncIn)
    , S(std::array{count, rank, K, K}, ttg::scope::SyncIn)
    { }
    ConvolutionData1D(tensor_type&& R_,
                      tensor_type&& S_)
    : R(std::move(R_))
    , S(std::move(S_))
    { }
    ConvolutionData1D(const ConvolutionData1D&) = default;
    ConvolutionData1D(ConvolutionData1D&&) = default;
    ~ConvolutionData1D() = default;
  };

  namespace detail {

#if defined(__cpp_lib_atomic_shared_ptr)
    /** True atomic<shared_ptr<T>> (C++20, P0718), used whenever the standard
     * library actually implements it. */
    template <typename SharedPtrT>
    using atomic_shared_ptr = std::atomic<SharedPtrT>;
#else
    /**
     * Fallback for standard libraries that advertise C++20 but do not (yet)
     * implement std::atomic<std::shared_ptr<T>> -- e.g. the libc++ shipped
     * with the Clang on this machine. Built on the free-standing
     * std::atomic_load/store/compare_exchange overloads for shared_ptr,
     * which every standard library has provided since C++11: they were
     * deprecated in C++20 in favor of the type above, but remain available
     * and are still the only portable option where the new API is missing.
     * Same interface as the subset of std::atomic<shared_ptr> used below
     * (default-construct, load/store, compare_exchange_strong), so callers
     * don't need to know which one they got.
     */
    template <typename SharedPtrT>
    class atomic_shared_ptr {
    public:
      atomic_shared_ptr() noexcept = default;
      atomic_shared_ptr(SharedPtrT desired) noexcept : m_ptr(std::move(desired)) { }

      atomic_shared_ptr(const atomic_shared_ptr&) = delete;
      atomic_shared_ptr& operator=(const atomic_shared_ptr&) = delete;

      SharedPtrT load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return std::atomic_load_explicit(&m_ptr, order);
      }

      void store(SharedPtrT desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
        std::atomic_store_explicit(&m_ptr, std::move(desired), order);
      }

      operator SharedPtrT() const noexcept { return load(); }

      bool compare_exchange_strong(SharedPtrT& expected, SharedPtrT desired,
                                    std::memory_order success,
                                    std::memory_order failure) noexcept {
        return std::atomic_compare_exchange_strong_explicit(&m_ptr, &expected, std::move(desired),
                                                             success, failure);
      }

    private:
      SharedPtrT m_ptr;
    };
#endif // __cpp_lib_atomic_shared_ptr

    /// Key for the per-dimension 1D convolution operator cache. Unlike the aggregate
    /// cache (keyed by Key<NDIM>, which already spans all dimensions), lookups here must
    /// also distinguish which dimension the entry belongs to.
    struct Op1DKey {
      Level n = 0;
      Dimension d = 0;
      Translation l = 0;

      Op1DKey() = default;
      Op1DKey(Level n, Dimension d, Translation l) : n(n), d(d), l(l) { }

      bool operator==(const Op1DKey& other) const {
        return n == other.n && d == other.d && l == other.l;
      }

      /// Combines n, d, l into a single hash value; follows the same style as Key<NDIM>::hash().
      HashValue hash() const {
        HashValue h = static_cast<HashValue>(static_cast<uint32_t>(l));
        h = (h << 16) ^ static_cast<HashValue>(d);
        h = (h << 16) ^ static_cast<HashValue>(static_cast<uint16_t>(n));
        return h;
      }
    };

    /// Key for the level-wide 1D operator generation gate -- see
    /// GaussianConvolutionOperator's header comment and try_get_op(). Exactly
    /// one caller across an entire level (every dimension, every translation
    /// in the bounded range) becomes responsible for generating the whole
    /// detail::LevelOp1DTable for that level in a single kernel launch.
    struct LevelKey {
      Level n = 0;

      LevelKey() = default;
      explicit LevelKey(Level n) : n(n) { }

      bool operator==(const LevelKey& other) const { return n == other.n; }

      HashValue hash() const { return static_cast<HashValue>(static_cast<uint16_t>(n)); }
    };

    /**
     * Hash functor bridging our own KeyT::hash() (returning HashValue, i.e. uint64_t) to
     * madness::ConcurrentHashMap's expected hashfunT (returning madness::hashT, i.e.
     * std::size_t). We cannot rely on madness's generic hash_value()/t.hash() plumbing
     * here: on platforms where size_t and uint64_t are distinct types of the same width
     * (e.g. macOS/LP64, where hashT is `unsigned long` but HashValue is `unsigned long
     * long`), its SFINAE check requires an exact type match and fails to compile.
     */
    template <typename KeyT>
    struct HashFunctor {
      madness::hashT operator()(const KeyT& key) const {
        return static_cast<madness::hashT>(key.hash());
      }
    };

    enum class CacheEntryState : uint8_t { Empty = 0, Requested = 1, Ready = 2 };

    /**
     * A concurrent, memoizing cache keyed by KeyT that computes each distinct value at
     * most once, even when many tasks request the same key concurrently.
     *
     * The first task to request a given key becomes its "owner" (claim() marks the entry
     * Requested/processing) and is responsible for computing the value and publish()-ing
     * it. Any other task requesting the same key while it is still Requested does not
     * redo the work; it calls acquire() to block (with progressive backoff, via
     * madness::MutexWaiter) until the owner publishes the result. This lets concurrent
     * tasks that need several related keys (e.g. one per dimension) split up the work
     * instead of every task recomputing everything itself: a task can claim() several
     * keys up front, compute the ones it owns, and only acquire()-wait on the rest.
     *
     * Built on madness::ConcurrentHashMap, which requires the mapped value to be
     * copy-constructible (entries are stored as copied std::pair<const K,V>). Since the
     * actual payload must be updated in place by whichever task ends up computing it, we
     * store a copyable std::shared_ptr<Cell> in the map and do the claim/publish/acquire
     * synchronization lock-free on the Cell itself via atomic_shared_ptr -- the map is
     * only ever used for the cheap "find-or-create the Cell for this key" step.
     */
    template <typename KeyT, typename ValueT>
    class SharedComputeCache {
    public:
      using pointer_type = std::shared_ptr<const ValueT>;

    private:
      struct Cell {
        std::atomic<CacheEntryState> state{CacheEntryState::Empty};
        atomic_shared_ptr<pointer_type> value;
      };
      using cellptr_type = std::shared_ptr<Cell>;
      using map_type = madness::ConcurrentHashMap<KeyT, cellptr_type, HashFunctor<KeyT>>;

      mutable map_type m_map;

      cellptr_type get_or_create_cell(const KeyT& key) const {
        {
          // fast path: entry already exists, only need a (shared) read lock
          typename map_type::const_accessor cacc;
          if (m_map.find(cacc, key)) {
            return cacc->second;
          }
        }
        // slow path: entry may not exist yet; insert() blocks until it can exclusively
        // create-or-observe the entry, so exactly one caller constructs the Cell.
        typename map_type::accessor acc;
        if (m_map.insert(acc, key)) {
          acc->second = std::make_shared<Cell>();
        }
        return acc->second;
      }

    public:
      /// Handle returned by claim(); pass to publish() (if owner) or acquire() (otherwise).
      struct Ticket {
        cellptr_type cell;
        bool owner = false;
        bool ready = false;
      };

      /**
       * Claims responsibility for computing the value for `key`. If `Ticket::owner` is
       * true, the caller must compute the value and call publish(). Otherwise some other
       * task already claimed (or already finished) this key; call acquire() to obtain the
       * result once it is ready.
       */
      Ticket claim(const KeyT& key) const {
        cellptr_type cell = get_or_create_cell(key);
        CacheEntryState expected = CacheEntryState::Empty;
        bool owner = cell->state.compare_exchange_strong(expected, CacheEntryState::Requested,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_acquire);
        return Ticket{std::move(cell), owner, expected == CacheEntryState::Ready};
      }

      /// Publishes the computed value for a ticket obtained via claim() with owner == true.
      void publish(const Ticket& ticket, pointer_type data) const {
        ticket.cell->value.store(std::move(data), std::memory_order_release);
        ticket.cell->state.store(CacheEntryState::Ready, std::memory_order_release);
      }

      /// Blocks (with progressive backoff) until the value is ready, then returns it.
      pointer_type acquire(const Ticket& ticket) const {
        madness::MutexWaiter waiter;
        while (ticket.cell->state.load(std::memory_order_acquire) != CacheEntryState::Ready) {
          waiter.wait();
        }
        return ticket.cell->value.load(std::memory_order_acquire);
      }
    };

    enum class Op1DCellState : uint8_t { Empty = 0, Initializing = 1, InProgress = 2, Ready = 3 };

    /**
     * Concurrent cache + work-splitting scheduler for per-(level, dimension, translation)
     * 1D convolution operator tensors.
     *
     * Building one such tensor loops over every (function, separated-term) pair, calling
     * into MADNESS to assemble that pair's block of the R/S tensors. That loop can be long
     * (separated-expansion rank is often tens to hundreds of terms), and a plain
     * claim/compute/publish scheme (as used by SharedComputeCache) has exactly one task
     * run the whole loop while every other task waiting on the same key sits in a pure
     * spin-wait. Each (c,i) pair writes to a disjoint slice of the tensor, and MADNESS's
     * own per-term operator accessors/caches (ConvolutionND::getop, Convolution1D's
     * SimpleCache-based nonstandard()) are self-contained and already safe to call
     * concurrently from different (c,i), so any task that shows up for the same key can
     * steal whichever (c,i) pairs are still unclaimed instead of only ever waiting.
     *
     * Protocol per key:
     *   claim()      -- get-or-create the entry; `creator` tells the caller whether it
     *                    must call init().
     *   init()       -- (creator only) allocate the shared tensor and the flat list of
     *                    (c,i) work items, then open the entry up for contributions.
     *   contribute() -- (anyone holding a ticket) grab and compute whatever unclaimed
     *                    items remain; returns as soon as none are left -- it does not
     *                    block waiting on other tasks' in-flight items.
     *   acquire()    -- (anyone) blocks until every item is done, then returns the
     *                    finished, immutable tensor.
     */
    template <typename T>
    class Op1DCache {
    public:
      using pointer_type = std::shared_ptr<const ConvolutionData1D<T>>;

      struct WorkItem {
        size_type c;
        size_type i;
      };

    private:
      struct Cell {
        std::atomic<Op1DCellState> state{Op1DCellState::Empty};
        // Written once by the creator, before the InProgress/Ready release-store below;
        // every other access happens-after observing that store (see contribute()), so
        // these need no synchronization of their own.
        std::shared_ptr<ConvolutionData1D<T>> tensor;
        std::vector<WorkItem> items;
        // Per-item claim flags and the outstanding-item counter *do* need to be atomic:
        // many contributors race on them concurrently.
        std::vector<std::atomic<bool>> claimed;
        std::atomic<size_type> remaining{0};
        atomic_shared_ptr<pointer_type> value;
      };
      using cellptr_type = std::shared_ptr<Cell>;
      using map_type = madness::ConcurrentHashMap<Op1DKey, cellptr_type, HashFunctor<Op1DKey>>;

      mutable map_type m_map;

      cellptr_type get_or_create_cell(const Op1DKey& key) const {
        {
          typename map_type::const_accessor cacc;
          if (m_map.find(cacc, key)) {
            return cacc->second;
          }
        }
        typename map_type::accessor acc;
        if (m_map.insert(acc, key)) {
          acc->second = std::make_shared<Cell>();
        }
        return acc->second;
      }

    public:
      /// Handle returned by claim(); pass to init()/contribute()/acquire().
      struct Ticket {
        cellptr_type cell;
        bool creator = false;
        bool ready = false;
      };

      /// Claims (get-or-creates) the entry for `key`. `creator == true` means this call
      /// must follow up with init() before anyone can contribute() or acquire().
      Ticket claim(const Op1DKey& key) const {
        cellptr_type cell = get_or_create_cell(key);
        Op1DCellState expected = Op1DCellState::Empty;
        bool creator = cell->state.compare_exchange_strong(expected, Op1DCellState::Initializing,
                                                             std::memory_order_acq_rel,
                                                             std::memory_order_acquire);
        return Ticket{std::move(cell), creator, expected == Op1DCellState::Ready};
      }

      /// Creator-only: installs the shared tensor and its flat list of (c,i) work items,
      /// then opens the entry for contributions (or marks it Ready immediately if there
      /// happen to be no items).
      void init(const Ticket& ticket, std::shared_ptr<ConvolutionData1D<T>> tensor,
                std::vector<WorkItem> items) const {
        auto& cell = *ticket.cell;
        cell.tensor = std::move(tensor);
        const size_type n_items = static_cast<size_type>(items.size());
        cell.items = std::move(items);
        cell.claimed = std::vector<std::atomic<bool>>(n_items);
        if (n_items == 0) {
          cell.value.store(pointer_type(cell.tensor), std::memory_order_relaxed);
          cell.state.store(Op1DCellState::Ready, std::memory_order_release);
        } else {
          cell.remaining.store(n_items, std::memory_order_relaxed);
          cell.state.store(Op1DCellState::InProgress, std::memory_order_release);
        }
      }

      /// Grabs and computes whatever (c,i) items nobody has claimed yet, calling
      /// `compute(c, i, tensor)` for each. Returns once no unclaimed items remain; does
      /// not wait for items other contributors are still working on (use acquire() for
      /// that). The last contribution to finish marks the entry Ready.
      template <typename F>
      void contribute(const Ticket& ticket, F&& compute) const {
        auto& cell = *ticket.cell;
        if (!ticket.creator) {
          // Wait out the brief Initializing window (tensor/work-list allocation) so
          // cell.items/cell.tensor are safe to read below.
          madness::MutexWaiter waiter;
          Op1DCellState s;
          while ((s = cell.state.load(std::memory_order_acquire)) == Op1DCellState::Empty ||
                 s == Op1DCellState::Initializing) {
            waiter.wait();
          }
        }
        for (size_type k = 0; k < cell.items.size(); ++k) {
          bool expected = false;
          if (cell.claimed[k].compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                       std::memory_order_relaxed)) {
            const WorkItem item = cell.items[k];
            compute(item.c, item.i, *cell.tensor);
            if (cell.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
              // We were the last item to finish: the tensor is now fully populated.
              cell.value.store(pointer_type(cell.tensor), std::memory_order_release);
              cell.state.store(Op1DCellState::Ready, std::memory_order_release);
            }
          }
        }
      }

      /// Blocks (with progressive backoff) until every item is done, then returns the
      /// finished tensor.
      pointer_type acquire(const Ticket& ticket) const {
        madness::MutexWaiter waiter;
        while (ticket.cell->state.load(std::memory_order_acquire) != Op1DCellState::Ready) {
          waiter.wait();
        }
        return ticket.cell->value.load(std::memory_order_acquire);
      }
    };

#ifndef MRA_ENABLE_HOST
    enum class AsyncCacheStatus { Available, Owned, Pending };

    /**
     * Non-blocking, tri-state, memoizing cache -- the device-path replacement for
     * SharedComputeCache/Op1DCache's claim()+acquire() above. There is no acquire()
     * at all: try_get() always returns immediately with one of:
     *   Available -- value is ready, here it is.
     *   Owned     -- the caller just became responsible for producing the value
     *                (allocating the destination tensor(s), submitting the
     *                generation kernel, and calling publish() once that kernel has
     *                been awaited to completion -- see GaussianConvolutionOperator::
     *                try_get_op). Ownership, once granted, cannot be
     *                un-claimed; the caller must follow through.
     *   Pending   -- someone else already owns it. The caller must yield --
     *                co_await ttg::device::wait() with NO arguments (a pure
     *                reschedule point; passing a buffer would force an unwanted
     *                device->host transfer) -- and call try_get() again on resume.
     *
     * This intentionally does NOT try to synchronize with whichever task's kernel
     * is filling a Pending entry -- PaRSEC's per-buffer version tracking exists to
     * pick which device holds the newest copy for data movement, not to order
     * kernels/tasks that share a buffer outside a normal TTG edge or a
     * ttg::device::coop() batch, and there is currently no cheaper TTG primitive
     * for that than a full stream-to-stream event (rejected here as too heavy for
     * a per-entry cost -- see the design discussion this class grew out of).
     * Instead: the OWNER is the only one who ever waits for the real GPU
     * completion (via co_await ttg::device::wait(), the plain "wait for kernels
     * this task itself submitted" form), and only calls publish() afterwards -- so
     * "Available" is only ever observed once the value is genuinely, fully
     * computed, at which point ordinary buffer placement (select()) is all a
     * later, unrelated consumer needs.
     */
    template <typename KeyT, typename ValueT>
    class AsyncCache {
    public:
      using pointer_type = std::shared_ptr<const ValueT>;

      struct Result {
        AsyncCacheStatus status;
        pointer_type value; // valid iff status == Available
      };

    private:
      struct Cell {
        std::atomic<CacheEntryState> state{CacheEntryState::Empty}; // Empty/Requested/Ready reused from above
        atomic_shared_ptr<pointer_type> value;
      };
      using cellptr_type = std::shared_ptr<Cell>;
      using map_type = madness::ConcurrentHashMap<KeyT, cellptr_type, HashFunctor<KeyT>>;

      mutable map_type m_map;

      cellptr_type get_or_create_cell(const KeyT& key) const {
        {
          typename map_type::const_accessor cacc;
          if (m_map.find(cacc, key)) {
            return cacc->second;
          }
        }
        typename map_type::accessor acc;
        if (m_map.insert(acc, key)) {
          acc->second = std::make_shared<Cell>();
        }
        return acc->second;
      }

    public:
      Result try_get(const KeyT& key) const {
        cellptr_type cell = get_or_create_cell(key);
        auto st = cell->state.load(std::memory_order_acquire);
        if (st == CacheEntryState::Ready) {
          return {AsyncCacheStatus::Available, cell->value.load(std::memory_order_acquire)};
        }
        CacheEntryState expected = CacheEntryState::Empty;
        if (cell->state.compare_exchange_strong(expected, CacheEntryState::Requested,
                                                 std::memory_order_acq_rel, std::memory_order_acquire)) {
          return {AsyncCacheStatus::Owned, nullptr};
        }
        if (expected == CacheEntryState::Ready) {
          return {AsyncCacheStatus::Available, cell->value.load(std::memory_order_acquire)};
        }
        return {AsyncCacheStatus::Pending, nullptr};
      }

      /// Called by whoever try_get() told Owned == true, once the value has been
      /// fully, genuinely computed (i.e. after co_await ttg::device::wait() on the
      /// generation kernel this task itself submitted) -- never before.
      void publish(const KeyT& key, pointer_type data) const {
        cellptr_type cell = get_or_create_cell(key);
        cell->value.store(std::move(data), std::memory_order_release);
        cell->state.store(CacheEntryState::Ready, std::memory_order_release);
      }
    };

    /**
     * Per-level table holding EVERY 1D convolution operator matrix a
     * GaussianConvolutionOperator<T,NDIM> may need at level `n`: every
     * dimension, every translation l in [-MRA_OP1D_MAX_DISPLACEMENT,
     * MRA_OP1D_MAX_DISPLACEMENT]. Generated by a single kernel launch (see
     * GaussianConvolutionOperator::try_get_op) instead of one tensor -- and
     * one kernel launch -- per (level, dimension, translation): there is
     * exactly one buffer per field to co_await ttg::device::select(),
     * regardless of how many displacements a task ultimately needs.
     * Consumers slice out the specific (dimension, translation) matrix they
     * need via Op1DSliceView below.
     */
    template <typename T, Dimension NDIM>
    struct LevelOp1DTable {
      Level n = 0;
      // [NDIM, MRA_OP1D_NUM_DISPLACEMENTS, count, rank, 2K, 2K]
      DenseTensor<T, 6> R;
      // [NDIM, MRA_OP1D_NUM_DISPLACEMENTS, count, rank, K, K]
      DenseTensor<T, 6> S;
      // [NDIM, MRA_OP1D_NUM_DISPLACEMENTS, count, rank, 5]
      DenseTensor<T, 5> norms1d;

      LevelOp1DTable() = default;
      LevelOp1DTable(Level n, size_type count, size_type rank, size_type K, ttg::scope scope)
      : n(n)
      , R(std::array<size_type, 6>{(size_type)NDIM, MRA_OP1D_NUM_DISPLACEMENTS, count, rank, 2 * K, 2 * K}, scope)
      , S(std::array<size_type, 6>{(size_type)NDIM, MRA_OP1D_NUM_DISPLACEMENTS, count, rank, K, K}, scope)
      , norms1d(std::array<size_type, 5>{(size_type)NDIM, MRA_OP1D_NUM_DISPLACEMENTS, count, rank, (size_type)5},
                scope)
      { }
    };

    /**
     * A read-only, single-(dimension, translation) slice of a shared
     * LevelOp1DTable. Cheap enough (a shared_ptr plus a couple of indices)
     * that GaussianConvolutionOperator::try_get_op() rebuilds a fresh one on
     * every call -- see OpResult::dims -- rather than caching it anywhere;
     * only the assembled norms tensor (which costs a kernel launch to
     * assemble, see OpResult::norms) is worth caching per (level,
     * displacement). Exposes R.buffer()/.current_view(),
     * S.buffer()/.current_view(), norms1d.buffer()/.view_on()/.current_view()
     * -- .buffer() returns the WHOLE table's buffer (there is only one to
     * select() per level -- see GaussianConvolutionOperator::
     * make_generation_input()), while .current_view()/.view_on() return
     * just this (dimension, translation)'s matrix.
     */
    template <typename T, Dimension NDIM>
    struct Op1DSliceView {
      template <typename TensorT>
      struct FieldView {
        const TensorT* tensor = nullptr;
        Dimension d = 0;
        size_type l_idx = 0;

        auto& buffer() const { return tensor->buffer(); }
        auto current_view() const { return tensor->current_view()(d, l_idx); }
        auto view_on(const ttg::device::Device& device) const { return tensor->view_on(device)(d, l_idx); }
      };

      std::shared_ptr<const LevelOp1DTable<T, NDIM>> table; // keeps R/S/norms1d alive
      FieldView<DenseTensor<T, 6>> R;
      FieldView<DenseTensor<T, 6>> S;
      FieldView<DenseTensor<T, 5>> norms1d;

      Op1DSliceView() = default;
      Op1DSliceView(std::shared_ptr<const LevelOp1DTable<T, NDIM>> tbl, Dimension d, Translation l)
      : table(std::move(tbl))
      , R{&table->R, d, (size_type)(l + MRA_OP1D_MAX_DISPLACEMENT)}
      , S{&table->S, d, (size_type)(l + MRA_OP1D_MAX_DISPLACEMENT)}
      , norms1d{&table->norms1d, d, (size_type)(l + MRA_OP1D_MAX_DISPLACEMENT)}
      { }

      // Lets call sites write res.dims[d]->R... (matching the pointer-style
      // access the host path's ConvolutionData1D<T> slots use), even though
      // this is a plain value, not a pointer.
      const Op1DSliceView* operator->() const { return this; }
    };
#endif // !MRA_ENABLE_HOST

  } // namespace detail

  /**
   * Host-path (get_op()) aggregate: bundles per-dimension R/S/norms1d
   * (owned, self-contained ConvolutionData1D<T> tensors -- see Op1DCache)
   * together with the assembled norms tensor. The device path has no
   * equivalent bundled type: its R/S views (detail::Op1DSliceView, a slice
   * of the shared, level-wide detail::LevelOp1DTable) are cheap enough to
   * rebuild on demand from GaussianConvolutionOperator::try_get_op()'s own
   * OpResult::dims every time, so only the norms tensor -- the one part
   * that costs a kernel launch (or, on the host path, an assembly loop) to
   * compute -- is worth caching there (see OpResult's norms_pointer_type).
   */
  template<typename T, size_type NDIM>
  struct ConvolutionData {
    std::array<std::shared_ptr<const ConvolutionData1D<T>>, NDIM> data;
    // also taken from MADNESS
    // 4D: veccount x rank x NDIM x [Rnorm, Snorm, Rnormf, Snormf, NSnormf]
    //     fac & munorm of each separated term is stored in the same tensor, at dim 0
    DenseTensor<T, 4> norms;

    ConvolutionData(size_type veccount, size_type rank)
    : data()
    , norms(std::array{veccount, rank, NDIM, (size_type)NormId::Count}, ttg::scope::SyncIn)
    { }
  };

  /**
   * MRA/TTG wrapper around the MADNESS SeparatedConvolution operator.
   * This class is responsible for generating the ConvolutionData for a given level and displacement.
   * Provides the operators in buffers so they can be used in device kernels.
   *
   * TODO: are all functions guaranteed to have the same rank? If so, we can just use the first one.
   *       It seems the rank is essentially K, but what do I know...
   */
  template <typename T, Dimension NDIM>
  class GaussianConvolutionOperator {

  public:

    /**
     * Construct a convolution operator
     */
    GaussianConvolutionOperator(std::shared_ptr<madness::SeparatedConvolution<T, NDIM>> mad_conv_sep)
    : m_mad_conv_sep_vec(std::move(std::vector<std::shared_ptr<madness::SeparatedConvolution<T, NDIM>>>(1, mad_conv_sep)))
    , m_max_rank(mad_conv_sep->get_rank())
    {
#ifndef MRA_ENABLE_HOST
      extract_gaussian_terms_for_device();
#endif // !MRA_ENABLE_HOST
    }

    /**
     * Construct a convolution operator
     */
    GaussianConvolutionOperator(const std::vector<std::shared_ptr<madness::SeparatedConvolution<T, NDIM>>>& mad_conv_sep)
    : m_mad_conv_sep_vec(mad_conv_sep)
    {
      // find the highest rank
      for (auto& mad_conv : m_mad_conv_sep_vec) {
        m_max_rank = std::max(m_max_rank, mad_conv->get_rank());
      }
#ifndef MRA_ENABLE_HOST
      extract_gaussian_terms_for_device();
#endif // !MRA_ENABLE_HOST
    }

    /**
     * Returns the number of operators
     */
    size_type count() const {
      return m_mad_conv_sep_vec.size();
    }

    /**
     * Returns the vector of displacements for a given operator and level.
     * NOTE: Returns the displacements as MADNESS keys, not MRA keys.
     */
    const auto& get_mad_displacements(int c, Level n) const {
      return m_mad_conv_sep_vec[c]->get_disp(n);
    }

    const auto& get_mad_op(int c) const {
      return m_mad_conv_sep_vec[c];
    }

#ifndef MRA_ENABLE_HOST
    /// Shared (not per-item) tensors needed by submit_generate_op1d_kernel --
    /// see try_get_op()'s has_generation_work()/submit_generation_kernel()
    /// comments for the sequence the caller must run these through (select()
    /// before reading their views).
    auto& shared_c_tensor() const { return m_shared_c; }
    auto& shared_hgT_tensor() const { return m_shared_hgT; }
    auto& shared_hgT2k_tensor() const { return m_shared_hgT2k; }
    auto& shared_quadx_tensor() const { return m_shared_quadx; }
    auto& shared_quadw_tensor() const { return m_shared_quadw; }
    size_type K() const { return m_K; }
    size_type npt() const { return m_npt; }
#endif // !MRA_ENABLE_HOST

    /**
     * Assembles ConvolutionData for the level and displacement.
     */
    std::shared_ptr<const ConvolutionData<T, NDIM>> get_op(Level n, Key<NDIM> disp) const {
      auto key = Key<NDIM>(0, n, disp.translation());
      auto agg_ticket = _datacache.claim(key);

      if (agg_ticket.ready) {
        // Someone else already finished this exact aggregate; just return it.
        return _datacache.acquire(agg_ticket);
      }

      /**
       * Claim and contribute to the per-dimension 1D tensors *before* checking whether
       * we own the aggregate entry. _op1d_cache is keyed only by (level, dimension,
       * translation), independent of which aggregate call is asking for it, so a task
       * that loses the race for this exact (level, displacement) aggregate can still
       * usefully help fill in the very same 1D tensors the winner needs -- instead of
       * only ever spin-waiting on a result it played no part in computing. Concretely:
       * if several tasks call get_op() for the same displacement concurrently, all of
       * them now pitch in on the (function, term) loop below; only the single winner
       * goes on to acquire() the finished tensors, run the norms computation and
       * publish the aggregate.
       */
      std::array<typename op1d_cache_type::Ticket, NDIM> op1d_tickets;
      for (Dimension d = 0; d < NDIM; ++d) {
        op1d_tickets[d] = _op1d_cache.claim(detail::Op1DKey(n, d, disp.translation()[d]));
        if (op1d_tickets[d].creator) {
          auto [tensor, items] = make_op1d_shell();
          _op1d_cache.init(op1d_tickets[d], std::move(tensor), std::move(items));
        }
      }
      for (Dimension d = 0; d < NDIM; ++d) {
        Translation l = disp.translation()[d];
        _op1d_cache.contribute(op1d_tickets[d], [&, n, l, d](size_type c, size_type i, ConvolutionData1D<T>& tensor) {
          compute_op1d_entry(n, l, d, c, i, tensor);
        });
      }

      if (!agg_ticket.owner) {
        // Someone else is already assembling (or has finished) this exact aggregate.
        // We've already helped compute whatever 1D pieces it needs above; nothing more
        // we can contribute for an identical request, so just wait for the publish.
        return _datacache.acquire(agg_ticket);
      }

      /**
       * We are responsible for computing the aggregate data for this Level/displacement.
       * The 1D tensors were already claimed/contributed to above; acquire() blocks only
       * on whatever items (if any) are still being finished by other contributors.
       */
      auto data = std::make_shared<ConvolutionData<T, NDIM>>(m_mad_conv_sep_vec.size(), m_max_rank);
      for (Dimension d = 0; d < NDIM; ++d) {
        data->data[d] = _op1d_cache.acquire(op1d_tickets[d]);
      }
      /**
       * Assemble the norms for each dimension and store the fac of each term.
       */
      auto norms_view = data->norms.view_on(ttg::device::Device::host());
      for (int c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        int i = 0;
        for (i = 0; i < mad_ops.size(); ++i) {
          for (int d = 0; d < NDIM; ++d) {
            auto cd_mad = mad_ops[i].getop(d)->nonstandard(n, disp.translation()[d]);
            norms_view(c, i, d, (int)NormId::Rnorm) = cd_mad->Rnorm;
            norms_view(c, i, d, (int)NormId::Snorm) = cd_mad->Tnorm;
            norms_view(c, i, d, (int)NormId::Rnormf) = cd_mad->Rnormf;
            norms_view(c, i, d, (int)NormId::Snormf) = cd_mad->Tnormf;
            norms_view(c, i, d, (int)NormId::NSnormf) = cd_mad->NSnormf;
          }
          auto fac = mad_ops[i].getfac();
          norms_view(c, i, 0, (int)NormId::Fac) = fac;
          norms_view(c, i, 0, (int)NormId::MUnorm) = munorm2_ns(c, n, i, data->norms) * std::abs(fac);
        }
        for (; i < m_max_rank; ++i) {
          for (int d = 0; d < NDIM; ++d) {
            norms_view(c, i, d, (int)NormId::Rnorm) = 0.0;
            norms_view(c, i, d, (int)NormId::Snorm) = 0.0;
            norms_view(c, i, d, (int)NormId::Rnormf) = 0.0;
            norms_view(c, i, d, (int)NormId::Snormf) = 0.0;
            norms_view(c, i, d, (int)NormId::NSnormf) = 0.0;
          }
          norms_view(c, i, 0, (int)NormId::Fac) = 0.0;
          norms_view(c, i, 0, (int)NormId::MUnorm) = 0.0;
        }
        /* Finally, store the norm of the whole operator */
        T norm = m_mad_conv_sep_vec[c]->norm(n, disp.to_madness_key(), disp.to_madness_key());
        norms_view(c, 0, 0, (int)NormId::Opnorm) = norm;
        norms_view(c, 0, 0, (int)NormId::Rank) = mad_ops.size();
      }
      _datacache.publish(agg_ticket, data);
      return data;
    }

#ifndef MRA_ENABLE_HOST
    using table_pointer_type = std::shared_ptr<const detail::LevelOp1DTable<T, NDIM>>;
    // Only the assembled norms tensor is cached on the device path -- the
    // R/S views (OpResult::dims) are cheap enough (a shared_ptr + a couple
    // of indices, see detail::Op1DSliceView) to rebuild fresh from the
    // level table on every try_get_op() call, so there's nothing else worth
    // caching per (level, displacement). What IS worth caching is the one
    // part that costs a kernel launch to assemble (see
    // submit_assemble_norms_kernel) -- exactly this tensor.
    using norms_pointer_type = std::shared_ptr<const DenseTensor<T, 4>>;

    enum class Status { Available, Owned, Pending };

    struct OpResult {
      Status status; // Available: norms is ready; dims is also valid (as always).
                      // Owned: caller now owns assembling norms for this
                      //   specific displacement -- device path: call
                      //   make_norms_tensor(), fold its buffer into the
                      //   round's one select()
                      //   (see add_assembly_buffers()), submit_assemble_
                      //   norms_kernel(), then publish_op(). Host path: call
                      //   finish_op_assembly() then publish_op().
                      // Pending: the norms tensor for this exact displacement
                      //   is still being assembled by an unrelated task
                      //   (dims/level_work may still be set below if THIS
                      //   call also owns the level's table -- see
                      //   level_work). Caller must submit level_work if
                      //   present regardless of status, then (if status ==
                      //   Pending) co_await ttg::device::suspend() once and
                      //   call try_get_op() again.
                      //
                      // level_work and status are independent: the first
                      // caller for a level not yet generated always gets
                      // level_work set (see its own comment below), whether
                      // or not it ALSO happens to be the first to touch this
                      // exact displacement's norms (status Owned) or not
                      // (status Pending/Available, decided from the same,
                      // not-yet-computed level_work table -- see dims below).
      norms_pointer_type norms;                                // valid iff Available
      std::shared_ptr<detail::LevelOp1DTable<T, NDIM>> level_work; // non-null iff THIS call must generate the level's table
      std::array<detail::Op1DSliceView<T, NDIM>, NDIM> dims{}; // populated whenever a table reference (level_work or an existing table) is known -- i.e. whenever status != Pending, OR status == Pending with level_work set
    };

    /**
     * Non-blocking query for the full (n, displacement) operator data --
     * the device-path replacement for the claim()+acquire() pair inside
     * get_op() above. See OpResult's comment for the contract.
     *
     * Every dimension and translation at a level are generated together, by
     * a single kernel launch, into one shared detail::LevelOp1DTable (see its
     * class comment): the first caller for a level not yet generated becomes
     * responsible for the whole table (out.level_work), and everyone else
     * -- for any displacement at that level -- waits on the very same table.
     * The norms cache (this exact displacement) is checked/claimed in this
     * SAME call regardless of whether the table itself still needs
     * generating: dims are sliced from out.level_work (not yet computed, but
     * already a valid, structurally-complete object -- see
     * detail::Op1DSliceView) rather than waiting for a later call once the
     * table becomes Available. This matters because a device task may
     * co_await ttg::device::select() only ONCE over its whole execution:
     * folding "does the table need generating" and "does the norms tensor
     * need assembling" into one query lets a caller learn everything it
     * needs to select() up front, in one round, instead of discovering
     * assembly ownership only on a LATER call (after the table already
     * exists), which would need a second, separate select().
     *
     * `disp` must be within [-MRA_OP1D_MAX_DISPLACEMENT,
     * MRA_OP1D_MAX_DISPLACEMENT] in every dimension; this is not expected to
     * ever trigger in practice (see MRA_OP1D_MAX_DISPLACEMENT's comment), so
     * it is a hard error rather than a fallback path.
     */
    OpResult try_get_op(Level n, Key<NDIM> disp) const {
      for (Dimension d = 0; d < NDIM; ++d) {
        Translation l = disp.translation()[d];
        if (l < -MRA_OP1D_MAX_DISPLACEMENT || l > MRA_OP1D_MAX_DISPLACEMENT) {
          throw std::runtime_error(
              "GaussianConvolutionOperator::try_get_op: translation out of the "
              "level-wide generation range [-MRA_OP1D_MAX_DISPLACEMENT, MRA_OP1D_MAX_DISPLACEMENT]");
        }
      }

      OpResult out;
      std::shared_ptr<const detail::LevelOp1DTable<T, NDIM>> table;
      auto level_res = _level_table_cache_async.try_get(detail::LevelKey(n));
      if (level_res.status == detail::AsyncCacheStatus::Pending) {
        out.status = Status::Pending;
        return out;
      }
      if (level_res.status == detail::AsyncCacheStatus::Owned) {
        out.level_work = std::make_shared<detail::LevelOp1DTable<T, NDIM>>(
            n, m_mad_conv_sep_vec.size(), m_max_rank, m_K, ttg::scope::Allocate);
        table = out.level_work; // not yet computed, but a valid object to slice views into
      } else {
        table = level_res.value; // Available
      }
      for (Dimension d = 0; d < NDIM; ++d) {
        out.dims[d] = detail::Op1DSliceView<T, NDIM>(table, d, disp.translation()[d]);
      }
      auto key = Key<NDIM>(0, n, disp.translation());
      auto norms_res = _norms_cache_async.try_get(key);
      if (norms_res.status == detail::AsyncCacheStatus::Available) {
        out.status = Status::Available;
        out.norms = norms_res.value;
        return out;
      }
      if (norms_res.status == detail::AsyncCacheStatus::Pending) {
        out.status = Status::Pending;
        return out;
      }
      out.status = Status::Owned;
      return out;
    }

    /**
     * Host-path (screener_tt) norms assembly: builds just the norms tensor
     * (not a whole ConvolutionData -- that bundling is a host/get_op()-only
     * concept, see ConvolutionData's own comment) once the level's table is
     * Available (i.e. after a try_get_op() call returned Owned). Caller must
     * have already done co_await ttg::device::wait(dims[d]->norms1d.buffer())
     * for every d (a small device->host transfer of the 5-float-per-(c,i)
     * norms1d slice) before calling this, so the host view read below is
     * valid -- this stays a plain (non-coroutine) function since only the
     * caller's own device-task body can co_await (see this class's header
     * comment).
     *
     * Mirrors get_op()'s norms-assembly loop above, but sources
     * Rnorm/Snorm/Rnormf/Snormf/NSnormf from each dimension's own norms1d
     * slice (already computed once, on-device, by the generation kernel)
     * instead of a second, redundant MADNESS nonstandard() call. Fac is
     * likewise already known (mad_ops[i].getfac(), extracted once at
     * construction -- see extract_gaussian_terms()) rather than re-read
     * here; kept as the mad_ops[i].getfac() call below only to stay
     * obviously-identical to get_op()'s existing padding-loop structure.
     *
     * Opnorm/Rank are computed from values already on hand, NOT via
     * SeparatedConvolution::norm(): that call would go through getop_ns()/
     * getmuop() to Convolution1D::nonstandard() on a cold ns_cache -- which
     * this device-generation path guarantees, since nothing here ever warms
     * MADNESS's own cache -- silently forcing the exact host-side R/S
     * regeneration (plus an SVD) that generating R/S on-device exists to
     * avoid. Opnorm = sqrt(sum_mu MUnorm_mu^2) (MADNESS's own getop_ns()
     * formula) is accumulated alongside MUnorm in the loop below; Rank is
     * just mad_ops.size().
     */
    norms_pointer_type finish_op_assembly(Level n, Key<NDIM> disp,
                                           const std::array<detail::Op1DSliceView<T, NDIM>, NDIM>& dims) const {
      // Constructed as a named local (not forwarded straight through
      // make_shared) to avoid make_shared's perfect-forwarding picking
      // Tensor's variadic Dims... constructor overload instead of the
      // intended (array, scope) one.
      const size_type norms_count = (size_type)m_mad_conv_sep_vec.size();
      const size_type norms_rank = (size_type)m_max_rank;
      const size_type norms_ndim = (size_type)NDIM;
      const size_type norms_kinds = (size_type)NormId::Count;
      DenseTensor<T, 4> norms_tensor(std::array{norms_count, norms_rank, norms_ndim, norms_kinds}, ttg::scope::SyncIn);
      auto norms = std::make_shared<DenseTensor<T, 4>>(std::move(norms_tensor));
      auto norms_view = norms->view_on(ttg::device::Device::host());
      for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        size_type i = 0;
        // Opnorm = sqrt(sum_mu MUnorm_mu^2) -- MADNESS's own SeparatedConvolution::
        // getop_ns() formula (operator.h), computed here from the MUnorm values
        // this loop already assembles below, rather than via
        // SeparatedConvolution::norm(): that call goes through getop_ns()/getmuop()
        // to Convolution1D::nonstandard() on a cold ns_cache -- which this
        // device-generation path guarantees, since nothing here ever warms MADNESS's
        // own cache -- silently forcing the exact host-side R/S regeneration (plus an
        // SVD) that generating R/S on-device was supposed to avoid.
        T opnorm_sumsq = T(0);
        for (; i < (size_type)mad_ops.size(); ++i) {
          for (Dimension d = 0; d < NDIM; ++d) {
            auto n1d = dims[d].norms1d.view_on(ttg::device::Device::host());
            norms_view(c, i, d, (int)NormId::Rnorm) = n1d(c, i, 0);
            norms_view(c, i, d, (int)NormId::Snorm) = n1d(c, i, 1);
            norms_view(c, i, d, (int)NormId::Rnormf) = n1d(c, i, 2);
            norms_view(c, i, d, (int)NormId::Snormf) = n1d(c, i, 3);
            norms_view(c, i, d, (int)NormId::NSnormf) = n1d(c, i, 4);
          }
          auto fac = mad_ops[i].getfac();
          norms_view(c, i, 0, (int)NormId::Fac) = fac;
          T munorm = munorm2_ns(c, n, i, *norms) * std::abs(fac);
          norms_view(c, i, 0, (int)NormId::MUnorm) = munorm;
          opnorm_sumsq += munorm * munorm;
        }
        for (; i < (size_type)m_max_rank; ++i) {
          for (Dimension d = 0; d < NDIM; ++d) {
            norms_view(c, i, d, (int)NormId::Rnorm) = 0.0;
            norms_view(c, i, d, (int)NormId::Snorm) = 0.0;
            norms_view(c, i, d, (int)NormId::Rnormf) = 0.0;
            norms_view(c, i, d, (int)NormId::Snormf) = 0.0;
            norms_view(c, i, d, (int)NormId::NSnormf) = 0.0;
          }
          norms_view(c, i, 0, (int)NormId::Fac) = 0.0;
          norms_view(c, i, 0, (int)NormId::MUnorm) = 0.0;
        }
        norms_view(c, 0, 0, (int)NormId::Opnorm) = std::sqrt(opnorm_sumsq);
        norms_view(c, 0, 0, (int)NormId::Rank) = mad_ops.size();
      }
      return norms;
    }

    /// Publishes the norms tensor assembled by finish_op_assembly() (host
    /// path) or submit_assemble_norms_kernel() (device path) for (n, disp).
    void publish_op(Level n, Key<NDIM> disp, norms_pointer_type data) const {
      auto key = Key<NDIM>(0, n, disp.translation());
      _norms_cache_async.publish(key, std::move(data));
    }

    /**
     * Device-path alternative to finish_op_assembly(): assembles just the
     * norms tensor ON-DEVICE, via two small dedicated kernels
     * (detail::assemble_op1d_norms_kernel then detail::assemble_op1d_opnorm_
     * kernel), instead of a host-side copy that would require pulling the
     * level table's norms1d back to host first -- AND, critically, instead
     * of calling MADNESS's SeparatedConvolution::norm() for Opnorm, which
     * would (see submit_assemble_norms_kernel()'s comment) silently force
     * the exact host-side R/S regeneration this whole file exists to avoid.
     * Needed because shell0_tt/accumulate_tt feed the result straight into
     * an actual device convolution kernel (unlike screener_tt, which only
     * ever reads Opnorm on the host, and so just uses finish_op_assembly()
     * -- see that function's comment, which takes the analogous host-side
     * shortcut). R/S for the kernel come straight from res.dims -- no
     * separate "assembling" object needed for those, see OpResult's
     * comment. Call sequence (see e.g. shell0_tt/accumulate_tt):
     *   bool need_assemble = (res.status == Status::Owned);
     *   std::shared_ptr<DenseTensor<T,4>> norms;
     *   if (need_assemble) {
     *     norms = op.make_norms_tensor();
     *   }
     *   auto input = op.make_generation_input(res, items_buf, ws_buf);
     *   if (need_assemble) op.add_assembly_buffers(input, *norms);
     *   co_await ttg::device::select(input);
     *   if (op.has_generation_work(res)) op.submit_generation_kernel(res, items, items_buf, ws_buf);
     *   if (need_assemble) op.submit_assemble_norms_kernel(res, n, disp, *norms);
     *   co_await ttg::device::wait(); // wait for OUR OWN submitted kernels before publishing
     *   if (op.has_generation_work(res)) op.publish_generation_work(res);
     *   if (need_assemble) op.publish_op(n, disp, norms);
     */

    /// Allocates the norms tensor a device-path Owned result must finish
    /// assembling -- device-resident garbage until submit_assemble_norms_
    /// kernel() fills it. Call before the round's one select() -- its
    /// buffer must be added to it (see add_assembly_buffers()).
    std::shared_ptr<DenseTensor<T, 4>> make_norms_tensor() const {
      // See finish_op_assembly()'s identical named-local workaround.
      const size_type norms_count = (size_type)m_mad_conv_sep_vec.size();
      const size_type norms_rank = (size_type)m_max_rank;
      const size_type norms_ndim = (size_type)NDIM;
      const size_type norms_kinds = (size_type)NormId::Count;
      DenseTensor<T, 4> norms_tensor(std::array{norms_count, norms_rank, norms_ndim, norms_kinds},
                                      ttg::scope::Allocate);
      return std::make_shared<DenseTensor<T, 4>>(std::move(norms_tensor));
    }

    /// Adds the buffers submit_assemble_norms_kernel() needs -- the shared
    /// per-(count-index,term) Fac and Rank tables, and the (not yet
    /// populated) norms tensor -- to `input`, so they ride along in the
    /// round's one select() alongside whatever make_generation_input()
    /// already added. No per-(level,displacement) Opnorm/Rank buffer needed
    /// here (unlike an earlier version of this function): Rank is an
    /// operator-wide constant (m_shared_rank, extracted once at
    /// construction) and Opnorm is now computed entirely on-device -- see
    /// submit_assemble_norms_kernel()'s comment.
    void add_assembly_buffers(ttg::device::Input& input, DenseTensor<T, 4>& norms) const {
      input.add(m_shared_fac.buffer());
      input.add(m_shared_rank.buffer());
      input.add(norms.buffer());
    }

    /// Call only after co_await-ing the select() that add_assembly_buffers()
    /// contributed to, and (if has_generation_work(res)) after
    /// submit_generation_kernel() -- same stream, so both kernels submitted
    /// here are guaranteed to see the generation kernel's writes to norms1d
    /// (and detail::assemble_op1d_opnorm_kernel is guaranteed to see
    /// detail::assemble_op1d_norms_kernel's writes to MUnorm) without any
    /// explicit wait() between any of them. Reads norms1d directly from
    /// whichever table backs res.dims (res.dims[0].table -- valid whether
    /// that table was already Available or is the one THIS round just
    /// generated) entirely on-device; no host round-trip, unlike
    /// finish_op_assembly()'s norms1d.view_on(host) reads.
    ///
    /// Deliberately does NOT call MADNESS's SeparatedConvolution::norm() for
    /// Opnorm (same reasoning as finish_op_assembly()'s host-side path,
    /// which likewise no longer calls it, computing it from the MUnorm
    /// values assembled a few lines earlier instead, to stay off MADNESS's
    /// own cache entirely). That call would go through
    /// getop_ns()/getmuop() to Convolution1D::nonstandard() on MADNESS's
    /// own ns_cache -- and on this path nothing ever warms that cache (this
    /// file's whole point is generating R/S on-device instead), so it would
    /// unconditionally re-trigger the exact host-side quadrature-projection/
    /// autocorrelation/two-scale-filter/SVD pipeline this file replaces.
    /// detail::assemble_op1d_opnorm_kernel computes the identical value
    /// (Opnorm = sqrt(sum_i MUnorm_i^2), MADNESS's own getop_ns() formula)
    /// on-device instead, from the MUnorm values detail::
    /// assemble_op1d_norms_kernel already wrote.
    void submit_assemble_norms_kernel(const OpResult& res, Level n, Key<NDIM> disp,
                                       DenseTensor<T, 4>& norms) const {
      auto& table = *res.dims[0].table;
      std::array<Translation, NDIM> lx;
      for (Dimension d = 0; d < NDIM; ++d) lx[d] = disp.translation()[d];
      const size_type count = m_mad_conv_sep_vec.size();
      const size_type rank = (size_type)m_max_rank;
      submit_assemble_op1d_norms_kernel<T, NDIM>(
          count, rank, n, lx, MRA_OP1D_MAX_DISPLACEMENT,
          table.norms1d.current_view(), m_shared_fac.current_view(),
          norms.current_view(), ttg::device::current_stream());
      submit_assemble_op1d_opnorm_kernel<T>(
          count, rank, norms.current_view(), m_shared_rank.current_view(), ttg::device::current_stream());
    }

    /**
     * The non-coroutine steps every call site's generation round needs,
     * factored here so tasks/convolution.h's three call sites only have to
     * interleave co_await calls around them, not duplicate the bookkeeping.
     *
     * A device task may call co_await ttg::device::select() only ONCE over
     * its whole execution, so collect_generation_items() must be (and is)
     * callable BEFORE that one select() -- it only reads host-side state
     * (m_term_params, res.level_work->n), never a device view -- letting the
     * caller fold every buffer this generation round needs (the shared
     * generator tables, the level table's R/S/norms1d, items_buf, and
     * workspace) into that single select() call alongside it. Typical
     * call-site sequence when the caller (like screener_tt) needs norms1d
     * back on the HOST afterward, via finish_op_assembly():
     *   if (op.has_generation_work(res)) {
     *     auto items = op.collect_generation_items(res);
     *     auto items_buf = make_generate_op1d_items_buffer(items);
     *     auto ws_buf = op.make_generation_workspace(items.size());
     *     auto input = op.make_generation_input(res, items_buf, ws_buf);
     *     co_await ttg::device::select(input);
     *     op.submit_generation_kernel(res, items, items_buf, ws_buf);
     *     co_await ttg::device::wait(res.level_work->norms1d.buffer());
     *     op.publish_generation_work(res);
     *   }
     * A caller that instead feeds the result into an actual device kernel
     * (shell0_tt/accumulate_tt) never needs norms1d on the host at all --
     * see make_assembling_data()'s comment for that variant, which folds
     * table generation and aggregate assembly into the very same select().
     */
    bool has_generation_work(const OpResult& res) const {
      return static_cast<bool>(res.level_work);
    }

    /// Every buffer this generation round needs -- the shared generator
    /// tables, the whole level table's R/S/norms1d (one buffer per field,
    /// not one per dimension/translation), items_buf, and workspace -- so
    /// the caller can co_await ttg::device::select() on all of it in the
    /// ONE select() call its task is allowed. Call after
    /// collect_generation_items()/make_generate_op1d_items_buffer()/
    /// make_generation_workspace(), which are all host-only and need no
    /// prior select() of their own.
    ttg::device::Input make_generation_input(const OpResult& res, ttg::Buffer<GenerateOp1DItem<T>>& items_buf,
                                              ttg::Buffer<T>& workspace) const {
      ttg::device::Input input;
      input.add(m_shared_c.buffer());
      input.add(m_shared_hgT.buffer());
      input.add(m_shared_hgT2k.buffer());
      input.add(m_shared_quadx.buffer());
      input.add(m_shared_quadw.buffer());
      if (res.level_work) {
        input.add(res.level_work->R.buffer());
        input.add(res.level_work->S.buffer());
        input.add(res.level_work->norms1d.buffer());
        input.add(items_buf);
        input.add(workspace);
      }
      return input;
    }

    /// Builds the flat (dimension, count-index, term) work-item list
    /// covering the WHOLE table in res.level_work -- each item generates
    /// every translation in [-MRA_OP1D_MAX_DISPLACEMENT,
    /// MRA_OP1D_MAX_DISPLACEMENT] for its own (d, c, i) -- so a single
    /// submit_generate_op1d_kernel call fills every matrix at this level.
    /// Padding entries (i beyond a given count-index's real rank) get a
    /// default-constructed (coeff==0) GaussianTermParams -- generate_op1d_one's
    /// own coeff==0 sentinel check zero-fills them on-device. Host-only (no
    /// device view read) -- see has_generation_work()'s comment for why that
    /// matters here -- so safe to call before the round's one select().
    std::vector<GenerateOp1DItem<T>> collect_generation_items(const OpResult& res) const {
      std::vector<GenerateOp1DItem<T>> items;
      if (!res.level_work) return items;
      const size_type count = m_mad_conv_sep_vec.size();
      const size_type rank = (size_type)m_max_rank;
      items.reserve((size_type)NDIM * count * rank);
      for (Dimension d = 0; d < NDIM; ++d) {
        for (size_type c = 0; c < count; ++c) {
          for (size_type i = 0; i < rank; ++i) {
            GenerateOp1DItem<T> item;
            item.params = m_term_params[(c * rank + i) * NDIM + d];
            item.n = res.level_work->n;
            item.d = d;
            item.c = c;
            item.i = i;
            // TODO: levels 0, 1, and 2 don't need the full displacement range;
            // query the actual displacements from madness::Displacements instead
            item.lmin = -MRA_OP1D_MAX_DISPLACEMENT;
            item.lmax =  MRA_OP1D_MAX_DISPLACEMENT;
            items.push_back(std::move(item));
          }
        }
      }
      return items;
    }

    /// Allocates (Allocate scope -- device-resident, no host fill needed) the
    /// per-(item, translation) cooperative scratch the generation kernel's
    /// threads share (see generate_op1d.h's Op1DWorkspaceOffsets/
    /// generate_op1d_workspace_size). Sized by MRA_OP1D_NUM_DISPLACEMENTS
    /// (the workspace stride submit_generation_kernel below passes as
    /// max_range_len -- see generate_op1d_kernel's comment on why the
    /// generator now needs one scratch region per (item, translation) pair,
    /// not just per item, to run every translation of an item concurrently
    /// instead of looping over them serially in one block) since
    /// collect_generation_items() currently always requests the full
    /// level-wide range for every item. Host-only (no device view read) --
    /// safe to call before the round's one select(), same as
    /// collect_generation_items().
    ttg::Buffer<T> make_generation_workspace(size_type n_items) const {
      return ttg::Buffer<T>(generate_op1d_workspace_size(m_K) * n_items * MRA_OP1D_NUM_DISPLACEMENTS,
                             ttg::scope::Allocate);
    }

    /// Call only after co_await-ing make_generation_input()'s select();
    /// `items`/`items_buf`/`workspace` must be the exact objects that went
    /// into that same select() call (via make_generation_input).
    void submit_generation_kernel(const OpResult& res, const std::vector<GenerateOp1DItem<T>>& items,
                                   ttg::Buffer<GenerateOp1DItem<T>>& items_buf,
                                   ttg::Buffer<T>& workspace) const {
      if (items.empty()) return;
      auto& table = *res.level_work;
      submit_generate_op1d_kernel<T>(
          items_buf.current_device_ptr(), items.size(), MRA_OP1D_NUM_DISPLACEMENTS, m_K,
          m_shared_c.current_view(), m_shared_hgT.current_view(), m_shared_hgT2k.current_view(),
          m_shared_quadx.current_view().data(), m_shared_quadw.current_view().data(), m_npt,
          workspace.current_device_ptr(), MRA_OP1D_MAX_DISPLACEMENT,
          table.R.current_view(), table.S.current_view(), table.norms1d.current_view(),
          ttg::device::current_stream());
    }

    /// Call only after co_await-ing the generation kernel to completion
    /// (co_await ttg::device::wait(), no argument). Publishes the whole
    /// table as Available so every other task at this level -- any
    /// dimension, any translation -- finds it immediately.
    void publish_generation_work(const OpResult& res) const {
      if (!res.level_work) return;
      _level_table_cache_async.publish(detail::LevelKey(res.level_work->n),
                                        table_pointer_type(res.level_work));
    }
#endif // !MRA_ENABLE_HOST

  private:
    using op1d_cache_type = detail::Op1DCache<T>;
    using data_cache_type = detail::SharedComputeCache<Key<NDIM>, ConvolutionData<T, NDIM>>;

    // madness separate convolution object, provided by application
    std::vector<std::shared_ptr<madness::SeparatedConvolution<T, NDIM>>> m_mad_conv_sep_vec;
    int m_max_rank = 0;
    // our own cache of 1D operator data for each [Level, Dimension, Translation]
    mutable op1d_cache_type _op1d_cache;
    // our own cache of full operator data for each [Level, Translation] (encoded as Key)
    // includes all terms and dimensions
    mutable data_cache_type _datacache;

    template<typename TV>
    void copy_from_madtensor(TV&& tv, const madness::Tensor<T>& m) const {
      assert(tv.size() == m.size());
      for (size_type i = 0; i < m.size(); ++i) {
        tv[i] = m.ptr()[i];
      }
    }

#ifndef MRA_ENABLE_HOST
    // Device-path caches: try_get()/publish() only, no acquire()/MutexWaiter --
    // see detail::AsyncCache's class comment and try_get_op() above.
    // Exactly one caller per level generates the whole
    // detail::LevelOp1DTable (every dimension, every translation in
    // [-MRA_OP1D_MAX_DISPLACEMENT, MRA_OP1D_MAX_DISPLACEMENT]) in a single
    // kernel launch; every other displacement at that level just reads it.
    mutable detail::AsyncCache<detail::LevelKey, detail::LevelOp1DTable<T, NDIM>> _level_table_cache_async;
    // Caches only the assembled norms tensor per (level, displacement) --
    // not a whole aggregate -- since R/S (OpResult::dims) are cheap enough
    // to rebuild from the level table on every try_get_op() call; see
    // OpResult's comment.
    mutable detail::AsyncCache<Key<NDIM>, DenseTensor<T, 4>> _norms_cache_async;

    // Shared (not per-term) tensors extracted once, at construction, from
    // MADNESS's Convolution1D base -- identical for every term/dimension of
    // this operator since they depend only on K and npt, never on
    // (coeff, expnt). See extract_gaussian_terms_for_device().
    size_type m_K = 0;
    size_type m_npt = 0;
    DenseTensor<T, 3> m_shared_c;      // K x K x 4K autocorrelation tensor
    DenseTensor<T, 2> m_shared_hgT;    // 2K x 2K two-scale filter (transpose)
    DenseTensor<T, 2> m_shared_hgT2k;  // 4K x 4K two-scale filter for order-2K
    DenseTensor<T, 1> m_shared_quadx;  // npt Gauss-Legendre quadrature points
    DenseTensor<T, 1> m_shared_quadw;  // npt Gauss-Legendre quadrature weights
    // Per-(count-index, term) Fac, d-invariant (mad_ops[i].getfac() doesn't
    // depend on dimension) -- read by submit_assemble_norms_kernel() so it
    // doesn't need m_term_params (a per-d-too, host-only vector) uploaded.
    DenseTensor<T, 2> m_shared_fac;    // count x rank
    // Per-count-index real separated-expansion rank (get_ops().size()) --
    // an operator-wide constant, independent of level/displacement, so
    // extracted once here rather than recomputed (or, worse, re-derived via
    // a MADNESS call) on every submit_assemble_norms_kernel() round.
    DenseTensor<T, 1> m_shared_rank;   // count

    // Per-(count-index c, term i, dimension d) Gaussian parameters, extracted
    // once. Flattened as m_term_params[(c*m_max_rank + i)*NDIM + d]; padding
    // slots (i beyond count-index c's real rank) are left default-constructed
    // (coeff==0), which generate_op1d_one's own sentinel check zero-fills.
    std::vector<GaussianTermParams<T>> m_term_params;

    /**
     * Extracts everything the on-device generator (mra/kernels/generate_op1d.h)
     * needs from MADNESS's already-built GaussianConvolution1D term objects --
     * called once, from both constructors, guarded to device builds only. This
     * still touches MADNESS, but it is a one-time, per-run setup cost, not the
     * per-(level,translation)-key cost that motivated moving generation to the
     * GPU in the first place (see this file's header comment).
     *
     * Requires every term/dimension to actually be a GaussianConvolution1D --
     * true for the Gaussian-separated-expansion operators this codebase
     * constructs (see misc/convolutiondata.h's from-scratch reference, which
     * assumes the same), but not guaranteed by SeparatedConvolution's API in
     * general (e.g. a non-Gaussian Convolution1D subclass would fail the
     * dynamic_pointer_cast below). Throws rather than silently mis-generating
     * if that assumption doesn't hold.
     */
    void extract_gaussian_terms_for_device() {
      m_K = (size_type)m_mad_conv_sep_vec.front()->get_k();
      const size_type K = m_K;

      auto first_ops = m_mad_conv_sep_vec.front()->get_ops();
      auto first = std::dynamic_pointer_cast<const madness::GaussianConvolution1D<T>>(first_ops[0].getop(0));
      if (!first) {
        throw std::runtime_error(
            "GaussianConvolutionOperator: operator terms are not GaussianConvolution1D -- "
            "on-device transition-matrix generation only supports Gaussian-based "
            "separated expansions.");
      }
      m_npt = (size_type)first->npt;

      m_shared_c = DenseTensor<T, 3>(std::array{K, K, 4 * K}, ttg::scope::SyncIn);
      m_shared_hgT = DenseTensor<T, 2>(std::array{2 * K, 2 * K}, ttg::scope::SyncIn);
      m_shared_hgT2k = DenseTensor<T, 2>(std::array{4 * K, 4 * K}, ttg::scope::SyncIn);
      m_shared_quadx = DenseTensor<T, 1>(m_npt, ttg::scope::SyncIn);
      m_shared_quadw = DenseTensor<T, 1>(m_npt, ttg::scope::SyncIn);
      auto c_view = m_shared_c.current_view();
      auto hgT_view = m_shared_hgT.current_view();
      auto hgT2k_view = m_shared_hgT2k.current_view();
      auto qx_view = m_shared_quadx.current_view();
      auto qw_view = m_shared_quadw.current_view();
      for (size_type i = 0; i < K; ++i)
        for (size_type j = 0; j < K; ++j)
          for (size_type k = 0; k < 4 * K; ++k)
            c_view(i, j, k) = static_cast<T>(first->c(i, j, k));
      for (size_type i = 0; i < 2 * K; ++i)
        for (size_type j = 0; j < 2 * K; ++j)
          hgT_view(i, j) = static_cast<T>(first->hgT(i, j));
      for (size_type i = 0; i < 4 * K; ++i)
        for (size_type j = 0; j < 4 * K; ++j)
          hgT2k_view(i, j) = static_cast<T>(first->hgT2k(i, j));
      for (size_type i = 0; i < m_npt; ++i) {
        qx_view(i) = static_cast<T>(first->quad_x(i));
        qw_view(i) = static_cast<T>(first->quad_w(i));
      }

      m_term_params.assign(m_mad_conv_sep_vec.size() * (size_type)m_max_rank * NDIM, GaussianTermParams<T>{});
      m_shared_fac = DenseTensor<T, 2>(std::array<size_type, 2>{m_mad_conv_sep_vec.size(), (size_type)m_max_rank},
                                       ttg::scope::SyncIn);
      {
        auto fac_view = m_shared_fac.current_view();
        for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c)
          for (size_type i = 0; i < (size_type)m_max_rank; ++i)
            fac_view(c, i) = T(0); // padding default; real entries overwritten below
      }
      for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        for (size_type i = 0; i < (size_type)mad_ops.size(); ++i) {
          T fac = static_cast<T>(mad_ops[i].getfac());
          m_shared_fac.current_view()(c, i) = fac;
          for (Dimension d = 0; d < NDIM; ++d) {
            auto term = std::dynamic_pointer_cast<const madness::GaussianConvolution1D<T>>(mad_ops[i].getop(d));
            if (!term) {
              throw std::runtime_error(
                  "GaussianConvolutionOperator: operator terms are not GaussianConvolution1D");
            }
            GaussianTermParams<T> p;
            p.coeff = static_cast<T>(term->coeff);
            p.expnt = static_cast<T>(term->expnt);
            p.natlev = static_cast<Level>(term->natlev);
            p.m = static_cast<int>(term->m);
            p.fac = fac;
            m_term_params[(c * (size_type)m_max_rank + i) * NDIM + d] = p;
          }
        }
        // padding entries (i >= mad_ops.size(), up to m_max_rank) are left
        // default-constructed (coeff==0) -- see this function's header comment.
      }

      m_shared_rank = DenseTensor<T, 1>(m_mad_conv_sep_vec.size(), ttg::scope::SyncIn);
      {
        auto rank_view = m_shared_rank.current_view();
        for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
          rank_view(c) = (T)m_mad_conv_sep_vec[c]->get_ops().size();
        }
      }
    }
#endif // !MRA_ENABLE_HOST

    /**
     * Allocates a fresh ConvolutionData1D tensor and returns it together with the flat
     * list of (function, term) work items whose computation actually needs MADNESS --
     * i.e. every (c,i) with i < the real rank of function c's separated expansion.
     * Padding entries (i beyond that rank, up to m_max_rank) are zero and cheap, so
     * they're filled in here directly rather than turned into their own work items.
     * Independent of level/dimension/displacement, so it's fast enough to always run on
     * the task that wins the Op1DCache claim() race, without needing to be split further.
     */
    std::pair<std::shared_ptr<ConvolutionData1D<T>>, std::vector<typename op1d_cache_type::WorkItem>>
    make_op1d_shell() const {
      auto tensor = std::make_shared<ConvolutionData1D<T>>(m_mad_conv_sep_vec.size(), m_max_rank,
                                                             m_mad_conv_sep_vec.front()->get_k());
      auto rv = tensor->R.view_on(ttg::device::Device::host());
      auto sv = tensor->S.view_on(ttg::device::Device::host());
      std::vector<typename op1d_cache_type::WorkItem> items;
      for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        size_type i = 0;
        for (; i < mad_ops.size(); ++i) {
          items.push_back({c, i});
        }
        // fill in the rest of the tensors with zeros
        for (; i < m_max_rank; ++i) {
          rv(c, i) = 0.0;
          sv(c, i) = 0.0;
        }
      }
      return {std::move(tensor), std::move(items)};
    }

    /**
     * Computes a single (function, term) entry of the 1D operator tensor for
     * (n, d, l) and writes it into that entry's disjoint slice of `tensor`.
     * Safe to run concurrently with other (c,i) entries of the very same tensor:
     * different terms use different MADNESS Convolution1D instances (each with its own
     * internal, thread-safe SimpleCache), get_ops()/getop() are plain accessors into
     * already-built state, and each (c,i) only ever touches its own slice of `tensor`.
     */
    void compute_op1d_entry(Level n, Translation l, Dimension d, size_type c, size_type i,
                             ConvolutionData1D<T>& tensor) const {
      auto rv = tensor.R.view_on(ttg::device::Device::host());
      auto sv = tensor.S.view_on(ttg::device::Device::host());
      auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
      std::shared_ptr<const madness::Convolution1D<T>> conv1d = mad_ops[i].getop(d);
      const madness::ConvolutionData1D<T>* cd_mad = conv1d->nonstandard(n, l);
      if (!(cd_mad->R.size() == 0 && cd_mad->T.size() == 0)) {
        copy_from_madtensor(rv(c, i), cd_mad->R);
        copy_from_madtensor(sv(c, i), cd_mad->T); // S = T for us
      }
    }


    /// Taken from MADNESS, since munorm2_ns is private in SeparatedConvolution
    /// and we have no way to get the norm otherwise.
    /// Computes the Frobenius norm of one of the separated terms for the NS form
    ///       ... WITHOUT FACTOR INCLUDED
    /// compute for 1 term, all dim, 1 disp, essentially for SeparatedConvolutionInternal
    ///
    /// Takes the norms tensor directly (not a whole ConvolutionData) so both
    /// get_op()'s host-path aggregate and finish_op_assembly()'s
    /// device-path norms-only tensor can share this without either needing
    /// the other's bundling.
    double munorm2_ns(size_type c, Level n, size_type mu, const DenseTensor<T, 4>& norms) const {

        double prodR=1.0;
        double prod=1.0, sum=0.0;
        auto norms_view = norms.view_on(ttg::device::Device::host());
        for (std::size_t d=0; d<NDIM; ++d) {
            double a = norms_view(c, mu, d, (int)NormId::NSnormf);
            double b = norms_view(c, mu, d, (int)NormId::Snormf);
            double aa = std::min(a,b);
            double bb = std::max(a,b);
            prod *= bb;
            if (bb > 0.0) sum +=(aa/bb);
        }
        if (n) prod *= sum;
        prodR = prod;

        return prodR;
    }
  };

} // namespace mra

#endif // CONV_MAD_H
