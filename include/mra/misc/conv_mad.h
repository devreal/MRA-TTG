#ifndef CONV_MAD_H
#define CONV_MAD_H

#include <atomic>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <cstdlib>
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

  /// Absolute ceiling on |translation| (in each dimension) the level-wide 1D
  /// operator generation table ever needs -- see GaussianConvolutionOperator's
  /// header comment and detail::LevelOp1DTable. A translation outside this
  /// range is a hard error (GaussianConvolutionOperator::try_get_op() throws),
  /// not a fallback -- screener_tt/accumulate_tt only ever walk MADNESS's own
  /// shell-ordered, quickly-decaying displacement list, so this is not
  /// expected to trigger in practice. The ACTUAL per-level bound used to size
  /// and generate a given level's table is max_op1d_displacement(n) below,
  /// which is tighter than this constant at low levels.
  inline constexpr Translation MRA_OP1D_MAX_DISPLACEMENT = 4;
  inline constexpr size_type MRA_OP1D_NUM_DISPLACEMENTS = 2 * MRA_OP1D_MAX_DISPLACEMENT + 1;

  /// A translation at level n can't exceed 2^n-1 in magnitude -- there are
  /// only that many boxes to either side at that level -- so levels below
  /// MRA_OP1D_MAX_DISPLACEMENT's own crossover point (0, 1, 2 for the default
  /// 4) need a narrower table/generation range than the worst case. Mirrors
  /// MADNESS's own Displacements<NDIM>::make_disp_periodic level clipping
  /// (madness/mra/displacements.h: "if (bmax > twon-1) bmax = twon-1;").
  /// Bails out before computing 2^n-1 for large n -- well before n reaches
  /// a point where the shift could matter, min() below always already
  /// selects MRA_OP1D_MAX_DISPLACEMENT -- to stay well-defined for every
  /// valid Level without relying on a signed-shift bound on Translation.
  inline constexpr Translation max_op1d_displacement(Level n) {
    if (n < 0 || n >= 20) return MRA_OP1D_MAX_DISPLACEMENT;
    const Translation twon_minus_1 = (Translation(1) << n) - Translation(1);
    return twon_minus_1 < MRA_OP1D_MAX_DISPLACEMENT ? twon_minus_1 : MRA_OP1D_MAX_DISPLACEMENT;
  }
  inline constexpr size_type num_op1d_displacements(Level n) {
    return (size_type)(2 * max_op1d_displacement(n) + 1);
  }

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
     * Non-blocking, tri-state, memoizing slot -- the device-path replacement for
     * SharedComputeCache/Op1DCache's claim()+acquire() above. There is no acquire()
     * at all: try_get() always returns immediately with one of:
     *   Available -- value is ready, here it is.
     *   Owned     -- the caller just became responsible for producing the value:
     *                installing the destination object via emplace(), submitting the
     *                kernel(s) that fill it, and calling publish() once those have
     *                been awaited to completion -- see GaussianConvolutionOperator::
     *                try_get_op. Ownership, once granted, cannot be un-claimed; the
     *                caller must follow through.
     *   Pending   -- someone else already owns it. The caller must yield --
     *                co_await ttg::device::suspend() (a pure reschedule point; waiting
     *                on a buffer would force an unwanted device->host transfer) --
     *                and call try_get() again on resume.
     *
     * This intentionally does NOT try to synchronize with whichever task's kernel
     * is filling a Pending entry -- PaRSEC's per-buffer version tracking exists to
     * pick which device holds the newest copy for data movement, not to order
     * kernels/tasks that share a buffer outside a normal TTG edge or a
     * ttg::device::coop() batch, and there is currently no cheaper TTG primitive
     * for that than a full stream-to-stream event (rejected here as too heavy for
     * a per-entry cost). Instead: the OWNER is the only one who ever waits for the
     * real GPU completion, and only calls publish() afterwards -- so "Available"
     * is only ever observed once the value is genuinely, fully computed, at which
     * point ordinary buffer placement (select()) is all a later, unrelated
     * consumer needs.
     *
     * The slot owns its value for its own lifetime (slots are never evicted, and
     * live as long as the GaussianConvolutionOperator holding them), so every
     * lookup hands out a plain pointer -- no reference counting on the hot path.
     * Only the owner ever writes m_value (emplace(), strictly between try_get()
     * returning Owned and publish()), and nobody else reads it before observing
     * Ready with acquire semantics, so m_value needs no synchronization of its own.
     */
    template <typename ValueT>
    class AsyncSlot {
    public:
      struct Result {
        AsyncCacheStatus status;
        const ValueT* value; // valid iff status == Available
      };

      Result try_get() {
        auto st = m_state.load(std::memory_order_acquire);
        if (st == CacheEntryState::Ready) {
          return {AsyncCacheStatus::Available, m_value.get()};
        }
        CacheEntryState expected = CacheEntryState::Empty;
        if (m_state.compare_exchange_strong(expected, CacheEntryState::Requested,
                                            std::memory_order_acq_rel, std::memory_order_acquire)) {
          return {AsyncCacheStatus::Owned, nullptr};
        }
        if (expected == CacheEntryState::Ready) {
          return {AsyncCacheStatus::Available, m_value.get()};
        }
        return {AsyncCacheStatus::Pending, nullptr};
      }

      /// Owner only: installs the (not yet computed) value. The returned pointer
      /// stays valid for the slot's whole lifetime.
      ValueT* emplace(std::unique_ptr<ValueT> value) {
        assert(m_state.load(std::memory_order_relaxed) == CacheEntryState::Requested && !m_value);
        m_value = std::move(value);
        return m_value.get();
      }

      /// Owner only: the value installed by emplace(), before it is published.
      ValueT* owned_value() const { return m_value.get(); }

      /// Called by whoever try_get() told Owned, once the value has been fully,
      /// genuinely computed (i.e. after its generating task has waited for its
      /// own kernels) -- never before.
      void publish() {
        assert(m_value && "AsyncSlot::publish() without a prior emplace()");
        m_state.store(CacheEntryState::Ready, std::memory_order_release);
      }

    private:
      std::atomic<CacheEntryState> m_state{CacheEntryState::Empty};
      std::unique_ptr<ValueT> m_value;
    };

    /**
     * One AsyncSlot per (level, displacement), for displacements within the
     * level-wide generation range [-max_op1d_displacement(n),
     * max_op1d_displacement(n)]^NDIM (anything outside it is rejected up front --
     * see GaussianConvolutionOperator::check_op_displacement). That range is
     * small and dense (at most MRA_OP1D_NUM_DISPLACEMENTS^NDIM slots per level),
     * so a lookup is plain array indexing: no hash map, no lock. Each level's
     * slot array is allocated on first use and never freed before the
     * container itself.
     */
    template <typename ValueT, Dimension NDIM>
    class LevelDisplacementSlots {
    public:
      using slot_type = AsyncSlot<ValueT>;

      LevelDisplacementSlots() = default;
      LevelDisplacementSlots(const LevelDisplacementSlots&) = delete;
      LevelDisplacementSlots& operator=(const LevelDisplacementSlots&) = delete;
      ~LevelDisplacementSlots() {
        for (auto& level : m_levels) {
          delete[] level.load(std::memory_order_acquire);
        }
      }

      slot_type& slot(Level n, const std::array<Translation, NDIM>& l) {
        assert(n >= 0 && n <= MAX_LEVEL);
        slot_type* slots = m_levels[n].load(std::memory_order_acquire);
        if (slots == nullptr) slots = allocate(n);
        return slots[index(n, l)];
      }

    private:
      static size_type index(Level n, const std::array<Translation, NDIM>& l) {
        const Translation max_disp = max_op1d_displacement(n);
        const size_type width = num_op1d_displacements(n);
        size_type idx = 0;
        for (Dimension d = 0; d < NDIM; ++d) {
          assert(l[d] >= -max_disp && l[d] <= max_disp);
          idx = idx * width + (size_type)(l[d] + max_disp);
        }
        return idx;
      }

      slot_type* allocate(Level n) {
        size_type num_slots = 1;
        for (Dimension d = 0; d < NDIM; ++d) num_slots *= num_op1d_displacements(n);
        slot_type* fresh = new slot_type[num_slots];
        slot_type* expected = nullptr;
        if (m_levels[n].compare_exchange_strong(expected, fresh, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
          return fresh;
        }
        delete[] fresh; // lost the race; use the winner's array
        return expected;
      }

      std::array<std::atomic<slot_type*>, MAX_LEVEL + 1> m_levels{};
    };

    /**
     * Per-level table holding EVERY 1D convolution operator matrix a
     * GaussianConvolutionOperator<T,NDIM> may need at level `n`: every
     * dimension, every translation l in [-max_op1d_displacement(n),
     * max_op1d_displacement(n)] -- NOT the full [-MRA_OP1D_MAX_DISPLACEMENT,
     * MRA_OP1D_MAX_DISPLACEMENT] ceiling once n is small enough that a
     * translation that large can't geometrically exist (see
     * max_op1d_displacement's comment): a level-0 table is just 1 wide, a
     * level-1 table 3 wide, a level-2 table 7 wide, every deeper level the
     * full 9. Generated by a single kernel launch (see
     * GaussianConvolutionOperator::try_get_op) instead of one tensor -- and
     * one kernel launch -- per (level, dimension, translation): there is
     * exactly one buffer per field to co_await ttg::device::select(),
     * regardless of how many displacements a task ultimately needs.
     * Consumers slice out the specific (dimension, translation) matrix they
     * need via Op1DSliceView below, which derives the same per-level width
     * from `n` (stored here) rather than needing it passed in again.
     */
    template <typename T, Dimension NDIM>
    struct LevelOp1DTable {
      Level n = 0;
      // [NDIM, num_op1d_displacements(n), count, rank, 2K, 2K]
      DenseTensor<T, 6> R;
      // [NDIM, num_op1d_displacements(n), count, rank, K, K]
      DenseTensor<T, 6> S;
      // [NDIM, num_op1d_displacements(n), count, rank, 5]
      DenseTensor<T, 5> norms1d;

      LevelOp1DTable() = default;
      LevelOp1DTable(Level n, size_type count, size_type rank, size_type K, ttg::scope scope)
      : n(n)
      , R(std::array<size_type, 6>{(size_type)NDIM, num_op1d_displacements(n), count, rank, 2 * K, 2 * K}, scope)
      , S(std::array<size_type, 6>{(size_type)NDIM, num_op1d_displacements(n), count, rank, K, K}, scope)
      // norms1d is read back on the host, so allocate its host memory up front
      , norms1d(std::array<size_type, 5>{(size_type)NDIM, num_op1d_displacements(n), count, rank, (size_type)5},
                (scope == ttg::scope::Allocate) ? ttg::scope::SyncOut : scope)
      { }
    };

    /**
     * A read-only, single-(dimension, translation) slice of a shared
     * LevelOp1DTable. Cheap enough (a pointer plus a couple of indices)
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

      // Owned by the GaussianConvolutionOperator's level table slot, which
      // outlives every task using it -- no need to keep it alive from here.
      const LevelOp1DTable<T, NDIM>* table = nullptr;
      FieldView<DenseTensor<T, 6>> R;
      FieldView<DenseTensor<T, 6>> S;
      FieldView<DenseTensor<T, 5>> norms1d;

      Op1DSliceView() = default;
      // The offset must match the per-level width THIS table was actually
      // allocated with (see LevelOp1DTable's comment), not the global ceiling.
      Op1DSliceView(const LevelOp1DTable<T, NDIM>* tbl, Dimension d, Translation l)
      : table(tbl)
      , R{&tbl->R, d, (size_type)(l + max_op1d_displacement(tbl->n))}
      , S{&tbl->S, d, (size_type)(l + max_op1d_displacement(tbl->n))}
      , norms1d{&tbl->norms1d, d, (size_type)(l + max_op1d_displacement(tbl->n))}
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
   * Functions are NOT assumed to share the same separated-expansion rank:
   * m_max_rank (the max over every function's own SeparatedConvolution::
   * get_rank()) sizes shared per-level/per-operator allocations (the table's
   * `rank` axis, m_term_params/m_shared_fac's layout), but the actual
   * generation work in collect_generation_items() only ever covers each
   * function's own real rank -- see that function's comment.
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
    // Plain pointers: tables and norms tensors are owned by this operator's
    // slots (_level_tables/_norms_slots), which are never evicted, and the
    // operator outlives every convolution task using it.
    using table_pointer_type = const detail::LevelOp1DTable<T, NDIM>*;
    // Only the assembled norms tensor is cached on the device path -- the
    // R/S views (OpResult::dims) are cheap enough (a pointer + a couple
    // of indices, see detail::Op1DSliceView) to rebuild fresh from the
    // level table on every try_get_op() call, so there's nothing else worth
    // caching per (level, displacement). What IS worth caching is the one
    // part that costs a kernel launch to assemble (see
    // submit_assemble_norms_kernel) -- exactly this tensor.
    using norms_pointer_type = const DenseTensor<T, 4>*;
    // Screener-only, host-only: per-operator Opnorm -- see try_get_opnorms().
    using opnorms_pointer_type = const std::vector<T>*;

    enum class Status { Available, Owned, Pending };

    struct OpResult {
      Status status; // Available: norms is ready; dims is also valid (as always).
                      // Owned: caller now owns assembling norms for this
                      //   specific displacement -- device path: call
                      //   make_norms_tensor(), fold its buffer into the
                      //   round's one select()
                      //   (see add_assembly_buffers()), submit_assemble_
                      //   norms_kernel(), then publish_op(). (The
                      //   screener never claims norms: it uses
                      //   try_get_level_table()/try_get_opnorms().)
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
      norms_pointer_type norms = nullptr;                                // valid iff Available
      detail::LevelOp1DTable<T, NDIM>* level_work = nullptr; // non-null iff THIS call must generate the level's table
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
     * `disp` must be within [-max_op1d_displacement(n),
     * max_op1d_displacement(n)] in every dimension; this is not expected to
     * ever trigger in practice (see MRA_OP1D_MAX_DISPLACEMENT's comment), so
     * it is a hard error rather than a fallback path.
     */
    OpResult try_get_op(Level n, Key<NDIM> disp) const {
      check_op_displacement(n, disp);

      OpResult out;
      table_pointer_type table;
      auto& level_slot = _level_tables[n];
      auto level_res = level_slot.try_get();
      if (level_res.status == detail::AsyncCacheStatus::Pending) {
        out.status = Status::Pending;
        return out;
      }
      if (level_res.status == detail::AsyncCacheStatus::Owned) {
        out.level_work = level_slot.emplace(std::make_unique<detail::LevelOp1DTable<T, NDIM>>(
            n, m_mad_conv_sep_vec.size(), m_max_rank, m_K, ttg::scope::Allocate));
        table = out.level_work; // not yet computed, but a valid object to slice views into
      } else {
        table = level_res.value; // Available
      }
      lookup_op_norms(out, table, n, disp);
      return out;
    }

    /**
     * Same as try_get_op(n, disp), but for a caller that already holds the
     * level's table -- e.g. one it generated itself and whose publication
     * (publish_generation_work()) is deferred until its task completes, so
     * the level table cache would still report it as Pending. Never sets
     * level_work.
     */
    OpResult try_get_op(Level n, Key<NDIM> disp, table_pointer_type table) const {
      check_op_displacement(n, disp);
      assert(table && table->n == n);
      OpResult out;
      lookup_op_norms(out, table, n, disp);
      return out;
    }

    /// Result of try_get_level_table(): the level's table once known, plus
    /// level_work if THIS call must generate it (same contract as
    /// OpResult::level_work).
    struct LevelTableResult {
      bool pending = false; // another task is generating the table: suspend and retry
      table_pointer_type table = nullptr;
      detail::LevelOp1DTable<T, NDIM>* level_work = nullptr;
    };

    /**
     * Screener-only level table lookup. Unlike try_get_op(), it never claims
     * the norms tensor of any displacement: those are consumed on the device
     * by shell0_tt/accumulate_tt and always assembled there (see
     * make_norms_tensor()), while the screener only needs each operator's
     * Opnorm on the host, which try_get_opnorms() keeps in a separate cache.
     * If level_work is set, the caller must generate the table
     * (collect_generation_items() ... publish_generation_work()).
     */
    LevelTableResult try_get_level_table(Level n) const {
      check_level(n);
      LevelTableResult out;
      auto& level_slot = _level_tables[n];
      auto level_res = level_slot.try_get();
      if (level_res.status == detail::AsyncCacheStatus::Pending) {
        out.pending = true;
      } else if (level_res.status == detail::AsyncCacheStatus::Owned) {
        out.level_work = level_slot.emplace(std::make_unique<detail::LevelOp1DTable<T, NDIM>>(
            n, m_mad_conv_sep_vec.size(), m_max_rank, m_K, ttg::scope::Allocate));
        out.table = out.level_work; // not yet computed
      } else {
        out.table = level_res.value;
      }
      return out;
    }

    /**
     * Screener-only: each operator's Opnorm (indexed by count-index) for
     * (n, disp), from a host-only cache kept apart from the norms tensors
     * shell0_tt/accumulate_tt select(). Were the screener to assemble those
     * tensors on the host instead, every one of them would have to be pushed
     * to the device by the first task consuming it.
     *
     * Computes the value on the spot from `table`'s norms1d if nobody has
     * yet, so the table's norms1d must be current on the host (whoever
     * generated the table pulls it back before publishing it -- see
     * publish_generation_work()). Returns nullptr if another task is
     * computing it right now: co_await ttg::device::suspend() and retry.
     *
     * Opnorm = sqrt(sum_mu MUnorm_mu^2) (MADNESS's own getop_ns() formula),
     * with MUnorm from munorm2_ns()'s formula on the generated 1D norms --
     * NOT via SeparatedConvolution::norm(), which would go through
     * getop_ns()/getmuop() to Convolution1D::nonstandard() on a cold
     * ns_cache (nothing on this path ever warms MADNESS's own cache),
     * silently forcing the host-side R/S regeneration (plus an SVD) that
     * generating R/S on-device exists to avoid.
     */
    opnorms_pointer_type try_get_opnorms(Level n, const Key<NDIM>& disp, table_pointer_type table) const {
      check_op_displacement(n, disp);
      assert(table && table->n == n);
      auto& slot = _opnorm_slots.slot(n, disp.translation());
      auto res = slot.try_get();
      if (res.status == detail::AsyncCacheStatus::Available) return res.value;
      if (res.status == detail::AsyncCacheStatus::Pending) return nullptr;

      std::array<detail::Op1DSliceView<T, NDIM>, NDIM> dims;
      for (Dimension d = 0; d < NDIM; ++d) {
        dims[d] = detail::Op1DSliceView<T, NDIM>(table, d, disp.translation()[d]);
      }
      const size_type count = m_mad_conv_sep_vec.size();
      auto opnorms = std::make_unique<std::vector<T>>(count);
      for (size_type c = 0; c < count; ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        T opnorm_sumsq = T(0);
        for (size_type i = 0; i < (size_type)mad_ops.size(); ++i) {
          double prod = 1.0, sum = 0.0;
          for (Dimension d = 0; d < NDIM; ++d) {
            auto n1d = dims[d].norms1d.view_on(ttg::device::Device::host());
            const double a = n1d(c, i, (int)NormId::NSnormf);
            const double b = n1d(c, i, (int)NormId::Snormf);
            const double aa = std::min(a, b), bb = std::max(a, b);
            prod *= bb;
            if (bb > 0.0) sum += aa / bb;
          }
          if (n) prod *= sum;
          const T munorm = prod * std::abs(mad_ops[i].getfac());
          opnorm_sumsq += munorm * munorm;
        }
        (*opnorms)[c] = std::sqrt(opnorm_sumsq);
      }
      if (validate_ops_enabled()) validate_opnorms(n, disp, *opnorms); // DEBUG
      opnorms_pointer_type result = slot.emplace(std::move(opnorms));
      slot.publish();
      return result;
    }

    /// Publishes the norms tensor for (n, disp) that make_norms_tensor() +
    /// submit_assemble_norms_kernel() installed in its slot. If op
    /// validation is enabled, the tensor must be current on the host.
    void publish_op(Level n, Key<NDIM> disp) const {
      auto& slot = norms_slot(n, disp);
      if (validate_ops_enabled()) validate_op_norms(n, disp, *slot.owned_value()); // DEBUG
      slot.publish();
    }

    /**
     * Assembles the norms tensor for shell0_tt/accumulate_tt ON-DEVICE, via two small dedicated kernels
     * (detail::assemble_op1d_norms_kernel then detail::assemble_op1d_opnorm_
     * kernel), instead of a host-side copy that would require pulling the
     * level table's norms1d back to host first -- AND, critically, instead
     * of calling MADNESS's SeparatedConvolution::norm() for Opnorm, which
     * would (see submit_assemble_norms_kernel()'s comment) silently force
     * the exact host-side R/S regeneration this whole file exists to avoid.
     * shell0_tt/accumulate_tt feed the result straight into an actual
     * device convolution kernel; screener_tt only ever reads Opnorm on the
     * host and keeps its own host-only copy (see try_get_opnorms()), so a
     * norms tensor never needs to travel host->device. R/S for the kernel
     * come straight from res.dims -- no
     * separate "assembling" object needed for those, see OpResult's
     * comment. Call sequence (see shell0_tt/accumulate_tt) -- the operator
     * buffers ride along in the task's ONE select() together with the
     * convolution's own buffers (see add_convolution_op_buffers()):
     *   // loop on try_get_op() + ttg::device::suspend() while Pending
     *   bool need_assemble = (res.status == Status::Owned);
     *   DenseTensor<T,4>* norms = nullptr;
     *   if (need_assemble) {
     *     norms = op.make_norms_tensor(n, disp);
     *   }
     *   ttg::device::Input input(...convolution buffers...);
     *   op.add_convolution_op_buffers(input, res, items_buf, norms);
     *   co_await ttg::device::select(input);
     *   if (op.has_generation_work(res)) op.submit_generation_kernel(res.level_work, items, items_buf);
     *   if (need_assemble) op.submit_assemble_norms_kernel(res, n, disp, *norms);
     *   ... submit the convolution kernel ...
     *   // if generated, pull the table's norms1d back to the host before
     *   // publishing -- try_get_opnorms() reads it there. The norms tensor
     *   // itself stays on the device (pulled back only for validation).
     *   if (op.has_generation_work(res)) co_await ttg::device::wait(res.level_work->norms1d.buffer());
     *   // on task completion:
     *   op.publish_generation_work(res.level_work); // no-op if nullptr
     *   if (need_assemble) op.publish_op(n, disp);
     */

    /// Allocates the norms tensor a device-path Owned result for (n, disp)
    /// must finish assembling -- device-resident garbage until submit_
    /// assemble_norms_kernel() fills it -- and installs it in the norms slot
    /// (owned there; the pointer stays valid as long as this operator).
    /// Call before the round's one select() -- its buffer must be added to
    /// it (see add_assembly_buffers()).
    DenseTensor<T, 4>* make_norms_tensor(Level n, const Key<NDIM>& disp) const {
      // Constructed as a named local (not forwarded straight through
      // make_unique) to avoid make_unique's perfect-forwarding picking
      // Tensor's variadic Dims... constructor overload instead of the
      // intended (array, scope) one.
      const size_type norms_count = (size_type)m_mad_conv_sep_vec.size();
      const size_type norms_rank = (size_type)m_max_rank;
      const size_type norms_ndim = (size_type)NDIM;
      const size_type norms_kinds = (size_type)NormId::Count;
      DenseTensor<T, 4> norms_tensor(std::array{norms_count, norms_rank, norms_ndim, norms_kinds},
                                      ttg::scope::Allocate);
      return norms_slot(n, disp).emplace(std::make_unique<DenseTensor<T, 4>>(std::move(norms_tensor)));
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

    /// Adds every operator buffer a device convolution task (shell0_tt/
    /// accumulate_tt) needs to `input`, for the task's ONE select():
    ///  - the R/S tables backing res.dims, each exactly once (all dimensions
    ///    slice the same level table, so adding them per dimension would
    ///    register the same data as several flows);
    ///  - if this task owns the level's table: everything
    ///    submit_generation_kernel() needs (see make_generation_input());
    ///  - if res is Owned: everything submit_assemble_norms_kernel() needs,
    ///    including the table's norms1d (see add_assembly_buffers());
    ///    `norms` must then be the tensor from make_norms_tensor();
    ///  - if res is Available: the cached norms tensor.
    /// res must not be Pending -- loop on try_get_op() and
    /// ttg::device::suspend() until it isn't, before calling this.
    void add_convolution_op_buffers(ttg::device::Input& input, const OpResult& res,
                                    ttg::Buffer<GenerateOp1DItem<T>>& items_buf,
                                    DenseTensor<T, 4>* norms) const {
      assert(res.status != Status::Pending);
      if (res.level_work) {
        // includes the table's R/S/norms1d
        add_generation_buffers(input, res.level_work, items_buf);
      } else {
        const auto& table = *res.dims[0].table;
        input.add(table.R.buffer());
        input.add(table.S.buffer());
        if (res.status == Status::Owned) {
          input.add(table.norms1d.buffer());
        }
      }
      if (res.status == Status::Owned) {
        assert(nullptr != norms);
        add_assembly_buffers(input, *norms);
      } else {
        input.add(res.norms->buffer());
      }
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
    /// generated) entirely on-device; no host round-trip.
    ///
    /// Deliberately does NOT call MADNESS's SeparatedConvolution::norm() for
    /// Opnorm (same reasoning as try_get_opnorms(), which likewise computes
    /// it from MUnorm values instead, to stay off MADNESS's own cache
    /// entirely). That call would go through
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
          count, rank, n, lx, max_op1d_displacement(n),
          table.norms1d.current_view(), m_shared_fac.current_view(),
          norms.current_view(), m_shared_rank.current_view(), ttg::device::current_stream());
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
     * generator tables, the level table's R/S/norms1d, and items_buf) into
     * that single select() call alongside it. Call-site sequence for
     * screener_tt, which needs norms1d back on the HOST afterward for
     * try_get_opnorms():
     *   auto lt = op.try_get_level_table(n);
     *   if (lt.level_work) {
     *     auto items = op.collect_generation_items(lt.level_work);
     *     auto items_buf = make_generate_op1d_items_buffer(items);
     *     auto input = op.make_generation_input(lt.level_work, items_buf);
     *     co_await ttg::device::select(input);
     *     op.submit_generation_kernel(lt.level_work, items, items_buf);
     *     co_await ttg::device::wait(lt.level_work->norms1d.buffer());
     *     // on task completion:
     *     op.publish_generation_work(lt.level_work);
     *   }
     * shell0_tt/accumulate_tt instead fold table generation and norms
     * assembly into the convolution's own select() -- see
     * add_convolution_op_buffers().
     */
    bool has_generation_work(const OpResult& res) const {
      return static_cast<bool>(res.level_work);
    }

    /// Every buffer this generation round needs -- the shared generator
    /// tables, the whole level table's R/S/norms1d (one buffer per field,
    /// not one per dimension/translation), and items_buf -- so the caller
    /// can co_await ttg::device::select() on all of it in the ONE select()
    /// call its task is allowed. Call after collect_generation_items()/
    /// make_generate_op1d_items_buffer(), which are both host-only and need
    /// no prior select() of their own.
    ttg::device::Input make_generation_input(detail::LevelOp1DTable<T, NDIM>* level_work, ttg::Buffer<GenerateOp1DItem<T>>& items_buf) const {
      ttg::device::Input input;
      add_generation_buffers(input, level_work, items_buf);
      return input;
    }

    /// Same buffers as make_generation_input(), appended to an existing
    /// `input`. Adds nothing unless level_work is set.
    void add_generation_buffers(ttg::device::Input& input, detail::LevelOp1DTable<T, NDIM>* level_work,
                                ttg::Buffer<GenerateOp1DItem<T>>& items_buf) const {
      if (!level_work) return;
      input.add(m_shared_c.buffer());
      input.add(m_shared_hgT.buffer());
      input.add(m_shared_hgT2k.buffer());
      input.add(m_shared_quadx.buffer());
      input.add(m_shared_quadw.buffer());
      input.add(level_work->R.buffer());
      input.add(level_work->S.buffer());
      input.add(level_work->norms1d.buffer());
      input.add(items_buf);
    }

    /// Builds the flat (dimension, count-index, term) work-item list
    /// covering the table in level_work -- each item generates every
    /// translation in [-max_op1d_displacement(n), max_op1d_displacement(n)]
    /// for its own (d, c, i), which is narrower than the full
    /// [-MRA_OP1D_MAX_DISPLACEMENT, MRA_OP1D_MAX_DISPLACEMENT] ceiling at low
    /// levels (see max_op1d_displacement's comment) -- so a single
    /// submit_generate_op1d_kernel call fills every matrix this level's table
    /// actually holds. The `i` loop only runs up to count-index c's own REAL
    /// separated-expansion rank (get_rank()), not the operator-wide
    /// m_max_rank every count-index's table row is allocated with -- a
    /// count-index with fewer terms than another simply gets fewer items,
    /// leaving its table's higher-i rows (and this level's norms1d there)
    /// untouched rather than generating a zero-filled placeholder for them
    /// (that used to be necessary because generate_op1d_one's coeff==0
    /// sentinel was the only way the old, uniform-m_max_rank item list could
    /// mark "nothing here"; now there's simply no item for those slots to
    /// begin with). Nothing downstream reads an untouched slot: the
    /// application kernel (mra/kernels/convolution.h) already loops only
    /// [0, real_rank) per count-index via the Rank norm, and
    /// assemble_op1d_norms_kernel/assemble_op1d_opnorm_kernel (generate_op1d.h)
    /// are bounded by m_shared_rank the same way -- see their own comments.
    /// Host-only (no device view read) -- see has_generation_work()'s comment
    /// for why that matters here -- so safe to call before the round's one
    /// select().
    std::vector<GenerateOp1DItem<T>> collect_generation_items(detail::LevelOp1DTable<T, NDIM>* level_work) const {
      std::vector<GenerateOp1DItem<T>> items;
      if (!level_work) return items;
      const size_type count = m_mad_conv_sep_vec.size();
      const size_type max_rank = (size_type)m_max_rank; // stride into m_term_params only -- see below
      const Level n = level_work->n;
      const Translation max_disp = max_op1d_displacement(n);
      items.reserve((size_type)NDIM * count * max_rank); // upper bound; real count is usually smaller once per-count-index rank is applied below
      for (Dimension d = 0; d < NDIM; ++d) {
        for (size_type c = 0; c < count; ++c) {
          const size_type rank = (size_type)m_mad_conv_sep_vec[c]->get_rank(); // this count-index's own REAL rank
          for (size_type i = 0; i < rank; ++i) {
            GenerateOp1DItem<T> item;
            // m_term_params is laid out with stride m_max_rank (see
            // extract_gaussian_terms_for_device()), regardless of this
            // count-index's own smaller real rank.
            item.params = m_term_params[(c * max_rank + i) * NDIM + d];
            item.n = n;
            item.d = d;
            item.c = c;
            item.i = i;
            item.lmin = -max_disp;
            item.lmax =  max_disp;
            items.push_back(std::move(item));
          }
        }
      }
      return items;
    }

    /// Call only after co_await-ing make_generation_input()'s select();
    /// `items`/`items_buf` must be the exact objects that went
    /// into that same select() call (via make_generation_input).
    void submit_generation_kernel(detail::LevelOp1DTable<T, NDIM>* level_work, const std::vector<GenerateOp1DItem<T>>& items,
                                   ttg::Buffer<GenerateOp1DItem<T>>& items_buf) const {
      if (items.empty()) return;
      auto& table = *level_work;
      const Translation max_disp = max_op1d_displacement(table.n);
      submit_generate_op1d_kernel<T>(
          items_buf.current_device_ptr(), items.size(), num_op1d_displacements(table.n), m_K,
          m_shared_c.current_view(), m_shared_hgT.current_view(), m_shared_hgT2k.current_view(),
          m_shared_quadx.current_view().data(), m_shared_quadw.current_view().data(), m_npt,
          max_disp,
          table.R.current_view(), table.S.current_view(), table.norms1d.current_view(),
          ttg::device::current_stream());
    }

    /// Call only once the generating task has completed (e.g. via
    /// ttg::device::on_complete()), after pulling the table's norms1d back to
    /// the host (co_await ttg::device::wait(); R/S stay on the device, where
    /// they are consumed -- other devices stage them in device-to-device, which
    /// requires peer access between the GPUs): until the task
    /// completes the runtime still holds its written device copies, which
    /// other tasks must not stage in. Publishes the whole
    /// table as Available so every other task at this level -- any
    /// dimension, any translation -- finds it immediately.
    void publish_generation_work(detail::LevelOp1DTable<T, NDIM>* level_work) const {
      if (!level_work) return;
      if (validate_ops_enabled()) validate_level_table(*level_work); // DEBUG
      _level_tables[level_work->n].publish();
    }
#endif // !MRA_ENABLE_HOST

    /// DEBUG: MRA_VALIDATE_OPS=1 checks every published table/norms tensor
    /// against MADNESS (expensive -- forces MADNESS to build its own ops).
    static bool validate_ops_enabled() {
      static const bool enabled = [] {
        const char* v = std::getenv("MRA_VALIDATE_OPS");
        return v != nullptr && *v != '\0' && std::string(v) != "0";
      }();
      return enabled;
    }

  private:
#ifndef MRA_ENABLE_HOST
    static void check_level(Level n) {
      if (n < 0 || n > MAX_LEVEL) {
        throw std::runtime_error("GaussianConvolutionOperator::try_get_op: level out of range");
      }
    }

    static void check_op_displacement(Level n, const Key<NDIM>& disp) {
      check_level(n);
      const Translation max_disp = max_op1d_displacement(n);
      for (Dimension d = 0; d < NDIM; ++d) {
        Translation l = disp.translation()[d];
        if (l < -max_disp || l > max_disp) {
          throw std::runtime_error(
              "GaussianConvolutionOperator::try_get_op: translation out of the "
              "level-wide generation range for this level (see max_op1d_displacement)");
        }
      }
    }

    /// The norms slot for (n, disp); disp must have passed check_op_displacement().
    detail::AsyncSlot<DenseTensor<T, 4>>& norms_slot(Level n, const Key<NDIM>& disp) const {
      return _norms_slots.slot(n, disp.translation());
    }

    /**
     * DEBUG (MRA_VALIDATE_OPS=1): compare device-generated operator data
     * against MADNESS's own Convolution1D::nonstandard() / SeparatedConvolution
     * data, i.e., what the host path (get_op()) would have produced. Called at
     * publish time, when the data is current on the host. Prints the largest
     * absolute and relative deviation per level (table) or per (level,
     * displacement) (norms). Expensive -- forces MADNESS to build its own
     * nonstandard ops.
     */
    struct ValidationError {
      double max_abs = 0.0, max_rel = 0.0, ref_at_max = 0.0;
      std::size_t nbad = 0, ntotal = 0;
      std::string where;
      static bool is_bad(double val, double ref) {
        return !(std::abs(val - ref) <= 1e-10 + 1e-8 * std::abs(ref)); // also catches NaN
      }
      template <typename Loc>
      bool update(double val, double ref, Loc&& loc) {
        const double abs = std::abs(val - ref);
        const double rel = abs / std::max(std::abs(ref), 1e-300);
        ++ntotal;
        const bool bad = is_bad(val, ref);
        if (bad) ++nbad;
        if (abs > max_abs || std::isnan(abs)) { max_abs = abs; ref_at_max = ref; where = loc(); }
        if (std::abs(ref) > 1e-12) max_rel = std::max(max_rel, rel);
        return bad;
      }
    };

    void validate_level_table(const detail::LevelOp1DTable<T, NDIM>& table) const {
      const Level n = table.n;
      const size_type count = m_mad_conv_sep_vec.size();
      const size_type rank = (size_type)m_max_rank;
      const size_type K = m_K, K2 = 2 * m_K;
      const Translation max_disp = max_op1d_displacement(n);
      const size_type num_disp = num_op1d_displacements(n);
      // R/S normally stay on the device (only norms1d is pulled back to the
      // host before publishing); check them only if they happen to be there.
      // norms1d is computed on the device from the same R/S, so it still
      // catches wrong matrices.
      const bool have_RS = table.R.buffer().is_current_on(ttg::device::Device::host()) &&
                           table.S.buffer().is_current_on(ttg::device::Device::host());
      const T* R  = have_RS ? table.R.view_on(ttg::device::Device::host()).data() : nullptr;
      const T* S  = have_RS ? table.S.view_on(ttg::device::Device::host()).data() : nullptr;
      const T* N1 = table.norms1d.view_on(ttg::device::Device::host()).data();
      ValidationError eR, eS, eN;
      // per (c,i): number of bad entries over all d, l and fields, and how many
      // of the generated values are exactly zero
      std::vector<std::size_t> bad_ci(count * rank, 0), zero_ci(count * rank, 0), total_ci(count * rank, 0);
      for (Dimension d = 0; d < NDIM; ++d) {
        for (size_type li = 0; li < num_disp; ++li) {
          const Translation l = (Translation)li - max_disp;
          for (size_type c = 0; c < count; ++c) {
            auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
            // Only i < mad_ops.size() (this count-index's REAL rank) was ever
            // generated -- see collect_generation_items()'s comment; higher i
            // is intentionally untouched (not zero), so there's nothing
            // meaningful to compare it against.
            for (size_type i = 0; i < mad_ops.size(); ++i) {
              const size_type entry = ((d * num_disp + li) * count + c) * rank + i;
              const T* r = have_RS ? R + entry * K2 * K2 : nullptr;
              const T* s = have_RS ? S + entry * K * K : nullptr;
              const T* nn = N1 + entry * 5;
              const madness::ConvolutionData1D<T>* cd = nullptr;
              if (i < mad_ops.size()) cd = mad_ops[i].getop(d)->nonstandard(n, l);
              auto loc = [&](const char* what, size_type k) {
                return [=]() {
                  std::ostringstream oss;
                  oss << what << "(d=" << d << ",l=" << l << ",c=" << c << ",i=" << i << ",k=" << k << ")";
                  return oss.str();
                };
              };
              const bool haveR = cd && cd->R.size() != 0;
              const bool haveT = cd && cd->T.size() != 0;
              auto& nbad = bad_ci[c * rank + i];
              auto& nzero = zero_ci[c * rank + i];
              auto& ntot = total_ci[c * rank + i];
              for (size_type k = 0; have_RS && k < K2 * K2; ++k) {
                nbad += eR.update(r[k], haveR ? cd->R.ptr()[k] : T(0), loc("R", k));
                nzero += (r[k] == T(0)); ++ntot;
              }
              for (size_type k = 0; have_RS && k < K * K; ++k) {
                nbad += eS.update(s[k], haveT ? cd->T.ptr()[k] : T(0), loc("S", k));
                nzero += (s[k] == T(0)); ++ntot;
              }
              const T refn[5] = {cd ? cd->Rnorm : T(0), cd ? cd->Tnorm : T(0), cd ? cd->Rnormf : T(0),
                                 cd ? cd->Tnormf : T(0), cd ? cd->NSnormf : T(0)};
              for (size_type k = 0; k < 5; ++k) {
                nbad += eN.update(nn[k], refn[k], loc("norms1d", k));
                nzero += (nn[k] == T(0)); ++ntot;
              }
            }
          }
        }
      }
      static std::mutex mtx;
      std::lock_guard<std::mutex> lg(mtx);
      std::cout << "MRA_VALIDATE_OPS table n=" << n << std::scientific << std::setprecision(3)
                << " | R bad " << eR.nbad << "/" << eR.ntotal << " max_abs " << eR.max_abs << " at " << eR.where << " (ref " << eR.ref_at_max << ")"
                << " | S bad " << eS.nbad << "/" << eS.ntotal << " max_abs " << eS.max_abs << " at " << eS.where << " (ref " << eS.ref_at_max << ")"
                << " | norms1d bad " << eN.nbad << "/" << eN.ntotal << " max_abs " << eN.max_abs << " at " << eN.where << " (ref " << eN.ref_at_max << ")"
                << std::defaultfloat << std::endl;
      // list the (c,i) terms with any bad entry: c:i(bad/zero/total)
      std::cout << "MRA_VALIDATE_OPS table n=" << n << " bad terms (c:i bad/zero/total):";
      std::size_t nlisted = 0;
      for (size_type c = 0; c < count; ++c) {
        for (size_type i = 0; i < rank; ++i) {
          const auto idx = c * rank + i;
          if (bad_ci[idx] == 0) continue;
          if (nlisted++ < 40) {
            std::cout << " " << c << ":" << i << "(" << bad_ci[idx] << "/" << zero_ci[idx] << "/" << total_ci[idx] << ")";
          }
        }
      }
      std::cout << " [" << nlisted << " terms of " << count * rank << "]" << std::endl;
    }

    void validate_op_norms(Level n, const Key<NDIM>& disp, const DenseTensor<T, 4>& norms) const {
      auto nv = norms.view_on(ttg::device::Device::host());
      const T* got = nv.data();
      const size_type count = m_mad_conv_sep_vec.size();
      const size_type rank = (size_type)m_max_rank;
      const size_type nk = (size_type)NormId::Count;
      auto at = [&](size_type c, size_type i, size_type d, NormId id) {
        return got[((c * rank + i) * NDIM + d) * nk + (size_type)id];
      };
      const char* names[] = {"Rnorm", "Snorm", "Rnormf", "Snormf", "NSnormf", "Fac", "MUnorm", "Opnorm", "Rank"};
      std::array<ValidationError, (size_type)NormId::Count> err;
      const madness::Key<NDIM> mkey = Key<NDIM>(0, n, disp.translation()).to_madness_key();
      for (size_type c = 0; c < count; ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        for (size_type i = 0; i < mad_ops.size(); ++i) {
          // reference 1D norms, and MUnorm via munorm2_ns's formula on them
          double prod = 1.0, sum = 0.0;
          for (Dimension d = 0; d < NDIM; ++d) {
            auto cd = mad_ops[i].getop(d)->nonstandard(n, disp.translation()[d]);
            const T ref[5] = {cd->Rnorm, cd->Tnorm, cd->Rnormf, cd->Tnormf, cd->NSnormf};
            for (size_type k = 0; k < 5; ++k) {
              err[k].update(at(c, i, d, (NormId)k), ref[k], [&] {
                std::ostringstream oss; oss << "(c=" << c << ",i=" << i << ",d=" << d << ")"; return oss.str(); });
            }
            const double a = cd->NSnormf, b = cd->Tnormf;
            const double aa = std::min(a, b), bb = std::max(a, b);
            prod *= bb;
            if (bb > 0.0) sum += aa / bb;
          }
          if (n) prod *= sum;
          const double fac = mad_ops[i].getfac();
          auto ci_loc = [&] { std::ostringstream oss; oss << "(c=" << c << ",i=" << i << ")"; return oss.str(); };
          err[(size_type)NormId::Fac].update(at(c, i, 0, NormId::Fac), fac, ci_loc);
          err[(size_type)NormId::MUnorm].update(at(c, i, 0, NormId::MUnorm), prod * std::abs(fac), ci_loc);
        }
        auto c_loc = [&] { std::ostringstream oss; oss << "(c=" << c << ")"; return oss.str(); };
        err[(size_type)NormId::Opnorm].update(at(c, 0, 0, NormId::Opnorm), m_mad_conv_sep_vec[c]->norm(n, mkey, mkey), c_loc);
        err[(size_type)NormId::Rank].update(at(c, 0, 0, NormId::Rank), (T)mad_ops.size(), c_loc);
      }
      static std::mutex mtx;
      std::lock_guard<std::mutex> lg(mtx);
      std::cout << "MRA_VALIDATE_OPS norms n=" << n << " disp=[";
      for (Dimension d = 0; d < NDIM; ++d) std::cout << disp.translation()[d] << (d + 1 < NDIM ? "," : "");
      std::cout << "]" << std::scientific << std::setprecision(3);
      for (size_type k = 0; k < nk; ++k) {
        std::cout << " | " << names[k] << " bad " << err[k].nbad << "/" << err[k].ntotal << " max " << err[k].max_abs << "/" << err[k].max_rel;
        if (err[k].max_abs > 0) std::cout << " at " << err[k].where << " (ref " << err[k].ref_at_max << ")";
      }
      std::cout << std::defaultfloat << std::endl;
    }

    void validate_opnorms(Level n, const Key<NDIM>& disp, const std::vector<T>& opnorms) const {
      ValidationError err;
      const madness::Key<NDIM> mkey = Key<NDIM>(0, n, disp.translation()).to_madness_key();
      for (size_type c = 0; c < opnorms.size(); ++c) {
        err.update(opnorms[c], m_mad_conv_sep_vec[c]->norm(n, mkey, mkey),
                   [&] { std::ostringstream oss; oss << "(c=" << c << ")"; return oss.str(); });
      }
      static std::mutex mtx;
      std::lock_guard<std::mutex> lg(mtx);
      std::cout << "MRA_VALIDATE_OPS opnorms n=" << n << " disp=[";
      for (Dimension d = 0; d < NDIM; ++d) std::cout << disp.translation()[d] << (d + 1 < NDIM ? "," : "");
      std::cout << "]" << std::scientific << std::setprecision(3)
                << " | Opnorm bad " << err.nbad << "/" << err.ntotal << " max " << err.max_abs << "/" << err.max_rel;
      if (err.max_abs > 0) std::cout << " at " << err.where << " (ref " << err.ref_at_max << ")";
      std::cout << std::defaultfloat << std::endl;
    }

    /// Slices out.dims from `table` and checks/claims the norms cache for
    /// (n, disp) -- the part of try_get_op() after the level table is known.
    void lookup_op_norms(OpResult& out, table_pointer_type table, Level n, const Key<NDIM>& disp) const {
      for (Dimension d = 0; d < NDIM; ++d) {
        out.dims[d] = detail::Op1DSliceView<T, NDIM>(table, d, disp.translation()[d]);
      }
      auto norms_res = norms_slot(n, disp).try_get();
      if (norms_res.status == detail::AsyncCacheStatus::Available) {
        out.status = Status::Available;
        out.norms = norms_res.value;
        return;
      }
      if (norms_res.status == detail::AsyncCacheStatus::Pending) {
        out.status = Status::Pending;
        return;
      }
      out.status = Status::Owned;
    }
#endif // !MRA_ENABLE_HOST

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
    // see detail::AsyncSlot's class comment and try_get_op() above. Both are
    // plain arrays indexed by level (and displacement), owning their values
    // for the operator's lifetime, so lookups take no lock and touch no
    // reference count.
    // Exactly one caller per level generates the whole
    // detail::LevelOp1DTable (every dimension, every translation in
    // [-max_op1d_displacement(n), max_op1d_displacement(n)]) in a single
    // kernel launch; every other displacement at that level just reads it.
    mutable std::array<detail::AsyncSlot<detail::LevelOp1DTable<T, NDIM>>, MAX_LEVEL + 1> _level_tables;
    // Caches only the assembled norms tensor per (level, displacement) --
    // not a whole aggregate -- since R/S (OpResult::dims) are cheap enough
    // to rebuild from the level table on every try_get_op() call; see
    // OpResult's comment.
    mutable detail::LevelDisplacementSlots<DenseTensor<T, 4>, NDIM> _norms_slots;
    // Screener-only, host-only Opnorm per (level, displacement) -- kept apart
    // from _norms_slots so nothing host-assembled is ever a device input; see
    // try_get_opnorms().
    mutable detail::LevelDisplacementSlots<std::vector<T>, NDIM> _opnorm_slots;

    // Shared (not per-term) tensors extracted once, at construction, from
    // MADNESS's Convolution1D base -- identical for every term/dimension of
    // this operator since they depend only on K and npt, never on
    // (coeff, expnt). See extract_gaussian_terms_for_device().
    size_type m_K = 0;
    size_type m_npt = 0;
    DenseTensor<T, 3> m_shared_c;      // 4K x K x K autocorrelation tensor (k outermost -- see extract_gaussian_terms_for_device())
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

      // Stored as (4K, K, K) rather than MADNESS's own (K, K, 4K) -- i.e.
      // transposed so the contraction axis (k, summed over in
      // make_rnlij_device's cten(k,qi,qj)*Rvec[k]) is the OUTERMOST/slowest
      // dimension and (qi,qj) are the fastest-varying/contiguous ones. This
      // is a one-time, host-side transpose (this whole extraction runs once
      // per operator construction), done so that on-device, consecutive
      // threads working on consecutive (qi,qj) output elements for the same
      // k read consecutive addresses -- coalesced -- instead of addresses
      // 4K elements apart under MADNESS's native layout.
      m_shared_c = DenseTensor<T, 3>(std::array{4 * K, K, K}, ttg::scope::SyncIn);
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
            c_view(k, i, j) = static_cast<T>(first->c(i, j, k));
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
      m_shared_fac = DenseTensor<T, 2>(std::array<size_type, 2>{(size_type)m_mad_conv_sep_vec.size(), (size_type)m_max_rank},
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

      m_shared_rank = DenseTensor<T, 1>((size_type)m_mad_conv_sep_vec.size(), ttg::scope::SyncIn);
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
    /// get_op()'s host-path aggregate and its validation can share it.
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
