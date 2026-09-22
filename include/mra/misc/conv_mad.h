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

  template <typename T>
  struct ConvolutionData1D {

    // 4D: count x rank x [R|S] x 2D operator matrix
    // count should either be 1 or the number of functions to which the operators are applied
    using tensor_type = DenseTensor<T, 4>;

    /**
     * We store R and S in separate tensors because they have different dimensions (2K and K).
     */
    tensor_type R, S;

    /**
     * [count, rank, 5] -- Rnorm,Snorm,Rnormf,Snormf,NSnormf per (count-index, term),
     * for this dimension only. On the MADNESS/host path these are read straight off
     * cd_mad; on the device path they are computed once, alongside R/S, by the
     * on-device generator (mra/kernels/generate_op1d.h) and copied into the
     * aggregate ConvolutionData<T,NDIM>::norms tensor when assembling it -- see
     * GaussianConvolutionOperator::try_get_op. Unused (default-constructed, empty)
     * on the host path.
     */
    DenseTensor<T, 3> norms1d;

    ConvolutionData1D() : R(), S(){}
    ConvolutionData1D(size_type count, size_type rank, size_type K)
    : R(std::array{count, rank, 2*K, 2*K}, ttg::scope::SyncIn)
    , S(std::array{count, rank, K, K}, ttg::scope::SyncIn)
    { }
#ifndef MRA_ENABLE_HOST
    /// Device-path constructor: R/S allocated device-resident (no host fill), plus norms1d.
    ConvolutionData1D(size_type count, size_type rank, size_type K, ttg::scope scope)
    : R(std::array{count, rank, 2*K, 2*K}, scope)
    , S(std::array{count, rank, K, K}, scope)
    , norms1d(std::array{count, rank, (size_type)5}, scope)
    { }
#endif // !MRA_ENABLE_HOST
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
     * Non-blocking, tri-state, memoizing cache -- the device-path replacement for
     * SharedComputeCache/Op1DCache's claim()+acquire() above. There is no acquire()
     * at all: try_get() always returns immediately with one of:
     *   Available -- value is ready, here it is.
     *   Owned     -- the caller just became responsible for producing the value
     *                (allocating the destination tensor(s), submitting the
     *                generation kernel, and calling publish() once that kernel has
     *                been awaited to completion -- see GaussianConvolutionOperator::
     *                try_get_op1d/try_get_op). Ownership, once granted, cannot be
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
#endif // !MRA_ENABLE_HOST

  } // namespace detail

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
    /// see try_get_op1d()'s Op1DGenerateWork comment for the sequence the
    /// caller must run these through (select() before reading their views).
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
          norms_view(c, i, 0, (int)NormId::MUnorm) = munorm2_ns(c, n, i, data) * std::abs(fac);
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
    using op1d_pointer_type = std::shared_ptr<const ConvolutionData1D<T>>;
    using agg_pointer_type = std::shared_ptr<const ConvolutionData<T, NDIM>>;

    enum class Status { Available, Owned, Pending };

    /// What try_get_op1d() hands back when it returns Owned: the freshly
    /// allocated (device-resident, uninitialized -- ttg::scope::Allocate)
    /// tensor, plus enough to later build its generation work items. The
    /// tensor's R/S/norms1d buffers have no valid current_view() yet (Allocate
    /// scope defers actual device allocation to the runtime's own callback,
    /// triggered by select()), so building the per-(count-index,term) views
    /// happens separately, in build_op1d_items(), AFTER select() -- not here.
    /// The caller must, in its OWN device-task coroutine (this can't be
    /// hidden in a helper coroutine of its own -- ttg::device::Task is driven
    /// directly by the backend, not composable via nested co_await; see this
    /// class's header comment):
    ///   1. co_await ttg::device::select(work.tensor->R.buffer(),
    ///      work.tensor->S.buffer(), work.tensor->norms1d.buffer()) (plus
    ///      whatever else it's already selecting) to get them resident on the
    ///      current device,
    ///   2. build_op1d_items(work) (now current_view() is valid), batched
    ///      together with any other dimensions' items this same call also
    ///      owns (see try_get_op) into one submit_generate_op1d_kernel call,
    ///   3. co_await ttg::device::wait() (NO buffer argument -- this only
    ///      needs to know the kernel *this task* just submitted has finished,
    ///      not to force an unwanted device->host transfer),
    ///   4. publish_op1d(work, std::move(work.tensor)).
    struct Op1DGenerateWork {
      std::shared_ptr<ConvolutionData1D<T>> tensor;
      Dimension d = 0;
      Level n = 0;
      Translation lx = 0;
    };

    struct Op1DResult {
      Status status;
      op1d_pointer_type value; // valid iff Available
      Op1DGenerateWork work;   // valid iff Owned
    };

    /**
     * Non-blocking query for the (n,d,l) 1D transition-matrix tensor -- the
     * device-path replacement for the claim()+acquire() pair inside get_op()
     * above. See detail::AsyncCache's class comment for what each of the
     * three outcomes means and what the caller must do about it. Safe to call
     * from ordinary (non-coroutine) code.
     */
    Op1DResult try_get_op1d(Level n, Dimension d, Translation l) const {
      auto res = _op1d_cache_async.try_get(detail::Op1DKey(n, d, l));
      if (res.status == detail::AsyncCacheStatus::Available) {
        return {Status::Available, res.value, {}};
      }
      if (res.status == detail::AsyncCacheStatus::Pending) {
        return {Status::Pending, nullptr, {}};
      }
      // Owned: just allocate the destination tensor (host-side bookkeeping
      // only -- Allocate scope does not touch the device here). Building the
      // per-item views happens later, in build_op1d_items(), after select().
      auto tensor = std::make_shared<ConvolutionData1D<T>>(m_mad_conv_sep_vec.size(), m_max_rank, m_K,
                                                             ttg::scope::Allocate);
      Op1DResult out;
      out.status = Status::Owned;
      out.work.tensor = std::move(tensor);
      out.work.d = d;
      out.work.n = n;
      out.work.lx = l;
      return out;
    }

    /// Builds the flat (count-index, term) work-item list for `work.tensor`.
    /// Call ONLY after co_await ttg::device::select()-ing work.tensor's
    /// R/S/norms1d buffers -- see Op1DGenerateWork's comment; current_view()
    /// has no valid data pointer before that. Padding entries (i beyond a
    /// given count-index's real rank) get a default-constructed (coeff==0)
    /// GaussianTermParams -- generate_op1d_one's own coeff==0 sentinel check
    /// zero-fills them on-device, so unlike make_op1d_shell there is no
    /// separate host-side zero-fill loop needed here.
    std::vector<GenerateOp1DItem<T>> build_op1d_items(const Op1DGenerateWork& work) const {
      auto rv = work.tensor->R.current_view();
      auto sv = work.tensor->S.current_view();
      auto nv = work.tensor->norms1d.current_view();
      std::vector<GenerateOp1DItem<T>> items;
      items.reserve(m_mad_conv_sep_vec.size() * (size_type)m_max_rank);
      for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        for (size_type i = 0; i < (size_type)m_max_rank; ++i) {
          GenerateOp1DItem<T> item;
          item.params = m_term_params[(c * (size_type)m_max_rank + i) * NDIM + work.d];
          item.n = work.n;
          item.lx = work.lx;
          item.R_dst = rv(c, i);
          item.S_dst = sv(c, i);
          item.norms_dst = nv(c, i);
          items.push_back(std::move(item));
        }
      }
      return items;
    }

    /// Publishes the tensor generated for work's (n,d,l) -- call only after
    /// the generation kernel submitted over build_op1d_items(work) has been
    /// co_await-ed to completion (co_await ttg::device::wait(), no argument).
    /// See Op1DGenerateWork's comment for the full sequence.
    void publish_op1d(const Op1DGenerateWork& work) const {
      _op1d_cache_async.publish(detail::Op1DKey(work.n, work.d, work.lx),
                                 op1d_pointer_type(work.tensor));
    }

    struct OpResult {
      Status status; // Available: value is ready.
                      // Owned: all NDIM dims were already Available; caller
                      //   now owns the AGGREGATE itself -- call
                      //   finish_op_assembly() then publish_op().
                      // Pending: at least one dim is not yet ready this round
                      //   (either genuinely owned by an unrelated task, or
                      //   just claimed by THIS call -- dim_work entries below
                      //   cover the latter). Caller must submit whatever
                      //   dim_work entries are present regardless of status,
                      //   then (if status == Pending) co_await
                      //   ttg::device::wait() once and call try_get_op()
                      //   again. This may yield once even when every
                      //   outstanding dim was actually owned by this same
                      //   call (no unrelated task involved) rather than
                      //   re-checking inline -- a deliberate simplification,
                      //   not a correctness issue: generation is fast, so one
                      //   extra reschedule round-trip is cheap, and it avoids
                      //   the caller needing to distinguish "why" a dim
                      //   wasn't ready.
      agg_pointer_type value;                        // valid iff Available
      std::array<Op1DGenerateWork, NDIM> dim_work{};  // non-null .tensor for dims THIS call just claimed
      std::array<op1d_pointer_type, NDIM> dims{};     // populated iff status != Pending
    };

    /// Non-blocking query for the full (n, displacement) aggregate operator
    /// data. See OpResult's comment for the contract.
    OpResult try_get_op(Level n, Key<NDIM> disp) const {
      OpResult out;
      bool any_not_ready = false;
      for (Dimension d = 0; d < NDIM; ++d) {
        auto r1 = try_get_op1d(n, d, disp.translation()[d]);
        if (r1.status == Status::Available) {
          out.dims[d] = r1.value;
        } else if (r1.status == Status::Owned) {
          out.dim_work[d] = std::move(r1.work);
          any_not_ready = true;
        } else {
          any_not_ready = true;
        }
      }
      if (any_not_ready) {
        out.status = Status::Pending;
        return out;
      }
      auto key = Key<NDIM>(0, n, disp.translation());
      auto agg = _datacache_async.try_get(key);
      if (agg.status == detail::AsyncCacheStatus::Available) {
        out.status = Status::Available;
        out.value = agg.value;
        return out;
      }
      if (agg.status == detail::AsyncCacheStatus::Pending) {
        out.status = Status::Pending;
        return out;
      }
      out.status = Status::Owned;
      return out;
    }

    /**
     * Assembles the aggregate ConvolutionData once every dimension is
     * Available (i.e. after a try_get_op() call returned Owned). Caller must
     * have already done co_await ttg::device::wait(dims[d]->norms1d.buffer())
     * for every d (a small device->host transfer of the 5-float-per-(c,i)
     * norms1d tensor) before calling this, so the host view read below is
     * valid -- this stays a plain (non-coroutine) function since only the
     * caller's own device-task body can co_await (see this class's header
     * comment).
     *
     * Mirrors get_op()'s norms-assembly loop above, but sources
     * Rnorm/Snorm/Rnormf/Snormf/NSnormf from each dimension's own norms1d
     * (already computed once, on-device, by the generation kernel) instead
     * of a second, redundant MADNESS nonstandard() call. Fac is likewise
     * already known (mad_ops[i].getfac(), extracted once at construction --
     * see extract_gaussian_terms()) rather than re-read here; kept as the
     * mad_ops[i].getfac() call below only to stay obviously-identical to
     * get_op()'s existing padding-loop structure.
     *
     * Opnorm/Rank still call into MADNESS's SeparatedConvolution::norm() --
     * a genuinely different, much smaller (one scalar per (c, displacement))
     * computation than the R/S/Rnorm/Snorm/Rnormf/Snormf/NSnormf generation
     * this file moved off the CPU; porting it is out of scope here.
     */
    agg_pointer_type finish_op_assembly(Level n, Key<NDIM> disp,
                                         const std::array<op1d_pointer_type, NDIM>& dims) const {
      auto data = std::make_shared<ConvolutionData<T, NDIM>>(m_mad_conv_sep_vec.size(), m_max_rank);
      for (Dimension d = 0; d < NDIM; ++d) data->data[d] = dims[d];
      auto norms_view = data->norms.view_on(ttg::device::Device::host());
      for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        size_type i = 0;
        for (; i < (size_type)mad_ops.size(); ++i) {
          for (Dimension d = 0; d < NDIM; ++d) {
            auto n1d = data->data[d]->norms1d.view_on(ttg::device::Device::host());
            norms_view(c, i, d, (int)NormId::Rnorm) = n1d(c, i, 0);
            norms_view(c, i, d, (int)NormId::Snorm) = n1d(c, i, 1);
            norms_view(c, i, d, (int)NormId::Rnormf) = n1d(c, i, 2);
            norms_view(c, i, d, (int)NormId::Snormf) = n1d(c, i, 3);
            norms_view(c, i, d, (int)NormId::NSnormf) = n1d(c, i, 4);
          }
          auto fac = mad_ops[i].getfac();
          norms_view(c, i, 0, (int)NormId::Fac) = fac;
          norms_view(c, i, 0, (int)NormId::MUnorm) = munorm2_ns(c, n, i, data) * std::abs(fac);
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
        T norm = m_mad_conv_sep_vec[c]->norm(n, disp.to_madness_key(), disp.to_madness_key());
        norms_view(c, 0, 0, (int)NormId::Opnorm) = norm;
        norms_view(c, 0, 0, (int)NormId::Rank) = mad_ops.size();
      }
      return data;
    }

    /// Publishes the aggregate assembled by finish_op_assembly() for (n, disp).
    void publish_op(Level n, Key<NDIM> disp, agg_pointer_type data) const {
      auto key = Key<NDIM>(0, n, disp.translation());
      _datacache_async.publish(key, std::move(data));
    }

    /**
     * The non-coroutine steps every call site's generation round needs,
     * factored here so tasks/convolution.h's three call sites only have to
     * interleave co_await calls around them, not duplicate the bookkeeping.
     * Typical call-site sequence (see e.g. shell0_tt):
     *   if (op.has_generation_work(res)) {
     *     auto input = op.make_generation_input(res);
     *     co_await ttg::device::select(input);
     *     auto items = op.collect_generation_items(res);
     *     auto items_buf = make_generate_op1d_items_buffer(items);
     *     auto ws_buf = op.make_generation_workspace(items.size());
     *     co_await ttg::device::select(items_buf, ws_buf);
     *     op.submit_generation_kernel(items, items_buf, ws_buf);
     *     co_await ttg::device::wait();
     *     op.publish_generation_work(res);
     *   }
     */
    bool has_generation_work(const OpResult& res) const {
      for (Dimension d = 0; d < NDIM; ++d) {
        if (res.dim_work[d].tensor) return true;
      }
      return false;
    }

    /// Buffers to co_await ttg::device::select() before reading any view into
    /// a newly-owned dimension's tensor or the shared generator tables.
    ttg::device::Input make_generation_input(const OpResult& res) const {
      ttg::device::Input input;
      input.add(m_shared_c.buffer());
      input.add(m_shared_hgT.buffer());
      input.add(m_shared_hgT2k.buffer());
      input.add(m_shared_quadx.buffer());
      input.add(m_shared_quadw.buffer());
      for (Dimension d = 0; d < NDIM; ++d) {
        if (res.dim_work[d].tensor) {
          input.add(res.dim_work[d].tensor->R.buffer());
          input.add(res.dim_work[d].tensor->S.buffer());
          input.add(res.dim_work[d].tensor->norms1d.buffer());
        }
      }
      return input;
    }

    /// Call only after co_await-ing make_generation_input()'s select().
    std::vector<GenerateOp1DItem<T>> collect_generation_items(const OpResult& res) const {
      std::vector<GenerateOp1DItem<T>> all_items;
      for (Dimension d = 0; d < NDIM; ++d) {
        if (res.dim_work[d].tensor) {
          auto items = build_op1d_items(res.dim_work[d]);
          all_items.insert(all_items.end(), items.begin(), items.end());
        }
      }
      return all_items;
    }

    /// Allocates (Allocate scope -- device-resident, no host fill needed) the
    /// per-item cooperative scratch the generation kernel's threads share
    /// (see generate_op1d.h's Op1DWorkspaceOffsets/generate_op1d_workspace_size).
    /// Must be co_await ttg::device::select()-ed (alongside items_buf) before
    /// submit_generation_kernel reads its device pointer.
    ttg::Buffer<T> make_generation_workspace(size_type n_items) const {
      return ttg::Buffer<T>(generate_op1d_workspace_size(m_K) * n_items, ttg::scope::Allocate);
    }

    /// Call only after co_await-ing items_buf's and workspace's own select();
    /// `items` must be the exact list items_buf was built from
    /// (make_generate_op1d_items_buffer), and `workspace` must be sized by
    /// make_generation_workspace(items.size()).
    void submit_generation_kernel(const std::vector<GenerateOp1DItem<T>>& items,
                                   ttg::Buffer<GenerateOp1DItem<T>>& items_buf,
                                   ttg::Buffer<T>& workspace) const {
      if (items.empty()) return;
      submit_generate_op1d_kernel<T>(
          items_buf.current_device_ptr(), items.size(), m_K,
          m_shared_c.current_view(), m_shared_hgT.current_view(), m_shared_hgT2k.current_view(),
          m_shared_quadx.current_view().data(), m_shared_quadw.current_view().data(), m_npt,
          workspace.current_device_ptr(),
          ttg::device::current_stream());
    }

    /// Call only after co_await-ing the generation kernel to completion
    /// (co_await ttg::device::wait(), no argument).
    void publish_generation_work(const OpResult& res) const {
      for (Dimension d = 0; d < NDIM; ++d) {
        if (res.dim_work[d].tensor) publish_op1d(res.dim_work[d]);
      }
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
    // see detail::AsyncCache's class comment and try_get_op1d()/try_get_op() above.
    mutable detail::AsyncCache<detail::Op1DKey, ConvolutionData1D<T>> _op1d_cache_async;
    mutable detail::AsyncCache<Key<NDIM>, ConvolutionData<T, NDIM>> _datacache_async;

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
      for (size_type c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        for (size_type i = 0; i < (size_type)mad_ops.size(); ++i) {
          T fac = static_cast<T>(mad_ops[i].getfac());
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
    double munorm2_ns(size_type c, Level n, size_type mu, const std::shared_ptr<const ConvolutionData<T, NDIM>>& data) const {

        double prodR=1.0;
        double prod=1.0, sum=0.0;
        auto norms_view = data->norms.view_on(ttg::device::Device::host());
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
