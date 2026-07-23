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
        return Ticket{std::move(cell), owner};
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
      };

      /// Claims (get-or-creates) the entry for `key`. `creator == true` means this call
      /// must follow up with init() before anyone can contribute() or acquire().
      Ticket claim(const Op1DKey& key) const {
        cellptr_type cell = get_or_create_cell(key);
        Op1DCellState expected = Op1DCellState::Empty;
        bool creator = cell->state.compare_exchange_strong(expected, Op1DCellState::Initializing,
                                                             std::memory_order_acq_rel,
                                                             std::memory_order_acquire);
        return Ticket{std::move(cell), creator};
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
    { }

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
    }

    /**
     * Assembles ConvolutionData for the level and displacement.
     */
    std::shared_ptr<const ConvolutionData<T, NDIM>> get_op(Level n, Key<NDIM> disp) const {
      auto key = Key<NDIM>(0, n, disp.translation());
      auto agg_ticket = _datacache.claim(key);
      if (!agg_ticket.owner) {
        // Someone else is already computing (or has finished) this exact aggregate;
        // nothing to split further for an identical request, so just wait for it.
        return _datacache.acquire(agg_ticket);
      }
      /**
       * We are responsible for computing the aggregate data for this Level/displacement.
       * We generate the data out of MADNESS and store our own version of it.
       * Start with assembling the ConvolutionData1D for each dimension. The 1D data is
       * cached so it may be reused across displacements/dimensions.
       *
       * This happens in three passes over the NDIM dimensions rather than one:
       *  1. claim (get-or-create) every per-dimension cache entry, and let whichever
       *     call is first to claim a given (level, dimension, translation) allocate its
       *     tensor and the flat list of (function, term) work items;
       *  2. contribute to every dimension we hold a ticket for -- grabbing and computing
       *     whatever work items nobody else has claimed yet;
       *  3. acquire the (by then likely-finished) result for each dimension, blocking
       *     only on whatever items are still in flight.
       * Splitting this way means that when several tasks need overlapping (or the same)
       * (level, dimension, translation) 1D tensors concurrently, they all pitch in on the
       * (function, term) loop that fills each tensor instead of one task running that
       * whole loop alone while the others just sit in a spin-wait on the result.
       */
      auto data = std::make_shared<ConvolutionData<T, NDIM>>(m_mad_conv_sep_vec.size(), m_max_rank);
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
