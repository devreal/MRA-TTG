#ifndef CONV_MAD_H
#define CONV_MAD_H

#include <atomic>
#include <memory>
#include <array>
#include <mutex>
#include <map>
#include <utility>

#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/mra/operator.h>
#include <madness/mra/convolution1d.h>
#include "mra/misc/types.h"

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

  } // namespace detail

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
   * Cache for ConvolutionData1D objects. The cache is indexed by the displacement, which is a signed integer.
   * The cache is used to avoid recomputing the ConvolutionData1D for the same displacement multiple times.
   * The cache is thread-safe and uses atomic shared pointers to manage the lifetime of
   */
  template <typename T, int MaxDistance>
  struct ConvolutionData1DCache {

    static_assert(MaxDistance > 0, "ConvolutionData1DCache: MaxDistance must be positive");

    using pointer_type = std::shared_ptr<const ConvolutionData1D<T>>;
    using atomic_pointer_type = detail::atomic_shared_ptr<pointer_type>;

  private:
    std::array<std::array<atomic_pointer_type, 2 * MaxDistance + 1>, MAX_LEVEL> m_data; // indexed by displacement + MaxDistance

  public:
    ConvolutionData1DCache() : m_data() { }
    ~ConvolutionData1DCache() = default;

    bool has_data(int level, int displacement) const {
      if (level > MAX_LEVEL || level < 0 || displacement < -MaxDistance || displacement > MaxDistance) {
        throw std::out_of_range("ConvolutionData1DCache: displacement or level out of range");
      }
      return !!m_data[level][displacement + MaxDistance].load(std::memory_order_relaxed);
    }

    pointer_type get_data(int level, int displacement) const {
      if (level > MAX_LEVEL || level < 0 || displacement < -MaxDistance || displacement > MaxDistance) {
        throw std::out_of_range("ConvolutionData1DCache: displacement or level out of range");
      }
      return m_data[level][displacement + MaxDistance].load(std::memory_order_acquire);
    }

    /**
     * Returns true if the new data was set, false if the data was already set by another thread.
     * The data is set atomically using compare_exchange_strong, so if another thread has already
     * set the data, this function will return false and the new data will be discarded.
     */
    bool set_data(int level, int displacement, pointer_type data) {
      if (level > MAX_LEVEL || level < 0 || displacement < -MaxDistance || displacement > MaxDistance) {
        throw std::out_of_range("ConvolutionData1DCache: displacement or level out of range");
      }
      pointer_type expected = nullptr;
      return m_data[level][displacement + MaxDistance].compare_exchange_strong(expected, std::move(data),
                                                                               std::memory_order_release,
                                                                               std::memory_order_relaxed);
    }
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
      cachemutex.lock();
      auto key = Key<NDIM>(0, n, disp.translation());
      auto it = _datacache.find(key);
      if (it != _datacache.end()) {
        auto& data = it->second;
        cachemutex.unlock();
        return data;
      }
      cachemutex.unlock();
      /**
       * First time looking for this Level/displacement.
       * We generate the data out of MADNESS and store our own version of it.
       * Start with assembling the ConvolutionData1D for each dimension.
       * The 1D data is cached so we might reuse if from other displacements.
       */
      auto data = std::make_shared<ConvolutionData<T, NDIM>>(m_mad_conv_sep_vec.size(), m_max_rank);
      for (int d = 0; d < NDIM; ++d) {
        if (!_op1d_cache.has_data(n, disp.translation()[d])) {
          // compute new data
          auto data = make_op1d(n, disp.translation()[d], d);
          // try to set the data in the cache, may be discarded if another thread already set it
          _op1d_cache.set_data(n, disp.translation()[d], std::move(data));
        }
        assert(_op1d_cache.has_data(n, disp.translation()[d])
              && "ConvolutionData1DCache should have data after make_op1d");
        data->data[d] = _op1d_cache.get_data(n, disp.translation()[d]);
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
      cachemutex.lock();
      it = _datacache.find(key);
      if (it != _datacache.end()) {
        auto& data = it->second;
        cachemutex.unlock();
        return data;
      }
      // insert new
      _datacache.insert(std::make_pair(key, data));
      cachemutex.unlock();
      return data;
    }

  private:
    // madness separate convolution object, provided by application
    std::vector<std::shared_ptr<madness::SeparatedConvolution<T, NDIM>>> m_mad_conv_sep_vec;
    int m_max_rank = 0;
    // our own cache of full operator data for each [Level, Translation] (encoded as Key)
    // includes all terms and dimensions
    mutable ConvolutionData1DCache<T, 4> _op1d_cache; // MADNESS uses [-4,4] as the maximum distance for screening
    mutable std::map<Key<NDIM>, std::shared_ptr<const ConvolutionData<T, NDIM>>> _datacache;
    mutable std::mutex cachemutex;

    template<typename TV>
    void copy_from_madtensor(TV&& tv, const madness::Tensor<T>& m) const {
      assert(tv.size() == m.size());
      for (size_type i = 0; i < m.size(); ++i) {
        tv[i] = m.ptr()[i];
      }
    }

    /**
     * Assembles ConvolutionData1D for the level and displacement, for the given dimension.
     * Note that the same 1D data may be shared across multiple dimensions and/or terms,
     * depending on what MADNESS provides.
     * This function does not modify the cache.
     * Assumes all convolution objects have the same K.
     */
    std::shared_ptr<const ConvolutionData1D<T>> make_op1d(Level n, Translation l, Dimension d) const {
      auto data = std::make_shared<ConvolutionData1D<T>>(m_mad_conv_sep_vec.size(), m_max_rank, m_mad_conv_sep_vec.front()->get_k());
      auto rv = data->R.view_on(ttg::device::Device::host());
      auto sv = data->S.view_on(ttg::device::Device::host());
      for (int c = 0; c < m_mad_conv_sep_vec.size(); ++c) {
        auto& mad_ops = m_mad_conv_sep_vec[c]->get_ops();
        int i = 0;
        for (i = 0; i < mad_ops.size(); ++i) {
          const madness::ConvolutionData1D<T>* cd_mad;
          std::shared_ptr<const madness::Convolution1D<T> > conv1d = mad_ops[i].getop(d);
          cd_mad = conv1d->nonstandard(n, l);
          if (!(cd_mad->R.size() == 0 && cd_mad->T.size() == 0)) {
            copy_from_madtensor(rv(c, i), cd_mad->R);
            //copy_from_madtensor(rv(i, 1), cd_mad->RU);
            //copy_from_madtensor(rv(i, 2), cd_mad->RVT);
            copy_from_madtensor(sv(c, i), cd_mad->T); // S = T for us
            //copy_from_madtensor(sv(i, 1), cd_mad->TU);
            //copy_from_madtensor(sv(i, 2), cd_mad->TVT);
          }
        }
        // fill in the rest of the tensors with zeros
        for (; i < m_max_rank; ++i) {
          rv(c, i) = 0.0;
          sv(c, i) = 0.0;
        }
      }
      return data;
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
