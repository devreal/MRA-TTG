#ifndef CONV_MAD_H
#define CONV_MAD_H

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
      cachemutex.lock();
      auto key = Key<NDIM>(0, n, disp.translation());
      auto it = _datacache.find(key);
      if (it != _datacache.end()) {
        cachemutex.unlock();
        return it->second;
      }
      /**
       * First time looking for this Level/displacement.
       * We generate the data out of MADNESS and store our own version of it.
       * Start with assembling the ConvolutionData1D for each dimension.
       * The 1D data is cached so we might reuse if from other displacements.
       */
      auto data = std::make_shared<ConvolutionData<T, NDIM>>(m_mad_conv_sep_vec.size(), m_max_rank);
      for (int d = 0; d < NDIM; ++d) {
        auto key_1d = std::make_pair(n, disp.translation()[d]);
        auto it = _opcache.find(key_1d);
        if (it == _opcache.end()) {
          cachemutex.unlock();
          // compute new data
          auto data = make_op1d(n, disp.translation()[d], d);
          cachemutex.lock();
          // check if someone else generated this data
          it = _opcache.find(key_1d);
          if (it == _opcache.end()) {
            auto [it_, inserted] = _opcache.insert(std::make_pair(key_1d, std::move(data)));
            it = it_;
          }
        }
        assert(it != _opcache.end());
        data->data[d] = it->second;
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
          norms_view(c, i, 0, (int)NormId::Fac) = mad_ops[i].getfac();
          norms_view(c, i, 0, (int)NormId::MUnorm) = munorm2_ns(c, n, i, data);
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
      }
      it = _datacache.find(key);
      if (it != _datacache.end()) {
        cachemutex.unlock();
        return it->second;
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
    mutable std::map<std::pair<Level, Translation>, std::shared_ptr<const ConvolutionData1D<T>>> _opcache;
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
