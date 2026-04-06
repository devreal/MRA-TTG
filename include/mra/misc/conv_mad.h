#ifndef CONV_MAD_H
#define CONV_MAD_H

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
    Count
  };

  template <typename T>
  struct ConvolutionData1D {
#if 0
    // 4D: rank x [R, RU, RVT] x 2D operator matrix
    Tensor<T, 4> R, S;
#endif // 0

    // 3D: rank x [R] x 2D operator matrix
    Tensor<T, 3> R, S;

    ConvolutionData1D() : R(), S(){}
    ConvolutionData1D(size_type rank, size_type K)
    : R(std::array{rank, 2*K, 2*K}, ttg::scope::SyncIn)
    , S(std::array{rank, K, K}, ttg::scope::SyncIn)
    { }
    ConvolutionData1D(Tensor<T, 3>&& R_,
                      Tensor<T, 3>&& S_)
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
    // 3D: rank x NDIM x [Rnorm, Snorm, Rnormf, Snormf, NSnormf]
    //     fac & munorm of each separated term is stored in the same tensor, at dim 0
    Tensor<T, 3> norms;
    T norm;

    ConvolutionData(size_type rank)
    : data()
    , norms(std::array{rank, NDIM, (size_type)NormId::Count}, ttg::scope::SyncIn)
    , norm(-1.0)
    { }

  };

  template <typename T, Dimension NDIM>
  class GaussianConvolutionOperator {

  public:

    /**
     * Construct a convolution operator
     */
    GaussianConvolutionOperator(madness::SeparatedConvolution<T, NDIM>& mad_conv_sep)
    : mad_conv_sep(mad_conv_sep)
    { }

    /**
     * Assembles ConvolutionData for the level and displacement.
     */
    std::shared_ptr<const ConvolutionData<T, NDIM>> get_op(Level n, Key<NDIM> disp) const {
      cachemutex.lock();
      auto key = Key<NDIM>(n, disp.translation());
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
      auto data = std::make_shared<ConvolutionData<T, NDIM>>(mad_conv_sep.get_rank());
      for (int d = 0; d < NDIM; ++d) {
        auto key_1d = std::make_pair(n, disp.translation()[d]);
        auto it = _opcache.find(key_1d);
        if (it == _opcache.end()) {
          cachemutex.unlock();
          // compute new data
          auto data = make_op1d(n, disp.translation()[d], d);
          cachemutex.lock();
          // check if someone else generated this data
          if (_opcache.find(key_1d) == _opcache.end()) {
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
      auto& mad_ops = mad_conv_sep.get_ops();
      auto norms_view = data->norms.view_on(ttg::device::Device::host());
      for (int i = 0; i < mad_ops.size(); ++i) {
        for (int d = 0; d < NDIM; ++d) {
          auto cd_mad = mad_ops[i].getop(d)->nonstandard(n, disp.translation()[d]);
          norms_view(i, d, (int)NormId::Rnorm) = cd_mad->Rnorm;
          norms_view(i, d, (int)NormId::Snorm) = cd_mad->Tnorm;
          norms_view(i, d, (int)NormId::Rnormf) = cd_mad->Rnormf;
          norms_view(i, d, (int)NormId::Snormf) = cd_mad->Tnormf;
          norms_view(i, d, (int)NormId::NSnormf) = cd_mad->NSnormf;
        }
        norms_view(i, 0, (int)NormId::Fac) = mad_ops[i].getfac();
        norms_view(i, 0, (int)NormId::MUnorm) = munorm2_ns(n, i, data);
      }
      /* Finally, store the norm of the whole operator */
      T norm = mad_conv_sep.norm(n, disp.to_madness_key(), disp.to_madness_key());
      data->norm = norm;
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
    // convolution1d madness object
    //madness::GaussianConvolution1D<double> conv1d;
    // madness separate convolution object, provided by application
    madness::SeparatedConvolution<T, NDIM>& mad_conv_sep;
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
     */
    std::shared_ptr<const ConvolutionData1D<T>> make_op1d(Level n, Translation l, Dimension d) const {

      auto& mad_ops = mad_conv_sep.get_ops();
      auto data = std::make_shared<ConvolutionData1D<T>>(mad_ops.size(), mad_conv_sep.get_k());
      auto rv = data->R.view_on(ttg::device::Device::host());
      auto sv = data->S.view_on(ttg::device::Device::host());
      for (int i = 0; i < mad_ops.size(); ++i) {
        const madness::ConvolutionData1D<T>* cd_mad;
        std::shared_ptr<const madness::Convolution1D<T> > conv1d = mad_ops[i].getop(d);
        cd_mad = conv1d->nonstandard(n, l);
        if (!(cd_mad->R.size() == 0 && cd_mad->T.size() == 0)) {
          copy_from_madtensor(rv(i, 0), cd_mad->R);
          //copy_from_madtensor(rv(i, 1), cd_mad->RU);
          //copy_from_madtensor(rv(i, 2), cd_mad->RVT);
          copy_from_madtensor(sv(i, 0), cd_mad->T); // S = T for us
          //copy_from_madtensor(sv(i, 1), cd_mad->TU);
          //copy_from_madtensor(sv(i, 2), cd_mad->TVT);
        }
      }
      return data;
    }


    /// Taken from MADNESS, since munorm2_ns is private in SeparatedConvolution
    /// and we have no way to get the norm otherwise.
    /// Computes the Frobenius norm of one of the separated terms for the NS form
    ///       ... WITHOUT FACTOR INCLUDED
    /// compute for 1 term, all dim, 1 disp, essentially for SeparatedConvolutionInternal
    double munorm2_ns(Level n, size_type mu, const std::shared_ptr<const ConvolutionData<T, NDIM>>& data) const {

        double prodR=1.0;
        double prod=1.0, sum=0.0;
        auto norms_view = data->norms.view_on(ttg::device::Device::host());
        for (std::size_t d=0; d<NDIM; ++d) {
            double a = norms_view(mu, d, (int)NormId::NSnormf);
            double b = norms_view(mu, d, (int)NormId::Snormf);
            double aa = std::min(a,b);
            double bb = std::max(a,b);
            prod *= bb;
            if (bb > 0.0) sum +=(aa/bb);
        }
        if (n) prod *= sum;
        prodR = prod;

        return prodR;
    }


#if 0
    std::shared_ptr<const ConvolutionData<T, NDIM>> make_op(Level n, Key<NDIM> disp) const {

      // call madness nonstandard function to populate GaussianConvolutionData for each dimension
      std::array<std::shared_ptr<const GaussianConvolutionData<T>>, NDIM> ops;

      size_type K = mad_conv_sep.get_k();

      assert(mad_conv_sep.get_ops().size() == 1); // TODO: FIXME

      //const madness::ConvolutionData1D<T>* cd_mad[NDIM];
      for (size_type i = 0; i < NDIM; ++i) {
        const madness::ConvolutionData1D<T>* cd_mad;
        std::shared_ptr<const madness::Convolution1D<T> > conv1d = mad_conv_sep.get_ops()[0].getop(i);
        cd_mad = conv1d->nonstandard(n, disp.translation()[i]);
        //cd_mad[i] = conv1d.nonstandard(n, disp.translation()[i]);
        if (!(cd_mad->R.size() == 0 && cd_mad->T.size() == 0)) {
          GaussianConvolutionData<T>  op_data;
          // op_data.Rnorm = cd_mad[i]->Rnorm;
          // op_data.Snorm = cd_mad[i]->Tnorm;
          // op_data.Rnormf = cd_mad[i]->Rnormf;
          // op_data.Snormf = cd_mad[i]->Tnormf;
          // op_data.NSnormf = cd_mad[i]->NSnormf;

          // op_data.R    = Tensor<T, 2>(2 * op_info.K, 2 * op_info.K);
          // op_data.RU   = Tensor<T, 2>(2 * op_info.K, 2 * op_info.K);
          // op_data.RVT  = Tensor<T, 2>(2 * op_info.K, 2 * op_info.K);
          // op_data.S    = Tensor<T, 2>(op_info.K, op_info.K);
          // op_data.SU   = Tensor<T, 2>(op_info.K, op_info.K);
          // op_data.SVT  = Tensor<T, 2>(op_info.K, op_info.K);
          // op_data.Rs   = Tensor<T, 1>(2 * op_info.K);
          // op_data.Ss   = Tensor<T, 1>(op_info.K);

          Tensor<T, 2> R(2 * K, 2 * K),
                        RU(2 * K, 2 * K),
                        RVT(2 * K, 2 * K),
                        S(K, K),
                        SU(K, K),
                        SVT(K, K);
          Tensor<T, 1> Rs(2 * K), Ss(K);
          auto R_view = R.view_on(ttg::device::Device::host());
          auto RU_view = RU.view_on(ttg::device::Device::host());
          auto RVT_view = RVT.view_on(ttg::device::Device::host());
          auto S_view = S.view_on(ttg::device::Device::host());
          auto SU_view = SU.view_on(ttg::device::Device::host());
          auto SVT_view = SVT.view_on(ttg::device::Device::host());
          auto Rs_view = Rs.view_on(ttg::device::Device::host());
          auto Ss_view = Ss.view_on(ttg::device::Device::host());

          for (size_type j=0; j<2*K; ++j){
            for (size_type k=0; k<2*K; ++k){
              R_view(j,k) = static_cast<T>(cd_mad->R(j,k));
              RU_view(j,k) = static_cast<T>(cd_mad->RU(j,k));
              RVT_view(j,k) = static_cast<T>(cd_mad->RVT(j,k));
            }
          }

          for (size_type j=0; j<K; ++j){
            for (size_type k=0; k<K; ++k){
              S_view(j,k) = static_cast<T>(cd_mad->T(j,k));
              SU_view(j,k) = static_cast<T>(cd_mad->TU(j,k));
              SVT_view(j,k) = static_cast<T>(cd_mad->TVT(j,k));
            }
          }

          for (size_type j=0; j<2*K; ++j){
            Rs_view(j) = static_cast<T>(cd_mad->Rs[j]);
          }

          for (size_type j=0; j<K; ++j){
            Ss_view(j) = static_cast<T>(cd_mad->Ts[j]);
          }
          ops[i] = std::make_shared<const GaussianConvolutionData<T>>(std::move(R), std::move(S),
                                                                      std::move(RU), std::move(RVT),
                                                                      std::move(SU), std::move(SVT),
                                                                      std::move(Rs), std::move(Ss),
                                                                      static_cast<T>(cd_mad->Rnorm),
                                                                      static_cast<T>(cd_mad->Tnorm),
                                                                      static_cast<T>(cd_mad->Rnormf),
                                                                      static_cast<T>(cd_mad->Tnormf),
                                                                      static_cast<T>(cd_mad->NSnormf),
                                                                      static_cast<T>(cd_mad->fac));
        }
        else {
          ops[i] = std::make_shared<const GaussianConvolutionData<T>>();
        }
      }
      //T norm = norm_ns(n, ops);
      T norm = mad_conv_sep.norm(n, disp.to_madness_key(), disp.to_madness_key());
      GaussianOperatorData<T, NDIM> ops_data;
      ops_data.ops = ops;
      ops_data.norm = norm;
      ops_data.fac = mad_conv_sep.get_ops()[0].getfac(); // TODO: FIXME

      cachemutex.lock();
      // check again if another thread has already populated the cache while we were computing
      it = _opcache.find(disp);
      if (it == _opcache.end()) {
        const auto result = std::make_shared<const GaussianOperatorData<T, NDIM>>(std::move(ops_data));
        _opcache.emplace(disp, std::move(result));
      }
      it = _opcache.find(disp);
      cachemutex.unlock();
      auto& r = it->second;
      return r;
    }
#endif // 0
  };

} // namespace mra

#endif // CONV_MAD_H
