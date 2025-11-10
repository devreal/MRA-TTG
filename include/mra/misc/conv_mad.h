#ifndef CONV_MAD_H
#define CONV_MAD_H

#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/mra/operator.h>
#include <madness/mra/convolution1d.h>
#include "mra/misc/types.h"

namespace mra {

  template <typename T>
  struct OperatorInfo {
    size_type K;
    T expnt;
    T coeff;

    void init_madness() {
      madness::World& world = madness::World::get_default();
      madness::startup(world, 0, nullptr, false);
    }
    OperatorInfo() : K(8), expnt(1500), coeff(1) {
      init_madness();
    }
    OperatorInfo(size_type K, T expnt, T coeff) : K(K), expnt(expnt), coeff(coeff) {
      init_madness();
    }
  };

  template <typename T>
  struct GaussianConvolutionData {
    Tensor<T, 2> R, S;
    Tensor<T, 2> RU, RVT, SU, SVT;
    Tensor<T, 1> Rs, Ss; // singular values of R and S matrix

    T Rnorm, Snorm, Rnormf, Snormf, NSnormf;

    GaussianConvolutionData() : R(), S(), RU(), RVT(), SU(), SVT(),
                                Rs(), Ss(),
                                Rnorm(0.0), Snorm(0.0),
                                Rnormf(0.0), Snormf(0.0), NSnormf(0.0) {}
    GaussianConvolutionData(const GaussianConvolutionData&) = default;
    GaussianConvolutionData(GaussianConvolutionData&&) = default;
    ~GaussianConvolutionData() = default;
  };

  template <typename T, Dimension NDIM>
  struct GaussianOperatorData {
    std::array<std::shared_ptr<const GaussianConvolutionData<T>>, NDIM> ops;
    T norm;
    T fac;
    GaussianOperatorData() : ops{}, norm(0.0), fac(1.0) {}
    GaussianOperatorData(const GaussianOperatorData&) = default;
    GaussianOperatorData(GaussianOperatorData&&) = default;
    ~GaussianOperatorData() = default;
  };

  template <typename T, Dimension NDIM>
  class GaussianConvolutionOperator {
  public:
    OperatorInfo<T> op_info;

    GaussianConvolutionOperator() : op_info(), conv1d(op_info.K, op_info.coeff, op_info.expnt, 0, false) {
      // startup(madness::World::get_default(), 0, nullptr, false);
    }

    GaussianConvolutionOperator(const OperatorInfo<T>& op_info) : op_info(op_info),
                                              conv1d(op_info.K, op_info.coeff, op_info.expnt, 0, false) {
      // startup(madness::World::get_default(), 0, nullptr, false);
    }

    std::shared_ptr<const GaussianOperatorData<T, NDIM>> get_op(Level n, Key<NDIM> disp) const {
      cachemutex.lock();
      auto it = _opcache.find(disp);
      cachemutex.unlock();
      if (it != _opcache.end()) {
        return it->second;
      }

      return make_op(n, disp);
    }
  private:
    // convolution1d madness object
    madness::GaussianConvolution1D<double> conv1d;
    mutable std::map<Key<NDIM>, std::shared_ptr<const GaussianOperatorData<T, NDIM>>> _opcache;
    mutable std::mutex cachemutex;

    T norm_ns(Level n, std::array<std::shared_ptr<const GaussianConvolutionData<T>>, NDIM>& ns) const {
      T prodR = 1.0, prodS = 1.0;
      for (size_type i = 0; i < NDIM; ++i) {
        prodR *= ns[i]->Rnormf;
        prodS *= ns[i]->Snormf;
      }

      T prod = 1.0, sum = 0.0;
      for (size_type i = 0; i < NDIM; ++i) {
        T a = ns[i]->NSnormf;
        T b = ns[i]->Snormf;
        T aa = std::min(a, b);
        T bb = std::max(a, b);
        prod *= bb;
        if (bb > 0) sum += aa / bb;
      }

      if (n) prod*=sum;
      prodR *= prod;
      return prodR;
    }

    std::shared_ptr<const GaussianOperatorData<T, NDIM>> make_op(Level n, Key<NDIM> disp) const {
      // call madness nonstandard function to populate GaussianConvolutionData for each dimension
      std::array<std::shared_ptr<const GaussianConvolutionData<T>>, NDIM> ops;

      for (size_type i = 0; i < NDIM; ++i) {
        const madness::ConvolutionData1D<T>* cd_mad = conv1d.nonstandard(n, disp.translation()[i]);
        GaussianConvolutionData<T>  op_data;
        op_data.Rnorm = cd_mad->Rnorm;
        op_data.Snorm = cd_mad->Tnorm;
        op_data.Rnormf = cd_mad->Rnormf;
        op_data.Snormf = cd_mad->Tnormf;
        op_data.NSnormf = cd_mad->NSnormf;

        op_data.R    = Tensor<T, 2>(2 * op_info.K, 2 * op_info.K);
        op_data.RU   = Tensor<T, 2>(2 * op_info.K, 2 * op_info.K);
        op_data.RVT  = Tensor<T, 2>(2 * op_info.K, 2 * op_info.K);
        op_data.S    = Tensor<T, 2>(op_info.K, op_info.K);
        op_data.SU   = Tensor<T, 2>(op_info.K, op_info.K);
        op_data.SVT  = Tensor<T, 2>(op_info.K, op_info.K);
        op_data.Rs   = Tensor<T, 1>(2 * op_info.K);
        op_data.Ss   = Tensor<T, 1>(op_info.K);

        auto R_view = op_data.R.current_view();
        auto RU_view = op_data.RU.current_view();
        auto RVT_view = op_data.RVT.current_view();
        auto S_view = op_data.S.current_view();
        auto SU_view = op_data.SU.current_view();
        auto SVT_view = op_data.SVT.current_view();
        auto Rs_view = op_data.Rs.current_view();
        auto Ss_view = op_data.Ss.current_view();

        for (size_type j=0; j<2*op_info.K; ++j){
          for (size_type k=0; k<2*op_info.K; ++k){
            R_view(j,k) = static_cast<T>(cd_mad->R(j,k));
            RU_view(j,k) = static_cast<T>(cd_mad->RU(j,k));
            RVT_view(j,k) = static_cast<T>(cd_mad->RVT(j,k));
          }
        }

        for (size_type j=0; j<op_info.K; ++j){
          for (size_type k=0; k<op_info.K; ++k){
            S_view(j,k) = static_cast<T>(cd_mad->T(j,k));
            SU_view(j,k) = static_cast<T>(cd_mad->TU(j,k));
            SVT_view(j,k) = static_cast<T>(cd_mad->TVT(j,k));
          }
        }

        for (size_type j=0; j<2*op_info.K; ++j){
          Rs_view(j) = static_cast<T>(cd_mad->Rs[j]);
        }

        for (size_type j=0; j<op_info.K; ++j){
          Ss_view(j) = static_cast<T>(cd_mad->Ts[j]);
        }
        ops[i] = std::make_shared<const GaussianConvolutionData<T>>(std::move(op_data));
      }

      T norm = norm_ns(n, ops);
      GaussianOperatorData<T, NDIM> ops_data;
      ops_data.ops = ops;
      ops_data.norm = norm;
      ops_data.fac = 1.0;

      cachemutex.lock();
      if (_opcache.find(disp) == _opcache.end()) {
        const auto result = std::make_shared<const GaussianOperatorData<T, NDIM>>(std::move(ops_data));
        _opcache.emplace(disp, std::move(result));
      }
      auto it = _opcache.find(disp);
      cachemutex.unlock();
      auto& r = it->second;
      return r;
      }
    };

} // namespace mra

#endif // CONV_MAD_H
