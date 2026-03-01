#ifndef CONV_MAD_H
#define CONV_MAD_H

#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/mra/operator.h>
#include <madness/mra/convolution1d.h>
#include "mra/misc/types.h"

namespace mra {

  /**
   * A weird class that stores parameters for the operator and makes sure that madness is initialized once.
   */
  template <typename T>
  struct OperatorInfo {
    size_type K;
    T expnt;
    T coeff;

    void init_madness() {
      static std::once_flag flag;
      std::call_once(flag, []() {
        int argc = 0;
        char** argv = nullptr;
        madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
        madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);
        madness::World& world = madness::World::get_default();
        madness::startup(world, 0, nullptr, false);
      });
    }

    OperatorInfo(size_type K, T expnt, T coeff)
    : K(K), expnt(expnt), coeff(coeff)
    {
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
    GaussianConvolutionData(Tensor<T, 2>&& R_, Tensor<T, 2>&& S_,
                              Tensor<T, 2>&& RU_, Tensor<T, 2>&& RVT_,
                              Tensor<T, 2>&& SU_, Tensor<T, 2>&& SVT_,
                              Tensor<T, 1>&& Rs_, Tensor<T, 1>&& Ss_,
                              T Rnorm_, T Snorm_, T Rnormf_, T Snormf_, T NSnormf_)
      : R(std::move(R_)), S(std::move(S_)),
        RU(std::move(RU_)), RVT(std::move(RVT_)),
        SU(std::move(SU_)), SVT(std::move(SVT_)),
        Rs(std::move(Rs_)), Ss(std::move(Ss_)),
        Rnorm(Rnorm_), Snorm(Snorm_),
        Rnormf(Rnormf_), Snormf(Snormf_), NSnormf(NSnormf_) {}
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

    GaussianConvolutionOperator(size_type K, T expnt, T coeff)
    : op_info(K, expnt, coeff)
    , conv1d(K, coeff, expnt, 0, false)
    { }

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

      const madness::ConvolutionData1D<T>* cd_mad[NDIM];
      for (size_type i = 0; i < NDIM; ++i) {
        cd_mad[i] = conv1d.nonstandard(n, disp.translation()[i]);
        if (!(cd_mad[i]->R.size() == 0 && cd_mad[i]->T.size() == 0)) {
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

          Tensor<T, 2> R(2 * op_info.K, 2 * op_info.K),
                        RU(2 * op_info.K, 2 * op_info.K),
                        RVT(2 * op_info.K, 2 * op_info.K),
                        S(op_info.K, op_info.K),
                        SU(op_info.K, op_info.K),
                        SVT(op_info.K, op_info.K);
          Tensor<T, 1> Rs(2 * op_info.K), Ss(op_info.K);
          auto R_view = R.view_on(ttg::device::Device::host());
          auto RU_view = RU.view_on(ttg::device::Device::host());
          auto RVT_view = RVT.view_on(ttg::device::Device::host());
          auto S_view = S.view_on(ttg::device::Device::host());
          auto SU_view = SU.view_on(ttg::device::Device::host());
          auto SVT_view = SVT.view_on(ttg::device::Device::host());
          auto Rs_view = Rs.view_on(ttg::device::Device::host());
          auto Ss_view = Ss.view_on(ttg::device::Device::host());

          for (size_type j=0; j<2*op_info.K; ++j){
            for (size_type k=0; k<2*op_info.K; ++k){
              R_view(j,k) = static_cast<T>(cd_mad[i]->R(j,k));
              RU_view(j,k) = static_cast<T>(cd_mad[i]->RU(j,k));
              RVT_view(j,k) = static_cast<T>(cd_mad[i]->RVT(j,k));
            }
          }

          for (size_type j=0; j<op_info.K; ++j){
            for (size_type k=0; k<op_info.K; ++k){
              S_view(j,k) = static_cast<T>(cd_mad[i]->T(j,k));
              SU_view(j,k) = static_cast<T>(cd_mad[i]->TU(j,k));
              SVT_view(j,k) = static_cast<T>(cd_mad[i]->TVT(j,k));
            }
          }

          for (size_type j=0; j<2*op_info.K; ++j){
            Rs_view(j) = static_cast<T>(cd_mad[i]->Rs[j]);
          }

          for (size_type j=0; j<op_info.K; ++j){
            Ss_view(j) = static_cast<T>(cd_mad[i]->Ts[j]);
          }
          ops[i] = std::make_shared<const GaussianConvolutionData<T>>(std::move(R), std::move(S),
                                                                      std::move(RU), std::move(RVT),
                                                                      std::move(SU), std::move(SVT),
                                                                      std::move(Rs), std::move(Ss),
                                                                      static_cast<T>(cd_mad[i]->Rnorm),
                                                                      static_cast<T>(cd_mad[i]->Tnorm),
                                                                      static_cast<T>(cd_mad[i]->Rnormf),
                                                                      static_cast<T>(cd_mad[i]->Tnormf),
                                                                      static_cast<T>(cd_mad[i]->NSnormf));
        }
        else {
          ops[i] = std::make_shared<const GaussianConvolutionData<T>>();
        }
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
