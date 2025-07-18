#ifndef MRA_CONVOLUTIONDATA_H
#define MRA_CONVOLUTIONDATA_H

#include "mra/ops/inner.h"
#include "mra/misc/gl.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/types.h"
#include "mra/misc/adquad.h"
#include "mra/misc/twoscale.h"
#include "mra/misc/platform.h"
#include "mra/misc/autocorr.h"
#include "mra/tensor/tensorview.h"

#define MRA_MAX_LX 7
namespace mra {

  template <typename T>
  struct ConvolutionData {
    Tensor<T, 2> R, S;
  };

  template <typename T, Dimension NDIM>
  struct OperatorData {
    std::array<const ConvolutionData<T>*, NDIM> ops;
    T norm;
    T fac;

    OperatorData() : ops{}, norm(0.0), fac(1.0) {
      for (int i = 0; i < NDIM; ++i) {
        ops[i] = nullptr;
      }
    }
    OperatorData(const OperatorData& op) {
      norm = op.norm;
      fac = op.fac;
      for (int i = 0; i < NDIM; ++i) {
        if (op.ops[i]) {
          ops[i] = op.ops[i];
        } else {
          ops[i] = nullptr;
        }
      }
    }
    ~OperatorData() = default;
  };

  template <typename T, Dimension NDIM>
  class Convolution {

    private:
      size_type K;
      int npt;                                         // number of quadrature points
      T expnt;                                         // exponent for the Gaussian
      T coeff;                                         // coefficient for the Gaussian
      const T* quad_x;                                 // quadrature points
      const T* quad_w;                                 // quadrature weights
      Tensor<T, 3> c;                                  // autocorrelation coefficients
      FunctionData<T, NDIM>& functiondata;             // function data
      std::map<Key<NDIM>, Tensor<T, 2>> rnlijcache;    // map for storing rnlij matrices
      std::map<Key<NDIM>, Tensor<T, 1>> rnlpcache;     // map for storing rnlp matrices
      std::map<Key<NDIM>, ConvolutionData<T>> nscache; // map for storing ns matrices
      mutable std::mutex cachemutex;                           // mutex for thread safety


      void autoc(){
        Tensor<T, 3> autocorrcoef(K, K, 4*K);
        auto autocorr_view = autocorrcoef.current_view();
        detail::autocorr_get<T>(K, autocorr_view);
        auto c_view = c.current_view();
        c_view = 0.0;
        std::array<Slice,NDIM> slices = {Slice(0, K), Slice(0, K), Slice(0, 2*K)};
        c_view(slices) = autocorr_view(slices);
        slices = {Slice(0, K), Slice(0, K), Slice(2*K, 4*K)};
        c_view(slices) = autocorr_view(slices);
      }

      // projection of a Gaussian onto double order polynomials
      const Tensor<T, 1>& make_rnlp(const Level n, Translation lx) {
        mra::Key<NDIM> key(n, std::array<Translation, NDIM>({lx}));
        auto it = rnlpcache.find(key);
        if (it != rnlpcache.end()) {
          const auto& r = it->second;
          return r;
        }

        Tensor<T, 1> rnlp(2*K);
        auto rnlp_view = rnlp.current_view();
        rnlp_view = 0.0;

        if (lx < 0) lx = -lx-1;
        T scaledcoeff  = coeff*std::pow(0.5, 0.5*n);
        T beta = expnt * std::pow(T(0.25), T(n));
        T h = 1.0/std::sqrt(beta);
        T nbox = 1.0/h;
        if (nbox < 1) nbox = 1;
        h = 1.0/nbox;
        T sch = std::abs(scaledcoeff*h);
        T argmax = std::abs(std::log(1e-22/sch));

        for (size_type box=0; box<nbox; ++box){
          T xlo = box*h + lx;
          if (beta*xlo*xlo > argmax) break;

          for (size_type i=0; i<npt; ++i){
            T* phix = new T[2*K];
            T xx = xlo + h*quad_x[i];
            T ee = scaledcoeff*std::exp(-beta*xx*xx)*quad_w[i]*h;
            legendre_scaling_functions(xx-lx, 2*K, &phix[0]);
            for (size_type p=0; p<2*K; ++p) {
              rnlp_view(p) += ee*phix[p];
            }
            delete[] phix;
          }
        }
        cachemutex.lock();
        if (rnlpcache.find(key) == rnlpcache.end()) {
          assert(rnlpcache.find(key) == rnlpcache.end());
          rnlpcache.emplace(key, std::move(rnlp));
        }
        it = rnlpcache.find(key);
        cachemutex.unlock();
        const auto& r = it->second;
        return r;
      }

    public:

      Convolution(size_type K, int npt, T coeff, T expnt, FunctionData<T, NDIM>& functiondata)
        : K(K), npt(npt), c(K, K, 4*K), coeff(coeff), expnt(expnt), functiondata(functiondata) {
        GLget(&quad_x, &quad_w, npt);
        autoc();
      }

      Convolution(Convolution&&) = default;
      Convolution(const Convolution&) = delete;
      Convolution& operator=(Convolution&&) = default;
      Convolution& operator=(const Convolution&) = delete;

      const Tensor<T, 2>& make_rnlij (const Level n, const Translation lx) {
        mra::Key<NDIM> key(n, std::array<Translation, NDIM>({lx}));
        cachemutex.lock();
        auto it = rnlijcache.find(key);
        cachemutex.unlock();
        if (it != rnlijcache.end()) {
          const auto& r = it->second;
          return r;
        }
        Tensor<T, 1> R(4*K);
        Tensor<T, 2> rnlij(K, K);
        auto R_view = R.current_view();

        const auto& rnlp1 = make_rnlp(n, lx-1);
        const auto& rnlp2 = make_rnlp(n, lx);
        auto rnlp1_view = rnlp1.current_view();
        auto rnlp2_view = rnlp2.current_view();

        std::array<Slice,1> slice1 = {Slice(0, 2*K)};
        R_view(slice1) = rnlp1_view(slice1);
        std::array<Slice,1> slice2 = {Slice(2*K, 4*K)};
        R_view(slice2) = rnlp2_view(slice1);

        T scale = std::pow(T(0.5), T(0.5*n));
        R_view *= scale;
        auto rnlij_view = rnlij.current_view();
        detail::inner(c.current_view(), R_view, rnlij_view);

        cachemutex.lock();
        if (rnlijcache.find(key) == rnlijcache.end()) {
          assert(rnlijcache.find(key) == rnlijcache.end());
          rnlijcache.emplace(key, std::move(rnlij));
        }

        it = rnlijcache.find(key);
        cachemutex.unlock();
        const auto& r = it->second;
        return r;
      }

      const ConvolutionData<T>& make_nonstandard (const Level n, const Translation lx) {
        mra::Key<NDIM> key(n, std::array<Translation, NDIM>({lx}));
        auto it = nscache.find(key);
        if (it != nscache.end()) {
          const auto& r = it->second;
          return r;
        }

        Tensor<T, 2> tmp(2*K, 2*K);
        const Tensor<T, 2>& rm = make_rnlij(n+1, 2*lx-1);
        const Tensor<T, 2>& r0 = make_rnlij(n+1, 2*lx);
        const Tensor<T, 2>& rp = make_rnlij(n+1, 2*lx+1);

        auto tmp_view = tmp.current_view();
        auto rm_view = rm.current_view();
        auto r0_view = r0.current_view();
        auto rp_view = rp.current_view();

        std::array<Slice,2> slice = {Slice(0, K), Slice(0, K)};
        tmp_view(slice) = r0_view;
        slice = {Slice(0, K), Slice(K, 2*K)};
        tmp_view(slice) = rm_view;
        slice = {Slice(K, 2*K), Slice(0, K)};
        tmp_view(slice) = rp_view;
        slice = {Slice(K, 2*K), Slice(K, 2*K)};
        tmp_view(slice) = r0_view;

        const auto& hgT = functiondata.get_hgT();
        auto hgT_view = hgT.current_view();
        Tensor<T, 2> R(2*K, 2*K), work(2*K, 2*K);
        auto R_view = R.current_view();
        transform(tmp_view, hgT_view, R_view, work.data());
        Tensor<T, 2> S(K, K);
        auto S_view = S.current_view();
        slice = {Slice(0, K), Slice(0, K)};
        S_view(slice) = R_view(slice);

        auto obj = ConvolutionData<T>();
        obj.R = std::move(R);
        obj.S = std::move(S);

        cachemutex.lock();
        if (nscache.find(key) == nscache.end()) {
          assert(nscache.find(key) == nscache.end());
          nscache.emplace(key, std::move(obj));
        }
        it = nscache.find(key);
        cachemutex.unlock();
        const auto& r = it->second;
        return r;
      }
    };

  template <typename T, Dimension NDIM>
  class ConvolutionOperator {

  private:
    size_type K;
    size_type seprank;
    Convolution<T, NDIM> conv;                       // convolution object
    std::map<Key<NDIM>, OperatorData<T, NDIM>> opdata;     // map for storing operator data
    mutable std::mutex cachemutex;                                 // mutex for thread safety

    T norm_ns(Level n, std::array<const ConvolutionData<T>*, NDIM>& ns) const {
      // ConvolutionData<T>* const ns[]
      T norm = 1.0, sum = 0.0;

      for (size_type d = 0; d < NDIM; ++d) {
        Tensor<T, 2> ns_r(2*K, 2*K);
        const auto& ref_view = ns[d]->R.current_view();
        const auto& ns_sview = ns[d]->S.current_view();
        auto ns_rview = ns_r.current_view();
        for (size_type i = 0; i < 2*K; ++i) {
          for (size_type j = 0; j < 2*K; ++j) {
            if(i<K && j<K) ns_rview(i, j) = 0.0;
            else ns_rview(i, j) = ref_view(i, j);
          }
        }
        T rnorm = normf(ns_rview);
        T snorm = normf(ns_sview);
        T aa = std::min(rnorm, snorm);
        T bb = std::max(rnorm, snorm);
        norm *= aa;
        if (bb > 0.0) sum += aa/bb;
      }
      if (n) norm *= sum;
      return norm;
    }

  public:

    ConvolutionOperator(size_type K, int npt, T coeff, T expnt, FunctionData<T, NDIM>& functiondata)
     : K(K), conv(K, npt, coeff, expnt, functiondata) {}

    ConvolutionOperator(ConvolutionOperator&&) = default;
    ConvolutionOperator(const ConvolutionOperator&) = delete;
    ConvolutionOperator& operator=(ConvolutionOperator&&) = default;
    ConvolutionOperator& operator=(const ConvolutionOperator&) = delete;

    const OperatorData<T, NDIM>& get_op(const Key<NDIM>& key) {
      auto it = opdata.find(key);
      if (it != opdata.end()) {
        return it->second;
      }
      OperatorData<T, NDIM> data;
      for (int i = 0; i < NDIM; ++i) {
        auto& cd = conv.make_nonstandard(key.level(), key.translation()[i]);
        data.ops[i] = &cd;
      }
      data.norm = norm_ns(key.level(), data.ops);
      cachemutex.lock();
      if (opdata.find(key) == opdata.end()) {
        assert(opdata.find(key) == opdata.end());
        opdata.emplace(key, std::move(data));
      }
      it = opdata.find(key);
      cachemutex.unlock();
      auto& r = it->second;
      return r;
    }
  };

} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
