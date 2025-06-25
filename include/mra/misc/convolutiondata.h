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
  class Convolution {

    private:
      size_type K;
      Level n;
      int npt;                                      // number of quadrature points
      T expnt;
      T coeff;
      Translation lx;
      const T* quad_x;                              // quadrature points
      const T* quad_w;                              // quadrature weights
      Tensor<T, 3> autocorrcoef;                    // autocorrelation coefficients
      Tensor<T, 3> c;                               // autocorrelation coefficients
      std::map<Key<NDIM>, Tensor<T, 2>> rnlijcache; // map for storing rnlij matrices
      std::map<Key<NDIM>, Tensor<T, 1>> rnlpcache;  // map for storing rnlp matrices

      std::mutex cachemutex;                        // mutex for thread safety


      void autoc(){
        auto c_view = c.current_view();
        auto autocorr_view = autocorrcoef.current_view();
        detail::autocorr_get<T>(K, autocorr_view);
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
          }
        }
        cachemutex.lock();
        if (rnlpcache.find(key) == rnlpcache.end()) {
          assert(rnlpcache.find(key) == rnlpcache.end());
          rnlpcache.emplace(std::move(key), std::move(rnlp));
        }
        it = rnlpcache.find(key);
        cachemutex.unlock();
        const auto& r = it->second;
        return r;
      }

    public:

      Convolution(size_type K, int npt, T coeff, T expnt)
        : K(K), npt(npt), autocorrcoef(K, K, 4*K),
          c(K, K, 4*K), coeff(coeff), expnt(expnt) {
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
          rnlijcache.emplace(std::move(key), std::move(rnlij));
        }

        it = rnlijcache.find(key);
        cachemutex.unlock();
        std::cout << "returning after computation" << std::endl;
        const auto& r = it->second;
        return r;
      }
    };

} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
