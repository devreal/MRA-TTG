#ifndef MRA_CONVOLUTIONDATA_H
#define MRA_CONVOLUTIONDATA_H

#include "mra/misc/types.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/adquad.h"
#include "mra/misc/platform.h"
#include "mra/misc/autocorr.h"


namespace mra {

  template <typename T, Dimension NDIM>
  class ConvolutionData {

    private:
      size_type K;
      Level n;
      int npt;                  // number of quadrature points
      Translation lx;
      Tensor<T, 1> quad_x;      // quadrature points
      Tensor<T, 1> quad_w;      // quadrature weights
      Tensor<T, NDIM> autocorr; // autocorrelation coefficients
      Tensor<T, 1> coeff;       // coefficients for the convolution
      Tensor<T, 1> rnlp;        // rnlp coefficients



      void autocorr_get() {
        detail::autocorr_get(K, p);
      }

      void autoc(const ){
        size_type twoK = 2*K;
        TensorSlice<T, NDIM> autocorr_view = autocorr.current_view();
        autocorr_get(K, autocorr_view.data());
      }

      void rnlij(const Level n, const Translation lx, TensorView<T, 2>& rnlij){

      }

      void get_rnlp(TensorView<T, NDIM>& rnlp) {
        size_type twoK = 2*K;

        Translation twoN = Translation(1) << n;

      }

      void rnlp(const Level n, const Translation lx) {

        Translation lkeep = lx;
        if (lx < 0) lx = -lx-1;

        T scaledcoeff  = coeff*std::pow(0.5, 0.5*n);
        T fourn = std::pow(T(4), T(n));
        T beta = expnt * std::pow(T(0.25), T(n));
        T h = 1.0/std::sqrt(beta);
        T nbox = T(1/h);
        if (nbox < 1) nbox = 1;
        h = 1.0/nbox;

        T sch = std::abs(scaledcoeff*h);

        T argmax = std::abs(std::log(1e-22/sch));

        for (size_type box=0; box<nbox; ++box){
          T xlo = box*h + lx;
          if (beta*xlo*xlo > argmax) break;

          for (size_type i=0; i<npt; ++i){
            T* phix = new T[2*K];
            T xx = xlo + h*quad_x(i);
            T ee = scaledcoeff*std::exp(-beta*xx*xx)*quad_w(i)*h;

            legendre_scaling_functions(xx-lx, 2*K, &phix[0]);

            for (size_type p=0; p<2*K; ++p) rnlp(p) = ee*phix[p];
          }
        }
      }

      void rnlij(
        const size_type K,
        const Level n,
        const Translation lx,
        bool do_transpose,
        const TensorView<T, NDIM>& c, // this is the array of coefficients from  autocorr.cc autocorr_get()
        TensorView<T, NDIM>& rnlij){
          size_type twoK = 2*K;

          TensorSlice<T, NDIM> rnlij_view = rnlij.current_view();
          get_rnlp(n, lx-1, rnlij_view); // append into rnlij_view
          get_rnlp(n, lx, rnlij_view); // append into rnlij_view
        }

    public:

      const auto& get_rnlp() const { return rnlp;}
      const auto& get_rnlij() const { return rnlij;}
  };


} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
