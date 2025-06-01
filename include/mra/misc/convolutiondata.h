#ifndef MRA_CONVOLUTIONDATA_H
#define MRA_CONVOLUTIONDATA_H

#include "mra/misc/types.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/adquad.h"
#include "mra/misc/platform.h"
#include "mra/misc/autocorr.h"
#include "mra/tensor/tensorview.h"


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
      Tensor<T, 3> autocorr;    // autocorrelation coefficients
      Tensor<T, 3> c;           // autocorrelation coefficients
      Tensor<T, 2> rnlij;       // rnlij coefficients
      Tensor<T, 1> coeff;       // coefficients for the convolution
      Tensor<T, 1> rnlp;        // rnlp coefficients


      void autoc(){
        detail::autocorr_get(K, autocorr.data());

        auto c_view = c.current_view();
        c_view(Slice(0, K-1), Slice(0, K-1), Slice(0, 2*K-1)) = autocorr(Slice(0, K-1), Slice(0, K-1), Slice(0, 2*K-1));
        c_view(Slice(0, K-1), Slice(0, K-1), Slice(2K, 4*K-1)) = autocorr(Slice(0, K-1), Slice(0, K-1), Slice(2*K, 4*K-1));
      }

      void rnlij(const Level n, const Translation lx){

      }


      // projection of a Gaussian onto double order polynomials
      void rnlp(const Level n, const Translation lx) {

        if (lx < 0) lx = -lx-1;

        auto rnlp_view = rnlp.current_view();
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
            T xx = xlo + h*quad_x(i);
            T ee = scaledcoeff*std::exp(-beta*xx*xx)*quad_w(i)*h;

            legendre_scaling_functions(xx-lx, 2*K, &phix[0]);

            for (size_type p=0; p<2*K; ++p) rnlp_view(p) += ee*phix[p];
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

      ConvolutionData(size_type K, Level n, int npt, Translation lx)
        : K(K), n(n), npt(npt), lx(lx),
          quad_x({K}), quad_w({K}),
          autocorr({K, K, K}),
          coeff({K}),
          rnlp({K})
      {
        autoc();
      }
      const auto& get_rnlp() const { return rnlp;}
      const auto& get_rnlij() const { return rnlij;}
  };


} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
