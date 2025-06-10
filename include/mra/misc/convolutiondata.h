#ifndef MRA_CONVOLUTIONDATA_H
#define MRA_CONVOLUTIONDATA_H

#include "mra/ops/inner.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/types.h"
#include "mra/misc/adquad.h"
#include "mra/misc/platform.h"
#include "mra/misc/autocorr.h"
#include "mra/tensor/tensorview.h"

#define MRA_MAX_LX 7
namespace mra {

  template <typename T, Dimension NDIM>
  class ConvolutionData {

    public:
      using rnl_key = std::tuple<Level, Translation>;

    private:
      size_type K;
      Level n;
      int npt;                  // number of quadrature points
      Translation lx;
      Tensor<T, 1> quad_x;      // quadrature points
      Tensor<T, 1> quad_w;      // quadrature weights
      Tensor<T, 3> autocorr;    // autocorrelation coefficients
      Tensor<T, 3> c;           // autocorrelation coefficients
      Tensor<T, 1> coeff;       // coefficients for the convolution
      Tensor<T, 1> rnlp;        // rnlp coefficients
      std::map<rnl_key, std::map<rnl_key, Tensor<T, 2>>> matrixmap; // map for storing rnlp matrices
      std::mutex mapmutex; // mutex for thread safety


      void autoc(){
        detail::autocorr_get(K, autocorr.data());

        auto c_view = c.current_view();
        std::array<Slice,NDIM> slices = {Slice(0, K-1), Slice(0, K-1), Slice(0, 2*K-1)};
        c_view(slices) = autocorr(slices);
        slices = {Slice(0, K-1), Slice(0, K-1), Slice(2K, 4*K-1)};
        c_view(slices) = autocorr(slices);
      }

      Tensor& make_rnlij(const Level n, const Translation lx){
        Tensor<T, 1> R(4*K);
        Tensor<T, 2> rnlij(2*K, 2*K);
        R_view = R.current_view();
        make_rnlp(n, lx-1);
        R_view(Slice(0, 2*K-1)) = rnlp;
        make_rnlp(n, lx);
        R_view(Slice(2*K, 4*K-1)) = rnlp;

        T scale = std::pow(T(0.5), T(0.5*n));
        R_view *= scale;

        inner(c, R, rnlij);

        return rnlij;
      }


      // projection of a Gaussian onto double order polynomials
      void make_rnlp(const Level n, const Translation lx) {

        if (lx < 0) lx = -lx-1;
        rnlp.fill(0.0);
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

      void construct_matrices(const Level n, const Translation lx) {
        auto key = std::make_tuple(n, lx);
        if (matrixmap.find(key) != matrixmap.end()) {
          return; // already constructed
        }

        std::map<rnl_key, Tensor<T, 2>> rnl0map;

        for (size_type l0 = -lx-MRA_MAX_LX; l0<=lx+MRA_MAX_LX; ++l0) {
          Tensor<T, 2> rnlij(2*K, 2*K) = make_rnlij(n, l0);
          auto rnl0_key = std::make_tuple(n, l0);
          mapmutex.lock();
          if (rnl0map.find(rnl0_key) == rnl0map.end()) {
            assert(rnl0map.find(rnl0_key) == rnl0map.end());
            rnl0_map[rnl0_key] = rnlij;
            mapmutex.unlock();
          }
        }
        mapmutex.lock();
        assert(matrixmap.find(key) == matrixmap.end());
        matrixmap[key] = rnl0_map;
        mapmutex.unlock();
      }

    public:

      ConvolutionData(size_type K, Level n, int npt, Translation lx)
        : K(K), n(n), npt(npt), lx(lx),
          quad_x(K), quad_w(K),
          autocorr(K, K, 4*K),
          c(K, K, 4*K), rnlij(K, K),
          coeff(K), rnlp(K)
      {
        autoc();
      }
      const auto& get_rnlp() const { return rnlp;}
      const auto& get_rnlij() const { return rnlij;}
      const auto& construct_matrices(const Level n){ return matrixmap;}
  };


} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
