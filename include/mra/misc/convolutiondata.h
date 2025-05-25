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
      Translation lx;
      Tensor<T, NDIM> autocorr; // autocorrelation coefficients
      Tensor<T, NDIM> rnlp;     // rnlp coefficients

      void autocorr_get() {
        detail::autocorr_get(K, p);
      }

      void autoc(){
        size_type twoK = 2*K;
        TensorSlice<T, NDIM> autocorr_view = autocorr.current_view();
        autocorr_get(K, autocorr_view.data());
      }

      void rnlij(){

      }

      void get_rnlp(TensorView<T, NDIM>& rnlp) {
        size_type twoK = 2*K;

        Translation twoN = Translation(1) << n;

      }

      void rnlp(const Level n, const Translation lx, TensorView<T, 2>& phi,
                TensorView<T, 1>& rnlp) {
        std::pair<T, T> integrange{0, 1};



      }

    public:

      void get_rnlp(const size_type K, TensorView<T, NDIM>& rnlp) {
        autocorr_get(K, rnlp.data());
      }

      // MADNESS -- function name preserved
      /// Computes the transition matrix elements for the convolution for n,l
      /// Returns the tensor
      /// \code
      ///   r(i,j) = int(K(x-y) phi[n0](x) phi[nl](y), x=0..1, y=0..1)
      /// \endcode
      /// This is computed from the matrix elements over the correlation
      /// function which in turn are computed from the matrix elements
      /// over the double order legendre polynomials.
      /// \note if `this->range_restricted()==true`, `θ(D/2 - |x-y|) K(x-y)` is used as the kernel
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

      void get_rnlp(const size_type K, const Translation lx, TensorView<T, NDIM>& rnlp) {
        size_type twoK = 2*K;

      }
  };

} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
