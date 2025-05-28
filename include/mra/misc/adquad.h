#ifndef MRA_ADQUAD_H
#define MRA_ADQUAD_H

#include "tensor/tensor.h"
#include "mra/misc/gl.h"
#include "mra/misc/types.h"

namespace mra {

  namespace detail{

    template <typename functorT, typename T>
    void do_adq(const T lo, const T hi, const functorT& func,
                int n, const T* x, const T* w, Tensor<T, 1>& adq_val){
      T range = hi - lo;
      for (int i = 0; i < n; ++i) adq_val += func(lo + range * x[i]) * w[i];
      adq_val *= range;
    }

    template <typename functorT, typename T>
    void adq1(const T lo, const T hi, const functorT& func,
              const T thresh, int n, int level, const T* x, const T* w, Tensor<T, 1>& adq){
      static const int MAX_LEVEL = 14;
      T d = (hi - lo) / 2;

      Tensor<T, 1> full(adq), half(adq);
      full.fill(0.0);
      half.fill(0.0);
      do_adq(lo, hi, func, n, x, w, full);
      do_adq(lo, lo+d, func, n, x, w, half);
      do_adq(lo+d, hi, func, n, x, w, half);

      T abserr = std::abs(full - half);
      T norm = std::abs(half);
      T relerr = (norm==0.0) ? 0.0 : abserr/norm;

      bool converged = (relerr < 1e-14) || (abserr<thresh && relerr<0.01);
      if (converged || level >= MAX_LEVEL) {
        adq = half;
      } else {
        adq1(lo, lo+d, func, thresh*0.5, n, level+1, x, w, adq);
        adq1(lo+d, hi, func, thresh*0.5, n, level+1, x, w, adq);
      }
    }


    template <typename functorT, typename T>
    void adq(const T lo, const T hi, const functorT& func,
             const T thresh, Tensor<T, 1> result){
      const int n = 20;
      T x[n], w[n];
      GLget(&x, &w, n);

      adq1(lo, hi, func, thresh, n, 0, x, w, result);
    }
  }
}

#endif // MRA_ADQUAD_H
