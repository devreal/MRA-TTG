#ifndef MRA_POLYNOMIAL_H
#define MRA_POLYNOMIAL_H

#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/platform.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensor.h"

namespace mra {
  template <typename T, Dimension NDIM>
  class Polynomial {
    T expnt;
    Coordinate<T,NDIM> origin;
    T fac;
    T maxr;
    Level initlev;

    public:
      Polynomial() = default;

      Polynomial(const Domain<NDIM>& domain, T expnt, const Coordinate<T,NDIM>& origin)
      : expnt(expnt)
      , origin(origin)
      , fac(std::pow(T(2.0*expnt/M_PI),T(0.25*NDIM)))
      , maxr(std::sqrt(std::log(fac/1e-12)/expnt))
      {
        const size_type N = 6;
        const size_type K = 6;
        const T log10 = std::log(10.0);
        const T log2 = std::log(2.0);
        const T L = domain.get_max_width();
        const T a = expnt*L*L;
        double n = std::log(a/(4*K*K*(N*log10+std::log(fac))))/(2*log2);
        initlev = Level(n<2 ? 2.0 : std::ceil(n));
      }

      Polynomial(const Polynomial&) = default;
      Polynomial(Polynomial&&) = default;
      Polynomial& operator=(const Polynomial&) = default;
      Polynomial& operator=(Polynomial&&) = default;

      SCOPE void operator()(const TensorView<T,2>& x, T* values, size_type N) const {
        assert(x.dim(0) == NDIM);
        assert(x.dim(1) == N);
        for (size_type i = thread_id(); i < N; i += block_size()) {
          values[i] = 1.0;
          }
        }
      SCOPE Level initial_level() const {
        return this->initlev;
        }
      SCOPE bool is_negligible(const std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>& box, T thresh) const {
        auto& lo = box.first;
        auto& hi = box.second;
        T rsq = 0.0;
        T maxw = 0.0; // max width of box
        for (Dimension d = 0; d < NDIM; ++d) {
          maxw = std::max(maxw,hi(d)-lo(d));
          T x = T(0.5)*(hi(d)+lo(d)) - origin(d);
          rsq += x*x;
        }
        T diagndim = T(0.5)*std::sqrt(T(NDIM));
        T boxradplusr = maxw*diagndim + maxr;
        // ttg::print(box, boxradplusr, bool(boxradplusr*boxradplusr < rsq));
        return (boxradplusr*boxradplusr < rsq);
      }
    };
  } // namespace mra

#endif // MRA_POLYNOMIAL_H
