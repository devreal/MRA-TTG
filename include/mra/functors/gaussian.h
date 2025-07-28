#ifndef MRA_GAUSSIAN_H
#define MRA_GAUSSIAN_H

#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/platform.h"
#include "mra/ops/functions.h"

#include <numbers>

/**
 * Defines a Gaussian functor that we use across this example.
 * We want to template the whole stack on the functor but we
 * have to verify that nvcc can compile all TTG code (incl. coroutines).
 */

namespace mra {
    // Test gaussian functor
    template <typename T, Dimension NDIM>
    class Gaussian {
        T expnt;
        Coordinate<T,NDIM> origin;
        T fac;
        T maxr;
        Level initlev;
    public:
        /* default construction required for ttg::Buffer */
        Gaussian() = default;

        Gaussian(const Domain<NDIM>& domain, T expnt, const Coordinate<T,NDIM>& origin, int initial_level = 0)
        : expnt(expnt)
        , origin(origin)
        , fac(std::pow(T(2.0*expnt/std::numbers::pi),T(0.25*NDIM)))
        , maxr(std::sqrt(std::log(fac/1e-12)/expnt))
        , initlev(initial_level)
        {
            // Pick initial level such that average gap between quadrature points
            // will find a significant value
            const size_type N = 6; // looking for where exp(-a*x^2) < 10**-N
            const size_type K = 6; // typically the lowest order of the polyn
            const T log10 = std::log(10.0);
            const T log2 = std::log(2.0);
            const T L = domain.get_max_width();
            const T a = expnt*L*L;
            double n = std::log(a/(4*K*K*(N*log10+std::log(fac))))/(2*log2);
            //std::cout << expnt << " " << a << " " << n << std::endl;
            if (initial_level == 0) {
                initlev = Level(n<2 ? 2.0 : std::ceil(n));
            }
        }

        /* default copy ctor and operator */
        Gaussian(const Gaussian&) = default;
        Gaussian(Gaussian&&) = default;
        Gaussian& operator=(const Gaussian&) = default;
        Gaussian& operator=(Gaussian&&) = default;

        // T operator()(const Coordinate<T,NDIM>& r) const {
        //     T rsq = 0.0;
        //     for (auto x : r) rsq += x*x;
        //     return fac*std::exp(-expnt*rsq);
        // }

        /**
         * Evaluate function at N points x and store result in \c values
         */
        SCOPE void operator()(const TensorView<T,2>& x, T* values, size_type N) const {
            assert(x.dim(0) == NDIM);
            assert(x.dim(1) == N);
            distancesq(origin, x, values, N);
            for (size_type i = thread_id(); i < N; i += block_size()) {
                values[i] = fac * std::exp(-expnt*values[i]);
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

    template <typename T, Dimension NDIM>
    class GaussianDerivative {
        T expnt;
        Coordinate<T,NDIM> origin;
        T fac;
        T maxr;
        Level initlev;
    public:
        /* default construction required for ttg::Buffer */
        GaussianDerivative() = default;

        GaussianDerivative(const Domain<NDIM>& domain, T expnt, const Coordinate<T,NDIM>& origin)
        : expnt(expnt)
        , origin(origin)
        , fac(std::pow(T(2.0*expnt/std::numbers::pi),T(0.25*NDIM)))
        , maxr(std::sqrt(std::log(fac/1e-12)/expnt))
        {
            // Pick initial level such that average gap between quadrature points
            // will find a significant value
            const size_type N = 6; // looking for where exp(-a*x^2) < 10**-N
            const size_type K = 6; // typically the lowest order of the polyn
            const T log10 = std::log(10.0);
            const T log2 = std::log(2.0);
            const T L = domain.get_max_width();
            const T a = expnt*L*L;
            double n = std::log(a/(4*K*K*(N*log10+std::log(fac))))/(2*log2);
            //std::cout << expnt << " " << a << " " << n << std::endl;
            initlev = Level(n<2 ? 2.0 : std::ceil(n));
        }

        /* default copy ctor and operator */
        GaussianDerivative(const GaussianDerivative&) = default;
        GaussianDerivative(GaussianDerivative&&) = default;
        GaussianDerivative& operator=(const GaussianDerivative&) = default;
        GaussianDerivative& operator=(GaussianDerivative&&) = default;

        // T operator()(const Coordinate<T,NDIM>& r) const {
        //     T rsq = 0.0;
        //     for (auto x : r) rsq += x*x;
        //     return fac*std::exp(-expnt*rsq);
        // }

        /**
         * Evaluate function at N points x and store result in \c values
         */
        SCOPE void operator()(const TensorView<T,2>& x, T* values, size_type N) const {
            assert(x.dim(0) == NDIM);
            assert(x.dim(1) == N);
            distancesq(origin, x, values, N);
            for (size_type i = thread_id(); i < N; i += block_size()) {
                values[i] = T(-2) * expnt * std::sqrt(values[i]) * fac * std::exp(-expnt*values[i]);
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

#endif // MRA_GAUSSIAN_H
