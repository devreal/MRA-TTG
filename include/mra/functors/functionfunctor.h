#ifndef MRA_FUNCTIONFUNCTOR_H
#define MRA_FUNCTIONFUNCTOR_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

namespace mra {


    /// Function functor is going away ... don't use!
    /// \code
    ///template <typename T, Dimension NDIM>
    /// class FunctionFunctor {
    ///public:
    /// T operator()(const Coordinate<T,NDIM>& r) const;
    /// template <size_type K> void operator()(const SimpleTensor<T,NDIM,K>& x, FixedTensor<T,K,NDIM>& values) const;
    /// Level initial_level() const;
    /// bool is_negligible(const std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>& box, T thresh) const;
    /// special point interface to be added
    ///  }
    /// \endcode

    /// Adapts a simple callable to the API needed for evaluation --- implement your own for full vectorization
    template <typename T, Dimension NDIM>
    class FunctionFunctor {
        std::function<T(const Coordinate<T,NDIM>&)> f;

    public:
        static const Level default_initial_level = 3; //< needs to become user configurable

        template <typename functionT>
        FunctionFunctor(functionT f) : f(f) {}

        /// Evaluate at a single point
        T operator()(const Coordinate<T,NDIM>& r) const {return f(r);}

    };

    /**
     * TODO: adapt eval_cube to CUDA
     */

    /// Evaluate at points formed by tensor product of npt points in each dimension
    template <typename functorT> SCOPE void eval_cube(const functorT& f, const concepts::TensorView<1> auto& x, concepts::TensorView<1> auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        for (size_type i=0; i<x.dim(0); i++) values(i) = f(Coordinate<T,1>{x(0,i)});
    }

    /// Evaluate at points formed by tensor product of npt points in each dimension
    template <typename functorT> SCOPE void eval_cube(const functorT& f, const concepts::TensorView<2> auto& x, concepts::TensorView<2> auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        for (size_type i=0; i<x.dim(0); i++) {
            for (size_type j=0; j<x.dim(1); j++) {
                values(i,j) = f(Coordinate<T,2>{x(0,i),x(1,j)});
            }
        }
    }

    /// Evaluate at points formed by tensor product of K points in each dimension
    template <typename functorT> SCOPE void eval_cube(const functorT& f, const concepts::TensorView<3> auto& x, concepts::TensorView<3> auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        for (size_type i=0; i<x.dim(0); i++) {
            for (size_type j=0; j<x.dim(1); j++) {
                for (size_type k=0; k<x.dim(2); k++) {
                    values(i,j,k) = f(Coordinate<T,3>{x(0,i),x(1,j),x(2,k)});
                }
            }
        }
    }

    /// Evaluate at points formed by tensor product of K points in each dimension using vectorized form
    template <typename functorT, typename T> SCOPE void eval_cube_vec(const functorT& f, const concepts::TensorView<1> auto& x, T* values) {
        for (size_type i=0; i<x.dim(0); i++) {
            values[i] = f(x(0,i));
        }
    }

    /// Evaluate at points formed by tensor product of K points in each dimension using vectorized form
    template <typename functorT, typename T> SCOPE void eval_cube_vec(const functorT& f, const concepts::TensorView<2> auto& x, T* values) {
        for (size_type i=0; i<x.dim(0); i++) {
            values[i] = f(x(0,i),x(1,i));
        }
    }

    /// Evaluate at points formed by tensor product of K points in each dimension using vectorized form
    template <typename functorT, typename T> SCOPE void eval_cube_vec(const functorT& f, const concepts::TensorView<3> auto& x, T* values) {
        for (size_type i=0; i<x.dim(0); i++) {
            values[i] = f(x(0,i),x(1,i),x(2,i));
        }
    }

    namespace detail {
        template <class functorT> using initial_level_t =
            decltype(std::declval<const functorT>().initial_level());
        template <class functorT> using supports_initial_level =
            ::mra::is_detected<initial_level_t,functorT>;

        template <class functorT, class pairT> using is_negligible_t =
            decltype(std::declval<const functorT>().is_negligible(std::declval<pairT>(),std::declval<double>()));
        template <class functorT, class pairT> using supports_is_negligible =
            ::mra::is_detected<is_negligible_t,functorT,pairT>;
    }

    template <typename functionT> SCOPE Level initial_level(const functionT& f) {
        if constexpr (detail::supports_initial_level<functionT>()) return f.initial_level();
        else return 2; // <<<<<<<<<<<<<<< needs updating to make user configurable
    }

    template <typename functionT, typename T, Dimension NDIM>
    SCOPE bool is_negligible(const functionT& f, const std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>& box, T thresh) {
        using pairT = std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>;
        if constexpr (detail::supports_is_negligible<functionT,pairT>()) return f.is_negligible(box, thresh);
        else return false;
    }
} // namespace mra

#endif // MRA_FUNCTIONFUNCTOR_H
