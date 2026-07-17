#ifndef MRA_KERNELS_FCUBE_H
#define MRA_KERNELS_FCUBE_H

#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

namespace mra{
  /// Make outer product of quadrature points for vectorized algorithms
  SCOPE void make_xvec(const concepts::TensorView<2> auto& x, concepts::TensorView<2> auto& xvec,
                          std::integral_constant<Dimension, 1>) {
    /* uses threads in 3 dimensions */
    xvec = x;
    /* TensorView assignment synchronizes */
  }

  /// Make outer product of quadrature points for vectorized algorithms
  SCOPE void make_xvec(const concepts::TensorView<2> auto& x, concepts::TensorView<2> auto& xvec,
                          std::integral_constant<Dimension, 2>) {
    const size_type K = x.dim(1);
    if (threadIdx.z == 0) {
      for (size_type i=threadIdx.y; i<K; i += blockDim.y) {
        for (size_type j=threadIdx.x; j<K; j += blockDim.x) {
          size_type ij = i*K + j;
          xvec(0,ij) = x(0,i);
          xvec(1,ij) = x(1,j);
        }
      }
    }
    SYNCTHREADS();
  }

  /// Make outer product of quadrature points for vectorized algorithms
  SCOPE void make_xvec(const concepts::TensorView<2> auto& x, concepts::TensorView<2> auto& xvec,
                          std::integral_constant<Dimension, 3>) {
    const size_type K = x.dim(1);
    for (size_type i=threadIdx.z; i<K; i += blockDim.z) {
      for (size_type j=threadIdx.y; j<K; j += blockDim.y) {
        for (size_type k=threadIdx.x; k<K; k += blockDim.x) {
          size_type ijk = i*K*K + j*K + k;
          xvec(0,ijk) = x(0,i);
          xvec(1,ijk) = x(1,j);
          xvec(2,ijk) = x(2,k);
        }
      }
    }
    SYNCTHREADS();
  }

  /// Set X(d,mu) to be the mu'th quadrature point in dimension d for the box described by key
  template<typename T, Dimension NDIM>
  SCOPE void make_quadrature_pts(
    const Domain<NDIM>& D,
    const T* gldata,
    const Key<NDIM>& key,
    concepts::TensorView<2> auto& X, size_type K)
  {
    assert(X.dim(0) == NDIM);
    assert(X.dim(1) == K);
    const Level n = key.level();
    const std::array<Translation,NDIM>& l = key.translation();
    const T h = std::pow(T(0.5),T(n));
    /* retrieve x[] from constant memory, use float */
    const T *x, *w;
    GLget(gldata, &x, &w, K);
    if (threadIdx.z == 0) {
      for (size_type d = threadIdx.y; d < X.dim(0); d += blockDim.y) {
        auto [lo,hi] = D.get(d);
        T width = h*D.get_width(d);
        for (size_type i = threadIdx.x; i < X.dim(1); i += blockDim.x) {
          X(d,i) = lo + width*(l[d] + x[i]);
        }
      }
    }
    /* wait for all to complete */
    SYNCTHREADS();
  }



  template <typename functorT, typename T, Dimension NDIM>
  SCOPE void fcube(
    const Domain<NDIM>& D,
    const T* gldata,
    const functorT& f,
    const Key<NDIM>& key,
    const T truncate_tol,
    // output
    concepts::TensorView<3> auto& values,
    size_type K,
    // temporaries
    concepts::TensorView<2> auto& x,
    concepts::TensorView<2> auto& xvec)
  {
    if (is_negligible(f, D.template bounding_box<T>(key), truncate_tol)) {
        values = 0.0;
        /* TensorView assigment synchronizes */
    }
    else {
      const size_type K = values.dim(0);
      const size_type K2NDIM = std::pow(K,NDIM);
      // sanity checks
      assert(x.dim(0) == NDIM);
      assert(x.dim(1) == K   );
      assert(xvec.dim(0) ==   NDIM);
      assert(xvec.dim(1) == K2NDIM);
      make_quadrature_pts(D, gldata, key, x, K);

      constexpr bool call_coord = std::is_invocable_r<T, decltype(f), Coordinate<T,NDIM>>(); // f(coord)
      constexpr bool call_1d = (NDIM==1) && std::is_invocable_r<T, decltype(f), T>(); // f(x)
      constexpr bool call_2d = (NDIM==2) && std::is_invocable_r<T, decltype(f), T, T>(); // f(x,y)
      constexpr bool call_3d = (NDIM==3) && std::is_invocable_r<T, decltype(f), T, T, T>(); // f(x,y,z)
      constexpr bool call_vec = std::is_invocable<decltype(f), decltype(xvec), T*, int>(); // vector API
      static_assert(std::is_invocable<decltype(f), decltype(xvec), T*, int>());
      static_assert(call_coord || call_1d || call_2d || call_3d || call_vec, "no working call");

      if constexpr (call_1d || call_2d || call_3d || call_vec) {
        make_xvec(x, xvec, std::integral_constant<Dimension, NDIM>{});
        if constexpr (call_vec) {
          f(xvec, values.data(), K2NDIM);
        }
        else if constexpr (call_1d || call_2d || call_3d) {
          eval_cube_vec(f, xvec, values);
        }
      }
      else if constexpr (call_coord) {
        eval_cube(f, x, values);
      }
      else {
        //throw "how did we get here?";
        // TODO: how to handle this?
        assert(!"Failed to handle eval call!");
      }
      SYNCTHREADS();
    }
  }

} // namespace mra

#endif // MRA_KERNELS_FCUBE_H
