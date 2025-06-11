#ifndef MRA_KERNELS_FCUBE_FOR_MUL_H
#define MRA_KERNELS_FCUBE_FOR_MUL_H

#include <cassert>
#include "mra/kernels.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/platform.h"
#include "mra/misc/functiondata.h"
#include "mra/tensor/tensorview.h"

#define MAX_ORDER 64
namespace mra {

  template <typename T>
  SCOPE void compute_legendre(
    const T x,
    const size_type order,
    T* p,
    const T* nn1)
  {
    p[0] = 1.0;
    if (order == 0) return;
    p[1] = x;
    for (size_type n=1; n<order; ++n) {
      p[n+1] = (x*p[n] - p[n-1]) * nn1[n] + x*p[n];
    }
  }

  template <typename T>
  SCOPE void compute_scaling(
    const T x,
    const size_type order,
    T* p,
    const T* phi_norm,
    const T* nn1)
  {
    compute_legendre(T(2)*x - 1, order - 1, p, nn1);
    for (size_type n=0; n<order; ++n) {
      p[n] *= phi_norm[n];
    }
  }

  template <typename T>
  SCOPE void phi_for_mul(
    const Level np,
    const Level nc,
    const Translation lp,
    const Translation lc,
    TensorView<T, 2>& phi,
    T* pv,
    const T* nn1,
    const T* phi_norms,
    const TensorView<T, 1>& quad_x,
    const size_type K)
  {
    T scale = pow(2.0, T(np-nc));

    for(size_type mu = 0; mu < K; ++mu) {
      T xmu = scale * (quad_x(mu) + lc) - lp;
      assert(xmu > -1e-15 && xmu < 1.0 + 1e-15);
      compute_scaling(xmu, K, pv, phi_norms, nn1);
      for (size_type i = 0; i < K; ++i) phi(i, mu) = pv[i];
    }
    T scale_phi = std::pow(2.0, 0.5*np);
    phi *= scale_phi;
  }


  template <typename T, Dimension NDIM>
  SCOPE void fcube_for_mul(
    const Domain<NDIM>& D,
    const Key<NDIM>& child,
    const Key<NDIM>& parent,
    const TensorView<T,NDIM>& coeffs,
    TensorView<T, NDIM>& result_values,
    const TensorView<T, 2>& phi_old,
    const TensorView<T, 2>& phibar,
    const TensorView<T, 1>& quad_x,
    const size_type K,
    T* workspace)
  {
    if (child.level() < parent.level()) {
      THROW("fcube_for_mul: bad child-parent relationship");
    }
    else if (child.level() == parent.level()) {
      // coeffs_to_values()
      transform(coeffs, phibar, result_values, workspace);
      T scale = std::pow(2.0, 0.5*NDIM*parent.level())/std::sqrt(D.template get_volume<T>());
      result_values *= scale;
    }
    else {
#ifdef HAVE_DEVICE_ARCH
      extern SHARED T phi[];
#else
      T* phi = new T[K*K*NDIM];
#endif
      SHARED T p[MAX_ORDER], nn1[MAX_ORDER], phi_norms[MAX_ORDER];
      SHARED std::array<TensorView<T, 2>, NDIM> phi_views;
      if(is_team_lead()){
        for (int d = 0; d < NDIM; ++d){
          phi_views[d] = TensorView<T, 2>(&phi[d*K*K], K, K);
        }
      }
      SYNCTHREADS();
      for (size_type i = thread_id(); i < MAX_ORDER; i+=block_size()) {
        nn1[i] = T(i) / T(i + 1.0);
        phi_norms[i] = std::sqrt(T(2*i + 1));
      }

      auto parent_l = parent.translation();
      auto child_l = child.translation();
      for (size_type d=0; d < NDIM; ++d){
        phi_for_mul<T>(parent.level(), child.level(), parent_l[d], child_l[d],
                       phi_views[d], p, nn1, phi_norms, quad_x, K);
      }

      SHARED TensorView<T, NDIM> result_tmp;
      if (is_team_lead()) {
        result_tmp = TensorView<T, NDIM>(workspace, result_values.dims());
      }
      SYNCTHREADS();

      general_transform(coeffs, phi_views, result_values, result_tmp);
      T scale = T(1)/std::sqrt(D.template get_volume<T>());
      result_values *= scale;
#ifndef HAVE_DEVICE_ARCH
      delete[] phi;
#endif
    }

  }

} // namespace mra

#endif // MRA_KERNELS_FCUBE_FOR_MUL_H
