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
    const size_type K,
    T* p,
    const T* phi_norm,
    const T* nn1)
  {
    /* compute_legendre has a loop-carried dependency so execute sequentially */
    if (is_team_lead()) {
      compute_legendre(T(2)*x - 1, K - 1, p, nn1);
    }
    SYNCTHREADS();
    for (size_type n=thread_id(); n<K; n+=block_size()) {
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
    const TensorView<T, 1>& quad_x,
    const size_type K)
  {
    T scale = std::pow(2.0, T(np-nc));

    /**
     * The first K threads compute.
     * compute_scaling has a loop-carried dependency so we cannot parallelize it.
     * If we inline everything into the mu-loop we can avoid the temporary array p.
     */
    for(size_type mu = thread_id(); mu < K; mu += block_size()) {
      T xmu = scale * (quad_x(mu) + lc) - lp;
      assert(xmu > -1e-15 && xmu < 1.0 + 1e-15);

      auto phi_norms = [](int i) {
        return std::sqrt(T(2*i + 1));
      };

      auto nn1 = [](int i) {
        return T(i) / T(i + 1.0);
      };

      // inlined compute_scaling and assignment to phi(:,mu)
      //compute_scaling(xmu, K, p, phi_norms, nn1);
      //for (size_type i = thread_id(); i < K; i += block_size()) phi(i, mu) = p[i];
      T x = T(2)*xmu - 1;
      T pm0 = 1.0, pm1 = x;
      phi(0, mu) = pm0 * phi_norms(0);
      phi(1, mu) = pm1 * phi_norms(1);
      for (int i = 2; i < K; ++i) {
        T pm2 = (x * pm1 - pm0) * nn1(i-1) + x * pm1;
        phi(i, mu) = pm2 * phi_norms(i);
        pm0 = pm1;
        pm1 = pm2;
      }
    }
    SYNCTHREADS();
    T scale_phi = std::pow(2.0, 0.5*np);
    phi *= scale_phi;

    std::cout << "phi_for_mul np " << np << " nc " << nc << " lp " << lp << " lc " << lc << "\n" << normf(phi) << std::endl;
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
      SHARED std::array<TensorView<T, 2>, NDIM> phi_views;
      if(is_team_lead()){
        for (int d = 0; d < NDIM; ++d){
          phi_views[d] = TensorView<T, 2>(&phi[d*K*K], K, K);
        }
      }
      SYNCTHREADS();

      const auto& parent_l = parent.translation();
      const auto& child_l = child.translation();
      for (size_type d=0; d < NDIM; ++d){
        phi_for_mul<T>(parent.level(), child.level(), parent_l[d], child_l[d],
                       phi_views[d], quad_x, K);
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
