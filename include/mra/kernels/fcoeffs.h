#ifndef MRA_KERNELS_FCOEFFS_H
#define MRA_KERNELS_FCOEFFS_H

#include "mra/misc/gl.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/platform.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensorview.h"
#include "mra/kernels/fcube.h"
#include "mra/kernels/transform.h"

namespace mra {

  /* Returns the total size of temporary memory needed for
  * the project() kernel. */
  template<mra::Dimension NDIM>
  SCOPE size_type fcoeffs_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    const size_type TWOK2NDIM = std::pow(2*K, NDIM);
    return (3*TWOK2NDIM) // workspace, values and r0
         + (NDIM*K2NDIM) // xvec in fcube
         + (NDIM*K)      // x in fcube
         + (2*K2NDIM);   // child_values, r1
  }

  namespace detail {

    template<typename Fn, typename T, Dimension NDIM>
    DEVSCOPE void fcoeffs_kernel_impl(
      const Domain<NDIM>& D,
      const T* gldata,
      const Fn& f,
      Key<NDIM> key,
      size_type K,
      size_type fnid,
      /* temporaries */
      TensorView<T, NDIM>& values,
      TensorView<T, NDIM>& r0,
      TensorView<T, NDIM>& r1,
      TensorView<T, NDIM>& child_values,
      TensorView<T, 2   >& x_vec,
      TensorView<T, 2   >& x,
      T* workspace, /* variable size so pointer only */
      /* constants */
      const TensorView<T, 2>& phibar,
      const TensorView<T, 2>& hgT,
      /* result */
      TensorView<T, NDIM>&  coeffs,
      bool *is_leaf,
      T thresh)
    {
      /* check for our function */
      if ((key.level() < initial_level(f))) {
        // std::cout << "project: key " << key << " below intial level " << initial_level(f) << std::endl;
        coeffs = T(1e7); // set to obviously bad value to detect incorrect use
        if (is_team_lead()) {
          *is_leaf = false;
        }
      }
      if (is_negligible<Fn,T,NDIM>(f, D.template bounding_box<T>(key), mra::truncate_tol(key,thresh))) {
        /* zero coeffs */
        coeffs = T(0.0);
        if (is_team_lead()) {
          *is_leaf = true;
        }
      } else {

        /* compute one child */
        for (int bid = 0; bid < key.num_children(); bid++) {
          Key<NDIM> child = key.child_at(bid);
          child_values = 0.0; // TODO: needed?
          fcube(D, gldata, f, child, thresh, child_values, K, x, x_vec);
          transform(child_values, phibar, r0, workspace);
          auto child_slice = get_child_slice<NDIM>(key, K, bid);
          values(child_slice) = r0;
        }

        T fac = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5),T(NDIM*(1+key.level()))));
        values *= fac;
        // Inlined: filter<T,K,NDIM>(values,r);
        transform<NDIM>(values, hgT, r1, workspace);

        auto child_slice = get_child_slice<NDIM>(key, K, 0);
        auto r_slice = r1(child_slice);
        coeffs = r_slice; // extract sum coeffs
        r_slice = 0.0; // zero sum coeffs so can easily compute norm of difference coeffs
        /* TensorView assignment synchronizes */
        T norm = mra::normf(r1);
        //std::cout << "project norm " << norm << " thresh " << thresh << std::endl;
        if (is_team_lead()) {
          *is_leaf = (norm < truncate_tol(key,thresh)); // test norm of difference coeffs
          if (!*is_leaf) {
            // std::cout << "fcoeffs not leaf " << key << " norm " << norm << std::endl;
          }
        }
      }
    }

    template<typename Fn, typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    fcoeffs_kernel(
      const Domain<NDIM>& D,
      const T* gldata,
      const Fn* fns,
      Key<NDIM> key,
      size_type N,
      size_type K,
      T* tmp,
      const TensorView<T, 2> phibar_view,
      const TensorView<T, 2> hgT_view,
      TensorView<T, NDIM+1>  coeffs_view,
      bool *is_leaf,
      T thresh)
    {
      /* set up temporaries once in each block */
      SHARED TensorView<T, NDIM> values, r0, r1, child_values, coeffs;
      SHARED TensorView<T, 2   > x_vec, x;
      SHARED T* workspace;
      if (is_team_lead()) {
        const size_type K2NDIM    = std::pow(K, NDIM);
        const size_type TWOK2NDIM = std::pow(2*K, NDIM);
        T* block_tmp = &tmp[blockIdx.x*fcoeffs_tmp_size<NDIM>(K)];
        values       = TensorView<T, NDIM>(&block_tmp[0], 2*K);
        r0           = TensorView<T, NDIM>(&block_tmp[TWOK2NDIM], K);
        r1           = TensorView<T, NDIM>(&block_tmp[TWOK2NDIM+K2NDIM], 2*K);
        child_values = TensorView<T, NDIM>(&block_tmp[2*TWOK2NDIM+K2NDIM], K);
        x_vec        = TensorView<T, 2   >(&block_tmp[2*TWOK2NDIM+2*K2NDIM], NDIM, K2NDIM);
        x            = TensorView<T, 2   >(&block_tmp[2*TWOK2NDIM+(NDIM+2)*K2NDIM], NDIM, K);
        workspace    = &block_tmp[2*TWOK2NDIM+(NDIM+2)*K2NDIM+NDIM*K];
      }

      /* adjust pointers for the function of each block */
      for (size_type fnid = blockIdx.x; fnid < N; fnid += gridDim.x) {
        if (is_team_lead()) {
          /* get the coefficient inputs */
          coeffs       = coeffs_view(fnid);
        }
        SYNCTHREADS();
        fcoeffs_kernel_impl(D, gldata, fns[fnid], key, K, fnid,
                            values, r0, r1, child_values, x_vec, x, workspace,
                            phibar_view, hgT_view, coeffs,
                            &is_leaf[fnid], thresh);
      }
    }
  } // namespace detail

  /**
   * Fcoeffs used in project
   */
  template<typename Fn, typename T, mra::Dimension NDIM>
  void submit_fcoeffs_kernel(
      const mra::Domain<NDIM>& D,
      const T* gldata,
      const Fn* fns,
      const mra::Key<NDIM>& key,
      size_type N,
      size_type K,
      T* tmp,
      const mra::TensorView<T, 2>& phibar_view,
      const mra::TensorView<T, 2>& hgT_view,
      mra::TensorView<T, NDIM+1>& coeffs_view,
      bool* is_leaf_scratch,
      T thresh,
      ttg::device::Stream stream)
  {
    /**
     * Launch the kernel with KxKxK threads in each of the N blocks.
     * Computation on functions is embarassingly parallel and no
     * synchronization is required.
     */
    Dim3 thread_dims = max_thread_dims(K);

    /* launch one block per child */
    CALL_KERNEL(detail::fcoeffs_kernel, N, thread_dims, 0, stream,
      (D, gldata, fns, key, N, K, tmp,
       phibar_view, hgT_view, coeffs_view,
       is_leaf_scratch, thresh));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_FCOEFFS_H
