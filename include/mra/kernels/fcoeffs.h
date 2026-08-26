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
#include "mra/tensor/leafstatus.h"

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
    DEVSCOPE bool fcoeffs_kernel_impl(
      const Domain<NDIM>& D,
      const T* gldata,
      const Fn& f,
      Key<NDIM> key,
      size_type K,
      size_type fnid,
      /* temporaries */
      concepts::TensorView<NDIM> auto& values,
      concepts::TensorView<NDIM> auto& r0,
      concepts::TensorView<NDIM> auto& r1,
      concepts::TensorView<NDIM> auto& child_values,
      concepts::TensorView<2> auto& x_vec,
      concepts::TensorView<2> auto& x,
      T* workspace, /* variable size so pointer only */
      /* constants */
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& hgT,
      /* result */
      concepts::TensorView<NDIM> auto& coeffs,
      T thresh,
      T truncate_tol)
    {

      /* compute all children */
      for (int bid = 0; bid < key.num_children(); bid++) {
        Key<NDIM> child = key.child_at(bid);
        child_values = 0.0; // TODO: needed?
        fcube(D, gldata, f, child, truncate_tol, child_values, K, x, x_vec);
        transform(child_values, phibar, r0, workspace);
        auto child_slice = get_child_slice<NDIM>(key, K, bid);
        values(child_slice) = r0;
      }

      T fac = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5),T(NDIM*(1+key.level()))));
      values *= fac;
      // Inlined: filter<T,K,NDIM>(values,r);
      transform(values, hgT, r1, workspace);

      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      auto r_slice = r1(child_slice);
      coeffs = r_slice; // extract sum coeffs
      r_slice = 0.0; // zero sum coeffs so can easily compute norm of difference coeffs
      /* TensorView assignment synchronizes */
      T norm = mra::normf(r1);
      //std::cout << "project norm " << norm << " thresh " << thresh << std::endl;
      bool is_leaf_val = (norm < truncate_tol);
      return is_leaf_val;
    }

    template<typename Fn, typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    fcoeffs_kernel(
      const Domain<NDIM>& D,
      const T* gldata,
      const DenseTensorView<Fn, 1> fns,
      Key<NDIM> key,
      size_type K,
      T* tmp,
      const concepts::TensorView<2> auto phibar_view,
      const concepts::TensorView<2> auto hgT_view,
      concepts::TensorView<NDIM+1> auto coeffs_view,
      T thresh,
      T truncate_tol,
      const DenseTensorView<LeafStatus, 1> leaf_info_view,
      DenseTensorView<LeafStatus, 1> result_leaf_info_view,
      size_type n_nonzero)
    {
      /* set up temporaries once in each block */
      SHARED DenseTensorView<T, NDIM> values, r0, r1, child_values, coeffs;
      SHARED DenseTensorView<T, 2   > x_vec, x;
      SHARED T* workspace;
      SHARED size_type fnid;
      size_type N = fns.dim(0);

      /* Functions excluded from coeffs_view's sparsity host-side (already
       * leaf/invalid, or negligible -- see mra/tasks/project.h) never reach
       * this loop at all; their result_leaf_info entries are pre-filled with
       * Invalid host-side, matching what this kernel would otherwise have
       * set for them below. The checks below are kept as-is (redundant, but
       * harmless) for functions that DO reach here -- e.g. the initial_level
       * check, which the host does not pre-evaluate per function. */
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          // coeffs_view has exactly n_nonzero non-zero entries, so this
          // always finds a valid function id -- see submit_fcoeffs_kernel.
          fnid = find_nth_nonzero(N, pos, coeffs_view);

          const size_type K2NDIM    = std::pow(K, NDIM);
          const size_type TWOK2NDIM = std::pow(2*K, NDIM);
          T* block_tmp = &tmp[pos*fcoeffs_tmp_size<NDIM>(K)];
          values       = DenseTensorView<T, NDIM>(&block_tmp[0], 2*K);
          r0           = DenseTensorView<T, NDIM>(&block_tmp[TWOK2NDIM], K);
          r1           = DenseTensorView<T, NDIM>(&block_tmp[TWOK2NDIM+K2NDIM], 2*K);
          child_values = DenseTensorView<T, NDIM>(&block_tmp[2*TWOK2NDIM+K2NDIM], K);
          x_vec        = DenseTensorView<T, 2   >(&block_tmp[2*TWOK2NDIM+2*K2NDIM], NDIM, K2NDIM);
          x            = DenseTensorView<T, 2   >(&block_tmp[2*TWOK2NDIM+(NDIM+2)*K2NDIM], NDIM, K);
          workspace    = &block_tmp[2*TWOK2NDIM+(NDIM+2)*K2NDIM+NDIM*K];

          /* carry over leaf info */
          result_leaf_info_view[fnid] = leaf_info_view[fnid];
        }
        SYNCTHREADS();

        auto& f = fns(fnid);
        // if we have seen a leaf for this function, skip and set the status to Invalid
        if (leaf_info_view(fnid) == LeafStatus::Leaf || leaf_info_view(fnid) == LeafStatus::Invalid) {
          if (is_team_lead()) {
            result_leaf_info_view(fnid) = LeafStatus::Invalid;
          }
          continue; // skip leaf and invalid entries
        }
        /* check for our function */
        if ((key.level() < initial_level(f))) {
          // std::cout << "project: key " << key << " below intial level " << initial_level(f) << std::endl;
          if (is_team_lead()) {
            //std::cout << "FCOEFFS: key " << key << " function " << fnid << " above initial level, marking as Inner" << std::endl;
            result_leaf_info_view(fnid) = LeafStatus::Inner;
            coeffs_view.set_zero(fnid);
          }
          continue;
        }
        if (is_negligible<Fn,T,NDIM>(f, D.template bounding_box<T>(key), truncate_tol)) {
          /* set leaf status to Inner */
          if (is_team_lead()) {
            //std::cout << "FCOEFFS: key " << key << " negligible with tol " << truncate_tol << std::endl;
            result_leaf_info_view(fnid) = LeafStatus::Invalid;
            /* zero coeffs */
            coeffs_view.set_zero(fnid);
          }
          continue;
        }
        // if we have not seen a leaf for this function it might still be zero (one of our siblings will be non-zero)
        if (coeffs_view.is_zero(fnid)) {
          continue;
        }
        if (is_team_lead()) {
          /* get the coefficient inputs */
          coeffs = coeffs_view(fnid);
        }
        SYNCTHREADS();
        bool is_leaf_val = fcoeffs_kernel_impl(D, gldata, f, key, K, fnid,
                            values, r0, r1, child_values, x_vec, x, workspace,
                            phibar_view, hgT_view, coeffs, thresh, truncate_tol);
        if (is_team_lead()) {
          if (is_leaf_val) {
            //std::cout << "FCOEFFS: key " << key << " function " << fnid << " is a leaf, zeroing coeffs" << std::endl;
            result_leaf_info_view(fnid) = LeafStatus::Leaf;
          } else {
            //std::cout << "FCOEFFS: key " << key << " function " << fnid << " is not a leaf, marking as Inner" << std::endl;
            result_leaf_info_view(fnid) = LeafStatus::Inner;
            coeffs_view.set_zero(fnid);
          }
        }
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
      const DenseTensorView<Fn, 1>& fns,
      const mra::Key<NDIM>& key,
      size_type K,
      T* tmp,
      const concepts::TensorView<2> auto& phibar_view,
      const concepts::TensorView<2> auto& hgT_view,
      concepts::TensorView<NDIM+1> auto& coeffs_view,
      T thresh,
      T truncate_tol,
      const DenseTensorView<LeafStatus, 1>& leaf_info_view,
      DenseTensorView<LeafStatus, 1>& result_leaf_info_view,
      size_type n_nonzero,
      ttg::device::Stream stream)
  {
    /**
     * Launch the kernel with KxKxK threads in each of the n_nonzero blocks.
     * Computation on functions is embarassingly parallel and no
     * synchronization is required.
     */
    Dim3 thread_dims = max_thread_dims(K);

    auto smem_size = mTxmq_shmem_size<T>(2*K);
    //CONFIGURE_KERNEL((detail::fcoeffs_kernel<Fn, T, NDIM>), smem_size);
    /* launch one block per non-zero function */
    CALL_KERNEL(detail::fcoeffs_kernel, n_nonzero, thread_dims, smem_size, stream,
      (D, gldata, fns, key, K, tmp,
       phibar_view, hgT_view, coeffs_view,
       thresh, truncate_tol, leaf_info_view, result_leaf_info_view, n_nonzero));
    checkSubmit();
  }

} // namespace mra

#endif // MRA_KERNELS_FCOEFFS_H
