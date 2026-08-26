#ifndef MRA_KERNELS_MULTIPLY_H
#define MRA_KERNELS_MULTIPLY_H

#include "mra/misc/maxk.h"
#include "mra/misc/key.h"
#include "mra/misc/domain.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"
#include "mra/tensor/child_slice.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type multiply_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    const size_type TWOK2NDIM = std::pow(2*K, NDIM);
    return 32*K2NDIM + 3 * TWOK2NDIM; // workspace, r1*8, cnodeA*8, cnodeB*8,
  }

  namespace detail {

    /**
     * The abbreviated (concept-constrained `auto`) function-parameter form is
     * used everywhere else in the kernels, but nvcc's EDG front end hits an
     * internal assertion ("check_name_hiding_by_template_parameters") once
     * enough compiler-synthesized template parameters accumulate across a
     * translation unit -- this function alone contributed 13 of them, enough
     * to push the whole file over that threshold regardless of instantiation
     * (reproduces from a bare #include). Spelling the same constraints as
     * explicitly-named template parameters is semantically identical --
     * still deduced from the call site, still concept-checked, no loss of
     * genericity -- it just avoids the specific (buggy) EDG code path for
     * abbreviated templates. See git history for the nvcc repro.
     */
    template <typename T, Dimension NDIM,
              concepts::TensorView<NDIM> ViewNodeA,
              concepts::TensorView<NDIM> ViewNodeB,
              concepts::TensorView<NDIM+1> ViewCnodesA,
              concepts::TensorView<NDIM+1> ViewCnodesB,
              concepts::TensorView<NDIM> ViewCnodeR,
              concepts::TensorView<NDIM> ViewCnodeD,
              concepts::TensorView<NDIM> ViewNodeR,
              concepts::TensorView<NDIM+1> ViewR1,
              concepts::TensorView<2> ViewHgT,
              concepts::TensorView<2> ViewPhi,
              concepts::TensorView<2> ViewPhiT,
              concepts::TensorView<2> ViewPhibar,
              concepts::TensorView<1> ViewQuadX>
    DEVSCOPE void multiply_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& keyA,
      const Key<NDIM>& keyB,
      const ViewNodeA& nodeA,
      const ViewNodeB& nodeB,
      ViewCnodesA& cnodesA,
      ViewCnodesB& cnodesB,
      ViewCnodeR& cnodeR,
      ViewCnodeD& cnodeD,
      ViewNodeR& nodeR,
      ViewR1& r1,
      T* workspace,
      const ViewHgT& hgT,
      const ViewPhi& phi,
      const ViewPhiT& phiT,
      const ViewPhibar& phibar,
      const ViewQuadX& quad_x,
      size_type K)
    {
      Key<NDIM> target;
      if (keyA.level()>keyB.level()) target = keyA;
      else target = keyB;
      T scale;

      for (int i=0; i< keyA.num_children(); ++i){
        auto child = target.child_at(i);
        auto cnodeA = cnodesA(i);
        auto cnodeB = cnodesB(i);
        fcube_for_mul(D, child, keyB, nodeB, cnodeB, phibar, phi, quad_x, K, workspace);
        fcube_for_mul(D, child, keyA, nodeA, cnodeA, phibar, phi, quad_x, K, workspace);
        scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*child.level())));
        cnodeB *= scale;
        cnodeA *= scale;
      }

      // fcube_for_mul() returns function values evaluated at quadrature points
      foreach_idx(cnodesA, [&](size_type i) {
        cnodesA[i] = cnodesA[i] * cnodesB[i];
      });

      // convert back to coeffs
      for (int i=0; i< keyA.num_children(); ++i){
        auto cnodeA = cnodesA(i);
        auto r = r1(i);
        transform(cnodeA, phibar, r, workspace);
      }

      // compress the result(r1 which is NDIM+1 tensorview) and store scaling functions to nodeR
      for (int i = 0; i<target.num_children(); ++i) {
        auto child_slice = get_child_slice<NDIM>(target, K, i);
        const auto& in = r1(i);
        cnodeR(child_slice) = in;
      }

      transform(cnodeR, hgT, cnodeD, workspace);
      if (keyA.level() > 0) {
        auto child_slice = get_child_slice<NDIM>(target, K, 0);
        nodeR = cnodeD(child_slice);
      }
    }

    template <typename T, Dimension NDIM>
    GLOBALSCOPE void
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    multiply_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM>& keyA,
      const Key<NDIM>& keyB,
      const concepts::TensorView<NDIM+1> auto nodeA_view,
      const concepts::TensorView<NDIM+1> auto nodeB_view,
      concepts::TensorView<NDIM+1> auto nodeR_view,
      T* tmp,
      const concepts::TensorView<2> auto hgT,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<2> auto phiT,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<1> auto quad_x,
      size_type N,
      size_type n_nonzero,
      size_type K)
    {
      SHARED DenseTensorView<const T, NDIM> nodeA, nodeB;
      SHARED DenseTensorView<T, NDIM> nodeR, cnodesR, cnodesD;
      SHARED DenseTensorView<T, NDIM+1> cnodesA, cnodesB, r1;
      SHARED T* workspace;
      SHARED size_type fnid;

      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          // nodeR_view has exactly n_nonzero non-zero entries, so this
          // always finds a valid function id -- see submit_multiply_kernel.
          fnid = find_nth_nonzero(N, pos, nodeR_view);
          const size_type K2NDIM = std::pow(K, NDIM);
          const size_type TWO2NDIM = std::pow(2, NDIM);
          const size_type TWOK2NDIM = std::pow(2*K, NDIM);
          T* block_tmp = &tmp[pos*multiply_tmp_size<NDIM>(K)];
          r1        = DenseTensorView<T, NDIM+1>(&block_tmp[        0], TWO2NDIM, K, K, K);
          cnodesA   = DenseTensorView<T, NDIM+1>(&block_tmp[ 8*K2NDIM], TWO2NDIM, K, K, K);
          cnodesB   = DenseTensorView<T, NDIM+1>(&block_tmp[16*K2NDIM], TWO2NDIM, K, K, K);
          cnodesR   = DenseTensorView<T, NDIM>(&block_tmp  [24*K2NDIM], 2*K, 2*K, 2*K);
          cnodesD   = DenseTensorView<T, NDIM>(&block_tmp  [32*K2NDIM + TWOK2NDIM], 2*K, 2*K, 2*K);
          workspace = &block_tmp[32*K2NDIM + 2*TWOK2NDIM];
          nodeA = nodeA_view(fnid);
          nodeB = nodeB_view(fnid);
          nodeR = nodeR_view(fnid);
        }
        SYNCTHREADS();
        multiply_kernel_impl<T, NDIM>(D, keyA, keyB, nodeA, nodeB, cnodesA, cnodesB,
           cnodesR, cnodesD, nodeR, r1, workspace, hgT, phi, phiT, phibar, quad_x, K);
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_multiply_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& keyA,
    const Key<NDIM>& keyB,
    const concepts::TensorView<NDIM+1> auto& funcA,
    const concepts::TensorView<NDIM+1> auto& funcB,
    concepts::TensorView<NDIM+1> auto& funcR,
    const concepts::TensorView<2> auto& hgT,
    const concepts::TensorView<2> auto& phi,
    const concepts::TensorView<2> auto& phiT,
    const concepts::TensorView<2> auto& phibar,
    const concepts::TensorView<1> auto& quad_x,
    size_type N,
    size_type n_nonzero,
    size_type K,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    //CONFIGURE_KERNEL((detail::multiply_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::multiply_kernel, n_nonzero, thread_dims, smem_size, stream,
      (D, keyA, keyB, funcA, funcB, funcR, tmp, hgT, phi, phiT, phibar,
        quad_x, N, n_nonzero, K));
    checkSubmit();
  }

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* explicit instanatiation */
  extern template
  void submit_multiply_kernel<double, 3>(
    const Domain<3>& D,
    const Key<3>& keyA,
    const Key<3>& keyB,
    const SparseTensorView<double, 3+1>& funcA,
    const SparseTensorView<double, 3+1>& funcB,
    SparseTensorView<double, 3+1>& funcR,
    const SparseTensorView<double, 2>& hgT,
    const SparseTensorView<double, 2>& phi,
    const SparseTensorView<double, 2>& phiT,
    const SparseTensorView<double, 2>& phibar,
    const SparseTensorView<double, 1>& quad_x,
    size_type N,
    size_type n_nonzero,
    size_type K,
    double* tmp,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_H
