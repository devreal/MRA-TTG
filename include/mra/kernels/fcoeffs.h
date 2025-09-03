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
#include "mra/tensor/functionnode.h"
#include "mra/misc/functiondata.h"
#include "mra/tensor/functionnorm.h"
#include "mra/kernels/kernel_state.h"

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

        /* compute all children */
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


  template<typename Fn, typename T, mra::Dimension NDIM>
  class FcoeffsKernel {

    using key_type = typename mra::Key<NDIM>;
    using node_type = typename mra::FunctionsReconstructedNode<T, NDIM>;

    const ttg::Buffer<mra::Domain<NDIM>>& m_domain;
    const ttg::Buffer<const T>& m_gl;
    const ttg::Buffer<Fn>& m_fns;
    const mra::Key<NDIM> m_key;
    const mra::FunctionData<T, NDIM>& m_fndata;
    const T m_thresh;
    node_type& m_result; // empty for fast-paths, no need to zero out
    ttg::Buffer<bool, DeviceAllocator<bool>> m_is_leafs;
    ttg::Buffer<T, DeviceAllocator<T>> m_scratch;
    FunctionNorms<T, NDIM> m_norms;
    detail::KernelState m_state = detail::KernelState::Initialized;

  public:

    FcoeffsKernel(const ttg::Buffer<mra::Domain<NDIM>>& domain,
                  const ttg::Buffer<const T>& gl,
                  const ttg::Buffer<Fn>& fns,
                  const mra::Key<NDIM>& key,
                  const mra::FunctionData<T, NDIM>& fndata,
                  const T thresh,
                  node_type& result,
                  std::string name = "fcoeffs")
    : m_domain(domain)
    , m_gl(gl)
    , m_fns(fns)
    , m_key(key)
    , m_fndata(fndata)
    , m_thresh(thresh)
    , m_result(result)
    , m_is_leafs(m_result.count())
    , m_scratch(fcoeffs_tmp_size<NDIM>(m_result.coeffs().dim(m_result.ndim()-1)), TempScope)
    , m_norms(name, result)
    { }


    /**
     * Returns the buffers to use to select a device, using ttg::device::select().
     */
    auto select() {
      assert(m_state == detail::KernelState::Initialized);
      const auto& phibar = m_fndata.get_phibar();
      const auto& hgT = m_fndata.get_hgT();
      m_state = detail::KernelState::Select;
#ifndef MRA_ENABLE_HOST
      return ttg::device::select(m_domain, m_gl, m_fb, m_result.coeffs().buffer(), phibar.buffer(),
                                 hgT.buffer(), m_scratch, m_is_leafs, m_norms.buffer());
#else
      return;
#endif
    }

    /**
     * Submit the kernel. select() must have been called before.
     * Returns the buffers that should be waited on, just like wait();
     */
    auto submit() {
      assert(m_state == detail::KernelState::Select);
      m_state = detail::KernelState::Submit;

      /**
       * Launch the kernel with KxKxK threads in each of the N blocks.
       * Computation on functions is embarassingly parallel and no
       * synchronization is required.
       */
      size_type K = m_result.coeffs().dim(m_result.ndim()-1);
      size_type N = m_result.count();
      Dim3 thread_dims = max_thread_dims(K);

      const auto& phibar = m_fndata.get_phibar();
      const auto& hgT = m_fndata.get_hgT();
      auto smem_size = mTxmq_shmem_size<T>(2*K);
      CONFIGURE_KERNEL((detail::fcoeffs_kernel<Fn, T, NDIM>), smem_size);
      /* launch one block per child */
      CALL_KERNEL(detail::fcoeffs_kernel, N, thread_dims, smem_size, ttg::device::current_stream(),
        (*m_domain.current_device_ptr(),
         m_gl.current_device_ptr(),
         m_fns.current_device_ptr(),
         m_key, N, K,
         m_scratch.current_device_ptr(),
         phibar.current_view(), hgT.current_view(), m_result.coeffs().current_view(),
         m_is_leafs.current_device_ptr(), m_thresh));
      checkSubmit();

      m_norms.compute();

#ifndef MRA_ENABLE_HOST
      return ttg::device::wait(m_is_leafs, m_norms.buffer());
#else
      return;
#endif
    }

    /**
     * Returns the buffers that should be transferred out of the device.
     * submit() must have been called before.
     */
    auto wait() {
      assert(m_state == detail::KernelState::Submit);
      m_state = detail::KernelState::Wait;
#ifndef MRA_ENABLE_HOST
      return ttg::device::wait(m_is_leafs, m_norms.buffer());
#else
      return;
#endif
    }

    /**
     * The epilogue, handling cleanup and norm checks (if enabled).
     * submit() or wait() must have been called before.
     */
    void epilogue() {
      assert(m_state == detail::KernelState::Wait || m_state == detail::KernelState::Submit);
      m_norms.verify(); // extracts the norms and stores them in the node
      m_state = detail::KernelState::Epilogue;
    }

    bool is_leaf(size_type idx) const {
      return m_is_leafs.host_ptr()[idx];
    }

  }; // FcoeffsKernel


#if 0
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

    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CONFIGURE_KERNEL((detail::fcoeffs_kernel<Fn, T, NDIM>), smem_size);
    /* launch one block per child */
    CALL_KERNEL(detail::fcoeffs_kernel, N, thread_dims, smem_size, stream,
      (D, gldata, fns, key, N, K, tmp,
       phibar_view, hgT_view, coeffs_view,
       is_leaf_scratch, thresh));
    checkSubmit();
  }
#endif // 0

} // namespace mra

#endif // MRA_KERNELS_FCOEFFS_H
