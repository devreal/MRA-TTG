#ifndef MRA_KERNELS_DERIVATIVE_H
#define MRA_KERNELS_DERIVATIVE_H

#include <assert.h>
#include "mra/misc/dims.h"
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/platform.h"
#include "mra/misc/functiondata.h"
#include "mra/tensor/tensorview.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type derivative_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    return 6*K2NDIM; // workspace, left_tmp, center_tmp, right_tmp and 2xtmp_result
  }

  template <typename T, Dimension NDIM>
  SCOPE void parent_to_child(
    const Domain<NDIM>& D,
    const Key<NDIM>& parent,
    const Key<NDIM>& child,
    const TensorView<T, NDIM>& coeffs,
    TensorView<T, NDIM>& result,
    TensorView<T, NDIM>& result_tmp,
    const TensorView<T, 2>& phibar,
    const TensorView<T, 2>& phi,
    const TensorView<T, 1>& quad_x,
    const size_type K,
    T* tmp)
    {
      if (parent == child || parent.is_invalid() || child.is_invalid()) {
        result = coeffs;
      } else {
        fcube_for_mul(D, child, parent, coeffs, result_tmp, phibar, phi, quad_x, K, tmp);
        //std::cout << "PARENT TO CHILD: parent " << parent << " child " << child
        //          << " phibar " << normf(phibar)
        //          << " fcube_for_mul result " << normf(result_tmp)
        //          << " scale " << std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*child.level()))) << std::endl;
        T scale = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*child.level())));
        result_tmp *= scale;
        //std::cout << "PARENT TO CHILD: result_tmp " << result_tmp << std::endl;
        //std::cout << "PARENT TO CHILD: phibar " << phibar << std::endl;
        transform(result_tmp, phibar, result, tmp);
        //std::cout << "PARENT TO CHILD: result " << result << std::endl;
      }
    }
  namespace detail {

  #if 0
    template <typename T, Dimension NDIM>
    SCOPE bool enforce_bc(int bc_left, int bc_right, const Level& n, Translation& l) {
      Translation two2n = 1ul << n;
        if (l < 0){
          if(bc_left == FunctionData<T, NDIM>::BC_ZERO || bc_left == FunctionData<T, NDIM>::BC_FREE ||
              bc_left == FunctionData<T, NDIM>::BC_DIRICHLET || bc_left == FunctionData<T, NDIM>::BC_ZERONEUMANN ||
              bc_left == FunctionData<T, NDIM>::BC_NEUMANN){
            return false;
          }
          else if (bc_left == FunctionData<T, NDIM>::BC_PERIODIC){
            l += two2n;
            assert(bc_left == bc_right);
        }
          else {
            throw std::runtime_error("Invalid boundary condition");
          }
        }
        else if (l >= two2n){
          if(bc_right == FunctionData<T, NDIM>::BC_ZERO || bc_right == FunctionData<T, NDIM>::BC_FREE ||
              bc_right == FunctionData<T, NDIM>::BC_DIRICHLET || bc_right == FunctionData<T, NDIM>::BC_ZERONEUMANN ||
              bc_right == FunctionData<T, NDIM>::BC_NEUMANN){
            return false;
          }
          else if (bc_right == FunctionData<T, NDIM>::BC_PERIODIC){
            l -= two2n;
            assert(bc_left == bc_right);
        }
          else {
            throw std::runtime_error("Invalid boundary condition");
          }
        }
        return true;
    }

    template <Dimension NDIM>
    Key<NDIM> neighbor(
      const Key<NDIM>& key,
      size_type step,
      size_type axis,
      const int bc_left,
      const int bc_right)
    { // TODO: check for boundary conditions to return an invalid key
      auto res = key.step(axis, step);
      if (!enforce_bc(bc_left, bc_right, key.level(), res.translation()[axis])){
        return Key<NDIM>::invalid();
      }
      return res;
    }
    #endif

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_inner(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>& left,
      const Key<NDIM>& center,
      const Key<NDIM>& right,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 3>& operators,
      TensorView<T, NDIM>& deriv,
      TensorView<T, NDIM+1>& tmp,
      TensorView<T, NDIM>& left_tmp,
      TensorView<T, NDIM>& center_tmp,
      TensorView<T, NDIM>& right_tmp,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 1>& quad_x,
      const int bc_left,
      const int bc_right,
      size_type axis,
      size_type K,
      T* workspace)
    {
      SHARED TensorView<T, NDIM> tmp_result;
      SHARED TensorView<T, NDIM> transform_result;
      if (is_team_lead()){
        tmp_result = tmp(0);
        transform_result = tmp(1);
      }
      SYNCTHREADS();

      deriv = 0;

      std::cout << "DERIVATIVE " << key << " axis " << axis << std::endl;

      parent_to_child(D, left,   key.neighbor(axis, -1), node_left, left_tmp, tmp_result, phibar, phi, quad_x, K, workspace);
      transform_dir(left_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::RPT), tmp_result, transform_result, axis);
      deriv += transform_result;
      //std::cout << "DERIVATIVE " << key << " axis " << axis  << " left " << left << " " << normf(node_left)<< " left_tmp " << normf(left_tmp) << std::endl;

      parent_to_child(D, center, key, node_center, center_tmp, tmp_result, phibar, phi, quad_x, K, workspace);
      transform_dir(center_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::R0T), tmp_result, transform_result, axis);
      deriv += transform_result;
      //std::cout << "DERIVATIVE " << key << " axis " << axis << " center " << center << " "
      //          << normf(node_center) << " center_tmp " << normf(center_tmp) << " deriv " << normf(deriv) << std::endl;

      parent_to_child(D, right,  key.neighbor(axis, 1), node_right, right_tmp, tmp_result, phibar, phi, quad_x, K, workspace);
      transform_dir(right_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::RMT), tmp_result, transform_result, axis);
      deriv += transform_result;

      //std::cout << "DERIVATIVE " << key << " axis " << axis << " right " << right << " neighbor " << key.neighbor(axis, 1) << " "
      //          << normf(node_right) << " right_tmp " << normf(right_tmp) << std::endl;

      T scale = D.template get_reciprocal_width<T>(axis)*std::pow(T(2), T(key.level()));
      T thresh = T(1e-12);
      deriv *= scale;
      deriv.reduce_rank(thresh);
      //std::cout << "INNER " << key << " axis " << axis << " scale " << scale << " RESULT "<< " deriv " << normf(deriv) << std::endl;
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_boundary(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>& left,
      const Key<NDIM>& center,
      const Key<NDIM>& right,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 2+1>& operators,
      TensorView<T, NDIM>& deriv,
      TensorView<T, NDIM+1>& tmp,
      TensorView<T, NDIM>& left_tmp,
      TensorView<T, NDIM>& center_tmp,
      TensorView<T, NDIM>& right_tmp,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 1>& quad_x,
      const T g1,
      const T g2,
      const int bc_left,
      const int bc_right,
      size_type axis,
      size_type K,
      T* workspace)
    {
      SHARED TensorView<T, NDIM> tmp_result;
      SHARED TensorView<T, NDIM> transform_result;
      if (is_team_lead()){
        tmp_result = tmp(0);
        transform_result = tmp(1);
      }
      SYNCTHREADS();


      deriv = T(0);

      if (key.is_left_boundary(axis)){

        parent_to_child(D, right, key.neighbor(axis, 1), node_right, right_tmp, tmp_result, phibar, phi, quad_x, K, workspace);
        parent_to_child(D, center, key, node_center, center_tmp, tmp_result, phibar, phi, quad_x, K, workspace);

        transform_dir(right_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::RMT), tmp_result, transform_result, axis);
        deriv += transform_result;
        //std::cout << "BOUNDARY " << key << " axis " << axis << " right_tmp " << normf(right_tmp) << " deriv " << normf(deriv) << std::endl;
        transform_dir(center_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::R0T), tmp_result, transform_result, axis);
        deriv += transform_result;
        //std::cout << "BOUNDARY " << key << " axis " << axis << " center_tmp " << normf(center_tmp) << " deriv " << normf(deriv) << std::endl;
      }
      else {
        parent_to_child(D, center, key, node_center, center_tmp, tmp_result, phibar, phi, quad_x, K, workspace);
        parent_to_child(D, left, key.neighbor(axis, -1), node_left, left_tmp, tmp_result, phibar, phi, quad_x, K, workspace);

        transform_dir(center_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::RIGHT_R0T), tmp_result, transform_result, axis);
        deriv += transform_result;
        //std::cout << "BOUNDARY " << key << " axis " << axis << " center_tmp " << normf(center_tmp) << " deriv " << normf(deriv) << std::endl;
        transform_dir(left_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::RIGHT_RPT), tmp_result, transform_result, axis);
        deriv += transform_result;
        //std::cout << "BOUNDARY " << key << " axis " << axis << " left_tmp " << normf(left_tmp) << " deriv " << normf(deriv) << std::endl;
      }

      T scale = D.template get_reciprocal_width<T>(axis)*std::pow(T(2), T(key.level()));
      T thresh = T(1e-12);
      deriv *= scale;
      deriv.reduce_rank(thresh);

    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>& left,
      const Key<NDIM>& center,
      const Key<NDIM>& right,
      const TensorView<T, NDIM>& node_left,
      const TensorView<T, NDIM>& node_center,
      const TensorView<T, NDIM>& node_right,
      const TensorView<T, 3>& operators,
      TensorView<T, NDIM>& deriv,
      const TensorView<T, 2>& phi,
      const TensorView<T, 2>& phibar,
      const TensorView<T, 1>& quad_x,
      T* tmp,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const int bc_left,
      const int bc_right)
      {
        // if we reached here, all checks have passed, and we do the transform to compute the derivative
        // for a given axis by calling either derivative_inner() or derivative_boundary()
        SHARED TensorView<T, NDIM> left_tmp, center_tmp, right_tmp;
        SHARED TensorView<T, NDIM+1> tmp_result;
        SHARED T* workspace;

        size_type blockId = blockIdx.x;
        T* block_tmp_ptr = &tmp[blockId*derivative_tmp_size<NDIM>(K)];
        const size_type K2NDIM = std::pow(K, NDIM);
        if(is_team_lead()){
          tmp_result = TensorView<T, NDIM+1>(&block_tmp_ptr[       0], make_dims<NDIM+1>(2, K));
          left_tmp   = TensorView<T, NDIM>(&block_tmp_ptr[2*K2NDIM], K);
          center_tmp = TensorView<T, NDIM>(&block_tmp_ptr[3*K2NDIM], K);
          right_tmp  = TensorView<T, NDIM>(&block_tmp_ptr[4*K2NDIM], K);
          workspace = &block_tmp_ptr[5*K2NDIM];
        }
        SYNCTHREADS();

        if (key.is_boundary(axis)){
          derivative_boundary<T, NDIM>(D, key, left, center, right, node_left, node_center, node_right,
            operators, deriv, tmp_result, left_tmp, center_tmp, right_tmp, phi, phibar, quad_x, g1, g2,
            bc_left, bc_right, axis, K, workspace);
        }
        else{
          derivative_inner<T, NDIM>(D, key, left, center, right, node_left, node_center, node_right,
            operators, deriv, tmp_result, left_tmp, center_tmp, right_tmp, phi, phibar, quad_x, bc_left,
            bc_right, axis, K, workspace);
        }
      }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void derivative_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const Key<NDIM> left,
      const Key<NDIM> center,
      const Key<NDIM> right,
      const TensorView<T, NDIM+1> node_left,
      const TensorView<T, NDIM+1> node_center,
      const TensorView<T, NDIM+1> node_right,
      const TensorView<T, 3> operators,
      TensorView<T, NDIM+1> deriv,
      const TensorView<T, 2> phi,
      const TensorView<T, 2> phibar,
      const TensorView<T, 1> quad_x,
      T* tmp,
      size_type N,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const int bc_left,
      const int bc_right)
    {
      SHARED TensorView<T, NDIM> node_left_view, node_center_view, node_right_view, deriv_view;
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (is_team_lead()) {
          node_left_view = node_left(blockid);
          node_center_view = node_center(blockid);
          node_right_view = node_right(blockid);
          deriv_view = deriv(blockid);
        }
        SYNCTHREADS();
        derivative_kernel_impl<T, NDIM>(D, key, left, center, right, node_left_view, node_center_view, node_right_view,
          operators, deriv_view, phi, phibar, quad_x, tmp, K, g1, g2, axis, bc_left, bc_right);
      }
    }

  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_derivative_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const Key<NDIM>& left,
    const Key<NDIM>& center,
    const Key<NDIM>& right,
    const TensorView<T, NDIM+1>& node_left,
    const TensorView<T, NDIM+1>& node_center,
    const TensorView<T, NDIM+1>& node_right,
    const TensorView<T, 3>& operators,
    TensorView<T, NDIM+1>& deriv,
    const TensorView<T, 2>& phi,
    const TensorView<T, 2>& phibar,
    const TensorView<T, 1>& quad_x,
    T* tmp,
    size_type N,
    size_type K,
    const T g1,
    const T g2,
    size_type axis,
    const int bc_left,
    const int bc_right,
    ttg::device::Stream stream)
  {
    size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
    Dim3 thread_dims = Dim3(max_threads, max_threads, 1);

    auto smem_size = std::max(static_cast<size_type>(K*K*NDIM*sizeof(T)), // used in fcube_for_mul
                              mTxmq_shmem_size<T>(2*K));

    CONFIGURE_KERNEL((detail::derivative_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::derivative_kernel, N, thread_dims, smem_size, stream,
      (D, key, left, center, right, node_left, node_center, node_right, operators,
        deriv, phi, phibar, quad_x, tmp, N, K, g1, g2, axis, bc_left, bc_right));
    checkSubmit();
  }

  /* explicit instanatiation */
  extern template
  void submit_derivative_kernel<double, 3>(
    const Domain<3>& D,
    const Key<3>& key,
    const Key<3>& left,
    const Key<3>& center,
    const Key<3>& right,
    const TensorView<double, 3+1>& node_left,
    const TensorView<double, 3+1>& node_center,
    const TensorView<double, 3+1>& node_right,
    const TensorView<double, 3>& operators,
    TensorView<double, 3+1>& deriv,
    const TensorView<double, 2>& phi,
    const TensorView<double, 2>& phibar,
    const TensorView<double, 1>& quad_x,
    double* tmp,
    size_type N,
    size_type K,
    const double g1,
    const double g2,
    size_type axis,
    const int bc_left,
    const int bc_right,
    ttg::device::Stream stream);

} // namespace mra

#endif // MRA_KERNELS_DERIVATIVE_H
