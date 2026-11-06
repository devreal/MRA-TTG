#ifndef MRA_KERNELS_DERIVATIVE_H
#define MRA_KERNELS_DERIVATIVE_H

#include <assert.h>
#include <tuple>
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/platform.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"

namespace mra {

  template<mra::Dimension NDIM>
  SCOPE size_type derivative_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K,NDIM);
    return 6*K2NDIM; // workspace, left_tmp, center_tmp, right_tmp and 2xtmp_result
  }

#ifdef MRA_HAVE_TRANSFORM_DMMA
  namespace detail {
    /**
     * parent_to_child for a child below its parent on FP64 tensor cores, see
     * parent_to_child: the values at the child's quadrature points of the
     * parent's polynomials, phi[d](i, mu) (phi_for_mul without its scaling, all
     * dimensions at once, in dynamic shared memory as in fcube_for_mul), are
     * applied by one general_transform, then the values are projected by phibar.
     * The scales of fcube_for_mul and parent_to_child (phi_for_mul's 2^(np/2) per
     * dimension, 1/sqrt(vol) and sqrt(vol*2^(-NDIM*nc))) are applied at once in
     * the first transform: they come to 2^(NDIM*(np-nc)/2).
     * Returns false, without touching anything, if there is no tensor-core path.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE bool parent_to_child_mma(
      const Key<NDIM>& parent,
      const Key<NDIM>& child,
      const T* coeffs,
      T* result,
      T* result_tmp,
      const T* phibar,
      const concepts::TensorView<1> auto& quad_x,
      const size_type K,
      T* tmp)
    {
      if constexpr (NDIM == 3 && std::is_same_v<T, double>) {
        if (K == 0 || K > accel::mma::transform_mma_max_dim || block_size() < MRA_WARP_SIZE) return false;
        extern SHARED T phi[];
        const int np = parent.level(), nc = child.level();
        const T xscale = std::ldexp(T(1), np - nc);
        for (size_type idx = thread_id(); idx < NDIM*K; idx += block_size()) {
          const size_type d = idx / K, mu = idx % K;
          T* phid = phi + d*K*K;
          const T xmu = xscale * (quad_x(mu) + child.translation()[d]) - parent.translation()[d];
          const T x = T(2)*xmu - 1;
          T pm0 = 1.0, pm1 = x;
          phid[mu] = pm0;
          if (K > 1) phid[K + mu] = pm1 * std::sqrt(T(3));
          for (size_type i = 2; i < K; ++i) {
            const T pm2 = (x * pm1 - pm0) * (T(i-1) / T(i)) + x * pm1;
            phid[i*K + mu] = pm2 * std::sqrt(T(2*i + 1));
            pm0 = pm1;
            pm1 = pm2;
          }
        }
        SYNCTHREADS();
        const T alpha = std::sqrt(std::ldexp(T(1), int(NDIM)*(np - nc)));
        accel::mma::general_transform<T, NDIM>(K, coeffs, phi, phi + K*K, phi + 2*K*K, result_tmp, tmp, alpha);
        /* phi is dead, the transform may use the shared memory */
        return accel::mma::transform<T, NDIM>(K, result_tmp, phibar, result, tmp);
      }
      return false;
    }
  } // namespace detail
#endif // MRA_HAVE_TRANSFORM_DMMA

  template <typename T, Dimension NDIM>
  SCOPE void parent_to_child(
    const Domain<NDIM>& D,
    const Key<NDIM>& parent,
    const Key<NDIM>& child,
    const concepts::TensorView<NDIM> auto& coeffs,
    concepts::TensorView<NDIM> auto& result,
    concepts::TensorView<NDIM> auto& result_tmp,
    const concepts::TensorView<2> auto& phibar,
    const concepts::TensorView<2> auto& phi,
    const concepts::TensorView<1> auto& quad_x,
    const size_type K,
    T* tmp)
    {
      if (coeffs.data() == nullptr) {
        result = T(0); // the function has no coefficients at parent
      } else if (parent == child || parent.is_invalid() || child.is_invalid()) {
        result = coeffs;
      } else {
#ifdef MRA_HAVE_TRANSFORM_DMMA
        if (coeffs.size() == K*K*K && result.size() == K*K*K && result_tmp.size() == K*K*K &&
            phibar.dim(0) == K && phibar.dim(1) == K &&
            detail::parent_to_child_mma<T, NDIM>(parent, child, coeffs.data(), result.data(), result_tmp.data(),
                                                 phibar.data(), quad_x, K, tmp)) {
          return;
        }
#endif // MRA_HAVE_TRANSFORM_DMMA
        fcube_for_mul(D, child, parent, coeffs, result_tmp, phibar, phi, quad_x, K, tmp);
        //std::cout << "PARENT TO CHILD: parent " << parent << " child " << child
        //          << " phibar " << normf(phibar)
        //          << " fcube_for_mul result " << normf(result_tmp)
        //          << " scale " << std::sqrt(D.template get_volume<T>()*std::pow(T(0.5), T(NDIM*child.level()))) << std::endl;
        T scale = std::sqrt(std::ldexp(D.template get_volume<T>(), -int(NDIM*child.level())));
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

    /**
     * deriv = scale * sum_t sum_k src[t](.., k, ..) * op[t](k, j) with k and j the index
     * along axis, i.e., the sum of transform_dir(src[t], op[t], axis) without the
     * intermediate results. Each thread computes a line of deriv along axis, KK is the
     * compile-time K (0 for K > MRA_MAX_K). src[t] are contiguous K^NDIM cubes or nullptr
     * (zero), op[t] the K x K operators, deriv may not alias any of src.
     */
    template <typename T, Dimension NDIM, int KK, int NT>
    DEVSCOPE void derivative_apply_k(
      const T* const (&src)[NT],
      const T* const (&op)[NT],
      T* __restrict__ deriv,
      const T scale,
      const size_type K,
      const size_type axis)
    {
      constexpr int MAXK = KK > 0 ? KK : 1;
      size_type s = 1;  // the stride of axis
      for (size_type d = axis+1; d < NDIM; ++d) s *= K;
      size_type nlines = 1;
      for (size_type d = 1; d < NDIM; ++d) nlines *= K;
      for (size_type line = thread_id(); line < nlines; line += block_size()) {
        const size_type base = (line / s) * s * K + line % s;
        if constexpr (KK > 0) {
          T acc[MAXK];
#pragma unroll
          for (int j = 0; j < KK; ++j) acc[j] = T(0);
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            if (src[t] == nullptr) continue;
            const T* __restrict__ in = src[t] + base;
            const T* __restrict__ o = op[t];
            T v[MAXK];
#pragma unroll
            for (int k = 0; k < KK; ++k) v[k] = in[k*s];
#pragma unroll
            for (int k = 0; k < KK; ++k) {
#pragma unroll
              for (int j = 0; j < KK; ++j) acc[j] += v[k] * o[k*KK + j];
            }
          }
#pragma unroll
          for (int j = 0; j < KK; ++j) deriv[base + j*s] = scale * acc[j];
        } else {
          for (size_type j = 0; j < K; ++j) {
            T acc = T(0);
            for (int t = 0; t < NT; ++t) {
              if (src[t] == nullptr) continue;
              for (size_type k = 0; k < K; ++k) acc += src[t][base + k*s] * op[t][k*K + j];
            }
            deriv[base + j*s] = scale * acc;
          }
        }
      }
      SYNCTHREADS();
    }

    template <typename T, Dimension NDIM, int NT>
    DEVSCOPE void derivative_apply(
      const T* const (&src)[NT],
      const T* const (&op)[NT],
      T* deriv,
      const T scale,
      const size_type K,
      const size_type axis)
    {
      switch (K) {
#define MRA_DERIV_APPLY_CASE(k) \
        case k: if constexpr (k <= MRA_MAX_K) { derivative_apply_k<T, NDIM, k, NT>(src, op, deriv, scale, K, axis); return; } break;
        MRA_DERIV_APPLY_CASE(4)  MRA_DERIV_APPLY_CASE(5)  MRA_DERIV_APPLY_CASE(6)  MRA_DERIV_APPLY_CASE(7)
        MRA_DERIV_APPLY_CASE(8)  MRA_DERIV_APPLY_CASE(9)  MRA_DERIV_APPLY_CASE(10) MRA_DERIV_APPLY_CASE(11)
        MRA_DERIV_APPLY_CASE(12)
#undef MRA_DERIV_APPLY_CASE
        default: break;
      }
      derivative_apply_k<T, NDIM, 0, NT>(src, op, deriv, scale, K, axis);
    }

    /**
     * The coefficients of node (the coefficients of the function at src_key) at
     * target_key: node itself if it is at target_key, else projected into result
     * (see parent_to_child). Returns nullptr if the function has no coefficients.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE const T* derivative_source(
      const Domain<NDIM>& D,
      const Key<NDIM>& src_key,
      const Key<NDIM>& target_key,
      const concepts::TensorView<NDIM> auto& node,
      concepts::TensorView<NDIM> auto& result,
      concepts::TensorView<NDIM> auto& result_tmp,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      const size_type K,
      T* workspace)
    {
      if (node.data() == nullptr) return nullptr;
      if (src_key == target_key || src_key.is_invalid() || target_key.is_invalid()) return node.data();
      parent_to_child(D, src_key, target_key, node, result, result_tmp, phibar, phi, quad_x, K, workspace);
      return result.data();
    }

    /**
     * See the comment on multiply_kernel_impl in mra/kernels/multiply.h for
     * why this uses explicitly-named template parameters instead of the
     * abbreviated (concept-constrained `auto`) form used elsewhere: too many
     * compiler-synthesized template parameters in one function trips an nvcc
     * EDG front-end assertion ("check_name_hiding_by_template_parameters").
     */
    template <typename T, Dimension NDIM,
              concepts::TensorView<NDIM> ViewNodeLeft,
              concepts::TensorView<NDIM> ViewNodeCenter,
              concepts::TensorView<NDIM> ViewNodeRight,
              concepts::TensorView<3> ViewOperators,
              concepts::TensorView<NDIM> ViewDeriv,
              concepts::TensorView<NDIM+1> ViewTmp,
              concepts::TensorView<NDIM> ViewLeftTmp,
              concepts::TensorView<NDIM> ViewCenterTmp,
              concepts::TensorView<NDIM> ViewRightTmp,
              concepts::TensorView<2> ViewPhi,
              concepts::TensorView<2> ViewPhibar,
              concepts::TensorView<1> ViewQuadX>
    DEVSCOPE void derivative_inner(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>& left,
      const Key<NDIM>& center,
      const Key<NDIM>& right,
      const ViewNodeLeft& node_left,
      const ViewNodeCenter& node_center,
      const ViewNodeRight& node_right,
      const ViewOperators& operators,
      ViewDeriv& deriv,
      ViewTmp& tmp,
      ViewLeftTmp& left_tmp,
      ViewCenterTmp& center_tmp,
      ViewRightTmp& right_tmp,
      const ViewPhi& phi,
      const ViewPhibar& phibar,
      const ViewQuadX& quad_x,
      const int bc_left,
      const int bc_right,
      size_type axis,
      size_type K,
      T* workspace)
    {
      SHARED DenseTensorView<T, NDIM> tmp_result;
      SHARED DenseTensorView<T, NDIM> transform_result;
      if (is_team_lead()){
        tmp_result = tmp(0);
        transform_result = tmp(1);
      }
      SYNCTHREADS();

      /* the coefficients of the neighbors at the neighbor keys of key, the center is key itself */
      const T* src[3];
      src[0] = derivative_source<T, NDIM>(D, left, key.neighbor(axis, -1), node_left, left_tmp, tmp_result,
                                          phibar, phi, quad_x, K, workspace);
      src[1] = derivative_source<T, NDIM>(D, center, key, node_center, center_tmp, tmp_result,
                                          phibar, phi, quad_x, K, workspace);
      src[2] = derivative_source<T, NDIM>(D, right, key.neighbor(axis, 1), node_right, right_tmp, tmp_result,
                                          phibar, phi, quad_x, K, workspace);
      const T* op[3] = {operators((int)FunctionData<T, NDIM>::DerivOp::RPT).data(),
                        operators((int)FunctionData<T, NDIM>::DerivOp::R0T).data(),
                        operators((int)FunctionData<T, NDIM>::DerivOp::RMT).data()};
      T scale = D.template get_reciprocal_width<T>(axis)*std::ldexp(T(1), int(key.level()));
      derivative_apply<T, NDIM>(src, op, deriv.data(), scale, K, axis);
      //std::cout << "INNER " << key << " axis " << axis << " scale " << scale << " RESULT "<< " deriv " << normf(deriv) << std::endl;
    }

    /** See the comment on derivative_inner above. */
    template <typename T, Dimension NDIM,
              concepts::TensorView<NDIM> ViewNodeLeft,
              concepts::TensorView<NDIM> ViewNodeCenter,
              concepts::TensorView<NDIM> ViewNodeRight,
              concepts::TensorView<3> ViewOperators,
              concepts::TensorView<NDIM> ViewDeriv,
              concepts::TensorView<NDIM+1> ViewTmp,
              concepts::TensorView<NDIM> ViewLeftTmp,
              concepts::TensorView<NDIM> ViewCenterTmp,
              concepts::TensorView<NDIM> ViewRightTmp,
              concepts::TensorView<2> ViewPhi,
              concepts::TensorView<2> ViewPhibar,
              concepts::TensorView<1> ViewQuadX>
    DEVSCOPE void derivative_boundary(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>& left,
      const Key<NDIM>& center,
      const Key<NDIM>& right,
      const ViewNodeLeft& node_left,
      const ViewNodeCenter& node_center,
      const ViewNodeRight& node_right,
      const ViewOperators& operators,
      ViewDeriv& deriv,
      ViewTmp& tmp,
      ViewLeftTmp& left_tmp,
      ViewCenterTmp& center_tmp,
      ViewRightTmp& right_tmp,
      const ViewPhi& phi,
      const ViewPhibar& phibar,
      const ViewQuadX& quad_x,
      const T g1,
      const T g2,
      const int bc_left,
      const int bc_right,
      size_type axis,
      size_type K,
      T* workspace)
    {
      SHARED DenseTensorView<T, NDIM> tmp_result;
      SHARED DenseTensorView<T, NDIM> transform_result;
      if (is_team_lead()){
        tmp_result = tmp(0);
        transform_result = tmp(1);
      }
      SYNCTHREADS();


      const T* src[2];
      const T* op[2];
      if (key.is_left_boundary(axis)){
        src[0] = derivative_source<T, NDIM>(D, right, key.neighbor(axis, 1), node_right, right_tmp, tmp_result,
                                            phibar, phi, quad_x, K, workspace);
        src[1] = derivative_source<T, NDIM>(D, center, key, node_center, center_tmp, tmp_result,
                                            phibar, phi, quad_x, K, workspace);
        op[0] = operators((int)FunctionData<T, NDIM>::DerivOp::LEFT_RMT).data();
        op[1] = operators((int)FunctionData<T, NDIM>::DerivOp::LEFT_R0T).data();
      }
      else {
        src[0] = derivative_source<T, NDIM>(D, center, key, node_center, center_tmp, tmp_result,
                                            phibar, phi, quad_x, K, workspace);
        src[1] = derivative_source<T, NDIM>(D, left, key.neighbor(axis, -1), node_left, left_tmp, tmp_result,
                                            phibar, phi, quad_x, K, workspace);
        op[0] = operators((int)FunctionData<T, NDIM>::DerivOp::RIGHT_R0T).data();
        op[1] = operators((int)FunctionData<T, NDIM>::DerivOp::RIGHT_RPT).data();
      }
      T scale = D.template get_reciprocal_width<T>(axis)*std::ldexp(T(1), int(key.level()));
      derivative_apply<T, NDIM>(src, op, deriv.data(), scale, K, axis);

    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_kernel_impl(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>& left,
      const Key<NDIM>& center,
      const Key<NDIM>& right,
      const concepts::TensorView<NDIM> auto& node_left,
      const concepts::TensorView<NDIM> auto& node_center,
      const concepts::TensorView<NDIM> auto& node_right,
      const concepts::TensorView<3> auto& operators,
      concepts::TensorView<NDIM> auto& deriv,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<1> auto& quad_x,
      T* tmp,
      size_type tmp_slot,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const int bc_left,
      const int bc_right)
      {
        // if we reached here, all checks have passed, and we do the transform to compute the derivative
        // for a given axis by calling either derivative_inner() or derivative_boundary()
        SHARED DenseTensorView<T, NDIM> left_tmp, center_tmp, right_tmp;
        SHARED DenseTensorView<T, NDIM+1> tmp_result;
        SHARED T* workspace;

        T* block_tmp_ptr = &tmp[tmp_slot*derivative_tmp_size<NDIM>(K)];
        const size_type K2NDIM = std::pow(K, NDIM);
        if (is_team_lead()) {
          tmp_result = DenseTensorView<T, NDIM+1>(&block_tmp_ptr[       0], make_dims<NDIM+1>(2, K));
          left_tmp   = DenseTensorView<T, NDIM>(&block_tmp_ptr[2*K2NDIM], K);
          center_tmp = DenseTensorView<T, NDIM>(&block_tmp_ptr[3*K2NDIM], K);
          right_tmp  = DenseTensorView<T, NDIM>(&block_tmp_ptr[4*K2NDIM], K);
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

    /**
     * Derivative of the pos-th non-zero function of deriv, executed by one block;
     * shared by derivative_kernel and derivative_kernel_batched. The block's scratch
     * is the tmp_slot-th one of tmp: the position for the unbatched kernel, the block
     * for the batched one (see submit_derivative_batch_leader).
     *
     * The views are named template parameters rather than abbreviated
     * `concepts::TensorView<...> auto&` parameters: mixing the explicit T, NDIM
     * with this many abbreviated-auto parameters crashes nvcc's front end with
     * "internal error: assertion failed ... in check_name_hiding_by_template_parameters"
     * (see reconstruct_process_one_leader in mra/kernels/reconstruct.h).
     */
    template <typename T, Dimension NDIM, typename NodeLeftT, typename InhLeftT, typename NodeCenterT,
              typename InhCenterT, typename NodeRightT, typename InhRightT, typename OperatorsT,
              typename DerivT, typename PhiT, typename PhibarT, typename QuadXT>
    DEVSCOPE void derivative_process_one(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>* src_keys,
      const NodeLeftT& node_left,
      const InhLeftT& inh_left,
      const NodeCenterT& node_center,
      const InhCenterT& inh_center,
      const NodeRightT& node_right,
      const InhRightT& inh_right,
      const OperatorsT& operators,
      DerivT& deriv,
      const PhiT& phi,
      const PhibarT& phibar,
      const QuadXT& quad_x,
      T* tmp,
      size_type N,
      size_type pos,
      size_type tmp_slot,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const int bc_left,
      const int bc_right)
    {
      SHARED DenseTensorView<T, NDIM> deriv_view;
      SHARED DenseTensorView<const T, NDIM> node_left_view, node_center_view, node_right_view;
      SHARED Key<NDIM> left, center, right;
      SHARED size_type blockid;
      if (is_team_lead()) {
        // deriv has exactly n_nonzero non-zero entries, so this always
        // finds a valid function id -- see submit_derivative_kernel.
        blockid = find_nth_nonzero(N, pos, deriv);
        /* a function's coefficients are in the node itself or else in the inherited node */
        node_left_view = node_left(blockid);
        if (node_left_view.data() == nullptr) node_left_view = inh_left(blockid);
        node_center_view = node_center(blockid);
        if (node_center_view.data() == nullptr) node_center_view = inh_center(blockid);
        node_right_view = node_right(blockid);
        if (node_right_view.data() == nullptr) node_right_view = inh_right(blockid);
        deriv_view = deriv(blockid);
        /* the nodes the coefficients of this function belong to */
        left = src_keys[3*blockid + 0];
        center = src_keys[3*blockid + 1];
        right = src_keys[3*blockid + 2];
      }
      SYNCTHREADS();
      derivative_kernel_impl<T, NDIM>(D, key, left, center, right, node_left_view, node_center_view, node_right_view,
        operators, deriv_view, phi, phibar, quad_x, tmp, tmp_slot, K, g1, g2, axis, bc_left, bc_right);
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    /* named template parameters for the views: see derivative_process_one */
    template <typename T, Dimension NDIM, typename NodeLeftT, typename InhLeftT, typename NodeCenterT,
              typename InhCenterT, typename NodeRightT, typename InhRightT, typename OperatorsT,
              typename DerivT, typename PhiT, typename PhibarT, typename QuadXT>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void derivative_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const Key<NDIM>* src_keys,
      const NodeLeftT node_left,
      const InhLeftT inh_left,
      const NodeCenterT node_center,
      const InhCenterT inh_center,
      const NodeRightT node_right,
      const InhRightT inh_right,
      const OperatorsT operators,
      DerivT deriv,
      const PhiT phi,
      const PhibarT phibar,
      const QuadXT quad_x,
      T* tmp,
      size_type N,
      size_type n_nonzero,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const int bc_left,
      const int bc_right)
    {

      static_assert(node_left.ndim() == NDIM+1, "node_left must be of dimension NDIM+1");
      static_assert(node_center.ndim() == NDIM+1, "node_center must be of dimension NDIM+1");
      static_assert(node_right.ndim() == NDIM+1, "node_right must be of dimension NDIM+1");

      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        derivative_process_one<T, NDIM>(D, key, src_keys, node_left, inh_left, node_center, inh_center,
                                        node_right, inh_right, operators, deriv,
                                        phi, phibar, quad_x, tmp, N, pos, pos, K, g1, g2, axis, bc_left, bc_right);
      }
    }

    template <typename T, Dimension NDIM>
    size_type derivative_smem_size(size_type K) {
      return std::max(static_cast<size_type>(K*K*NDIM*sizeof(T)), // used in fcube_for_mul
                      static_cast<size_type>(transform_shmem_size<T, NDIM>(K)));
    }

    /* K^2 threads (K <= MRA_MAX_K) in whole warps: the derivative is a line along the
     * axis per thread, and the tensor-core transforms of parent_to_child run on whole warps */
    inline Dim3 derivative_thread_dims(size_type K) {
#if defined(MRA_ENABLE_HIP)
      constexpr size_type warp = 64;
#else
      constexpr size_type warp = 32;
#endif
      const size_type k = std::min(K, MRA_MAX_K_SIZET);
      return Dim3((k*k + warp - 1) / warp * warp, 1, 1);
    }

  } // namespace detail

  /**
   * Derivative at key of every function that is non-zero in deriv. The coefficients
   * of a function at the left neighbor, at key, and at the right neighbor are those in
   * node_left, node_center, and node_right, or, where the function has none there, those
   * in inh_left, inh_center, and inh_right (coefficients inherited from an ancestor).
   * src_keys (device memory, 3 per function: left, center, right) holds the nodes that
   * these coefficients belong to, per function: they are projected from there to the
   * left neighbor, to key, and to the right neighbor. The views are named template
   * parameters, see detail::derivative_process_one.
   */
  template <typename T, Dimension NDIM, typename NodeLeftT, typename InhLeftT, typename NodeCenterT,
            typename InhCenterT, typename NodeRightT, typename InhRightT, typename OperatorsT,
            typename DerivT, typename PhiT, typename PhibarT, typename QuadXT>
  void submit_derivative_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const Key<NDIM>* src_keys,
    const NodeLeftT node_left,
    const InhLeftT inh_left,
    const NodeCenterT node_center,
    const InhCenterT inh_center,
    const NodeRightT node_right,
    const InhRightT inh_right,
    const OperatorsT operators,
    DerivT deriv,
    const PhiT phi,
    const PhibarT phibar,
    const QuadXT quad_x,
    T* tmp,
    size_type N,
    size_type n_nonzero,
    size_type K,
    const T g1,
    const T g2,
    size_type axis,
    const int bc_left,
    const int bc_right,
    ttg::device::Stream stream)
  {
    const size_type smem_size = detail::derivative_smem_size<T, NDIM>(K);
    //CONFIGURE_KERNEL((detail::derivative_kernel<T, NDIM>), smem_size);
    CALL_KERNEL(detail::derivative_kernel, n_nonzero, detail::derivative_thread_dims(K), smem_size, stream,
      (D, key, src_keys, node_left, inh_left, node_center, inh_center, node_right, inh_right, operators,
        deriv, phi, phibar, quad_x, tmp, N, n_nonzero, K, g1, g2, axis, bc_left, bc_right));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the derivative kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/derivative.h. The operator data (D, the
   * derivative operators, phi, phibar, quad_x, K, axis, boundary conditions) is the
   * same for every node of the TT, so batching is unrestricted; the views and source
   * keys of each node travel per member. The combined launch is flattened over the sum
   * of each member's non-zero functions, see compress_kernel_batched; its scratch is
   * one occupancy-sized region of the batch slot, indexed by block.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using DerivativeBatchArg = std::tuple<
      Key<NDIM>,                     // key
      const Key<NDIM>*,              // src_keys
      SparseTensorView<T, NDIM+1>,   // node_left
      SparseTensorView<T, NDIM+1>,   // inh_left
      SparseTensorView<T, NDIM+1>,   // node_center
      SparseTensorView<T, NDIM+1>,   // inh_center
      SparseTensorView<T, NDIM+1>,   // node_right
      SparseTensorView<T, NDIM+1>,   // inh_right
      SparseTensorView<T, NDIM+1>,   // deriv
      size_type                      // N: this member's function count
    >;

    struct DerivativeBatchArgIdx {
      static constexpr std::size_t key         = 0;
      static constexpr std::size_t src_keys    = 1;
      static constexpr std::size_t node_left   = 2;
      static constexpr std::size_t inh_left    = 3;
      static constexpr std::size_t node_center = 4;
      static constexpr std::size_t inh_center  = 5;
      static constexpr std::size_t node_right  = 6;
      static constexpr std::size_t inh_right   = 7;
      static constexpr std::size_t deriv       = 8;
      static constexpr std::size_t n           = 9;
    };

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void derivative_kernel_batched(
      DerivativeBatchArg<T, NDIM>* args,  // device ptr, size == num_members
      const size_type* member_offsets,    // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      T* shared_tmp,                      // occupancy-sized scratch of the batch slot, indexed by block
      const Domain<NDIM>& D,
      const concepts::TensorView<3> auto operators,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<1> auto quad_x,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const int bc_left,
      const int bc_right)
    {
      using idx = DerivativeBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        derivative_process_one<T, NDIM>(D, std::get<idx::key>(arg), std::get<idx::src_keys>(arg),
                                        std::get<idx::node_left>(arg), std::get<idx::inh_left>(arg),
                                        std::get<idx::node_center>(arg), std::get<idx::inh_center>(arg),
                                        std::get<idx::node_right>(arg), std::get<idx::inh_right>(arg),
                                        operators, std::get<idx::deriv>(arg),
                                        phi, phibar, quad_x, shared_tmp, std::get<idx::n>(arg),
                                        local_pos, blockIdx.x, K, g1, g2, axis, bc_left, bc_right);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(key, src_keys, left, inh_left, center,
     * inh_center, right, inh_right, deriv_view, N, deriv, n_nonzero)`; the leader marshals every member and
     * submits one combined launch. The sparsity of each member's result is copied to
     * its device buffer as part of the single H2D copy of the batch (see
     * submit_grouped_copy). total_functions (of the whole function set) bounds the
     * sparsity staging.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_derivative_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<DerivativeBatchArg<T, NDIM>>& registry,
      const Domain<NDIM>& D,
      const concepts::TensorView<3> auto& operators,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<1> auto& quad_x,
      size_type K,
      const T g1,
      const T g2,
      size_type axis,
      const int bc_left,
      const int bc_right,
      size_type total_functions)
    {
      using arg_t = DerivativeBatchArg<T, NDIM>;
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      const auto device = ttg::device::current_device();
      auto& pool = registry.get(device);
      const std::size_t max_batch = registry.get_max_batch_size();
      auto& slot = pool.acquire(max_batch, max_batch + 1, max_batch * total_functions);
      slot.args.clear();
      slot.offsets.resize(nb + 1);
      slot.offsets[0] = 0;
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();

      size_type total_sparsity_bytes = 0;
      for (std::size_t m = 0; m < nb; ++m) total_sparsity_bytes += batch[m].template get<9>();
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        const size_type m_N = batch[m].template get<9>();
        auto& m_deriv_tensor = batch[m].template get<10>();
        sparsity_to_bytes(m_deriv_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), m_N);
        slot.extra_dsts.push_back(const_cast<T*>(m_deriv_tensor.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(m_N);
        sparsity_offset += m_N;
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(), batch[m].template get<2>(),
                               batch[m].template get<3>(), batch[m].template get<4>(), batch[m].template get<5>(),
                               batch[m].template get<6>(), batch[m].template get<7>(), batch[m].template get<8>(),
                               m_N);
        slot.offsets[m + 1] = slot.offsets[m] + batch[m].template get<11>();

      }
      const size_type total_nonzero = slot.offsets[nb];

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      const size_type smem_size = derivative_smem_size<T, NDIM>(K);
      const Dim3 thread_dims = derivative_thread_dims(K);
      const int threads = static_cast<int>(thread_dims.x * thread_dims.y * thread_dims.z);
      /* cap the grid at occupancy (the kernel's grid-stride loop covers the rest) so the
       * scratch is one region per slot, sized from the cap and not from this launch --
       * see submit_compress_kernel_batched */
      auto kernel = derivative_kernel_batched<T, NDIM, std::decay_t<decltype(operators)>, std::decay_t<decltype(phi)>,
                                              std::decay_t<decltype(phibar)>, std::decay_t<decltype(quad_x)>>;
      const size_type grid_cap = kernel_grid_cap(kernel, threads, smem_size);
      const size_type grid_x = std::min<size_type>(total_nonzero, grid_cap);
      T* shared_tmp = pool.template ensure_scratch<T>(slot, static_cast<std::size_t>(grid_cap) * derivative_tmp_size<NDIM>(K));
      // an empty grid (no nonzero function in the batch) is an invalid launch
      if (grid_x > 0) {
        CALL_KERNEL((derivative_kernel_batched<T, NDIM>), grid_x, thread_dims, smem_size, stream,
                    (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_nonzero, shared_tmp, D,
                     operators, phi, phibar, quad_x, K, g1, g2, axis, bc_left, bc_right));
      }
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* explicit instanatiation */
  extern template
  void submit_derivative_kernel<double, 3>(
    const Domain<3>& D,
    const Key<3>& key,
    const Key<3>* src_keys,
    const SparseTensorView<double, 3+1>& node_left,
    const SparseTensorView<double, 3+1>& inh_left,
    const SparseTensorView<double, 3+1>& node_center,
    const SparseTensorView<double, 3+1>& inh_center,
    const SparseTensorView<double, 3+1>& node_right,
    const SparseTensorView<double, 3+1>& inh_right,
    const DenseTensorView<double, 3>& operators,
    SparseTensorView<double, 3+1>& deriv,
    const DenseTensorView<double, 2>& phi,
    const DenseTensorView<double, 2>& phibar,
    const DenseTensorView<double, 1>& quad_x,
    double* tmp,
    size_type N,
    size_type n_nonzero,
    size_type K,
    const double g1,
    const double g2,
    size_type axis,
    const int bc_left,
    const int bc_right,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra

#endif // MRA_KERNELS_DERIVATIVE_H
