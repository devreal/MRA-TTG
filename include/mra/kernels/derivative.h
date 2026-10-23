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

      deriv = 0;

      //std::cout << "DERIVATIVE " << key << " axis " << axis << std::endl;

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


      deriv = T(0);

      if (key.is_left_boundary(axis)){

        parent_to_child(D, right, key.neighbor(axis, 1), node_right, right_tmp, tmp_result, phibar, phi, quad_x, K, workspace);
        parent_to_child(D, center, key, node_center, center_tmp, tmp_result, phibar, phi, quad_x, K, workspace);

        transform_dir(right_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::LEFT_RMT), tmp_result, transform_result, axis);
        deriv += transform_result;
        //std::cout << "BOUNDARY " << key << " axis " << axis << " right_tmp " << normf(right_tmp) << " deriv " << normf(deriv) << std::endl;
        transform_dir(center_tmp, operators((int)FunctionData<T, NDIM>::DerivOp::LEFT_R0T), tmp_result, transform_result, axis);
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
      const concepts::TensorView<NDIM> auto& node_left,
      const concepts::TensorView<NDIM> auto& node_center,
      const concepts::TensorView<NDIM> auto& node_right,
      const concepts::TensorView<3> auto& operators,
      concepts::TensorView<NDIM> auto& deriv,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<1> auto& quad_x,
      T* tmp,
      size_type tmp_pos,
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

        T* block_tmp_ptr = &tmp[tmp_pos*derivative_tmp_size<NDIM>(K)];
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
     * shared by derivative_kernel and derivative_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void derivative_process_one(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const Key<NDIM>* src_keys,
      const concepts::TensorView<NDIM+1> auto& node_left,
      const concepts::TensorView<NDIM+1> auto& node_center,
      const concepts::TensorView<NDIM+1> auto& node_right,
      const concepts::TensorView<3> auto& operators,
      concepts::TensorView<NDIM+1> auto& deriv,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<1> auto& quad_x,
      T* tmp,
      size_type N,
      size_type pos,
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
        node_left_view = node_left(blockid);
        node_center_view = node_center(blockid);
        node_right_view = node_right(blockid);
        deriv_view = deriv(blockid);
        /* the nodes the coefficients of this function belong to */
        left = src_keys[3*blockid + 0];
        center = src_keys[3*blockid + 1];
        right = src_keys[3*blockid + 2];
      }
      SYNCTHREADS();
      derivative_kernel_impl<T, NDIM>(D, key, left, center, right, node_left_view, node_center_view, node_right_view,
        operators, deriv_view, phi, phibar, quad_x, tmp, pos, K, g1, g2, axis, bc_left, bc_right);
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void derivative_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const Key<NDIM>* src_keys,
      const concepts::TensorView<NDIM+1> auto node_left,
      const concepts::TensorView<NDIM+1> auto node_center,
      const concepts::TensorView<NDIM+1> auto node_right,
      const concepts::TensorView<3> auto operators,
      concepts::TensorView<NDIM+1> auto deriv,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<1> auto quad_x,
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
        derivative_process_one<T, NDIM>(D, key, src_keys, node_left, node_center, node_right, operators, deriv,
                                        phi, phibar, quad_x, tmp, N, pos, K, g1, g2, axis, bc_left, bc_right);
      }
    }

    template <typename T, Dimension NDIM>
    size_type derivative_smem_size(size_type K) {
      return std::max(static_cast<size_type>(K*K*NDIM*sizeof(T)), // used in fcube_for_mul
                      static_cast<size_type>(mTxmq_shmem_size<T>(2*K)));
    }

    inline Dim3 derivative_thread_dims(size_type K) {
      size_type max_threads = std::min(K, MRA_MAX_K_SIZET);
      return Dim3(max_threads, max_threads, 1);
    }

  } // namespace detail

  /**
   * Derivative at key of every function that is non-zero in deriv. src_keys (device
   * memory, 3 per function: left, center, right) holds the nodes that the coefficients
   * of node_left, node_center, and node_right belong to, per function: they are
   * projected from there to the left neighbor, to key, and to the right neighbor.
   */
  template <typename T, Dimension NDIM>
  void submit_derivative_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const Key<NDIM>* src_keys,
    const concepts::TensorView<NDIM+1> auto node_left,
    const concepts::TensorView<NDIM+1> auto node_center,
    const concepts::TensorView<NDIM+1> auto node_right,
    const concepts::TensorView<3> auto operators,
    concepts::TensorView<NDIM+1> auto deriv,
    const concepts::TensorView<2> auto phi,
    const concepts::TensorView<2> auto phibar,
    const concepts::TensorView<1> auto quad_x,
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
      (D, key, src_keys, node_left, node_center, node_right, operators,
        deriv, phi, phibar, quad_x, tmp, N, n_nonzero, K, g1, g2, axis, bc_left, bc_right));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the derivative kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/derivative.h. The operator data (D, the
   * derivative operators, phi, phibar, quad_x, K, axis, boundary conditions) is the
   * same for every node of the TT, so batching is unrestricted; the views, source
   * keys and scratch of each node travel per member. The combined launch is flattened
   * over the sum of each member's non-zero functions, see compress_kernel_batched.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using DerivativeBatchArg = std::tuple<
      Key<NDIM>,                     // key
      const Key<NDIM>*,              // src_keys
      SparseTensorView<T, NDIM+1>,   // node_left
      SparseTensorView<T, NDIM+1>,   // node_center
      SparseTensorView<T, NDIM+1>,   // node_right
      SparseTensorView<T, NDIM+1>,   // deriv
      T*,                            // tmp: this member's scratch, indexed by local position
      size_type                      // N: this member's function count
    >;

    struct DerivativeBatchArgIdx {
      static constexpr std::size_t key         = 0;
      static constexpr std::size_t src_keys    = 1;
      static constexpr std::size_t node_left   = 2;
      static constexpr std::size_t node_center = 3;
      static constexpr std::size_t node_right  = 4;
      static constexpr std::size_t deriv       = 5;
      static constexpr std::size_t tmp         = 6;
      static constexpr std::size_t n           = 7;
    };

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void derivative_kernel_batched(
      DerivativeBatchArg<T, NDIM>* args,  // device ptr, size == num_members
      const size_type* member_offsets,    // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
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
                                        std::get<idx::node_left>(arg), std::get<idx::node_center>(arg),
                                        std::get<idx::node_right>(arg), operators, std::get<idx::deriv>(arg),
                                        phi, phibar, quad_x, std::get<idx::tmp>(arg), std::get<idx::n>(arg),
                                        local_pos, K, g1, g2, axis, bc_left, bc_right);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(key, src_keys, left, center, right,
     * deriv_view, tmp, N, deriv, n_nonzero)`; the leader marshals every member and
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
      for (std::size_t m = 0; m < nb; ++m) total_sparsity_bytes += batch[m].template get<7>();
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        const size_type m_N = batch[m].template get<7>();
        auto& m_deriv_tensor = batch[m].template get<8>();
        sparsity_to_bytes(m_deriv_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), m_N);
        slot.extra_dsts.push_back(const_cast<T*>(m_deriv_tensor.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(m_N);
        sparsity_offset += m_N;
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(), batch[m].template get<2>(),
                               batch[m].template get<3>(), batch[m].template get<4>(), batch[m].template get<5>(),
                               batch[m].template get<6>(), m_N);
        slot.offsets[m + 1] = slot.offsets[m] + batch[m].template get<9>();
      }
      const size_type total_nonzero = slot.offsets[nb];

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      const size_type smem_size = derivative_smem_size<T, NDIM>(K);
      CALL_KERNEL((derivative_kernel_batched<T, NDIM>), total_nonzero, derivative_thread_dims(K), smem_size, stream,
                  (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_nonzero, D,
                   operators, phi, phibar, quad_x, K, g1, g2, axis, bc_left, bc_right));
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
    const SparseTensorView<double, 3+1>& node_center,
    const SparseTensorView<double, 3+1>& node_right,
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
