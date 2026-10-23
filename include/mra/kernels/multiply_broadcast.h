#ifndef MRA_KERNELS_MULTIPLY_BROADCAST_H
#define MRA_KERNELS_MULTIPLY_BROADCAST_H

#include <array>
#include <cmath>
#include <tuple>

#include "mra/misc/key.h"
#include "mra/misc/domain.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"

/**
 * Kernels for multiplying every function of a set by one function, on
 * reconstructed (scaling-function) coefficients. Where a factor's tree ends
 * above the node, its values at the node's quadrature points are computed from
 * the leaf coefficients of the ancestor directly (fcube_for_mul), which is exact.
 */

namespace mra {

  template<Dimension NDIM>
  SCOPE size_type multiply_broadcast_tmp_size(size_type K) {
    return 3*size_type(std::pow(K, NDIM));         // two value cubes, workspace
  }

  namespace detail {

    /**
     * out_j = a * b_j at node `key` for the pos-th non-zero output function j, where
     * a and b_j are given by their scaling coefficients: a from a_leaf (at key) if it
     * holds them, else from a_inh (at inh_keys[0], an ancestor of key), likewise b_j
     * (inh_keys[1+j]). As madness::FunctionImpl::do_mul: values at the quadrature
     * points of the node, product, back to coefficients. Executed by one block; shared
     * by multiply_broadcast_kernel and multiply_broadcast_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_broadcast_process_one(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const concepts::TensorView<NDIM+1> auto& a_leaf,
      const concepts::TensorView<NDIM+1> auto& a_inh,
      const concepts::TensorView<NDIM+1> auto& b_leaf,
      const concepts::TensorView<NDIM+1> auto& b_inh,
      const Key<NDIM>* inh_keys,
      concepts::TensorView<NDIM+1> auto& out_view,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      size_type N,
      size_type K,
      T* tmp,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> a, b;
      SHARED DenseTensorView<T, NDIM> out, acube, bcube;
      SHARED Key<NDIM> a_key, b_key;
      SHARED T* workspace;
      SHARED size_type j;
      const size_type K2NDIM = std::pow(K, NDIM);
      if (is_team_lead()) {
        j = find_nth_nonzero(N, pos, out_view);
        T* block_tmp = &tmp[pos*multiply_broadcast_tmp_size<NDIM>(K)];
        acube = DenseTensorView<T, NDIM>(&block_tmp[0], K, K, K);
        bcube = DenseTensorView<T, NDIM>(&block_tmp[K2NDIM], K, K, K);
        workspace = &block_tmp[2*K2NDIM];
        /* the leaf's own coefficients belong to key, inherited ones to their origin */
        const bool a_here = a_leaf.is_nonzero(0), b_here = b_leaf.is_nonzero(j);
        a = a_here ? a_leaf(0) : a_inh(0);
        b = b_here ? b_leaf(j) : b_inh(j);
        a_key = a_here ? key : inh_keys[0];
        b_key = b_here ? key : inh_keys[1 + j];
        out = out_view(j);
      }
      SYNCTHREADS();
      /* values at the quadrature points of the node, from coefficients at the node or at
       * an ancestor; a factor without coefficients (a leaf that is zero) is zero */
      if (a.data() == nullptr) acube = T(0);
      else fcube_for_mul<T, NDIM>(D, key, a_key, a, acube, phibar, phi, quad_x, K, workspace);
      if (b.data() == nullptr) bcube = T(0);
      else fcube_for_mul<T, NDIM>(D, key, b_key, b, bcube, phibar, phi, quad_x, K, workspace);
      foreach_idx(acube, [&](size_type i) { acube[i] *= bcube[i]; });
      SYNCTHREADS();
      transform(acube, phibar, out, workspace);
      const T scale = std::pow(T(0.5), T(0.5*NDIM*key.level())) * std::sqrt(D.template get_volume<T>());
      out *= scale;
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    /**
     * One block per non-zero output function, see multiply_broadcast_process_one.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void multiply_broadcast_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a_leaf,
      const concepts::TensorView<NDIM+1> auto a_inh,
      const concepts::TensorView<NDIM+1> auto b_leaf,
      const concepts::TensorView<NDIM+1> auto b_inh,
      const Key<NDIM>* inh_keys,
      concepts::TensorView<NDIM+1> auto out_view,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<1> auto quad_x,
      size_type N,
      size_type n_nonzero,
      size_type K,
      T* tmp)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        multiply_broadcast_process_one<T, NDIM>(D, key, a_leaf, a_inh, b_leaf, b_inh, inh_keys, out_view,
                                                phibar, phi, quad_x, N, K, tmp, pos);
      }
    }

  } // namespace detail

  namespace detail {
    template <typename T, Dimension NDIM>
    size_type multiply_broadcast_smem_size(size_type K) {
      return std::max(static_cast<size_type>(K*K*NDIM*sizeof(T)), // phi of fcube_for_mul from an ancestor
                      static_cast<size_type>(mTxmq_shmem_size<T>(2*K)));
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_multiply_broadcast_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a_leaf,
    const concepts::TensorView<NDIM+1> auto& a_inh,
    const concepts::TensorView<NDIM+1> auto& b_leaf,
    const concepts::TensorView<NDIM+1> auto& b_inh,
    const Key<NDIM>* inh_keys,
    concepts::TensorView<NDIM+1> auto& out,
    const concepts::TensorView<2> auto& phibar,
    const concepts::TensorView<2> auto& phi,
    const concepts::TensorView<1> auto& quad_x,
    size_type N,
    size_type n_nonzero,
    size_type K,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    const size_type smem_size = detail::multiply_broadcast_smem_size<T, NDIM>(K);
    CALL_KERNEL(detail::multiply_broadcast_kernel, n_nonzero, thread_dims, smem_size, stream,
      (D, key, a_leaf, a_inh, b_leaf, b_inh, inh_keys, out, phibar, phi, quad_x, N, n_nonzero, K, tmp));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the multiply_broadcast kernel, used by
   * ttg::device::coop()/TT::set_batch_matcher() in mra/tasks/multiply_broadcast.h.
   * The operator data (D, phibar, phi, quad_x, K) is the same for every node of the
   * TT, so batching is unrestricted; the views, source keys and scratch of each node
   * travel per member. The combined launch is flattened over the sum of each
   * member's non-zero output functions, see compress_kernel_batched.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using MultiplyBroadcastBatchArg = std::tuple<
      Key<NDIM>,                     // key
      SparseTensorView<T, NDIM+1>,   // a_leaf
      SparseTensorView<T, NDIM+1>,   // a_inh
      SparseTensorView<T, NDIM+1>,   // b_leaf
      SparseTensorView<T, NDIM+1>,   // b_inh
      const Key<NDIM>*,              // inh_keys
      SparseTensorView<T, NDIM+1>,   // out
      size_type,                     // N: this member's function count
      T*                             // tmp: this member's scratch, indexed by local position
    >;

    struct MultiplyBroadcastBatchArgIdx {
      static constexpr std::size_t key      = 0;
      static constexpr std::size_t a_leaf   = 1;
      static constexpr std::size_t a_inh    = 2;
      static constexpr std::size_t b_leaf   = 3;
      static constexpr std::size_t b_inh    = 4;
      static constexpr std::size_t inh_keys = 5;
      static constexpr std::size_t out      = 6;
      static constexpr std::size_t n        = 7;
      static constexpr std::size_t tmp      = 8;
    };

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void multiply_broadcast_kernel_batched(
      MultiplyBroadcastBatchArg<T, NDIM>* args, // device ptr, size == num_members
      const size_type* member_offsets,          // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      const Domain<NDIM>& D,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<1> auto quad_x,
      size_type K)
    {
      using idx = MultiplyBroadcastBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        multiply_broadcast_process_one<T, NDIM>(D, std::get<idx::key>(arg),
                                                std::get<idx::a_leaf>(arg), std::get<idx::a_inh>(arg),
                                                std::get<idx::b_leaf>(arg), std::get<idx::b_inh>(arg),
                                                std::get<idx::inh_keys>(arg), std::get<idx::out>(arg),
                                                phibar, phi, quad_x, std::get<idx::n>(arg), K,
                                                std::get<idx::tmp>(arg), local_pos);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(key, a_leaf, a_inh, b_leaf, b_inh,
     * inh_keys, out_view, N, tmp, out, n_nonzero, ...)`; the leader marshals every member
     * and submits one combined launch. A member without non-zero output functions
     * (it only merges) takes part without contributing work. The sparsity of each
     * member's output is copied to its device buffer as part of the single H2D copy
     * of the batch, instead of by each member (see submit_grouped_copy).
     * total_functions (of the whole function set) bounds the sparsity staging.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_multiply_broadcast_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<MultiplyBroadcastBatchArg<T, NDIM>>& registry,
      const Domain<NDIM>& D,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      size_type K,
      size_type total_functions)
    {
      using arg_t = MultiplyBroadcastBatchArg<T, NDIM>;
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      size_type total_nonzero = 0;
      for (std::size_t m = 0; m < nb; ++m) total_nonzero += batch[m].template get<10>();
      if (total_nonzero == 0) return;

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
      for (std::size_t m = 0; m < nb; ++m) {
        if (batch[m].template get<10>() > 0) total_sparsity_bytes += batch[m].template get<7>();
      }
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        const size_type m_N = batch[m].template get<7>();
        const size_type m_n_nonzero = batch[m].template get<10>();
        if (m_n_nonzero > 0) {
          auto& m_out_tensor = batch[m].template get<9>();
          sparsity_to_bytes(m_out_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), m_N);
          slot.extra_dsts.push_back(const_cast<T*>(m_out_tensor.buffer().device_ptr_on(device)));
          slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
          slot.extra_sizes.push_back(m_N);
          sparsity_offset += m_N;
        }
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(), batch[m].template get<2>(),
                               batch[m].template get<3>(), batch[m].template get<4>(), batch[m].template get<5>(),
                               batch[m].template get<6>(), m_N, batch[m].template get<8>());
        slot.offsets[m + 1] = slot.offsets[m] + m_n_nonzero;
      }

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      Dim3 thread_dims = max_thread_dims(2*K);
      const size_type smem_size = multiply_broadcast_smem_size<T, NDIM>(K);
      CALL_KERNEL((multiply_broadcast_kernel_batched<T, NDIM>), total_nonzero, thread_dims, smem_size, stream,
                  (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_nonzero, D, phibar, phi, quad_x, K));
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST


} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_BROADCAST_H
