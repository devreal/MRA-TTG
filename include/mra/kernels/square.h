#ifndef MRA_KERNELS_SQUARE_H
#define MRA_KERNELS_SQUARE_H

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
 * Pointwise square of the scaling coefficients of a node, for every function of
 * a set (cf. madness::FunctionImpl::square_inplace, without autorefinement).
 */

namespace mra {

  template<Dimension NDIM>
  SCOPE size_type square_tmp_size(size_type K) {
    return 2*size_type(std::pow(K, NDIM));         // value cube, workspace
  }

  namespace detail {

    /**
     * out_j = in_j^2 at node `key` for the pos-th non-zero output function j: values
     * at the quadrature points of the node, squared, back to coefficients. Executed
     * by one block; shared by square_kernel and square_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void square_process_one(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const concepts::TensorView<NDIM+1> auto& in_view,
      concepts::TensorView<NDIM+1> auto& out_view,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      size_type N,
      size_type K,
      T* tmp,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> out, cube;
      SHARED T* workspace;
      SHARED size_type j;
      const size_type K2NDIM = std::pow(K, NDIM);
      if (is_team_lead()) {
        j = find_nth_nonzero(N, pos, out_view);
        T* block_tmp = &tmp[pos*square_tmp_size<NDIM>(K)];
        cube = DenseTensorView<T, NDIM>(&block_tmp[0], K, K, K);
        workspace = &block_tmp[K2NDIM];
        in = in_view(j);
        out = out_view(j);
      }
      SYNCTHREADS();
      fcube_for_mul<T, NDIM>(D, key, key, in, cube, phibar, phi, quad_x, K, workspace);
      foreach_idx(cube, [&](size_type i) { cube[i] *= cube[i]; });
      SYNCTHREADS();
      transform(cube, phibar, out, workspace);
      const T scale = std::pow(T(0.5), T(0.5*NDIM*key.level())) * std::sqrt(D.template get_volume<T>());
      out *= scale;
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    /**
     * One block per non-zero output function, see square_process_one.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void square_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto in_view,
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
        square_process_one<T, NDIM>(D, key, in_view, out_view, phibar, phi, quad_x, N, K, tmp, pos);
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_square_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& in,
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
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CALL_KERNEL(detail::square_kernel, n_nonzero, thread_dims, smem_size, stream,
      (D, key, in, out, phibar, phi, quad_x, N, n_nonzero, K, tmp));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the square kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/square.h. The operator data (D, phibar,
   * phi, quad_x, K) is the same for every node of the TT, so batching is
   * unrestricted; the views and scratch of each node travel per member. The combined
   * launch is flattened over the sum of each member's non-zero output functions.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using SquareBatchArg = std::tuple<
      Key<NDIM>,                     // key
      SparseTensorView<T, NDIM+1>,   // in
      SparseTensorView<T, NDIM+1>,   // out
      size_type,                     // N: this member's function count
      T*                             // tmp: this member's scratch, indexed by local position
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void square_kernel_batched(
      SquareBatchArg<T, NDIM>* args,    // device ptr, size == num_members
      const size_type* member_offsets,  // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      const Domain<NDIM>& D,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<1> auto quad_x,
      size_type K)
    {
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        square_process_one<T, NDIM>(D, std::get<0>(arg), std::get<1>(arg), std::get<2>(arg),
                                    phibar, phi, quad_x, std::get<3>(arg), K, std::get<4>(arg), local_pos);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(key, in_view, out_view, N, tmp, out,
     * n_nonzero)`; the leader marshals every member and submits one combined launch.
     * The sparsity of each member's output is copied to its device buffer as part of
     * the single H2D copy of the batch. total_functions bounds the sparsity staging.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_square_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<SquareBatchArg<T, NDIM>>& registry,
      const Domain<NDIM>& D,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      size_type K,
      size_type total_functions)
    {
      using arg_t = SquareBatchArg<T, NDIM>;
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
      for (std::size_t m = 0; m < nb; ++m) total_sparsity_bytes += batch[m].template get<3>();
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        const size_type m_N = batch[m].template get<3>();
        auto& m_out_tensor = batch[m].template get<5>();
        sparsity_to_bytes(m_out_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), m_N);
        slot.extra_dsts.push_back(const_cast<T*>(m_out_tensor.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(m_N);
        sparsity_offset += m_N;
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(), batch[m].template get<2>(),
                               m_N, batch[m].template get<4>());
        slot.offsets[m + 1] = slot.offsets[m] + batch[m].template get<6>();
      }
      const size_type total_nonzero = slot.offsets[nb];

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      Dim3 thread_dims = max_thread_dims(2*K);
      auto smem_size = mTxmq_shmem_size<T>(2*K);
      CALL_KERNEL((square_kernel_batched<T, NDIM>), total_nonzero, thread_dims, smem_size, stream,
                  (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_nonzero, D, phibar, phi, quad_x, K));
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_SQUARE_H
