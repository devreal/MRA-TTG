#ifndef MRA_KERNELS_TRANSFORM_FUNCTIONS_H
#define MRA_KERNELS_TRANSFORM_FUNCTIONS_H

#include <tuple>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/ops/functions.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"

/**
 * Linear combination of a set of functions at one node of the tree:
 *   out_j = sum_i U(i,j) in_i
 * (cf. madness::transform). U is N x M, row-major. Being linear, this applies
 * to the coefficients of compressed and reconstructed nodes alike.
 */

namespace mra {
  namespace detail {

    /**
     * out_j for the pos-th non-zero output function j. Input functions that are zero
     * at this node, or whose coefficient U(i,j) is zero, are skipped. Executed by one
     * block; shared by transform_functions_kernel and transform_functions_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void transform_functions_process_one(
      const concepts::TensorView<NDIM+1> auto& in_view,
      concepts::TensorView<NDIM+1> auto& out_view,
      const T* U,
      size_type N,
      size_type M,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> out;
      SHARED size_type j;
      SHARED bool skip;
      if (is_team_lead()) {
        // out_view has exactly n_nonzero non-zero entries
        j = find_nth_nonzero(M, pos, out_view);
        out = out_view(j);
      }
      SYNCTHREADS();
      foreach_idx(out, [&](size_type k) { out[k] = T(0); });
      for (size_type i = 0; i < N; ++i) {
        if (is_team_lead()) {
          skip = in_view.is_zero(i) || U[i*M + j] == T(0);
          if (!skip) in = in_view(i);
        }
        SYNCTHREADS();
        if (!skip) {
          const T u = U[i*M + j];
          /* every thread updates the same elements in each iteration */
          foreach_idx(out, [&](size_type k) { out[k] += u*in[k]; });
        }
        SYNCTHREADS();
      }
    }

    /**
     * One block per non-zero output function, see transform_functions_process_one.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void transform_functions_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto in_view,
      concepts::TensorView<NDIM+1> auto out_view,
      const T* U,
      size_type N,
      size_type M,
      size_type n_nonzero)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        transform_functions_process_one<T, NDIM>(in_view, out_view, U, N, M, pos);
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_transform_functions_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& in,
    concepts::TensorView<NDIM+1> auto& out,
    const T* U,
    size_type N,
    size_type M,
    size_type n_nonzero,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::transform_functions_kernel, n_nonzero, thread_dims, 0, stream,
                (key, in, out, U, N, M, n_nonzero));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the transform_functions kernel, used by
   * ttg::device::coop()/TT::set_batch_matcher() in mra/tasks/transform.h. U is the
   * same for every node of the TT, so batching is unrestricted; the views of each
   * node travel per member. The combined launch is flattened over the sum of each
   * member's non-zero output functions.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using TransformFunctionsBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,   // in
      SparseTensorView<T, NDIM+1>,   // out
      size_type,                     // N: this member's input function count
      size_type                      // M: this member's output function count
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void transform_functions_kernel_batched(
      TransformFunctionsBatchArg<T, NDIM>* args, // device ptr, size == num_members
      const size_type* member_offsets,           // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      const T* U)
    {
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        transform_functions_process_one<T, NDIM>(std::get<0>(arg), std::get<1>(arg), U,
                                                 std::get<2>(arg), std::get<3>(arg), local_pos);
        SYNCTHREADS(); // member and local_pos are rewritten in the next iteration
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(in_view, out_view, N, M, out, n_nonzero)`;
     * the leader marshals every member and submits one combined launch. The sparsity
     * of each member's output is copied to its device buffer as part of the single
     * H2D copy of the batch. total_functions bounds the sparsity staging.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_transform_functions_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<TransformFunctionsBatchArg<T, NDIM>>& registry,
      const T* U,
      size_type K,
      size_type total_functions)
    {
      using arg_t = TransformFunctionsBatchArg<T, NDIM>;
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
        const size_type m_M = batch[m].template get<3>();
        auto& m_out_tensor = batch[m].template get<4>();
        sparsity_to_bytes(m_out_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), m_M);
        slot.extra_dsts.push_back(const_cast<T*>(m_out_tensor.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(m_M);
        sparsity_offset += m_M;
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(),
                               batch[m].template get<2>(), m_M);
        slot.offsets[m + 1] = slot.offsets[m] + batch[m].template get<5>();
      }
      const size_type total_nonzero = slot.offsets[nb];

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      Dim3 thread_dims = max_thread_dims(2*K);
      CALL_KERNEL((transform_functions_kernel_batched<T, NDIM>), total_nonzero, thread_dims, 0, stream,
                  (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_nonzero, U));
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_TRANSFORM_FUNCTIONS_H
