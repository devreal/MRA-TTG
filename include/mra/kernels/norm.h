#ifndef MRA_KERNELS_NORM_H
#define MRA_KERNELS_NORM_H

#include <array>
#include <tuple>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {
    /**
     * Squared norm of function pos at this node (zero where the function is zero
     * here). Executed by one block; shared by norm_kernel and norm_kernel_batched.
     * The squared norms are summed up the tree by the host-side reduction in
     * mra/tasks/norm.h, which takes the square root at the root.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void norm_process_one(
      const concepts::TensorView<NDIM+1> auto& node,
      T* result_norms,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> n;
      SHARED bool is_zero;
      SHARED T node_normsq;
      if (is_team_lead()) {
        is_zero = node.is_zero(pos);
        if (!is_zero) n = node(pos);
      }
      SYNCTHREADS();
      if (is_zero) {
        if (is_team_lead()) result_norms[pos] = T(0);
      } else {
        sumabssq(n, &node_normsq);
        /* wait for all threads to contribute */
        SYNCTHREADS();
        if (is_team_lead()) result_norms[pos] = node_normsq;
      }
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void norm_kernel(
      const concepts::TensorView<NDIM+1> auto node,
      T* result_norms,
      size_type N,
      size_type K)
    {
      // One block per function, including functions that are zero at this node,
      // so that every entry of result_norms is written.
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        norm_process_one<T, NDIM>(node, result_norms, pos);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_norm_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    const concepts::TensorView<NDIM+1> auto& in,
    concepts::TensorView<1> auto& result_norms,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL((detail::norm_kernel<T, NDIM>), N, thread_dims, 0, stream,
        (in, result_norms.data(), N, K));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the norm kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/norm.h. There is no operator data, so
   * batching is unrestricted. The norms are dense and written in full, so no sparsity
   * needs to be pushed. The combined launch has one block per function of every
   * member, as the unbatched kernel.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using NormBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,   // node
      T*                             // result norms
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void norm_kernel_batched(
      NormBatchArg<T, NDIM>* args,      // device ptr, size == num_members
      const size_type* member_offsets,  // device ptr, size == num_members+1
      size_type num_members,
      size_type total_blocks,
      size_type K)
    {
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_blocks; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        norm_process_one<T, NDIM>(std::get<0>(arg), std::get<1>(arg), local_pos);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(node_view, result, N)`;
     * the leader marshals every member and submits one combined launch.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_norm_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<NormBatchArg<T, NDIM>>& registry,
      size_type K)
    {
      using arg_t = NormBatchArg<T, NDIM>;
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      size_type total_blocks = 0;
      for (std::size_t m = 0; m < nb; ++m) total_blocks += batch[m].template get<2>();
      if (total_blocks == 0) return;

      const auto device = ttg::device::current_device();
      auto& pool = registry.get(device);
      const std::size_t max_batch = registry.get_max_batch_size();
      auto& slot = pool.acquire(max_batch, max_batch + 1, /*num_sparsity=*/0);
      slot.args.clear();
      slot.offsets.resize(nb + 1);
      slot.offsets[0] = 0;
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();
      for (std::size_t m = 0; m < nb; ++m) {
        auto& b = batch[m];
        slot.args.emplace_back(b.template get<0>(), b.template get<1>());
        slot.offsets[m + 1] = slot.offsets[m] + b.template get<2>();
      }

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      Dim3 thread_dims = max_thread_dims(2*K);
      CALL_KERNEL((norm_kernel_batched<T, NDIM>), total_blocks, thread_dims, 0, stream,
                  (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_blocks, K));
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_NORM_H
