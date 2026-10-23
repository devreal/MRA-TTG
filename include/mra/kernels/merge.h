#ifndef MRA_KERNELS_MERGE_H
#define MRA_KERNELS_MERGE_H

#include <tuple>
#include <type_traits>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"

/**
 * Combine the coefficients of two nodes of a set of functions into one node.
 */

namespace mra {

  namespace detail {

    /**
     * out_j = b_j where b holds coefficients for j, else a_j, for the pos-th function
     * j that is non-zero in out. Executed by one block; shared by merge_kernel and
     * merge_kernel_batched.
     */
    template <Dimension NDIM>
    DEVSCOPE void merge_process_one(
      const concepts::TensorView<NDIM+1> auto& a,
      const concepts::TensorView<NDIM+1> auto& b,
      concepts::TensorView<NDIM+1> auto& out_view,
      size_type N,
      size_type pos)
    {
      using T = std::remove_const_t<typename std::remove_reference_t<decltype(out_view)>::value_type>;
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> out;
      SHARED size_type j;
      if (is_team_lead()) {
        j = find_nth_nonzero(N, pos, out_view);
        in = b.is_nonzero(j) ? b(j) : a(j);
        out = out_view(j);
      }
      SYNCTHREADS();
      out = in;
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    /**
     * One block per non-zero output function, see merge_process_one.
     */
    template <Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void merge_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a,
      const concepts::TensorView<NDIM+1> auto b,
      concepts::TensorView<NDIM+1> auto out_view,
      size_type N,
      size_type n_nonzero)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        merge_process_one<NDIM>(a, b, out_view, N, pos);
      }
    }
  } // namespace detail

  template <Dimension NDIM>
  void submit_merge_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a,
    const concepts::TensorView<NDIM+1> auto& b,
    concepts::TensorView<NDIM+1> auto& out,
    size_type N,
    size_type n_nonzero,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(K);
    CALL_KERNEL(detail::merge_kernel, n_nonzero, thread_dims, 0, stream, (key, a, b, out, N, n_nonzero));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the merge kernel. Merging has no operator data, so batching
   * is unrestricted. A task merges either in a batch of its own TT (make_derivative's
   * assemble) or as part of another batched kernel's batch (make_multiply_broadcast),
   * since a task can join only one batch; submit_merge_batch serves both.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using MergeBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,   // a
      SparseTensorView<T, NDIM+1>,   // b
      SparseTensorView<T, NDIM+1>,   // out
      size_type                      // N: this member's function count
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void merge_kernel_batched(
      MergeBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets,  // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero)
    {
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        merge_process_one<NDIM>(std::get<0>(arg), std::get<1>(arg), std::get<2>(arg), std::get<3>(arg), local_pos);
      }
    }

    /**
     * Merges for nb tasks in one launch. member(m) returns the m-th task's
     * {a, b, out_view, out tensor, N, n_nonzero}; tasks with n_nonzero == 0 are
     * skipped. The sparsity of each output is copied to its device buffer as part
     * of the single H2D copy of the batch. total_functions bounds the sparsity
     * staging. Must be called by the batch leader only.
     */
    template <typename T, Dimension NDIM, typename MemberFn>
    void submit_merge_batch(
      GroupedBatchPoolRegistry<MergeBatchArg<T, NDIM>>& registry,
      std::size_t nb,
      MemberFn&& member,
      size_type K,
      size_type total_functions)
    {
      using arg_t = MergeBatchArg<T, NDIM>;
      size_type total_nonzero = 0, total_sparsity_bytes = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        auto&& [a, b, out_view, out_tensor, N, n_nonzero] = member(m);
        if (n_nonzero == 0) continue;
        total_nonzero += n_nonzero;
        total_sparsity_bytes += N;
      }
      if (total_nonzero == 0) return;

      const auto device = ttg::device::current_device();
      auto& pool = registry.get(device);
      const std::size_t max_batch = registry.get_max_batch_size();
      auto& slot = pool.acquire(max_batch, max_batch + 1, max_batch * total_functions);
      slot.args.clear();
      slot.offsets.clear();
      slot.offsets.push_back(0);
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        auto&& [a, b, out_view, out_tensor, N, n_nonzero] = member(m);
        if (n_nonzero == 0) continue;
        sparsity_to_bytes(out_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), N);
        slot.extra_dsts.push_back(const_cast<T*>(out_tensor.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(N);
        sparsity_offset += N;
        slot.args.emplace_back(a, b, out_view, N);
        slot.offsets.push_back(slot.offsets.back() + n_nonzero);
      }

      const size_type num_members = static_cast<size_type>(slot.args.size());
      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, num_members, slot.offsets.size(), pool.device_id, stream);
      Dim3 thread_dims = max_thread_dims(K);
      CALL_KERNEL((merge_kernel_batched<T, NDIM>), total_nonzero, thread_dims, 0, stream,
                  (slot.dev_args, slot.dev_offsets, num_members, total_nonzero));
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_MERGE_H
