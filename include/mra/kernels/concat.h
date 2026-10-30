#ifndef MRA_KERNELS_CONCAT_H
#define MRA_KERNELS_CONCAT_H

#include <tuple>
#include <type_traits>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"

/**
 * Concatenate the coefficients of two nodes of sets of functions into one node.
 */

namespace mra {

  namespace detail {

    /**
     * out_j = a_j for j < NA, b_{j-NA} otherwise, for the pos-th function j that is
     * non-zero in out. Executed by one block; shared by concat_kernel and
     * concat_kernel_batched.
     */
    template <Dimension NDIM>
    DEVSCOPE void concat_process_one(
      const concepts::TensorView<NDIM+1> auto& a,
      const concepts::TensorView<NDIM+1> auto& b,
      concepts::TensorView<NDIM+1> auto& out_view,
      size_type NA,
      size_type N,
      size_type pos)
    {
      using T = std::remove_const_t<typename std::remove_reference_t<decltype(out_view)>::value_type>;
      SHARED DenseTensorView<const T, NDIM> in;
      SHARED DenseTensorView<T, NDIM> out;
      SHARED size_type j;
      if (is_team_lead()) {
        j = find_nth_nonzero(N, pos, out_view);
        in = (j < NA) ? a(j) : b(j - NA);
        out = out_view(j);
      }
      SYNCTHREADS();
      out = in;
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    /**
     * One block per non-zero output function, see concat_process_one.
     */
    template <Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void concat_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a,
      const concepts::TensorView<NDIM+1> auto b,
      concepts::TensorView<NDIM+1> auto out_view,
      size_type NA,
      size_type N,
      size_type n_nonzero)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        concat_process_one<NDIM>(a, b, out_view, NA, N, pos);
      }
    }
  } // namespace detail

  template <Dimension NDIM>
  void submit_concat_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a,
    const concepts::TensorView<NDIM+1> auto& b,
    concepts::TensorView<NDIM+1> auto& out,
    size_type NA,
    size_type N,
    size_type n_nonzero,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::concat_kernel, n_nonzero, thread_dims, 0, stream, (key, a, b, out, NA, N, n_nonzero));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the concat kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/concat.h. Concatenation has no operator
   * data, so batching is unrestricted; the views and function counts of each node
   * travel per member. The combined launch is flattened over the sum of each
   * member's non-zero output functions.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using ConcatBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,   // a
      SparseTensorView<T, NDIM+1>,   // b
      SparseTensorView<T, NDIM+1>,   // out
      size_type,                     // NA: this member's function count of a
      size_type                      // N: this member's function count of out
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void concat_kernel_batched(
      ConcatBatchArg<T, NDIM>* args,    // device ptr, size == num_members
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
        concat_process_one<NDIM>(std::get<0>(arg), std::get<1>(arg), std::get<2>(arg),
                                 std::get<3>(arg), std::get<4>(arg), local_pos);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(a_view, b_view, out_view, NA, N, out, n_nonzero)`;
     * the leader marshals every member and submits one combined launch. The sparsity
     * of each member's output is copied to its device buffer as part of the single
     * H2D copy of the batch. total_functions bounds the sparsity staging.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_concat_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<ConcatBatchArg<T, NDIM>>& registry,
      size_type K,
      size_type total_functions)
    {
      using arg_t = ConcatBatchArg<T, NDIM>;
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
      for (std::size_t m = 0; m < nb; ++m) total_sparsity_bytes += batch[m].template get<4>();
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        const size_type m_NA = batch[m].template get<3>();
        const size_type m_N = batch[m].template get<4>();
        auto& m_out_tensor = batch[m].template get<5>();
        sparsity_to_bytes(m_out_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), m_N);
        slot.extra_dsts.push_back(const_cast<T*>(m_out_tensor.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(m_N);
        sparsity_offset += m_N;
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(),
                               batch[m].template get<2>(), m_NA, m_N);
        slot.offsets[m + 1] = slot.offsets[m] + batch[m].template get<6>();
      }
      const size_type total_nonzero = slot.offsets[nb];

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      if (total_nonzero > 0) {
        Dim3 thread_dims = max_thread_dims(2*K);
        CALL_KERNEL((concat_kernel_batched<T, NDIM>), total_nonzero, thread_dims, 0, stream,
                    (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_nonzero));
        checkSubmit();
      }
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_CONCAT_H
