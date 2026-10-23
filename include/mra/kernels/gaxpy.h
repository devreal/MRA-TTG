#ifndef MRA_KERNELS_GAXPY_H
#define MRA_KERNELS_GAXPY_H

#include <tuple>

#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {
    template <typename T, Dimension NDIM>
    DEVSCOPE void axpy_kernel_impl(
      const concepts::TensorView<NDIM> auto& nodeA,
      concepts::TensorView<NDIM> auto& nodeR,
      const T scalarA)
    {
      foreach_idx(nodeR, [&](size_type i) {
        nodeR[i] += scalarA*nodeA[i];
      });
    }

    template <typename T, Dimension NDIM>
    DEVSCOPE void gaxpy_kernel_impl(
      const concepts::TensorView<NDIM> auto& nodeA,
      const concepts::TensorView<NDIM> auto& nodeB,
      concepts::TensorView<NDIM> auto& nodeR,
      const T scalarA,
      const T scalarB)
    {
      foreach_idx(nodeR, [&](size_type i) {
        nodeR[i] = scalarA*nodeA[i] + scalarB*nodeB[i];
      });
    }

    /**
     * R_j = scalarA*A_j + scalarB*B_j for the pos-th non-zero function j of R.
     * Executed by one block; shared by gaxpy_kernel and gaxpy_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void gaxpy_process_one(
      const concepts::TensorView<NDIM+1> auto& nodeA_view,
      const concepts::TensorView<NDIM+1> auto& nodeB_view,
      concepts::TensorView<NDIM+1> auto& nodeR_view,
      const T scalarA,
      const T scalarB,
      size_type N,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> nodeA, nodeB;
      SHARED DenseTensorView<T, NDIM> nodeR;
      SHARED size_type blockid;
      if (is_team_lead()) {
        // nodeR_view has exactly n_nonzero non-zero entries, so this
        // always finds a valid function id -- see submit_gaxpy_kernel.
        blockid = find_nth_nonzero(N, pos, nodeR_view);
        nodeA = nodeA_view(blockid);
        nodeB = nodeB_view(blockid);
        nodeR = nodeR_view(blockid);
      }
      SYNCTHREADS();
      gaxpy_kernel_impl<T, NDIM>(nodeA, nodeB, nodeR, scalarA, scalarB);
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gaxpy_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto nodeA_view,
      const concepts::TensorView<NDIM+1> auto nodeB_view,
      concepts::TensorView<NDIM+1> auto nodeR_view,
      const T scalarA,
      const T scalarB,
      size_type N,
      size_type n_nonzero)
    {
      for (size_type pos = blockIdx.x; pos < n_nonzero; pos += gridDim.x) {
        gaxpy_process_one<T, NDIM>(nodeA_view, nodeB_view, nodeR_view, scalarA, scalarB, N, pos);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_gaxpy_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& funcA,
    const concepts::TensorView<NDIM+1> auto& funcB,
    concepts::TensorView<NDIM+1> auto& funcR,
    const T scalarA,
    const T scalarB,
    size_type N,
    size_type n_nonzero,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);

    CALL_KERNEL(detail::gaxpy_kernel, n_nonzero, thread_dims, 0, stream,
      (key, funcA, funcB, funcR, scalarA, scalarB, N, n_nonzero));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the gaxpy kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/gaxpy.h. The scalars are the same for every
   * node of the TT, so batching is unrestricted; the views of each node travel per
   * member. The combined launch is flattened over the sum of each member's non-zero
   * output functions.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using GaxpyBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,   // A
      SparseTensorView<T, NDIM+1>,   // B
      SparseTensorView<T, NDIM+1>,   // R
      size_type                      // N: this member's function count
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gaxpy_kernel_batched(
      GaxpyBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets,  // device ptr, size == num_members+1
      size_type num_members,
      size_type total_nonzero,
      const T scalarA,
      const T scalarB)
    {
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        gaxpy_process_one<T, NDIM>(std::get<0>(arg), std::get<1>(arg), std::get<2>(arg),
                                   scalarA, scalarB, std::get<3>(arg), local_pos);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(A_view, B_view, R_view, N, R, n_nonzero)`;
     * the leader marshals every member and submits one combined launch. The sparsity
     * of each member's output is copied to its device buffer as part of the single
     * H2D copy of the batch, also where it has no non-zero function.
     * total_functions bounds the sparsity staging.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_gaxpy_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<GaxpyBatchArg<T, NDIM>>& registry,
      const T scalarA,
      const T scalarB,
      size_type K,
      size_type total_functions)
    {
      using arg_t = GaxpyBatchArg<T, NDIM>;
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
        auto& m_out_tensor = batch[m].template get<4>();
        sparsity_to_bytes(m_out_tensor.sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), m_N);
        slot.extra_dsts.push_back(const_cast<T*>(m_out_tensor.buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
        slot.extra_sizes.push_back(m_N);
        sparsity_offset += m_N;
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(),
                               batch[m].template get<2>(), m_N);
        slot.offsets[m + 1] = slot.offsets[m] + batch[m].template get<5>();
      }
      const size_type total_nonzero = slot.offsets[nb];

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      if (total_nonzero > 0) {
        Dim3 thread_dims = max_thread_dims(2*K);
        CALL_KERNEL((gaxpy_kernel_batched<T, NDIM>), total_nonzero, thread_dims, 0, stream,
                    (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_nonzero, scalarA, scalarB));
        checkSubmit();
      }
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_GAXPY_H
