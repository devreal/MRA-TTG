#ifndef MRA_KERNELS_INNER_PRODUCT_H
#define MRA_KERNELS_INNER_PRODUCT_H

#include <array>
#include <tuple>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/ops/functions.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"

/**
 * Kernels for inner products between sets of functions in compressed form.
 *
 * Each kernel processes one node of the tree: it computes the node's own
 * contribution and adds the partial results already reduced over the node's
 * children, so that the result at the root is the inner product over the
 * whole tree. Children that do not exist pass a nullptr.
 *
 * In compressed (non-standard=false) form only the root carries scaling
 * coefficients, so the inner product of two functions is the sum over all
 * nodes of the dot product of their coefficients.
 */

namespace mra {
  namespace detail {

    /**
     * result[pos] = <a_pos|b_pos> at this node + sum over children.
     * Executed by one block; shared by inner_kernel and inner_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void inner_process_one(
      const concepts::TensorView<NDIM+1> auto& a_view,
      const concepts::TensorView<NDIM+1> auto& b_view,
      T* result,
      const std::array<const T*, Key<NDIM>::num_children()>& child_results,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> a, b;
      SHARED bool have_both;
      SHARED T node_sum;
      if (is_team_lead()) {
        have_both = a_view.is_nonzero(pos) && b_view.is_nonzero(pos);
        if (have_both) {
          a = a_view(pos);
          b = b_view(pos);
        }
      }
      SYNCTHREADS();
      if (have_both) {
        sumprod(a, b, &node_sum);
      }
      SYNCTHREADS();
      if (is_team_lead()) {
        T r = have_both ? node_sum : T(0);
        for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
          if (child_results[c] != nullptr) r += child_results[c][pos];
        }
        result[pos] = r;
      }
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    /**
     * result[i] = <a_i|b_i> at this node + sum over children.
     * One block per function.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void inner_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a_view,
      const concepts::TensorView<NDIM+1> auto b_view,
      T* result,
      std::array<const T*, Key<NDIM>::num_children()> child_results,
      size_type N)
    {
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        inner_process_one<T, NDIM>(a_view, b_view, result, child_results, pos);
      }
    }

    /**
     * result[pos*M+j] = <a_pos|b_j> at this node + sum over children, for all j.
     * Executed by one block; shared by matrix_inner_kernel and inner_kernel_batched.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void matrix_inner_process_one(
      const concepts::TensorView<NDIM+1> auto& a_view,
      const concepts::TensorView<NDIM+1> auto& b_view,
      T* result,
      const std::array<const T*, Key<NDIM>::num_children()>& child_results,
      size_type M,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> a, b;
      SHARED bool a_nonzero, b_nonzero;
      SHARED T node_sum;
      T* row = result + pos*M;
      if (is_team_lead()) {
        a_nonzero = a_view.is_nonzero(pos);
        if (a_nonzero) a = a_view(pos);
      }
      /* start the row from the sum over the children */
      for (size_type j = thread_id(); j < M; j += block_size()) {
        T r = 0;
        for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
          if (child_results[c] != nullptr) r += child_results[c][pos*M + j];
        }
        row[j] = r;
      }
      SYNCTHREADS();
      if (a_nonzero) {
        for (size_type j = 0; j < M; ++j) {
          if (is_team_lead()) {
            b_nonzero = b_view.is_nonzero(j);
            if (b_nonzero) b = b_view(j);
          }
          SYNCTHREADS();
          if (b_nonzero) {
            sumprod(a, b, &node_sum);
            SYNCTHREADS();
            if (is_team_lead()) row[j] += node_sum;
          }
          SYNCTHREADS();
        }
      }
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    /**
     * result[i*M+j] = <a_i|b_j> at this node + sum over children.
     * One block per row i.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void matrix_inner_kernel(
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a_view,
      const concepts::TensorView<NDIM+1> auto b_view,
      T* result,
      std::array<const T*, Key<NDIM>::num_children()> child_results,
      size_type N,
      size_type M)
    {
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        matrix_inner_process_one<T, NDIM>(a_view, b_view, result, child_results, M, pos);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_inner_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a,
    const concepts::TensorView<NDIM+1> auto& b,
    T* result,
    const std::array<const T*, Key<NDIM>::num_children()>& child_results,
    size_type N,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::inner_kernel, N, thread_dims, 0, stream,
                (key, a, b, result, child_results, N));
    checkSubmit();
  }

  template <typename T, Dimension NDIM>
  void submit_matrix_inner_kernel(
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a,
    const concepts::TensorView<NDIM+1> auto& b,
    T* result,
    const std::array<const T*, Key<NDIM>::num_children()>& child_results,
    size_type N,
    size_type M,
    size_type K,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    CALL_KERNEL(detail::matrix_inner_kernel, N, thread_dims, 0, stream,
                (key, a, b, result, child_results, N, M));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the inner product kernels, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/inner_product.h. There is no operator data,
   * so batching is unrestricted. The result of each node is dense and written in
   * full, so no sparsity needs to be pushed. The combined launch has one block per
   * function (Matrix == false) or row (Matrix == true) of every member.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using InnerBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>,                        // a
      SparseTensorView<T, NDIM+1>,                        // b
      T*,                                                 // result
      std::array<const T*, Key<NDIM>::num_children()>,    // child_results
      size_type,                                          // N: functions of a (rows)
      size_type                                           // M: functions of b (columns)
    >;

    template <typename T, Dimension NDIM, bool Matrix>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void inner_kernel_batched(
      InnerBatchArg<T, NDIM>* args,     // device ptr, size == num_members
      const size_type* member_offsets,  // device ptr, size == num_members+1
      size_type num_members,
      size_type total_blocks)
    {
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_blocks; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        if constexpr (Matrix) {
          matrix_inner_process_one<T, NDIM>(std::get<0>(arg), std::get<1>(arg), std::get<2>(arg),
                                            std::get<3>(arg), std::get<5>(arg), local_pos);
        } else {
          inner_process_one<T, NDIM>(std::get<0>(arg), std::get<1>(arg), std::get<2>(arg),
                                     std::get<3>(arg), local_pos);
        }
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(a_view, b_view, result, child_results,
     * N, M)`; the leader marshals every member and submits one combined launch.
     */
    template <typename T, Dimension NDIM, bool Matrix, typename BatchView>
    void submit_inner_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<InnerBatchArg<T, NDIM>>& registry,
      size_type K)
    {
      using arg_t = InnerBatchArg<T, NDIM>;
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      size_type total_blocks = 0;
      for (std::size_t m = 0; m < nb; ++m) total_blocks += batch[m].template get<4>();
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
        slot.args.emplace_back(b.template get<0>(), b.template get<1>(), b.template get<2>(),
                               b.template get<3>(), b.template get<4>(), b.template get<5>());
        slot.offsets[m + 1] = slot.offsets[m] + b.template get<4>();
      }

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      Dim3 thread_dims = max_thread_dims(2*K);
      CALL_KERNEL((inner_kernel_batched<T, NDIM, Matrix>), total_blocks, thread_dims, 0, stream,
                  (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_blocks));
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_INNER_PRODUCT_H
