#ifndef MRA_KERNELS_SIMPLE_NORM_H
#define MRA_KERNELS_SIMPLE_NORM_H

#include <tuple>

#include "mra/misc/platform.h"
#include "mra/misc/types.h"
#include "mra/misc/key.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {

    /**
     * Processes one function of one node: computes its norm, or writes 0 for
     * a zero function -- the per-block body shared by both the unbatched
     * simple_norm_kernel below and simple_norm_kernel_batched further down,
     * so there is exactly one copy of this logic to maintain.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void simple_norm_process_one(
      const concepts::TensorView<NDIM+1> auto& node,
      concepts::TensorView<1> auto& result_norms,
      size_type pos)
    {
      SHARED DenseTensorView<const T, NDIM> n;
      if (node.is_zero(pos)) {
        if (is_team_lead()) {
          result_norms[pos] = T(0);
        }
        return; // skip zero-function entries
      }
      if (is_team_lead()) {
        n = node(pos);
      }
      SYNCTHREADS();
      T norm = normf(n);
      if (is_team_lead()) {
        result_norms[pos] = norm;
      }
    }

    template <Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void simple_norm_kernel(
      Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto node,
      concepts::TensorView<1> auto result_norms,
      size_type N)
    {
      using T = typename std::remove_reference_t<decltype(node)>::value_type;
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        simple_norm_process_one<T, NDIM>(node, result_norms, pos);
      }
    }
  } // namespace detail


  template <Dimension NDIM>
  void submit_simple_norm_kernel(
    Key<NDIM> key,
    const concepts::TensorView<NDIM+1> auto&& in,
    size_type N,
    concepts::TensorView<1> auto&& result_norms)
  {
    /* simple norm calculation can use as many threads as are available */
    CALL_KERNEL(detail::simple_norm_kernel, N, MAX_THREADS_PER_BLOCK, 0, ttg::device::current_stream(),
        (key, in, result_norms, N));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the simple_norm kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/convolution.h. Unlike compress,
   * simple_norm has no shared operator data at all -- every per-member
   * argument is just its own node/result views and its own function count N.
   * Batching is therefore fully unrestricted (any two members may batch,
   * regardless of level or position). And, as with the unbatched kernel
   * above, there is no upfront nonzero detection: this simply spawns one
   * thread-block per function (0..N-1) for every member and lets
   * simple_norm_process_one skip zero functions itself -- the combined
   * launch is flattened over the SUM of each member's own N, not a nonzero
   * count.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using SimpleNormBatchArg = std::tuple<
      SparseTensorView<T, NDIM+1>, // node
      DenseTensorView<T, 1>,       // result_norms
      size_type                    // N: this member's own function count
    >;

    /* Named indices into SimpleNormBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct SimpleNormBatchArgIdx {
      static constexpr std::size_t node         = 0;
      static constexpr std::size_t result_norms = 1;
      static constexpr std::size_t n            = 2;
    };

    /**
     * One combined launch covering every function (0..N-1, not just non-zero
     * ones) across all members of the batch, flattened into a single 1D grid
     * of size total_blocks (the sum, across members, of each member's own
     * N). member_offsets (size num_members+1) names, for a given global grid
     * position, which member a block belongs to and that member's own local
     * position (find_member_for_pos) -- see compress_kernel_batched in
     * mra/kernels/compress.h for the fuller version of this comment.
     */
    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void simple_norm_kernel_batched(
      SimpleNormBatchArg<T, NDIM>* args,  // device ptr, size == num_members
      const size_type* member_offsets,    // device ptr, size == num_members+1
      size_type num_members,
      size_type total_blocks)
    {
      using idx = SimpleNormBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;

      for (size_type pos = blockIdx.x; pos < total_blocks; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        simple_norm_process_one<T, NDIM>(std::get<idx::node>(arg), std::get<idx::result_norms>(arg), local_pos);
      }
    }

  } // namespace detail

  /**
   * Batched counterpart of submit_simple_norm_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.args (by the caller,
   * via detail::submit_simple_norm_batch_leader below).
   */
  template <typename T, Dimension NDIM>
  void submit_simple_norm_kernel_batched(
    detail::GroupedBatchPool<detail::SimpleNormBatchArg<T, NDIM>>& pool,
    typename detail::GroupedBatchPool<detail::SimpleNormBatchArg<T, NDIM>>::slot_t& slot,
    size_type total_blocks,
    ttg::device::Stream stream)
  {
    using arg_t = detail::SimpleNormBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.args.size());

    // Single combined H2D transfer for args/offsets -- no extra per-member
    // copies needed (unlike compress's sparsity staging): result_norms is a
    // plain dense buffer, fully covered by this member's own N blocks, and
    // node's sparsity already lives on the device from whatever produced it.
    detail::submit_grouped_copy<arg_t>(slot, num_members, slot.offsets.size(), pool.device_id, stream);

    CALL_KERNEL((detail::simple_norm_kernel_batched<T, NDIM>), total_blocks, MAX_THREADS_PER_BLOCK, 0, stream,
                (slot.dev_args, slot.dev_offsets, num_members, total_blocks));
    checkSubmit();

    pool.mark_submitted(slot, stream);
  }

  namespace detail {

    /**
     * Shared by convolution.h's norm_tt: given the batch_view returned by
     * its own `co_await ttg::device::coop<Key<NDIM>>(...)` (which must stay
     * inline in the coroutine -- only the ordinary, non-suspending code
     * below is worth sharing), marshal every member into the current
     * device's pool and submit one combined kernel launch if this task is
     * the batch's leader.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_simple_norm_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<SimpleNormBatchArg<T, NDIM>>& registry)
    {
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      const auto device = ttg::device::current_device();
      auto& pool = registry.get(device);
      auto& slot = pool.acquire(registry.get_max_batch_size(), registry.get_max_batch_size() + 1, /*num_sparsity=*/0);
      slot.args.clear();
      slot.offsets.resize(nb + 1);
      slot.offsets[0] = 0;
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();

      for (std::size_t m = 0; m < nb; ++m) {
        auto& m_node         = batch[m].template get<0>();
        auto& m_result_norms = batch[m].template get<1>();
        const size_type m_N  = batch[m].template get<2>();

        slot.args.emplace_back(m_node, m_result_norms, m_N);
        slot.offsets[m + 1] = slot.offsets[m] + m_N;
      }
      const size_type total_blocks = slot.offsets[nb];
      submit_simple_norm_kernel_batched<T, NDIM>(pool, slot, total_blocks, ttg::device::current_stream());
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_SIMPLE_NORM_H
