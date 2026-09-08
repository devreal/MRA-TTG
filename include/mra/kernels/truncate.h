#ifndef MRA_KERNELS_TRUNCATE_H
#define MRA_KERNELS_TRUNCATE_H

#include <array>
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
     * Decides whether one function's node coefficients can be dropped: they
     * can be dropped only if none of the children still hold coefficients
     * for that function (child_kept[c][pos] == 0 for all c) and the node's
     * own norm is below tol. Nodes at level 0/1 are never dropped. The
     * node's sparsity is updated in place; `kept` receives the post-decision
     * keep/drop flag so it can be forwarded to the parent. Shared by both
     * the unbatched truncate_kernel below and truncate_kernel_batched
     * further down, so there is exactly one copy of this logic to maintain.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void truncate_process_one(
      Key<NDIM> key,
      concepts::TensorView<NDIM+1> auto& node,
      concepts::TensorView<1> auto& kept,
      const std::array<const T*, Key<NDIM>::num_children()>& child_kept,
      size_type pos,
      T tol)
    {
      constexpr auto num_children = Key<NDIM>::num_children();
      SHARED DenseTensorView<T, NDIM> n;
      if (node.is_zero(pos)) {
        if (is_team_lead()) {
          kept[pos] = T(0);
        }
        return; // skip zero-function entries
      }
      if (is_team_lead()) {
        n = node(pos);
      }
      SYNCTHREADS();
      bool keep;
      bool any_child_nonzero = false;
      for (size_type c = 0; c < num_children; ++c) {
        if (child_kept[c] != nullptr && child_kept[c][pos] != T(0.0)) {
          any_child_nonzero = true;
        }
      }
      if (any_child_nonzero || key.level() <= 1) {
        keep = true;
      } else {
        keep = !(normf(n) < tol);
      }
      if (is_team_lead()) {
        if (!keep) {
          node.set_zero(pos);
        }
        kept[pos] = keep ? T(1.0) : T(0.0);
      }
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void truncate_kernel(
      Key<NDIM> key,
      concepts::TensorView<NDIM+1> auto node,
      concepts::TensorView<1> auto kept,
      std::array<const T*, Key<NDIM>::num_children()> child_kept,
      size_type N,
      T tol)
    {
      for (size_type pos = blockIdx.x; pos < N; pos += gridDim.x) {
        truncate_process_one<T, NDIM>(key, node, kept, child_kept, pos, tol);
      }
    }
  } // namespace detail


  template <typename T, Dimension NDIM>
  void submit_truncate_kernel(
    const Key<NDIM>& key,
    concepts::TensorView<NDIM+1> auto&& node,
    concepts::TensorView<1> auto&& kept,
    const std::array<const T*, Key<NDIM>::num_children()>& child_kept,
    size_type N,
    T tol,
    ttg::device::Stream stream)
  {
    CALL_KERNEL(detail::truncate_kernel, N, MAX_THREADS_PER_BLOCK, 0, stream,
        (key, node, kept, child_kept, N, tol));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the truncate kernel, used by ttg::device::coop()/
   * TT::set_batch_matcher() in mra/tasks/truncate.h. Truncate has no shared
   * operator data either (tol itself varies per member -- it depends on
   * each member's own key.level() -- so it travels PER MEMBER, same as
   * key/node/kept/child_kept). Batching is therefore fully unrestricted. As
   * with the unbatched kernel above, there is no upfront nonzero detection:
   * this simply spawns one thread-block per function (0..N-1) for every
   * member and lets truncate_process_one skip zero functions itself -- the
   * combined launch is flattened over the SUM of each member's own N, not a
   * nonzero count.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using TruncateBatchArg = std::tuple<
      Key<NDIM>,                                                       // key (level needed for the keep decision)
      SparseTensorView<T, NDIM+1>,                                     // node
      DenseTensorView<T, 1>,                                           // kept
      std::array<const T*, Key<NDIM>::num_children()>,                 // child_kept
      size_type,                                                       // N: this member's own function count
      T                                                                // tol: this member's own truncate_tol(...)
    >;

    /* Named indices into TruncateBatchArg, so callers don't sprinkle magic
     * std::get<N> numbers across the kernel, submit function, and marshaling loop. */
    struct TruncateBatchArgIdx {
      static constexpr std::size_t key         = 0;
      static constexpr std::size_t node        = 1;
      static constexpr std::size_t kept        = 2;
      static constexpr std::size_t child_kept  = 3;
      static constexpr std::size_t n           = 4;
      static constexpr std::size_t tol         = 5;
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
    GLOBALSCOPE void truncate_kernel_batched(
      TruncateBatchArg<T, NDIM>* args,    // device ptr, size == num_members
      const size_type* member_offsets,    // device ptr, size == num_members+1
      size_type num_members,
      size_type total_blocks)
    {
      using idx = TruncateBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;

      for (size_type pos = blockIdx.x; pos < total_blocks; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        truncate_process_one<T, NDIM>(std::get<idx::key>(arg), std::get<idx::node>(arg),
                                      std::get<idx::kept>(arg), std::get<idx::child_kept>(arg),
                                      local_pos, std::get<idx::tol>(arg));
      }
    }

  } // namespace detail

  /**
   * Batched counterpart of submit_truncate_kernel: launches one kernel on
   * behalf of every member already marshaled into slot.args (by the caller,
   * via detail::submit_truncate_batch_leader below).
   */
  template <typename T, Dimension NDIM>
  void submit_truncate_kernel_batched(
    detail::GroupedBatchPool<detail::TruncateBatchArg<T, NDIM>>& pool,
    typename detail::GroupedBatchPool<detail::TruncateBatchArg<T, NDIM>>::slot_t& slot,
    size_type total_blocks,
    ttg::device::Stream stream)
  {
    using arg_t = detail::TruncateBatchArg<T, NDIM>;
    const size_type num_members = static_cast<size_type>(slot.args.size());

    // Single combined H2D transfer for args/offsets -- no extra per-member
    // copies needed: kept is a plain dense buffer, fully covered by this
    // member's own N blocks, and node's sparsity already lives on the
    // device from whatever produced it.
    detail::submit_grouped_copy<arg_t>(slot, num_members, slot.offsets.size(), pool.device_id, stream);

    CALL_KERNEL((detail::truncate_kernel_batched<T, NDIM>), total_blocks, MAX_THREADS_PER_BLOCK, 0, stream,
                (slot.dev_args, slot.dev_offsets, num_members, total_blocks));
    checkSubmit();

    pool.mark_submitted(slot, stream);
  }

  namespace detail {

    /**
     * Shared by truncate.h's merge task: given the batch_view returned by
     * its own `co_await ttg::device::coop<Key<NDIM>>(...)` (which must stay
     * inline in the coroutine -- only the ordinary, non-suspending code
     * below is worth sharing), marshal every member into the current
     * device's pool and submit one combined kernel launch if this task is
     * the batch's leader.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_truncate_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<TruncateBatchArg<T, NDIM>>& registry)
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
        auto& m_key        = batch[m].template get<0>();
        auto& m_node       = batch[m].template get<1>();
        auto& m_kept       = batch[m].template get<2>();
        auto& m_child_kept = batch[m].template get<3>();
        const size_type m_N = batch[m].template get<4>();
        const T m_tol        = batch[m].template get<5>();

        slot.args.emplace_back(m_key, m_node, m_kept, m_child_kept, m_N, m_tol);
        slot.offsets[m + 1] = slot.offsets[m] + m_N;
      }
      const size_type total_blocks = slot.offsets[nb];
      submit_truncate_kernel_batched<T, NDIM>(pool, slot, total_blocks, ttg::device::current_stream());
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_TRUNCATE_H
