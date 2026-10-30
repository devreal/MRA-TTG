#ifndef MRA_KERNELS_GRADIENT_COMPRESSED_H
#define MRA_KERNELS_GRADIENT_COMPRESSED_H

#include <cmath>
#include <array>
#include <cstdint>
#include <tuple>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/reconstruct.h"
#include "mra/kernels/derivative.h"
#include "mra/misc/device_batch_pool.h"
#include "mra/tensor/sparsity.h"

/**
 * Spawn kernel of make_gradient_compressed: the scaling coefficients of the children
 * of a node, per function either unfiltered from [s, d] (the input tree continues
 * here) or the node's own coefficients passed on unchanged (the input ended here or
 * above; the children record that they belong to an ancestor, see
 * FunctionsReconstructedNode::origin).
 */

namespace mra {

  namespace detail {

    /**
     * Function `j` of node `key`, executed by one block: inner[j] != 0 unfilters
     * [s, d] (d from the compressed node, s passed down, or the s-block of the
     * compressed root) into the children, else every child receives s unchanged.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void gradient_spawn_process_one(
      const Key<NDIM>& key,
      size_type K,
      const concepts::TensorView<NDIM+1> auto& node_view,
      const concepts::TensorView<NDIM+1> auto& s_view,
      const concepts::TensorView<2> auto& hg,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& children,
      const std::uint8_t* inner,
      T* tmp,
      size_type j)
    {
      SHARED DenseTensorView<const T, NDIM> d, s;
      SHARED DenseTensorView<T, NDIM> u, t, dst;
      SHARED T* workspace;
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      if (is_team_lead()) {
        T* block_tmp = &tmp[j*reconstruct_tmp_size<NDIM>(K)];
        u = DenseTensorView<T, NDIM>(&block_tmp[0], 2*K);
        t = DenseTensorView<T, NDIM>(&block_tmp[TWOK2NDIM], 2*K);
        workspace = &block_tmp[2*TWOK2NDIM];
        d = node_view(j);
        s = s_view(j);
      }
      SYNCTHREADS();
      if (inner[j]) {
        t = d; // zero without a compressed node
        SYNCTHREADS();
        if (key.level() > 0) {
          t(get_child_slice<NDIM>(key, K, 0)) = s;
          SYNCTHREADS();
        }
        transform(t, hg, u, workspace);
        SYNCTHREADS();
      }
      for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
        if (is_team_lead()) dst = children[c](j);
        SYNCTHREADS();
        if (inner[j]) dst = u(get_child_slice<NDIM>(key, K, c));
        else          dst = s; // zero where the function has no coefficients
        SYNCTHREADS();
      }
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gradient_spawn_kernel(
      const Key<NDIM> key,
      size_type N,
      size_type K,
      const concepts::TensorView<NDIM+1> auto node_view,
      const concepts::TensorView<NDIM+1> auto s_view,
      const concepts::TensorView<2> auto hg,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto children,
      const std::uint8_t* inner,
      T* tmp)
    {
      for (size_type j = blockIdx.x; j < N; j += gridDim.x) {
        gradient_spawn_process_one<T, NDIM>(key, K, node_view, s_view, hg, children, inner, tmp, j);
      }
    }
  } // namespace detail

  /**
   * One block per function; every child holds every function (dense), tmp holds
   * reconstruct_tmp_size(K) per function.
   */
  template <typename T, Dimension NDIM>
  void submit_gradient_spawn_kernel(
    const Key<NDIM>& key,
    size_type N,
    size_type K,
    const concepts::TensorView<NDIM+1> auto& node_view,
    const concepts::TensorView<NDIM+1> auto& s_view,
    const concepts::TensorView<2> auto& hg,
    const concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& children,
    const std::uint8_t* inner,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CALL_KERNEL(detail::gradient_spawn_kernel, N, thread_dims, smem_size, stream,
      (key, N, K, node_view, s_view, hg, children, inner, tmp));
    checkSubmit();
  }

  namespace detail {
    /**
     * The derivatives along every axis at key in one launch: the blocks are flattened
     * over the non-zero functions of the results of all axes, axis by axis
     * (axis_offsets[a] is the first position of axis a, axis_offsets[NDIM] the total),
     * so the axes run concurrently. Every coefficient is in the nodes themselves, so
     * the "inherited" views of derivative_process_one are the same nodes. The views
     * are named template parameters, see derivative_process_one.
     */
    template <typename T, Dimension NDIM, typename ViewT, typename OperatorsT,
              typename PhiT, typename PhibarT, typename QuadXT>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gradient_derivative_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const Key<NDIM>* src_keys,
      const std::array<ViewT, 2*NDIM> neighbors,
      const ViewT center,
      std::array<ViewT, NDIM> results,
      const OperatorsT operators,
      const PhiT phi,
      const PhibarT phibar,
      const QuadXT quad_x,
      T* tmp,
      size_type N,
      const std::array<size_type, NDIM+1> axis_offsets,
      size_type K,
      const int bc_left,
      const int bc_right)
    {
      for (size_type pos = blockIdx.x; pos < axis_offsets[NDIM]; pos += gridDim.x) {
        size_type a = 0;
        while (pos >= axis_offsets[a+1]) ++a;
        derivative_process_one<T, NDIM>(D, key, src_keys + 3*a*N, neighbors[2*a], neighbors[2*a], center, center,
                                        neighbors[2*a+1], neighbors[2*a+1], operators, results[a],
                                        phi, phibar, quad_x, tmp, N, pos - axis_offsets[a], pos, K,
                                        T(0), T(0), a, bc_left, bc_right);
      }
    }
  } // namespace detail

  /**
   * The derivatives along every axis at key, see detail::gradient_derivative_kernel:
   * neighbors holds the left and right neighbor of each axis (2*a, 2*a+1), src_keys
   * (device memory) 3*N keys per axis (see submit_derivative_kernel), n_nonzero the
   * number of non-zero functions of each result, and tmp derivative_tmp_size(K) per
   * non-zero function of all results together.
   */
  template <typename T, Dimension NDIM, typename ViewT, typename OperatorsT,
            typename PhiT, typename PhibarT, typename QuadXT>
  void submit_gradient_derivative_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const Key<NDIM>* src_keys,
    const std::array<ViewT, 2*NDIM>& neighbors,
    const ViewT& center,
    std::array<ViewT, NDIM>& results,
    const OperatorsT& operators,
    const PhiT& phi,
    const PhibarT& phibar,
    const QuadXT& quad_x,
    T* tmp,
    size_type N,
    const std::array<size_type, NDIM>& n_nonzero,
    size_type K,
    const int bc_left,
    const int bc_right,
    ttg::device::Stream stream)
  {
    std::array<size_type, NDIM+1> axis_offsets{};
    for (Dimension a = 0; a < NDIM; ++a) axis_offsets[a+1] = axis_offsets[a] + n_nonzero[a];
    if (axis_offsets[NDIM] == 0) return;
    const size_type smem_size = detail::derivative_smem_size<T, NDIM>(K);
    CALL_KERNEL((detail::gradient_derivative_kernel<T, NDIM>), axis_offsets[NDIM], detail::derivative_thread_dims(K),
      smem_size, stream,
      (D, key, src_keys, neighbors, center, results, operators, phi, phibar, quad_x, tmp, N, axis_offsets, K,
       bc_left, bc_right));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for the compute task of make_gradient_compressed. A task joins
   * one batch only, so the leader issues every launch the members need: one batched
   * derivative launch over every member and axis with derivatives (one entry per
   * member and axis, the axis travelling with the entry), and one batched spawn
   * launch over the members with children. The operator data is the same for every
   * node, so batching is unrestricted.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using GradientDerivativeBatchArg = std::tuple<
      Key<NDIM>,                   // key
      const Key<NDIM>*,            // src_keys of the axis
      SparseTensorView<T, NDIM+1>, // left neighbor along the axis
      SparseTensorView<T, NDIM+1>, // S
      SparseTensorView<T, NDIM+1>, // right neighbor along the axis
      SparseTensorView<T, NDIM+1>, // derivative along the axis
      size_type,                   // N
      size_type                    // axis
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gradient_derivative_kernel_batched(
      GradientDerivativeBatchArg<T, NDIM>* args, // device ptr, size == num_entries
      const size_type* entry_offsets,            // device ptr, size == num_entries+1
      size_type num_entries,
      size_type total_nonzero,
      T* shared_tmp,                             // occupancy-sized scratch of the batch slot, indexed by block
      const Domain<NDIM>& D,
      const concepts::TensorView<3> auto operators,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<1> auto quad_x,
      size_type K,
      const int bc_left,
      const int bc_right)
    {
      SHARED size_type entry;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_nonzero; pos += gridDim.x) {
        if (is_team_lead()) {
          entry = find_member_for_pos(entry_offsets, num_entries, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[entry];
        derivative_process_one<T, NDIM>(D, std::get<0>(arg), std::get<1>(arg),
                                        std::get<2>(arg), std::get<2>(arg), std::get<3>(arg), std::get<3>(arg),
                                        std::get<4>(arg), std::get<4>(arg), operators, std::get<5>(arg),
                                        phi, phibar, quad_x, shared_tmp, std::get<6>(arg), local_pos, blockIdx.x,
                                        K, T(0), T(0), std::get<7>(arg), bc_left, bc_right);
      }
    }

    template <typename T, Dimension NDIM>
    using GradientSpawnBatchArg = std::tuple<
      Key<NDIM>,                                                          // key
      SparseTensorView<T, NDIM+1>,                                        // compressed node
      SparseTensorView<T, NDIM+1>,                                        // S
      std::array<SparseTensorView<T, NDIM+1>, Key<NDIM>::num_children()>, // S of the children
      const std::uint8_t*,                                                // unfilter flags
      T*,                                                                 // tmp: indexed by function
      size_type                                                           // N
    >;

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void gradient_spawn_kernel_batched(
      GradientSpawnBatchArg<T, NDIM>* args, // device ptr, size == num_members
      const size_type* member_offsets,      // device ptr, size == num_members+1
      size_type num_members,
      size_type total,
      size_type K,
      const concepts::TensorView<2> auto hg)
    {
      SHARED size_type member;
      SHARED size_type j;
      for (size_type pos = blockIdx.x; pos < total; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &j);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        gradient_spawn_process_one<T, NDIM>(std::get<0>(arg), K, std::get<1>(arg), std::get<2>(arg), hg,
                                            std::get<3>(arg), std::get<4>(arg), std::get<5>(arg), j);
        SYNCTHREADS(); // member and j are rewritten by the next iteration
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(key, S_view, nb_views, result_views,
     * results, n_nonzero, src_keys, tmp, has_children, cnode_view, child_views,
     * children, unfilter, N)`: nb_views the six neighbors (left and right of each axis),
     * results/children the nodes (for their sparsity), n_nonzero the number of
     * derivatives along each axis, tmp the derivatives' scratch for the axes in order
     * followed by the children's (batched, only the children's is used: the derivatives'
     * scratch is the batch slot's). max_functions bounds the number of functions of a node.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_gradient_compute_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<GradientDerivativeBatchArg<T, NDIM>>& deriv_registry,
      GroupedBatchPoolRegistry<GradientSpawnBatchArg<T, NDIM>>& spawn_registry,
      const Domain<NDIM>& D,
      const concepts::TensorView<3> auto& operators,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<1> auto& quad_x,
      const concepts::TensorView<2> auto& hg,
      size_type K,
      int bc_left,
      int bc_right,
      size_type max_functions)
    {
      constexpr size_type num_children = Key<NDIM>::num_children();
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      const auto device = ttg::device::current_device();
      auto stream = ttg::device::current_stream();
      const std::size_t max_batch = deriv_registry.get_max_batch_size();

      /* push the sparsity of `node` with the slot's grouped copy */
      auto stage_sparsity = [&](auto& slot, auto& node, size_type& offset) {
        const size_type n = node.coeffs().dim(0);
        sparsity_to_bytes(node.coeffs().sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[offset]), n);
        slot.extra_dsts.push_back(const_cast<T*>(node.coeffs().buffer().device_ptr_on(device)));
        slot.extra_srcs.push_back(&slot.sparsity[offset]);
        slot.extra_sizes.push_back(n);
        offset += n;
      };
      auto reset = [](auto& slot) {
        slot.args.clear();
        slot.offsets.clear();
        slot.offsets.push_back(0);
        slot.extra_dsts.clear();
        slot.extra_srcs.clear();
        slot.extra_sizes.clear();
      };

      /* the derivatives of every member and axis, one launch */
      size_type deriv_bytes = 0;
      std::size_t entries = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        for (Dimension a = 0; a < NDIM; ++a) {
          if (batch[m].template get<5>()[a] == 0) continue;
          deriv_bytes += batch[m].template get<4>()[a].coeffs().dim(0);
          ++entries;
        }
      }
      if (entries > 0) {
        auto& deriv_pool = deriv_registry.get(device);
        auto& slot = deriv_pool.acquire(NDIM * max_batch, NDIM * max_batch + 1, NDIM * max_batch * max_functions);
        reset(slot);
        slot.sparsity.resize(deriv_bytes);
        size_type sparsity_offset = 0;
        for (std::size_t m = 0; m < nb; ++m) {
          const auto& n_nonzero = batch[m].template get<5>();
          const size_type N = batch[m].template get<13>();
          auto& nbv = batch[m].template get<2>();
          auto& center = batch[m].template get<1>();
          for (Dimension a = 0; a < NDIM; ++a) {
            if (n_nonzero[a] == 0) continue;
            stage_sparsity(slot, batch[m].template get<4>()[a], sparsity_offset);
            slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<6>() + 3*a*N,
                                   nbv[2*a], center, nbv[2*a + 1], batch[m].template get<3>()[a], N, size_type(a));
            slot.offsets.push_back(slot.offsets.back() + n_nonzero[a]);
          }
        }
        const size_type total = slot.offsets.back();
        submit_grouped_copy<GradientDerivativeBatchArg<T, NDIM>>(slot, entries, slot.offsets.size(),
                                                                 deriv_pool.device_id, stream);
        /* the scratch is the slot's, one region per block of the occupancy-capped grid,
         * see submit_derivative_batch_leader */
        const Dim3 thread_dims = derivative_thread_dims(K);
        const int threads = static_cast<int>(thread_dims.x * thread_dims.y * thread_dims.z);
        const size_type smem_size = derivative_smem_size<T, NDIM>(K);
        auto kernel = gradient_derivative_kernel_batched<T, NDIM, std::decay_t<decltype(operators)>,
                                                         std::decay_t<decltype(phi)>, std::decay_t<decltype(phibar)>,
                                                         std::decay_t<decltype(quad_x)>>;
        const size_type grid_cap = kernel_grid_cap(kernel, threads, smem_size);
        const size_type grid_x = std::min<size_type>(total, grid_cap);
        T* shared_tmp = deriv_pool.template ensure_scratch<T>(slot, static_cast<std::size_t>(grid_cap) * derivative_tmp_size<NDIM>(K));
        CALL_KERNEL((gradient_derivative_kernel_batched<T, NDIM>), grid_x, thread_dims, smem_size, stream,
                    (slot.dev_args, slot.dev_offsets, static_cast<size_type>(entries), total, shared_tmp, D,
                     operators, phi, phibar, quad_x, K, bc_left, bc_right));
        checkSubmit();
        deriv_pool.mark_submitted(slot, stream);
      }

      /* S of the children */
      size_type bytes = 0;
      std::size_t members = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        if (!batch[m].template get<8>()) continue;
        for (auto& c : batch[m].template get<11>()) bytes += c.coeffs().dim(0);
        ++members;
      }
      if (members == 0) return;
      auto& spawn_pool = spawn_registry.get(device);
      auto& slot = spawn_pool.acquire(max_batch, max_batch + 1, max_batch * num_children * max_functions);
      reset(slot);
      slot.sparsity.resize(bytes);
      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        if (!batch[m].template get<8>()) continue;
        const auto& n_nonzero = batch[m].template get<5>();
        const size_type N = batch[m].template get<13>();
        size_type deriv_items = 0;
        for (Dimension b = 0; b < NDIM; ++b) deriv_items += n_nonzero[b];
        for (auto& c : batch[m].template get<11>()) stage_sparsity(slot, c, sparsity_offset);
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<9>(), batch[m].template get<1>(),
                               batch[m].template get<10>(), batch[m].template get<12>(),
                               batch[m].template get<7>() + deriv_items*derivative_tmp_size<NDIM>(K), N);
        slot.offsets.push_back(slot.offsets.back() + N);
      }
      const size_type total = slot.offsets.back();
      submit_grouped_copy<GradientSpawnBatchArg<T, NDIM>>(slot, members, slot.offsets.size(), spawn_pool.device_id, stream);
      CALL_KERNEL((gradient_spawn_kernel_batched<T, NDIM>), total, max_thread_dims(2*K), mTxmq_shmem_size<T>(2*K),
                  stream, (slot.dev_args, slot.dev_offsets, static_cast<size_type>(members), total, K, hg));
      checkSubmit();
      spawn_pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_GRADIENT_COMPRESSED_H
