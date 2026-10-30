#ifndef MRA_KERNELS_MULTIPLY_COMPRESSED_H
#define MRA_KERNELS_MULTIPLY_COMPRESSED_H

#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "mra/misc/key.h"
#include "mra/misc/domain.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"
#include "mra/kernels/transform.h"
#include "mra/kernels/fcube_for_mul.h"
#include "mra/misc/device_batch_pool.h"

/**
 * Kernels for the product of functions given in compressed form, fused with the
 * descent that reconstruct would do (Algorithm B of "Compressed-form
 * multiplication in MADNESS"): at a node, [s, d] of each factor is unfiltered
 * into the scaling coefficients of the children, which are either multiplied
 * right away (where the product ends at the child) or passed down. The product
 * itself is the leaf product of madness::FunctionImpl::do_mul: values at the
 * quadrature points of the leaf, product, back to coefficients.
 */

namespace mra {

  namespace detail {

    /** what a work item does for one child of the node */
    enum MulCompressedChild : std::uint8_t {
      MULC_NONE = 0,  // nothing
      MULC_LEAF = 1,  // the product ends at the child: multiply into the child's product node
      MULC_PASS = 2,  // the child has a task: pass the scaling coefficients of both factors down
    };

    /**
     * One output function at one node. type 0: the product ends at the node itself,
     * multiply the scaling coefficients passed down by the parent. type 1: unfilter both
     * factors and handle the children as `child` says.
     */
    template <Dimension NDIM>
    struct MulCompressedWork {
      size_type j;        // output function, and function of the second factor
      size_type ia;       // function of the first factor (j, or 0 when it is broadcast)
      std::uint8_t type;
      std::uint8_t a_inner; // the first factor has wavelet coefficients for ia at the node
      std::uint8_t b_inner; // likewise the second factor for j
      std::uint8_t write_a; // this item passes the first factor down (one item per function ia)
      std::array<std::uint8_t, Key<NDIM>::num_children()> child;
    };

  } // namespace detail

  template<Dimension NDIM>
  SCOPE size_type multiply_compressed_tmp_size(size_type K) {
    const size_type K2NDIM = std::pow(K, NDIM);
    const size_type TWOK2NDIM = std::pow(2*K, NDIM);
    return 4*TWOK2NDIM + 4*K2NDIM; // t, ua, ub, workspace; sa, sb, value cubes a and b
  }

  namespace detail {

    /**
     * Scaling coefficients of the children of `key` from [s, d] of one function:
     * d from the compressed node (null if there is none), s passed down by the
     * parent, or the s-block of the compressed node at the root (as reconstruct).
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_compressed_unfilter(
      const Key<NDIM>& key,
      size_type K,
      const DenseTensorView<const T, NDIM>& d,
      const DenseTensorView<const T, NDIM>& s,
      DenseTensorView<T, NDIM>& t,
      DenseTensorView<T, NDIM>& u,
      const concepts::TensorView<2> auto& hg,
      T* workspace)
    {
      t = d; // zero if there is no node
      SYNCTHREADS();
      if (key.level() > 0) {
        t(get_child_slice<NDIM>(key, K, 0)) = s;
        SYNCTHREADS();
      }
      transform(t, hg, u, workspace);
      SYNCTHREADS();
    }

    /**
     * out = a*b at `key` (b == a for a square), from scaling coefficients at `key`.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_compressed_product(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const DenseTensorView<const T, NDIM>& a,
      const DenseTensorView<const T, NDIM>& b,
      bool square,
      DenseTensorView<T, NDIM>& acube,
      DenseTensorView<T, NDIM>& bcube,
      DenseTensorView<T, NDIM>& out,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      size_type K,
      T* workspace)
    {
      if (a.data() == nullptr) acube = T(0);
      else fcube_for_mul<T, NDIM>(D, key, key, a, acube, phibar, phi, quad_x, K, workspace);
      SYNCTHREADS();
      if (square) {
        foreach_idx(acube, [&](size_type i) { acube[i] *= acube[i]; });
      } else {
        if (b.data() == nullptr) bcube = T(0);
        else fcube_for_mul<T, NDIM>(D, key, key, b, bcube, phibar, phi, quad_x, K, workspace);
        SYNCTHREADS();
        foreach_idx(acube, [&](size_type i) { acube[i] *= bcube[i]; });
      }
      SYNCTHREADS();
      transform(acube, phibar, out, workspace);
      const T scale = std::pow(T(0.5), T(0.5*NDIM*key.level())) * std::sqrt(D.template get_volume<T>());
      out *= scale;
      SYNCTHREADS();
    }

    /**
     * Work item `pos` at node `key`, executed by one block.
     *
     * s_in holds the scaling coefficients passed down by the parent: functions
     * [0, NA) of the first factor, then (unless square) [NA, NA+N) of the second.
     * out_arr[c] is, for each child c, either the product node of c (a child without
     * a task, indexed by j) or the scaling coefficients passed to the task of c (laid
     * out as s_in). result is the product node of `key`.
     */
    template <typename T, Dimension NDIM>
    DEVSCOPE void multiply_compressed_process_one(
      const Domain<NDIM>& D,
      const Key<NDIM>& key,
      const concepts::TensorView<NDIM+1> auto& a_node,
      const concepts::TensorView<NDIM+1> auto& b_node,
      const concepts::TensorView<NDIM+1> auto& s_in,
      concepts::TensorView<NDIM+1> auto& result,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& out_arr,
      const concepts::TensorView<2> auto& hg,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      const MulCompressedWork<NDIM>* work,
      size_type NA,
      bool square,
      size_type K,
      T* tmp,
      size_type pos)
    {
      SHARED MulCompressedWork<NDIM> w;
      SHARED DenseTensorView<const T, NDIM> da, db, sa_in, sb_in, ca, cb;
      SHARED DenseTensorView<T, NDIM> t, ua, ub, sa, sb, acube, bcube, dst;
      SHARED T* workspace;
      const size_type K2NDIM = std::pow(K, NDIM);
      const size_type TWOK2NDIM = std::pow(2*K, NDIM);
      if (is_team_lead()) {
        w = work[pos];
        T* block_tmp = &tmp[pos*multiply_compressed_tmp_size<NDIM>(K)];
        t  = DenseTensorView<T, NDIM>(&block_tmp[0*TWOK2NDIM], 2*K, 2*K, 2*K);
        ua = DenseTensorView<T, NDIM>(&block_tmp[1*TWOK2NDIM], 2*K, 2*K, 2*K);
        ub = DenseTensorView<T, NDIM>(&block_tmp[2*TWOK2NDIM], 2*K, 2*K, 2*K);
        workspace = &block_tmp[3*TWOK2NDIM];
        sa    = DenseTensorView<T, NDIM>(&block_tmp[4*TWOK2NDIM + 0*K2NDIM], K, K, K);
        sb    = DenseTensorView<T, NDIM>(&block_tmp[4*TWOK2NDIM + 1*K2NDIM], K, K, K);
        acube = DenseTensorView<T, NDIM>(&block_tmp[4*TWOK2NDIM + 2*K2NDIM], K, K, K);
        bcube = DenseTensorView<T, NDIM>(&block_tmp[4*TWOK2NDIM + 3*K2NDIM], K, K, K);
        /* wavelet coefficients only where the factor's tree continues below the node */
        if (w.a_inner) da = a_node(w.ia);
        else           da = DenseTensorView<const T, NDIM>(nullptr, 2*K, 2*K, 2*K);
        sa_in = s_in(w.ia);
        if (!square) {
          if (w.b_inner) db = b_node(w.j);
          else           db = DenseTensorView<const T, NDIM>(nullptr, 2*K, 2*K, 2*K);
          sb_in = s_in(NA + w.j);
        }
      }
      SYNCTHREADS();
      if (w.type == 0) {
        /* the product ends here */
        if (is_team_lead()) dst = result(w.j);
        SYNCTHREADS();
        multiply_compressed_product<T, NDIM>(D, key, sa_in, sb_in, square, acube, bcube, dst,
                                             phibar, phi, quad_x, K, workspace);
      } else {
        multiply_compressed_unfilter<T, NDIM>(key, K, da, sa_in, t, ua, hg, workspace);
        if (!square) multiply_compressed_unfilter<T, NDIM>(key, K, db, sb_in, t, ub, hg, workspace);
        for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
          const auto child_slice = get_child_slice<NDIM>(key, K, c);
          if (w.child[c] == MULC_LEAF) {
            if (is_team_lead()) {
              dst = out_arr[c](w.j);
              ca = DenseTensorView<const T, NDIM>(sa.data(), K, K, K);
              cb = DenseTensorView<const T, NDIM>(sb.data(), K, K, K);
            }
            SYNCTHREADS();
            sa = ua(child_slice);
            if (!square) sb = ub(child_slice);
            SYNCTHREADS();
            multiply_compressed_product<T, NDIM>(D, key.child_at(c), ca, cb, square, acube, bcube, dst,
                                                 phibar, phi, quad_x, K, workspace);
          } else if (w.child[c] == MULC_PASS) {
            if (w.write_a) {
              if (is_team_lead()) dst = out_arr[c](w.ia);
              SYNCTHREADS();
              dst = ua(child_slice);
              SYNCTHREADS();
            }
            if (!square) {
              if (is_team_lead()) dst = out_arr[c](NA + w.j);
              SYNCTHREADS();
              dst = ub(child_slice);
              SYNCTHREADS();
            }
          }
        }
      }
      SYNCTHREADS(); // shared state is rewritten by the next call
    }

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void multiply_compressed_kernel(
      const Domain<NDIM>& D,
      const Key<NDIM> key,
      const concepts::TensorView<NDIM+1> auto a_node,
      const concepts::TensorView<NDIM+1> auto b_node,
      const concepts::TensorView<NDIM+1> auto s_in,
      concepts::TensorView<NDIM+1> auto result,
      concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto out_arr,
      const concepts::TensorView<2> auto hg,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<1> auto quad_x,
      const MulCompressedWork<NDIM>* work,
      size_type NA,
      bool square,
      size_type n_items,
      size_type K,
      T* tmp)
    {
      for (size_type pos = blockIdx.x; pos < n_items; pos += gridDim.x) {
        multiply_compressed_process_one<T, NDIM>(D, key, a_node, b_node, s_in, result, out_arr,
                                                 hg, phibar, phi, quad_x, work, NA, square, K, tmp, pos);
      }
    }
  } // namespace detail

  template <typename T, Dimension NDIM>
  void submit_multiply_compressed_kernel(
    const Domain<NDIM>& D,
    const Key<NDIM>& key,
    const concepts::TensorView<NDIM+1> auto& a_node,
    const concepts::TensorView<NDIM+1> auto& b_node,
    const concepts::TensorView<NDIM+1> auto& s_in,
    concepts::TensorView<NDIM+1> auto& result,
    const concepts::TensorViewArray<NDIM+1, Key<NDIM>::num_children()> auto& out_arr,
    const concepts::TensorView<2> auto& hg,
    const concepts::TensorView<2> auto& phibar,
    const concepts::TensorView<2> auto& phi,
    const concepts::TensorView<1> auto& quad_x,
    const detail::MulCompressedWork<NDIM>* work,
    size_type NA,
    bool square,
    size_type n_items,
    size_type K,
    T* tmp,
    ttg::device::Stream stream)
  {
    Dim3 thread_dims = max_thread_dims(2*K);
    auto smem_size = mTxmq_shmem_size<T>(2*K);
    CALL_KERNEL(detail::multiply_compressed_kernel, n_items, thread_dims, smem_size, stream,
      (D, key, a_node, b_node, s_in, result, out_arr, hg, phibar, phi, quad_x, work, NA, square, n_items, K, tmp));
    checkSubmit();
  }

#ifndef MRA_ENABLE_HOST
  /**
   * Batching support for make_multiply_compressed/make_square_compressed, used by
   * ttg::device::coop()/TT::set_batch_matcher(). The operator data (D, hg, phibar,
   * phi, quad_x, K) and whether the TT squares are the same for every node of the
   * TT, so batching is unrestricted; the views, work items and scratch of each node
   * travel per member. The combined launch is flattened over the sum of each member's
   * work items.
   */
  namespace detail {

    template <typename T, Dimension NDIM>
    using MulCompressedBatchArg = std::tuple<
      Key<NDIM>,                                                          // key
      SparseTensorView<T, NDIM+1>,                                        // a_node
      SparseTensorView<T, NDIM+1>,                                        // b_node
      SparseTensorView<T, NDIM+1>,                                        // s_in
      SparseTensorView<T, NDIM+1>,                                        // result
      std::array<SparseTensorView<T, NDIM+1>, Key<NDIM>::num_children()>, // out_arr
      const MulCompressedWork<NDIM>*,                                     // work items
      size_type,                                                          // NA
      T*                                                                  // tmp: indexed by local position
    >;

    struct MulCompressedBatchArgIdx {
      static constexpr std::size_t key = 0, a_node = 1, b_node = 2, s_in = 3, result = 4, out_arr = 5,
                                   work = 6, NA = 7, tmp = 8;
    };

    template <typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void multiply_compressed_kernel_batched(
      MulCompressedBatchArg<T, NDIM>* args, // device ptr, size == num_members
      const size_type* member_offsets,      // device ptr, size == num_members+1
      size_type num_members,
      size_type total_items,
      const Domain<NDIM>& D,
      const concepts::TensorView<2> auto hg,
      const concepts::TensorView<2> auto phibar,
      const concepts::TensorView<2> auto phi,
      const concepts::TensorView<1> auto quad_x,
      bool square,
      size_type K)
    {
      using idx = MulCompressedBatchArgIdx;
      SHARED size_type member;
      SHARED size_type local_pos;
      for (size_type pos = blockIdx.x; pos < total_items; pos += gridDim.x) {
        if (is_team_lead()) {
          member = find_member_for_pos(member_offsets, num_members, pos, &local_pos);
        }
        SYNCTHREADS();
        auto& arg = args[member];
        multiply_compressed_process_one<T, NDIM>(D, std::get<idx::key>(arg), std::get<idx::a_node>(arg),
                                                 std::get<idx::b_node>(arg), std::get<idx::s_in>(arg),
                                                 std::get<idx::result>(arg), std::get<idx::out_arr>(arg),
                                                 hg, phibar, phi, quad_x, std::get<idx::work>(arg),
                                                 std::get<idx::NA>(arg), square, K, std::get<idx::tmp>(arg),
                                                 local_pos);
      }
    }

    /**
     * Called by every member of a batch with the batch_view returned by its
     * `co_await ttg::device::coop<Key<NDIM>>(key, a_view, b_view, s_view, result_view,
     * out_views, work, NA, tmp, result, out, n_items)` (result and out the nodes, for
     * their sparsity); the leader marshals every member and submits one combined
     * launch. The sparsity of the result and of every allocated child node of each
     * member is copied to its device buffer as part of the single H2D copy of the
     * batch. max_functions bounds the number of functions of any of those nodes.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_multiply_compressed_batch_leader(
      BatchView& batch,
      GroupedBatchPoolRegistry<MulCompressedBatchArg<T, NDIM>>& registry,
      const Domain<NDIM>& D,
      const concepts::TensorView<2> auto& hg,
      const concepts::TensorView<2> auto& phibar,
      const concepts::TensorView<2> auto& phi,
      const concepts::TensorView<1> auto& quad_x,
      bool square,
      size_type K,
      size_type max_functions)
    {
      using arg_t = MulCompressedBatchArg<T, NDIM>;
      constexpr size_type num_children = Key<NDIM>::num_children();
      if (!batch.is_leader()) return;

      const std::size_t nb = batch.size();
      const auto device = ttg::device::current_device();
      auto& pool = registry.get(device);
      const std::size_t max_batch = registry.get_max_batch_size();
      auto& slot = pool.acquire(max_batch, max_batch + 1, max_batch * (num_children + 1) * max_functions);
      slot.args.clear();
      slot.offsets.resize(nb + 1);
      slot.offsets[0] = 0;
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();

      /* the nodes whose sparsity is pushed: the result, and the children that were allocated */
      auto for_each_output = [&](std::size_t m, auto&& fn) {
        fn(batch[m].template get<9>());
        for (auto& o : batch[m].template get<10>()) if (!o.empty()) fn(o);
      };
      size_type total_sparsity_bytes = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        for_each_output(m, [&](auto& node) { total_sparsity_bytes += node.coeffs().dim(0); });
      }
      slot.sparsity.resize(total_sparsity_bytes);

      size_type sparsity_offset = 0;
      for (std::size_t m = 0; m < nb; ++m) {
        for_each_output(m, [&](auto& node) {
          const size_type n = node.coeffs().dim(0);
          sparsity_to_bytes(node.coeffs().sparsity(), reinterpret_cast<SparsityState*>(&slot.sparsity[sparsity_offset]), n);
          slot.extra_dsts.push_back(const_cast<T*>(node.coeffs().buffer().device_ptr_on(device)));
          slot.extra_srcs.push_back(&slot.sparsity[sparsity_offset]);
          slot.extra_sizes.push_back(n);
          sparsity_offset += n;
        });
        slot.args.emplace_back(batch[m].template get<0>(), batch[m].template get<1>(), batch[m].template get<2>(),
                               batch[m].template get<3>(), batch[m].template get<4>(), batch[m].template get<5>(),
                               batch[m].template get<6>(), batch[m].template get<7>(), batch[m].template get<8>());
        slot.offsets[m + 1] = slot.offsets[m] + batch[m].template get<11>();
      }
      const size_type total_items = slot.offsets[nb];

      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      Dim3 thread_dims = max_thread_dims(2*K);
      auto smem_size = mTxmq_shmem_size<T>(2*K);
      CALL_KERNEL((multiply_compressed_kernel_batched<T, NDIM>), total_items, thread_dims, smem_size, stream,
                  (slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb), total_items, D,
                   hg, phibar, phi, quad_x, square, K));
      checkSubmit();
      pool.mark_submitted(slot, stream);
    }

  } // namespace detail
#endif // !MRA_ENABLE_HOST

} // namespace mra

#endif // MRA_KERNELS_MULTIPLY_COMPRESSED_H
