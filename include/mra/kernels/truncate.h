#ifndef MRA_KERNELS_TRUNCATE_H
#define MRA_KERNELS_TRUNCATE_H

#include "mra/misc/platform.h"
#include "mra/misc/types.h"
#include "mra/misc/key.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensorview.h"

namespace mra {
  namespace detail {

    /**
     * Decides, for each function, whether the node's own coefficients can be
     * dropped: they can be dropped only if none of the children still hold
     * coefficients for that function (child_kept[c][blockid] == 0 for all c)
     * and the node's own norm is below tol. Nodes at level 0/1 are never
     * dropped. The node's sparsity is updated in place; `kept` receives the
     * post-decision keep/drop flag for each function so it can be forwarded
     * to the parent.
     */
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
      constexpr auto num_children = Key<NDIM>::num_children();
      SHARED DenseTensorView<T, NDIM> n;
      for (size_type blockid = blockIdx.x; blockid < N; blockid += gridDim.x) {
        if (node.is_zero(blockid)) {
          if (is_team_lead()) {
            kept[blockid] = T(0);
          }
          continue; // skip zero-function entries
        } else if (is_team_lead()) {
          n = node(blockid);
        }
        SYNCTHREADS();
        bool keep;
        assert(!node.is_zero(blockid) &&
               "blockid must have node non-zero -- find_nth_nonzero only returns non-zero ids");
        {
          bool any_child_nonzero = false;
          for (size_type c = 0; c < num_children; ++c) {
            if (child_kept[c] != nullptr && child_kept[c][blockid] != T(0.0)) {
              any_child_nonzero = true;
            }
          }
          if (any_child_nonzero || key.level() <= 1) {
            keep = true;
          } else {
            if (is_team_lead()) {
              n = node(blockid);
            }
            SYNCTHREADS();
            keep = !(normf(n) < tol);
          }
        }
        if (is_team_lead()) {
          if (!keep) {
            node.set_zero(blockid);
          }
          kept[blockid] = keep ? T(1.0) : T(0.0);
        }
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

} // namespace mra

#endif // MRA_KERNELS_TRUNCATE_H
