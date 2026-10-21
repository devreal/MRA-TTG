#ifndef MRA_TASKS_SQUARE_H
#define MRA_TASKS_SQUARE_H

#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/functiondata.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/functionnode.h"
#include "mra/kernels/square.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Square every function of a set pointwise, in reconstructed form:
   *   out_j = f_j^2
   * at the leaves of f_j, without refinement (cf. madness::square(world, v) with
   * autorefine off). The tree of out_j is the tree of f_j; inner nodes carry no
   * coefficients, as produced by make_multiply_broadcast. Input and output are
   * reconstructed trees with a node at every key of the tree, as produced by
   * make_reconstruct.
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_square(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetT>& fns,
    const mra::FunctionData<T, NDIM>& functiondata,
    const size_type K,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const std::string& name = "square",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;

    auto square_fn = [&db, &functiondata, fns, K](const mra::Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      for (size_type j = 0; j < N; ++j) {
        if (node.is_leaf(j) && !node.empty() && node.is_nonzero(j)) sparsity.set_nonzero(j);
      }
      const size_type n_nonzero = sparsity.count_nonzero();
      node_type result = (n_nonzero > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate) : node_type(key, N);
      for (size_type j = 0; j < N; ++j) result.set_leaf(j, node.leaf_status(j));

      /* scratch must outlive the kernel: destroying a buffer releases its device memory
       * right away, while the kernel launched below is still running */
      ttg::Buffer<T, DeviceAllocator<T>> tmp;
      if (n_nonzero > 0) {
        const auto& phibar = functiondata.get_phibar();
        const auto& phi = functiondata.get_phi();
        const auto& quad_x = functiondata.get_quad_x();
        tmp = ttg::Buffer<T, DeviceAllocator<T>>(square_tmp_size<NDIM>(K)*n_nonzero, TempScope);
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(db, phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp,
                                        node.coeffs().buffer(), result.coeffs().buffer());
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto& D = *db.current_device_ptr();
        auto out_view = result.coeffs().current_view();
        auto sparseman = make_sparsity_manager(result);
        sparseman.populate_device_sparsity();
        submit_square_kernel<T, NDIM>(D, key, node.coeffs().current_view(), out_view,
                                      phibar.current_view(), phi.current_view(), quad_x.current_view(),
                                      N, n_nonzero, K, tmp.current_device_ptr(), ttg::device::current_stream());
      }
#ifndef MRA_ENABLE_HOST
      co_await ttg::device::send<0>(key, std::move(result));
#else  // MRA_ENABLE_HOST
      ttg::send<0>(key, std::move(result));
#endif // MRA_ENABLE_HOST
    };

    auto tt = ttg::make_tt<Space>(std::move(square_fn), ttg::edges(in), ttg::edges(out), name, {"in"}, {"out"});
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
    return tt;
  }

} // namespace mra

#endif // MRA_TASKS_SQUARE_H
