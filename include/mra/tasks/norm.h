#ifndef MRA_TASKS_NORM_H
#define MRA_TASKS_NORM_H

#include <cmath>
#include <vector>
#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/batch_size.h"
#include "mra/misc/domain.h"
#include "mra/misc/options.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/functionset.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/functionnorm.h"
#include "mra/functors/gaussian.h"
#include "mra/functors/functionfunctor.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>
#include <ttg/serialization/std/vector.h>

namespace mra{
  /**
   * Compute the norms of a set of functions in compressed form. The squared norms
   * of every node are computed independently (and in parallel) by the compute task
   * and summed up the tree by a separate, host-side reduction task, which takes the
   * square root at the root and sends the result out.
   *
   * Partial norms travel as std::vector<T> rather than a norm tensor, so that only
   * the compute task holds (pinned) tensor memory, and only for the kernel's output.
   * An empty vector stands for all-zero norms.
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_norm(const std::shared_ptr<FunctionSetT>& fns,
                 size_type K,
                 ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> input,
                 ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 1>> result,
                 const char* name = "norm",
                 ProcMap procmap = {},
                 DeviceMap devicemap = {}) {
    using norm_tensor_type = mra::DenseTensor<T, 1>;
    using norm_vector_type = std::vector<T>;
    static constexpr const int num_children = mra::Key<NDIM>::num_children();
    /* squared norms into the reduction at a key: one from the node itself, one per child */
    ttg::Edge<mra::Key<NDIM>, norm_vector_type> norm_e(std::string(name) + "-norms");

    /* batching is controlled process-wide (mra/misc/batch_size.h), read once here;
     * there is no operator data, so any two tasks may be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::NormBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::NormBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    /**
     * Computes the squared norms of the functions at this node and sends them to the
     * reduction at the same key. For every child that does not exist (all functions
     * have a leaf there), an empty vector is sent in its place.
     */
    auto compute_fn = [fns, K, name, enable_batching, batch_pool](
                        const mra::Key<NDIM>& key,
                        const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      const size_type n_nonzero = in.empty() ? 0 : in.coeffs().sparsity().count_nonzero();

      norm_vector_type norms;
      if (n_nonzero > 0) {
        /* the kernel writes every function's entry, no need to sync the tensor in */
#ifndef MRA_ENABLE_HOST
        auto norms_result = norm_tensor_type(N, ttg::scope::Allocate);
#else  // MRA_ENABLE_HOST
        auto norms_result = norm_tensor_type(N);
#endif // MRA_ENABLE_HOST
        auto fnnorms = FunctionNorms(name, in);
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::select(norms_result.buffer(), in.coeffs().buffer(), fnnorms.buffer());
#endif // MRA_ENABLE_HOST
        auto node_view = in.coeffs().current_view();
        auto norm_result_view = norms_result.current_view();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          T* norm_result_ptr = norm_result_view.data();
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(node_view, norm_result_ptr, N);
          detail::submit_norm_batch_leader<T, NDIM>(batch, *batch_pool, K);
        } else
#endif // MRA_ENABLE_HOST
        {
          submit_norm_kernel<T, NDIM>(key, N, K, node_view, norm_result_view, ttg::device::current_stream());
        }

        fnnorms.compute();
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::wait(norms_result.buffer(), fnnorms.buffer());
#endif // MRA_ENABLE_HOST
        fnnorms.verify();

        /* the tensor only carries the norms out of the kernel */
        const T* host_norms = norms_result.buffer().host_ptr();
        norms.assign(host_norms, host_norms + N);
      }

      std::vector<mra::Key<NDIM>> missing; // keys at which to stand in for a missing child
      for (auto child : children(key)) {
        if (in.is_all_child_leaf(child)) missing.push_back(key);
      }
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(ttg::device::send<0>(key, std::move(norms)));
      if (!missing.empty()) sends.push_back(ttg::device::broadcast<0>(std::move(missing), norm_vector_type{}));
      co_await std::move(sends);
#else  // MRA_ENABLE_HOST
      ttg::send<0>(key, std::move(norms));
      for (auto& k : missing) ttg::send<0>(k, norm_vector_type{});
#endif // MRA_ENABLE_HOST
    };

    auto compute_tt = ttg::make_tt<Space>(std::move(compute_fn), ttg::edges(input), ttg::edges(norm_e),
                                          std::string(name) + "-compute", {"input"}, {"norms"});

    /**
     * Receives the sum of the squared norms of the node and its children (see the
     * input reducer below) and sends it up to the parent, or, at the root, sends the
     * norms out.
     */
    auto reduce_fn = [fns](const mra::Key<NDIM>& key, norm_vector_type&& normsq) {
      if (key.level() == 0) {
        const size_type N = fns->num_functions(key);
        auto norms = norm_tensor_type(N);
        T* ptr = norms.buffer().host_ptr();
        for (size_type j = 0; j < N; ++j) {
          ptr[j] = normsq.empty() ? T(0) : std::sqrt(normsq[j]);
        }
        ttg::send<1>(key, std::move(norms));
      } else {
        ttg::send<0>(key.parent(), std::move(normsq));
      }
    };

    auto reduce_tt = ttg::make_tt(std::move(reduce_fn), ttg::edges(norm_e), ttg::edges(norm_e, result),
                                  std::string(name) + "-reduce", {"norms"}, {"parent", "result"});
    reduce_tt->template set_input_reducer<0>([](norm_vector_type& a, const norm_vector_type& b) {
      if (b.empty()) return;
      if (a.empty()) {
        a = b;
        return;
      }
      assert(a.size() == b.size());
      for (std::size_t j = 0; j < a.size(); ++j) a[j] += b[j];
    }, 1 + num_children);

    // set maps if provided
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      compute_tt->set_keymap(procmap);
      reduce_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      compute_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      compute_tt->set_batch_matcher([](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; }, max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(compute_tt->template in<0>());
    auto outs = std::make_tuple(reduce_tt->template out<1>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(compute_tt);
    ops[1] = std::move(reduce_tt);

    return make_ttg(std::move(ops), ins, outs, name);
  }
} // namespace mra

#endif // MRA_TASKS_NORM_H
