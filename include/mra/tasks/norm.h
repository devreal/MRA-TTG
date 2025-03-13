#ifndef MRA_TASKS_NORM_H
#define MRA_TASKS_NORM_H

#include <ttg.h>
#include "mra/kernels.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/options.h"
#include "mra/misc/functiondata.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/functionnode.h"
#include "mra/functors/gaussian.h"
#include "mra/functors/functionfunctor.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra{
  template <typename T, Dimension NDIM>
  auto make_norm(size_type N, size_type K,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> input,
                ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, 1>> result,
                const char* name = "norm") {
    static_assert(NDIM == 3); // TODO: worth fixing?
    using norm_tensor_type = mra::Tensor<T, 1>;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> node_e;
    ttg::Edge<mra::Key<NDIM>, norm_tensor_type> norm_e0, norm_e1, norm_e2, norm_e3, norm_e4, norm_e5, norm_e6, norm_e7;
    static constexpr const int num_children = mra::Key<NDIM>::num_children();

    /**
     * Takes a tensor of norms from each child
     */
    auto norm_fn = [N, K, name](const mra::Key<NDIM>& key,
                                const norm_tensor_type& norm0,
                                const norm_tensor_type& norm1,
                                const norm_tensor_type& norm2,
                                const norm_tensor_type& norm3,
                                const norm_tensor_type& norm4,
                                const norm_tensor_type& norm5,
                                const norm_tensor_type& norm6,
                                const norm_tensor_type& norm7,
                                const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
      // TODO: pass ttg::scope::Allocate once that's possible
      // TODO: reuse of one of the input norms?
      auto norms_result = norm_tensor_type(N);
      auto fnnorms = FunctionNorms("norm", norm0, norm1, norm2, norm3, norm4, norm5, norm6, norm7);
      //std::cout << name << " " << key << std::endl;
#ifndef MRA_ENABLE_HOST
      co_await ttg::device::select(norms_result.buffer(), in.coeffs().buffer(), fnnorms.buffer(),
                                  norm0.buffer(), norm1.buffer(), norm2.buffer(), norm3.buffer(),
                                  norm4.buffer(), norm5.buffer(), norm6.buffer(), norm7.buffer());
#endif // MRA_ENABLE_HOST
      auto node_view = in.coeffs().current_view();
      auto norm_result_view = norms_result.current_view();
      std::array<const T*, mra::Key<NDIM>::num_children()> child_norms =
          {norm0.buffer().current_device_ptr(), norm1.buffer().current_device_ptr(),
          norm2.buffer().current_device_ptr(), norm3.buffer().current_device_ptr(),
          norm4.buffer().current_device_ptr(), norm5.buffer().current_device_ptr(),
          norm6.buffer().current_device_ptr(), norm7.buffer().current_device_ptr()};
      submit_norm_kernel(key, N, K, node_view, norm_result_view, child_norms, ttg::device::current_stream());

      fnnorms.compute();
#ifndef MRA_ENABLE_HOST
      co_await ttg::device::wait(fnnorms);
#endif // MRA_ENABLE_HOST
      fnnorms.verify();

#ifndef MRA_ENABLE_HOST
      if (key.level() == 0) {
        // send to result
        //std::cout << name << " " << key << " sending to result " << std::endl;
        co_await ttg::device::send<num_children>(key, std::move(norms_result));
      } else {
        // send norms upstream
        co_await select_send_up(key, std::move(norms_result), std::make_index_sequence<num_children>{}, "norm");
      }
#else
      if (key.level() == 0) {
        // send to result
        //std::cout << "norm send to result " << key << std::endl;
        ttg::send<num_children>(key, std::move(norms_result));
      } else {
        // send norms upstream
        //std::cout << "norm send up " << key << std::endl;
        select_send_up(key, std::move(norms_result), std::make_index_sequence<num_children>{}, "norm");
      }
#endif // MRA_ENABLE_HOST
    };

    auto norm_tt = ttg::make_tt<Space>(std::move(norm_fn),
                                      ttg::edges(norm_e0, norm_e1, norm_e2, norm_e3, norm_e4, norm_e5, norm_e6, norm_e7, node_e),
                                      ttg::edges(norm_e0, norm_e1, norm_e2, norm_e3, norm_e4, norm_e5, norm_e6, norm_e7, result),
                                      name);

    /**
     * Task to dispatch incoming compressed nodes and forward empty
     * nodes for children that do not exist
     */
    auto dispatch_fn = [N, K, name]
                    (const mra::Key<NDIM>& key,
                      const mra::FunctionsCompressedNode<T, NDIM>& in) -> TASKTYPE {
      //std::cout << name << "-dispatch " << key << " sending node to " << num_children << std::endl;
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(ttg::device::send<num_children>(key, in));
#else  // MRA_ENABLE_HOST
      ttg::send<num_children>(key, in);
#endif // MRA_ENABLE_HOST
      /* feed empty tensor to all */
      for (auto child : children(key)) {
        bool is_all_leaf = true;
        const auto childidx = child.childindex();
        for (size_type i = 0; i < N && is_all_leaf; ++i) {
          is_all_leaf &= in.is_child_leaf(i, childidx);
        }
        //std::cout << "norm dispatch " << key << " child " << child << " all leaf " << is_all_leaf << std::endl;
        if (is_all_leaf) {
          //std::cout << name << "-dispatch " << key << " sending empty norms to child " << childidx << " " << child << std::endl;
          // pass up a null tensor
#ifndef MRA_ENABLE_HOST
          sends.push_back(select_send_up(child, mra::Tensor<T, 1>(), std::make_index_sequence<num_children>{}, "dispatch"));
#else  // MRA_ENABLE_HOST
          select_send_up(child, mra::Tensor<T, 1>(), std::make_index_sequence<num_children>{}, "dispatch");
#endif // MRA_ENABLE_HOST
        } else {
          /* if not all children are leafs the norm task will receive norms from somewhere
          * so there is nothing to be done here */
          //std::cout << name << "-dispatch " << key << " child " << childidx << " " << child << " has not all leaf" << std::endl;
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn),
                                          ttg::edges(input),     // main input
                                          ttg::edges(norm_e0, norm_e1, norm_e2, norm_e3,
                                                      norm_e4, norm_e5, norm_e6, norm_e7, node_e),
                                          "norm-dispatch");
    /* compile everything into tasks */
    return std::make_tuple(std::move(norm_tt),
                          std::move(dispatch_tt));
  }
} // namespace mra

#endif // MRA_TASKS_NORM_H
