#ifndef MRA_TASKS_GAXPY_H
#define MRA_TASKS_GAXPY_H

#include <any>
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
  template<typename T, mra::Dimension NDIM>
  auto make_gaxpy(ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in1,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in2,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
                const T scalarA, const T scalarB, const size_t N, const size_t K,
                const char* name = "gaxpy")
  {
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> S1, S2; // to balance trees

    auto func = [N, K, scalarA, scalarB, name](
              const mra::Key<NDIM>& key,
              const mra::FunctionsCompressedNode<T, NDIM>& t1,
              const mra::FunctionsCompressedNode<T, NDIM>& t2) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      auto send_out = [&]<typename S>(S&& out){
        sends.push_back(ttg::device::send<0>(key, std::forward<S>(out)));
      };
#else
      auto send_out = [&]<typename S>(S&& out){
        ttg::send<0>(key, std::forward<S>(out));
      };
#endif

      //std::cout << name << " " << key << " t1 empty " << t1.empty() << " t2 empty " << t2.empty() << std::endl;

      /**
       * We can forward inputs only if the scalars are right and if the other input is empty and
       * all its children are leafs. Otherwise we need to compute the GAXPY and/or adjust the leaf information.
       */
      if ((t1.empty() && t2.empty())) {
        /* send out an empty result */
        auto out = mra::FunctionsCompressedNode<T, NDIM>(key, N); // out -> result
        mra::apply_leaf_info(out, t1, t2);
        //std::cout << name << " " << key << " t1 empty, t2 empty, all leafs " << out.is_all_child_leaf() << std::endl;
        send_out(std::move(out));
      } else if ((scalarA == 0.0 || (t1.empty() && scalarB == 1.0)) && t1.is_all_child_leaf()) {
        // just send t2, t1 is empty and all children are leafs
        send_out(t2);
      } else if ((scalarB == 0.0 || (t2.empty() && scalarA == 1.0)) && t2.is_all_child_leaf()) {
        // just send t1, t2 is empty and all children are leafs
        send_out(t1);
      } else {

        auto out = mra::FunctionsCompressedNode<T, NDIM>(key, N, K, ttg::scope::Allocate);

        /* adapt the leaf information of the result: if the children of both nodes are leafs then
        * the children of the output node are leafs as well. */
        mra::apply_leaf_info(out, t1, t2);
        //std::cout << name << " " << key << " all leafs " << out.is_all_child_leaf() << std::endl;
#if 0
        for (size_type i = 0; i < N; ++i) {
          for (auto child : children(key)) {
            auto childidx = child.childindex();
            out.is_child_leaf(i)[childidx] = t1.is_child_leaf(i)[childidx] && t2.is_child_leaf(i)[childidx];
          }
        }
#endif // 0

        auto norms = FunctionNorms("gaxpy", out, t1, t2);

#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(out.coeffs().buffer());
        if (!t1.empty()) {
          input.add(t1.coeffs().buffer());
        }
        if (!t2.empty()) {
          input.add(t2.coeffs().buffer());
        }
        input.add(norms.buffer());
        co_await ttg::device::select(input);
#endif
        auto t1_view = t1.coeffs().current_view();
        auto t2_view = t2.coeffs().current_view();
        auto out_view = out.coeffs().current_view();

        submit_gaxpy_kernel(key, t1_view, t2_view, out_view,
                            scalarA, scalarB, N, K, ttg::device::current_stream());

        norms.compute();

#ifndef MRA_ENABLE_HOST
        co_await ttg::device::wait(norms.buffer());
#endif // MRA_ENABLE_HOST

        norms.verify();

        send_out(std::move(out));
      }

      std::vector<mra::Key<NDIM>> child_keys_left, child_keys_right;
      /**
       * Check for each child whether all functions are leafs in t1 and t2.
       * For any child where all functions are leafs in only one side, we need to broadcast an empty node.
       */
      for (auto child : children(key)) {
        const auto childidx = child.childindex();
        bool t1_all_child_leaf = true, t2_all_child_leaf = true;
        for (size_type i = 0; i < N && (t1_all_child_leaf | t2_all_child_leaf); ++i) {
          if (!t1.is_child_leaf(i, childidx)) {
            t1_all_child_leaf = false;
          }
          if (!t2.is_child_leaf(i, childidx)) {
            t2_all_child_leaf = false;
          }
        }
        if (t1_all_child_leaf && !t2_all_child_leaf) {
          //std::cout << name << " " << key << " balancing tree to left " << child << std::endl;
          child_keys_left.push_back(child);
        }
        if (!t1_all_child_leaf && t2_all_child_leaf) {
          //std::cout << name << " " << key << " balancing tree to right " << child << std::endl;
          child_keys_right.push_back(child);
        }
      }
      if (child_keys_left.size() > 0) {
#ifndef MRA_ENABLE_HOST
        sends.push_back(ttg::device::broadcast<1>(
                          std::move(child_keys_left),
                          mra::FunctionsCompressedNode<T, NDIM>(key, N)));
#else
        ttg::broadcast<1>(std::move(child_keys_left),
                          mra::FunctionsCompressedNode<T, NDIM>(key, N));
#endif // MRA_ENABLE_HOST
      }
      if (child_keys_right.size() > 0) {
#ifndef MRA_ENABLE_HOST
        sends.push_back(ttg::device::broadcast<2>(
                          std::move(child_keys_right),
                          mra::FunctionsCompressedNode<T, NDIM>(key, N)));
#else
        ttg::broadcast<2>(std::move(child_keys_right),
                          mra::FunctionsCompressedNode<T, NDIM>(key, N));
#endif // MRA_ENABLE_HOST
      }

#ifndef MRA_ENABLE_HOST
       co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    return ttg::make_tt<Space>(std::move(func),
                              ttg::edges(ttg::fuse(S1, in1), ttg::fuse(S2, in2)),
                              ttg::edges(out, S1, S2), name,
                              {"in1", "in2"},
                              {"out", "S1", "S2"});
}

  template<typename T, mra::Dimension NDIM>
  auto make_gaxpy(ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in1,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
                const T scalarA,
                const size_t N,
                const size_t K,
                const char* name = "gaxpy_ovld")
  {
    auto func = [N](const mra::Key<NDIM>& key){
      return FunctionsCompressedNode<T, NDIM>(key, N);
    };

    auto rank = ttg::default_execution_context().rank();

    auto pmap = [rank](const mra::Key<NDIM>& key){
      return rank;
    };

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> dummy{"gaxpy_dummy", true,
                                                                            {std::any{}, std::move(func), std::move(pmap)}};


    return make_gaxpy(in1, dummy, out, scalarA, 0.0, N, K, name);
  }
} // namespace mra

#endif // MRA_TASKS_GAXPY_H
