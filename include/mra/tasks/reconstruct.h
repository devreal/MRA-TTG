#ifndef MRA_TASKS_RECONSTRUCT_H
#define MRA_TASKS_RECONSTRUCT_H

#include <ttg.h>
#include "mra/kernels.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
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

namespace mra{
  template <typename T, mra::Dimension NDIM, typename FunctionSetT, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_reconstruct(
    const std::shared_ptr<FunctionSetT>& fns,
    const std::size_t K,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
    const char* name = "reconstruct",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> S("S");  // passes scaling functions down
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T,NDIM>> P("Primer"); // primer for root

    auto primer = [&, fns, name](const mra::Key<NDIM>& key,
                                    const mra::FunctionsCompressedNode<T, NDIM>& node) -> TASKTYPE {
      //std::cout << name << " primer " << key << std::endl;
      if (key.level() == 0) {
        /* root node: need to send an empty node as the parent to do_reconstruct */
        size_type N = fns->num_functions(key);
        auto r_empty = mra::FunctionsReconstructedNode<T,NDIM>(key, N);
        r_empty.set_all_leaf(false);
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, std::move(r_empty));
#else
        ttg::send<0>(key, std::move(r_empty));
#endif
      }
    };

    auto p = ttg::make_tt(std::move(primer), ttg::edges(in), edges(P), std::string(name) + "-primer");

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) p->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) p->set_devicemap(devicemap);

    auto do_reconstruct = [&, fns, K, name](const mra::Key<NDIM>& key,
                                            const mra::FunctionsCompressedNode<T, NDIM>& node,
                                            const mra::FunctionsReconstructedNode<T, NDIM>& from_parent) -> TASKTYPE {
      size_type N = fns->num_functions(key);
      const std::size_t tmp_size = reconstruct_tmp_size<NDIM>(K)*N;
      ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch(tmp_size, TempScope);
      const auto& hg = functiondata.get_hg();
      mra::KeyChildren<NDIM> children(key);

      // Send empty interior node to result tree
      auto r_empty = mra::FunctionsReconstructedNode<T,NDIM>(key, N);
      r_empty.set_all_leaf(false);
      //std::cout << name << " " << key << std::endl;
#ifndef MRA_ENABLE_HOST
      // forward() returns a vector that we can push into
      auto sends = ttg::device::forward(ttg::device::send<1>(key, std::move(r_empty)));
      auto do_send = [&]<std::size_t I, typename S>(auto& child, S&& node) {
            sends.push_back(ttg::device::send<I>(child, std::forward<S>(node)));
      };
#else
      ttg::send<1>(key, std::move(r_empty));
      auto do_send = []<std::size_t I, typename S>(auto& child, S&& node) {
        ttg::send<I>(child, std::forward<S>(node));
      };
#endif // MRA_ENABLE_HOST

      // array of child nodes
      std::array<mra::FunctionsReconstructedNode<T,NDIM>, mra::Key<NDIM>::num_children()> r_arr;
      for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        auto& r = r_arr[it.index()];
        r = mra::FunctionsReconstructedNode<T,NDIM>(key, N);
        // collect leaf information
        for (std::size_t i = 0; i < N; ++i) {
          r.is_leaf(i) = node.is_child_leaf(i, it.index());
        }
      }

      if (node.empty() && from_parent.empty()) {
        //std::cout << "reconstruct " << key << " node and parent empty " << std::endl;
        /* both the node and the parent are empty so we can shortcut with empty results */
        for (auto it=children.begin(); it!=children.end(); ++it) {
          const mra::Key<NDIM> child= *it;
          auto& r = r_arr[it.index()];
          if (r.is_all_leaf()) {
            do_send.template operator()<1>(child, std::move(r));
          } else {
            do_send.template operator()<0>(child, std::move(r));
          }
        }
#ifndef MRA_ENABLE_HOST
        // won't return
        co_await std::move(sends);
        assert(0);
#else  // MRA_ENABLE_HOST
        return; // we're done
#endif // MRA_ENABLE_HOST
      }

      /* once we are here we know we need to invoke the reconstruct kernel */

      /* populate the vector of r's
      * TODO: TTG/PaRSEC supports only a limited number of inputs so for higher dimensions
      *       we may have to consolidate the r's into a single buffer and pick them apart afterwards.
      *       That will require the ability to ref-count 'parent buffers'. */
      for (int i = 0; i < key.num_children(); ++i) {
        r_arr[i].allocate(K, ttg::scope::Allocate);
      }

      // compute norms
      auto norms = [&]<std::size_t... Is>(std::index_sequence<Is...>){
        return FunctionNorms(name, node, from_parent, r_arr[Is]...);
      }(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});

#ifndef MRA_ENABLE_HOST
      // helper lambda to pick apart the std::array
      auto make_inputs = [&]<std::size_t... Is>(std::index_sequence<Is...>){
        return ttg::device::Input(hg.buffer(), tmp_scratch,
                                  (r_arr[Is].coeffs().buffer())...);
      };
      auto inputs = make_inputs(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
      inputs.add(from_parent.coeffs().buffer());
      inputs.add(node.coeffs().buffer());
      inputs.add(norms.buffer());
      /* select a device */
      co_await ttg::device::select(inputs);
#endif

      // helper lambda to pick apart the std::array
      auto assemble_tensors = [&]<std::size_t... Is>(std::index_sequence<Is...>){
        return std::array{(r_arr[Is].coeffs().current_view())...};
      };
      auto r_ptrs = assemble_tensors(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
      auto node_view = node.coeffs().current_view();
      auto hg_view = hg.current_view();
      auto from_parent_view = from_parent.coeffs().current_view();
      submit_reconstruct_kernel(key, N, K, node_view, hg_view, from_parent_view,
                                r_ptrs, tmp_scratch.current_device_ptr(), ttg::device::current_stream());

#ifdef MRA_CHECK_NORMS
      norms.compute();
#ifndef MRA_ENABLE_HOST
    /* wait for norms to come back and verify */
      co_await ttg::device::wait(norms.buffer());
#endif // MRA_ENABLE_HOST
      norms.verify();
#endif // MRA_CHECK_NORMS

      for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        mra::FunctionsReconstructedNode<T,NDIM>& r = r_arr[it.index()];
        r.key() = child;
        if (r.is_all_leaf()) {
          do_send.template operator()<1>(child, std::move(r));
        } else {
          do_send.template operator()<0>(child, std::move(r));
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };


    auto s = ttg::make_tt<Space>(std::move(do_reconstruct),
                                 ttg::edges(in, ttg::fuse(S, P)), // inputs
                                 ttg::edges(S, out),              // outputs
                                 name, {"input", "s/p"}, {"s", "output"});

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) s->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) s->set_devicemap(devicemap);

    /* assemble the Reconstruct TTG */
    auto ins = std::make_tuple(s->template in<0>(), s->template in<0>());
    auto outs = std::make_tuple(s->template out<0>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(s);
    ops[1] = std::move(p);

    return ttg::make_ttg(std::move(ops), std::move(ins), std::move(outs), std::string(name) + " TTG");
  }
} // namespace mra

#endif // MRA_TASKS_RECONSTRUCT_H
