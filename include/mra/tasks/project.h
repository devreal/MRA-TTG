#ifndef MRA_TASKS_PROJECT_H
#define MRA_TASKS_PROJECT_H

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
#include "mra/tensor/functionnorm.h"
#include "mra/functors/gaussian.h"
#include "mra/functors/functionfunctor.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra{
  template<typename FnT, typename T, mra::Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_project(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const ttg::Buffer<FnT>& fb,
    std::size_t N,
    std::size_t K,
    int max_level,
    const mra::FunctionData<T, NDIM>& functiondata,
    const T thresh, /// should be scalar value not complex
    ttg::Edge<mra::Key<NDIM>, void> control,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> result,
    const char *name = "project",
    ProcMap procmap = {},
    DeviceMap devicemap = {})
  {
    /* create a non-owning buffer for domain and capture it */
    auto fn = [&, N, K, max_level, thresh, gl = mra::GLbuffer<T>()]
              (const mra::Key<NDIM>& key) -> TASKTYPE {
      using tensor_type = typename mra::Tensor<T, NDIM+1>;
      using key_type = typename mra::Key<NDIM>;
      using node_type = typename mra::FunctionsReconstructedNode<T, NDIM>;
      node_type result(key, N); // empty for fast-paths, no need to zero out
#ifndef MRA_ENABLE_HOST
      auto outputs = ttg::device::forward();
#endif // MRA_ENABLE_HOST
      auto* fn_arr = fb.host_ptr();
      bool all_initial_level = true;
      for (std::size_t i = 0; i < N; ++i) {
        if (key.level() >= initial_level(fn_arr[i])) {
          all_initial_level = false;
          break;
        }
      }
      if (all_initial_level) {
        //std::cout << "project " << key << " all initial " << std::endl;
        std::vector<mra::Key<NDIM>> bcast_keys;
        /* TODO: children() returns an iteratable object but broadcast() expects a contiguous memory range.
                  We need to fix broadcast to support any ranges */
        for (auto child : children(key)) bcast_keys.push_back(child);

#ifndef MRA_ENABLE_HOST
        outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
        ttg::broadcastk<0>(std::move(bcast_keys));
#endif
        result.set_all_leaf(false);
      } else {
        bool all_negligible = true;
        for (std::size_t i = 0; i < N; ++i) {
          all_negligible &= mra::is_negligible<FnT,T,NDIM>(
                                      fn_arr[i], db.host_ptr()->template bounding_box<T>(key),
                                      mra::truncate_tol(key,thresh));
        }
        //std::cout << "project " << key << " all negligible " << all_negligible << std::endl;
        if (all_negligible) {
          result.set_all_leaf(true);
        } else {
          /* here we actually compute: first select a device */
          //result.is_leaf = fcoeffs(f, functiondata, key, thresh, coeffs);
          /**
           * BEGIN FCOEFFS HERE
           * TODO: figure out a way to outline this into a function or coroutine
           */
          // allocate tensor
          result = node_type(key, N, K, ttg::scope::Allocate);
          tensor_type& coeffs = result.coeffs();

          // compute the norm of functions
          auto result_norms = FunctionNorms(name, result);

          /* global function data */
          // TODO: need to make our own FunctionData with dynamic K
          const auto& phibar = functiondata.get_phibar();
          const auto& hgT = functiondata.get_hgT();

          /* temporaries */
          const std::size_t tmp_size = fcoeffs_tmp_size<NDIM>(K)*N;
          ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch(tmp_size, TempScope);
          auto is_leafs = ttg::Buffer<bool, DeviceAllocator<bool>>(N, TempScope);

          /* TODO: cannot do this from a function, had to move it into the main task */
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::select(db, gl, fb, coeffs.buffer(), phibar.buffer(),
                                      hgT.buffer(), tmp_scratch, is_leafs, result_norms.buffer());
#endif
          auto coeffs_view = coeffs.current_view();
          auto phibar_view = phibar.current_view();
          auto hgT_view    = hgT.current_view();
          T* tmp_device = tmp_scratch.current_device_ptr();
          bool *is_leafs_device = is_leafs.current_device_ptr();
          auto *f_ptr   = fb.current_device_ptr();
          auto& domain = *db.current_device_ptr();
          auto  gldata = gl.current_device_ptr();

          /* submit the kernel */
          submit_fcoeffs_kernel(domain, gldata, f_ptr, key, N, K, tmp_device,
                                phibar_view, hgT_view, coeffs_view,
                                is_leafs_device, thresh, ttg::device::current_stream());

          result_norms.compute();

          /* wait and get is_leaf back */
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::wait(is_leafs, result_norms.buffer());
#endif

          result_norms.verify(); // extracts the norms and stores them in the node
          const bool* is_leafs_arr = is_leafs.host_ptr();
          for (std::size_t i = 0; i < N; ++i) {
            result.is_leaf(i) = is_leafs_arr[i];
          }
          /**
           * END FCOEFFS HERE
           */
        }

        if (max_level > 0){
          if (!all_initial_level && result.key().level() < max_level) { // && pass in max_level
            std::vector<mra::Key<NDIM>> bcast_keys;
            for (auto child : children(key)) bcast_keys.push_back(child);
#ifndef MRA_ENABLE_HOST
            outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
            ttg::broadcastk<0>(bcast_keys);
#endif
          }
          if (key.level() == max_level) {
            result.set_all_leaf(true);
          }
          else {
            result.set_all_leaf(false);
          }
        }
        else {
          if (!result.is_all_leaf()) {
            std::vector<mra::Key<NDIM>> bcast_keys;
            for (auto child : children(key)) bcast_keys.push_back(child);
#ifndef MRA_ENABLE_HOST
            outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
            ttg::broadcastk<0>(bcast_keys);
#endif
          }
        }
      }
#ifndef MRA_ENABLE_HOST
      outputs.push_back(ttg::device::send<1>(key, std::move(result))); // always produce a result
      co_await std::move(outputs);
#else
      ttg::send<1>(key, std::move(result));
#endif
    };

    ttg::Edge<mra::Key<NDIM>, void> refine("refine");
    auto tt = ttg::make_tt<Space>(std::move(fn), ttg::edges(ttg::fuse(control, refine)), ttg::edges(refine,result), name);
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
    return tt;
  }
} // namespace mra

#endif // MRA_TASKS_PROJECT_H
