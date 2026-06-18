#ifndef MRA_TASKS_PROJECT_H
#define MRA_TASKS_PROJECT_H

#include <ttg.h>
#include "mra/kernels.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/options.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/functionset.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/functionnorm.h"
#include "mra/functors/gaussian.h"
#include "mra/functors/functionfunctor.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra{
  template<typename T, mra::Dimension NDIM, typename FunctionSetT,
           typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_project(
    const ttg::Buffer<mra::Domain<NDIM>>& db,
    const std::shared_ptr<FunctionSetT>& fns,
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
    /**
     * We need to track which functions have reached their leaf level at each position in the tree.
     */
    using LeafInfo = typename mra::DenseTensor<LeafStatus, 1>;
    ttg::Edge<mra::Key<NDIM>, LeafInfo> refine("refine");

    /**
     * Takes the control input and sends an empty LeafInfo to the root of project.
     */
    auto dispatch_fn = [fns](const Key<NDIM>& key) {
      LeafInfo leaf_info(fns->num_functions(key), ttg::scope::SyncIn);
      auto host_ptr = leaf_info.buffer().host_ptr();
      assert(host_ptr != nullptr);
      std::fill(host_ptr, host_ptr + leaf_info.size(), LeafStatus::Inner);
      ttg::send<0>(key, std::move(leaf_info));
    };
    auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn), ttg::edges(control), edges(refine), std::string(name) + "-dispatch");

    /* create a non-owning buffer for domain and capture it */
    auto fn = [&, K, max_level, thresh, gl = mra::GLbuffer<T>(), fns, name]
              (const mra::Key<NDIM>& key, const LeafInfo& leaf_info) -> TASKTYPE {
      using key_type = typename mra::Key<NDIM>;
      using node_type = typename mra::FunctionsReconstructedNode<T, NDIM>;
      using function_type = typename FunctionSetT::function_type;

      size_type N = fns->num_functions(key);
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllNonZero); // start with all non-zero, we'll remove the zero ones as we go
      node_type result(key, N); // empty for fast-paths, no need to zero out

#ifndef MRA_ENABLE_HOST
      auto outputs = ttg::device::forward();
#endif // MRA_ENABLE_HOST
      auto fn_host_view = fns->host_view(key); // force the host view to be used
      bool all_initial_level = true;
      for (std::size_t i = 0; i < N; ++i) {
        if (key.level() >= initial_level(fn_host_view[i])) {
          all_initial_level = false;
          break;
        } else {
          // above initial level, mark as zero
          sparsity.remove(i);
        }
      }
      if (all_initial_level) {
        //std::cout << "project " << key << " all initial " << std::endl;
        std::vector<mra::Key<NDIM>> bcast_keys;
        /* TODO: children() returns an iteratable object but broadcast() expects a contiguous memory range.
                  We need to fix broadcast to support any ranges */
        for (auto child : children(key)) bcast_keys.push_back(child);

#ifndef MRA_ENABLE_HOST
        outputs.push_back(ttg::device::broadcast<0>(std::move(bcast_keys), leaf_info));
#else
        ttg::broadcast<0>(std::move(bcast_keys), leaf_info);
#endif
        result.set_all_leaf(LeafStatus::Inner); // set to inner since we haven't computed coeffs yet, the kernel will update this for the children
      } else {
        bool all_negligible = true;
        bool all_leaf_or_invalid = true;
        auto trunc = mra::truncate_tol(key,thresh);
        LeafInfo result_leaf_info;
        auto leaf_info_view = leaf_info.current_view();
        for (std::size_t i = 0; i < N; ++i) {
          if (leaf_info_view[i] == LeafStatus::Leaf || leaf_info_view[i] == LeafStatus::Invalid) {
            /* if the parent is a leaf, then this must be a zero child */
            //std::cout << name << " " << key << " function " << i << " is leaf or invalid, setting to invalid" << std::endl;
            result.set_leaf(i, LeafStatus::Invalid);
            sparsity.remove(i);
            continue;
          }
          if (sparsity.is_zero(i)) {
            /* already marked as zero by the check for initial level, skip */
            continue;
          }
          bool is_negligible = mra::is_negligible<function_type,T,NDIM>(
                                      fn_host_view[i], db.host_ptr()->template bounding_box<T>(key), trunc);
          if (is_negligible) {
            // don't set the function as sparse, we still need to get to the kernel to set the leaf info correctly for the children

            // if the parent is an inner node and we are negligible we mark as leaf
            result.set_leaf(i, LeafStatus::Leaf);

            // set node as zero and don't allocate
            sparsity.remove(i);
            //std::cout << "" << name << " " << key << " function " << i << " is negligible, setting to zero" << std::endl;
          } else {
            //std::cout << "" << name << " " << key << " function " << i << " is non-negligible" << std::endl;
          }
          all_negligible &= is_negligible;
        }
        if (!all_negligible) {
          /**
           * BEGIN FCOEFFS HERE
           * TODO: figure out a way to outline this into a function or coroutine
           */
          // allocate tensor
          result.allocate(sparsity, K, ttg::scope::Allocate);
          auto& coeffs = result.coeffs();

          result_leaf_info = LeafInfo(N, ttg::scope::Allocate);

          //std::cout << name << " " << key << " all negligible " << all_negligible << " sparsity: " << sparsity << std::endl;

          // compute the norm of functions
          auto result_norms = FunctionNorms(name, result);

          /* global function data */
          const auto& phibar = functiondata.get_phibar();
          const auto& hgT = functiondata.get_hgT();

          /* temporaries */
          const std::size_t tmp_size = fcoeffs_tmp_size<NDIM>(K)*N;
          ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch(tmp_size, TempScope);

          /* TODO: cannot do this from a function, had to move it into the main task */
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::select(db, gl, fns->buffer(), coeffs.buffer(), phibar.buffer(),
                                      hgT.buffer(), tmp_scratch, result_norms.buffer(),
                                      leaf_info.buffer(), result_leaf_info.buffer());
#endif
          auto coeffs_view      = coeffs.current_view();
          auto phibar_view      = phibar.current_view();
          auto hgT_view         = hgT.current_view();
          T* tmp_device         = tmp_scratch.current_device_ptr();
          auto  fn_view         = fns->current_view(key); // the view for the functions in this batch
          auto& domain          = *db.current_device_ptr();
          auto  gldata          = gl.current_device_ptr();
          auto leaf_info_view    = leaf_info.current_view();
          auto result_leaf_info_view = result_leaf_info.current_view();

          SparsityManager sparseman(coeffs);
          sparseman.populate_device_sparsity();

          /* submit the kernel */
          submit_fcoeffs_kernel(domain, gldata, fn_view, key, K, tmp_device,
                                phibar_view, hgT_view, coeffs_view,
                                thresh, leaf_info_view, result_leaf_info_view,
                                ttg::device::current_stream());

          result_norms.compute();

          /* wait and get is_leaf back */
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::wait(result_leaf_info.buffer(), result_norms.buffer());
#endif

          result_norms.verify(); // extracts the norms and stores them in the node
          const LeafStatus* is_leafs_arr = result_leaf_info.buffer().host_ptr();
          for (std::size_t i = 0; i < N; ++i) {
            //std::cout << name << " " << key << ", function " << i << " leaf_info in " << (int)leaf_info_view[i]
            //          << " leaf_info out " << (int)is_leafs_arr[i] << std::endl;
            result.set_leaf(i, is_leafs_arr[i]);
          }
          all_leaf_or_invalid = result.is_all_leaf_or_invalid();
          /**
           * END FCOEFFS HERE
           */
        }

        //std::cout << name << " " << key << " result " << result << ", sparsity " << sparsity << std::endl;

        /**
         * Handle forced level if provided by user.
         */
        if (max_level > 0) {
          if (key.level() == max_level) {
            result.set_all_leaf(LeafStatus::Leaf);
          }
          else {
            result.set_all_leaf(LeafStatus::Inner);
          }
        }

        if (!all_leaf_or_invalid) {
          if (!result.is_any_leaf()) {
            result = node_type(key, N); // drop coeffs if none of the functions are leafs
          }
          std::vector<mra::Key<NDIM>> bcast_keys;
          for (auto child : children(key)) bcast_keys.push_back(child);
#ifndef MRA_ENABLE_HOST
          outputs.push_back(ttg::device::broadcast<0>(std::move(bcast_keys), std::move(result_leaf_info)));
#else
          ttg::broadcast<0>(bcast_keys, std::move(result_leaf_info));
#endif
        }
      }
#ifndef MRA_ENABLE_HOST
      outputs.push_back(ttg::device::send<1>(key, std::move(result))); // always produce a result
      co_await std::move(outputs);
#else
      ttg::send<1>(key, std::move(result));
#endif
    };

    auto tt = ttg::make_tt<Space>(std::move(fn), ttg::edges(refine),
                                  ttg::edges(refine,result), name);
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      tt->set_keymap(procmap);
      dispatch_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      tt->set_devicemap(devicemap);
    }

    auto ins = std::make_tuple(dispatch_tt->template in<0>());
    auto outs = std::make_tuple(tt->template out<1>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(dispatch_tt);
    ops[1] = std::move(tt);

    return make_ttg(std::move(ops), ins, outs, name);
  }
} // namespace mra

#endif // MRA_TASKS_PROJECT_H
