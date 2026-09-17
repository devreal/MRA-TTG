#ifndef MRA_TASKS_COMMON_H
#define MRA_TASKS_COMMON_H

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

  /**
   * Make a start task that sends the keys of the batches to be projected
   * to the control edge.
   * This should be invoked only by Process 0 after the `ttg::execute()` was called
   * and will take care of all batches in the FunctionSet.
   *
   * Example:
   *
   * ```
   *   auto gaussians = ttg::make_functionset<Gaussian<T, NDIM>, NDIM>(pmap.batch_manager());
   *   auto start = make_start(gaussians, project_control);
   *   ttg::execute();
   *   if (ttg::default_execution_context().rank() == 0) {
   *     start->invoke();
   *   }
   * ```
   */
  template <mra::Dimension NDIM, typename Functions>
  auto make_start(std::shared_ptr<Functions>& functions,
                  ttg::Edge<mra::Key<NDIM>, void>& ctl) {
    auto func = [functions]() {
      for (Batch batch = 0; batch < functions->num_batches(); ++batch) {
        ttg::sendk<0>(mra::Key<NDIM>(batch, 0));
      }
    };
    return ttg::make_tt<void>(func, ttg::edges(), edges(ctl), "start", {}, {"control"});
  }

  template <typename keyT, typename valueT>
  auto make_printer(const ttg::Edge<keyT, valueT>& in, const char* str = "", const bool doprint=true) {
    auto func = [str,doprint](const keyT& key, const valueT& value) -> TASKTYPE {
      if (doprint) {
        static std::mutex printer_guard;
#ifndef MRA_ENABLE_HOST
        /* pull the data back to the host */
        co_await ttg::device::select(value.coeffs().buffer());
        co_await ttg::device::wait(value.coeffs().buffer());
#endif // MRA_ENABLE_HOST

        // sanity check
        assert(value.coeffs().buffer().is_current_on(ttg::device::Device()));
        std::lock_guard<std::mutex> obolus(printer_guard);
        std::cout << str << " (" << key << "," << value << ")" << std::endl;
      }
    };
    auto tt = ttg::make_tt<Space>(func, ttg::edges(in), ttg::edges(), "printer", {"input"});

    // always execute on the rank that asks
    tt->set_keymap([&](const keyT&){ return ttg::default_execution_context().rank(); });
    return tt;
  }

  /* forward a reconstructed function node to the right input of do_compress
  * this is a device task to prevent data from being pulled back to the host
  * even though it will not actually perform any computation */
  template<typename T, mra::Dimension NDIM>
  static TASKTYPE do_send_leafs_up(const mra::Key<NDIM>& key, const mra::FunctionsReconstructedNode<T, NDIM>& node) {
    /* drop all inputs from nodes that are not leafs, they will be upstreamed by compress */
    if (!node.any_have_children()) {
#ifndef MRA_ENABLE_HOST
    co_await select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "do_send_leafs_up");
#else
    select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "do_send_leafs_up");
#endif
    }
  }

  /**
   * Companion to do_send_leafs_up used by make_compress's optional in-line
   * truncation support: forwards an empty "kept" placeholder up for the same
   * leaf nodes do_send_leafs_up forwards `p` for. A true tree leaf has no
   * wavelet coefficients of its own, so there is nothing to have "kept" --
   * an empty tensor is the same "nothing here" placeholder
   * make_truncate's own dispatch_fn synthesizes for structurally-absent
   * children (see tasks/truncate.h).
   */
  template<typename T, mra::Dimension NDIM>
  static TASKTYPE do_send_leafs_kept_up(const mra::Key<NDIM>& key, const mra::FunctionsReconstructedNode<T, NDIM>& node) {
    if (!node.any_have_children()) {
      mra::DenseTensor<T, 1> kept; // empty placeholder
#ifndef MRA_ENABLE_HOST
      co_await select_send_up(key, std::move(kept), std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "do_send_leafs_kept_up");
#else
      select_send_up(key, std::move(kept), std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "do_send_leafs_kept_up");
#endif
    }
  }

  /**
   * `Base` shifts which output terminal this ends up sending to (terminal
   * `Base + childindex()`), so a task with several same-shaped "send up to
   * parent" terminal groups (e.g. compress's optional `p` and `kept` groups)
   * can reuse this same helper for each group instead of hand-writing the
   * childindex dispatch per group. Defaulted to 0 so existing call sites are
   * unaffected.
   */
  template<std::size_t Base = 0, mra::Dimension NDIM, typename Value, std::size_t I, std::size_t... Is>
  static auto select_send_up(const mra::Key<NDIM>& key, Value&& value,
                            std::index_sequence<I, Is...>, const char *name = "select_send_up") {
    if (key.childindex() == I) {
      //std::cout << name << "-select_send_up " << key << " sending to " << key.parent() << " on " << (Base+I) << std::endl;
#ifndef MRA_ENABLE_HOST
      return ttg::device::send<Base + I>(key.parent(), std::forward<Value>(value));
#else
      return ttg::send<Base + I>(key.parent(), std::forward<Value>(value));
#endif
    } else if constexpr (sizeof...(Is) > 0){
      return select_send_up<Base>(key, std::forward<Value>(value), std::index_sequence<Is...>{}, name);
    }
    /* if we get here we messed up */
    throw std::runtime_error("Mismatching number of children!");
  }

} // namespace mra

#endif // MRA_TASKS_COMMON_H
