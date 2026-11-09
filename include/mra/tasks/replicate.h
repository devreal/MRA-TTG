#ifndef MRA_TASKS_REPLICATE_H
#define MRA_TASKS_REPLICATE_H

#include <stdexcept>
#include <string>
#include <vector>

#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/tensor/functionnode.h"

/**
 * Moving the nodes of sets of functions between batches: a node is the same, at the same
 * position, in another batch (a different key, so other tasks and processes).
 */

namespace mra {

  /**
   * Send each node of a set, read-only, to the same position in the batches targets(b), b the
   * batch of the node (none: the node is dropped). TTG sends a node once to every process that
   * receives it, so each process group gets one copy, shared by its tasks. The input is only
   * read: it may come from a FunctionVector fed with keep. A device task, though it does no
   * device work, so that a node on a device is not pushed out to the host.
   *
   * targets: a callable Batch -> std::vector<Batch>.
   */
  template <typename T, Dimension NDIM, typename Targets, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_rebatch(ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> in,
                    ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> out,
                    Targets targets,
                    const std::string& name = "rebatch",
                    ProcMap procmap = {},
                    DeviceMap devicemap = {})
  {
    using node_type = FunctionsCompressedNode<T, NDIM>;
    auto fn = [targets](const Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      const std::vector<Batch> to = targets(key.batch());
      if (!to.empty()) {
        std::vector<Key<NDIM>> keys;
        keys.reserve(to.size());
        for (Batch b : to) keys.push_back(Key<NDIM>(b, key.level(), key.translation()));
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::broadcast<0>(std::move(keys), node);
#else  // MRA_ENABLE_HOST
        ttg::broadcast<0>(std::move(keys), node);
#endif // MRA_ENABLE_HOST
      }
#ifndef MRA_ENABLE_HOST
      co_return;
#endif // MRA_ENABLE_HOST
    };
    auto tt = ttg::make_tt<Space>(std::move(fn), ttg::edges(in), ttg::edges(out), name, {"input"}, {"output"});
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
    return tt;
  }

  /**
   * Replicate a set of functions held in batch 0 into every batch 0 ... num_batches - 1, for
   * functions that the functions of every batch of another set need, e.g., a potential applied
   * to all orbitals (make_multiply_compressed with a single factor). See make_rebatch.
   */
  template <typename T, Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_replicate(ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> in,
                      ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> out,
                      Batch num_batches,
                      const std::string& name = "replicate",
                      ProcMap procmap = {},
                      DeviceMap devicemap = {})
  {
    if (num_batches < 1) throw std::invalid_argument(name + ": need at least one batch");
    auto targets = [num_batches, name](Batch b) {
      if (b != 0) throw std::runtime_error(name + ": the input must be in batch 0");
      std::vector<Batch> all(num_batches);
      for (Batch t = 0; t < num_batches; ++t) all[t] = t;
      return all;
    };
    return make_rebatch<T, NDIM>(in, out, targets, name, procmap, devicemap);
  }

  /**
   * Take the functions of batch `from` of a set into batch `to` of another, the nodes of other
   * batches dropped. One instance per batch, followed by gaxpys across them, sums the functions
   * of all batches into one batch, e.g., the partial densities of the batches of the orbitals;
   * gaxpy reconciles their trees. See make_rebatch.
   */
  template <typename T, Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_take_batch(ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> in,
                       ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> out,
                       Batch from, Batch to = 0,
                       const std::string& name = "take_batch",
                       ProcMap procmap = {},
                       DeviceMap devicemap = {})
  {
    auto targets = [from, to](Batch b) { return b == from ? std::vector<Batch>{to} : std::vector<Batch>{}; };
    return make_rebatch<T, NDIM>(in, out, targets, name, procmap, devicemap);
  }

} // namespace mra

#endif // MRA_TASKS_REPLICATE_H
