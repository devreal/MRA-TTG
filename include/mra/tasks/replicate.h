#ifndef MRA_TASKS_REPLICATE_H
#define MRA_TASKS_REPLICATE_H

#include <stdexcept>
#include <string>
#include <vector>

#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/tensor/functionnode.h"

namespace mra {

  /**
   * Replicate a set of functions held in batch 0 into every batch 0 ... num_batches - 1: the
   * node at (0, n, l) is broadcast, read-only, to (b, n, l) for all b. TTG sends a node once to
   * every process that receives it, so each batch's process group gets one copy, shared by its
   * tasks. For functions that the functions of every batch of another set need, e.g., a
   * potential applied to all orbitals (make_multiply_compressed with a single factor).
   *
   * The input is only read: it may come from a FunctionVector fed with keep. A device task,
   * though it does no device work, so that a node on a device is not pushed out to the host.
   */
  template <typename T, Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_replicate(ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> in,
                      ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> out,
                      Batch num_batches,
                      const std::string& name = "replicate",
                      ProcMap procmap = {},
                      DeviceMap devicemap = {})
  {
    using node_type = FunctionsCompressedNode<T, NDIM>;
    if (num_batches < 1) throw std::invalid_argument(name + ": need at least one batch");
    auto fn = [num_batches, name](const Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      if (key.batch() != 0) throw std::runtime_error(name + ": the input must be in batch 0");
      std::vector<Key<NDIM>> keys;
      keys.reserve(num_batches);
      for (Batch b = 0; b < num_batches; ++b) keys.push_back(Key<NDIM>(b, key.level(), key.translation()));
#ifndef MRA_ENABLE_HOST
      co_await ttg::device::broadcast<0>(std::move(keys), node);
#else  // MRA_ENABLE_HOST
      ttg::broadcast<0>(std::move(keys), node);
#endif // MRA_ENABLE_HOST
    };
    auto tt = ttg::make_tt<Space>(std::move(fn), ttg::edges(in), ttg::edges(out), name, {"input"}, {"output"});
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
    return tt;
  }

  /**
   * Take the functions of batch `from` of a set into batch `to` of another: the nodes of batch
   * `from` are sent, read-only, to the same position in batch `to`, the nodes of other batches are
   * dropped. One instance per batch, followed by gaxpys across them, sums the functions of all
   * batches into one batch, e.g., the partial densities of the batches of the orbitals; gaxpy
   * reconciles their trees. As make_replicate, a device task without device work.
   */
  template <typename T, Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_take_batch(ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> in,
                       ttg::Edge<Key<NDIM>, FunctionsCompressedNode<T, NDIM>> out,
                       Batch from, Batch to = 0,
                       const std::string& name = "take_batch",
                       ProcMap procmap = {},
                       DeviceMap devicemap = {})
  {
    using node_type = FunctionsCompressedNode<T, NDIM>;
    auto fn = [from, to](const Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      if (key.batch() == from) {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(Key<NDIM>(to, key.level(), key.translation()), node);
#else  // MRA_ENABLE_HOST
        ttg::send<0>(Key<NDIM>(to, key.level(), key.translation()), node);
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

} // namespace mra

#endif // MRA_TASKS_REPLICATE_H
