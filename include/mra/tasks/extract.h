#ifndef MRA_TASKS_EXTRACT_H
#define MRA_TASKS_EXTRACT_H

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

  template <Dimension NDIM, typename NodeT>
  auto make_extract(ttg::Edge<Key<NDIM>, NodeT>& in, std::map<Key<NDIM>, NodeT>& map) {
    /* TODO: need to bring together functions from different batches */
    auto func = [&map](const Key<NDIM>& key, NodeT&& node) {
      static std::mutex m;
      std::lock_guard<std::mutex> lock(m);
      map[key] = std::move(node);
    };
    auto tt = ttg::make_tt(func, ttg::edges(in), ttg::edges(), "extract", {"input"});
    // allow TTG to defer this task until all readers have completed
    // we can do this here because no other task depends on the output of this task
    tt->set_defer_writer(true);
    return tt;
  }
}

#endif // MRA_TASKS_EXTRACT_H
