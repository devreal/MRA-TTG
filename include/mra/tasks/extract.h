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

  template <typename T, Dimension NDIM, typename NodeT>
  auto make_extract(ttg::Edge<Key<NDIM>, NodeT>& in, std::map<Key<NDIM>, NodeT>& map){
    auto func = [&map](const Key<NDIM>& key, NodeT&& node) {
      map[key] = std::move(node);
    };
    return ttg::make_tt(func, ttg::edges(in), ttg::edges(), "extract", {"input"});
  }
}

#endif // MRA_TASKS_EXTRACT_H