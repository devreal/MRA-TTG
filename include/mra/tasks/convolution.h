#ifndef MRA_TASKS_CONVOLUTION_H
#define MRA_TASKS_CONVOLUTION_H

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

  template <typename T, Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_convolution(size_type N, size_type K,
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsNSNode<T, NDIM>> input,
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsNSNode<T, NDIM>> result,
                        const char* name = "convolution",
                        ProcMap procmap = {},
                        DeviceMap devicemap = {}) {}