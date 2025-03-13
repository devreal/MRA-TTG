#ifndef MRA_H_INCL
#define MRA_H_INCL

/**
 * Collect all available mra headers here
 */

#include "mra/kernels.h"
#include "mra/tasks.h"

#include "mra/misc/gl.h"
#include "mra/misc/key.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/maxk.h"
#include "mra/misc/range.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/options.h"
#include "mra/misc/twoscale.h"
#include "mra/misc/platform.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/keymaps.h"

#include "mra/ops/mxm.h"
#include "mra/ops/inner.h"
#include "mra/ops/outer.h"
#include "mra/ops/functions.h"

#include "mra/tensor/tensor.h"
#include "mra/tensor/cycledim.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/tensoriter.h"
#include "mra/tensor/child_slice.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/functionnorm.h"

#include "mra/functors/gaussian.h"
#include "mra/functors/functionfunctor.h"

#endif // MRA_H_INCL
