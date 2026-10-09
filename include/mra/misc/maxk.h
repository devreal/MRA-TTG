#ifndef MRA_MAXK_H
#define MRA_MAXK_H

#include "mra/misc/types.h"

/* Set a reasonable upper bound for K.
 * This is used to inform the CUDA compiler of the maximum
 * size of thread blocks we're using. Can be changed by
 * setting MRA_MAX_K at compile-time.
 * Higher K are allowed but performance may be impacted
 * because we are launching at max K^2 threads per block
 * on accelerators. */
#ifndef MRA_MAX_K
#define MRA_MAX_K 10
#endif // MRA_MAX_K

#define MRA_MAX_K_SIZET ((size_type)MRA_MAX_K)

/**
 * Maximum number of quadrature points.
 * It appears that MADNESS uses a maximum of K + 11 quadrature points, with a hardcoded 11.
 */
#define MRA_MAX_NPT (MRA_MAX_K + 11)

#endif // MRA_MAXK_H
