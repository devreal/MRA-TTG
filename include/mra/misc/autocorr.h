#ifndef MRA_AUTOCORR_H
#define MRA_AUTOCORR_H

#include "mra/misc/types.h"


namespace mra{
  namespace detail {

    /// Copies the multiwavelet autocorrelation coefficients into p which should be [2k][2k] and either double or float
    template <typename T>
    void autocorr_get(size_type K, T* p);

  } // namespace detail

}

#endif // MRA_AUTOCORR_H
