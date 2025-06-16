#ifndef MRA_AUTOCORR_H
#define MRA_AUTOCORR_H

#include "mra/misc/types.h"
#include "mra/tensor/tensorview.h"


namespace mra{
  namespace detail {
    /// \brief Get autocorrelation coefficients for a given K
    template <typename T>
    void autocorr_get(size_type K, TensorView<T, 3>& cread);

  } // namespace detail

}

#endif // MRA_AUTOCORR_H
