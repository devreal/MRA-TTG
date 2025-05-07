#ifndef MRA_CONVOLUTIONDATA_H
#define MRA_CONVOLUTIONDATA_H

#include "mra/misc/types.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/platform.h"
#include "mra/misc/autocorr.h"


namespace mra {

  template <typename T, Dimension NDIM>
  class ConvolutionData {
    public:

      void get_rnlp(const size_type K, TensorView<T, NDIM>& rnlp) {
        autocorr_get(K, rnlp.data());
      }
  };

} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
