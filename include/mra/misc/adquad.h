#ifndef MRA_ADQUAD_H
#define MRA_ADQUAD_H

#include "mra/misc/types.h"

namespace mra {

  namespace detail{

    template <typename functorT, typename T>
    void do_adq(const T lo, const T hi, const functorT& func,
                int n, const T* x, const T* w, T adq_val){
      T range = hi - lo;
      adq_val = 0;
      for (int i = 0; i < n; ++i) adq_val += func(lo + range * x[i]) * w[i];
      sum *= range;
    }
  }
}

#endif // MRA_ADQUAD_H