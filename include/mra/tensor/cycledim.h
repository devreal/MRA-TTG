#ifndef MRA_CYCLEDIM_H
#define MRA_CYCLEDIM_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

namespace mra{
  namespace detail{
    template<typename T, Dimension NDIM>
    SCOPE void cycledim(const TensorView<T, NDIM>& in, TensorView<T, NDIM>& out, int nshift, int start, int end){

      SHARED std::array<int, NDIM> permute;

      if (is_team_lead()) {
        // support python-style negative indexing
        if (start < 0) start += NDIM;
        if (end < 0) end += NDIM;

        // sanity checks
        assert(start >= 0 && start < NDIM);
        assert(end >= 0 && end >= start && end <= NDIM);

        int ndshift = end - start + 1;
        // fill shifts with identity
        std::iota(permute.begin(), permute.end(), 0);
        for (int i = end; i >= start; --i) {
          int j = i + nshift;
          while (j > end)   j -= ndshift;
          while (j < start) j += ndshift;
          permute[i] = j;
        }
      }
      SYNCTHREADS();
      // assign using new index positions
      foreach_idxs(in, [&](auto... idxs){
        std::array<int, NDIM> newidxs;
        std::array<int, NDIM> idxs_arr = {static_cast<int>(idxs)...};
        /* mutate the indices */
        for (int i = 0; i < NDIM; ++i) {
          newidxs[permute[i]] = idxs_arr[i];
        }
        auto do_assign = [&]<std::size_t... Is>(T val, std::index_sequence<Is...>){
          out(newidxs[Is]...) = val;
        };
        do_assign(in(idxs...), std::make_index_sequence<NDIM>{});
      });
    }
  }
}

#endif // MRA_CYCLEDIM_H
