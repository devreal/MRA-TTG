#ifndef MRA_CYCLEDIM_H
#define MRA_CYCLEDIM_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"

namespace mra{
  namespace detail{
    SCOPE void cycledim(const concepts::TensorView auto& in, concepts::TensorView auto& out, int nshift, int start, int end){

      using in_type = std::decay_t<decltype(in)>;
      using out_type = std::decay_t<decltype(out)>;
      static_assert(std::is_same_v<in_type, out_type>, "Input and output tensor views must have the same type.");
      using T = typename in_type::value_type;
      constexpr Dimension NDIM = in_type::ndim();
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
