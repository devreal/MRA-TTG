#ifndef MRA_TASKS_MULTIPLY_H
#define MRA_TASKS_MULTIPLY_H

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
  template<typename T, mra::Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_multiply(ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in1,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in2,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> out,
                const mra::FunctionData<T, NDIM>& functiondata,
                const ttg::Buffer<mra::Domain<NDIM>>& db, const size_t N, const size_t K,
                const char* name = "multiply",
                ProcMap procmap = {},
                DeviceMap devicemap = {})
  {
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> S1, S2; // to balance trees

    auto func = [&, N, K](
              const mra::Key<NDIM>& key,
              const mra::FunctionsReconstructedNode<T, NDIM>& t1,
              const mra::FunctionsReconstructedNode<T, NDIM>& t2) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      auto send_out = [&]<typename S>(S&& out){
        sends.push_back(ttg::device::send<0>(key, std::forward<S>(out)));
      };
#else
      auto send_out = [&]<typename S>(S&& out){
        ttg::send<0>(key, std::forward<S>(out));
      };
#endif

      if (t1.empty() || t2.empty()) {
        /* send out an empty result */
        auto out = mra::FunctionsReconstructedNode<T, NDIM>(key, N);
        mra::apply_leaf_info(out, t1, t2);
        send_out(std::move(out));
      } else {
        auto out = mra::FunctionsReconstructedNode<T, NDIM>(key, N, K, ttg::scope::Allocate);
        mra::apply_leaf_info(out, t1, t2);
        const auto& phibar = functiondata.get_phibar();
        const auto& phiT = functiondata.get_phiT();
        const std::size_t tmp_size = multiply_tmp_size<NDIM>(K)*N;
        ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch(tmp_size, TempScope);
        auto norms = FunctionNorms("multiply", t1, t2, out);

  #ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(out.coeffs().buffer(), phibar.buffer(), phiT.buffer(),
                                        tmp_scratch, norms.buffer());
        // if (!t1.empty()) {
        //   input.add(t1.coeffs().buffer());
        // }
        // if (!t2.empty()) {
        //   input.add(t2.coeffs().buffer());
        // }
        // pass in phibar, phiT, gl, and tmp_scratch to select
        co_await ttg::device::select(input);
  #endif
        auto t1_view = t1.coeffs().current_view();
        auto t2_view = t2.coeffs().current_view();
        auto out_view = out.coeffs().current_view();

        auto phiT_view = phiT.current_view();
        auto phibar_view = phibar.current_view();
        auto& D = *db.current_device_ptr();
        T* tmp_device = tmp_scratch.current_device_ptr();

        submit_multiply_kernel(D, t1_view, t2_view, out_view, phiT_view, phibar_view,
                            N, K, key, tmp_device, ttg::device::current_stream());

        norms.compute();
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::wait(norms.buffer());
#endif // MRA_ENABLE_HOST
        norms.verify();


        send_out(std::move(out));
      }

#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto tt = ttg::make_tt<Space>(std::move(func),
                                  ttg::edges(ttg::fuse(S1, in1), ttg::fuse(S2, in2)),
                                  ttg::edges(out, S1, S2), name,
                                  {"in1", "in2"},
                                  {"out", "S1", "S2"});
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt.set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt.set_devicemap(devicemap);
    return tt;
  }


  /* forward a reconstructed function node to the right input of do_compress
  * this is a device task to prevent data from being pulled back to the host
  * even though it will not actually perform any computation */
  template<typename T, mra::Dimension NDIM>
  static TASKTYPE send_norms_up(const mra::Key<NDIM>& key, const mra::Tensor<T, 1>& node) {
#ifndef MRA_ENABLE_HOST
    co_await select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "send-norms-up");
#else
    select_send_up(key, node, std::make_index_sequence<mra::Key<NDIM>::num_children()>{}, "send-norms-up");
#endif
  }
} // namespace mra

#endif // MRA_TASKS_MULTIPLY_H
