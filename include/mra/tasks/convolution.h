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
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> input,
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> result,
                        const mra::OperatorData<T, NDIM>& op,
                        const char* name = "convolution",
                        ProcMap procmap = {},
                        DeviceMap devicemap = {}) {

    auto conv_fn = [&, N, K, op, name](const mra::Key<NDIM>& key,
                          const mra::FunctionsCompressedNode<T, NDIM>& in_node) -> TASKTYPE {

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

      bool is_ns = true;
      mra::FunctionsCompresssedNode<T, NDIM> result(key, N, K, ttg::scope::Allocate);
      result.set_ns(is_ns);
      auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K)*N, TempScope);

      T normr = 1.0, norms = 1.0;
      for (size_type i = 0; i < NDIM; ++i) normr *= op->ops[i]->normR;
      for (size_type i = 0; i < NDIM; ++i) normr *= op->ops[i]->normR;

      std::array<TensorView<T, 2>, NDIM> transr;
      for (size_type d = 0; d < NDIM; ++d){
        transr[d].current_view() = op->ops[d]->R;
      }
      std::array<TensorView<T, 2>, NDIM> transs;

      for (size_type d = 0; d < NDIM; ++d){
        transs[d].current_view() = op->ops[d]->S;
      }

#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(in_node.coeffs().buffer(), result.coeffs().buffer(), tmp);
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

      auto result_view = result.coeffs().current_view();
      submit_convolution_kernel<T, NDIM>(K, normr, norms, in_node.coeffs.current_view(), result_view, transr, transs,
        tmp.current_device_ptr(), ttg::device::current_stream());

#ifndef MRA_ENABLE_HOST
      co_await ttg::device::wait(result.coeffs().buffer());
#endif // MRA_ENABLE_HOST

      send_out(std::move(result));

#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto tt = ttg::make_tt(conv_fn, ttg::edges(input), ttg::edges(result), name);
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
    return tt;
  }