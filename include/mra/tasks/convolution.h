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
                        const mra::ConvolutionOperator<T, NDIM>& op,
                        const char* name = "convolution",
                        ProcMap procmap = {},
                        DeviceMap devicemap = {}) {

    auto conv_fn = [&, N, K, name](
                    const mra::Key<NDIM>& key,
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
      mra::FunctionsCompressedNode<T, NDIM> result(key, N, K, ttg::scope::Allocate);
      result.set_ns(is_ns);
      auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K)*N, TempScope);

      std::shared_ptr<const mra::OperatorData<T, NDIM>> op_data = op.get_op(key);

      T normr = 1.0;
      T norms = 1.0;
      T fac = op_data->fac;
      for (size_type i = 0; i < NDIM; ++i) normr *= op_data->ops[i]->normR;
      for (size_type i = 0; i < NDIM; ++i) normr *= op_data->ops[i]->normR;

      auto transr = std::array{op_data->ops[0]->R.current_view(), op_data->ops[1]->R.current_view(), op_data->ops[2]->R.current_view()};
      auto transs = std::array{op_data->ops[0]->S.current_view(), op_data->ops[1]->S.current_view(), op_data->ops[2]->S.current_view()};

#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(in_node.coeffs().buffer(), result.coeffs().buffer(), tmp);
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

      auto result_view = result.coeffs().current_view();
      auto in_node_view = in_node.coeffs().current_view();

      submit_convolution_kernel<T, NDIM>(K, N, normr, norms, fac, in_node_view, result_view, transr, transs,
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

} // namespace mra

#endif // MRA_TASKS_CONVOLUTION_H
