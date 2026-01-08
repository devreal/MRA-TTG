#ifndef MRA_TASKS_CONVOLUTION_H
#define MRA_TASKS_CONVOLUTION_H

#include <ttg.h>
#include "mra/kernels.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/options.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/conv_mad.h"
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
                        const mra::GaussianConvolutionOperator<T, NDIM>& op,
                        const T thresh,
                        const char* name = "convolution",
                        ProcMap procmap = {},
                        DeviceMap devicemap = {}) {

    auto conv_fn = [&, N, K, thresh, name](
                    const mra::Key<NDIM>& key,
                    const mra::FunctionsCompressedNode<T, NDIM>& in_node) -> TASKTYPE {
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      auto send_out = [&]<typename S>(auto& k, S&& out){
        sends.push_back(ttg::device::send<0>(k, std::forward<S>(out)));
      };
#else
      auto send_out = [&]<typename S>(auto& k, S&& out){
        ttg::send<0>(k, std::forward<S>(out));
      };
#endif

      if (in_node.empty()) {
        send_out(key, in_node);
      }
      else {
        // for fixed distance, the 2nd arg to get_op needs to be mra::Key<NDIM>(key.level(), {0, 0, 0})
        auto in_node_view = in_node.coeffs().current_view();
        std::shared_ptr<const mra::GaussianOperatorData<T, NDIM>> op_data = op.get_op(key.level(), mra::Key<NDIM>(key.level(), {0, 0, 0}));
        T opnorm = op_data->norm * op_data->fac;
        T cnorm = normf(in_node_view);
        T tol = thresh*0.01;
        /// allocate space, and move the check to kernel so the norm is computed on device
        /// TODO: revisit when norms are available on the host
        /*
        if (opnorm < 0.01*tol || opnorm*cnorm < tol) {
          mra::FunctionsCompressedNode<T, NDIM> out(key, N);
          out.set_ns();
          send_out(key, std::move(out)); // send empty node
        }

        else {
        */
        {
          // fac is the volume of sphere
          // auto fac_thresh = [&](const T R) {
          //   return std::pow(std::numbers::pi,0.5*NDIM)*std::pow(R,NDIM)/std::tgamma(1+0.5*NDIM);
          // };
          // T fac = fac_thresh(op_data->ops[0]->Rmax);

          mra::FunctionsCompressedNode<T, NDIM> out(key, N, K, ttg::scope::Allocate);
          out.set_ns();
          // set child leaf information
          for (size_type i = 0; i < N; ++i) {
            for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
              out.set_child_leaf(i, c, in_node.is_child_leaf(i, c));
            }
          }
          T normr = 1.0;
          T norms = 1.0;
          T fac = op_data->fac;
          std::array<bool, 2> at = {true, key.level()>0}; // apply terms analogue in MADNESS
          // if (key.level() == 0) at[1] = false; // do not apply S at level 0

          auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K)*N, TempScope);
          auto out_view = out.coeffs().current_view();

          for (size_type i = 0; i < NDIM; ++i) normr *= op_data->ops[i]->Rnorm;
          for (size_type i = 0; i < NDIM; ++i) norms *= op_data->ops[i]->Snorm;

          // std::cout << "MRA:: For Key: " << key << "\n the operators being passed are \n R\n" << op_data->ops[0]->R.current_view() << "\nand S: \n" << op_data->ops[0]->S.current_view() << std::endl;

          auto transr = std::array{op_data->ops[0]->R.current_view(), op_data->ops[1]->R.current_view(), op_data->ops[2]->R.current_view()};
          auto transs = std::array{op_data->ops[0]->S.current_view(), op_data->ops[1]->S.current_view(), op_data->ops[2]->S.current_view()};

#ifndef MRA_ENABLE_HOST
          auto input = ttg::device::Input(in_node.coeffs().buffer(), out.coeffs().buffer(), tmp);
          co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

          submit_convolution_kernel<T, NDIM>(key, K, N, opnorm, normr, norms, fac, in_node_view, out_view, transr, transs, at, tol,
          tmp.current_device_ptr(), ttg::device::current_stream());

#ifndef MRA_ENABLE_HOST
          co_await ttg::device::wait(out.coeffs().buffer());
#endif // MRA_ENABLE_HOST

          send_out(key, std::move(out));
        }
      }

#ifndef MRA_ENABLE_HOST
    co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto tt = ttg::make_tt(std::move(conv_fn), ttg::edges(input), ttg::edges(result), name);
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
    return tt;
  }

} // namespace mra

#endif // MRA_TASKS_CONVOLUTION_H
