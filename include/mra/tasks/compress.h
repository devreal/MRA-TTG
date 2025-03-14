#ifndef MRA_TASKS_COMPRESS_H
#define MRA_TASKS_COMPRESS_H

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

namespace mra
{
/// Make a composite operator that implements compression for a single function
  template <typename T, mra::Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  static auto make_compress(
    const std::size_t N,
    const std::size_t K,
    const mra::FunctionData<T, NDIM>& functiondata,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>& in,
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>& out,
    const char *name = "compress",
    ProcMap&& procmap = {},
    DeviceMap&& devicemap = {})
  {
    static_assert(NDIM == 3); // TODO: worth fixing?

    constexpr const std::size_t num_children = mra::Key<NDIM>::num_children();
    // creates the right number of edges for nodes to flow from send_leafs_up to compress
    // send_leafs_up will select the right input for compress
    auto create_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>{})...);
    };
    auto send_to_compress_edges = create_edges(std::make_index_sequence<num_children>{});
    /* append out edge to set of edges */
    auto compress_out_edges = std::tuple_cat(send_to_compress_edges, std::make_tuple(out));
    /* use the tuple variant to handle variable number of inputs while suppressing the output tuple */
    auto do_compress = [&, N, K, name](const mra::Key<NDIM>& key,
                          //const std::tuple<const FunctionsReconstructedNodeTypes&...>& input_frns
                          const mra::FunctionsReconstructedNode<T,NDIM> &in0,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in1,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in2,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in3,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in4,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in5,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in6,
                          const mra::FunctionsReconstructedNode<T,NDIM> &in7) -> TASKTYPE {
      //const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
      //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
        constexpr const auto num_children = mra::Key<NDIM>::num_children();
        constexpr const auto out_terminal_id = num_children;
        mra::FunctionsCompressedNode<T,NDIM> result(key, N); // The eventual result
        // create empty, may be reset if needed
        mra::FunctionsReconstructedNode<T, NDIM> p(key, N);

        /* check if all inputs are empty */
        bool all_empty = in0.empty() && in1.empty() && in2.empty() && in3.empty() &&
                        in4.empty() && in5.empty() && in6.empty() && in7.empty();

        if (all_empty) {
          // Collect child leaf info
          mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
          /* all data is still on the host so the coefficients are zero */
          for (std::size_t i = 0; i < N; ++i) {
            p.sum(i) = 0.0;
          }
          p.set_all_leaf(false);
          // std::cout << name << " " << key << " all empty, all children leafs " << result.is_all_child_leaf() << " ["
          //           << in0.is_all_leaf() << ", " << in1.is_all_leaf() << ", "
          //           << in2.is_all_leaf() << ", " << in3.is_all_leaf() << ", "
          //           << in4.is_all_leaf() << ", " << in5.is_all_leaf() << ", "
          //           << in6.is_all_leaf() << ", " << in7.is_all_leaf() << "] "
          //           << std::endl;
        } else {

          /* some inputs are on the device so submit a kernel */

          // allocate the result
          result = mra::FunctionsCompressedNode<T, NDIM>(key, N, K, ttg::scope::Allocate);
          auto& d = result.coeffs();
          // Collect child leaf info
          mra::apply_leaf_info(result, in0, in1, in2, in3, in4, in5, in6, in7);
          p = mra::FunctionsReconstructedNode<T, NDIM>(key, N, K, ttg::scope::Allocate);
          p.set_all_leaf(false);
          assert(p.is_all_leaf() == false);
          FunctionNorms<T, NDIM> norms(name, in0, in1, in2, in3, in4, in5, in6, in7, result);

          const std::size_t tmp_size = compress_tmp_size<NDIM>(K)*N;
          ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch(tmp_size, TempScope);
          const auto& hgT = functiondata.get_hgT();
          /* stores sumsq for each child and for result at the end of the kernel */
          auto d_sumsq = ttg::Buffer<T, DeviceAllocator<T>>(N, TempScope);

#ifndef MRA_ENABLE_HOST
          auto input = ttg::device::Input(p.coeffs().buffer(), d.buffer(), hgT.buffer(),
                                          tmp_scratch, d_sumsq);
          auto select_in = [&](const auto& in) {
            if (!in.empty()) {
              input.add(in.coeffs().buffer());
            }
          };
          select_in(in0); select_in(in1);
          select_in(in2); select_in(in3);
          select_in(in4); select_in(in5);
          select_in(in6); select_in(in7);
          input.add(norms.buffer());

          co_await ttg::device::select(input);
#endif

          /* some constness checks for the API */
          static_assert(std::is_const_v<std::remove_reference_t<decltype(in0)>>);
          static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs())>>);
          static_assert(std::is_const_v<std::remove_reference_t<decltype(in0.coeffs().buffer())>>);
          static_assert(std::is_const_v<std::remove_reference_t<std::remove_reference_t<decltype(*in0.coeffs().buffer().current_device_ptr())>>>);

          /* assemble input array and submit kernel */
          //auto input_ptrs = std::apply([](auto... ins){ return std::array{(ins.coeffs.buffer().current_device_ptr())...}; });
          auto input_views = std::array{in0.coeffs().current_view(), in1.coeffs().current_view(), in2.coeffs().current_view(), in3.coeffs().current_view(),
                                        in4.coeffs().current_view(), in5.coeffs().current_view(), in6.coeffs().current_view(), in7.coeffs().current_view()};

          auto coeffs_view = p.coeffs().current_view();
          auto rcoeffs_view = d.current_view();
          auto hgT_view = hgT.current_view();

          submit_compress_kernel(key, N, K, coeffs_view, rcoeffs_view, hgT_view,
                                tmp_scratch.current_device_ptr(), d_sumsq.current_device_ptr(), input_views,
                                ttg::device::current_stream());
          norms.compute();
          /* wait for kernel and transfer sums back */
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::wait(d_sumsq, norms.buffer());
#endif
          norms.verify();

          auto* d_sumsq_arr = d_sumsq.host_ptr();
          for (std::size_t i = 0; i < N; ++i) {
            auto sumsqs = std::array{in0.sum(i), in1.sum(i), in2.sum(i), in3.sum(i),
                                    in4.sum(i), in5.sum(i), in6.sum(i), in7.sum(i)};
            auto child_sumsq = std::reduce(sumsqs.begin(), sumsqs.end());
            p.sum(i) = d_sumsq_arr[i] + child_sumsq; // result sumsq is last element in sumsqs
            //std::cout << "compress " << key << " fn " << i << "/" << N << " d_sumsq " << d_sumsq_arr[i]
            //          << " child_sumsq " << child_sumsq << " sum " << p.sum(i) << std::endl;
          }

        }

        // Recur up
        if (key.level() > 0) {
          // will not return
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::forward(
            // select to which child of our parent we send
            //ttg::device::send<0>(key, std::move(p)),
            select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress"),
            // Send result to output tree
            ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
            select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress");
            ttg::send<out_terminal_id>(key, std::move(result));
#endif
        } else {
          for (std::size_t i = 0; i < N; ++i) {
            std::cout << "At root of compressed tree fn " << i << ": total normsq is " << p.sum(i) << std::endl;
          }
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::forward(
            // Send result to output tree
            ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
          ttg::send<out_terminal_id>(key, std::move(result));
#endif
        }
    };
    auto ttt = std::make_tuple(ttg::make_tt<Space>(&do_send_leafs_up<T,NDIM>, edges(in), send_to_compress_edges, "send_leaves_up"),
                               ttg::make_tt<Space>(std::move(do_compress), send_to_compress_edges, compress_out_edges, name));

    // set maps if provided
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      std::get<0>(ttt)->set_keymap(procmap);
      std::get<1>(ttt)->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      std::get<0>(ttt)->set_devicemap(devicemap);
      std::get<1>(ttt)->set_devicemap(devicemap);
    }

    return ttt;
  }

}

#endif // MRA_TASKS_COMPRESS_H