#ifndef MRA_TASKS_DERIVATIVE_H
#define MRA_TASKS_DERIVATIVE_H

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
  auto make_derivative(size_type N, size_type K,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> in,
                ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> result,
                const mra::FunctionData<T, NDIM>& functiondata,
                const ttg::Buffer<mra::Domain<NDIM>>& db,
                const T g1,
                const T g2,
                const Dimension axis,
                const int bc_left,
                const int bc_right,
                const std::string& name = "derivative",
                ProcMap&& procmap = {},
                DeviceMap&& devicemap = {})
  {
    // TODO: we could generalize this to NDIM by using the tuple-based API
    static_assert(NDIM == 3, "Derivative currently only supported in 3D!");
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> left, center, right;
    // output terminal offsets
    constexpr static const int LEFT  = 0;
    constexpr static const int CENTER  = 1;
    constexpr static const int RIGHT = 2;
    constexpr static const int RESULT = RIGHT+1;

    if (axis >= NDIM) {
      throw std::runtime_error("Invalid axis for derivative");
    }

    auto dispatch_fn = [&, axis](const mra::Key<NDIM>& key,
                          const mra::FunctionsReconstructedNode<T, NDIM>& in_node) -> TASKTYPE {

      //std::cout << "derivative dispatch " << key << " axis " << axis << std::endl;
#ifndef MRA_ENABLE_HOST
      // forward() returns a vector that we can push into
      auto sends = ttg::device::forward(ttg::device::send<CENTER>(key, in_node));
      auto do_send = [&]<std::size_t I, typename S>(auto& child, S&& node) {
        sends.push_back(ttg::device::send<I>(child, std::forward<S>(node)));
      };
#else
      ttg::send<CENTER>(key, in_node);
      auto do_send = []<std::size_t I, typename S>(auto& k, S&& node) {
        ttg::send<I>(k, std::forward<S>(node));
      };
#endif // MRA_ENABLE_HOST


      bool has_right = !key.is_right_boundary(axis);
      bool has_left  = !key.is_left_boundary(axis);

      //std::cout << "derivative dispatch " << key << " axis " << D << " has_left " << has_left << " has_right " << has_right << std::endl;

      if (has_right){
        mra::Key<NDIM> right_key = key.neighbor(axis, 1);
        //std::cout << "derivative dispatch " << key << " has right, sending to right neighbor " << right_key << std::endl;
        do_send.template operator()<LEFT>(right_key, in_node);
      } else {
        //std::cout << "derivative dispatch " << key << " has no right, sending to right input " << key << std::endl;
        do_send.template operator()<RIGHT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
      }

      if (has_left){
        mra::Key<NDIM> left_key = key.neighbor(axis, -1);
        //std::cout << "derivative dispatch " << key << " has left, sending to left neighbor " << left_key << std::endl;
        do_send.template operator()<RIGHT>(left_key, in_node);
      } else {
        //std::cout << "derivative dispatch " << key << " has no left, sending to left input " << key << std::endl;
        do_send.template operator()<LEFT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
      }
    };

    auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn),
                                          ttg::edges(in),
                                          ttg::edges(left, center, right),
                                          name+"-dispatch");

    auto derivative_fn = [&, N, K, g1, g2, axis, bc_left, bc_right](
                                const mra::Key<NDIM>& key,
                                const mra::FunctionsReconstructedNode<T, NDIM>& left,
                                const mra::FunctionsReconstructedNode<T, NDIM>& center,
                                const mra::FunctionsReconstructedNode<T, NDIM>& right) -> TASKTYPE {

      /* tuple of references to inputs */
      auto inputs = std::array{std::cref(left), std::cref(center), std::cref(right)};

#ifndef MRA_ENABLE_HOST
      // forward() returns a vector that we can push into
      auto sends = ttg::device::forward();
      auto do_send = [&]<std::size_t I, typename S>(auto& child, S&& node) {
        sends.push_back(ttg::device::send<I>(child, std::forward<S>(node)));
      };
#else
      auto do_send = []<std::size_t I, typename S>(auto& k, S&& node) {
        ttg::send<I>(k, std::forward<S>(node));
      };
#endif // MRA_ENABLE_HOST

      //std::cout << "derivative " << key << std::endl;

      if (center.empty()){
        //std::cout << "derivative " << key << " center empty" << std::endl;
        /**
         * We received an empty center. If the left node is not empty, we need to refine it to the left child.
         * If the right node is not empty, we need to refine it to the right child.
         * These children will eventually receive a non-empty center node.
         */
        for (auto child : children(key)) {
          if (!left.empty()){
            if (child.is_left_child(axis)) { // skip right children
              //std::cout << "derivative " << key << " left not empty, sending to left child " << child << std::endl;
              do_send.template operator()<LEFT>(child, left);
            }
          }

          if (!right.empty()){
            if (child.is_right_child(axis)) { // skip left children
              //std::cout << "derivative " << key << " right not empty, sending to right child " << child << std::endl;
              do_send.template operator()<RIGHT>(child, right);
            }
          }
        }
        /* send an empty node as the result */
        do_send.template operator()<RESULT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
      } else { // center is not empty

        /**
         * Check if we have to refine down in all dimensions. This is necessary if one
         * of the neighbors is empty and we are not at a boundary.
         */
        bool need_refinement = (left.empty() && !key.is_left_boundary(axis)) || (right.empty() && !key.is_right_boundary(axis));

        if (need_refinement) {

          auto make_empty = []{ return mra::FunctionsReconstructedNode<T, NDIM>(); };

          /**
           * Send center to all children.
           */
          for (auto child : children(key)) {
            //std::cout << "derivative " << key << " left or right empty, sending center " << center.key() << " to child " << child << " center input" << std::endl;
            do_send.template operator()<CENTER>(child, center);

            //std::cout << "derivative " << key << " center not empty, balance axis " << D << std::endl;

            /**
             * Handle left input
             */

            /* only refine down if the left node is not empty or we are at a boundary
            * if the left node is empty the children will receive a left node from their neighbor */
            if (!left.empty() || key.is_left_boundary(axis)) {
              if (child.is_left_child(axis)) { // skip right children
                // send left (if not empty) or center to left children
                if (key.is_left_boundary(axis)) {
                  //std::cout << "derivative " << key << " center not empty, sending " << make_empty().key()
                  //          << " to left child " << child << " left input" << std::endl;
                  do_send.template operator()<LEFT>(child, make_empty());
                } else {
                  //std::cout << "derivative " << key << " center not empty, sending " << left.key()
                  //          << " to left child " << child << " left input" << std::endl;
                  do_send.template operator()<LEFT>(child, left);
                }
              }
            }

            /* Send the center node to the left inputs of right children */
            if (child.is_right_child(axis)) { // skip left children
              //std::cout << "derivative " << key << " center not empty, sending " << center.key()
              //          << " to right child " << child << " left input" << std::endl;
              do_send.template operator()<LEFT>(child, center);
            }

            /**
             * Handle right input
             */
            if (!right.empty() || key.is_right_boundary(axis)) {
              if (child.is_right_child(axis)) { // skip left children
                if (key.is_right_boundary(axis)) {
                  //std::cout << "derivative " << key << " center not empty, sending " << make_empty().key()
                  //          << " to right child " << child << " right input" << std::endl;
                  do_send.template operator()<RIGHT>(child, make_empty());
                } else {
                  //std::cout << "derivative " << key << " center not empty, sending " << right.key()
                  //          << " to right child " << child << " right input" << std::endl;
                  do_send.template operator()<RIGHT>(child, right);
                }
              }
            }

            /* Send the center node to the right inputs of left children */
            if (child.is_left_child(axis)) { // skip right children
              //std::cout << "derivative " << key << " center not empty, sending " << center.key()
              //          << " to left child " << child << " right input" << std::endl;
              do_send.template operator()<RIGHT>(child, center);
            }
          }

          /* send an empty node as result */
          do_send.template operator()<RESULT>(key, mra::FunctionsReconstructedNode<T, NDIM>());
        } else {
          /**
           * We can finally compute the derivative.
           */
          if ((!left.empty() || key.is_left_boundary(axis)) && (!right.empty() || key.is_right_boundary(axis))){
            mra::FunctionsReconstructedNode<T, NDIM> result(key, N, K);
            result.set_all_leaf(true);
            ttg::Buffer<T> tmp = ttg::Buffer<T>(derivative_tmp_size<NDIM>(K)*N);
            const Tensor<T, 2+1>& operators = functiondata.get_operators();
            const Tensor<T, 2>& phibar= functiondata.get_phibar();
            const Tensor<T, 2>& phi= functiondata.get_phi();
            const Tensor<T, 1>& quad_x = functiondata.get_quad_x();

            FunctionNorms<T, NDIM> norms(name, left, center, right);

#ifndef MRA_ENABLE_HOST
            co_await ttg::device::select(db, left.coeffs().buffer(), center.coeffs().buffer(), norms.buffer(),
                                        right.coeffs().buffer(), result.coeffs().buffer(), operators.buffer(),
                                        phibar.buffer(), phi.buffer(), quad_x.buffer(), tmp);
#endif // MRA_ENABLE_HOST

            auto& D = *db.current_device_ptr();
            auto result_view = result.coeffs().current_view();
            submit_derivative_kernel(D, key, left.key(), center.key(), right.key(), left.coeffs().current_view(),
                                    center.coeffs().current_view(), right.coeffs().current_view(), operators.current_view(),
                                    result_view, phi.current_view(), phibar.current_view(), quad_x.current_view(),
                                    tmp.current_device_ptr(), N, K, g1, g2, axis, bc_left, bc_right, ttg::device::current_stream());

            norms.compute();

#if !defined(MRA_ENABLE_HOST) && defined(MRA_CHECK_NORMS)
            co_await ttg::device::wait(norms.buffer());
#endif // !defined(MRA_ENABLE_HOST) && defined(MRA_CHECK_NORMS)

            norms.verify();

            do_send.template operator()<RESULT>(key, std::move(result));
          }
        }
#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif
      }
    };

    auto deriv_tt = ttg::make_tt<Space>(std::move(derivative_fn),
                              ttg::edges(left, center, right),
                              ttg::edges(left, center, right, result),
                              name);

    // set maps if provided
    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      deriv_tt->set_keymap(procmap);
      dispatch_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      deriv_tt->set_devicemap(devicemap);
      dispatch_tt->set_devicemap(devicemap);
    }

    return std::make_tuple(std::move(deriv_tt),
                           std::move(dispatch_tt));
  }
}   // namespace mra

#endif // MRA_TASKS_DERIVATIVE_H