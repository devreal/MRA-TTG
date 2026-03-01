#ifndef MRA_TASKS_CONVOLUTION_H
#define MRA_TASKS_CONVOLUTION_H

#include <iostream>
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

namespace mra {


  namespace detail {
    template<Dimension NDIM>
    struct KeyPair {
      Key<NDIM> source;
      Key<NDIM> dest;

      auto operator<=>(const KeyPair&) const = default;
      auto hash() const {
        return source.hash() + dest.hash(); // TODO: make this better
      }
    };

    template<Dimension NDIM>
    std::ostream& operator<<(std::ostream& os, const KeyPair<NDIM>& kp) {
      return os << "(" << kp.source << "->" << kp.dest << ")";
    }


    /**
     * Struct that carries the leaf status for each child of a node.
     * We ignore the per-function info here, only flagging it per child.
     * This is sent from the down task to the leaf-adjust task, which receives the node
     * from the shell0 task (or the down task if no prior node exists).
     * Once the info is adjusted, the node is forwarded to the shallN tasks.
     */
    template<Dimension NDIM>
    struct ChildLeafInfo : public ttg::TTValue<ChildLeafInfo<NDIM>> {
      std::array<bool, Key<NDIM>::num_children()> is_child_leaf = { false };

      void set_all_child_leaf(bool value) {
        for (size_type i = 0; i < is_child_leaf.size(); ++i) {
          is_child_leaf[i] = value;
        }
      }

      template <typename Archive>
      void serialize(Archive& ar) {
        ar& this->is_child_leaf;
      }

      template <typename Archive>
      void serialize(Archive& ar, const unsigned int) {
        serialize(ar);
      }
    };
  } // namespace detail


  /**
   * Convolution entails many steps. For each node:
   *
   * 1) Apply the shell0 contribution.
   * 2) Screen the contributions based on the norms of the input node and the operator norm using the supplied threshold.
   *    Send the input node to the destination nodes and send the list contributions up to the parent.
   * 3) Receive contributions from the children and merge them. Distribute the relevant contributions to our direct
   *    neighbors (-x, +x, -y, +y, -z, +z), to ourselves, and to our parent.
   * 4) Receive contributions from neighbors, ourselves, and parent, combine them, and send them down to the appropriate children and ourselves.
   * 5) Iterate over all contributions we will receive by recursively instantiating a task using the pair of keys that
   *    describe the contribution (source and destination). This task will apply the contribution to the input node and send the result
   *    to the next contribution task in the list of contributions. After the last contribution has been applied we send the result to the output.
   *
   * TODO: we need to make sure there are tasks to receive the contributions from each neighbor. Each task needs to know whether their neighbor has
   *       children and if they do but the task has no children itself the task needs to send empty lists of contributions for to each child's input.
   *       Alternatively, we can use accumulate terminals that accumulate the contribution keys and have the parent adjust the number of accumulated
   *       values based on the number of neighbors each task has. We still need to create the children but do not have to send empty inputs anymore.
   */

  template <typename T, Dimension NDIM, typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_convolution(size_type N, size_type K,
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> input,
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> result,
                        const mra::GaussianConvolutionOperator<T, NDIM>& op,
                        const T thresh,
                        const std::string& name = "convolution",
                        ProcMap procmap = {},
                        DeviceMap devicemap = {}) {

    static_assert(NDIM == 3); // TODO: worth fixing?

    using ChildLeafInfo = detail::ChildLeafInfo<NDIM>;

    /**
     * A set of edges to communicate neighbor nodes.
     */
    std::array<ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>, 6> neighbor_edges;


    /**
     * Edges used with reducer terminals to accumulate contributions on the way and down.
     */
    ttg::Edge<mra::Key<NDIM>, std::vector<detail::KeyPair<NDIM>>> up_contribution_edge;
    ttg::Edge<mra::Key<NDIM>, std::vector<detail::KeyPair<NDIM>>> down_to_accumulate_edge;
    ttg::Edge<mra::Key<NDIM>, std::vector<detail::KeyPair<NDIM>>> down_contribution_edge;

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> to_shellN;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> shell0_to_dispatch;

    ttg::Edge<mra::Key<NDIM>, std::vector<detail::KeyPair<NDIM>>> contribution_edge; // connecting the down task to the accumulate dispatch task

    ttg::Edge<detail::KeyPair<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> screener_to_accumulate;

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> down_to_adjust_leaf_node;
    ttg::Edge<mra::Key<NDIM>, ChildLeafInfo> down_to_adjust_leaf_child_info;


    /***************************************************************************************************************************************
     * Task that receives inputs from its children, filters out the contributions we are ancestors for, and sends the rest up to the parent.
     ****************************************************************************************************************************************/
    auto up_contributions_tt = ttg::make_tt(
      [=](const mra::Key<NDIM>& key,
          std::vector<detail::KeyPair<NDIM>>&& contributions) {
        /**
         * Combine contributions from children and send them to the appropriate neighbors and parent.
         * We need to be careful to only send one message per neighbor/parent, so we will combine contributions that go to the same destination.
         */

        //ttg::trace(name + "-up", key, contributions.size());
        //std::cout << "UP " << key << " received " << contributions.size() << " contributions" << std::endl;

        if (key.level() == 0) {
          // root has no neighbors and no parent, so forward the contributions to the down task
          ttg::send<0>(key, std::move(contributions));
          std::cout << "UP " << key << " is root with " << contributions.size() << " contributions" << std::endl;

        } else {

          /**
           * Not the root.
           * Iterate over our neighbors and send the contributions they are responsible for.
           * We remove each key we have sent from the contributions vector so that at the end
           * we are left with only contributions that need to be sent up to the parent.
           */

          auto backiter = contributions.end();
          std::vector<detail::KeyPair<NDIM>> my_contributions; // contributions for myself and my children

          for (auto it = contributions.begin(); it != backiter; ++it) {
            const auto& contribution = *it;
            if (key == contribution.dest || key.is_ancestor_of(contribution.dest)) {
              my_contributions.push_back(contribution);
              // replace with last element
              --backiter;
              *it = *backiter;
              --it; // step back one so that the next iteration will not skip the element we just swapped in
            }
          }
          //std::cout << "UP " << key << " sending " << my_contributions.size() << " contributions to myself" << std::endl;
          ttg::send<0>(key, std::move(my_contributions));

          // shrink the contributions vector to include only elements we have not sent yet
          contributions.erase(backiter, contributions.end());

          // send the rest up to the parent
          //std::cout << "UP " << key << " sending " << contributions.size() << " contributions to parent " << key.parent() << std::endl;
          ttg::send<1>(key.parent(), std::move(contributions));

        }

      }, ttg::edges(up_contribution_edge), ttg::edges(down_contribution_edge, up_contribution_edge), "Up");

    /* Set the contribution reducer. On the way up, we receive from ourself and our children and send them to our ourself and 6 neighbors,
       as well as our parent.
       Some nodes (leaves, boundaries) receive fewer contributions and will have to be adjusted dynamically by the screener. */
    constexpr std::size_t num_up_contributions = 1 + Key<NDIM>::num_children(); // contributions from self and children
    up_contributions_tt->template set_input_reducer<0>([](std::vector<detail::KeyPair<NDIM>>& a, const std::vector<detail::KeyPair<NDIM>>& b){
      a.insert(a.end(), b.begin(), b.end());
    }, num_up_contributions);





    /************************************************************************************************
     * Task that receives input from the corresponding UP task and its parent and distributes the keys to
     * the task that applies contributions on itself and to the child tasks.
     *
     * TODO: need to adjust the leaf status of children here if we start sending further down!
     *       Route the result of shell0 through this task and adjust the leaf status based on
     *       whether our children receive contributions.
     ***********************************************************************************************/
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> down_recursive_edge;
    auto down_contributions_tt = ttg::make_tt<Space>(
      [=](const Key<NDIM>& key,
          std::vector<detail::KeyPair<NDIM>>&& contributions,
          const mra::FunctionsCompressedNode<T, NDIM>& node) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
        };
#endif

        /**
         * Receive contributions from neighbors, parent, and self send them down to the appropriate children and ourselves.
         */

        //std::cout << "DOWN " << key << " received " << contributions.size() << " contributions" << std::endl;

        auto backiter = contributions.end();
        // send to children
        auto filter_dest = [&]<std::size_t I>(const Key<NDIM>& dest, bool with_ancestors, std::index_sequence<I>) {
          std::vector<detail::KeyPair<NDIM>> dest_contributions;
          for (auto it = contributions.begin(); it != backiter; ++it) {
            const auto& contribution = *it;
            if (dest == contribution.dest || (with_ancestors && dest.is_ancestor_of(contribution.dest))) {
              dest_contributions.push_back(contribution);
              // replace with last element
              --backiter;
              *it = *backiter;
              --it;
            }
          }
          return dest_contributions;
        };

        auto dest_contributions = filter_dest(key, false, std::index_sequence<1>{}); // send our key pairs to the accumulate task
        auto num_contributions = dest_contributions.size();
        //if (num_contributions > 0) {
          //std::cout << "DOWN " << key << " sending " << dest_contributions.size() << " contributions to accumulate dispatch" << std::endl;
          send_out(key, std::move(dest_contributions), std::integral_constant<std::size_t, 1>{});
        //}

        contributions.erase(backiter, contributions.end());

        ChildLeafInfo child_leaf_info;
        child_leaf_info.set_all_child_leaf(true); // assume all children are leafs and adjust below

        // send to all children
        for (auto child : children(key)) {
          auto dest_contributions = filter_dest(child, true, std::index_sequence<0>{});
          int num_contributions = dest_contributions.size();
          if (dest_contributions.size() > 0 || !node.is_child_leaf(child)) {
            // we have contributions or an existing child, send down to child
            //std::cout << "DOWN " << key << " sending " << dest_contributions.size() << " contributions to dest " << child << std::endl;
            //send_out(child, std::move(dest_contributions), std::integral_constant<std::size_t, 0>{});
            ttg::send<0>(child, std::move(dest_contributions));
            child_leaf_info.is_child_leaf[child.childindex()] = false;
          }
          if (num_contributions > 0 && (node.invalid() || node.is_child_leaf(child))) {
            // if the child is a leaf we need to send an empty contribution list to satisfy the second input on the way down
            //std::cout << "DOWN " << key << " node empty or child " << child << " is leaf, sending empty node " << std::endl;
            ttg::send<0>(child, std::vector<detail::KeyPair<NDIM>>{}); // send an empty contribution list to the child since it will expect one
            ttg::send<2>(child, mra::FunctionsCompressedNode<T, NDIM>{}); // also send an empty node since the child task will expect one
            //send_out(child, std::vector<detail::KeyPair<NDIM>>{}, std::integral_constant<std::size_t, 0>{});
            //send_out(child, mra::FunctionsCompressedNode<T, NDIM>{}, std::integral_constant<std::size_t, 2>{}); // also send an empty node since the child task will expect one
            child_leaf_info.is_child_leaf[child.childindex()] = false;
          }
        }

        if (node.invalid()) {
          // send an empty node to the leaf-adjust task because it will not get once from shell0
          //std::cout << "DOWN " << key << " node is invalid, sending empty node to adjust leaf task" << std::endl;
          send_out(key, mra::FunctionsCompressedNode<T, NDIM>{}, std::integral_constant<std::size_t, 3>{});
        }

        send_out(key, std::move(child_leaf_info), std::integral_constant<std::size_t, 4>{});

        contributions.erase(backiter, contributions.end());
#if 0
        if (contributions.size() > 0) {
          std::cout << "DOWN " << key << " has " << contributions.size() << " contributions left! " << std::endl;
          for (const auto& contribution : contributions) {
            std::cout << "DOWN " << key << " LEFT contribution " << contribution.source << "->" << contribution.dest << std::endl;
          }
        }
#endif // 0

        assert(contributions.empty() && "All contributions should have been sent!");

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST

      }, ttg::edges(down_contribution_edge, ttg::fuse(input, down_recursive_edge)),
         ttg::edges(down_contribution_edge, contribution_edge, down_recursive_edge, down_to_adjust_leaf_node, down_to_adjust_leaf_child_info),
         "Down");

    /* Set the contribution reducer. On the way down, we receive from ourself, our parent, and 6 neighbors.
       Some nodes (root, boundaries) receive fewer contributions and will have to be adjusted dynamically. */
    constexpr std::size_t num_down_contributions = 2; // contributions from self and parent
    down_contributions_tt->template set_input_reducer<0>([](std::vector<detail::KeyPair<NDIM>>& a, const std::vector<detail::KeyPair<NDIM>>& b){
      a.insert(a.end(), b.begin(), b.end());
    }, num_down_contributions);


    /****************************************************************************************************************************
     * Task that receives a node (either from shell0 if it exists already, or from down task if its new) and the leaf status of
     * its children, adjusts the leaf status based on whether it has children or receives contributions, and sends it to shellN.
     ****************************************************************************************************************************/
    auto adjust_leaf_tt = ttg::make_tt<Space>(
      [=](const Key<NDIM>& key, FunctionsCompressedNode<T, NDIM>&& node, const ChildLeafInfo& child_info) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
        };
#endif

        if (node.invalid()) {
          node = FunctionsCompressedNode<T, NDIM>(key, N);
        }

        for (auto child : children(key)) {
          // TODO: drop the per-function leaf info from nodes
          for (int i = 0; i < N; ++i) {
            node.set_child_leaf(i, child.childindex(), child_info.is_child_leaf[child.childindex()]);
          }
        }

        send_out(key, std::move(node), std::integral_constant<std::size_t, 0>{});

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(down_to_adjust_leaf_node, down_to_adjust_leaf_child_info), ttg::edges(to_shellN), "AdjustLeaf");


    /**
     * TODO: TTG needs a way to programatically set the number of inputs from within another TT, i.e., from an output to an input terminal.
     *       Taking the raw pointer here is a dirty hack!
     */
    auto screener_tt = ttg::make_tt<Space>(
      [&, N, K, thresh, name, up_tt_ptr = up_contributions_tt.get(), down_tt_ptr = down_contributions_tt.get()](const mra::Key<NDIM>& key,
                              const mra::FunctionsCompressedNode<T, NDIM>& in_node) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
        };
#endif
        /**
         * Compute the norm of the input node and apply a screening threshold based on the operator and the input node norms.
         * Assemble a list of key pairs that pass the screening and send them up the tree.
         * We follow MADNESS here and use a maximum distance of [-3,3] in each dimension for the screening.
         */

        std::vector<detail::KeyPair<NDIM>> contributions;

        if (in_node.empty()){
          //std::cout << "SCREENER " << key << " empty, sending empty to children" << std::endl;
        } else {

          /**
           * Compute the cnorm using the norm kernel.
           */

          Tensor<T, 1> cnorms(N, TempScope);
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::select(in_node.buffer(), cnorms.buffer());
#endif

          submit_simple_norm_kernel(key, in_node.coeffs().current_view(), N, cnorms.current_view());

#ifndef MRA_ENABLE_HOST
          co_await ttg::device::wait(cnorms.buffer());
#endif

          auto cnorm_view = cnorms.view_on(ttg::device::Device::host());

          /**
           * Assemble our list of contributions.
           */
          for (int d0 = -3; d0 <= 3; ++d0) {
            for (int d1 = -3; d1 <= 3; ++d1) {
              for (int d2 = -3; d2 <= 3; ++d2) {
                auto disp_key = mra::Key<NDIM>(key.level(), {d0, d1, d2});
                mra::Key<NDIM> neighbor_key = key.neighbor(disp_key);
                if (!neighbor_key.is_valid() || neighbor_key == key) {
                  continue;
                }
                auto op_data = op.get_op(key.level(), disp_key);
                for (int i = 0; i < N; ++i) {
                  if (op_data->norm * cnorm_view(i) > thresh) {
                    contributions.push_back({key, neighbor_key});
                    break; // if any of the coefficients pass the threshold we add the contribution
                  }
                }
              }
            }
          }

          //std::cout << "SCREEN " << key << " computed contributions " << contributions.size() << std::endl;
        }

        /**
         * Count existing children and adjust UP reduction count.
         */
        int num_empty = 0;
        for (auto child : children(key)) {
          bool all_child_fn_leaf = true;
          for (int i = 0; i < N; ++i) {
            if (!in_node.is_child_leaf(i, child.childindex())) {
              all_child_fn_leaf = false;
              break;
            }
          }
          if (all_child_fn_leaf) {
            ++num_empty;
          }
        }

        if (num_empty > 0) {
          // if all children are leafs the up task will receive only its contributions.
          //std::cout << "SCREEN " << key << " is all leaf, adjusting up contributions to 1" << std::endl;
          up_tt_ptr->template set_argstream_size<0>(key, num_up_contributions - num_empty);
        }
        if (key.level() == 0) {
          // if we are the root we have no parent or neighbors, so we receive contributions only from ourselves on the way down.
          //std::cout << "SCREEN " << key << " is root, adjusting down contributions to 1" << std::endl;
          down_tt_ptr->template set_argstream_size<0>(key, 1);
        }

        send_out(key, contributions, std::integral_constant<std::size_t, 1>{}); // send contributions on the way up

#ifndef MRA_ENABLE_HOST
        if (!contributions.empty()) {
          sends.push_back(ttg::device::broadcast<0>(std::move(contributions), in_node)); // broadcast the input node to the accumulate tasks
        }
        co_await std::move(sends);
#else
        if (!contributions.empty()) {
          ttg::broadcast<0>(std::move(contributions), in_node); // broadcast the input node to the accumulate tasks
        }
#endif // MRA_ENABLE_HOST
      },
      ttg::edges(input), ttg::edges(screener_to_accumulate, up_contribution_edge), "Screen");

    /**
     * The task that applies the convolution operator on shell 0.
     * The result is sent to the task that applies the contributions that have been identified and communicated up and down the tree.
     */
    auto shell0_tt = ttg::make_tt<Space>(
      [&, N, K, thresh, name](
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

      //std::cout << "SHELL0 " << key << " applying shell 0 contribution (empty " << in_node.empty()
      //          << ", all child leaves " << in_node.is_all_child_leaf() << ")" << std::endl;

      mra::FunctionsCompressedNode<T, NDIM> out(key, N);
      out.set_ns();

      if (!in_node.empty()) {

        // for fixed distance, the 2nd arg to get_op needs to be mra::Key<NDIM>(key.level(), {0, 0, 0})
        auto op_data = op.get_op(key.level(), mra::Key<NDIM>(key.level(), {0, 0, 0}));

        out.allocate(K, ttg::scope::Allocate);
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
        T opnorm = op_data->norm * op_data->fac;
        T tol = thresh*0.01;
        std::array<bool, 2> at = {true, key.level()>0}; // apply terms analogue in MADNESS
        // if (key.level() == 0) at[1] = false; // do not apply S at level 0

        auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K)*N, TempScope);

        for (size_type i = 0; i < NDIM; ++i) normr *= op_data->ops[i]->Rnorm;
        for (size_type i = 0; i < NDIM; ++i) norms *= op_data->ops[i]->Snorm;

        // std::cout << "MRA:: For Key: " << key << "\n the operators being passed are \n R\n" << op_data->ops[0]->R.current_view() << "\nand S: \n" << op_data->ops[0]->S.current_view() << std::endl;


#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(in_node.coeffs().buffer(), out.coeffs().buffer(), tmp);
        for (Dimension d = 0; d < NDIM; ++d) {
          input.add(op_data->ops[d]->R.buffer());
          input.add(op_data->ops[d]->S.buffer());
        }
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

        auto in_node_view = in_node.coeffs().current_view();
        auto out_view = out.coeffs().current_view();

        auto transr = std::array{op_data->ops[0]->R.current_view(), op_data->ops[1]->R.current_view(), op_data->ops[2]->R.current_view()};
        auto transs = std::array{op_data->ops[0]->S.current_view(), op_data->ops[1]->S.current_view(), op_data->ops[2]->S.current_view()};
        // empty in node view
        auto empty_node = mra::FunctionsCompressedNode<T, NDIM>();
        auto empty_node_view = empty_node.coeffs().current_view();
        submit_convolution_kernel<T, NDIM>(key, K, N, opnorm, normr, norms, fac, empty_node_view,
                                            in_node_view, out_view, transr, transs, at, tol,
                                            tmp.current_device_ptr(), ttg::device::current_stream());

      }

      send_out(key, std::move(out));

#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    }, ttg::edges(input), ttg::edges(down_to_adjust_leaf_node), "Shell0");


    /**
     * Recursive edges
     */
    ttg::Edge<detail::KeyPair<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> accumulate_node_recurse;
    ttg::Edge<detail::KeyPair<NDIM>, std::vector<detail::KeyPair<NDIM>>> accumulate_contribution_recurse;

    /**
     * Dispatch task receives the
     */
    auto accumulate_dispatch = ttg::make_tt<Space>(
      [=](const mra::Key<NDIM>& key,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node,
          const std::vector<detail::KeyPair<NDIM>>& contributions) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
        };
#endif

        if (contributions.empty()) {
          // if we have no contributions to apply, just forward the input to the output
          //std::cout << "ACCUMULATE DISPATCH " << key << " has no contributions, forwarding input to output" << std::endl;
          send_out(key, in_node, std::integral_constant<std::size_t, 2>{});
        } else {
          // send the input node and the list of contributions to the accumulate task that applies contributions one by one recursively
          //std::cout << "ACCUMULATE DISPATCH " << key << " dispatching to accumulate " << contributions.back() << " with " << contributions.size() << " contributions" << std::endl;
          send_out(contributions.back(), in_node, std::integral_constant<std::size_t, 0>{});
          send_out(contributions.back(), contributions, std::integral_constant<std::size_t, 1>{});
        }

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(to_shellN, contribution_edge),
         ttg::edges(accumulate_node_recurse, accumulate_contribution_recurse, result),
         "AccumulateDispatch");

    /**
     * Task that applies the contributions from non-zero shells, as determined on the way down the tree.
     * The task recurses over the list of contributions to apply them one by one and send the result to the next contribution task.
     * After the last contribution is applied, the result is sent to the output.
     *
     * NOTE: because we use coroutines we cannot outline most of the code and instead have to copy past it here.
     */
    auto accumulate_tt = ttg::make_tt<Space>(
      [&, N, K, thresh, name](
          const detail::KeyPair<NDIM>& keypair,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node,
          const mra::FunctionsCompressedNode<T, NDIM>& contribution,
          std::vector<detail::KeyPair<NDIM>>&& contribution_keys) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward();
      auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
        sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
      };
#else
      auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
        ttg::send<I>(k, std::forward<S>(out));
      };
#endif

      auto key = keypair.dest;

      //std::cout << "ACCUMULATE " << key << " in_node " << in_node.key() << " applying contribution " << contribution_keys.back()
      //          << " with " << contribution_keys.size() << " contributions left" << std::endl;

      assert(!contribution_keys.empty());
      assert(contribution_keys.back() == keypair);
      assert(!contribution.empty());
      assert(!in_node.invalid()); // we should always get a valid node from child adjust task
      // we allow for invalid nodes here because we may have contributions for nodes that previously did not exist
      assert(key == in_node.key());

      // remove the current key
      contribution_keys.pop_back();

      mra::FunctionsCompressedNode<T, NDIM> out(key, N);

      auto op_data = op.get_op(key.level(), mra::Key<NDIM>(key.level(), {0, 0, 0}));

      out.allocate(K, ttg::scope::Allocate);

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
      T opnorm = op_data->norm * op_data->fac;
      T tol = thresh*0.01;
      std::array<bool, 2> at = {true, key.level()>0}; // apply terms analogue in MADNESS
      // if (key.level() == 0) at[1] = false; // do not apply S at level 0

      auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K)*N, TempScope);

      for (size_type i = 0; i < NDIM; ++i) normr *= op_data->ops[i]->Rnorm;
      for (size_type i = 0; i < NDIM; ++i) norms *= op_data->ops[i]->Snorm;

      // std::cout << "MRA:: For Key: " << key << "\n the operators being passed are \n R\n" << op_data->ops[0]->R.current_view() << "\nand S: \n" << op_data->ops[0]->S.current_view() << std::endl;

#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(in_node.coeffs().buffer(), out.coeffs().buffer(), contribution.coeffs().buffer(), tmp);
      for (Dimension d = 0; d < NDIM; ++d) {
        input.add(op_data->ops[d]->R.buffer());
        input.add(op_data->ops[d]->S.buffer());
      }
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

      auto transr = std::array{op_data->ops[0]->R.current_view(), op_data->ops[1]->R.current_view(), op_data->ops[2]->R.current_view()};
      auto transs = std::array{op_data->ops[0]->S.current_view(), op_data->ops[1]->S.current_view(), op_data->ops[2]->S.current_view()};

      auto out_view = out.coeffs().current_view();
      auto contribution_view = contribution.coeffs().current_view();
      auto in_node_view = in_node.coeffs().current_view();
      submit_convolution_kernel<T, NDIM>(key, K, N, opnorm, normr, norms, fac, in_node_view,
                                          contribution_view, out_view, transr, transs, at, tol,
                                          tmp.current_device_ptr(), ttg::device::current_stream());

      if (contribution_keys.empty()) {
         // if this was the last contribution to apply, send the result to the output
         send_out(keypair.dest, std::move(out), std::integral_constant<std::size_t, 2>{});
      } else {
        // send the result to the next contribution task or output
        send_out(contribution_keys.back(), std::move(out), std::integral_constant<std::size_t, 0>{});
        send_out(contribution_keys.back(), std::move(contribution_keys), std::integral_constant<std::size_t, 1>{});
      }

#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(accumulate_node_recurse, screener_to_accumulate, accumulate_contribution_recurse),
         ttg::edges(accumulate_node_recurse, accumulate_contribution_recurse, result),
         "Accumulate");

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      up_contributions_tt->set_keymap(procmap);
      down_contributions_tt->set_keymap(procmap);
      screener_tt->set_keymap(procmap);
      //neighbor_dispatch_tt->set_keymap(procmap);
      //rebalance_down->set_keymap(procmap);
      shell0_tt->set_keymap(procmap);
      adjust_leaf_tt->set_keymap(procmap);
      accumulate_dispatch->set_keymap(procmap);
      accumulate_tt->set_keymap([=](const detail::KeyPair<NDIM>& kp) { return procmap(kp.dest); });
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      up_contributions_tt->set_devicemap(devicemap);
      down_contributions_tt->set_devicemap(devicemap);
      screener_tt->set_devicemap(devicemap);
      //neighbor_dispatch_tt->set_devicemap(devicemap);
      //rebalance_down->set_devicemap(devicemap);
      shell0_tt->set_devicemap(devicemap);
      adjust_leaf_tt->set_devicemap(devicemap);
      accumulate_dispatch->set_devicemap(devicemap);
      accumulate_tt->set_devicemap([=](const detail::KeyPair<NDIM>& kp) { return devicemap(kp.dest); });
    }
    // TODO: assemble TTG

    auto ins = std::make_tuple(screener_tt->template in<0>());
    auto outs = std::make_tuple(accumulate_tt->template out<2>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(7);
    ops[0] = std::move(up_contributions_tt);
    ops[1] = std::move(down_contributions_tt);
    ops[2] = std::move(screener_tt);
    ops[3] = std::move(shell0_tt);
    ops[4] = std::move(adjust_leaf_tt);
    ops[5] = std::move(accumulate_dispatch);
    ops[6] = std::move(accumulate_tt);
    return make_ttg(std::move(ops), ins, outs, name);
#if 0
    return std::make_tuple(std::move(up_contributions_tt), std::move(down_contributions_tt), std::move(screener_tt),
                           //std::move(neighbor_dispatch_tt),
                           //std::move(rebalance_down),
                           std::move(adjust_leaf_tt),
                           std::move(shell0_tt), std::move(accumulate_dispatch), std::move(accumulate_tt));
#endif
  }




#if 0
    /************************************************************************************************************
     * A dispatch task that receives the input nodes and distributes them to the neighbors in all dimensions.
     *************************************************************************************************************/
    auto neighbor_dispatch_tt = ttg::make_tt(
      [=](const mra::Key<NDIM>& key,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
        };
#endif

        assert(in_node.key() == key);

        std::tuple<std::vector<Key<NDIM>>, std::vector<Key<NDIM>>, std::vector<Key<NDIM>>,
                   std::vector<Key<NDIM>>, std::vector<Key<NDIM>>, std::vector<Key<NDIM>>> neighbor_keys_tuple;
        auto send_to_neighbor = [&]<std::size_t I>(const Key<NDIM>& neighbor, std::index_sequence<I>) {
          if (neighbor.is_valid()) {
            std::cout << "NEIGHBOR DISPATCH " << key << " sending to neighbor " << neighbor << " on terminal " << I << std::endl;
            std::get<I>(neighbor_keys_tuple).push_back(neighbor);
          }
        };
        /**
         * NOTE: outputs are swapped so that our left neighbor receives us on the right input.
         */
        send_to_neighbor(key.neighbor(0, -1), std::index_sequence<1>{});
        send_to_neighbor(key.neighbor(0,  1), std::index_sequence<0>{});
        send_to_neighbor(key.neighbor(1, -1), std::index_sequence<3>{});
        send_to_neighbor(key.neighbor(1,  1), std::index_sequence<2>{});
        send_to_neighbor(key.neighbor(2, -1), std::index_sequence<5>{});
        send_to_neighbor(key.neighbor(2,  1), std::index_sequence<4>{});

        if (key.level() == 0) {
          // root has no neighbors, so just send to the down task
          std::cout << "NEIGHBOR DISPATCH " << key << " is root, sending empty nodes to all neighbor inputs" << std::endl;
          auto empty_node = mra::FunctionsCompressedNode<T, NDIM>();
          std::get<0>(neighbor_keys_tuple).push_back(key);
          std::get<1>(neighbor_keys_tuple).push_back(key);
          std::get<2>(neighbor_keys_tuple).push_back(key);
          std::get<3>(neighbor_keys_tuple).push_back(key);
          std::get<4>(neighbor_keys_tuple).push_back(key);
          std::get<5>(neighbor_keys_tuple).push_back(key);
#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::broadcast<0, 1, 2, 3, 4, 5>(std::move(neighbor_keys_tuple), std::move(empty_node)));
          co_await std::move(sends);
#else
          ttg::broadcast<0, 1, 2, 3, 4, 5>(std::move(neighbor_keys_tuple), std::move(empty_node));
#endif // MRA_ENABLE_HOST

        } else {

#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::broadcast<0, 1, 2, 3, 4, 5>(std::move(neighbor_keys_tuple), in_node));
          co_await std::move(sends);
#else
          ttg::broadcast<0, 1, 2, 3, 4, 5>(std::move(neighbor_keys_tuple), in_node);
#endif // MRA_ENABLE_HOST
        }
      }, ttg::edges(input),
         ttg::edges(neighbor_edges[0], neighbor_edges[1], neighbor_edges[2], neighbor_edges[3], neighbor_edges[4], neighbor_edges[5]),
         name + "-neighbor-dispatch");

    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> rebalance_recurse_edge;


    /*****************************************************************************************************************************
     * TT that receives lists of contributions from its children, combines them through a reducer terminal,
     * and sends them to their neighbors and parent.
     * TODO: this does not work as planned because the receiver's ancestors do not know that they receive something.
     *       We will have to rethink this.
     *****************************************************************************************************************************/

    auto up_contributions_tt = ttg::make_tt(
      [=](const mra::Key<NDIM>& key,
          std::vector<detail::KeyPair<NDIM>>&& contributions) {
        /**
         * Combine contributions from children and send them to the appropriate neighbors and parent.
         * We need to be careful to only send one message per neighbor/parent, so we will combine contributions that go to the same destination.
         */

        //ttg::trace(name + "-up", key, contributions.size());
        std::cout << "UP " << key << " received " << contributions.size() << " contributions" << std::endl;

        if (key.level() == 0) {
          // root has no neighbors and no parent, so forward the contributions to the down task
          ttg::send<0>(key, std::move(contributions));

        } else {

          /**
           * Not the root.
           * Iterate over our neighbors and send the contributions they are responsible for.
           * We remove each key we have sent from the contributions vector so that at the end
           * we are left with only contributions that need to be sent up to the parent.
           */

          auto backiter = contributions.end();

          auto send_to_neighbor = [&](std::array<int, NDIM> displacement) {
            std::vector<detail::KeyPair<NDIM>> neighbor_contributions;
            auto neighbor = key + displacement;
            if (!neighbor.is_valid()) {
              return;
            }
            for (auto it = contributions.begin(); it != backiter; ++it) {
              const auto& contribution = *it;
              if (neighbor.is_ancestor_of(contribution.dest)) {
                neighbor_contributions.push_back(contribution);
                // replace with last element
                --backiter;
                *it = *backiter;
                --it; // step back one so that the next iteration will not skip the element we just swapped in
              }
            }
            std::cout << "UP " << key << " sending " << neighbor_contributions.size() << " contributions to neighbor " << neighbor << std::endl;
            ttg::send<0>(neighbor, std::move(neighbor_contributions));
          };

          constexpr int num_neighbors = 6;
          constexpr int num_children = Key<NDIM>::num_children();

          // send to neighbors in the order of -x, +x, -y, +y, -z, +z
          send_to_neighbor({-1,  0,  0});
          send_to_neighbor({ 1,  0,  0});
          send_to_neighbor({ 0, -1,  0});
          send_to_neighbor({ 0,  1,  0});
          send_to_neighbor({ 0,  0, -1});
          send_to_neighbor({ 0,  0,  1});

          // find the ones we are an ancestor for
          send_to_neighbor({0, 0, 0});


          // shrink the contributions vector to include only elements we have not sent yet
          contributions.erase(backiter, contributions.end());

          // send the rest up to the parent
          std::cout << "UP " << key << " sending " << contributions.size() << " contributions to parent " << key.parent() << std::endl;
          ttg::send<1>(key.parent(), std::move(contributions));

        }

      }, ttg::edges(up_contribution_edge, input), ttg::edges(down_contribution_edge, up_contribution_edge), name + "-up");
#endif // 0

#if 0
    /**************************************************************************************************************************************************
     * The TT that walks down the tree where each node receives its 6 neighbors and sets the reducer terminal size
     * for the down-tree tasks where not all neighbors exist. On the way, we need to send empty nodes to our children that have no neighbors,
     * based on whether our neighbors have children or not.
     *
     * TODO: sending sideways does not currently work. We have to go through the through until we figure out how to balance the tree properly.
     *************************************************************************************************************************************************/
     auto rebalance_down = ttg::make_tt([=, down_tt_ptr = down_contributions_tt.get()](
                                            const mra::Key<NDIM>& key,
                                            const mra::FunctionsCompressedNode<T, NDIM>& in_node,
                                            const mra::FunctionsCompressedNode<T, NDIM>& neighbor_l0,
                                            const mra::FunctionsCompressedNode<T, NDIM>& neighbor_r0,
                                            const mra::FunctionsCompressedNode<T, NDIM>& neighbor_l1,
                                            const mra::FunctionsCompressedNode<T, NDIM>& neighbor_r1,
                                            const mra::FunctionsCompressedNode<T, NDIM>& neighbor_l2,
                                            const mra::FunctionsCompressedNode<T, NDIM>& neighbor_r2) -> TASKTYPE
      {

#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
        };
#endif

        if (key.level() == 0) {
          // root has no neighbors, so just fill in the neighbors of the children with empty nodes
          assert(neighbor_l0.empty() && neighbor_r0.empty() && neighbor_l1.empty() && neighbor_r1.empty() &&
                 neighbor_l2.empty() && neighbor_r2.empty() && "Root should have only neighbors!");
        }

#if 0
        if (in_node.is_all_child_leaf()) {
          // if all children are leafs we have no contributions coming up from the children, so we can just forward the contributions from our parent and neighbors down without rebalancing.
          std::cout << "REBALANCE " << key << " is all leaf, nothing to do" << std::endl;
        } else {
#endif // 0

          constexpr auto Left = std::integral_constant<typename Key<NDIM>::Direction, Key<NDIM>::Direction::Left>{};
          constexpr auto Right = std::integral_constant<typename Key<NDIM>::Direction, Key<NDIM>::Direction::Right>{};
          const auto neighbors = std::array{
            std::make_tuple(std::ref(neighbor_l0), 0, Key<NDIM>::Direction::Left), std::make_tuple(std::ref(neighbor_r0), 0, Key<NDIM>::Direction::Right),
            std::make_tuple(std::ref(neighbor_l1), 1, Key<NDIM>::Direction::Left), std::make_tuple(std::ref(neighbor_r1), 1, Key<NDIM>::Direction::Right),
            std::make_tuple(std::ref(neighbor_l2), 2, Key<NDIM>::Direction::Left), std::make_tuple(std::ref(neighbor_r2), 2, Key<NDIM>::Direction::Right)
          };

          const std::array<bool, 2*NDIM> has_neighbor{!neighbor_l0.empty(), !neighbor_r0.empty(),
                                                      !neighbor_l1.empty(), !neighbor_r1.empty(),
                                                      !neighbor_l2.empty(), !neighbor_r2.empty()};

          bool all_neighbors_empty = std::all_of(has_neighbor.begin(), has_neighbor.end(), [](bool b){ return !b; });
          bool any_neighbor_nonempty = std::any_of(has_neighbor.begin(), has_neighbor.end(), [](bool b){ return b; });

          bool all_child_leaf = in_node.is_all_child_leaf();

          std::cout << "REBALANCE " << key << " has neighbors " << has_neighbor << ", all neighbors empty: " << all_neighbors_empty
                    << ", any neighbor non-empty: " << any_neighbor_nonempty << ", all child leaf: " << all_child_leaf << "\n"
                    << ", neighbor_l0 " << neighbor_l0.key() << " all child leaf: " << neighbor_l0.is_all_child_leaf() << ", invalid neighbor_l0: " << neighbor_l0.invalid() << "\n"
                    << ", neighbor_r0 " << neighbor_r0.key() << " all child leaf: " << neighbor_r0.is_all_child_leaf() << ", invalid neighbor_r0: " << neighbor_r0.invalid() << "\n"
                    << ", neighbor_l1 " << neighbor_l1.key() << " all child leaf: " << neighbor_l1.is_all_child_leaf() << ", invalid neighbor_l1: " << neighbor_l1.invalid() << "\n"
                    << ", neighbor_r1 " << neighbor_r1.key() << " all child leaf: " << neighbor_r1.is_all_child_leaf() << ", invalid neighbor_r1: " << neighbor_r1.invalid() << "\n"
                    << ", neighbor_l2 " << neighbor_l2.key() << " all child leaf: " << neighbor_l2.is_all_child_leaf() << ", invalid neighbor_l2: " << neighbor_l2.invalid() << "\n"
                    << ", neighbor_r2 " << neighbor_r2.key()  <<" all child leaf: " << neighbor_r2.is_all_child_leaf() << ", invalid neighbor_r2: " << neighbor_r2.invalid()
                    << std::endl;

          if (all_neighbors_empty && all_child_leaf) {
            // if all neighbors are empty and we have no contributions coming up from the children, we can just forward the contributions from our parent down without rebalancing.
            std::cout << "REBALANCE " << key << " is all leaf and all neighbors empty, nothing to do" << std::endl;
          } else {
             std::cout << "REBALANCE " << key << " has non-empty neighbors, checking children" << std::endl;

            /**
             * For each dimension, check whether
             * 1) a child is located on the boundary; or
             * 2) no neighbor in that direction exists (i.e., our neighbor is empty or has no children).
             * In either case, we need to send an empty node to the child in that direction since the child
             * will not receive any contributions from that direction and needs to know to expect that.
             *
             * The lambda also returns the number of empty nodes sent out so that we can adjust the number of inputs
             * to the reduction terminal on the way down.
             */
            auto check_child_exists = [&](const auto& node, const auto& child){
              if (node.invalid()) {
                return false;
              }
              for (int i = 0; i < N; ++i) {
                if (node.is_child_leaf(i, child.childindex())) {
                  return false;
                }
              }
              return true;
            };
#if 0
            auto rebalance_dir = [&](const auto& child, const auto& neighbor_node, auto disp, auto d, const std::string& dir_name) {
              int num_empty = 0;
              auto neighbor_key = neighbor_node.key();
              auto child_neighbor = child.neighbor(d, static_cast<int>(disp()));
              constexpr auto terminal_id = 2*d() + (disp() == Key<NDIM>::Direction::Left ? 0 : 1);
#if 0
              bool child_neighbor_exists = key.level() == 0 // level 1 always exists
                                        || (child_neighbor.parent() == key)
                                        || (!child.is_boundary(d, disp)
                                          && (neighbor_key == child_neighbor.parent())
                                          && !neighbor_node.is_all_child_leaf()
                                          && !neighbor_node.invalid());

              if (child.is_boundary(d, disp))
                std::cout << "REBALANCE " << key << " child " << child << " is on the "
                          << dir_name << " boundary in dimension " << d() << std::endl;
              if (key.level() > 0 && (neighbor_key == child_neighbor.parent()) && (neighbor_node.is_all_child_leaf() || neighbor_node.invalid()))
                std::cout << "REBALANCE " << key << " child " << child << " has " << dir_name << " neighbor " << child_neighbor << " parent "
                          << neighbor_key << " that is all child leaf or is invalid in dimension " << d() << std::endl;
#endif // 0

              bool needs_empty_neighbor = false;
              if (child.is_boundary(d, disp) && !check_is_child_leaf(in_node, child)) {
                // if the child is on the boundary and is a leaf, it needs an empty neighbor to know to expect no contributions from that direction
                std::cout << "REBALANCE " << key << " child " << child << " is on the "
                          << dir_name << " boundary in dimension " << d() << " and is not a leaf, needs empty neighbor" << std::endl;
                needs_empty_neighbor = true;
              } else if (key.level() > 0) { // children of level 0 keys have all neighbors
                if (child_neighbor.parent() == key) {
                  // check if the child is a leaf and its neighbor is not, in which case we need to fill the input with an empty node.
                  if (!check_is_child_leaf(in_node, child) && check_is_child_leaf(in_node, child_neighbor)) {
                    std::cout << "REBALANCE " << key << " child " << child << " has neighbor " << child_neighbor
                              << " that is our child but is a leaf, needs empty neighbor" << std::endl;
                    needs_empty_neighbor = true;
                  }
                } else if (check_is_child_leaf(neighbor_node, child_neighbor) && neighbor_node.invalid()) {
                  // if the neighbor in that direction is empty/invalid or all child leaf, we need to send an empty node to the child since it will not receive any contributions from that direction and needs to know to expect that.


                }
              }
              } else if (key.level() > 0 && (neighbor_key == child_neighbor.parent()) && (neighbor_node.is_all_child_leaf() || neighbor_node.invalid())) {
                // the neighbor is empty/invalid
                needs_empty_neighbor = true;
              }

              if (needs_empty_neighbor) {
                std::stringstream reason;
#if 0
                if (child.is_boundary(d, disp)) reason << "is on the " << dir_name << " boundary in dimension " << d() << " ";
                if (neighbor_node.invalid()) reason << "has invalid " << dir_name << " neighbor " << child_neighbor << " in dimension " << d() << " ";
                if (neighbor_node.is_all_child_leaf()) reason << "has " << dir_name << " neighbor " << child_neighbor
                                                                << " parent " << neighbor_key << " all child leaf neighbor in dimension " << d() << " ";
#endif // 0
                std::cout << "REBALANCE " << key << " child " << child << " " << reason.str()
                          << "sending empty node on terminal " << terminal_id << std::endl;
                send_out(child, mra::FunctionsCompressedNode<T, NDIM>(), std::integral_constant<std::size_t, terminal_id>{});
                ++num_empty;
              }

              if (any_neighbor_nonempty && neighbor_node.is_all_child_leaf() && (all_child_leaf || in_node.invalid()) && !child.is_boundary(d, disp)) {
                // we have to make sure that our children's neighbors have their inputs satisfied
                constexpr auto rev_terminal_id = 2*d() + (disp() == Key<NDIM>::Direction::Left ? 1 : 0);
                std::cout << "REBALANCE " << key << " child " << child << " has at least one non-empty neighbor, sending empty node to child neighbor "
                          << child_neighbor << " on terminal " << rev_terminal_id << std::endl;
                send_out(child_neighbor, FunctionsCompressedNode<T, NDIM>(), std::integral_constant<std::size_t, rev_terminal_id>{});
              }
              return num_empty;
            };
#endif // 0

            for (auto child : children(key)) {
              int num_empty = 0;
              bool child_exists = check_child_exists(in_node, child);
              std::array<bool, 2*NDIM> child_has_neighbor{false, false, false, false, false, false};
              /**
               * First: check whether any of the neighbors of that child have contributions to send.
               */
              for (auto& neighbor : neighbors) {
                auto& neighbor_node = std::get<0>(neighbor);
                auto neighbor_key = neighbor_node.key();
                int d = std::get<1>(neighbor);
                int dir = static_cast<std::size_t>(std::get<2>(neighbor));
                auto child_neighbor = child.neighbor(d, dir); // check the neighbor
                std::cout << "REBALANCE " << key << " child " << child << "(exists " << child_exists << ") checking neighbor " << d << " " << dir << " " << neighbor_key << " child neighbor " << child_neighbor << std::endl;

                if (child.is_boundary(d, std::get<2>(neighbor))) {
                  std::cout << "REBALANCE " << key << " child " << child << " is on the boundary in direction " << d << " " << dir
                            << ", treating as empty neighbor" << std::endl;
                  continue;
                }
                if (child_neighbor.parent() == neighbor_key) {
                  if (check_child_exists(neighbor_node, child_neighbor)) {
                    std::cout << "REBALANCE " << key << " child " << child << " has neighbor " << child_neighbor
                              << " that is a leaf, treating as empty neighbor" << std::endl;
                    child_has_neighbor[&neighbor - &neighbors[0]] = true;
                  } else {
                    std::cout << "REBALANCE " << key << " child " << child << " child neighbor " << child_neighbor
                              << " does not exist in " << neighbor_key << " (invalid " << neighbor_node.invalid()
                              << ", all child leafs " << neighbor_node.is_all_child_leaf() << ")" << std::endl;
                  }
                } else if (child_neighbor.parent() == key) {
                  // the child's neighbor is one of our children, so check whether that is a leaf.
                  if (check_child_exists(in_node, child_neighbor)) {
                    std::cout << "REBALANCE " << key << " child " << child << " has neighbor " << child_neighbor
                              << " that is our child and is a leaf, treating as empty neighbor" << std::endl;
                    child_has_neighbor[&neighbor - &neighbors[0]] = true;
                  } else {
                    std::cout << "REBALANCE " << key << " child " << child << " has neighbor " << child_neighbor
                              << " that is our child but does not exist in our node (invalid " << in_node.invalid()
                              << ", all child leafs " << in_node.is_all_child_leaf() << ")" << std::endl;
                  }
                }
              }

              bool has_neighbors = std::any_of(child_has_neighbor.begin(), child_has_neighbor.end(), [](bool b){ return b; });
              std::cout << "REBALANCE " << key << " child " << child << " has neighbors: " << child_has_neighbor << std::endl;
              if (child_exists || has_neighbors) {
                /**
                 * This child has at least one neighbor that sent something. Now go over all neighbors that don't exist and send an empty node
                 * to fill their inputs.
                 */
                auto send_empty_if_no_neighbor = [&](auto dir, auto d, const std::string& dir_name) {
                  constexpr auto terminal_id = 2*d() + (dir == Left ? 0 : 1);
                  if (!child_has_neighbor[terminal_id]) {
                    std::cout << "REBALANCE " << key << " child " << child << " has no " << dir_name << " neighbor, sending empty node on terminal "
                              << terminal_id << std::endl;
                    send_out(child, mra::FunctionsCompressedNode<T, NDIM>(), std::integral_constant<std::size_t, terminal_id>{});
                    return 1;
                  }
                  return 0;
                };

                num_empty += send_empty_if_no_neighbor( Left, std::integral_constant<std::size_t, 0>{}, "left x");
                num_empty += send_empty_if_no_neighbor(Right, std::integral_constant<std::size_t, 0>{}, "right x");
                num_empty += send_empty_if_no_neighbor( Left, std::integral_constant<std::size_t, 1>{}, "left y");
                num_empty += send_empty_if_no_neighbor(Right, std::integral_constant<std::size_t, 1>{}, "right y");
                num_empty += send_empty_if_no_neighbor( Left, std::integral_constant<std::size_t, 2>{}, "left z");
                num_empty += send_empty_if_no_neighbor(Right, std::integral_constant<std::size_t, 2>{}, "right z");

                if (has_neighbors && !child_exists) {
                  std::cout << "REBALANCE " << key << " child " << child << " is a leaf but has neighbors, sending down empty node" << std::endl;
                  send_out(child, mra::FunctionsCompressedNode<T, NDIM>(), std::integral_constant<std::size_t, 6>{});
                  ++num_empty;
                }

                if (num_empty > 0) {
                  // set the input stream size for the child in the down task to expect the correct number of contributions
                  std::cout << "REBALANCE " << key << " child " << child << " has " << num_empty
                            << " empty neighbors, adjusting down contributions to " << num_down_contributions - num_empty << std::endl;
                  down_tt_ptr->template set_argstream_size<0>(child, num_down_contributions - num_empty);
                }

              }

#if 0
              if (has_neighbors) {

                num_empty += rebalance_dir(child, neighbor_l0,  Left, std::integral_constant<std::size_t, 0>{}, "left");
                num_empty += rebalance_dir(child, neighbor_r0, Right, std::integral_constant<std::size_t, 0>{}, "right");
                num_empty += rebalance_dir(child, neighbor_l1,  Left, std::integral_constant<std::size_t, 1>{}, "left");
                num_empty += rebalance_dir(child, neighbor_r1, Right, std::integral_constant<std::size_t, 1>{}, "right");
                num_empty += rebalance_dir(child, neighbor_l2,  Left, std::integral_constant<std::size_t, 2>{}, "left");
                num_empty += rebalance_dir(child, neighbor_r2, Right, std::integral_constant<std::size_t, 2>{}, "right");

              }
#endif // 0
            }
          }

#ifndef MRA_ENABLE_HOST
          co_await std::move(sends);
#endif // MRA_ENABLE_HOST
//        }
      }, ttg::edges(ttg::fuse(input, rebalance_recurse_edge), neighbor_edges[0], neighbor_edges[1], neighbor_edges[2], neighbor_edges[3], neighbor_edges[4], neighbor_edges[5]),
         ttg::edges(neighbor_edges[0], neighbor_edges[1], neighbor_edges[2], neighbor_edges[3], neighbor_edges[4], neighbor_edges[5], rebalance_recurse_edge),
         name + "-rebalance");

#endif // 0



} // namespace mra

#endif // MRA_TASKS_CONVOLUTION_H
