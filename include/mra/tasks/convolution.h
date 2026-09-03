#ifndef MRA_TASKS_CONVOLUTION_H
#define MRA_TASKS_CONVOLUTION_H

#include <iostream>
#include <optional>
#include <ttg.h>
#include "mra/kernels.h"
#include "mra/misc/batch_size.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/ops/functions.h"
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


/**
 * TODO: check the DOWN task to make sure we adjust the child leaf info
 *       correctly based on the information we receive from the parent.
 */

namespace mra {


  namespace detail {
    template<Dimension NDIM>
    struct KeyPair {
      Key<NDIM> source;
      Key<NDIM> dest;

      auto operator<=>(const KeyPair&) const = default;
      auto hash() const {
        // Combine every distinguishing field via mulhash (rotate + multiply-XOR)
        // instead of shift-OR: translations can need up to MAX_LEVEL (31) bits,
        // so packing them 7 bits at a time silently overlaps/truncates data,
        // and batch (originally placed at bit 48) gets shifted out entirely
        // by the two translation loops (6 * 7 = 42 bit total shift).
        HashValue hashvalue = mulhash(HashValue(0), source.batch());
        hashvalue = mulhash(hashvalue, source.level());
        for (Dimension d=0; d<NDIM; d++) hashvalue = mulhash(hashvalue, source.translation()[d]);
        hashvalue = mulhash(hashvalue, dest.batch());
        hashvalue = mulhash(hashvalue, dest.level());
        for (Dimension d=0; d<NDIM; d++) hashvalue = mulhash(hashvalue, dest.translation()[d]);
        return hashvalue;
      }
    };

    template<Dimension NDIM>
    std::ostream& operator<<(std::ostream& os, const KeyPair<NDIM>& kp) {
      return os << "(" << kp.source << "->" << kp.dest << ")";
    }

    template<size_type NDIM, std::size_t I, std::size_t... Is>
    void foreach_child_impl(const Key<NDIM>& key, auto&& fn,
                            std::index_sequence<I, Is...> s) {
      fn.template operator()<I>(key.child_at(I));
      if constexpr (sizeof...(Is) > 0) {
        foreach_child_impl(key, fn, std::index_sequence<Is...>{});
      }
    }

    template<size_type NDIM>
    void foreach_child(const Key<NDIM>& key, auto&& fn) {
      foreach_child_impl(key, fn, std::make_index_sequence<Key<NDIM>::num_children()>{});
    }
  } // namespace detail


  /**
   * Convolution entails several steps. For each node:
   *
   * 1) Apply the shell0 contribution.
   * 2) Screen the contributions based on the norms of the input node and the operator norm using the supplied threshold.
   *    Send the input node to the destination nodes and send the list contributions up to the parent.
   * 3) Receive contributions from the children and merge them. Distribute the relevant contributions to our
   *    children and send the remaining merged contribution up to the parent.
   * 4) Receive combined contributions from parent and distribute down to children.
   *    Adjust the leaf status of the node based on the contributions (e.g. if we receive a contribution
   *    from a neighbor but have no children then we must be a leaf, if we receive no contributions but have children then we must be empty).
   *    Send the adjusted node to the shellN task.
   * 5) Iterate over all contributions we will receive by recursively instantiating a task using the pair of keys that
   *    describe the contribution (source and destination). This task will apply the contribution to the input node and send the result
   *    to the next contribution task in the list of contributions. After the last contribution has been applied we send the result to the output.
   */

  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_convolution(const std::shared_ptr<FunctionSetT>& fns, size_type K,
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> input,
                        ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> result,
                        const mra::GaussianConvolutionOperator<T, NDIM>& op,
                        const T thresh,
                        const int truncate_mode,
                        const T cell_min_width,
                        const std::string& name = "convolution",
                        ProcMap procmap = {},
                        DeviceMap devicemap = {}) {

    static_assert(NDIM == 3); // TODO: worth fixing?

    // Batching is controlled process-wide via mra::set_batch_size(), not per
    // call here -- see mra/misc/batch_size.h. Read once, at graph-construction
    // time: this bakes the decision into the tasks/matchers built below, so
    // calling set_batch_size() again after this graph exists has no effect on
    // it (deliberate -- see batch_size.h for why).
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_conv_batching = mra::batching_enabled();
    // Total function count across the whole FunctionSet -- fixed for this
    // operation's entire run (unlike a single node's N, which varies with
    // key.batch()). Used to size each batch leader's sparsity-byte staging
    // buffer to a fixed upper bound so it never needs to grow after its
    // first allocation.
    const size_type total_functions = fns->num_functions();

#ifndef MRA_ENABLE_HOST
    // Shared by shell0_tt and accumulate_tt: they never contend for the same
    // batch group (TTG only ever batches tasks of the same TT together), but
    // the registry itself is one-per-device, lazily constructing a
    // BatchPool<ConvolutionBatchArg<T,NDIM>> for a given device the first time
    // either task type actually runs there -- BatchPool/BatchPoolRegistry
    // (mra/misc/device_batch_pool.h) are generic over the tuple type, so any
    // future batched kernel can reuse them with its own tuple instead of a
    // bespoke pool. max_batch_size no longer sizes the pool (it now grows on
    // demand); it only bounds how many tasks set_batch_matcher below will
    // ever group into one launch.
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::ConvolutionBatchArg<T, NDIM>>> conv_pool;
    if (enable_conv_batching) {
      conv_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::ConvolutionBatchArg<T, NDIM>>>(ttg::device::num_devices(), mra::get_batch_size());
    }
#else
    // GroupedBatchPoolRegistry only exists on device builds; this placeholder only
    // exists so the (shared host/device) task lambdas below can
    // unconditionally list conv_pool in their capture list -- it is never
    // accessed on host builds.
    std::nullptr_t conv_pool = nullptr;
#endif // MRA_ENABLE_HOST

    using ChildLeafInfo = typename mra::FunctionsCompressedNode<T, NDIM>::child_info_type;

    static constexpr const size_type num_children = Key<NDIM>::num_children();


    // TAKEN FROM MADNESS:
    // Tuning here is based on observation that with
    // sufficiently high-order wavelet relative to the
    // precision, that only nearest neighbor boxes contribute,
    // whereas for low-order wavelets more neighbors will
    // contribute.  Sufficiently high is picked as
    // k>=2-log10(eps) which is our empirical rule for
    // efficiency/accuracy and code instrumentation has
    // previously indicated that (in 3D) just unit
    // displacements are invoked.  The error decays as R^-(k+1),
    // and the number of boxes increases as R^d.
    //
    // Fac is the expected number of contributions to a given
    // box, so the error permitted per contribution will be
    // tol/fac

    // radius of shell (nearest neighbor is diameter of 3 boxes, so radius=1.5)
    double radius = 1.5 + 0.33 * std::max(0.0, 2 - std::log10(thresh) -
                                                    K); // 0.33 was 0.5
    //double radius = 2.5;
    const double fac = vol_nsphere(NDIM, radius);

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
    //ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> accumulate_result;

    ttg::Edge<mra::Key<NDIM>, std::vector<detail::KeyPair<NDIM>>> contribution_edge; // connecting the down task to the accumulate dispatch task

    ttg::Edge<detail::KeyPair<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> screener_to_accumulate;

    //ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> shell0_to_shellN;
    ttg::Edge<mra::Key<NDIM>, std::array<bool, Key<NDIM>::num_children()>> down_to_accumulate_leaf_info;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> accumulate_to_adjust_leaf;
    std::array<ttg::Edge<Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>>, Key<NDIM>::num_children()> adjust_leaf_edges;


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
          std::cout << "MRA CONV UP " << contributions.size() << " contributions to distribute at root" << std::endl;

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
    constexpr std::size_t num_up_contributions = 1 + num_children; // contributions from self and children
    up_contributions_tt->template set_input_reducer<0>([](std::vector<detail::KeyPair<NDIM>>& a, const std::vector<detail::KeyPair<NDIM>>& b){
      a.insert(a.end(), b.begin(), b.end());
    }, num_up_contributions);





    /************************************************************************************************
     * Task that receives input from the corresponding UP task and its parent and distributes the keys to
     * the task that applies contributions on itself and to the child tasks.
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


        size_type N = fns->num_functions(key);

        /**
         * We don't know which function(s) each contribution applies to.
         * We mark whether any contribution extend the node's children
         * and send that information to the adjust-leaf task.
         */
        std::array<bool, mra::Key<NDIM>::num_children()> child_empty;
        for (auto& child : children(key)) {
          child_empty[child.childindex()] = node.is_all_child_leaf(child);
        }

        std::vector<Key<NDIM>> child_empty_contributions;
        std::vector<Key<NDIM>> child_empty_nodes;
        // send to all children
        for (auto child : children(key)) {
          auto dest_contributions = filter_dest(child, true, std::index_sequence<0>{});
          int num_contributions = dest_contributions.size();
          if (num_contributions > 0) {
            // send down contributions
            send_out(child, std::move(dest_contributions), std::integral_constant<std::size_t, 0>{});
            child_empty[child.childindex()] = false;
            if (node.invalid() || node.is_all_child_leaf(child)) {
              //std::cout << "DOWN " << key << " child " << child << " is empty but receiving contributions" << std::endl;
              // if the child is a leaf we need to send an empty contribution list to satisfy the second input on the way down
              //std::cout << "DOWN " << key << " node empty or child " << child << " is leaf, sending empty node " << std::endl;
              //send_out(child, std::vector<detail::KeyPair<NDIM>>{}, std::integral_constant<std::size_t, 0>{});
              child_empty_contributions.push_back(child);
              child_empty_nodes.push_back(child);
              //send_out(child, mra::FunctionsCompressedNode<T, NDIM>{}, std::integral_constant<std::size_t, 2>{}); // also send an empty node since the child task will expect one
            }
          } else if (!node.is_all_child_leaf(child)) {
            // we have no contributions but an existing child, send down an empty contribution list to the child
            //std::cout << "DOWN " << key << " sending empty contributions to dest " << child << std::endl;
            //send_out(child, std::vector<detail::KeyPair<NDIM>>{}, std::integral_constant<std::size_t, 0>{});
            child_empty_contributions.push_back(child);
            child_empty[child.childindex()] = false;
          }
        }

        if (!child_empty_contributions.empty()) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::broadcast<0>(std::move(child_empty_contributions), std::vector<detail::KeyPair<NDIM>>{}));
#else
          ttg::broadcast<0>(std::move(child_empty_contributions), std::vector<detail::KeyPair<NDIM>>{});
#endif
        }

        if (!child_empty_nodes.empty()) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(ttg::device::broadcast<2>(std::move(child_empty_nodes), mra::FunctionsCompressedNode<T, NDIM>{}));
#else
          ttg::broadcast<2>(std::move(child_empty_nodes), mra::FunctionsCompressedNode<T, NDIM>{});
#endif
        }

        if (node.invalid()) {
          // the accumulate task won't receive a node from shell0, so we need to send down an empty node
          //std::cout << "DOWN " << key << " node is invalid, sending empty node to shellN" << std::endl;
          send_out(key, node, std::integral_constant<std::size_t, 3>{});
        }

        //std::cout << "DOWN " << key << " sending child leaf info " << child_empty << " to adjust leaf task" << std::endl;
        send_out(key, std::move(child_empty), std::integral_constant<std::size_t, 4>{});

        contributions.erase(backiter, contributions.end());

        assert(contributions.empty() && "All contributions should have been sent!");

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST

      }, ttg::edges(down_contribution_edge, ttg::fuse(input, down_recursive_edge)),
         ttg::edges(down_contribution_edge, contribution_edge, down_recursive_edge, to_shellN, down_to_accumulate_leaf_info),
         "Down");

    /* Set the contribution reducer. On the way down, we receive from ourself, our parent, and 6 neighbors.
       Some nodes (root, boundaries) receive fewer contributions and will have to be adjusted dynamically. */
    constexpr std::size_t num_down_contributions = 2; // contributions from self and parent
    down_contributions_tt->template set_input_reducer<0>([](std::vector<detail::KeyPair<NDIM>>& a, const std::vector<detail::KeyPair<NDIM>>& b){
      a.insert(a.end(), b.begin(), b.end());
    }, num_down_contributions);


    ttg::Edge<mra::Key<NDIM>, DenseTensor<T, 1>> norm_edge; // edge to send the cnorms from the screener to the accumulate task


    /****************************************************
     * Task that computes the norm and forwards it.
     * NOTE: we separate this into its own task because
     *       we otherwise serialize the screening on
     *       the device manager thread.
     ****************************************************/
    auto norm_tt = ttg::make_tt<Space>(
      [=](const Key<NDIM>& key,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node) -> TASKTYPE {

        size_type N = fns->num_functions(key);

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

        DenseTensor<T, 1> cnorms;

        if (!in_node.empty()) {
          cnorms = DenseTensor<T, 1>(N, ttg::scope::Allocate);
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::select(in_node.buffer(), cnorms.buffer());
#endif

          submit_simple_norm_kernel(key, in_node.coeffs().current_view(), N, cnorms.current_view());

#ifndef MRA_ENABLE_HOST
          co_await ttg::device::wait(cnorms.buffer());
#endif
        }
        send_out(key, std::move(cnorms), std::integral_constant<std::size_t, 0>{});

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(input), ttg::edges(norm_edge), "Norm");

    /**
     * TODO: TTG needs a way to programatically set the number of inputs from within another TT, i.e., from an output to an input terminal.
     *       Taking the raw pointer here is a dirty hack!
     */
    auto screener_tt = ttg::make_tt<Space>(
      [&, K, thresh, truncate_mode, cell_min_width, fac, name, up_tt_ptr = up_contributions_tt.get(), down_tt_ptr = down_contributions_tt.get()](
                              const mra::Key<NDIM>& key,
                              const mra::FunctionsCompressedNode<T, NDIM>& in_node,
                              const DenseTensor<T, 1>& cnorms) -> TASKTYPE {

        size_type N = fns->num_functions(key);

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

        if (!in_node.empty()){

          const double tol = truncate_tol(key, thresh, cell_min_width, truncate_mode);

          /**
           * Compute the cnorm using the norm kernel.
           */

          assert(cnorms.buffer().is_current_on(ttg::device::Device::host()) && "cnorms should be on host at this point");

          auto cnorm_view = cnorms.view_on(ttg::device::Device::host());

          const auto real_distance_squared = [&](const auto& mad_op, const auto &displacement)
              -> double {
            return displacement.real_distsq_bc(mad_op->lattice_summed(), madness::FunctionDefaults<NDIM>::get_cell_width());
          };
          const auto lattice_distance_squared = [&](const auto& mad_op, const auto &displacement)
              -> std::uint64_t {
            return displacement.distsq_bc(mad_op->lattice_summed());
          };
          for (int i = 0; i < N; ++i) {
            // we may have either one operator for all functions or one operator per function
            int opnorm_index = (op.count() == 1) ? 0 : i;
            // safe because get_disp returns a reference that remains valid
            const auto& mad_disps = op.get_mad_displacements(opnorm_index, key.level());
            const auto& mad_op = op.get_mad_op(opnorm_index);
            int nused = 1, nvalid = 1;
            std::optional<double> real_last_distsq;
            std::optional<std::uint64_t> lattice_last_distsq;
            for (auto& mad_disp : mad_disps) {

              // Screen out shells. We assume shells are grouped into shells so that the operator decays with shell index.
              // Shells are indexed by least distance from box to the central box.
              // Cells touching so much as a corner of the central box are further grouped by their lattice distance.
              // N.B. lattice-summed decaying kernel is periodic (i.e. does decay w.r.t. r), so loop over shells of displacements sorted by distances modulated by periodicity (Key::distsq_bc)
              const auto real_distsq = real_distance_squared(mad_op, mad_disp);
              const std::uint64_t lattice_distsq = real_distsq ? 0 : lattice_distance_squared(mad_op, mad_disp);
              if (!real_last_distsq.has_value() ||
                  !madness::nearlyEqual(real_distsq, *real_last_distsq) ||
                  (madness::nearlyEqual(*real_last_distsq, 0) && lattice_distsq != *lattice_last_distsq)) { // Moved to next shell of neighbors
                if (nvalid > 0 && nused == 0 && (real_distsq > 0 || lattice_distsq > 1)) {
                  // Have at least done the input box and all first
                  // nearest neighbors, and none of the last set
                  // of neighbors made significant contributions.  Thus,
                  // assuming monotonic decrease, we are done.
                  break;
                }
                nused = 0;
                nvalid = 0;
                real_last_distsq = real_distsq;
                // After real_last_distsq > 0, we stop caring about keeping lattice_last_distsq up-to-date.
                lattice_last_distsq = real_distsq ? std::optional<std::uint64_t>{} : lattice_distsq;
              }
              // Convert MADNESS key to MRA key
              auto disp_key = mra::Key<NDIM>(0, mad_disp);
              mra::Key<NDIM> neighbor_key = key.neighbor(disp_key);
              if (!neighbor_key.is_valid()) {
                continue; // neighbor is outside the domain
              }
              nvalid++;
              if (key == neighbor_key){ // shell 0 is handled by the shell0 task, so we don't need to add it to the contributions
                nused++;
                continue;
              }

              auto op_data = op.get_op(key.level(), disp_key);
              auto opnorm_view = op_data->norms.view_on(ttg::device::Device::host());
              auto opnorm = opnorm_view(opnorm_index, 0, 0, (int)NormId::Opnorm);
              //std::cout << "MRA-SCREEN " << key << " disp " << disp_key << " neighbor " << neighbor_key << " cnorm " << cnorm_view(i)
              //          << " op norm " << op_data->norm << " fac " << fac << " tol/fac " << tol/fac << std::endl;
              if (opnorm * cnorm_view(i) > tol / fac) {
                assert(neighbor_key.level() == key.level() && "neighbor key should be at the same level as the current key");
                if (std::find(contributions.begin(), contributions.end(), detail::KeyPair<NDIM>{key, neighbor_key}) == contributions.end()) {
                  contributions.push_back({key, neighbor_key});
                }
                nused++;
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
          if (in_node.is_all_child_leaf(child)) {
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
      ttg::edges(input, norm_edge), ttg::edges(screener_to_accumulate, up_contribution_edge), "Screen");

    /**
     * The task that applies the convolution operator on shell 0.
     * The result is sent to the task that applies the contributions that have been identified and communicated up and down the tree.
     */
    auto shell0_tt = ttg::make_tt<Space>(
      [&, K, fac, thresh, truncate_mode, cell_min_width, name, enable_conv_batching, conv_pool, total_functions](
          const mra::Key<NDIM>& key,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node) -> TASKTYPE {

      size_type N = fns->num_functions(key);

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
        auto op_data = op.get_op(key.level(), mra::Key<NDIM>(key.batch(), key.level(), {0, 0, 0}));

        /**
         * TODO: instead of allocating a new sparsity object, come up with a way to pass in_node's sparsity
         *       to allocate().
         */
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        sparsity.nonzero_if_any(in_node);
        // out's own sparsity (built below) is exactly this OR, so a single
        // on-device scan of out's sparsity (find_nth_nonzero) suffices to
        // recover each real function id -- no union scan needed here.
        const size_type n_nonzero = sparsity.count_nonzero();
        out.allocate(sparsity, K, ttg::scope::Allocate);
        out.set_ns();
        mra::apply_leaf_info(out, in_node);

        // No host zero-fill needed: entries the compute kernel doesn't visit
        // are finalized to 0 device-side instead -- convolution_kernel's own
        // tail pass for the unbatched path, convolution_scatter_sparsity_kernel
        // for the batched path (see their comments in kernels/convolution.h).
        DenseTensor<T, 1> resnorms(N, ttg::scope::Allocate);
        T tol = truncate_tol(key, thresh, cell_min_width, truncate_mode);
        std::array<bool, 2> at = {true, key.level()>0}; // apply terms analogue in MADNESS
        // if (key.level() == 0) at[1] = false; // do not apply S at level 0

        auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K) * n_nonzero, TempScope);

        // std::cout << "MRA:: For Key: " << key << "\n the operators being passed are \n R\n" << op_data->ops[0]->R.current_view() << "\nand S: \n" << op_data->ops[0]->S.current_view() << std::endl;


#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(in_node.coeffs().buffer(), resnorms.buffer(),
                                        out.coeffs().buffer(), tmp);
        input.add(op_data->norms.buffer());
        for (Dimension d = 0; d < NDIM; ++d) {
          input.add(op_data->data[d]->R.buffer());
          input.add(op_data->data[d]->S.buffer());
        }
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

        auto in_node_view = in_node.coeffs().current_view();
        auto out_view = out.coeffs().current_view();

        auto opnorms_view = op_data->norms.current_view();
        auto transr = std::array{op_data->data[0]->R.current_view(), op_data->data[1]->R.current_view(), op_data->data[2]->R.current_view()};
        auto transs = std::array{op_data->data[0]->S.current_view(), op_data->data[1]->S.current_view(), op_data->data[2]->S.current_view()};
        // empty in node view
        auto empty_node = mra::FunctionsCompressedNode<T, NDIM>();
        auto empty_node_view = empty_node.coeffs().current_view();
        auto resnorms_view = resnorms.current_view();

#ifndef MRA_ENABLE_HOST
        if (enable_conv_batching) {
          // shell0's kernel roles: "in" is always the empty accumulator, "f" is the
          // node's own coefficients -- coop() must expose both in that order.
          // transr/transs/opnorms_view/tol/at also travel through coop() since
          // batching is unrestricted now (any level, any displacement), so a
          // batch mate may have entirely different operator data -- see the
          // batching-support comment on ConvolutionBatchArg in kernels/convolution.h.
          // out.coeffs() (the real tensor, not just its view) travels through
          // too, so the batch leader can read its sparsity and aggregate every
          // member's bytes into one pinned buffer + one H2D copy instead of
          // each member pushing its own via SparsityManager here. n_nonzero
          // (this member's own, computed above independent of batching)
          // travels through as well, so the leader can flatten every
          // member's own non-zero work items into one combined 1D launch
          // (see submit_convolution_batch_leader).
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(empty_node_view, in_node_view, out_view, resnorms_view, tmp,
                                                                  transr, transs, opnorms_view, tol, at, out.coeffs(),
                                                                  n_nonzero);
          // followers: the leader's batched launch already wrote our slice of out/resnorms.
          detail::submit_convolution_batch_leader<T, NDIM>(batch, *conv_pool, K, fac, total_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          auto sparseman = make_sparsity_manager(out);
          sparseman.populate_device_sparsity();
          // out_view's device sparsity is demoted inline, per position, by
          // convolution_kernel itself (via convolution_process_one) for any
          // function whose computed norm is exactly zero -- before the
          // host-only out.set_zero(i) loop below narrows the *host* side of
          // the same sparsity. When batching is enabled this is instead done
          // once for the whole batch by submit_convolution_batch_leader (see
          // convolution_prune_zero_norm_kernel_batched in kernels/convolution.h).
          submit_convolution_kernel<T, NDIM>(key, key-key, K, N, n_nonzero, fac, tol, /*in_node_view*/ empty_node_view,
                                              in_node_view, out_view, resnorms_view, transr, transs, opnorms_view,
                                              at, tmp.current_device_ptr(), ttg::device::current_stream());
        }

#ifndef MRA_ENABLE_HOST
        // wait for the norms to come back
        co_await ttg::device::wait(resnorms.buffer());
#endif // MRA_ENABLE_HOST

        /* check if the result norms are >0.0 and send an empty node otherwise */
        bool empty = true;
        auto resnorms_host_view = resnorms.view_on(ttg::device::Device::host());
        for (size_type i = 0; i < N; ++i) {
          if (resnorms_host_view[i] != 0.0) {
            empty = false;
          } else {
            // set function to zero
            out.set_zero(i);
          }
        }
        if (empty) {
          //std::cout << "SHELL0 " << key << " result is empty after applying shell 0 contribution, sending empty node" << std::endl;
          out.make_empty();
        }
      }

      send_out(key, std::move(out));

#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    }, ttg::edges(input), ttg::edges(to_shellN), "Shell0");


    /**
     * Recursive edges
     */
    ttg::Edge<detail::KeyPair<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> accumulate_node_recurse;
    ttg::Edge<detail::KeyPair<NDIM>, std::vector<detail::KeyPair<NDIM>>> accumulate_contribution_recurse;

    /**
     * Dispatch task receives the input node and the list of contributions to apply to it.
     * It sends the node and the contributions to the first shellN task, or sends the node directly
     * to the output if there are no contributions to apply.
     */
    auto accumulate_dispatch = ttg::make_tt<Space>(
      [=](const mra::Key<NDIM>& key,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node,
          const std::vector<detail::KeyPair<NDIM>>& contributions,
          const std::array<bool, mra::Key<NDIM>::num_children()>& child_empty) -> TASKTYPE {

#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto&& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
          return false; // to allow using send_out in an initializer list
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto&& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
          return false; // to allow using send_out in an initializer list
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

        /**
         * Feed empty nodes to adjust-leaf task if needed, to make sure they have sufficient inputs.
         */
        [&]<std::size_t... Is>(std::index_sequence<Is...>){
          return ((
            child_empty[Is] && send_out(key, mra::FunctionsCompressedNode<T, NDIM>{},
                                        std::integral_constant<std::size_t, 3+Is>{})
          ), ...);
        }(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(to_shellN, contribution_edge, down_to_accumulate_leaf_info),
         ttg::edges(accumulate_node_recurse, accumulate_contribution_recurse, accumulate_to_adjust_leaf,
                    adjust_leaf_edges[0], adjust_leaf_edges[1], adjust_leaf_edges[2], adjust_leaf_edges[3],
                    adjust_leaf_edges[4], adjust_leaf_edges[5], adjust_leaf_edges[6], adjust_leaf_edges[7]),
         "AccumulateDispatch");

    /**
     * Task that applies the contributions from non-zero shells, as determined on the way down the tree.
     * The task recurses over the list of contributions to apply them one by one and send the result to the next contribution task.
     * After the last contribution is applied, the result is sent to the output.
     *
     * NOTE: because we use coroutines we cannot outline most of the code and instead have to copy past it here.
     */
    auto accumulate_tt = ttg::make_tt<Space>(
      [&, K, fac, thresh, truncate_mode, cell_min_width, name, enable_conv_batching, conv_pool, total_functions](
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
      auto source = keypair.source;
      auto displacement = key - source;

      size_type N = fns->num_functions(key);

      //std::cout << "ACCUMULATE " << key << " in_node " << in_node << " applying contribution " << contribution
      //          << " with " << contribution_keys.size() << " contributions left" << std::endl;

      assert(!contribution_keys.empty());
      assert(contribution_keys.back() == keypair);
      assert(!contribution.empty());
      // we allow for invalid nodes here because we may have contributions for nodes that previously did not exist
      assert(in_node.invalid() || key == in_node.key());

      // remove the current key
      contribution_keys.pop_back();

      bool last_key = contribution_keys.empty();
      SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
      sparsity.nonzero_if_any(in_node, contribution);
      // out's own sparsity (built below) is exactly this OR, so a single
      // on-device scan of out's sparsity (find_nth_nonzero) suffices to
      // recover each real function id -- no union scan needed here.
      const size_type n_nonzero = sparsity.count_nonzero();

      mra::FunctionsCompressedNode<T, NDIM> out(key, N);

      auto op_data = op.get_op(key.level(), displacement);

      out.allocate(sparsity, K, ttg::scope::Allocate);

      out.set_ns();
      mra::apply_leaf_info(out, in_node);

      DenseTensor<T, 1> resnorms;
      const double tol = truncate_tol(key, thresh, cell_min_width, truncate_mode);
      std::array<bool, 2> at = {true, source.level()>0}; // apply terms analogue in MADNESS

      auto tmp = ttg::Buffer<T>(convolution_tmp_size<NDIM>(K) * n_nonzero, TempScope);

      // std::cout << "MRA:: For Key: " << key << "\n the operators being passed are \n R\n" << op_data->ops[0]->R.current_view() << "\nand S: \n" << op_data->ops[0]->S.current_view() << std::endl;

      if (last_key) {
        // No host zero-fill needed: entries the compute kernel doesn't visit
        // are finalized to 0 device-side instead -- convolution_kernel's own
        // tail pass for the unbatched path, convolution_scatter_sparsity_kernel
        // for the batched path (see their comments in kernels/convolution.h).
        resnorms = DenseTensor<T, 1>(N, ttg::scope::Allocate);
      }
#ifndef MRA_ENABLE_HOST
      auto input = ttg::device::Input(in_node.coeffs().buffer(), out.coeffs().buffer(), contribution.coeffs().buffer(), tmp);
      input.add(op_data->norms.buffer());
      for (Dimension d = 0; d < NDIM; ++d) {
        input.add(op_data->data[d]->R.buffer());
        input.add(op_data->data[d]->S.buffer());
      }
      if (last_key) {
        // if this is the last we want to get the norms of the result back
        input.add(resnorms.buffer());
      }
      co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

      auto transr = std::array{op_data->data[0]->R.current_view(), op_data->data[1]->R.current_view(), op_data->data[2]->R.current_view()};
      auto transs = std::array{op_data->data[0]->S.current_view(), op_data->data[1]->S.current_view(), op_data->data[2]->S.current_view()};

      auto opnorms_view = op_data->norms.current_view();
      auto out_view = out.coeffs().current_view();
      auto contribution_view = contribution.coeffs().current_view();
      auto in_node_view = in_node.coeffs().current_view();
      auto resnorms_view = resnorms.current_view();

#ifndef MRA_ENABLE_HOST
      if (enable_conv_batching) {
        // transr/transs/opnorms_view/tol/at also travel through coop() since
        // batching is unrestricted now (any level, any displacement), so a
        // batch mate may have entirely different operator data -- see the
        // batching-support comment on ConvolutionBatchArg in kernels/convolution.h.
        // out.coeffs() (the real tensor, not just its view) travels through
        // too, so the batch leader can read its sparsity and aggregate every
        // member's bytes into one pinned buffer + one H2D copy instead of
        // each member pushing its own via SparsityManager here. n_nonzero
        // (this member's own, computed above independent of batching)
        // travels through as well, so the leader can flatten every
        // member's own non-zero work items into one combined 1D launch
        // (see submit_convolution_batch_leader).
        auto batch = co_await ttg::device::coop<detail::KeyPair<NDIM>>(in_node_view, contribution_view, out_view, resnorms_view, tmp,
                                                                       transr, transs, opnorms_view, tol, at, out.coeffs(),
                                                                       n_nonzero);
        // followers: the leader's batched launch already wrote our slice of out/resnorms.
        detail::submit_convolution_batch_leader<T, NDIM>(batch, *conv_pool, K, fac, total_functions);
      } else
#endif // MRA_ENABLE_HOST
      {
        auto sparseman = make_sparsity_manager(out);
        sparseman.populate_device_sparsity();
        // out_view's device sparsity is demoted inline, per position, by
        // convolution_kernel itself (via convolution_process_one) for any
        // function whose computed norm is exactly zero -- gated the same way
        // as resnorms itself (empty unless last_key, see above) -- before the
        // host-only out.set_zero(i) loop below (only reached when last_key)
        // narrows the *host* side of the same sparsity. When batching is
        // enabled this is instead done once for the whole batch by
        // submit_convolution_batch_leader (see
        // convolution_prune_zero_norm_kernel_batched in kernels/convolution.h).
        submit_convolution_kernel<T, NDIM>(key, displacement, K, N, n_nonzero, fac, tol, in_node_view,
                                            contribution_view, out_view, resnorms_view, transr, transs,
                                            opnorms_view, at,
                                            tmp.current_device_ptr(), ttg::device::current_stream());
      }

#ifndef MRA_ENABLE_HOST
      // wait for norms to come back
      if (last_key) {
        co_await ttg::device::wait(resnorms.buffer());
      }
#endif // MRA_ENABLE_HOST



      if (last_key) {
        bool empty = true;
        auto resnorms_host_view = resnorms.view_on(ttg::device::Device::host());
        for (size_type i = 0; i < N; ++i) {
          //std::cout << "ACCUMULATE " << key << " result norm for function " << i << ": " << resnorms_host_view(i) << std::endl;
          if (resnorms_host_view(i) != 0.0) {
            assert(out.is_nonzero(i) && "if the norm is non-zero we should have non-zero coefficients");
            empty = false;
          } else {
            // TODO: should we allow modifying the sparsity on the host?
            out.set_zero(i);
          }
        }

        /**
         * Drop the coefficients if the node is empty to save memory.
         */
        if (empty) {
          out.make_empty(); // drop the memory but keep child info
        }

        // if this was the last contribution to apply, send the result to the output
        ttg::trace(name, key, ": last contribution, node empty: ", empty);
        //std::cout << "ACCUMULATE " << key << " last contribution applied, sending result to output" << out << std::endl;
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
         ttg::edges(accumulate_node_recurse, accumulate_contribution_recurse, accumulate_to_adjust_leaf),
         "Accumulate");

    /***************************************************************************************
     * Task that receives the final output node and its children (or an empty node) and
     * adjusts the leaf status of the node based on whether its children have data.
     * Sends the result to the final output.
     ***************************************************************************************/
    auto adjust_leaf_info_tt = ttg::make_tt<Space>(
      [=](const Key<NDIM>& key,
          FunctionsCompressedNode<T, NDIM>&& node,
          const mra::FunctionsCompressedNode<T, NDIM>& child0,
          const mra::FunctionsCompressedNode<T, NDIM>& child1,
          const mra::FunctionsCompressedNode<T, NDIM>& child2,
          const mra::FunctionsCompressedNode<T, NDIM>& child3,
          const mra::FunctionsCompressedNode<T, NDIM>& child4,
          const mra::FunctionsCompressedNode<T, NDIM>& child5,
          const mra::FunctionsCompressedNode<T, NDIM>& child6,
          const mra::FunctionsCompressedNode<T, NDIM>& child7) -> TASKTYPE {
        ttg::trace("AdjustLeafInfo", key, "is_all_child_leaf", node.is_all_child_leaf());

        assert(!node.invalid() && "we should have received a valid node from the accumulate task");


#ifndef MRA_ENABLE_HOST
        auto sends = ttg::device::forward();
        auto send_out = [&]<std::size_t I, typename S>(auto&& k, S&& out, std::integral_constant<std::size_t, I>){
          sends.push_back(ttg::device::send<I>(k, std::forward<S>(out)));
        };
#else
        auto send_out = [&]<std::size_t I, typename S>(auto&& k, S&& out, std::integral_constant<std::size_t, I>){
          ttg::send<I>(k, std::forward<S>(out));
        };
#endif

        size_type N = fns->num_functions(key);
        constexpr size_type num_children = mra::Key<NDIM>::num_children();
        constexpr std::size_t result_terminal = mra::Key<NDIM>::num_children();
        std::array<const mra::FunctionsCompressedNode<T, NDIM>*, num_children> children
                                    = {&child0, &child1, &child2, &child3, &child4, &child5, &child6, &child7};


        //for (auto child : mra::children(key)) {
        //  std::cout << "ADJUST LEAF INFO " << key << " child " << child << " " << *children[child.childindex()] << std::endl;
        //}

        /**
         * For each function in each child, check if the children of the child are leafs and the child itself is empty.
         * If so, mark the child as a leaf in the node.
         */
        for (size_type i = 0; i < N; ++i) {
          for (size_type c = 0; c < num_children; ++c) {
            bool is_child_leaf = false;
            if (children[c]->invalid() || (children[c]->is_all_child_leaf(i) && children[c]->is_zero(i))) {
              is_child_leaf = true;
            }
            node.set_child_leaf(i, c, is_child_leaf);
          }
        }

        //std::cout << "ADJUST LEAF INFO " << key << " after adjustment node " << node << std::endl;

        if (key.level() > 0) {
          if (!(node.is_all_child_leaf() && node.is_all_zero())) {
            /**
             * Broadcast the node to the parent and the result (have to select the right output terminal).
             * We send the node to the result only if it is not empty and has no children.
             */
            [&]<std::size_t... I>(std::index_sequence<I...>){
              auto bcast = [&]<std::size_t J>(){
#ifndef MRA_ENABLE_HOST
                sends.push_back(ttg::device::broadcast<J, result_terminal>(std::make_tuple(key.parent(), key),
                                                                           std::move(node)));
#else
                ttg::broadcast<J, result_terminal>(std::make_tuple(key.parent(), key),
                                                   std::move(node));
#endif
                return true;
              };
              ((
                (I == key.childindex() ? bcast.template operator()<I>() : false)
              ), ...);
            }(std::make_index_sequence<result_terminal>{});
          } else {

            /**
             * If the node is empty and all children are leafs, we just send it up to the parent, but not to the result.
             * This way we are dropping empty nodes on the way up.
             * The lambda below helps us select the right output terminal based on which child we are.
             */
            [&]<std::size_t... I>(std::index_sequence<I...>){
              auto do_send = [&]<std::size_t J>(){
                send_out(key.parent(), std::move(node), std::integral_constant<std::size_t, J>{});
                return true;
              };
              ((
                (I == key.childindex() ? do_send.template operator()<I>() : false)
              ), ...);
            }(std::make_index_sequence<num_children>{});
          }
        } else {
          // if we are the root we have no parent to send to, so we send the result directly to the output
          send_out(key, std::move(node), std::integral_constant<std::size_t, result_terminal>{});
        }

#ifndef MRA_ENABLE_HOST
          co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(accumulate_to_adjust_leaf,
                    adjust_leaf_edges[0], adjust_leaf_edges[1], adjust_leaf_edges[2], adjust_leaf_edges[3],
                    adjust_leaf_edges[4], adjust_leaf_edges[5], adjust_leaf_edges[6], adjust_leaf_edges[7]),
         ttg::edges(adjust_leaf_edges[0], adjust_leaf_edges[1], adjust_leaf_edges[2], adjust_leaf_edges[3],
                    adjust_leaf_edges[4], adjust_leaf_edges[5], adjust_leaf_edges[6], adjust_leaf_edges[7],
                    result),
         "AdjustLeafInfo");



#if 0

    /****************************************************************************************************************************
     * Task that receives a node (either from shell0 if it exists already, or from down task if its new) and the leaf status of
     * its children, adjusts the leaf status based on whether it has children or receives contributions, and sends it to shellN.
     ****************************************************************************************************************************/
    auto adjust_leaf_tt = ttg::make_tt<Space>(
      [=](const Key<NDIM>& key, FunctionsCompressedNode<T, NDIM>&& node, const std::array<bool, 8>& child_info) -> TASKTYPE {

        size_type N = fns->num_functions(key);

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

        // TODO: drop the per-function leaf info from nodes
        for (auto child : children(key)) {
          node.set_child_empty(child.childindex(), child_info[child.childindex()]);
        }

        send_out(key, std::move(node), std::integral_constant<std::size_t, 0>{});

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(shell0_to_shellN, down_to_adjust_leaf_child_info), ttg::edges(to_shellN), "AdjustLeaf");

    /***************************************************************************************
     * Task that dispatches the result to adjust_parent, filling the LeafStatus inputs
     * for leaf nodes.
     * TODO: this task should be inlined somehow
     ***************************************************************************************/

    std::array<ttg::Edge<Key<NDIM>, bool>, num_children> adjust_parent_edges;

    auto dispatch_adjust_parent_tt = ttg::make_tt<Space>(
      [=](const Key<NDIM>& key, const FunctionsCompressedNode<T, NDIM>& node) -> TASKTYPE {
        ttg::trace("DispatchAdjustParent", key, "is_all_child_empty", node.is_all_child_empty());
        size_type N = fns->num_functions(key);
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

        detail::foreach_child(key, [&]<std::size_t I>(const Key<NDIM>& child){
          ttg::trace("DispatchAdjustParent", key, "checking child ", child, " is leaf or invalid ", node.is_child_empty(child));
          if (node.is_child_empty(child)) {
            detail::LeafInfo leaf_info(N);
            auto leaf_info_view = leaf_info.host_view();
            for (int n = 0; n < N; ++n) {
              // if we are zero we are invalid, otherwise we are a leaf
              leaf_info_view(n) = node.is_zero(n) ? LeafStatus::Invalid : LeafStatus::Leaf;
            }
            //std::cout << "DISPATCH ADJUST PARENT " << key << " child " << child << " is leaf, sending true" << std::endl;
            send_out(key, std::move(leaf_info), std::integral_constant<std::size_t, I>{});
          }
          send_out(key, node.is_child_empty(child), std::integral_constant<std::size_t, I>{});
        });

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(accumulate_result),
         ttg::edges(adjust_parent_edges[0], adjust_parent_edges[1], adjust_parent_edges[2], adjust_parent_edges[3],
                    adjust_parent_edges[4], adjust_parent_edges[5], adjust_parent_edges[6], adjust_parent_edges[7]),
         "DispatchAdjustParent");

    /***************************************************************************************
     * Task that receives the result node and LeafInfo from their children indicating
     * whether the child is empty. This allows us to adjust the child-leaf info of the
     * parent after all contributions have been accumulated.
     ***************************************************************************************/
    auto adjust_parent_tt = ttg::make_tt<Space>(
      [=](const Key<NDIM>& key, FunctionsCompressedNode<T, NDIM>&& node,
          bool child0, bool child1,
          bool child2, bool child3,
          bool child4, bool child5,
          bool child6, bool child7) -> TASKTYPE {
        // this task receives the leaf status of our children from the down task and sends it to our parent so that it can adjust its contribution count if necessary
        auto child_info = std::array{child0, child1, child2, child3,
                                     child4, child5, child6, child7};

        size_type N = fns->num_functions(key);

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

        for (size_t i = 0; i < num_children; ++i) {
          node.set_child_empty(i, child_info[i]);
        }

        assert(node.is_all_child_empty() || !node.invalid());

        bool empty = false;
        /**
         * If our children are all empty we may be empty ourselves, if we're invalid or all zero.
         * If our children are not all empty we cannot mark ourselves as empty.
         */
        if (node.is_all_child_empty() && (node.invalid() || node.empty() || node.is_all_zero())) {
          empty = true;
        }

        //std::cout << "ADJUST PARENT " << key << " is empty " << empty << std::endl;
        if (key.level() > 0) {
#ifndef MRA_ENABLE_HOST
          sends.push_back(select_send_up(key, std::move(empty), std::make_index_sequence<num_children>{}, "adjust_parent"));
#else  // MRA_ENABLE_HOST
          select_send_up(key, std::move(empty), std::make_index_sequence<num_children>{}, "adjust_parent");
#endif // MRA_ENABLE_HOST

        }

        // we drop the node if it is empty, the parent will adjust its child-leaf info
        // accordingly
        if (!empty) {
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::send<8>(key, std::move(node));
#else
          ttg::send<8>(key, std::move(node));
#endif // MRA_ENABLE_HOST
        }
      }, ttg::edges(accumulate_result,
                    adjust_parent_edges[0], adjust_parent_edges[1],
                    adjust_parent_edges[2], adjust_parent_edges[3],
                    adjust_parent_edges[4], adjust_parent_edges[5],
                    adjust_parent_edges[6], adjust_parent_edges[7]),
         ttg::edges(adjust_parent_edges[0], adjust_parent_edges[1],
                    adjust_parent_edges[2], adjust_parent_edges[3],
                    adjust_parent_edges[4], adjust_parent_edges[5],
                    adjust_parent_edges[6], adjust_parent_edges[7],
                    result), "AdjustParent");
#endif // 0

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      up_contributions_tt->set_keymap(procmap);
      down_contributions_tt->set_keymap(procmap);
      norm_tt->set_keymap(procmap);
      screener_tt->set_keymap(procmap);
      //neighbor_dispatch_tt->set_keymap(procmap);
      //rebalance_down->set_keymap(procmap);
      shell0_tt->set_keymap(procmap);
      adjust_leaf_info_tt->set_keymap(procmap);
      accumulate_dispatch->set_keymap(procmap);
      accumulate_tt->set_keymap([=](const detail::KeyPair<NDIM>& kp) { return procmap(kp.dest); });
      //dispatch_adjust_parent_tt->set_keymap(procmap);
      //adjust_parent_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      up_contributions_tt->set_devicemap(devicemap);
      down_contributions_tt->set_devicemap(devicemap);
      norm_tt->set_devicemap(devicemap);
      screener_tt->set_devicemap(devicemap);
      //neighbor_dispatch_tt->set_devicemap(devicemap);
      //rebalance_down->set_devicemap(devicemap);
      shell0_tt->set_devicemap(devicemap);
      adjust_leaf_info_tt->set_devicemap(devicemap);
      accumulate_dispatch->set_devicemap(devicemap);
      accumulate_tt->set_devicemap([=](const detail::KeyPair<NDIM>& kp) { return devicemap(kp.dest); });
      //dispatch_adjust_parent_tt->set_devicemap(devicemap);
      //adjust_parent_tt->set_devicemap(devicemap);
    }

#ifndef MRA_ENABLE_HOST
    if (enable_conv_batching) {
      // Unrestricted matchers: any two tasks of the same TT may batch
      // together, regardless of level or displacement, up to max_batch_size.
      // Level-only matching (an earlier, narrower version of this) still made
      // accumulate_tt's batches mostly size 1 in practice (two tasks rarely
      // reach even the same level at the same moment, let alone the same
      // displacement). Everything that could differ across levels/displacements
      // -- tol, at, transr, transs, opnorms -- now travels per-member (see
      // ConvolutionBatchArg), so nothing here needs to be verified equal
      // between head and cand; the cost is a few hundred extra bytes of view/
      // scalar descriptors per member, not the underlying filter-matrix data,
      // which TensorView only points to.
      shell0_tt->set_batch_matcher(
          [](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
          max_batch_size);

      accumulate_tt->set_batch_matcher(
          [](const detail::KeyPair<NDIM>&, const detail::KeyPair<NDIM>&) { return true; },
          max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    // TODO: assemble TTG

    auto ins = std::make_tuple(screener_tt->template in<0>());
    auto outs = std::make_tuple(adjust_leaf_info_tt->template out<2>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(8);
    ops[0] = std::move(up_contributions_tt);
    ops[1] = std::move(down_contributions_tt);
    ops[2] = std::move(norm_tt);
    ops[3] = std::move(screener_tt);
    ops[4] = std::move(shell0_tt);
    ops[5] = std::move(adjust_leaf_info_tt);
    ops[6] = std::move(accumulate_dispatch);
    ops[7] = std::move(accumulate_tt);
    //ops[8] = std::move(dispatch_adjust_parent_tt);
    //ops[9] = std::move(adjust_parent_tt);

    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_CONVOLUTION_H
