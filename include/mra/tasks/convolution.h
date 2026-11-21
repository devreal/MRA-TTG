#ifndef MRA_TASKS_CONVOLUTION_H
#define MRA_TASKS_CONVOLUTION_H

#include <iostream>
#include <optional>
#include <sstream>
#include <atomic>
#include <cstdlib>
#include <numeric>
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

    /**
     * The operator norms of the functions of a batch starting at function `offset` of the set:
     * with one operator per function, the view from the batch's first operator on, since the
     * kernels index the operators by the function's position in the node (one operator for all
     * functions: unchanged).
     */
    template <typename View>
    View batch_opnorms(const View& opnorms, size_type offset) {
      if (offset == 0 || opnorms.dim(0) == 1) return opnorms;
      if (offset >= opnorms.dim(0)) throw std::runtime_error("convolution: fewer operators than functions");
      const size_type stride = opnorms.dim(1) * opnorms.dim(2) * opnorms.dim(3);
      return View(opnorms.data() + offset * stride, opnorms.dim(0) - offset, opnorms.dim(1), opnorms.dim(2), opnorms.dim(3));
    }
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

    /**
     * Key of a chunk of the contributions to one destination node (make_convolution's Apply
     * task): the contributions of `dest` are split into `nchunks` chunks, this is chunk `chunk`
     * with `size` contributions. The counts travel in the key because the chunk's aggregator
     * needs its target when the first value arrives.
     */
    template<Dimension NDIM>
    struct ChunkKey {
      Key<NDIM> dest;
      std::uint32_t chunk = 0;
      std::uint32_t size = 0;
      std::uint32_t nchunks = 0;

      bool operator==(const ChunkKey&) const = default;
      auto hash() const {
        HashValue hashvalue = mulhash(dest.hash(), chunk);
        return hashvalue;
      }

      template <typename Archive>
      void serialize(Archive& ar) {
        ar & dest & chunk & size & nchunks;
      }

      template <typename Archive>
      void serialize(Archive& ar, const unsigned int) {
        serialize(ar);
      }
    };

    template<Dimension NDIM>
    std::ostream& operator<<(std::ostream& os, const ChunkKey<NDIM>& ck) {
      return os << "(" << ck.dest << " chunk " << ck.chunk << "/" << ck.nchunks << " size " << ck.size << ")";
    }

    /** Key of the sum of the `nchunks` partial results of `dest` and its shell-0 result (Sum task) */
    template<Dimension NDIM>
    struct SumKey {
      Key<NDIM> dest;
      std::uint32_t nchunks = 0;

      bool operator==(const SumKey&) const = default;
      auto hash() const { return dest.hash(); }

      template <typename Archive>
      void serialize(Archive& ar) {
        ar & dest & nchunks;
      }

      template <typename Archive>
      void serialize(Archive& ar, const unsigned int) {
        serialize(ar);
      }
    };

    template<Dimension NDIM>
    std::ostream& operator<<(std::ostream& os, const SumKey<NDIM>& sk) {
      return os << "(" << sk.dest << " sum of " << sk.nchunks << ")";
    }

    /* At most 2*C + 9 buffers per chunk task (C sources and their operator norms, the level's
     * R/S, the shared assembly tables and norms1d, the result, a node's own shell-0 result and
     * the result norms) must fit into MAX_PARAM_COUNT (32) flows. */
    inline constexpr std::size_t conv_max_chunk_size = 11;
    /* a Sum task selects its result, the result norms, the shell-0 result and the partial sums */
    inline constexpr std::size_t conv_max_sum_parts = 28;

    /**
     * Contributions per chunk task (MRA_TTG_CONV_CHUNK, default 8, at most conv_max_chunk_size);
     * 0 applies the contributions to a node one task at a time, in a chain. Host builds always
     * use the chain.
     */
    inline std::size_t conv_chunk_size() {
#ifdef MRA_ENABLE_HOST
      return 0;
#else
      static const std::size_t c = [] {
        const char* v = std::getenv("MRA_TTG_CONV_CHUNK");
        if (v == nullptr || v[0] == '\0') return std::size_t{8};
        return std::min<std::size_t>(std::strtoul(v, nullptr, 10), conv_max_chunk_size);
      }();
      return c;
#endif
    }

    /** scratch memory for the contributions of one batch of chunk tasks (MRA_TTG_CONV_CHUNK_SCRATCH_MB, default 256) */
    inline std::size_t conv_chunk_scratch_bytes() {
      static const std::size_t b = [] {
        const char* v = std::getenv("MRA_TTG_CONV_CHUNK_SCRATCH_MB");
        std::size_t mb = (v == nullptr || v[0] == '\0') ? 256 : std::strtoul(v, nullptr, 10);
        return std::max<std::size_t>(mb, 1) << 20;
      }();
      return b;
    }

    /**
     * Distribution of the number of contributions per destination node of one convolution graph,
     * printed when the graph is destroyed. Enabled with MRA_TTG_CONV_STATS=1; diagnostic only.
     */
    struct ConvContributionStats {
      static constexpr std::size_t max_count = 4096;
      std::string name;
      size_type num_functions;
      std::vector<std::atomic<std::uint64_t>> hist;
      std::array<std::atomic<std::uint64_t>, 32> per_level_nodes{};
      std::array<std::atomic<std::uint64_t>, 32> per_level_contribs{};

      ConvContributionStats(std::string name, size_type num_functions)
      : name(std::move(name)), num_functions(num_functions), hist(max_count + 1) {}

      static bool enabled() {
        static const bool e = [] {
          const char* v = std::getenv("MRA_TTG_CONV_STATS");
          return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        return e;
      }

      void record(Level level, std::size_t n) {
        hist[std::min(n, max_count)].fetch_add(1, std::memory_order_relaxed);
        const std::size_t l = std::min<std::size_t>(level, 31);
        per_level_nodes[l].fetch_add(1, std::memory_order_relaxed);
        per_level_contribs[l].fetch_add(n, std::memory_order_relaxed);
      }

      ~ConvContributionStats() {
        std::uint64_t nodes = 0, total = 0, maxn = 0;
        for (std::size_t i = 0; i <= max_count; ++i) {
          const auto c = hist[i].load();
          nodes += c; total += c * i;
          if (c) maxn = i;
        }
        if (nodes == 0) return;
        auto quantile = [&](double q) {
          std::uint64_t acc = 0;
          const auto goal = static_cast<std::uint64_t>(q * nodes);
          for (std::size_t i = 0; i <= max_count; ++i) {
            acc += hist[i].load();
            if (acc > goal) return i;
          }
          return max_count;
        };
        std::ostringstream os;
        os << "MRA CONV STATS " << name << " functions " << num_functions << " dest nodes " << nodes
           << " contributions " << total << " mean " << double(total) / nodes << " p50 " << quantile(0.5)
           << " p90 " << quantile(0.9) << " p99 " << quantile(0.99) << " max " << maxn << " | per level (nodes:mean)";
        for (std::size_t l = 0; l < 32; ++l) {
          const auto nn = per_level_nodes[l].load();
          if (nn) os << " " << l << "=" << nn << ":" << double(per_level_contribs[l].load()) / nn;
        }
        os << "\n";
        std::cout << os.str() << std::flush;
      }
    };

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

    /**
     * True iff two sparsity sets mark the exact same function indices
     * non-zero. Used by accumulate_tt to decide whether its newly-computed
     * output sparsity exactly matches its input node's own -- if so, the
     * input node's already-allocated storage has the right shape to be
     * reused as the output in place (see accumulate_tt's comment), instead
     * of allocating a fresh tensor. No existing comparator for this in the
     * tensor/sparsity headers (SparsityInfo and a node's own sparsity() are
     * different RangeSparsityBase instantiations with no operator== between
     * them), so compare through the common is_nonzero()/count() interface
     * both expose.
     */
    template <typename SparsityA, typename SparsityB>
    bool sparsity_equal(const SparsityA& a, const SparsityB& b) {
      if (a.count() != b.count() || a.count_nonzero() != b.count_nonzero()) return false;
      for (size_type i = 0; i < a.count(); ++i) {
        if (a.is_nonzero(i) != b.is_nonzero(i)) return false;
      }
      return true;
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
      conv_pool = detail::shared_pool_registry<detail::ConvolutionBatchArg<T, NDIM>>(mra::get_batch_size());
    }
    // Separate pool/registry for norm_tt's simple_norm kernel: its own
    // SimpleNormBatchArg<T,NDIM> element type differs from
    // ConvolutionBatchArg<T,NDIM>, so it cannot share conv_pool -- see
    // mra/kernels/simple_norm.h's batching-support comment. Driven by the
    // same process-wide enable_conv_batching/max_batch_size, since norm_tt
    // is unrestricted-batchable exactly like the other tasks here.
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::SimpleNormBatchArg<T, NDIM>>> simple_norm_pool;
    if (enable_conv_batching) {
      simple_norm_pool = detail::shared_pool_registry<detail::SimpleNormBatchArg<T, NDIM>>(mra::get_batch_size());
    }
#else
    // GroupedBatchPoolRegistry only exists on device builds; this placeholder only
    // exists so the (shared host/device) task lambdas below can
    // unconditionally list conv_pool in their capture list -- it is never
    // accessed on host builds.
    std::nullptr_t conv_pool = nullptr;
    std::nullptr_t simple_norm_pool = nullptr;
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

    /**
     * Chunked contributions (conv_chunk_size() > 0): the screener sends each source node to a
     * Route task per (source, dest) pair, which forwards it to the chunk of dest that
     * AccumulateDispatch assigned to the pair. An Apply task convolves its chunk's sources in
     * parallel and adds them up; a node with several chunks has its partial sums and its
     * shell-0 result added by a Sum task, a node with one chunk gets its shell-0 result added
     * by the Apply task itself.
     */
    const std::size_t chunk_size = detail::conv_chunk_size();
    ttg::Edge<detail::KeyPair<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> screener_to_route;
    ttg::Edge<detail::KeyPair<NDIM>, detail::ChunkKey<NDIM>> route_chunk_edge;
    ttg::Edge<detail::ChunkKey<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> apply_in;
    ttg::Edge<detail::SumKey<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> sum_in;

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
          cnorms = DenseTensor<T, 1>(N, ttg::scope::SyncOut);
#ifndef MRA_ENABLE_HOST
          co_await ttg::device::select(in_node.buffer(), cnorms.buffer());
#endif

          auto node_view = in_node.coeffs().current_view();
          auto result_view = cnorms.current_view();

#ifndef MRA_ENABLE_HOST
          if (enable_conv_batching) {
            // simple_norm has no operator data at all, so batching here is
            // unrestricted -- see mra/kernels/simple_norm.h's
            // batching-support comment. No upfront nonzero detection either:
            // the leader launches N blocks per member, same as the
            // unbatched path just below.
            auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(node_view, result_view, N);
            detail::submit_simple_norm_batch_leader<T, NDIM>(batch, *simple_norm_pool);
          } else
#endif // MRA_ENABLE_HOST
          {
            // submit_simple_norm_kernel's `in` parameter only binds rvalues,
            // so pass fresh views here rather than the named node_view/
            // result_view above (those exist only so coop(), in the
            // batching branch, has lvalues that survive the co_await
            // suspension).
            submit_simple_norm_kernel(key, in_node.coeffs().current_view(), N, cnorms.current_view());
          }

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
      [&, K, thresh, truncate_mode, cell_min_width, fac, name, chunk_size, up_tt_ptr = up_contributions_tt.get(), down_tt_ptr = down_contributions_tt.get()](
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

#ifndef MRA_ENABLE_HOST
          /**
           * A device task may co_await ttg::device::select() only ONCE over its
           * whole execution, but this task needs a DIFFERENT Opnorm per
           * mad_disp in the loop below -- one select() per displacement is not
           * an option. Instead, obtain (and if need be generate) this level's
           * shared 1D operator table ONCE here, up front: generation is
           * level-wide, not per-displacement. Once the table exists, each
           * displacement's Opnorm is computed host-only from the table's
           * norms1d (already current on the host, pulled back by whoever
           * generated the table) -- so the loop below never needs another
           * select(). The screener never claims the device-side norms tensors
           * that shell0_tt/accumulate_tt consume (see try_get_opnorms()).
           */
          // If we generate the table ourselves, its publication is deferred
          // until this task completes (see below), so the per-displacement
          // lookups below take it explicitly.
          typename mra::GaussianConvolutionOperator<T, NDIM>::table_pointer_type level_table = nullptr;
          for (;;) {
            auto lt = op.try_get_level_table(key.level());
            if (lt.pending) {
              // a pure reschedule point, no kernel of ours to wait for
              co_await ttg::device::suspend();
              continue;
            }
            if (lt.level_work) {
              /**
               * We need to generate the transition matrices for this level's operator.
               */
              auto gen_items = op.collect_generation_items(lt.level_work);
              auto items_buf = mra::make_generate_op1d_items_buffer(gen_items);
              auto gen_input = op.make_generation_input(lt.level_work, items_buf);
              co_await ttg::device::select(gen_input);
              op.submit_generation_kernel(lt.level_work, gen_items, items_buf);
              // wait for the kernel and pull the table's norms back to the host
              // for try_get_opnorms(); R/S are only ever used on the device
              co_await ttg::device::wait(lt.level_work->norms1d.buffer());
              // Publish only once this task has completed: until then the
              // runtime still holds our (written) device copies of the table,
              // and other tasks must not stage them in.
              sends.push_back(ttg::device::on_complete(
                [opp = &op, level_work = lt.level_work]() { opp->publish_generation_work(level_work); }));
            }
            level_table = lt.table;
            break;
          }
#endif // MRA_ENABLE_HOST

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
#ifndef MRA_ENABLE_HOST
          // Opnorms per position in mad_disps. With a single operator every
          // function walks the same displacement list, so each displacement's
          // Opnorms (one per operator index) are looked up once per task
          // rather than once per function.
          std::vector<typename mra::GaussianConvolutionOperator<T, NDIM>::opnorms_pointer_type> disp_opnorms;
#endif // MRA_ENABLE_HOST
          const size_type fn_offset = fns->global_offset(key);
          for (int i = 0; i < N; ++i) {
            // we may have either one operator for all functions or one operator per function
            // (by the function's index in the whole set: the key's batch starts at fn_offset)
            int opnorm_index = (op.count() == 1) ? 0 : fn_offset + i;
            // safe because get_disp returns a reference that remains valid
            const auto& mad_disps = op.get_mad_displacements(opnorm_index, key.level());
            const auto& mad_op = op.get_mad_op(opnorm_index);
#ifndef MRA_ENABLE_HOST
            if (op.count() != 1 || disp_opnorms.empty()) {
              disp_opnorms.assign(mad_disps.size(), nullptr);
            }
#endif // MRA_ENABLE_HOST
            int nused = 1, nvalid = 1;
            std::optional<double> real_last_distsq;
            std::optional<std::uint64_t> lattice_last_distsq;
            for (std::size_t disp_idx = 0; disp_idx < mad_disps.size(); ++disp_idx) {
              const auto& mad_disp = mad_disps[disp_idx];

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

#ifdef MRA_ENABLE_HOST
              auto op_data = op.get_op(key.level(), disp_key);
              auto opnorm_view = op_data->norms.view_on(ttg::device::Device::host());
              auto opnorm = opnorm_view(opnorm_index, 0, 0, (int)NormId::Opnorm);
#else
              // Host-only (the level's table is already known, see above):
              // only Pending -- another screener computing the same Opnorms
              // right now -- makes us wait.
              auto& opnorms = disp_opnorms[disp_idx];
              while (opnorms == nullptr) {
                opnorms = op.try_get_opnorms(key.level(), disp_key, level_table);
                if (opnorms == nullptr) co_await ttg::device::suspend();
              }
              auto opnorm = (*opnorms)[opnorm_index];
#endif // MRA_ENABLE_HOST

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
          // broadcast the input node to the accumulate (chain) or route (chunked) tasks
          if (chunk_size > 0) {
            sends.push_back(ttg::device::broadcast<2>(std::move(contributions), in_node));
          } else {
            sends.push_back(ttg::device::broadcast<0>(std::move(contributions), in_node));
          }
        }
        co_await std::move(sends);
#else
        if (!contributions.empty()) {
          ttg::broadcast<0>(std::move(contributions), in_node); // broadcast the input node to the accumulate tasks
        }
#endif // MRA_ENABLE_HOST
      },
      ttg::edges(input, norm_edge), ttg::edges(screener_to_accumulate, up_contribution_edge, screener_to_route), "Screen");

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
        mra::Key<NDIM> op_disp(key.batch(), key.level(), {0, 0, 0});
#ifdef MRA_ENABLE_HOST
        auto op_data = op.get_op(key.level(), op_disp);
#else
        // Survives the loop so res.dims/res.norms are usable afterward --
        // R/S (res.dims) are cheap enough to rebuild fresh on every
        // try_get_op() call that there's no separate cached "aggregate" to
        // hold onto; only the norms tensor is cached (see OpResult's comment
        // in conv_mad.h), and res.norms carries that once assembled/found.
        //
        // A device task may co_await ttg::device::select() only ONCE, so we
        // spin here (host-only, before that select()) until the operator is
        // either Available or ours to build. In the latter case the
        // generation/assembly buffers ride along in the convolution's own
        // select() below and their kernels are queued ahead of it.
        using OpResult = typename mra::GaussianConvolutionOperator<T, NDIM>::OpResult;
        using OpStatus = typename mra::GaussianConvolutionOperator<T, NDIM>::Status;
        OpResult res;
        for (;;) {
          res = op.try_get_op(key.level(), op_disp);
          if (res.status != OpStatus::Pending) break;
          if (op.has_generation_work(res)) {
            // Owning the level table implies nobody else could have claimed
            // this displacement's norms yet (claiming requires the table to
            // be owned or available) -- see try_get_op().
            throw std::runtime_error("Shell0: own level table generation but norms assembly is pending elsewhere");
          }
          co_await ttg::device::suspend();
        }
        const bool need_generate = op.has_generation_work(res);
        const bool need_assemble = (res.status == OpStatus::Owned);
        // All host-side (safe before the select(), see has_generation_work()/
        // make_norms_tensor()'s comments in conv_mad.h).
        auto gen_items = op.collect_generation_items(res.level_work); // empty if !need_generate
        auto items_buf = mra::make_generate_op1d_items_buffer(gen_items);
        mra::DenseTensor<T, 4>* norms = nullptr; // owned by op's norms cache
        if (need_assemble) {
          norms = op.make_norms_tensor(key.level(), op_disp);
        }
#endif // MRA_ENABLE_HOST

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
        // tail pass for the unbatched path, a queued zero-fill copy from
        // submit_convolution_batch_leader for the batched path (see their
        // comments in kernels/convolution.h).
        DenseTensor<T, 1> resnorms(N, ttg::scope::SyncOut);
        T tol = truncate_tol(key, thresh, cell_min_width, truncate_mode);
        std::array<bool, 2> at = {true, key.level()>0}; // apply terms analogue in MADNESS
        // if (key.level() == 0) at[1] = false; // do not apply S at level 0

        // When batching, submit_convolution_kernel_batched's own occupancy-
        // capped scratch pool (GroupedBatchPool::ensure_scratch) supplies
        // every member's real per-block scratch -- no per-task buffer is
        // needed at all in that case, so tmp_size is 0 and tmp is never
        // allocated. The unbatched fallback (host, or batching disabled at
        // runtime) still needs the full size, since it's the only thing
        // submit_convolution_kernel uses as scratch.
#ifndef MRA_ENABLE_HOST
        const std::size_t tmp_size = enable_conv_batching ? 0 : convolution_tmp_size<NDIM>(K) * n_nonzero;
#else
        const std::size_t tmp_size = convolution_tmp_size<NDIM>(K) * n_nonzero;
#endif
        auto tmp = ttg::Buffer<T>();
        if (tmp_size > 0) tmp.reset(tmp_size, ttg::scope::Allocate);

        // std::cout << "MRA:: For Key: " << key << "\n the operators being passed are \n R\n" << op_data->ops[0]->R.current_view() << "\nand S: \n" << op_data->ops[0]->S.current_view() << std::endl;


#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(in_node.coeffs().buffer(), resnorms.buffer(),
                                        out.coeffs().buffer());
        if (tmp_size > 0) input.add(tmp);
        op.add_convolution_op_buffers(input, res, items_buf, norms);
        co_await ttg::device::select(input);
        if (need_generate) {
          op.submit_generation_kernel(res.level_work, gen_items, items_buf);
        }
        if (need_assemble) {
          // Same stream as submit_generation_kernel (if it ran) and the
          // convolution kernel below -- CUDA stream ordering guarantees each
          // sees its predecessor's writes with no wait() in between.
          op.submit_assemble_norms_kernel(res, key.level(), op_disp, *norms);
          res.norms = norms;
        }
#endif // MRA_ENABLE_HOST

        auto in_node_view = in_node.coeffs().current_view();
        auto out_view = out.coeffs().current_view();

#ifdef MRA_ENABLE_HOST
        auto opnorms_view = detail::batch_opnorms(op_data->norms.current_view(), fns->global_offset(key));
        std::array<DenseTensorView<const T, 3>, NDIM> transr = op_data->transr();
        std::array<DenseTensorView<const T, 3>, NDIM> transs = op_data->transs();
#else
        auto opnorms_view = detail::batch_opnorms(res.norms->current_view(), fns->global_offset(key));
        std::array<DenseTensorView<const T, 3>, NDIM> transr = {res.dims[0]->R.current_view(), res.dims[1]->R.current_view(), res.dims[2]->R.current_view()};
        std::array<DenseTensorView<const T, 3>, NDIM> transs = {res.dims[0]->S.current_view(), res.dims[1]->S.current_view(), res.dims[2]->S.current_view()};
#endif // MRA_ENABLE_HOST
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
          // too, so the batch leader can read its sparsity and queue a direct
          // copy into its own device buffer instead of each member pushing
          // its own via SparsityManager here. n_nonzero (this member's own,
          // computed above independent of batching) travels through as well,
          // so the leader can flatten every member's own non-zero work items
          // into one combined 1D launch. resnorms (the real tensor, not just
          // resnorms_view) travels through too, so the leader can likewise
          // queue a direct zero-fill copy into its own device buffer (see
          // submit_convolution_batch_leader).
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(empty_node_view, in_node_view, out_view, resnorms_view,
                                                                  transr, transs, opnorms_view, tol, at, out.coeffs(),
                                                                  n_nonzero, resnorms);
          // followers: the leader's batched launch already wrote our slice of out/resnorms.
          detail::submit_convolution_batch_leader<T, NDIM>(batch, *conv_pool, K, fac, total_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          auto sparseman = make_sparsity_manager(out);
          sparseman.populate_device_sparsity();
          // out_view's device sparsity is demoted inline, per position, by
          // convolution_process_one for any function whose computed norm is
          // exactly zero -- before the host-only out.set_zero(i) loop below
          // narrows the *host* side of the same sparsity. This is shared by
          // both the unbatched (submit_convolution_kernel, below) and batched
          // (convolution_kernel_batched, see submit_convolution_batch_leader)
          // paths, so both get it with no separate prune kernel either way.
          submit_convolution_kernel<T, NDIM>(key, key-key, K, N, n_nonzero, fac, tol, /*in_node_view*/ empty_node_view,
                                              in_node_view, out_view, resnorms_view, transr, transs, opnorms_view,
                                              at, tmp.current_device_ptr(), ttg::device::current_stream());
        }

#ifndef MRA_ENABLE_HOST
        // wait for the result norms to come back. If we generated the level
        // table, also pull its norms1d back to the host before publishing:
        // screener_tt computes its Opnorms from it there (try_get_opnorms()).
        // R/S and the assembled norms tensor are only ever consumed on the
        // device, so they stay there.
        // need_generate implies need_assemble (see the try_get_op() loop above).
        if (need_generate) {
          co_await ttg::device::wait(resnorms.buffer(), res.level_work->norms1d.buffer());
        } else {
          co_await ttg::device::wait(resnorms.buffer());
        }
        if (need_assemble && op.validate_ops_enabled()) {
          co_await ttg::device::wait(norms->buffer()); // DEBUG: publish_op() validates it on the host
        }
        if (need_assemble) {
          // Publish only once this task has completed: until then the
          // runtime still holds our (written) device copies of the table and
          // norms, and other tasks must not stage them in.
          sends.push_back(ttg::device::on_complete(
            [opp = &op, level_work = res.level_work, level = key.level(), op_disp]() {
              opp->publish_generation_work(level_work); // no-op unless we generated the table
              opp->publish_op(level, op_disp);
            }));
        }
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
     * Recursive edges. Two separate node edges, not one, because the SAME
     * node value dispatch hands accumulate_tt as in_node is also broadcast
     * elsewhere as the (read-only) `contribution` input of OTHER accumulate_tt
     * chains (one source node's coefficients contribute to many destination
     * keys) -- TTG cannot reconcile one consumer wanting to move/mutate a
     * value that other consumers still need to read. accumulate_node_recurse
     * therefore stays const/read-only (fed by dispatch on the first hop, and
     * by accumulate_tt itself with an empty placeholder on every later hop);
     * accumulate_node_recurse_owned is exclusively owned by this one chain
     * (an empty placeholder from dispatch on the first hop, real,
     * move-from-safe data from accumulate_tt's own previous iteration on
     * every later hop) -- see accumulate_tt's comment for how it picks
     * whichever of the two actually arrived non-empty.
     */
    ttg::Edge<detail::KeyPair<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> accumulate_node_recurse;
    ttg::Edge<detail::KeyPair<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> accumulate_node_recurse_owned;
    ttg::Edge<detail::KeyPair<NDIM>, std::vector<detail::KeyPair<NDIM>>> accumulate_contribution_recurse;

    /**
     * Dispatch task receives the input node and the list of contributions to apply to it.
     * It sends the node and the contributions to the first shellN task, or sends the node directly
     * to the output if there are no contributions to apply.
     */
    std::shared_ptr<detail::ConvContributionStats> conv_stats;
    if (detail::ConvContributionStats::enabled()) {
      conv_stats = std::make_shared<detail::ConvContributionStats>(name, total_functions);
    }
    auto accumulate_dispatch = ttg::make_tt<Space>(
      [=](const mra::Key<NDIM>& key,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node,
          const std::vector<detail::KeyPair<NDIM>>& contributions,
          const std::array<bool, mra::Key<NDIM>::num_children()>& child_empty) -> TASKTYPE {

        if (conv_stats) conv_stats->record(key.level(), contributions.size());

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
          send_out(key, in_node, std::integral_constant<std::size_t, 3>{});
        } else if (chunk_size > 0) {
          // split the contributions into chunks of (nearly) equal size, at most chunk_size each
          // unless that would give a Sum task more partial sums than it can select
          const std::size_t n = contributions.size();
          std::size_t nchunks = (n + chunk_size - 1) / chunk_size;
          if (nchunks > detail::conv_max_sum_parts) {
            nchunks = detail::conv_max_sum_parts;
            if ((n + nchunks - 1) / nchunks > detail::conv_max_chunk_size) {
              throw std::runtime_error("convolution: too many contributions to one node for the chunked path; "
                                       "set MRA_TTG_CONV_CHUNK=0");
            }
          }
          std::size_t begin = 0;
          for (std::size_t j = 0; j < nchunks; ++j) {
            const std::size_t size = n / nchunks + (j < n % nchunks ? 1 : 0);
            detail::ChunkKey<NDIM> ck{key, static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(size),
                                      static_cast<std::uint32_t>(nchunks)};
            std::vector<detail::KeyPair<NDIM>> pairs(contributions.begin() + begin, contributions.begin() + begin + size);
            begin += size;
#ifndef MRA_ENABLE_HOST
            // an rvalue: the send keeps its own copy (it runs when the task completes)
            sends.push_back(ttg::device::broadcast<12>(std::move(pairs), std::move(ck)));
#else
            ttg::broadcast<12>(std::move(pairs), std::move(ck));
#endif
          }
          if (nchunks == 1) {
            // the only chunk adds the shell-0 result itself
            send_out(detail::ChunkKey<NDIM>{key, 0, static_cast<std::uint32_t>(n), 1}, in_node,
                     std::integral_constant<std::size_t, 13>{});
          } else {
            send_out(detail::SumKey<NDIM>{key, static_cast<std::uint32_t>(nchunks)}, in_node,
                     std::integral_constant<std::size_t, 14>{});
          }
        } else {
          // send the input node and the list of contributions to the accumulate task that applies contributions one by one recursively.
          // in_node goes out on the const edge (index 0): it's read-only here since the
          // same value is also broadcast elsewhere as other chains' `contribution` input.
          // The owned edge (index 1) gets an empty placeholder on this, the first hop --
          // see the edges' own comment and accumulate_tt's for why.
          //std::cout << "ACCUMULATE DISPATCH " << key << " dispatching to accumulate " << contributions.back() << " with " << contributions.size() << " contributions" << std::endl;
          send_out(contributions.back(), in_node, std::integral_constant<std::size_t, 0>{});
          send_out(contributions.back(), mra::FunctionsCompressedNode<T, NDIM>{}, std::integral_constant<std::size_t, 1>{});
          send_out(contributions.back(), contributions, std::integral_constant<std::size_t, 2>{});
        }

        /**
         * Feed empty nodes to adjust-leaf task if needed, to make sure they have sufficient inputs.
         */
        [&]<std::size_t... Is>(std::index_sequence<Is...>){
          return ((
            child_empty[Is] && send_out(key, mra::FunctionsCompressedNode<T, NDIM>{},
                                        std::integral_constant<std::size_t, 4+Is>{})
          ), ...);
        }(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});

#ifndef MRA_ENABLE_HOST
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(to_shellN, contribution_edge, down_to_accumulate_leaf_info),
         ttg::edges(accumulate_node_recurse, accumulate_node_recurse_owned, accumulate_contribution_recurse, accumulate_to_adjust_leaf,
                    adjust_leaf_edges[0], adjust_leaf_edges[1], adjust_leaf_edges[2], adjust_leaf_edges[3],
                    adjust_leaf_edges[4], adjust_leaf_edges[5], adjust_leaf_edges[6], adjust_leaf_edges[7],
                    route_chunk_edge, apply_in, sum_in),
         "AccumulateDispatch");

    /**
     * Task that applies the contributions from non-zero shells, as determined on the way down the tree.
     * The task recurses over the list of contributions to apply them one by one and send the result to the next contribution task.
     * After the last contribution is applied, the result is sent to the output.
     *
     * Receives the running-sum node on TWO edges (see accumulate_node_recurse's
     * own comment): in_node_const (read-only -- the same value dispatch sends
     * here is also broadcast as other chains' `contribution` input) and
     * in_node_owned (exclusively ours -- real data only when this task is a
     * later hop fed by a previous accumulate_tt's own recursive send). Exactly
     * one of the two is ever non-empty for a given invocation; pick that one
     * as the effective running sum below. Only in_node_owned's data may ever
     * be moved from.
     *
     * NOTE: because we use coroutines we cannot outline most of the code and instead have to copy past it here.
     */
    auto accumulate_tt = ttg::make_tt<Space>(
      [&, K, fac, thresh, truncate_mode, cell_min_width, name, enable_conv_batching, conv_pool, total_functions](
          const detail::KeyPair<NDIM>& keypair,
          const mra::FunctionsCompressedNode<T, NDIM>& in_node_const,
          mra::FunctionsCompressedNode<T, NDIM>&& in_node_owned,
          const mra::FunctionsCompressedNode<T, NDIM>& contribution,
          std::vector<detail::KeyPair<NDIM>>&& contribution_keys) -> TASKTYPE {
      // Exactly one of in_node_const/in_node_owned is non-empty (see the
      // class comment above and accumulate_node_recurse's) -- in_node is a
      // read-only alias for whichever that is, used everywhere the old
      // single in_node parameter used to be read; only in_node_owned itself
      // may ever be moved from (see the reuse branch further down).
      const bool in_node_from_owned = !in_node_owned.empty();
      const auto& in_node = in_node_from_owned ? in_node_owned : in_node_const;

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

#ifdef MRA_ENABLE_HOST
      auto op_data = op.get_op(key.level(), displacement);
#else
      // Survives the loop so res.dims/res.norms are usable afterward -- see
      // shell0_tt's identical pattern for why there's no separate cached
      // "aggregate" object on this path (conv_mad.h's OpResult comment), and
      // for why we only spin on Pending here: the operator buffers ride along
      // in the convolution's one select() below.
      using OpResult = typename mra::GaussianConvolutionOperator<T, NDIM>::OpResult;
      using OpStatus = typename mra::GaussianConvolutionOperator<T, NDIM>::Status;
      OpResult res;
      for (;;) {
        res = op.try_get_op(key.level(), displacement);
        if (res.status != OpStatus::Pending) break;
        if (op.has_generation_work(res)) {
          // see shell0_tt
          throw std::runtime_error("Accumulate: own level table generation but norms assembly is pending elsewhere");
        }
        co_await ttg::device::suspend();
      }
      const bool need_generate = op.has_generation_work(res);
      const bool need_assemble = (res.status == OpStatus::Owned);
      auto gen_items = op.collect_generation_items(res.level_work); // empty if !need_generate
      auto items_buf = mra::make_generate_op1d_items_buffer(gen_items);
      mra::DenseTensor<T, 4>* norms = nullptr; // owned by op's norms cache
      if (need_assemble) {
        norms = op.make_norms_tensor(key.level(), displacement);
      }
#endif // MRA_ENABLE_HOST

      // When the newly-computed sparsity exactly matches in_node's own, its
      // already-allocated storage has exactly the right shape to serve as
      // out directly -- move it in instead of allocating a fresh tensor and
      // copying/convolving into that. The move is an O(1) buffer-ownership
      // transfer (see Tensor/ttg::Buffer's move ctor), not a device copy. Only
      // ever valid when in_node came from the owned edge: in_node_const may
      // be the SAME value other accumulate_tt chains are still reading as
      // their own `contribution` (see accumulate_node_recurse's comment), so
      // it must never be moved from, regardless of sparsity. Also excluded:
      // an invalid (no corresponding prior node) or empty (no coefficients
      // yet) in_node -- those fall through to the ordinary allocate path
      // below.
      // set_ns()/apply_leaf_info() must NOT run in the reuse branch the way
      // they do below: set_ns() would be clobbered by the move if called
      // first, and apply_leaf_info(out, in_node) reads in_node -- already
      // moved-from by that point -- when out already carries in_node's exact
      // child-leaf info verbatim via the move.
      const bool reuse_in_node = in_node_from_owned && !in_node.invalid() && !in_node.empty() &&
                                  detail::sparsity_equal(sparsity, in_node.sparsity());
      if (reuse_in_node) {
        // not `out = std::move(in_node_owned)`: that would also move TTG's
        // data-copy bookkeeping out of the task's input (see move_contents_from)
        out.move_contents_from(std::move(in_node_owned));
        out.set_ns();
      } else {
        out.allocate(sparsity, K, ttg::scope::Allocate);
        out.set_ns();
        mra::apply_leaf_info(out, in_node);
      }

      DenseTensor<T, 1> resnorms;
      const double tol = truncate_tol(key, thresh, cell_min_width, truncate_mode);
      std::array<bool, 2> at = {true, source.level()>0}; // apply terms analogue in MADNESS

      // See shell0_tt's identical comment above: batched launches get their
      // scratch from submit_convolution_kernel_batched's occupancy-capped
      // pool instead -- no per-task buffer is needed at all in that case.
#ifndef MRA_ENABLE_HOST
      const std::size_t tmp_size = enable_conv_batching ? 0 : convolution_tmp_size<NDIM>(K) * n_nonzero;
#else
      const std::size_t tmp_size = convolution_tmp_size<NDIM>(K) * n_nonzero;
#endif
      auto tmp = ttg::Buffer<T>();
      if (tmp_size > 0) tmp.reset(tmp_size, ttg::scope::Allocate);

      // std::cout << "MRA:: For Key: " << key << "\n the operators being passed are \n R\n" << op_data->ops[0]->R.current_view() << "\nand S: \n" << op_data->ops[0]->S.current_view() << std::endl;

      if (last_key) {
        // No host zero-fill needed: entries the compute kernel doesn't visit
        // are finalized to 0 device-side instead -- convolution_kernel's own
        // tail pass for the unbatched path, a queued zero-fill copy from
        // submit_convolution_batch_leader for the batched path (see their
        // comments in kernels/convolution.h).
        resnorms = DenseTensor<T, 1>(N, ttg::scope::SyncOut);
      }
#ifndef MRA_ENABLE_HOST
      // When reusing in_node's storage as out, in_node is already moved-from
      // -- out.coeffs().buffer() is now the SAME underlying buffer, so there
      // is nothing further to select for it separately.
      auto input = ttg::device::Input(out.coeffs().buffer(), contribution.coeffs().buffer());
      if (!reuse_in_node) input.add(in_node.coeffs().buffer());
      if (tmp_size > 0) input.add(tmp);
      op.add_convolution_op_buffers(input, res, items_buf, norms);
      if (last_key) {
        // if this is the last we want to get the norms of the result back
        input.add(resnorms.buffer());
      }
      co_await ttg::device::select(input);
      if (need_generate) {
        op.submit_generation_kernel(res.level_work, gen_items, items_buf);
      }
      if (need_assemble) {
        // same stream as the generation and convolution kernels, see shell0_tt
        op.submit_assemble_norms_kernel(res, key.level(), displacement, *norms);
        res.norms = norms;
      }
#endif // MRA_ENABLE_HOST

#ifdef MRA_ENABLE_HOST
      std::array<DenseTensorView<const T, 3>, NDIM> transr = op_data->transr();
      std::array<DenseTensorView<const T, 3>, NDIM> transs = op_data->transs();
      auto opnorms_view = detail::batch_opnorms(op_data->norms.current_view(), fns->global_offset(key));
#else
      std::array<DenseTensorView<const T, 3>, NDIM> transr = {res.dims[0]->R.current_view(), res.dims[1]->R.current_view(), res.dims[2]->R.current_view()};
      std::array<DenseTensorView<const T, 3>, NDIM> transs = {res.dims[0]->S.current_view(), res.dims[1]->S.current_view(), res.dims[2]->S.current_view()};
      auto opnorms_view = detail::batch_opnorms(res.norms->current_view(), fns->global_offset(key));
#endif // MRA_ENABLE_HOST
      auto out_view = out.coeffs().current_view();
      auto contribution_view = contribution.coeffs().current_view();
      // When reusing in_node's storage as out, in_node is already moved-from
      // -- out_view IS what used to be in_node's own view, now aliasing the
      // kernel's write target. convolution_process_one's staged_in (see its
      // comment in kernels/convolution.h) is exactly what makes that safe.
      auto in_node_view = reuse_in_node ? out_view : in_node.coeffs().current_view();
      auto resnorms_view = resnorms.current_view();

#ifndef MRA_ENABLE_HOST
      if (enable_conv_batching) {
        // transr/transs/opnorms_view/tol/at also travel through coop() since
        // batching is unrestricted now (any level, any displacement), so a
        // batch mate may have entirely different operator data -- see the
        // batching-support comment on ConvolutionBatchArg in kernels/convolution.h.
        // out.coeffs() (the real tensor, not just its view) travels through
        // too, so the batch leader can read its sparsity and queue a direct
        // copy into its own device buffer instead of each member pushing its
        // own via SparsityManager here. n_nonzero (this member's own,
        // computed above independent of batching) travels through as well,
        // so the leader can flatten every member's own non-zero work items
        // into one combined 1D launch. resnorms (the real tensor, not just
        // resnorms_view -- empty unless last_key, same as resnorms_view)
        // travels through too, so the leader can likewise queue a direct
        // zero-fill copy into its own device buffer (see
        // submit_convolution_batch_leader).
        auto batch = co_await ttg::device::coop<detail::KeyPair<NDIM>>(in_node_view, contribution_view, out_view, resnorms_view,
                                                                       transr, transs, opnorms_view, tol, at, out.coeffs(),
                                                                       n_nonzero, resnorms);
        // followers: the leader's batched launch already wrote our slice of out/resnorms.
        detail::submit_convolution_batch_leader<T, NDIM>(batch, *conv_pool, K, fac, total_functions);
      } else
#endif // MRA_ENABLE_HOST
      {
        auto sparseman = make_sparsity_manager(out);
        sparseman.populate_device_sparsity();
        // out_view's device sparsity is demoted inline, per position, by
        // convolution_process_one for any function whose computed norm is
        // exactly zero -- gated the same way as resnorms itself (empty
        // unless last_key, see above) -- before the host-only out.set_zero(i)
        // loop below (only reached when last_key) narrows the *host* side of
        // the same sparsity. Shared by both the unbatched (submit_convolution_kernel,
        // below) and batched (convolution_kernel_batched, see
        // submit_convolution_batch_leader) paths, so both get it with no
        // separate prune kernel either way.
        submit_convolution_kernel<T, NDIM>(key, displacement, K, N, n_nonzero, fac, tol, in_node_view,
                                            contribution_view, out_view, resnorms_view, transr, transs,
                                            opnorms_view, at,
                                            tmp.current_device_ptr(), ttg::device::current_stream());
      }

#ifndef MRA_ENABLE_HOST
      if (need_generate) {
        // pull the generated table's norms1d back to the host for
        // screener_tt before publishing -- see shell0_tt
        co_await ttg::device::wait(res.level_work->norms1d.buffer());
      }
      if (need_assemble && op.validate_ops_enabled()) {
        co_await ttg::device::wait(norms->buffer()); // DEBUG: publish_op() validates it on the host
      }
      if (need_assemble) {
        // publish once this task has completed -- see shell0_tt
        sends.push_back(ttg::device::on_complete(
          [opp = &op, level_work = res.level_work, level = key.level(), displacement]() {
            opp->publish_generation_work(level_work); // no-op unless we generated the table
            opp->publish_op(level, displacement);
          }));
      }
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
        send_out(keypair.dest, std::move(out), std::integral_constant<std::size_t, 3>{});
      } else {
        // send the result to the next contribution task or output: the real
        // data goes out on the owned edge (index 1) -- the next accumulate_tt
        // invocation is this chain's sole consumer of it, so it's safe for
        // that task to move from it -- and an empty placeholder goes out on
        // the const edge (index 0), since nothing real needs to travel there
        // for this hop (see accumulate_node_recurse's comment).
        send_out(contribution_keys.back(), mra::FunctionsCompressedNode<T, NDIM>{}, std::integral_constant<std::size_t, 0>{});
        send_out(contribution_keys.back(), std::move(out), std::integral_constant<std::size_t, 1>{});
        send_out(contribution_keys.back(), std::move(contribution_keys), std::integral_constant<std::size_t, 2>{});
      }

#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(accumulate_node_recurse, accumulate_node_recurse_owned, screener_to_accumulate, accumulate_contribution_recurse),
         ttg::edges(accumulate_node_recurse, accumulate_node_recurse_owned, accumulate_contribution_recurse, accumulate_to_adjust_leaf),
         "Accumulate");

#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::ConvolutionBatchArg<T, NDIM>>> chunk_conv_pool;
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::ConvSumBatchArg<T, NDIM>>> chunk_sum_pool;
    if (chunk_size > 0) {
      chunk_conv_pool = conv_pool ? conv_pool
                                  : detail::shared_pool_registry<detail::ConvolutionBatchArg<T, NDIM>>(mra::get_batch_size());
      chunk_sum_pool = detail::shared_pool_registry<detail::ConvSumBatchArg<T, NDIM>>(mra::get_batch_size());
    }
#endif // MRA_ENABLE_HOST

    /**
     * Route: joins a source node (from the screener) with the chunk its (source, dest) pair was
     * assigned to (from AccumulateDispatch) and forwards the node to that chunk. No device work.
     */
    auto route_tt = ttg::make_tt<Space>(
      [=](const detail::KeyPair<NDIM>& keypair,
          const mra::FunctionsCompressedNode<T, NDIM>& source,
          const detail::ChunkKey<NDIM>& chunk) -> TASKTYPE {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(chunk, source);
#else
        ttg::send<0>(chunk, source);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(screener_to_route, route_chunk_edge), ttg::edges(apply_in), "ConvRoute");

    /**
     * Apply: convolves the sources of one chunk in parallel, each into its own slot of scratch
     * memory, and adds the slots into the chunk's partial sum, which goes to the node's Sum task.
     * The only chunk of a node also receives the node's shell-0 result (its key is the node's,
     * or it is invalid), adds it, and finishes the node as the last link of a chain does.
     *
     * Operator norms a chunk has to assemble itself (Owned) are claimed in a fixed order of
     * the displacements, so two chunks waiting for each other's norms cannot deadlock.
     */
    auto apply_target = [](const detail::ChunkKey<NDIM>& ck) -> std::size_t {
      return ck.size + (ck.nchunks == 1 ? 1 : 0);
    };
    auto apply_tt = ttg::make_tt<Space>(
#ifndef MRA_ENABLE_HOST
      [&, K, fac, thresh, truncate_mode, cell_min_width, enable_conv_batching, chunk_conv_pool, chunk_sum_pool](
#else
      [&, K](
#endif // MRA_ENABLE_HOST
          const detail::ChunkKey<NDIM>& ck,
          const ttg::Aggregator<mra::FunctionsCompressedNode<T, NDIM>>& values) -> TASKTYPE {
#ifdef MRA_ENABLE_HOST
        throw std::logic_error("convolution: chunked contributions need a device build");
#else
        auto sends = ttg::device::forward();
        const auto& key = ck.dest;
        const Level level = key.level();
        const size_type N = fns->num_functions(key);
        const bool final_chunk = (ck.nchunks == 1);

        /* the sources, and (only chunk) the node's own shell-0 result */
        const mra::FunctionsCompressedNode<T, NDIM>* base = nullptr;
        std::vector<const mra::FunctionsCompressedNode<T, NDIM>*> sources;
        sources.reserve(values.size());
        for (const auto& v : values) {
          if (v.invalid() || v.key() == key) {
            if (base != nullptr || !final_chunk) throw std::logic_error("convolution Apply: unexpected shell-0 result");
            base = &v;
          } else {
            sources.push_back(&v);
          }
        }
        if (sources.size() != ck.size || (final_chunk && base == nullptr)) {
          throw std::logic_error("convolution Apply: chunk does not match its key");
        }
        const std::size_t nc = sources.size();

        /* the operator of every displacement, claimed in a fixed order */
        using OpResult = typename mra::GaussianConvolutionOperator<T, NDIM>::OpResult;
        using OpStatus = typename mra::GaussianConvolutionOperator<T, NDIM>::Status;
        std::vector<Key<NDIM>> disps(nc);
        for (std::size_t j = 0; j < nc; ++j) disps[j] = key - sources[j]->key();
        std::vector<std::size_t> order(nc);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
          return disps[a].translation() < disps[b].translation();
        });
        std::vector<OpResult> res(nc);
        std::vector<mra::DenseTensor<T, 4>*> norms(nc, nullptr); // owned by op's norms cache
        std::vector<Key<NDIM>> owned_disps;
        for (std::size_t j : order) {
          for (;;) {
            res[j] = op.try_get_op(level, disps[j]);
            if (res[j].level_work != nullptr) {
              // the screeners of this level obtained the level's table before sending
              throw std::runtime_error("convolution Apply: level table not generated by the screener");
            }
            if (res[j].status != OpStatus::Pending) break;
            co_await ttg::device::suspend();
          }
          if (res[j].status == OpStatus::Owned) {
            norms[j] = op.make_norms_tensor(level, disps[j]);
            owned_disps.push_back(disps[j]);
          }
        }
        if (!owned_disps.empty() && op.validate_ops_enabled()) {
          throw std::logic_error("convolution: MRA_VALIDATE_OPS needs MRA_TTG_CONV_CHUNK=0");
        }

        /* the result: the union of the non-zero functions of the sources (and the base) */
        const bool base_data = base != nullptr && !base->invalid() && !base->empty();
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type i = 0; i < N; ++i) {
          bool nonzero = base_data && base->sparsity().is_nonzero(i);
          for (const auto* src : sources) nonzero = nonzero || src->sparsity().is_nonzero(i);
          if (nonzero) sparsity.set_nonzero(i);
        }
        mra::FunctionsCompressedNode<T, NDIM> out(key, N);
        out.allocate(sparsity, K, ttg::scope::Allocate);
        DenseTensor<T, 1> resnorms;
        if (final_chunk) {
          out.set_ns();
          mra::apply_leaf_info(out, *base);
          resnorms = DenseTensor<T, 1>(N, ttg::scope::SyncOut);
        } // a partial sum is not non-standard: that is how Sum tells it from the shell-0 result

        auto input = ttg::device::Input(out.coeffs().buffer());
        for (const auto* src : sources) input.add(src->coeffs().buffer());
        if (base_data) input.add(base->coeffs().buffer());
        if (final_chunk) input.add(resnorms.buffer());
        op.add_chunk_op_buffers(input, res, norms);
        co_await ttg::device::select(input);
        for (std::size_t j = 0; j < nc; ++j) {
          if (norms[j] != nullptr) {
            op.submit_assemble_norms_kernel(res[j], level, disps[j], *norms[j]);
            res[j].norms = norms[j];
          }
        }

        const auto device = ttg::device::current_device();
        detail::ConvChunkWork<T, NDIM> work;
        work.N = N;
        work.tol = truncate_tol(key, thresh, cell_min_width, truncate_mode);
        work.at = {true, level > 0};
        auto empty_node = mra::FunctionsCompressedNode<T, NDIM>();
        detail::rebind_view(work.empty_view, empty_node.coeffs().current_view());
        DenseTensor<T, 1> no_norms;
        detail::rebind_view(work.no_norms, no_norms.current_view());
        work.contribs.resize(nc);
        for (std::size_t j = 0; j < nc; ++j) {
          auto& c = work.contribs[j];
          const auto* src = sources[j];
          detail::rebind_view(c.f, src->coeffs().current_view());
          c.transr = {res[j].dims[0]->R.current_view(), res[j].dims[1]->R.current_view(), res[j].dims[2]->R.current_view()};
          c.transs = {res[j].dims[0]->S.current_view(), res[j].dims[1]->S.current_view(), res[j].dims[2]->S.current_view()};
          detail::rebind_view(c.opnorms, detail::batch_opnorms(res[j].norms->current_view(), fns->global_offset(key)));
          SparsityInfo src_sparsity(N, SparsityInfo::InitType::AllZero);
          for (size_type i = 0; i < N; ++i) {
            if (src->sparsity().is_nonzero(i)) src_sparsity.set_nonzero(i);
          }
          c.n_nonzero = src_sparsity.count_nonzero();
          c.sparsity.resize(N);
          detail::sparsity_to_bytes(src_sparsity, c.sparsity.data(), N);
        }
        detail::rebind_view(work.out_view, out.coeffs().current_view());
        work.out_sparsity.resize(N);
        detail::sparsity_to_bytes(out.coeffs().sparsity(), work.out_sparsity.data(), N);
        work.out_dev = const_cast<T*>(out.coeffs().buffer().device_ptr_on(device));
        work.out_nonzero = sparsity.count_nonzero();
        if (base_data) {
          work.has_base = true;
          detail::rebind_view(work.base_view, base->coeffs().current_view());
        }
        if (final_chunk) {
          work.has_norms = true;
          detail::rebind_view(work.resnorms_view, resnorms.current_view());
          work.resnorms_dev = const_cast<T*>(resnorms.buffer().device_ptr_on(device));
        }

        if (enable_conv_batching) {
          auto batch = co_await ttg::device::coop<detail::ChunkKey<NDIM>>(work);
          if (batch.is_leader()) {
            std::vector<detail::ConvChunkWork<T, NDIM>*> works(batch.size());
            for (std::size_t m = 0; m < batch.size(); ++m) works[m] = &batch[m].template get<0>();
            detail::submit_conv_chunk_works<T, NDIM>(works.data(), works.size(), *chunk_conv_pool, *chunk_sum_pool,
                                                     K, fac, detail::conv_chunk_scratch_bytes());
          }
        } else {
          detail::ConvChunkWork<T, NDIM>* w = &work;
          detail::submit_conv_chunk_works<T, NDIM>(&w, 1, *chunk_conv_pool, *chunk_sum_pool, K, fac,
                                                   detail::conv_chunk_scratch_bytes());
        }

        if (!owned_disps.empty()) {
          // publish once this task has completed, see shell0_tt
          sends.push_back(ttg::device::on_complete(
            [opp = &op, level, owned_disps = std::move(owned_disps)]() {
              for (const auto& d : owned_disps) opp->publish_op(level, d);
            }));
        }

        if (final_chunk) {
          co_await ttg::device::wait(resnorms.buffer());
          bool empty = true;
          auto resnorms_host_view = resnorms.view_on(ttg::device::Device::host());
          for (size_type i = 0; i < N; ++i) {
            if (resnorms_host_view(i) != 0.0) {
              empty = false;
            } else {
              out.set_zero(i);
            }
          }
          if (empty) out.make_empty(); // drop the memory but keep child info
          sends.push_back(ttg::device::send<1>(key, std::move(out)));
        } else {
          sends.push_back(ttg::device::send<0>(detail::SumKey<NDIM>{key, ck.nchunks}, std::move(out)));
        }
        co_await std::move(sends);
#endif // MRA_ENABLE_HOST
      }, ttg::edges(ttg::make_aggregator(apply_in, apply_target)),
         ttg::edges(sum_in, accumulate_to_adjust_leaf), "ConvApply");

    /**
     * Sum: adds the partial sums of a node with several chunks and its shell-0 result (the
     * non-standard or invalid value), and finishes the node as the last link of a chain does.
     */
    auto sum_target = [](const detail::SumKey<NDIM>& sk) -> std::size_t { return sk.nchunks + 1; };
    auto sum_tt = ttg::make_tt<Space>(
#ifndef MRA_ENABLE_HOST
      [&, K, enable_conv_batching, chunk_sum_pool](
#else
      [&, K](
#endif // MRA_ENABLE_HOST
          const detail::SumKey<NDIM>& sk,
          const ttg::Aggregator<mra::FunctionsCompressedNode<T, NDIM>>& values) -> TASKTYPE {
#ifdef MRA_ENABLE_HOST
        throw std::logic_error("convolution: chunked contributions need a device build");
#else
        const auto& key = sk.dest;
        const size_type N = fns->num_functions(key);
        const mra::FunctionsCompressedNode<T, NDIM>* base = nullptr;
        std::vector<const mra::FunctionsCompressedNode<T, NDIM>*> parts;
        for (const auto& v : values) {
          if (v.invalid() || v.is_ns()) {
            if (base != nullptr) throw std::logic_error("convolution Sum: two shell-0 results");
            base = &v;
          } else {
            parts.push_back(&v);
          }
        }
        if (base == nullptr || parts.size() != sk.nchunks) {
          throw std::logic_error("convolution Sum: values do not match the key");
        }
        const bool base_data = !base->invalid() && !base->empty();

        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        for (size_type i = 0; i < N; ++i) {
          bool nonzero = base_data && base->sparsity().is_nonzero(i);
          for (const auto* p : parts) nonzero = nonzero || (!p->empty() && p->sparsity().is_nonzero(i));
          if (nonzero) sparsity.set_nonzero(i);
        }
        mra::FunctionsCompressedNode<T, NDIM> out(key, N);
        out.allocate(sparsity, K, ttg::scope::Allocate);
        out.set_ns();
        mra::apply_leaf_info(out, *base);
        DenseTensor<T, 1> resnorms(N, ttg::scope::SyncOut);

        auto input = ttg::device::Input(out.coeffs().buffer(), resnorms.buffer());
        if (base_data) input.add(base->coeffs().buffer());
        for (const auto* p : parts) if (!p->empty()) input.add(p->coeffs().buffer());
        co_await ttg::device::select(input);

        const auto device = ttg::device::current_device();
        detail::ConvSumWork<T, NDIM> work;
        work.N = N;
        if (base_data) work.srcs.push_back(base->coeffs().current_view());
        for (const auto* p : parts) if (!p->empty()) work.srcs.push_back(p->coeffs().current_view());
        detail::rebind_view(work.out_view, out.coeffs().current_view());
        work.out_sparsity.resize(N);
        detail::sparsity_to_bytes(out.coeffs().sparsity(), work.out_sparsity.data(), N);
        work.out_dev = const_cast<T*>(out.coeffs().buffer().device_ptr_on(device));
        work.out_nonzero = sparsity.count_nonzero();
        detail::rebind_view(work.resnorms_view, resnorms.current_view());
        work.resnorms_dev = const_cast<T*>(resnorms.buffer().device_ptr_on(device));

        if (enable_conv_batching) {
          auto batch = co_await ttg::device::coop<detail::SumKey<NDIM>>(work);
          if (batch.is_leader()) {
            std::vector<detail::ConvSumWork<T, NDIM>*> works(batch.size());
            for (std::size_t m = 0; m < batch.size(); ++m) works[m] = &batch[m].template get<0>();
            detail::submit_conv_sum_works<T, NDIM>(works.data(), works.size(), *chunk_sum_pool);
          }
        } else {
          detail::ConvSumWork<T, NDIM>* w = &work;
          detail::submit_conv_sum_works<T, NDIM>(&w, 1, *chunk_sum_pool);
        }
        co_await ttg::device::wait(resnorms.buffer());

        bool empty = true;
        auto resnorms_host_view = resnorms.view_on(ttg::device::Device::host());
        for (size_type i = 0; i < N; ++i) {
          if (resnorms_host_view(i) != 0.0) {
            empty = false;
          } else {
            out.set_zero(i);
          }
        }
        if (empty) out.make_empty(); // drop the memory but keep child info
        co_await ttg::device::send<0>(key, std::move(out));
#endif // MRA_ENABLE_HOST
      }, ttg::edges(ttg::make_aggregator(sum_in, sum_target)),
         ttg::edges(accumulate_to_adjust_leaf), "ConvSum");

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
      route_tt->set_keymap([=](const detail::KeyPair<NDIM>& kp) { return procmap(kp.dest); });
      apply_tt->set_keymap([=](const detail::ChunkKey<NDIM>& ck) { return procmap(ck.dest); });
      sum_tt->set_keymap([=](const detail::SumKey<NDIM>& sk) { return procmap(sk.dest); });
      //dispatch_adjust_parent_tt->set_keymap(procmap);
      //adjust_parent_tt->set_keymap(procmap);
    }
    /* adjust-leaf modifies the node it receives, which AccumulateDispatch may forward while
     * other tasks (or outgoing transfers) still read it: the node is not copyable, so the writer
     * must wait for the readers */
    adjust_leaf_info_tt->set_defer_writer(true);
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
      route_tt->set_devicemap([=](const detail::KeyPair<NDIM>& kp) { return devicemap(kp.dest); });
      apply_tt->set_devicemap([=](const detail::ChunkKey<NDIM>& ck) { return devicemap(ck.dest); });
      sum_tt->set_devicemap([=](const detail::SumKey<NDIM>& sk) { return devicemap(sk.dest); });
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

      apply_tt->set_batch_matcher(
          [](const detail::ChunkKey<NDIM>&, const detail::ChunkKey<NDIM>&) { return true; },
          max_batch_size);
      sum_tt->set_batch_matcher(
          [](const detail::SumKey<NDIM>&, const detail::SumKey<NDIM>&) { return true; },
          max_batch_size);

      // norm_tt: unrestricted just like the two above -- simple_norm has no
      // operator data to begin with, so there is nothing to keep uniform
      // across a batch (see mra/kernels/simple_norm.h's batching-support
      // comment).
      norm_tt->set_batch_matcher(
          [](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
          max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    // TODO: assemble TTG

    auto ins = std::make_tuple(screener_tt->template in<0>());
    auto outs = std::make_tuple(adjust_leaf_info_tt->template out<2>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(11);
    ops[0] = std::move(up_contributions_tt);
    ops[1] = std::move(down_contributions_tt);
    ops[2] = std::move(norm_tt);
    ops[3] = std::move(screener_tt);
    ops[4] = std::move(shell0_tt);
    ops[5] = std::move(adjust_leaf_info_tt);
    ops[6] = std::move(accumulate_dispatch);
    ops[7] = std::move(accumulate_tt);
    ops[8] = std::move(route_tt);
    ops[9] = std::move(apply_tt);
    ops[10] = std::move(sum_tt);
    //ops[8] = std::move(dispatch_adjust_parent_tt);
    //ops[9] = std::move(adjust_parent_tt);

    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_CONVOLUTION_H
