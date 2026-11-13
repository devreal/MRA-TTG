#ifndef MRA_TASKS_TRANSFORM_H
#define MRA_TASKS_TRANSFORM_H

#include <algorithm>
#include <array>
#include <memory>
#include <ttg.h>
#include <utility>
#include <vector>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/batch_size.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/functionnode.h"
#include "mra/kernels/transform_functions.h"
#include "mra/tasks/common.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

namespace mra {

  /**
   * Linear combination of a set of functions in compressed form,
   *   out_j = sum_i U(i,j) in_i,   i < N, j < M
   * (cf. madness::transform). U is an N x M row-major matrix, given when the graph
   * is built or arriving through an edge (see the overloads below). It travels down
   * the tree with the nodes; its host copy is read to determine the structure of
   * the result.
   *
   * The tree of out_j is the union of the trees of the in_i with U(i,j) != 0.
   * It is determined locally from the child information of the input node: a
   * function not present at a key has zero coefficients and only leaf children
   * there, so it contributes nothing. An output node is sent at every key where
   * some output function continues. Below a key where only input functions that
   * no output depends on (zero rows of U) continue, the input nodes are consumed
   * without output: the propagate task sends an empty U down those subtrees.
   * (Adding such trees to an output instead would make that output's tree, and
   * the products and convolutions computed from it, depend on the other
   * functions of the input set.)
   *
   * U is passed down the tree by a separate propagate task that does not touch
   * device data, so it runs ahead of the transform kernels and exposes more tasks
   * for batching.
   *
   * The input must be compressed, not non-standard. No screening is applied;
   * truncate the result if needed.
   */
  namespace detail {
    /**
     * The host metadata of an input node that the propagate task reads: whether it is
     * non-standard, which functions are non-zero, and which children are leaves. It holds
     * no buffer, so sending it from the (device) dispatch task to the host propagate task
     * does not make TTG push the node's coefficients out to the host. (TTG pushes out
     * every datum a device task sends to a host task, by turning the flow into a write
     * while other device tasks may still read the copy.)
     */
    template <Dimension NDIM>
    struct TransformNodeStructure {
      static constexpr const int num_children = Key<NDIM>::num_children();
      bool ns = false;
      std::vector<char> nonzero;                              // per function
      std::vector<std::array<bool, num_children>> child_leaf; // per function and child

      TransformNodeStructure() = default;

      template <typename T>
      explicit TransformNodeStructure(const FunctionsCompressedNode<T, NDIM>& node, size_type N)
      : ns(node.is_ns()), nonzero(N), child_leaf(N)
      {
        for (size_type i = 0; i < N; ++i) {
          nonzero[i] = !node.is_zero(i);
          for (int c = 0; c < num_children; ++c) child_leaf[i][c] = node.is_child_leaf(i, c);
        }
      }

      bool is_ns() const { return ns; }
      bool is_zero(size_type i) const { return !nonzero[i]; }
      bool is_child_leaf(size_type i, int c) const { return child_leaf[i][c]; }
      bool is_all_child_leaf(const Key<NDIM>& child) const {
        const int c = child.childindex();
        return std::all_of(child_leaf.begin(), child_leaf.end(), [c](const auto& leaf) { return leaf[c]; });
      }

      template <typename Archive>
      void serialize(Archive& ar) { ar & ns & nonzero & child_leaf; }
      template <typename Archive>
      void serialize(Archive& ar, const unsigned int) { serialize(ar); }
    };

    /**
     * Determines which output functions have a non-zero node at the key of in_node and
     * which have children below it. NodeT is the input node or its TransformNodeStructure.
     */
    template <typename T, Dimension NDIM, typename NodeT>
    void transform_structure(const NodeT& in_node,
                             const T* U, size_type N, size_type M,
                             std::vector<std::array<bool, mra::Key<NDIM>::num_children()>>& has_child,
                             SparsityInfo& sparsity)
    {
      has_child.assign(M, {});
      for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < M; ++j) {
          if (U[i*M + j] == T(0)) continue;
          if (!in_node.is_zero(i)) sparsity.set_nonzero(j);
          for (int c = 0; c < mra::Key<NDIM>::num_children(); ++c) {
            has_child[j][c] = has_child[j][c] || !in_node.is_child_leaf(i, c);
          }
        }
      }
    }

    /** the index in the set of the first function of a batch */
    template <typename FunctionSet>
    size_type batch_offset(const FunctionSet& fns, Batch b) {
      size_type offset = 0;
      for (Batch c = 0; c < b; ++c) offset += fns.batchman()->batch_size(c);
      return offset;
    }

    /**
     * The block of U, over all input and output functions, for the functions of batch b: its
     * rows of b's input functions and columns of b's output functions. Each batch transforms
     * its own functions, so U must not couple the functions of another batch into b's outputs.
     */
    template <typename T, typename FunctionSetIn, typename FunctionSetOut>
    mra::DenseTensor<T, 2> batch_block(const mra::DenseTensor<T, 2>& U, const FunctionSetIn& fns_in,
                                       const FunctionSetOut& fns_out, Batch b, const std::string& name) {
      const size_type N = U.dim(0), M = U.dim(1);
      const size_type i0 = batch_offset(fns_in, b), n = fns_in.batchman()->batch_size(b);
      const size_type j0 = batch_offset(fns_out, b), m = fns_out.batchman()->batch_size(b);
      const T* u = U.buffer().host_ptr();
      for (size_type i = 0; i < N; ++i) {
        if (i >= i0 && i < i0 + n) continue;
        for (size_type j = j0; j < j0 + m; ++j) {
          if (u[i*M + j] != T(0))
            throw std::runtime_error(name + ": U couples the functions of another batch into this one");
        }
      }
      mra::DenseTensor<T, 2> block(std::array<size_type, 2>{n, m});
      T* blk = block.buffer().host_ptr();
      for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < m; ++j) blk[i*m + j] = u[(i0 + i)*M + j0 + j];
      return block;
    }

    /* U arrives at the root of each batch on U_in, sized for the batch, or, if U_fixed is given,
     * is sent there by the dispatch task: with several batches U_fixed is over the functions of
     * all and each root gets its batch's block (batch_block), the transform is block-diagonal in
     * the batches. The propagate tasks pass U on to the children that receive input nodes. */
    template <typename T, Dimension NDIM, typename FunctionSetIn, typename FunctionSetOut,
              typename ProcMap, typename DeviceMap>
    auto make_transform_impl(const std::shared_ptr<FunctionSetIn>& fns_in,
                             const std::shared_ptr<FunctionSetOut>& fns_out,
                             size_type K,
                             ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>> U_in,
                             const std::shared_ptr<mra::DenseTensor<T, 2>> U_fixed,
                             ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
                             ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
                             const std::string& name,
                             ProcMap procmap,
                             DeviceMap devicemap)
  {
    using node_type = mra::FunctionsCompressedNode<T, NDIM>;
    using matrix_type = mra::DenseTensor<T, 2>;
    static constexpr const int num_children = mra::Key<NDIM>::num_children();

    /* batching is controlled process-wide (mra/misc/batch_size.h), read once here; U is
     * the same for every node (also when it arrives through an edge: one matrix is passed
     * down the tree), so any two tasks may be batched */
    const std::size_t max_batch_size = mra::get_batch_size();
    const bool enable_batching = mra::batching_enabled();
    const size_type total_functions = fns_out->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::GroupedBatchPoolRegistry<detail::TransformFunctionsBatchArg<T, NDIM>>> batch_pool;
    if (enable_batching) {
      batch_pool = std::make_shared<detail::GroupedBatchPoolRegistry<detail::TransformFunctionsBatchArg<T, NDIM>>>(
                     ttg::device::num_devices(), max_batch_size);
    }
#else  // MRA_ENABLE_HOST
    std::nullptr_t batch_pool = nullptr; // never used, only captured
#endif // MRA_ENABLE_HOST

    using structure_type = detail::TransformNodeStructure<NDIM>;
    ttg::Edge<mra::Key<NDIM>, structure_type> node_p;
    ttg::Edge<mra::Key<NDIM>, node_type> node_t;
    ttg::Edge<mra::Key<NDIM>, matrix_type> U_p(name + "-U"), U_t;

    /**
     * Forward the input nodes to the transform task, their structure to the propagate
     * task, and seed the root with U, if it is fixed. The propagate task receives only
     * the structure: sending it the node would push its coefficients out to the host.
     */
    auto dispatch_fn = [U_fixed, fns_in, fns_out, name](const mra::Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      structure_type structure(node, fns_in->num_functions(key));
#ifndef MRA_ENABLE_HOST
      auto sends = ttg::device::forward(ttg::device::send<0>(key, std::move(structure)),
                                        ttg::device::send<1>(key, node));
#else  // MRA_ENABLE_HOST
      ttg::send<0>(key, std::move(structure));
      ttg::send<1>(key, node);
#endif // MRA_ENABLE_HOST
      if (key.level() == 0 && U_fixed) {
        matrix_type U = batch_block<T>(*U_fixed, *fns_in, *fns_out, key.batch(), name);
#ifndef MRA_ENABLE_HOST
        sends.push_back(ttg::device::send<2>(key, std::move(U)));
#else  // MRA_ENABLE_HOST
        ttg::send<2>(key, std::move(U));
#endif // MRA_ENABLE_HOST
      }
#ifndef MRA_ENABLE_HOST
      co_await std::move(sends);
#endif // MRA_ENABLE_HOST
    };

    auto dispatch_tt = ttg::make_tt<Space>(std::move(dispatch_fn),
                                           ttg::edges(in), ttg::edges(node_p, node_t, U_p),
                                           name + "-dispatch");

    /**
     * Pass U to the transform task at this key and to the children that receive input
     * nodes: U where some output function continues, an empty U (no output there or
     * below) where only functions that no output depends on do. Only host metadata is
     * read, so this runs ahead of the kernels.
     */
    /* The propagate task is a host task (also in device builds): it is the only task that
     * receives U from another process, and a device task receives remote buffers into device
     * memory only, while the transform task reads U on the host (transform_structure). As a
     * host task it receives U into host memory and passes it on locally. */
    auto propagate_fn = [fns_in, fns_out, name](const mra::Key<NDIM>& key,
                                                const structure_type& in_node,
                                                const matrix_type& U) {
      if (in_node.is_ns()) {
        throw std::runtime_error(name + ": input must be compressed, not non-standard");
      }
      const bool skip = (U.dim(0) == 0);
      if (!skip && (U.dim(0) != fns_in->num_functions(key) || U.dim(1) != fns_out->num_functions(key))) {
        throw std::runtime_error(name + ": U must be (number of input functions) x (number of output functions)");
      }

      ttg::send<0>(key, U);

      /* the output functions that continue below each child */
      std::array<bool, num_children> live{};
      if (!skip) {
        const size_type N = fns_in->num_functions(key), M = fns_out->num_functions(key);
        SparsityInfo sparsity(M, SparsityInfo::InitType::AllZero);
        std::vector<std::array<bool, num_children>> has_child;
        detail::transform_structure<T, NDIM>(in_node, U.buffer().host_ptr(), N, M, has_child, sparsity);
        for (size_type j = 0; j < M; ++j)
          for (int c = 0; c < num_children; ++c) live[c] = live[c] || has_child[j][c];
      }
      std::vector<mra::Key<NDIM>> U_children, skip_children;
      int c = 0;
      for (auto child : children(key)) {
        if (!in_node.is_all_child_leaf(child)) (live[c] ? U_children : skip_children).push_back(child);
        ++c;
      }
      if (!U_children.empty()) {
        ttg::broadcast<1>(std::move(U_children), U);
      }
      if (!skip_children.empty()) {
        matrix_type none(std::array<size_type, 2>{0, 0});
        ttg::broadcast<1>(std::move(skip_children), std::move(none));
      }
    };

    auto propagate_tt = ttg::make_tt<ttg::ExecutionSpace::Host>(std::move(propagate_fn),
                                            ttg::edges(node_p, U_fixed ? U_p : ttg::fuse(U_in, U_p)),
                                            ttg::edges(U_t, U_p),
                                            name + "-propagate", {"node", "U"}, {"U", "child U"});

    auto transform_fn = [fns_in, fns_out, K, enable_batching, batch_pool, total_functions](
                                                      const mra::Key<NDIM>& key,
                                                      const node_type& in_node,
                                                      const matrix_type& U) -> TASKTYPE {
      const size_type N = fns_in->num_functions(key);
      const size_type M = fns_out->num_functions(key);
      if (U.dim(0) == 0) {   // no output function here or below (see propagate)
#ifndef MRA_ENABLE_HOST
        co_return;
#else  // MRA_ENABLE_HOST
        return;
#endif // MRA_ENABLE_HOST
      }

      /* output functions with a non-zero node here, and with children below here */
      SparsityInfo sparsity(M, SparsityInfo::InitType::AllZero);
      std::vector<std::array<bool, num_children>> has_child;
      detail::transform_structure<T, NDIM>(in_node, U.buffer().host_ptr(), N, M, has_child, sparsity);

      const size_type n_nonzero = sparsity.count_nonzero();
      node_type out = (n_nonzero > 0) ? node_type(key, sparsity, K, ttg::scope::Allocate)
                                      : node_type(key, M);
      for (size_type j = 0; j < M; ++j) {
        for (int c = 0; c < num_children; ++c) {
          out.set_child_leaf(j, c, !has_child[j][c]);
        }
      }
      if (n_nonzero > 0) {
#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(out.coeffs().buffer(), U.buffer());
        if (!in_node.empty()) input.add(in_node.coeffs().buffer());
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST
        auto in_view = in_node.coeffs().current_view();
        auto out_view = out.coeffs().current_view();
        const T* U_ptr = U.buffer().current_device_ptr();
#ifndef MRA_ENABLE_HOST
        if (enable_batching) {
          /* the output's sparsity is pushed by the leader along with the batch */
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(in_view, out_view, N, M, out.coeffs(), n_nonzero);
          detail::submit_transform_functions_batch_leader<T, NDIM>(batch, *batch_pool, U_ptr, K, total_functions);
        } else
#endif // MRA_ENABLE_HOST
        {
          auto sparseman = make_sparsity_manager(out);
          sparseman.populate_device_sparsity();
          submit_transform_functions_kernel<T, NDIM>(key, in_view, out_view, U_ptr, N, M, n_nonzero, K,
                                                     ttg::device::current_stream());
        }
      }
#ifndef MRA_ENABLE_HOST
      co_await ttg::device::send<0>(key, std::move(out));
#else  // MRA_ENABLE_HOST
      ttg::send<0>(key, std::move(out));
#endif // MRA_ENABLE_HOST
    };

    auto transform_tt = ttg::make_tt<Space>(std::move(transform_fn),
                                            ttg::edges(node_t, U_t),
                                            ttg::edges(out),
                                            name, {"node", "U"}, {"out"});

    if constexpr (!std::is_same_v<ProcMap, ttg::Void>) {
      dispatch_tt->set_keymap(procmap);
      propagate_tt->set_keymap(procmap);
      transform_tt->set_keymap(procmap);
    }
    if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) {
      dispatch_tt->set_devicemap(devicemap);
      /* no device map for the (host) propagate task: with one, a remote U is received on a device */
      transform_tt->set_devicemap(devicemap);
    }
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      /* the tasks of a batch of functions share U: its block when there are several */
      transform_tt->set_batch_matcher([](const mra::Key<NDIM>& a, const mra::Key<NDIM>& b) { return a.batch() == b.batch(); },
                                      max_batch_size);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(dispatch_tt->template in<0>());
    auto outs = std::make_tuple(transform_tt->template out<0>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(3);
    ops[0] = std::move(dispatch_tt);
    ops[1] = std::move(propagate_tt);
    ops[2] = std::move(transform_tt);
    return make_ttg(std::move(ops), ins, outs, name);
  }
  } // namespace detail

  /// with U fixed when the graph is built
  template <typename T, Dimension NDIM, typename FunctionSetIn, typename FunctionSetOut,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_transform(const std::shared_ptr<FunctionSetIn>& fns_in,
                      const std::shared_ptr<FunctionSetOut>& fns_out,
                      size_type K,
                      const std::shared_ptr<mra::DenseTensor<T, 2>> U,
                      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
                      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
                      const std::string& name = "transform",
                      ProcMap procmap = {},
                      DeviceMap devicemap = {})
  {
    if (U->dim(0) != fns_in->num_functions() || U->dim(1) != fns_out->num_functions()) {
      throw std::runtime_error(name + ": U must be (number of input functions) x (number of output functions)");
    }
    if (fns_in->num_batches() != fns_out->num_batches()) {
      throw std::runtime_error(name + ": the input and output must have the same number of batches");
    }
    /* each batch transforms its own functions: refuse a U that couples batches now, not in a task */
    for (Batch b = 0; fns_in->num_batches() > 1 && b < Batch(fns_in->num_batches()); ++b) {
      (void)detail::batch_block<T>(*U, *fns_in, *fns_out, b, name);
    }
    return detail::make_transform_impl<T, NDIM>(fns_in, fns_out, K, ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>>(),
                                                U, in, out, name, procmap, devicemap);
  }

  /**
   * With U computed while the graph executes: U arrives on `U` at the root key (once),
   * e.g., from a task that computes it from a reduction over another set of functions.
   * The input nodes wait in the graph until it is there.
   */
  template <typename T, Dimension NDIM, typename FunctionSetIn, typename FunctionSetOut,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_transform(const std::shared_ptr<FunctionSetIn>& fns_in,
                      const std::shared_ptr<FunctionSetOut>& fns_out,
                      size_type K,
                      ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>> U,
                      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> in,
                      ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> out,
                      const std::string& name = "transform",
                      ProcMap procmap = {},
                      DeviceMap devicemap = {})
  {
    return detail::make_transform_impl<T, NDIM>(fns_in, fns_out, K, U, nullptr, in, out, name, procmap, devicemap);
  }

} // namespace mra

#endif // MRA_TASKS_TRANSFORM_H
