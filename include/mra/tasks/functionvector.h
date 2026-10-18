#ifndef MRA_TASKS_FUNCTIONVECTOR_H
#define MRA_TASKS_FUNCTIONVECTOR_H

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ttg.h>
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/tensor/functionnode.h"

namespace mra {

  /**
   * A vector of N functions in compressed form that outlives graph executions.
   *
   * The nodes produced by one graph are stored with make_store() and fed into a
   * later graph with make_feed(). The nodes are moved in and out, never copied,
   * so their coefficient buffers stay wherever TTG left them: on a device build
   * the data stays in device memory (unless evicted by TTG) and does not travel
   * through the host between graphs.
   *
   * Every process holds the nodes that were stored on it, i.e., whatever the
   * keymap of the store task assigned to it. Feeding sends each process's nodes
   * from that process.
   *
   * Only compressed (not non-standard) trees are supported.
   *
   * Within one graph, a vector may be fed and stored to as long as make_feed()
   * is called first: feeding without `keep` takes the nodes out of the vector
   * when the graph is built, so the vector is empty and ready for new nodes when
   * it executes. A vector fed with `keep` must not be stored to in the same graph.
   *
   * Nodes are not copyable, so a node fed with `keep` may only be read by the
   * graph: putting it back takes it over, and TTG cannot hand the same node to a
   * second task that takes it over. That includes tasks downstream of operations
   * that forward an input node unchanged, e.g., make_gaxpy forwards a node whose
   * scalar is 1 where the other operand is empty, so a kept node must not reach
   * a store (or make_truncate) that way.
   */
  template <typename T, Dimension NDIM>
  class FunctionVector {
  public:
    using node_type = FunctionsCompressedNode<T, NDIM>;
    using key_type = Key<NDIM>;
    using map_type = std::map<key_type, node_type>;

  private:
    struct Impl {
      size_type size = 0;
      map_type nodes;
      std::mutex mtx;

      void insert(const key_type& key, node_type&& node, const std::string& name) {
        if (node.is_ns()) {
          throw std::runtime_error(name + ": FunctionVector only holds compressed, not non-standard, nodes");
        }
        std::lock_guard<std::mutex> lock(mtx);
        auto [it, inserted] = nodes.emplace(key, std::move(node));
        if (!inserted) {
          throw std::runtime_error(name + ": node stored twice");
        }
        reset_ttg_state(it->second);
      }

      /**
       * A node is its own TTG data copy (it derives from ttg::TTValue), and moving it
       * carries over the bookkeeping of the copy it was moved from: the number of
       * readers (or the mark of a task about to modify it), the deferred writer, and
       * the reference count. Taken out of the graph that state is stale, and a node
       * fed into a later graph would start out as if a task were writing to it.
       * Reset it to that of a newly constructed node.
       */
      static void reset_ttg_state(node_type& node) {
        node.reset_readers();   // modified by a task -> one reader
        while (node.num_readers() > 1) node.template decrement_readers<false>();
        while (node.num_readers() < 1) node.template increment_readers<false>();
        node.set_next_task(nullptr);
        while (node.num_ref() > 1) node.drop_ref();
        while (node.num_ref() < 1) node.add_ref();
      }
    };
    /* heap-allocated so that the tasks of a graph can hold on to it while the vector is moved */
    std::unique_ptr<Impl> m_impl;

  public:
    explicit FunctionVector(size_type N = 0)
    : m_impl(std::make_unique<Impl>())
    {
      m_impl->size = N;
    }

    FunctionVector(FunctionVector&&) = default;
    FunctionVector& operator=(FunctionVector&&) = default;
    FunctionVector(const FunctionVector&) = delete;
    FunctionVector& operator=(const FunctionVector&) = delete;

    /// number of functions
    size_type size() const { return m_impl->size; }

    /// number of nodes held by this process
    size_type num_local_nodes() const { return m_impl->nodes.size(); }

    /// true if this process holds no nodes
    bool empty() const { return m_impl->nodes.empty(); }

    /// drop all nodes held by this process
    void clear() { m_impl->nodes.clear(); }

    /// call f(key, node) for every node held by this process (not while a graph using the vector executes)
    template <typename Fn>
    void for_each_local_node(Fn&& f) const {
      for (const auto& [key, node] : m_impl->nodes) f(key, node);
    }

    /**
     * Store the nodes arriving on `in`. The vector must be empty (or have been
     * fed without `keep` in the same graph) and takes the size `N`.
     * The task is a deferred writer: it takes the nodes over once all other
     * consumers of `in` are done with them, without copying.
     */
    auto make_store(ttg::Edge<key_type, node_type>& in, size_type N, const std::string& name = "store") {
      if (!empty()) {
        throw std::runtime_error(name + ": FunctionVector must be empty before storing into it");
      }
      m_impl->size = N;
      Impl* impl = m_impl.get();
      auto tt = ttg::make_tt([impl, name](const key_type& key, node_type&& node) {
        impl->insert(key, std::move(node), name);
      }, ttg::edges(in), ttg::edges(), name, {"input"}, {});
      tt->set_defer_writer(true);
      return tt;
    }

    /**
     * Feed the stored nodes into `out` once `control` fires (as for make_start,
     * once per batch at the root key). Without `keep`, the nodes are taken out of
     * the vector now, while the graph is built. With `keep`, they are put back
     * after all consumers of `out` are done with them, so the vector is unchanged.
     */
    auto make_feed(ttg::Edge<key_type, void>& control,
                   ttg::Edge<key_type, node_type>& out,
                   bool keep,
                   const std::string& name = "feed") {
      /* the nodes this process sends, taken out of the vector now */
      struct Source {
        map_type nodes;
        std::mutex mtx;
      };
      auto source = std::make_shared<Source>();
      source->nodes = std::move(m_impl->nodes);
      m_impl->nodes.clear();
      ttg::Edge<int, void> source_e(name + "-source");
      ttg::Edge<key_type, node_type> forward_e(name + "-forward");

      /* every process sends its own nodes: one source task per (batch, process) */
      auto dispatch_tt = ttg::make_tt([](const key_type& key) {
        const int nproc = ttg::default_execution_context().size();
        for (int p = 0; p < nproc; ++p) ttg::sendk<0>(int(key.batch()*nproc + p));
      }, ttg::edges(control), ttg::edges(source_e), name + "-dispatch", {"control"}, {"source"});

      auto source_tt = ttg::make_tt([source](const int& id) {
        const int nproc = ttg::default_execution_context().size();
        const Batch batch = id / nproc;
        std::lock_guard<std::mutex> lock(source->mtx);
        for (auto it = source->nodes.begin(); it != source->nodes.end(); ) {
          if (it->first.batch() == batch) {
            ttg::send<0>(it->first, std::move(it->second));
            it = source->nodes.erase(it);
          } else {
            ++it;
          }
        }
      }, ttg::edges(source_e), ttg::edges(forward_e), name, {"source"}, {"output"});
      source_tt->set_keymap([](const int& id) {
        return id % ttg::default_execution_context().size();
      });

      /* A node the source sends becomes a new, mutable TTG datum, which TTG cannot
       * share with read-only consumers without copying it. Moving it on from the
       * task that took it over marks it read-only, after which TTG shares it among
       * all consumers of `out` (and defers a single consumer that takes it over).
       * Same (default) keymap as the store, so the node stays on its process. */
      auto forward_tt = ttg::make_tt([](const key_type& key, node_type&& node) {
        ttg::send<0>(key, std::move(node));
      }, ttg::edges(forward_e), ttg::edges(out), name + "-forward", {"input"}, {"output"});

      auto ins = std::make_tuple(dispatch_tt->template in<0>());
      auto outs = std::make_tuple(forward_tt->template out<0>());
      std::vector<std::unique_ptr<ttg::TTBase>> ops;
      ops.push_back(std::move(dispatch_tt));
      ops.push_back(std::move(source_tt));
      ops.push_back(std::move(forward_tt));
      if (keep) {
        Impl* impl = m_impl.get();
        auto restore_tt = ttg::make_tt([impl, name](const key_type& key, node_type&& node) {
          impl->insert(key, std::move(node), name + "-restore");
        }, ttg::edges(out), ttg::edges(), name + "-restore", {"input"}, {});
        restore_tt->set_defer_writer(true);
        ops.push_back(std::move(restore_tt));
      }
      return make_ttg(std::move(ops), ins, outs, name);
    }
  };

} // namespace mra

#endif // MRA_TASKS_FUNCTIONVECTOR_H
