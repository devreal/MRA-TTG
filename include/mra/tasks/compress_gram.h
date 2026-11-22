#ifndef MRA_TASKS_COMPRESS_GRAM_H
#define MRA_TASKS_COMPRESS_GRAM_H

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <ttg.h>
#include "mra/kernels.h"
#include "mra/kernels/inner_product.h"
#include "mra/tasks/common.h"
#include "mra/misc/batch_group.h"
#include "mra/misc/batch_size.h"
#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/functiondata.h"
#include "mra/misc/functionset.h"
#include "mra/tensor/sparsitymanager.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/functionnode.h"

namespace mra {

  namespace detail {
#ifndef MRA_ENABLE_HOST
    /** the batch pools of make_compress_gram: its compress and inner product launches */
    template <typename T, Dimension NDIM>
    struct CompressGramPools {
      std::shared_ptr<GroupedBatchPoolRegistry<CompressBatchArg<T, NDIM>>> compress;
      std::shared_ptr<GroupedBatchPoolRegistry<InnerBatchArg<T, NDIM>>> inner;
    };

    /**
     * The inner product launch of a batch of make_compress_gram tasks, after their compress
     * launch (submit_compress_batch_leader) on the same stream: r = <d_i|d_j> of each member's
     * compressed node d plus its children's partial results. The members' coop arguments are
     * those of compress followed by (r, child results, N), see make_compress_gram.
     */
    template <typename T, Dimension NDIM, typename BatchView>
    void submit_gram_batch_leader(BatchView& batch, GroupedBatchPoolRegistry<InnerBatchArg<T, NDIM>>& registry,
                                  size_type K)
    {
      using arg_t = InnerBatchArg<T, NDIM>;
      if (!batch.is_leader()) return;
      BatchRange batch_range("compress_gram", batch.size());
      const std::size_t nb = batch.size();
      size_type total_blocks = 0;
      for (std::size_t m = 0; m < nb; ++m) total_blocks += batch[m].template get<14>();
      if (total_blocks == 0) return;
      auto& pool = registry.get(ttg::device::current_device());
      auto& slot = pool.acquire(nb, nb + 1, 0);
      slot.args.clear();
      slot.offsets.resize(nb + 1);
      slot.offsets[0] = 0;
      slot.sparsity.clear();
      slot.extra_dsts.clear();
      slot.extra_srcs.clear();
      slot.extra_sizes.clear();
      for (std::size_t m = 0; m < nb; ++m) {
        auto& b = batch[m];
        const size_type N = b.template get<14>();
        slot.args.emplace_back(b.template get<3>(), b.template get<3>(), b.template get<12>(),
                               b.template get<13>(), N, N);
        slot.offsets[m + 1] = slot.offsets[m] + N;
      }
      auto stream = ttg::device::current_stream();
      submit_grouped_copy<arg_t>(slot, nb, slot.offsets.size(), pool.device_id, stream);
      launch_inner_kernel_batched<T, NDIM, true>(slot.dev_args, slot.dev_offsets, static_cast<size_type>(nb),
                                                 total_blocks, K, stream);
      pool.mark_submitted(slot, stream);
    }
#endif // MRA_ENABLE_HOST
  } // namespace detail

  /**
   * The Gram matrix r[i,j] = <f_i|f_j> of a set of N functions given in reconstructed form,
   * computed while compressing them: compress followed by make_matrix_inner of the result
   * with itself, without the compressed tree. Each compress task forms its node's
   * compressed coefficients d as usual, but only for the duration of the task: the same
   * task (batch) also launches the inner product of d with itself and adds the partial
   * results of its children, which travel up the tree with the scaling coefficients. The
   * result (N x N, row-major) is sent on `result` at the root key.
   *
   * Without truncation, and the input tree must not be in non-standard form.
   */
  template <typename T, Dimension NDIM, typename FunctionSetT,
            typename ProcMap = ttg::Void, typename DeviceMap = ttg::Void>
  auto make_compress_gram(const std::shared_ptr<FunctionSetT>& fns,
                          size_type K,
                          const mra::FunctionData<T, NDIM>& functiondata,
                          ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>>& in,
                          ttg::Edge<mra::Key<NDIM>, mra::DenseTensor<T, 2>>& result,
                          const std::string& name = "compress_gram",
                          ProcMap procmap = {},
                          DeviceMap devicemap = {},
                          const BatchGroup& batch_group = {})
  {
    static_assert(NDIM == 3);
    using node_type = mra::FunctionsReconstructedNode<T, NDIM>;
    using result_type = mra::DenseTensor<T, 2>;
    constexpr std::size_t num_children = mra::Key<NDIM>::num_children();
    constexpr auto children_seq = std::make_index_sequence<num_children>{};

    const bool enable_batching = mra::batching_enabled();
    const std::size_t max_batch_size = mra::get_batch_size();
    const size_type total_functions = fns->num_functions();
#ifndef MRA_ENABLE_HOST
    std::shared_ptr<detail::CompressGramPools<T, NDIM>> pools;
    if (enable_batching) {
      pools = batch_group.pool<detail::CompressGramPools<T, NDIM>>([&] {
        auto p = new detail::CompressGramPools<T, NDIM>;
        p->compress = detail::shared_pool_registry<detail::CompressBatchArg<T, NDIM>>(max_batch_size);
        p->inner = detail::shared_pool_registry<detail::InnerBatchArg<T, NDIM>>(max_batch_size);
        return p;
      }, K);
    }
#else
    std::nullptr_t pools = nullptr;
#endif // MRA_ENABLE_HOST

    auto p_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, node_type>{})...);
    }(children_seq);
    auto r_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, result_type>{})...);
    }(children_seq);
    ttg::Edge<mra::Key<NDIM>, node_type> filter_in(name + "-filter_in");

    /* the nodes with children go to compress */
    auto filter_fn = [](const mra::Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      if (!node.is_all_leaf_or_invalid()) {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<0>(key, node);
#else
        ttg::send<0>(key, node);
#endif
      }
    };

    /* a leaf goes up as is, with an empty partial result (it has no wavelet coefficients) */
    auto leaf_fn = [](const mra::Key<NDIM>& key, const node_type& node) -> TASKTYPE {
      if (!node.any_have_children()) {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(
          select_send_up(key, node, std::make_index_sequence<num_children>{}, "compress_gram-leaf"),
          select_send_up<num_children>(key, result_type(), std::make_index_sequence<num_children>{},
                                       "compress_gram-leaf-r"));
#else
        select_send_up(key, node, std::make_index_sequence<num_children>{}, "compress_gram-leaf");
        select_send_up<num_children>(key, result_type(), std::make_index_sequence<num_children>{},
                                     "compress_gram-leaf-r");
#endif
      }
    };

    auto do_compress = [fns, K, &functiondata, name, enable_batching, pools, total_functions](
                          const mra::Key<NDIM>& key,
                          const node_type& in0, const node_type& in1, const node_type& in2, const node_type& in3,
                          const node_type& in4, const node_type& in5, const node_type& in6, const node_type& in7,
                          const result_type& r0, const result_type& r1, const result_type& r2,
                          const result_type& r3, const result_type& r4, const result_type& r5,
                          const result_type& r6, const result_type& r7,
                          const node_type& in) -> TASKTYPE {
      const size_type N = fns->num_functions(key);
      constexpr auto out_terminal_id = 2*num_children;
      std::array<const result_type*, num_children> rchildren = {&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7};
      node_type p(key, N);
      result_type r;
      /* the compressed node and the scratch buffers live until the kernels using them are done,
       * i.e. past the final forward/send, not just to the end of the block launching them */
      mra::FunctionsCompressedNode<T, NDIM> d(key, N);
      ttg::Buffer<T, DeviceAllocator<T>> tmp_scratch;
      ttg::Buffer<T, DeviceAllocator<T>> d_sumsq;

      const bool all_empty = in.empty() && in0.empty() && in1.empty() && in2.empty() && in3.empty() &&
                             in4.empty() && in5.empty() && in6.empty() && in7.empty();
      if (all_empty) {
        /* without truncation, a node without coefficients below has no contributions below */
        for (auto rc : rchildren) {
          if (!rc->empty()) throw std::logic_error(name + ": partial result below a node without coefficients");
        }
      } else {
        SparsityInfo sparsity(N, SparsityInfo::InitType::AllZero);
        sparsity.nonzero_if_any(in0, in1, in2, in3, in4, in5, in6, in7);
        /* the compressed node, only for the duration of this task */
        d.allocate(sparsity, K, ttg::scope::Allocate);
        d.set_ns(false);
        sparsity.set_all_zero();
        sparsity.nonzero_if_any(d, in);
        p.allocate(sparsity, K, ttg::scope::Allocate);
        const size_type n_nonzero = sparsity.count_nonzero();

        r = result_type(std::array<size_type, 2>{N, N}, ttg::scope::Allocate);
#ifndef MRA_ENABLE_HOST
        const bool use_batching = enable_batching;
#else
        const bool use_batching = false;
#endif
        const std::size_t tmp_size = use_batching ? 0 : compress_tmp_size<NDIM>(K)*n_nonzero;
        if (tmp_size > 0) tmp_scratch.reset(tmp_size, ttg::scope::Allocate);
        /* the sums are scratch here: in the batch's slot with results_via_slot() */
#ifndef MRA_ENABLE_HOST
        const bool slot_results = use_batching && detail::results_via_slot();
#else
        const bool slot_results = false;
#endif
        if (!slot_results) d_sumsq.reset(std::max<size_type>(n_nonzero, 1), ttg::scope::Allocate);
        const auto& hgT = functiondata.get_hgT();
        auto& dc = d.coeffs();

#ifndef MRA_ENABLE_HOST
        auto input = ttg::device::Input(p.coeffs().buffer(), dc.buffer(), hgT.buffer(), r.buffer());
        if (!slot_results) input.add(d_sumsq);
        if (tmp_size > 0) input.add(tmp_scratch);
        auto select_in = [&](const node_type& n) {
          if (!n.empty()) input.add(n.coeffs().buffer());
        };
        select_in(in);
        select_in(in0); select_in(in1); select_in(in2); select_in(in3);
        select_in(in4); select_in(in5); select_in(in6); select_in(in7);
        for (auto rc : rchildren) {
          if (!rc->empty()) input.add(rc->buffer());
        }
        co_await ttg::device::select(input);
#endif // MRA_ENABLE_HOST

        auto input_views = std::array{in0.coeffs().current_view(), in1.coeffs().current_view(),
                                      in2.coeffs().current_view(), in3.coeffs().current_view(),
                                      in4.coeffs().current_view(), in5.coeffs().current_view(),
                                      in6.coeffs().current_view(), in7.coeffs().current_view()};
        auto in_view = in.coeffs().current_view();
        auto coeffs_view = p.coeffs().current_view();
        auto rcoeffs_view = dc.current_view();
        auto hgT_view = hgT.current_view();
        T* r_ptr = r.buffer().current_device_ptr();
        std::array<const T*, num_children> child_results;
        for (std::size_t c = 0; c < num_children; ++c) {
          child_results[c] = rchildren[c]->empty() ? nullptr : rchildren[c]->buffer().current_device_ptr();
        }

#ifndef MRA_ENABLE_HOST
        if (use_batching) {
          /* the arguments of compress (see make_compress), then those of the inner product */
          /* coop holds its arguments by reference: no temporaries */
          const std::array<const mra::KeptInfo<T>*, num_children>* no_kept = nullptr;
          T no_trunc_tol = T{};
          auto no_ns_view = decltype(in_view){};
          size_type n_functions = N;
          auto batch = co_await ttg::device::coop<mra::Key<NDIM>>(key, in_view, coeffs_view, rcoeffs_view,
                                                                  d_sumsq, input_views, p.coeffs(), dc, n_nonzero,
                                                                  no_kept, no_trunc_tol, no_ns_view,
                                                                  r_ptr, child_results, n_functions);
          detail::submit_compress_batch_leader<T, NDIM>(batch, *pools->compress, K, false, hgT_view,
                                                        total_functions, false);
          detail::submit_gram_batch_leader<T, NDIM>(batch, *pools->inner, K);
        } else
#endif // MRA_ENABLE_HOST
        {
          auto stream = ttg::device::current_stream();
          auto sparseman = make_sparsity_manager(dc, p);
          sparseman.populate_device_sparsity();
          const unsigned char* no_child_kept = nullptr;
          submit_compress_kernel(key, N, n_nonzero, K, false, in_view, coeffs_view, rcoeffs_view, hgT_view,
                                 tmp_scratch.current_device_ptr(), d_sumsq.current_device_ptr(), input_views,
                                 decltype(in_view){}, no_child_kept, T{}, stream);
          /* the views make_matrix_inner passes (of a const node) */
          auto d_view = std::as_const(d).coeffs().current_view();
          submit_matrix_inner_kernel<T, NDIM>(key, d_view, d_view, r_ptr, child_results, N, N, K, stream);
        }
      }

      if (key.level() > 0) {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::forward(
          select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress_gram"),
          select_send_up<num_children>(key, std::move(r), std::make_index_sequence<num_children>{},
                                       "compress_gram-r"));
#else
        select_send_up(key, std::move(p), std::make_index_sequence<num_children>{}, "compress_gram");
        select_send_up<num_children>(key, std::move(r), std::make_index_sequence<num_children>{}, "compress_gram-r");
#endif
      } else {
#ifndef MRA_ENABLE_HOST
        co_await ttg::device::send<out_terminal_id>(key, std::move(r));
#else
        ttg::send<out_terminal_id>(key, std::move(r));
#endif
      }
    };

    auto in_edges = std::tuple_cat(p_edges, r_edges, std::make_tuple(filter_in));
    auto out_edges = std::tuple_cat(p_edges, r_edges, std::make_tuple(result));
    auto compress_tt = ttg::make_tt<Space>(std::move(do_compress), in_edges, out_edges, name);
    auto filter_tt = ttg::make_tt<Space>(std::move(filter_fn), ttg::edges(in), ttg::edges(filter_in), name + "-filter");
    auto leaf_tt = ttg::make_tt<Space>(std::move(leaf_fn), ttg::edges(in), std::tuple_cat(p_edges, r_edges),
                                       name + "-leaves");
    auto set_maps = [&](auto& tt) {
      if constexpr (!std::is_same_v<ProcMap, ttg::Void>) tt->set_keymap(procmap);
      if constexpr (!std::is_same_v<DeviceMap, ttg::Void>) tt->set_devicemap(devicemap);
    };
    set_maps(compress_tt);
    set_maps(filter_tt);
    set_maps(leaf_tt);
#ifndef MRA_ENABLE_HOST
    if (enable_batching) {
      compress_tt->set_batch_matcher([](const mra::Key<NDIM>&, const mra::Key<NDIM>&) { return true; },
                                     max_batch_size);
      batch_group.add(compress_tt);
    }
#endif // MRA_ENABLE_HOST

    auto ins = std::make_tuple(filter_tt->template in<0>());
    auto outs = std::make_tuple(compress_tt->template out<2*num_children>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops;
    ops.push_back(std::move(filter_tt));
    ops.push_back(std::move(leaf_tt));
    ops.push_back(std::move(compress_tt));
    return make_ttg(std::move(ops), ins, outs, name);
  }

} // namespace mra

#endif // MRA_TASKS_COMPRESS_GRAM_H
