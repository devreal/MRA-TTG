#ifndef MRA_TENSOR_COMBINEDNODE_H
#define MRA_TENSOR_COMBINEDNODE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <ttg.h>
#include <ttg/serialization/std/vector.h>

#include "mra/misc/key.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/sparsityinfo.h"
#include "mra/tensor/functionnode.h"
#include "mra/tensor/sparsitymanager.h"

namespace mra {

  /**
   * Reconstructed nodes at several keys in one buffer, so they travel as one data
   * copy (one select, one send or broadcast).
   *
   * Each slot has the storage layout of a SparseTensor of shape (M, K, ..., K): one
   * SparsityState byte per function, padded (align_to_type), then the coefficients of
   * the allocated functions, packed in order (SparseArrayBase::offset_of). A slot's
   * view is therefore an ordinary SparseTensorView and kernels need no changes. Unlike
   * a Tensor, which reserves room for every function, a slot holds only its allocated
   * functions; slots start aligned to MRA_BUFFER_ALIGNMENT.
   *
   * The sparsity bytes and leaf statuses of every slot are host metadata that travels
   * with the node. The sparsity bytes must be copied into the device buffer before a
   * kernel reads or writes a slot: populate_device_sparsity() unbatched, or one extra
   * copy per slot in a batch's grouped copy (see slot_sparsity_data()).
   *
   * A slot has the same layout as the start of a FunctionsReconstructedNode's buffer with
   * the same sparsity, so a slot is turned into a node by copying its slot_size() values
   * to the start of the node's buffer (make_node() creates the node).
   */
  template <typename T, Dimension NDIM>
  class CombinedReconstructedNode : public ttg::TTValue<CombinedReconstructedNode<T, NDIM>> {
  public:
    using value_type = T;
    using key_type = Key<NDIM>;
    using view_type = SparseTensorView<T, NDIM+1>;
    using const_view_type = SparseTensorView<const T, NDIM+1>;
    using buffer_type = ttg::Buffer<T, DeviceAllocator<T>>;
    using node_type = FunctionsReconstructedNode<T, NDIM>;

  private:
    struct slot_type {
      key_type key;
      size_type offset = 0; // in values of T from the start of the buffer
      std::vector<detail::SparsityState> sparsity;
      std::vector<LeafStatus> status;
      std::uint8_t tag = 0; // what the slot holds, for the user of the node

      template <typename Archive>
      void serialize(Archive& ar) {
        ar & key & offset & sparsity & status & tag;
      }

      template <typename Archive>
      void serialize(Archive& ar, const unsigned int) {
        serialize(ar);
      }
    };

    key_type m_key = key_type::invalid();
    size_type m_K = 0;
    size_type m_size = 0; // values of T over all slots, with padding
    std::vector<slot_type> m_slots;
    buffer_type m_buffer;

    static size_type header_size(size_type M) {
      return detail::align_to_type<T>(M) / sizeof(T);
    }

    static size_type align_offset(size_type offset) {
      constexpr size_type align = std::max<size_type>(MRA_BUFFER_ALIGNMENT / sizeof(T), 1);
      return (offset + align - 1) / align * align;
    }

    size_type num_allocated(size_type s) const {
      return std::count_if(m_slots[s].sparsity.begin(), m_slots[s].sparsity.end(),
                           [](detail::SparsityState st) { return st & detail::SparsityState::ALLOCATED; });
    }

  public:
    /* needed for serialization */
    CombinedReconstructedNode() = default;

    /** an empty node whose slots are added with add_slot() before allocate() */
    CombinedReconstructedNode(const key_type& key, size_type K)
    : m_key(key)
    , m_K(K)
    { }

    CombinedReconstructedNode(CombinedReconstructedNode&&) = default;
    CombinedReconstructedNode& operator=(CombinedReconstructedNode&&) = default;
    CombinedReconstructedNode(const CombinedReconstructedNode&) = delete;
    CombinedReconstructedNode& operator=(const CombinedReconstructedNode&) = delete;

    /**
     * Adds a slot for the node at `key` with the sparsity and leaf status of each of
     * its functions; returns the slot's index. Only before allocate().
     */
    size_type add_slot(const key_type& key, const SparsityInfo& sparsity, std::vector<LeafStatus> status,
                       std::uint8_t tag = 0) {
      if (!m_buffer.empty()) throw std::logic_error("CombinedReconstructedNode: add_slot after allocate");
      const size_type M = sparsity.dim(0);
      if (status.size() != M) throw std::logic_error("CombinedReconstructedNode: one status per function");
      slot_type slot;
      slot.key = key;
      slot.offset = align_offset(m_size);
      slot.sparsity.resize(M);
      detail::sparsity_to_bytes(sparsity, slot.sparsity.data(), M);
      slot.status = std::move(status);
      slot.tag = tag;
      m_slots.push_back(std::move(slot));
      m_size = m_slots.back().offset + slot_size(m_slots.size() - 1);
      return m_slots.size() - 1;
    }

    /** allocates the buffer for the slots added so far */
    void allocate(ttg::scope scope = ttg::scope::Allocate) {
      if (!m_buffer.empty()) throw std::logic_error("CombinedReconstructedNode: allocated twice");
      m_buffer = buffer_type(std::max<size_type>(m_size, 1), scope);
    }

    const key_type& key() const { return m_key; }
    size_type K() const { return m_K; }
    size_type num_slots() const { return m_slots.size(); }
    bool empty() const { return m_buffer.empty(); }

    buffer_type& buffer() { return m_buffer; }
    const buffer_type& buffer() const { return m_buffer; }

    /** index of the slot of `key`, or -1 */
    int find_slot(const key_type& key) const {
      for (size_type s = 0; s < m_slots.size(); ++s) if (m_slots[s].key == key) return s;
      return -1;
    }

    const key_type& slot_key(size_type s) const { return m_slots[s].key; }
    /** number of functions (M) of slot s */
    size_type slot_count(size_type s) const { return m_slots[s].status.size(); }
    LeafStatus slot_status(size_type s, size_type m) const { return m_slots[s].status[m]; }
    const std::vector<LeafStatus>& slot_statuses(size_type s) const { return m_slots[s].status; }
    size_type slot_offset(size_type s) const { return m_slots[s].offset; }
    std::uint8_t slot_tag(size_type s) const { return m_slots[s].tag; }

    /** values of T slot s occupies: sparsity bytes (padded) and the allocated functions */
    size_type slot_size(size_type s) const {
      const size_type K2NDIM = std::pow(m_K, NDIM);
      return header_size(slot_count(s)) + num_allocated(s) * K2NDIM;
    }

    /** the sparsity bytes of slot s, to be copied to the start of the slot on the device */
    const detail::SparsityState* slot_sparsity_data(size_type s) const { return m_slots[s].sparsity.data(); }

    /** the sparsity of slot s */
    SparsityInfo slot_sparsity(size_type s) const {
      const size_type M = slot_count(s);
      SparsityInfo sparsity(M, SparsityInfo::InitType::AllZero);
      for (size_type m = 0; m < M; ++m) {
        const auto st = m_slots[s].sparsity[m];
        if (st & detail::SparsityState::NONZERO) sparsity.set_nonzero(m);
        else if (st & detail::SparsityState::ALLOCATED) sparsity.set_allocated(m);
      }
      return sparsity;
    }

    /** view of slot s in the current memory space */
    view_type slot_view(size_type s) {
      return view_type(m_buffer.current_device_ptr() + slot_offset(s), make_dims<NDIM+1>(slot_count(s), m_K));
    }

    /* the view type of a non-const node, as Tensor::current_view() const, so kernels see one type */
    const view_type slot_view(size_type s) const {
      return view_type(const_cast<T*>(m_buffer.current_device_ptr()) + slot_offset(s),
                       make_dims<NDIM+1>(slot_count(s), m_K));
    }

    /**
     * Copies the sparsity bytes of every slot to the buffer on `device`, from inside a
     * task (the device's current stream), as SparsityManager does for a tensor.
     */
    void populate_device_sparsity(ttg::device::Device device = ttg::device::current_device()) {
      if (device.is_host()) {
        T* base = m_buffer.host_ptr();
        for (size_type s = 0; s < m_slots.size(); ++s) {
          std::memcpy(base + slot_offset(s), slot_sparsity_data(s), slot_count(s));
        }
        return;
      }
#ifndef MRA_ENABLE_HOST
      assert(m_buffer.is_current_on(device));
      size_type total = 0;
      for (size_type s = 0; s < m_slots.size(); ++s) total += slot_count(s);
      auto& pool = detail::sparsity_pool_registry().get(device);
      auto& staging = pool.acquire(total);
      staging.host_args.resize(total);
      parsec_device_gpu_module_t* device_module = ttg_parsec::detail::parsec_ttg_caller->dev_ptr->device;
      auto* stream = ttg_parsec::detail::parsec_ttg_caller->dev_ptr->stream;
      T* base = const_cast<T*>(m_buffer.device_ptr_on(device));
      size_type pos = 0;
      for (size_type s = 0; s < m_slots.size(); ++s) {
        std::memcpy(&staging.host_args[pos], slot_sparsity_data(s), slot_count(s));
        int ret = device_module->memcpy_async(device_module, stream, base + slot_offset(s), &staging.host_args[pos],
                                              slot_count(s), parsec_device_gpu_transfer_direction_h2d);
        if (ret != PARSEC_SUCCESS) throw std::runtime_error("CombinedReconstructedNode: sparsity copy failed");
        pos += slot_count(s);
      }
      pool.mark_submitted(staging, ttg::device::current_stream());
#endif // !MRA_ENABLE_HOST
    }

    /**
     * A node at slot s's key with its sparsity and leaf statuses, allocated: copy
     * slot_size(s) values from the slot to the start of its buffer (copy_slot_to()) to
     * fill it.
     */
    node_type make_node(size_type s, ttg::scope scope = ttg::scope::Allocate) const {
      node_type node(slot_key(s), slot_sparsity(s), m_K, scope);
      for (size_type m = 0; m < slot_count(s); ++m) node.set_leaf(m, slot_status(s, m));
      return node;
    }

    /**
     * Copies slot s, sparsity bytes included, to the start of node's buffer on `device`,
     * from inside a task (the device's current stream). node comes from make_node(s).
     */
    void copy_slot_to(size_type s, node_type& node, ttg::device::Device device = ttg::device::current_device()) const {
      const std::size_t bytes = slot_size(s) * sizeof(T);
      if (device.is_host()) {
        std::memcpy(node.coeffs().buffer().host_ptr(), m_buffer.host_ptr() + slot_offset(s), bytes);
        return;
      }
#ifndef MRA_ENABLE_HOST
      parsec_device_gpu_module_t* device_module = ttg_parsec::detail::parsec_ttg_caller->dev_ptr->device;
      auto* stream = ttg_parsec::detail::parsec_ttg_caller->dev_ptr->stream;
      int ret = device_module->memcpy_async(device_module, stream,
                                            const_cast<T*>(node.coeffs().buffer().device_ptr_on(device)),
                                            const_cast<T*>(m_buffer.device_ptr_on(device) + slot_offset(s)),
                                            bytes, parsec_device_gpu_transfer_direction_d2d);
      if (ret != PARSEC_SUCCESS) throw std::runtime_error("CombinedReconstructedNode: slot copy failed");
#endif // !MRA_ENABLE_HOST
    }

    template <typename Archive>
    void serialize(Archive& ar) {
      ar & m_key & m_K & m_size & m_slots & m_buffer;
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int) {
      serialize(ar);
    }
  };

} // namespace mra

#endif // MRA_TENSOR_COMBINEDNODE_H
