#ifndef MRA_SPARSITY_MANAGER_H
#define MRA_SPARSITY_MANAGER_H


#include "mra/misc/allocator.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/functionnode.h"

#include <ttg/parsec/devicefunc.h>

#ifndef MRA_ENABLE_HOST
#include "mra/misc/device_batch_pool.h"
#endif // !MRA_ENABLE_HOST

namespace mra {

#ifndef MRA_ENABLE_HOST
  namespace detail {
    /**
     * Process-wide, per-device pool of pinned staging buffers shared by every
     * SparsityManager/MockTensor construction, so that pushing a tensor's
     * sparsity bytes to a device no longer allocates a fresh pinned buffer
     * (contending on TTG's single global pinned-allocator mutex) on every
     * single task invocation. Sized in units of SparsityState (1 byte per
     * function); BatchPool grows slots on demand and reuses a slot once its
     * previous copy has completed (non-blocking event query), exactly like
     * it already does for ConvolutionBatchArg in mra/tasks/convolution.h.
     */
    inline BatchPoolRegistry<SparsityState>& sparsity_pool_registry() {
      static BatchPoolRegistry<SparsityState> registry(ttg::device::num_devices(), /* max_batch_size unused here */ 1);
      return registry;
    }
  } // namespace detail
#endif // !MRA_ENABLE_HOST

  /**
   * Manager for sparsity information of a tensor.
   * This class can be used to manage sparsity information for tensors
   * where the first dimension is sparse.
   */
  template<typename... TensorTypes>
  struct SparsityManager {

  private:

    /**
     * A simplified tensor mimicking only the sparsity aspects of the real tensor.
     */
    template<typename TensorType>
    struct MockTensor : SparseArrayBase<MockTensor<TensorType>, typename TensorType::value_type> {
      using value_type = typename TensorType::value_type;
      using sparsity_type = SparseArrayBase<MockTensor<TensorType>, value_type>;
      using tensor_type = MockTensor<TensorType>;
      using sparsity_traits = mra::sparsity_traits<sparsity_type>;

      constexpr static size_type ndim() {
        return TensorType::ndim();
      }

      MockTensor() = default;

      MockTensor(const MockTensor&) = delete;
      MockTensor(MockTensor&&) = default;
      MockTensor& operator=(const MockTensor&) = delete;
      MockTensor& operator=(MockTensor&&) = default;

#ifndef MRA_ENABLE_HOST
      MockTensor(TensorType& tensor)
      : sparsity_type()
      , m_tensor(tensor)
      , m_pool(&detail::sparsity_pool_registry().get(ttg::device::current_device()))
      , m_slot(&m_pool->acquire(byte_size()))
      {
        m_slot->host_args.resize(byte_size());
        this->apply_sparsity(m_tensor.sparsity());
      }
#else
      MockTensor(TensorType& tensor)
      : sparsity_type()
      , m_tensor(tensor)
      , m_buffer(sparsity_traits::required_space(tensor.dims()), ttg::scope::SyncIn)
      {
        this->apply_sparsity(m_tensor.sparsity());
      }
#endif // !MRA_ENABLE_HOST

      void populate_device_sparsity(ttg::device::Device device = ttg::device::current_device()) {
        if (device.is_host()) {
          std::memcpy(m_tensor.buffer().host_ptr(),
                      storage(),
                      byte_size());
        } else {
          // sanity checks
          assert(m_tensor.buffer().is_current_on(device));
          /**
           * TODO: TTG should provide a proper API for copying between host and device.
           */
          parsec_device_gpu_module_t *device_module = ttg_parsec::detail::parsec_ttg_caller->dev_ptr->device;
          int ret = device_module->memcpy_async(device_module, ttg_parsec::detail::parsec_ttg_caller->dev_ptr->stream,
                                                const_cast<value_type*>(m_tensor.buffer().device_ptr_on(device)),
                                                storage(),
                                                byte_size(),
                                                parsec_device_gpu_transfer_direction_h2d);
          if (ret != PARSEC_SUCCESS) throw std::runtime_error("Failed to copy sparsity data from host to device!");
#ifndef MRA_ENABLE_HOST
          m_pool->mark_submitted(*m_slot, ttg::device::current_stream());
#endif // !MRA_ENABLE_HOST
        }
      }


      /**
       * Mock tensor API
       */
      size_type dim(size_type d) const {
        return m_tensor.dim(d);
      }

#ifndef MRA_ENABLE_HOST
      value_type* storage() {
        return reinterpret_cast<value_type*>(m_slot->host_args.data());
      }
#else
      value_type* storage() {
        return m_buffer.host_ptr();
      }
#endif // !MRA_ENABLE_HOST

    private:
      /* number of bytes needed to store this tensor's sparsity bitfield */
      std::size_t byte_size() const {
        return sparsity_traits::required_space(m_tensor.dims()) * sizeof(value_type);
      }

      TensorType& m_tensor;
#ifndef MRA_ENABLE_HOST
      detail::BatchPool<detail::SparsityState>* m_pool = nullptr;
      typename detail::BatchPool<detail::SparsityState>::slot_t* m_slot = nullptr;
#else
      ttg::Buffer<value_type, DeviceAllocator<value_type>> m_buffer;
#endif // !MRA_ENABLE_HOST
    };

    template<std::size_t... Is>
    void populate_device_sparsity_impl(ttg::device::Device device, std::index_sequence<Is...>) {
      (std::get<Is>(*m_tensors).populate_device_sparsity(device), ...);
    }

  public:
    using mocktensor_tuple_type = std::tuple<MockTensor<TensorTypes>...>;
    using buffer_tuple_type = std::tuple<ttg::Buffer<typename TensorTypes::value_type, DeviceAllocator<typename TensorTypes::value_type>>...>;

  private:

    /**
     * Helper function to construct the buffers from each tensor.
     */
    template<typename TensorTuple, std::size_t... Is>
    buffer_tuple_type construct_buffers(TensorTuple&& tensors, std::index_sequence<Is...>) {
      return std::make_tuple(std::tuple_element_t<Is, buffer_tuple_type>(sparsity_traits<typename TensorTypes::sparsity_type>::required_space(std::get<Is>(tensors).dims()),
                                                                         ttg::scope::SyncIn)...);
    }

    template<typename TensorTuple, std::size_t... Is>
    auto construct_mocktensors(TensorTuple&& tensors, std::index_sequence<Is...>) {
      return std::make_unique<mocktensor_tuple_type>(std::make_tuple(MockTensor<TensorTypes>(std::get<Is>(tensors))...));
    }

  public:

    SparsityManager() = default;

    /**
     * Allow move but not copy, since the underlying buffers are not copyable.
     */
    SparsityManager(const SparsityManager&) = delete;
    SparsityManager(SparsityManager&&) = default;
    SparsityManager& operator=(const SparsityManager&) = delete;
    SparsityManager& operator=(SparsityManager&&) = default;

    SparsityManager(TensorTypes&... tensors)
    : m_tensors(construct_mocktensors(std::forward_as_tuple(tensors...), std::make_index_sequence<sizeof...(TensorTypes)>{}))
    { }


    /**
     * Overload for tuple of tensors.
     */
    SparsityManager(std::tuple<TensorTypes...>& tensors)
    : m_tensors(construct_mocktensors(tensors, std::make_index_sequence<sizeof...(TensorTypes)>{}))
    { }

    /**
     * Overload for tuple of tensors refs.
     */
    SparsityManager(const std::tuple<TensorTypes&...>& tensors)
    : m_tensors(construct_mocktensors(tensors, std::make_index_sequence<sizeof...(TensorTypes)>{}))
    { }


    void populate_device_sparsity(ttg::device::Device device = ttg::device::current_device()) {
      populate_device_sparsity_impl(device, std::make_index_sequence<sizeof...(TensorTypes)>{});
    }



    /**
     * TODO: do we need a way to get the sparsity info back out of the device?
     */

  private:
    std::unique_ptr<mocktensor_tuple_type> m_tensors;
  };


  namespace detail {
    template<typename T>
    struct sparseman_base;

    /**
     *
     */
    template<typename... TensorTypes>
    struct sparseman_base<std::tuple<TensorTypes...>> {
      using type = SparsityManager<std::decay_t<TensorTypes>...>;
    };

    /**
     * Array overload constructs a tuple.
     */
    template<typename T, std::size_t N>
    struct sparseman_base<std::array<T, N>> : public sparseman_base<decltype(std::tuple_cat(std::declval<std::array<T, N>>()))>
    { };

    template<typename T>
    using sparseman_base_type = typename sparseman_base<T>::type;

  } // namespace detail


  /**
   * Deduction guide for tuple of tensors overload.
   */
  template<typename... TensorTypes>
  SparsityManager(std::tuple<TensorTypes...>& tensors) -> SparsityManager<TensorTypes...>;


  namespace detail {

    auto extract_tensor_ref(concepts::Tensor auto& tensor) {
      return std::tie(tensor);
    }

    auto extract_tensor_ref(concepts::FunctionNode auto& node) {
      return std::tie(node.coeffs());
    }

    template<typename... TensorTypes>
    auto extract_tensor_ref(std::tuple<TensorTypes...>& tuple) {
      auto extract = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::tuple_cat(extract_tensor_ref(std::get<Is>(tuple))...);
      };
      return extract(std::make_index_sequence<sizeof...(TensorTypes)>{});
    }

    template<typename T, std::size_t N>
    auto extract_tensor_ref(std::array<T, N>& array) {
      auto extract = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::tuple_cat(extract_tensor_ref(std::get<Is>(array))...);
      };
      return extract(std::make_index_sequence<N>{});
    }
  } // namespace detail

  /**
   * Factory function to create a SparsityManager from a list of tensors and function nodes or arrays/tuples thereof.
   */
  template<typename... TensorTypes>
  auto make_sparsity_manager(TensorTypes&... tensors) {
    auto reftuple = std::tuple_cat(detail::extract_tensor_ref(tensors)...);
    using manager_type = detail::sparseman_base_type<decltype(reftuple)>;
    return manager_type{reftuple};
  }

} // namespace mra

#endif // MRA_SPARSITY_MANAGER_H