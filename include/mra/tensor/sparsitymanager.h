#ifndef MRA_SPARSITY_MANAGER_H
#define MRA_SPARSITY_MANAGER_H


#include "mra/tensor/sparsity.h"

#include <ttg/parsec/devicefunc.h>

namespace mra {

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

      MockTensor(TensorType& tensor)
      : sparsity_type()
      , m_tensor(tensor)
      , m_buffer(sparsity_traits::required_space(tensor.dims()), ttg::scope::SyncIn)
      {
        this->apply_sparsity(m_tensor.sparsity());
      }

      void populate_device_sparsity() {
#ifdef MRA_ENABLE_HOST
        //for (size_t i = 0; i < sparsity_traits::required_space(m_tensor.dims()); ++i) {
        //  std::cout << "SPARSITY INFO [" << i << "]: " << *reinterpret_cast<uint64_t*>(&m_buffer.host_ptr()[i]) << std::endl;
        //  m_tensor.buffer().host_ptr()[i] = m_buffer.host_ptr()[i];
        //}
        std::memcpy(m_tensor.buffer().host_ptr(),
                    m_buffer.host_ptr(),
                    sparsity_traits::required_space(m_tensor.dims()) * sizeof(typename sparsity_traits::value_type));
#else  // MRA_ENABLE_HOST
        // sanity checks
        assert(ttg::device::current_device().is_gpu());
        assert(m_tensor.buffer().is_current_on(ttg::device::current_device()));
        /**
         * TODO: TTG should provide a proper API for copying between host and device.
         */
        parsec_device_gpu_module_t *device_module = ttg_parsec::detail::parsec_ttg_caller->dev_ptr->device;
        int ret = device_module->memcpy_async(device_module, ttg_parsec::detail::parsec_ttg_caller->dev_ptr->stream,
                                              m_buffer.host_ptr(),
                                              const_cast<value_type*>(m_tensor.buffer().current_device_ptr()),
                                              sparsity_traits::required_space(m_tensor.dims()) * sizeof(typename sparsity_traits::value_type),
                                              parsec_device_gpu_transfer_direction_h2d);
        if (ret != PARSEC_SUCCESS) throw std::runtime_error("Failed to copy sparsity data from host to device!");
#endif // MRA_ENABLE_HOST
      }


      /**
       * Mock tensor API
       */
      size_type dim(size_type d) const {
        return m_tensor.dim(d);
      }

      value_type* storage() {
        return m_buffer.host_ptr();
      }

    private:
      TensorType& m_tensor;
      ttg::Buffer<value_type, DeviceAllocator<value_type>> m_buffer;
    };

    template<std::size_t... Is>
    void populate_device_sparsity_impl(std::index_sequence<Is...>) {
      (std::get<Is>(m_tensors).populate_device_sparsity(), ...);
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
    mocktensor_tuple_type construct_mocktensors(TensorTuple&& tensors, std::index_sequence<Is...>) {
      return std::make_tuple(MockTensor<TensorTypes>(std::get<Is>(tensors))...);
    }

  public:

    SparsityManager(TensorTypes&... tensors)
    : m_tensors(construct_mocktensors(std::forward_as_tuple(tensors...), std::make_index_sequence<sizeof...(TensorTypes)>{}))
    , m_buffers(construct_buffers(std::forward_as_tuple(tensors...), std::make_index_sequence<sizeof...(TensorTypes)>{}))
    { }


    /**
     * Overload for tuple of tensors.
     */
    SparsityManager(std::tuple<TensorTypes...>& tensors)
    : m_tensors(construct_mocktensors(tensors, std::make_index_sequence<sizeof...(TensorTypes)>{}))
    , m_buffers(construct_buffers(tensors, std::make_index_sequence<sizeof...(TensorTypes)>{}))
    { }

    /**
     * Overload for tuple of tensors refs.
     */
    SparsityManager(const std::tuple<TensorTypes&...>& tensors)
    : m_tensors(construct_mocktensors(tensors, std::make_index_sequence<sizeof...(TensorTypes)>{}))
    , m_buffers(construct_buffers(tensors, std::make_index_sequence<sizeof...(TensorTypes)>{}))
    { }


    void populate_device_sparsity() {
      populate_device_sparsity_impl(std::make_index_sequence<sizeof...(TensorTypes)>{});
    }



    /**
     * TODO: do we need a way to get the sparsity info back out of the device?
     */

  private:
    mocktensor_tuple_type m_tensors;
    buffer_tuple_type m_buffers;
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