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
                                              sparsity_traits::required_space(m_tensor.dims()),
                                              parsec_device_gpu_transfer_direction_h2d);
        if (ret != PARSEC_SUCCESS) throw std::runtime_error("Failed to copy sparsity data from host to device!");
      }


      /**
       * Mock tensor API
       */
      const size_type dim(size_type d) const {
        return m_tensor.dim(d);
      }

      value_type* storage() {
        return m_buffer.host_ptr();
      }

    private:
      const TensorType& m_tensor;
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
    template<std::size_t... Is>
    buffer_tuple_type construct_buffers(TensorTypes&... tensors, std::index_sequence<Is...>) {
      return std::make_tuple(std::tuple_element_t<Is, buffer_tuple_type>(sparsity_traits<typename TensorTypes::sparsity_type>::required_space(tensors.dims()),
                                                                         ttg::scope::SyncIn)...);
    }

  public:

    SparsityManager(TensorTypes&... tensors)
    : m_tensors({tensors}...)
    , m_buffers(construct_buffers(tensors..., std::make_index_sequence<sizeof...(TensorTypes)>{}))
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

} // namespace mra

#endif // MRA_SPARSITY_MANAGER_H