#ifndef HAVE_MRA_FUNCTIONSET_H
#define HAVE_MRA_FUNCTIONSET_H

#include "mra/misc/batching.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/misc/key.h"
#include "mra/misc/hash.h"
#include "mra/misc/functiondata.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/tensor.h"


namespace mra {

  /**
   * A set of functions that can be evaluated in parallel.
   * Given a key, the FunctionSet returns a view of the functions
   * that belong to the batch encoded in the key.
   *
   */
  template <typename FunctionType, typename BatchManager>
  class FunctionSet {

  public:

    using tensor_type = Tensor<FunctionType, 1>;
    using view_type = typename tensor_type::view_type;

    using function_type = FunctionType;

  private:
    tensor_type                    m_tensor;     // Number of batches for this function set
    std::shared_ptr<BatchManager> m_batchman;   // Batch Manager for this function set

    static size_type accumulate_local_batch_sizes(const std::shared_ptr<BatchManager>& batchman) {
      // Default implementation assumes each batch has the same size
      size_type size = 0;
      batchman->foreach_local_batch([&](Batch batch){
        size += batchman->batch_size(batch);
        return true; // continue iterating
      });
      return size;
    }

    size_type offset_for_batch(Batch batch) const {
      // Default implementation returns the offset for the given key
      size_type offset = 0;
      m_batchman->foreach_local_batch([&](Batch b) {
        if (b == batch) {
          return false; // stop iterating
        }
        offset += m_batchman->batch_size(b);
        return true; // continue iterating
      });
      return offset;
    }

  public:

    FunctionSet(std::shared_ptr<BatchManager> batchman)
    : m_tensor(accumulate_local_batch_sizes(batchman))
    , m_batchman(std::move(batchman))
    { }

    /// Returns the number of batches for this function set
    size_type num_functions() const {
      return m_batchman->num_functions();
    }

    size_type num_batches() const {
      return m_batchman->num_batches();
    }

    size_type num_local_batches() const {
      return m_batchman->num_local_batches();
    }

    size_type num_local_functions() const {
      return accumulate_local_batch_sizes(m_batchman);
    }

    /**
     * Returns the number of functions for the given key.
     */
    template<Dimension NDIM>
    size_type num_functions(const Key<NDIM>& key) const {
      // Default implementation returns 1 function per key
      return m_batchman->batch_size(key.batch());
    }

    /**
     * Returns the view for all functions in the set.
     */
    const view_type current_view() const {
      return m_tensor.current_view();
    }

    /**
     * Returns the view for the given key.
     */
    template<Dimension NDIM>
    const view_type current_view(const Key<NDIM>& key) const {
      // Default implementation returns the current view for the key
      size_type offset = offset_for_batch(key.batch());

      //std::cout << "Current view for key " << key << " at offset " << offset << std::endl;

      // view at proper offset
      return view_type(m_tensor.current_view().data() + offset, {num_functions(key)});
    }

    template<Dimension NDIM>
    const view_type host_view(const Key<NDIM>& key) const {
      // Default implementation returns the host view for the key
      size_type offset = offset_for_batch(key.batch());

      // view at proper offset
      return view_type(m_tensor.buffer().host_ptr() + offset, {num_functions(key)});
    }

    /**
     * Returns the batch manager of this function set.
     */
    const std::shared_ptr<BatchManager>& batchman() const {
      return m_batchman;
    }

    /**
     * Returns the buffer containing the functions.
     */
    const auto& buffer() const {
      return m_tensor.buffer();
    }
  };

  template<typename FunctionType, typename BatchManager>
  std::shared_ptr<FunctionSet<FunctionType, BatchManager>>
  make_functionset(const std::shared_ptr<BatchManager>& batchman) {
    return std::make_shared<FunctionSet<FunctionType, BatchManager>>(batchman);
  }

  template<typename FunctionType>
  std::shared_ptr<FunctionSet<FunctionType, SimpleBatchManager>>
  make_functionset(size_type num_functions) {
    return std::make_shared<FunctionSet<FunctionType, SimpleBatchManager>>(SimpleBatchManager(num_functions));
  }

} // namespace mra

#endif // HAVE_MRA_FUNCTIONSET_H