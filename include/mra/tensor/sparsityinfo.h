#ifndef MRA_TENSOR_SPARSITYINFO_H
#define MRA_TENSOR_SPARSITYINFO_H

#include "mra/tensor/sparsity.h"

namespace mra {

  /**
   * Sparsity information collect sparsity data for a function node.
   * Sparsity info can be manipulated using the base type RangeSparsityBase
   * methods.
   */
  struct SparsityInfo : public RangeSparsityBase<SparsityInfo, void> {
    using base_type = RangeSparsityBase<SparsityInfo, void>;
    using value_type = void;

  private:

    void check_valid() {
      if (m_num_functions == 0) {
        throw std::logic_error("SparsityInfo must have positive number of functions");
      }
    }

  public:

    enum class InitType {
      AllZero,
      AllNonZero,
      Allocated
    };

    static constexpr Dimension ndim() {
      return 1;
    }

    /**
     * Creates an invalid SparsityInfo with zero functions. Must be assigned to a valid SparsityInfo before use.
      * This is needed for default construction in arrays (e.g., reconstruct()).
     */
    SparsityInfo() = default;

    /**
     * Constructs sparsity info for num_functions functions.
     * If all_zero is true, all functions are marked as zero.
     * Otherwise, all functions are marked as non-zero.
     */
    SparsityInfo(size_type num_functions, InitType init_type)
    : base_type()
    , m_num_functions(num_functions)
    {
      if (init_type == InitType::AllZero) {
        base_type::set_all_zero();
      } else {
        base_type::set_all_nonzero();
      }
    }

    SparsityInfo(const SparsityInfo& other) = delete;
    SparsityInfo(SparsityInfo&& other) = default;
    SparsityInfo& operator=(const SparsityInfo& other) = delete;
    SparsityInfo& operator=(SparsityInfo&& other) = default;
    ~SparsityInfo() = default;

    /**
     * Returns the number of functions. Used by the sparsity base type.
     */
    SCOPE size_type dim(Dimension d) const {
      assert(d == 0);
      return m_num_functions;
    }

    template<typename... Nodes>
    void nonzero_if_any(const Nodes&... nodes) {
      check_valid();
      for (size_type i = 0; i < m_num_functions; ++i) {
        bool any_nonzero = (nodes.sparsity().is_nonzero(i) || ...);
        std::array<bool, sizeof...(Nodes)> nonzero_array = { nodes.sparsity().is_nonzero(i)... };
        std::array<size_type, sizeof...(Nodes)> size_array = { static_cast<size_type>(nodes.sparsity().count())... };
        if (any_nonzero) {
          base_type::set_nonzero(i);
        } else {
          base_type::remove(i);
        }
      }
    }

    template<typename... Nodes>
    void nonzero_if_all(const Nodes&... nodes) {
      check_valid();
      for (size_type i = 0; i < m_num_functions; ++i) {
        if ((nodes.sparsity().is_nonzero(i) && ...)) {
          base_type::set_nonzero(i);
        } else {
          base_type::remove(i);
        }
      }
    }

  private:
    size_type m_num_functions = 0;
  };

  inline std::ostream& operator<<(std::ostream& os, const SparsityInfo& si) {
    os << "SparsityInfo(" << static_cast<const RangeSparsityBase<SparsityInfo, void>&>(si) << ")";
    return os;
  }

  /**
   * Number of non-zero function slots in `tensor`'s own sparsity (dimension 0
   * must be N, the node's total function count). Thin wrapper so callers can
   * get a node's non-zero count directly from a coeffs tensor without going
   * through sparsity() themselves.
   */
  template<typename TensorT>
  size_type count_nonzero_any(size_type N, const TensorT& tensor) {
    assert(static_cast<size_type>(tensor.dim(0)) == N);
    return tensor.sparsity().count_nonzero();
  }

} // namespace mra

#endif // MRA_TENSOR_SPARSITYINFO_H
