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

    static constexpr Dimension ndim() {
      return 1;
    }

    /**
     * Constructs sparsity info for num_functions functions.
     * If all_zero is true, all functions are marked as zero.
     * Otherwise, all functions are marked as non-zero.
     */
    SparsityInfo(size_type num_functions, bool all_zero = true)
    : base_type()
    , m_num_functions(num_functions)
    {
      if (all_zero) {
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
    size_type dim(Dimension d) const {
      assert(d == 0);
      return m_num_functions;
    }

    template<typename... Nodes>
    void nonzero_if_any(const Nodes&... nodes) {
      for (size_type i = 0; i < m_num_functions; ++i) {
        if ((nodes.sparsity().is_nonzero(i) || ...)) {
          base_type::set_nonzero(i);
        } else {
          base_type::set_zero(i);
        }
      }
    }

    template<typename... Nodes>
    void nonzero_if_all(const Nodes&... nodes) {
      for (size_type i = 0; i < m_num_functions; ++i) {
        if ((nodes.sparsity().is_nonzero(i) && ...)) {
          base_type::set_nonzero(i);
        } else {
          base_type::set_zero(i);
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

} // namespace mra