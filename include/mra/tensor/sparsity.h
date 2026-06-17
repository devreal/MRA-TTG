#ifndef MRA_TENSOR_SPARSITY_H
#define MRA_TENSOR_SPARSITY_H

#include "mra/misc/types.h"

namespace mra {


  namespace detail {

    template<typename T>
    SCOPE std::size_t align_to_type(std::size_t size) {
      std::size_t mask = alignof(T) - 1;
      return size + (-size & mask);
    }



    enum class SparsityState : std::uint8_t { NONZERO = 1, ALLOCATED = 2, NONZERO_ALLOCATED = 3, SPARSE = 0 };

    SCOPE bool operator&(SparsityState a, SparsityState b) {
      return (static_cast<std::uint8_t>(a) & static_cast<std::uint8_t>(b)) != 0;
    }

    SCOPE bool operator|(SparsityState a, SparsityState b) {
      return (static_cast<std::uint8_t>(a) | static_cast<std::uint8_t>(b)) != 0;
    }

    SCOPE SparsityState& operator|=(SparsityState& a, SparsityState b) {
      a = static_cast<SparsityState>(static_cast<std::uint8_t>(a) | static_cast<std::uint8_t>(b));
      return a;
    }

    SCOPE SparsityState& operator&=(SparsityState& a, SparsityState b) {
      a = static_cast<SparsityState>(static_cast<std::uint8_t>(a) & static_cast<std::uint8_t>(b));
      return a;
    }

    /**
     * TODO: is that actually a good idea?
     */
    SCOPE SparsityState operator~(SparsityState a) {
      return static_cast<SparsityState>(~static_cast<std::uint8_t>(a));
    }

  } // namespace detail

  /**
   * Traits for sparsity information of a tensor.
   */
  template<typename T>
  struct sparsity_traits {};

  /**
   * View for sparsity information of a tensor where the first dimension is sparse.
   * A TensorView can derive from this class to manage sparsity information contained
   * in the memory of the Tensor. This is useful for on-device representations of sparse tensors
   * and for transferring sparsity information between hosts.
   *
   * The derived class must implement:
   *  std::size_t dim(Dimension d) const; // returns the size of dimension d
   *  std::size_t ndim() const; // returns the number of dimensions
   *  std::array<std::size_t, N> dims() const; // returns sizes of all dimensions
   *  ValueType* storage(); // returns pointer to the beginning of the allocated storage
   *  const ValueType* storage() const; // returns pointer to the beginning of the allocated storage
   */
  template<typename Derived, typename ValueType>
  struct SparseArrayBase {

    using value_type = ValueType;
    using sparsity_type = SparseArrayBase<Derived, value_type>;

  private:

    using unit_type = detail::SparsityState;


    SCOPE void* storage() {
      return static_cast<Derived*>(this)->storage();
    }

    SCOPE const void* storage() const {
      return static_cast<const Derived*>(this)->storage();
    }

    SCOPE unit_type* sparsity_data() {
      return static_cast<unit_type*>(storage());
    }

    SCOPE const unit_type* sparsity_data() const {
      return static_cast<const unit_type*>(storage());
    }

    /**
     * Returns the number of value_type entries needed to store the sparsity data.
     */
    SCOPE std::size_t value_count() const {
      return detail::align_to_type<value_type>(count()) / sizeof(value_type);
    }


  public:


    class sparsity_iterator {
      const unit_type* m_start = nullptr;
      const unit_type* m_pos   = nullptr;
      detail::SparsityState m_type;

    public:
      sparsity_iterator(const unit_type* data, detail::SparsityState type)
      : m_start(data)
      , m_pos(data)
      , m_type(type)
      { }

      sparsity_iterator& operator++() {
        while (!(*(++m_pos) & m_type))
        { }
        return *this;
      }

      size_type operator*() const {
        return (m_pos - m_start); // return the index
      }

      bool operator!=(const sparsity_iterator& other) const {
        return m_pos != other.m_pos;
      }
    }; // class sparsity_iterator


    SparseArrayBase() = default;

    /**
     * Returns the size of the sparse dimension (dimension 0).
     */
    SCOPE std::size_t count() const {
      return static_cast<const Derived*>(this)->dim(0);
    }

    /**
     * Returns the offset to the data portion of the tensor.
     */
    SCOPE std::size_t data_offset() const {
      return detail::align_to_type<value_type>(count());
    }

    /**
     * Returns pointer to the data portion of the tensor.
     */
    SCOPE value_type* data() {
      return static_cast<value_type*>(storage()) + value_count();
    }

    /**
     * Returns pointer to the data portion of the tensor.
     */
    SCOPE const value_type* data() const {
      return static_cast<const value_type*>(storage()) + value_count();
    }

    /**
     * Returns true if the i'th element in the sparse dimension is non-zero.
     */
    SCOPE bool is_nonzero(std::size_t i) const {
      const unit_type byte = sparsity_data()[i];
      return byte & static_cast<unit_type>(detail::SparsityState::NONZERO);
    }

    SCOPE bool is_zero(std::size_t i) const {
      auto sd = sparsity_data();
      if (nullptr == sd) {
        return true; // if no sparsity data, treat as all zero
      }
      const unit_type byte = sd[i];
      return (byte & static_cast<unit_type>(detail::SparsityState::NONZERO)) == 0;
    }

    SCOPE bool is_any_nonzero() const {
      const std::size_t n = count();
      for (std::size_t i = 0; i < n; ++i) {
        if (is_nonzero(i)) {
          return true;
        }
      }
      return false;
    }

    SCOPE bool is_all_zero() const {
      const std::size_t n = count();
      for (std::size_t i = 0; i < n; ++i) {
        if (!is_zero(i)) {
          return false;
        }
      }
      return true;
    }

    /**
     * Sets the i'th entry in the sparse dimension to non-zero and allocated.
     */
    SCOPE void set_nonzero(std::size_t i) {
      unit_type& byte = sparsity_data()[i];
      byte = static_cast<unit_type>(detail::SparsityState::NONZERO_ALLOCATED);
    }

    /**
     * Sets the i'th entry in the sparse dimension to zero.
     */
    SCOPE void set_zero(std::size_t i) {
      unit_type& byte = sparsity_data()[i];
      byte &= static_cast<unit_type>(detail::SparsityState::ALLOCATED); // keep allocated bit but clear non-zero bit
    }

    SCOPE void set_nonzero_all() {
      const std::size_t n = count();
      for (std::size_t i = 0; i < n; ++i) {
        set_nonzero(i);
      }
    }

    SCOPE void set_zero_all() {
      const std::size_t n = count();
      for (std::size_t i = 0; i < n; ++i) {
        set_zero(i);
      }
    }

    /**
     * Mark the given id as unallocated and zero.
     */
    void remove(size_type id) {
      sparsity_data()[id] = detail::SparsityState::SPARSE;
    }

    void reset() {
      std::fill(static_cast<value_type*>(storage()), static_cast<value_type*>(storage()) + value_count(), value_type{});
    }

    SCOPE std::size_t count_nonzero() const {
      const std::size_t n = count();
      std::size_t count = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (is_nonzero(i)) {
          ++count;
        }
      }
      return count;
    }

    /**
     * Returns true if the i'th entry in the sparse dimension is allocated (non-zero or zero).
     */
    SCOPE bool is_allocated(std::size_t i) const {
      const unit_type byte = sparsity_data()[i];
      return byte & static_cast<unit_type>(detail::SparsityState::ALLOCATED);
    }

    SCOPE void set_allocated(std::size_t i) {
      unit_type& byte = sparsity_data()[i];
      byte = static_cast<unit_type>(detail::SparsityState::ALLOCATED);
    }

    SCOPE void set_deallocated(std::size_t i) {
      unit_type& byte = sparsity_data()[i];
      byte &= ~static_cast<unit_type>(detail::SparsityState::ALLOCATED);
    }

    SCOPE void set_deallocated_all() {
      const std::size_t n = static_cast<const Derived*>(this)->dim(0);
      for (std::size_t i = 0; i < n; ++i) {
        set_deallocated(i);
      }
    }

    SCOPE void set_allocated_all() {
      const std::size_t n = static_cast<const Derived*>(this)->dim(0);
      for (std::size_t i = 0; i < n; ++i) {
        set_allocated(i);
      }
    }

    SCOPE std::size_t count_allocated() const {
      const std::size_t n = static_cast<const Derived*>(this)->dim(0);
      std::size_t count = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (is_allocated(i)) {
          ++count;
        }
      }
      return count;
    }

    /**
     * Returns the offset in the data portion corresponding to the i'th element.
     */
    SCOPE std::size_t offset_of(std::size_t i) const {
      std::size_t offset = 0;
      /* count the size of each non-zero tensor in the 1:NDIM-1 dimensions */
      std::size_t data_size = static_cast<const Derived*>(this)->dim(1);
      auto dims = static_cast<const Derived*>(this)->dims();
      for (std::size_t i = 2; i < static_cast<const Derived*>(this)->ndim(); ++i) {
        data_size *= dims[i];
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (is_allocated(j)) {
          offset += data_size;
        }
      }
      return offset;
    }


    void apply_sparsity(const SparseArrayBase& s) {
      assert(count() == s.count());
      std::memcpy(storage(), s.storage(), s.count());
    }

    template<typename SparsityT>
    void apply_sparsity(const SparsityT& s) {
      assert(count() == s.count());
      /* zero out first */
      reset();
      for (size_type i = 0; i < count(); ++i) {
        if (s.is_nonzero(i)) {
          set_nonzero(i);
        } else if (s.is_allocated(i)) {
          set_allocated(i);
        } else {
          remove(i);
        }
      }
    }

    /* form the union with the given SparseArrayBase */
    void union_sparsity(const SparseArrayBase& s) {
      assert(count() == s.count());
      for (size_type i = 0; i < count(); ++i) {
        if (s.is_nonzero(i)) {
          set_nonzero(i);
        }
      }
    }

    /* form the union with the given sparsity */
    template<typename SparsityT>
    void union_sparsity(const SparsityT& s) {
      assert(count() == s.count());
      for (auto iter = s.begin_nonzero(); iter != s.end_nonzero(); ++iter) {
        set_nonzero(*iter);
      }
    }

    /* form the intersection with the given sparsity array */
    void intersect_sparsity(const SparseArrayBase& s) {
      assert(count() == s.count());
      for (size_type i = 0; i < count(); ++i) {
        if (!s.is_nonzero(i)) {
          remove(i);
        }
      }
    }

    /* form the intersection with the given sparsity */
    template<typename SparsityT>
    void intersect_sparsity(const SparsityT& s) {
      assert(count() == s.count());
      size_type i = 0;
      for (auto iter = s.begin_nonzero(); iter != s.end_nonzero(); ++iter, ++i) {
        while (i < *iter) {
          remove(i++);
        }
      }
      while (i < count()) {
        remove(i++);
      }
    }

    using iterator = sparsity_iterator;

    iterator begin_nonzero() {
      return iterator(sparsity_data(), detail::SparsityState::NONZERO);
    }

    iterator end_nonzero() {
      return iterator(sparsity_data() + count(), detail::SparsityState::NONZERO);
    }

    iterator begin_allocated() {
      return iterator(sparsity_data(), detail::SparsityState::ALLOCATED);
    }

    iterator end_allocated() {
      return iterator(sparsity_data() + count(), detail::SparsityState::ALLOCATED);
    }
  };


  template<typename Derived, typename ValueType>
  struct sparsity_traits<SparseArrayBase<Derived, ValueType>> {
    using derived_type = Derived;
    using value_type = ValueType;
    template<typename T, typename U>
    using sparsity_type = SparseArrayBase<T, U>;
    static constexpr bool is_sparse() { return true; }
    /**
     * Whether the sparsity information is stored inline with the data.
     * If true, query the space_required() function to get the additional space needed.
     */
    static constexpr bool inline_storage() { return true; }
    /**
     * Whether the sparsity information allocates additional storage.
     */
    static constexpr bool allocates_storage() { return false; }
    /**
     * How much additional space (in units of \ref value_type) is required to store the sparsity information.
     */
    template<std::size_t NDIM>
    static size_type required_space(const std::array<size_type, NDIM>& dims) {
      // worst case: every entry is its own range
      return detail::align_to_type<value_type>(dims[0]) / sizeof(value_type);
    }

    static constexpr std::string name() {
      return "SparseArrayBase";
    }
  };


  namespace detail {

    struct Range {
      ssize_type from = -1; // inclusive
      ssize_type to   = -1; // inclusive

      Range() = default;

      Range(ssize_type i)
      : from(i)
      , to(i)
      { }

      Range(ssize_type from, ssize_type to)
      : from(from)
      , to(to)
      { }

      void append(ssize_type i) {
        if (from == -1) {
          from = i;
          to   = i;
        } else if (i == to + 1) {
          to = i;
        } else if (i == from - 1) {
          from = i;
        } else {
          throw std::invalid_argument("Cannot append non-contiguous index to range");
        }
      }

      bool is_contiguous(ssize_type i) const {
        return is_empty() || from-1 == i || to+1 == i;
      }

      bool is_empty() const {
        return from == -1 && to == -1;
      }

      bool contains(ssize_type i) const {
        return from <= i && i <= to;
      }

      template <typename Archive>
      void serialize(Archive &ar) {
        ar & from & to;
      }

      template <typename Archive>
      void serialize(Archive &ar, const unsigned int) {
        serialize(ar);
      }

    }; // class Range



    inline std::ostream& operator<<(std::ostream& os, const Range& r) {
      os << "[" << r.from << ", " << r.to << "]";
      return os;
    }


  } // namespace detail


  /**
   * Encoding of sparsity information using ranges.
   *
   * This encoding is more efficient than SparseArrayBase but is less flexible
   * with a higher cost for updates. Also does not support concurrent updates on the device.
   *
   * \tparam ValueT The type of the values used by the owning container.
   *                Sparsity will ensure proper alignment for this type but will
   *                repurpose the memory to the type needed to encode sparsity.
   */
  template<typename Derived, typename ValueT>
  struct RangeSparsityBase {
    using value_type = std::decay_t<ValueT>;
    using sparsity_type = RangeSparsityBase<Derived, value_type>;

  private:
    class sparsity_iterator {
      using iter_type = typename std::vector<detail::Range>::iterator;
      iter_type m_iter, m_end;
      size_type m_id = 0;

    public:
      sparsity_iterator(const iter_type& ranges_iter, const iter_type& ranges_end)
      : m_iter(ranges_iter)
      , m_end(ranges_end)
      {
        if (m_iter != m_end) {
          m_id = m_iter->from;
        }
      }

      sparsity_iterator& operator++() {
        if (m_iter->to == m_id) {
          ++m_iter;
          if (m_iter != m_end) {
            m_id = m_iter->from;
          }
        } else {
          ++m_id;
        }
        return *this;
      }

      bool operator!=(const sparsity_iterator& other) const {
        return m_iter != other.m_iter || m_id != other.m_id;
      }

      size_type operator*() const {
        return m_id;
      }
    }; // class sparsity_iterator

    std::vector<detail::Range> m_non_zero_ranges;   // ranges of non-zero entries
    std::vector<detail::Range> m_allocated_ranges;  // ranges of allocated entries

    void add(size_type id, std::vector<detail::Range>& ranges) {
      for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        if (it->contains(id)) {
          return;
        }
        if (it->is_contiguous(id)) {
          // extend existing range
          it->append(id);
          // check for possible merge with previous range first
          if (it != ranges.begin()) {
            auto prev = std::prev(it);
            if (prev->is_contiguous(it->from)) {
              // merge with previous range; it is now invalid, use prev
              prev->to = it->to;
              it = ranges.erase(it);  // it now points to element after the erased one
              it = prev;              // step back so the next-merge check uses the merged range
            }
          }
          // check for possible merge with next range
          auto next = std::next(it);
          if (next != ranges.end() && it->is_contiguous(next->from)) {
            // merge with next range
            it->to = next->to;
            ranges.erase(next);
          }
          return;
        }
        if (id < it->from) {
          // insert before the current range
          auto next = ++it;
          auto prev = --it;
          if (next != ranges.end() && next->from == id+1) {
            // append to front
            next->from = id;
          } else if (it != ranges.begin() && prev->to == id-1) {
            // append to back
            prev->to = id;
          } else {
            // insert new range
            ranges.insert(it, detail::Range(id));
          }
          return;
        }
      }
      // add to the end
      ranges.push_back(detail::Range(id));
    }

    void remove(size_type id, std::vector<detail::Range>& ranges) {
      for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        if (it->contains(id)) {
          if (it->from == it->to) {
            /* remove entire range */
            ranges.erase(it);
          } else if (it->from == id) {
            /* remove from beginning of range */
            it->from++;
          } else if (it->to == id) {
            /* remove from end of range */
            it->to--;
          } else {
            /* split the range: save original end before modifying it */
            auto original_to = it->to;
            auto next = it + 1;
            it->to = id - 1;
            ranges.insert(next, detail::Range(id + 1, original_to));
          }
          return;
        }
      }
    }

    bool contains(size_type id, const std::vector<detail::Range>& ranges) const {
      assert(count() > 0 || ranges.empty());
      for (const auto& r : ranges) {
        if (r.contains(id)) {
          return true;
        }
      }
      return false;
    }

    // allow other base classes access
    template<typename, typename>
    friend struct RangeSparsityBase;

  public:

    constexpr RangeSparsityBase() = default;

    /* pointer is ignored, the sparsity information manages its own memory */
    template<typename T>
    RangeSparsityBase(T* ptr)
    { }

    /* copy construction allocates memory */
    RangeSparsityBase(const RangeSparsityBase&) = delete;

    /* move construction */
    RangeSparsityBase(RangeSparsityBase&& other) = default;

    RangeSparsityBase& operator=(RangeSparsityBase&& other) = default;

    RangeSparsityBase& operator=(const RangeSparsityBase& other) = delete;

    ~RangeSparsityBase() = default;


    /**
     * Returns the size of the sparse dimension (dimension 0).
     */
    SCOPE std::size_t count() const {
      return static_cast<const Derived*>(this)->dim(0);
    }

    /* returns true if value is not zero */
    bool is_nonzero(size_type id) const {
      return contains(id, m_non_zero_ranges);
    }

    bool is_zero(size_type id) const {
      return !contains(id, m_non_zero_ranges);
    }

    SCOPE bool is_any_nonzero() const {
      return !m_non_zero_ranges.empty();
    }

    SCOPE bool is_all_nonzero() const {
      return m_non_zero_ranges.size() == 1 && m_non_zero_ranges[0].from == 0 && m_non_zero_ranges[0].to == count() - 1;
    }

    /**
     * Returns true if the given id is allocated.
     */
    bool is_allocated(size_type id) const {
      return contains(id, m_allocated_ranges);
    }

    /**
     * The number of non-zero entries.
     */
    size_type count_nonzero() const {
      size_type res = 0;
      for (const auto& r : m_non_zero_ranges) {
        res += r.to - r.from + 1;
      }
      return res;
    }

    /**
     * The offset of a given id, i.e., the sum of
     * all non-zero entries before the given id.
     */
    size_type offset(size_type id) const {
      size_type offset = 0;
      for (const auto& r : m_allocated_ranges) {
        if (r.to < id) {
          offset += r.to - r.from + 1;
        } else {
          return offset + id - r.from;
        }
      }
      return offset;
    }

    /**
     * Mark the given id as allocated and non-zero.
     */
    void set_nonzero(size_type id) {
      add(id, m_non_zero_ranges);
      // if it's nonzero it's also allocated
      add(id, m_allocated_ranges);
    }

    /**
     * Mark the given id as allocated only, if it was allocated before.
     * Otherwise the id is marked as unallocated and zero.
     */
    void set_allocated(size_type id) {
      add(id, m_allocated_ranges);
    }

    /* Mark all as zero */
    void set_all_zero() {
      m_non_zero_ranges.clear();
      m_allocated_ranges.clear();
    }

    /* Mark all as zero */
    void reset() {
      set_all_zero();
    }

    /**
     * Mark all ids as allocated and non-zeros
     */
    void set_all_nonzero() {
      m_non_zero_ranges.clear();
      m_non_zero_ranges.emplace_back(0, count() - 1);
      m_allocated_ranges.clear();
      m_allocated_ranges.emplace_back(0, count() - 1);
      assert(m_allocated_ranges.size() == 1);
    }

    /**
     * Mark all ids as allocated only.
     */
    void set_all_allocated() {
      m_allocated_ranges.clear();
      m_allocated_ranges.emplace_back(0, count() - 1);
      assert(m_allocated_ranges.size() == 1);
    }

    /**
     * Mark the given id as and zero. It will still be marked as allocated.
     */
    void set_zero(size_type id) {
      remove(id, m_non_zero_ranges);
    }

    /**
     * Remove the id from the sparsity information.
     * Marks it both zero and not allocated.
     */
    void remove(size_type id) {
      remove(id, m_non_zero_ranges);
      remove(id, m_allocated_ranges);
    }

    /* apply sparsity information from input
     * the count must be the same on both sparsity objects
     * and both sparsity objects must point to the same memory space */
    template<typename Derived_, typename Value_>
    void apply_sparsity(const SparseArrayBase<Derived_, Value_>& s) {
      m_non_zero_ranges.clear();
      m_allocated_ranges.clear();
      detail::Range ra  = {-1, -1}; // range for allocated entries
      detail::Range rnz = {-1, -1}; // range for nonzero entries
      auto add_to_range = [&](size_type i, detail::Range& r, std::vector<detail::Range>& ranges) {
        if (r.is_contiguous(i)) {
          r.append(i);
        } else {
          if (!r.is_empty()) {
            ranges.push_back(r);
          }
          r = detail::Range(i);
        }
      };
      /* iterate over all entries and form the ranges for non-zero and allocated elements */
      for (size_type i = 0; i < count(); ++i) {
        if (s.is_nonzero(i)) {
          add_to_range(i, rnz, m_non_zero_ranges);
          add_to_range(i, ra,  m_allocated_ranges);
        } else if (s.is_allocated(i)) {
          add_to_range(i, ra,  m_allocated_ranges);
        }
      }
    }

    /* apply sparsity information from input
     * the count must be the same on both sparsity objects
     * and both sparsity objects must point to the same memory space */
    template<typename Derived_, typename Value_>
    void apply_sparsity(const RangeSparsityBase<Derived_, Value_>& s) {
      m_non_zero_ranges = s.m_non_zero_ranges;
      m_allocated_ranges = s.m_allocated_ranges;
    }


    template <typename Archive>
    void serialize(Archive &ar) {
      ar & m_non_zero_ranges & m_allocated_ranges;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }

    using iterator = sparsity_iterator;

    iterator begin_nonzero() const {
      return iterator(m_non_zero_ranges.begin(), m_non_zero_ranges.end());
    }

    iterator end_nonzero() const {
      return iterator(m_non_zero_ranges.end(), m_non_zero_ranges.end());
    }

    iterator begin_allocated() const {
      return iterator(m_allocated_ranges.begin(), m_allocated_ranges.end());
    }

    iterator end_allocated() const {
      return iterator(m_allocated_ranges.end(), m_allocated_ranges.end());
    }
  };


  template<typename Derived, typename ValueType>
  struct sparsity_traits<RangeSparsityBase<Derived, ValueType>> {
    using derived_type = Derived;
    using value_type = ValueType;
    template<typename T, typename U>
    using sparsity_type = RangeSparsityBase<T, U>;
    static constexpr bool is_sparse() { return true; }
    /**
     * Whether the sparsity information is stored inline with the data.
     * If true, query the space_required() function to get the additional space needed.
     */
    static constexpr bool inline_storage() { return false; }
    /**
     * Whether the sparsity information allocates additional storage.
     */
    static constexpr bool allocates_storage() { return true; }

    /**
     * How much additional space (in units of size_type) is required to store the sparsity information.
     */
    template<std::size_t NDIM>
    static size_type required_space(const std::array<size_type, NDIM>&) {
      // worst case: every entry is its own range
      return 0;
    }

    static constexpr std::string name() {
      return "RangeSparsityBase";
    }
  };



  /**
   * Sparsity information for a dense tensor.
   */

  /**
   * View for sparsity information of a dense tensor.
   */
  template<typename Derived, typename ValueType>
  struct DenseViewBase {

    using derived_type = Derived;
    using value_type = ValueType;
    using sparsity_type = DenseViewBase<Derived, ValueType>;


  private:

    size_type dim0() const {
      return static_cast<const Derived*>(this)->dim(0);
    }

  public:


    constexpr DenseViewBase() = default;

    /**
     * Returns the offset to the data portion of the tensor.
     */
    SCOPE constexpr std::size_t data_offset() const {
      return 0;
    }


    /**
     * Returns true if the i'th element in the sparse dimension is non-zero.
     */
    SCOPE bool is_nonzero(std::size_t i) const {
      return dim0() > 0;
    }

    /**
     * Returns true if the i'th element in the sparse dimension is non-zero.
     */
    SCOPE bool is_zero(std::size_t i) const {
      return dim0() == 0;
    }

    SCOPE bool is_any_nonzero() const {
      return dim0() > 0;
    }

    SCOPE bool is_all_nonzero() const {
      return dim0() > 0;
    }

    SCOPE std::size_t count_nonzero() const {
      return dim0();
    }

    /**
     * Returns the size of the sparse dimension (dimension 0).
     */
    SCOPE std::size_t count() const {
      return dim0();
    }


    /**
     * Returns true if the i'th entry in the sparse dimension is allocated (non-zero or zero).
     */
    SCOPE bool is_allocated(std::size_t i) const {
      return dim0() > 0;
    }

    SCOPE size_type count_allocated() const {
      return dim0();
    }

    /**
     * Returns the offset in the data portion corresponding to the i'th element.
     */
    SCOPE size_type offset_of(size_type i) const {
      if (dim0() == 0) {
        return 0;
      }
      /* count the size of each non-zero tensor in the 1:NDIM-1 dimensions */
      size_type data_size = 1;
      if (Derived::ndim() > 1) {
        auto dims = static_cast<const Derived*>(this)->dims();
        data_size = dims[1];
        for (size_type i = 2; i < static_cast<const Derived*>(this)->ndim(); ++i) {
          data_size *= dims[i];
        }
      }
      return i*data_size;
    }

    SCOPE value_type* data() {
      return static_cast<Derived*>(this)->storage();
    }

    SCOPE const value_type* data() const {
      return static_cast<const Derived*>(this)->storage();
    }
  };


  template<typename Derived, typename ValueType>
  struct sparsity_traits<DenseViewBase<Derived, ValueType>> {
    using derived_type = Derived;
    using value_type = ValueType;
    template<typename T, typename U>
    using sparsity_type = DenseViewBase<T, U>;
    static constexpr bool is_sparse() { return false; }
    /**
     * Whether the sparsity information is stored inline with the data.
     * If true, query the space_required() function to get the additional space needed.
     */
    static constexpr bool inline_storage() { return false; }
    /**
     * Whether the sparsity information allocates additional storage.
     */
    static constexpr bool allocates_storage() { return false; }
    /**
     * How much additional space (in units of size_type) is required to store the sparsity information.
     */
    template<std::size_t NDIM>
    static constexpr size_type required_space(const std::array<size_type, NDIM>&) {
      return 0;
    }

    static constexpr std::string name() {
      return "DenseViewBase";
    }
  };

  namespace concepts {
    template<typename T,
             typename DerivedType = typename sparsity_traits<std::decay_t<T>>::derived_type,
             typename ValueType = typename sparsity_traits<std::decay_t<T>>::value_type>
    concept SparsityBase = std::is_same_v<T, DenseViewBase<DerivedType, ValueType>> ||
                          std::is_same_v<T, SparseArrayBase<DerivedType, ValueType>> ||
                          std::is_same_v<T, RangeSparsityBase<DerivedType, ValueType>>;
  } // namespace concepts

  /**
   * Traits for sparsity views.
   */
  template<typename T>
  constexpr bool is_sparsity_view_v = sparsity_traits<std::decay_t<T>>::is_sparse();


  inline std::ostream& operator<<(std::ostream& os, const concepts::SparsityBase auto& si) {
    auto count = si.count();
    os << "[" << count << ": ";
    for (size_type i = 0; i < count; ++i) {
      if (si.is_nonzero(i)) {
        os << "N";
      } else if (si.is_allocated(i)) {
        os << "A";
      } else {
        os << "Z";
      }
      if (i + 1 < count) {
        os << ",";
      }
    }
    os << "]";
    return os;
  }


} // namespace mra

#endif // MRA_TENSOR_SPARSITY_H