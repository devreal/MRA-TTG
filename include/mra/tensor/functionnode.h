#ifndef MRA_FUNCTIONNODE_H
#define MRA_FUNCTIONNODE_H

#include "mra/misc/key.h"
#include "mra/ops/functions.h"
#include "mra/tensor/leafstatus.h"
#include "mra/tensor/tensor.h"

#include <ttg/serialization/std/vector.h>
#include <ttg/serialization/std/array.h>


namespace mra {

    namespace detail {

      /**
       * Notes on Sparsity in the functionnode:
       * - The FunctionNodes store data in Tensors, that in turn encode sparsity.
       * - We need to split the tensor allocation from the construction of the sparsity of the node
       *   so that we can gather the sparsity information and then allocate the underlying tensor memory.
       * - The sparsity should be gathered in a separate object and passed to the functionnode/tensor
       *   constructor and allocate() function to apply it there. The fallback will be the DenseSparsity.
       */

      template<typename T, Dimension NDIM>
      class FunctionNodeBase {
      public: // temporarily make everything public while we figure out what we are doing
        static constexpr Dimension ndim() { return NDIM; }
        using key_type = Key<NDIM>;
        using value_type = T;
        using tensor_type = SparseTensor<value_type,NDIM+1>;
        using view_type   = SparseTensorView<value_type, NDIM>;
        using const_view_type   = SparseTensorView<const value_type, NDIM>;
        using subview_type   = DenseTensorView<value_type, NDIM>;
        using const_subview_type  = DenseTensorView<const value_type, NDIM>;
        using norm_tensor_type = DenseTensor<value_type, 1>;
        using norm_tensor_view_type = DenseTensorView<const value_type, NDIM>;
        using sparsity_type = typename tensor_type::sparsity_type;

      protected:
        key_type m_key = key_type::invalid(); //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        tensor_type m_coeffs; //< if !is_leaf these are junk (and need not be communicated)
        size_type m_num_func = 0;
#ifdef MRA_CHECK_NORMS
        norm_tensor_type m_norms;
#endif // MRA_CHECK_NORMS


        /**
         * Non-const sparsity accessor, not public.
         */
        sparsity_type& sparsity() {
          return m_coeffs.sparsity();
        }


      public:

        FunctionNodeBase() = default;

        /* constructs a node with metadata for N functions and all coefficients zero */
        FunctionNodeBase(const key_type& key)
        : m_key(key)
        , m_coeffs()
        { }

        /* constructs a node with metadata for N functions and all coefficients zero */
        FunctionNodeBase(const key_type& key, size_type N)
        : m_key(key)
        , m_coeffs()
        , m_num_func(N)
        { }

        FunctionNodeBase(const key_type& key, size_type N, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : m_key(key)
#ifdef MRA_ENABLE_HOST
        , m_coeffs(make_dims<ndim()+1>(N, K), ttg::scope::SyncIn) // make sure we allocate on host
#else
        , m_coeffs(make_dims<ndim()+1>(N, K), scope)
#endif
        , m_num_func(N)
        { }

        FunctionNodeBase(const key_type& key, const SparsityInfo& sparsity, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : m_key(key)
#ifdef MRA_ENABLE_HOST
        , m_coeffs(sparsity, K, ttg::scope::SyncIn) // make sure we allocate on host
#else
        , m_coeffs(sparsity, K, scope)
#endif
        , m_num_func(sparsity.dim(0))
        { }

        FunctionNodeBase(FunctionNodeBase&& other) = default;
        FunctionNodeBase(const FunctionNodeBase& other) = delete;

        FunctionNodeBase& operator=(FunctionNodeBase&& other) = default;
        FunctionNodeBase& operator=(const FunctionNodeBase& other) = delete;


        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(size_type K, ttg::scope scope = ttg::scope::SyncIn) {
          if (!empty()) throw std::runtime_error("Reallocating non-empty FunctionNode not allowed!");
          if (m_num_func == 0) throw std::runtime_error("Cannot reallocate FunctionNode with N = 0");
#ifndef MRA_ENABLE_HOST
          m_coeffs = tensor_type(make_dims<ndim()+1>(m_num_func, K), scope);
#else
          m_coeffs = tensor_type(make_dims<ndim()+1>(m_num_func, K), ttg::scope::SyncIn); // make sure we allocate on host
#endif
        }


        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(const SparsityInfo& sparsity, size_type K, ttg::scope scope = ttg::scope::SyncIn) {
          if (!empty()) throw std::runtime_error("Reallocating non-empty FunctionNode not allowed!");
          if (m_num_func == 0) throw std::runtime_error("Cannot reallocate FunctionNode with N = 0");

#ifndef MRA_ENABLE_HOST
          /**
           * Currently needed to ensure the buffer is allocated on the host.
           */
          scope = ttg::scope::SyncIn;
#endif // MRA_ENABLE_HOST
          m_coeffs = tensor_type(sparsity, K, scope);
        }

        /* with C++23 we could the following:
        auto& coeffs(this FunctionsReconstructedNode&& self) {
          return self.m_coeffs;
        }
        */
        auto& coeffs() {
          return m_coeffs;
        }

        const auto& coeffs() const {
          return m_coeffs;
        }

        subview_type coeffs_view(size_type i){
          return m_coeffs.current_view()(i);
        }

        const_subview_type coeffs_view(size_type i) const {
          return m_coeffs.current_view()(i);
        }

#ifdef MRA_CHECK_NORMS
        auto& norms() {
          return m_norms;
        }

        const auto& norms() const {
          return m_norms;
        }

        view_type norms_view(size_type i) {
          return m_norms.current_view()(i);
        }

        const view_type norms_view(size_type i) const {
          return m_norms.current_view()(i);
        }

#else  // MRA_CHECK_NORMS

        auto norms() {
          return ttg::Void{};
        }

        const auto norms() const {
          return ttg::Void{};
        }

        view_type norms_view(size_type i) {
          return ttg::Void{};
        }

        const view_type norms_view(size_type i) const {
          return ttg::Void{};
        }

#endif // MRA_CHECK_NORMS

        key_type& key() {
          return m_key;
        }

        const key_type& key() const {
          return m_key;
        }

        size_type count() const {
          return m_num_func;
        }

        bool empty() const {
          return m_coeffs.empty();
        }

        bool invalid() const {
          return m_key.is_invalid();
        }

        auto& buffer() {
          return m_coeffs.buffer();
        }

        const auto& buffer() const {
          return m_coeffs.buffer();
        }

        const auto& sparsity() const {
          return m_coeffs.sparsity();
        }

        void clear() {
          m_coeffs.clear();
        }

        void make_empty() {
          clear();
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          ar& this->m_key;
          ar& this->m_coeffs;
#ifdef MRA_CHECK_NORMS
          ar& this->m_norms;
#endif // MRA_CHECK_NORMS
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
      };
    } // namespace detail

    /* like FunctionReconstructedNode but for N functions */
    template <typename T, Dimension NDIM>
    class FunctionsReconstructedNode : public ttg::TTValue<FunctionsReconstructedNode<T, NDIM>>,
                                       public detail::FunctionNodeBase<T, NDIM> {
      public:
        using base_type = detail::FunctionNodeBase<T, NDIM>;
        using key_type = Key<NDIM>;
        using value_type = T;
        using tensor_type = typename base_type::tensor_type;
        using view_type   = typename base_type::view_type;
        using const_view_type   = typename base_type::const_view_type;
        using norm_tensor_type = typename base_type::norm_tensor_type;
        using norm_tensor_view_type = typename base_type::norm_tensor_view_type;
        constexpr static Dimension ndim() { return NDIM; }

      private:

        struct function_metadata {
          T sum = 0.0;
          LeafStatus status = LeafStatus::Inner;
          //std::array<bool, Key<NDIM>::num_children()> is_child_leaf = { false };
          template<typename Archive>
          void serialize(Archive& ar){
            ar & sum;
            ar & status;
            //ar & is_child_leaf;
          }
        };

        std::vector<function_metadata> m_metadata;

      public:
        /* constructs an empty node without key information,
         * needed for default construction during serialization
         * but should otherwise not be used */
        FunctionsReconstructedNode() = default;

        /* constructs an empty node with key information */
        FunctionsReconstructedNode(const Key<NDIM>& key)
        : base_type(key)
        { }

        /* constructs a node with metadata for N functions and all coefficients zero */
        FunctionsReconstructedNode(const Key<NDIM>& key, size_type N)
        : base_type(key, N)
        , m_metadata(N)
        { }

        FunctionsReconstructedNode(const Key<NDIM>& key, size_type N, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, N, K, scope)
        , m_metadata(N)
        { }

        FunctionsReconstructedNode(const Key<NDIM>& key, const SparsityInfo& sparsity, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, sparsity, K, scope)
        , m_metadata(sparsity.dim(0))
        { }


        FunctionsReconstructedNode(FunctionsReconstructedNode&& other) = default;
        FunctionsReconstructedNode(const FunctionsReconstructedNode& other) = delete;

        FunctionsReconstructedNode& operator=(FunctionsReconstructedNode&& other) = default;
        FunctionsReconstructedNode& operator=(const FunctionsReconstructedNode& other) = delete;


#if 0
        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(size_type K, ttg::scope scope = ttg::scope::SyncIn) {
          base_type::allocate(K, scope);
        }

        void allocate(const SparsityInfo& sparsity, ttg::scope scope = ttg::scope::SyncIn) {
          this->apply_sparsity(sparsity);
          base_type::allocate(sparsity.dim(0), scope);
        }
#endif // 0

        const auto& sparsity() const {
          return base_type::sparsity();
        }

        bool has_children(size_type i) const {
          return m_metadata[i].status == LeafStatus::Inner;
        }

        bool any_have_children() const {
          return std::any_of(m_metadata.begin(), m_metadata.end(), [](const function_metadata& data){
                    return data.status == LeafStatus::Inner;
                  });
        }


        /**
         * Set the status of the function.
         * Updates the host-side sparsity accordingly. The device-side sparsity must be handled separately.
         */
        void set_leaf(size_type i, LeafStatus status = LeafStatus::Leaf) {
          m_metadata[i].status = status;
        }

        /**
         * Set the status of all functions.
         */
        void set_all_leaf(LeafStatus status = LeafStatus::Leaf) {
          for (auto& data : m_metadata) {
            data.status = status;
          }
        }

        /**
         * Evaluates the given function on all functions and sets the leaf status to its returned value.
         */
        template<typename Fn>
        requires std::invocable<Fn, size_type>
        void set_all_leaf(Fn&& fn){
          for (size_type i = 0; i < m_metadata.size(); ++i) {
            set_leaf(i, fn(i));
          }
        }

        /**
         * Returns true if all nonzero nodes are leaf nodes.
         */
        bool is_all_leaf() const {
          return std::all_of(m_metadata.begin(), m_metadata.end(), [](const function_metadata& data){
                    return data.status != LeafStatus::Inner;
                  });
        }

        /**
         * Returns true if all nodes are either leaf nodes or invalid (zero) nodes.
         */
        bool is_all_leaf_or_invalid() const {
          return std::all_of(m_metadata.begin(), m_metadata.end(), [](const function_metadata& data){
                    return data.status == LeafStatus::Leaf || data.status == LeafStatus::Invalid;
                  });
        }

        /**
         * Returns true if any node is a leaf node (i.e., nonzero).
         */
        bool is_any_leaf() const {
          return std::any_of(m_metadata.begin(), m_metadata.end(), [](const function_metadata& data){
                    return data.status == LeafStatus::Leaf;
                  });
        }

        bool is_leaf(size_type i) const {
          return m_metadata[i].status == LeafStatus::Leaf;
        }

        bool is_invalid(size_type i) const {
          return m_metadata[i].status == LeafStatus::Invalid;
        }

        LeafStatus leaf_status(size_type i) const {
          return m_metadata[i].status;
        }

#if 0
        // TODO: needed?

        bool is_child_leaf(size_type i, size_type child) {
          return m_metadata[i].is_child_leaf[child];
        }
#endif // 0

        T& sum(size_type i) {
          return m_metadata[i].sum;
        }

        T sum(size_type i) const {
          return m_metadata[i].sum;
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          base_type::serialize(ar);
          ar& this->m_metadata;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }

        bool is_nonzero(size_type i) const {
          return !base_type::sparsity().is_zero(i);
        }

        bool is_any_nonzero() const {
          return base_type::sparsity().is_any_nonzero();
        }

        bool is_all_nonzero() const {
          return base_type::sparsity().is_all_nonzero();
        }

    };


    template <typename T, Dimension NDIM>
    class FunctionsCompressedNode : public ttg::TTValue<FunctionsCompressedNode<T, NDIM>>,
                                    public detail::FunctionNodeBase<T, NDIM> {
      public: // temporarily make everything public while we figure out what we are doing
        using base_type = detail::FunctionNodeBase<T, NDIM>;
        using key_type          = Key<NDIM>;
        using view_type         = typename base_type::view_type;
        using const_view_type   = typename base_type::const_view_type;
        using norm_tensor_type  = typename base_type::norm_tensor_type;
        using norm_tensor_view_type = typename base_type::norm_tensor_view_type;

      private:
        std::vector<std::array<LeafStatus, Key<NDIM>::num_children()>> m_child_leaf_status; //< True if that child is leaf on tree
        bool m_ns = false; //< True if node is non-standard

      public:
        /* constructs an empty node without key information,
        * needed for default construction during serialization
        * but should otherwise not be used */
        FunctionsCompressedNode() = default; // needed for serialization

        /* constructs a node for N functions with zero coefficients */
        FunctionsCompressedNode(const Key<NDIM>& key)
        : base_type(key)
        { }

        /* constructs a node for N functions with zero coefficients */
        FunctionsCompressedNode(const Key<NDIM>& key, size_type N)
        : base_type(key, N)
        , m_child_leaf_status(N)
        {
          set_all_child_leaf(LeafStatus::Leaf);
        }

        FunctionsCompressedNode(const Key<NDIM>& key, size_type N, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, N, 2*K, scope)
        , m_child_leaf_status(N)
        { }

        FunctionsCompressedNode(const Key<NDIM>& key, const SparsityInfo& sparsity, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, sparsity, 2*K, scope)
        , m_child_leaf_status(sparsity.dim(0))
        { }

        const auto& sparsity() const {
          return base_type::sparsity();
        }

        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(size_type K, ttg::scope scope = ttg::scope::SyncIn) {
          base_type::allocate(2*K, scope);
        }

        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(const SparsityInfo& sparsity, size_type K, ttg::scope scope = ttg::scope::SyncIn) {
          base_type::allocate(sparsity, 2*K, scope);
        }

        FunctionsCompressedNode(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode(const FunctionsCompressedNode& other) = delete;

        FunctionsCompressedNode& operator=(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode& operator=(const FunctionsCompressedNode& other) = delete;

#if 0
        bool has_children(size_type i, int childindex) const {
            assert(childindex < Key<NDIM>::num_children());
            assert(i < m_child_leaf_status.size());
            return !m_child_leaf_status[i][childindex];
        }
#endif // 0

        std::array<LeafStatus, Key<NDIM>::num_children()>& child_leaf_status(size_type i) {
          return m_child_leaf_status[i];
        }

        const std::array<LeafStatus, Key<NDIM>::num_children()>& child_leaf_status(size_type i) const {
          return m_child_leaf_status[i];
        }

        bool is_child_leaf(size_type i, size_type child) const {
          return m_child_leaf_status[i][child] == LeafStatus::Leaf;
        }

        bool is_child_leaf_or_invalid(size_type i, size_type child) const {
          return m_child_leaf_status[i][child] == LeafStatus::Leaf || m_child_leaf_status[i][child] == LeafStatus::Invalid;
        }

        bool is_child_all_leaf(const Key<NDIM>& child) const {
          bool result = true;
          for (size_type i = 0; i < m_child_leaf_status.size(); ++i) {
            result &= is_child_leaf(i, child.childindex());
          }
          return result;
        }

        bool is_child_all_leaf_or_invalid(const Key<NDIM>& child) const {
          bool result = true;
          for (size_type i = 0; i < m_child_leaf_status.size(); ++i) {
            result &= is_child_leaf_or_invalid(i, child.childindex());
          }
          return result;
        }


        void set_child_leaf(size_type i, size_type child, LeafStatus arg = LeafStatus::Leaf) {
          m_child_leaf_status[i][child] = arg;
        }

        void set_all_child_leaf(LeafStatus arg = LeafStatus::Leaf) {
          for (auto& node : m_child_leaf_status) {
            for (auto& c : node) {
              c = arg;
            }
          }
        }

        void set_ns(bool arg = true) {
          m_ns = arg;
        }

        bool is_ns() const {
          return m_ns;
        }

        void clear() {
          base_type::clear();
          set_all_child_leaf(LeafStatus::Inner);
        }

        void make_empty() {
          base_type::clear();
        }

        bool is_all_child_leaf() const {
          bool result = true;
          for (const auto& node : m_child_leaf_status) {
            for (const auto& c : node) {
              result &= c == LeafStatus::Leaf;
            }
          }
          return result;
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          base_type::serialize(ar);
          ar& this->m_child_leaf_status;
          ar& this->m_ns;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }
    };

    /**
     * Takes one or more reconstructed function nodes and applies the leaf information to the target node.
     * If the nodes of all functions of the source nodes are leafs then the the target
     * node will be leaf as well.
     */
    template<typename T, Dimension NDIM, typename... Nodes>
    requires((std::is_same_v<FunctionsReconstructedNode<T, NDIM>, std::decay_t<Nodes>> && ...)
          && sizeof...(Nodes) > 0)
    void apply_leaf_info(FunctionsReconstructedNode<T, NDIM>& target, Nodes&&... src) {
      for (size_type i = 0; i < target.count(); ++i) {
        bool any_is_leaf = (src.is_leaf(i) || ...); // actual leaf
        bool any_is_inner = (src.is_invalid(i) || ...); // inner node
        if (any_is_leaf || any_is_inner) {
          target.set_leaf(i, LeafStatus::Inner);
        } else { // TODO: not sure what to set here, since we don't know what the status of the current node is
          target.set_leaf(i, LeafStatus::Invalid);
        }
      }
    }

    /**
     * Takes one or more compressed function nodes and applies the child information to the target node.
     * If the children of all functions of the source nodes are leafs then the children of the target
     * node will be leafs as well.
     */
    template<typename T, Dimension NDIM, typename... Nodes>
    requires((std::is_same_v<FunctionsCompressedNode<T, NDIM>, std::decay_t<Nodes>> && ...)
          && sizeof...(Nodes) > 0)
    void apply_leaf_info(FunctionsCompressedNode<T, NDIM>& target, Nodes&&... src) {
      for (size_type i = 0; i < target.count(); ++i) {
        for (size_type j = 0; j < Key<NDIM>::num_children(); ++j) {
          bool is_child_leaf = (src.is_child_leaf_or_invalid(i, j) && ...);
          target.child_leaf_status(i)[j] = is_child_leaf ? LeafStatus::Leaf : LeafStatus::Inner;
        }
      }
    }

    template<typename T, Dimension NDIM, typename... Nodes>
    requires((std::is_same_v<FunctionsReconstructedNode<T, NDIM>, std::decay_t<Nodes>> && ...)
          && sizeof...(Nodes) >= Key<NDIM>::num_children())
    void apply_leaf_info(FunctionsCompressedNode<T, NDIM>& target, Nodes&&... src) {
      for (std::size_t i = 0; i < target.count(); ++i) {
        target.child_leaf_status(i) = std::array{src.leaf_status(i)...};
      }
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsReconstructedNode<T,NDIM>& node) {
      std::cout << "FunctionsReconstructedNode(" << node.key() << ", all_leafs " << node.is_all_leaf() << ", " << node.coeffs().dims() << ", " << node.sparsity() << ")";
      //for (size_type i = 0; i < node.count(); ++i) {
      //  s << "FunctionsReconstructedNode[" << i << "](" << node.key() << ", leaf " << node.is_leaf(i) << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      //}
      return s;
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsCompressedNode<T,NDIM>& node) {
      std::cout << "FunctionsCompressedNode(" << node.key() << ", ns " << node.is_ns() << ", all_child_leaf "
                << node.is_all_child_leaf() << ", norm " << mra::normf(node.coeffs_view(0)) << ", " << node.coeffs().dims() << ", " << node.sparsity() << ")";
      //for (size_type i = 0; i < node.count(); ++i) {
      //  s << "FunctionsCompressedNode[" << i << "](" << node.key() << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      //}
      return s;
    }

    namespace detail {
      template<typename T>
      struct is_functionnode : std::false_type {};

      template<typename T, Dimension NDIM>
      struct is_functionnode<FunctionsReconstructedNode<T, NDIM>> : std::true_type {};

      template<typename T, Dimension NDIM>
      struct is_functionnode<FunctionsCompressedNode<T, NDIM>> : std::true_type {};

      template<typename T>
      constexpr bool is_functionnode_v = is_functionnode<std::decay_t<T>>::value;
    } // namespace detail


    namespace concepts {

      template<typename T>
      concept FunctionNode = detail::is_functionnode_v<T>;
    } // namespace concepts


} // namespace mra

#endif // HAVE_MRA_FUNCTIONNODE_H
