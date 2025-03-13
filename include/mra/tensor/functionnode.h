#ifndef MRA_FUNCTIONNODE_H
#define MRA_FUNCTIONNODE_H

#include "mra/misc/key.h"
#include "mra/ops/functions.h"
#include "mra/tensor/tensor.h"

#include <ttg/serialization/std/vector.h>
#include <ttg/serialization/std/array.h>


namespace mra {

    namespace detail {

      template<Dimension NDIM, std::size_t... Is>
      std::array<size_type, NDIM> make_dims_helper(size_type N, size_type K, std::index_sequence<Is...>) {
        return std::array<size_type, NDIM>{N, ((void)Is, K)...};
      }
      /* helper to create {N, K, K, K, ...} dims array */
      template<Dimension NDIM>
      std::array<size_type, NDIM> make_dims(size_type N, size_type K) {
        return make_dims_helper<NDIM>(N, K, std::make_index_sequence<NDIM-1>{});
      }

      template<typename T, Dimension NDIM>
      class FunctionNodeBase {
      public: // temporarily make everything public while we figure out what we are doing
        static constexpr Dimension ndim() { return NDIM; }
        using key_type = Key<NDIM>;
        using value_type = T;
        using tensor_type = Tensor<value_type,NDIM+1>;
        using view_type   = TensorView<value_type, NDIM>;
        using const_view_type   = TensorView<const value_type, NDIM>;
        static constexpr bool is_function_node = true;
        using norm_tensor_type = Tensor<value_type, 1>;
        using norm_tensor_view_type = TensorView<const value_type, NDIM>;

      protected:
        key_type m_key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        tensor_type m_coeffs; //< if !is_leaf these are junk (and need not be communicated)
        size_type m_num_func = 0;
#ifdef MRA_CHECK_NORMS
        norm_tensor_type m_norms;
#endif // MRA_CHECK_NORMS

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
          m_coeffs = tensor_type(detail::make_dims<ndim()+1>(m_num_func, K), scope);
#else
          m_coeffs = tensor_type(detail::make_dims<ndim()+1>(m_num_func, K), ttg::scope::SyncIn); // make sure we allocate on host
#endif
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

        view_type coeffs_view(size_type i){
          return m_coeffs.current_view()(i);
        }

        const view_type coeffs_view(size_type i) const {
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

        auto& buffer() {
          return m_coeffs.buffer();
        }

        const auto& buffer() const {
          return m_coeffs.buffer();
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
        using key_type = Key<NDIM>;
        using value_type = T;
        using tensor_type = Tensor<T,NDIM+1>;
        using view_type   = TensorView<T, NDIM>;
        using const_view_type   = TensorView<const T, NDIM>;
        static constexpr bool is_function_node = true;
        using norm_tensor_type = Tensor<T, 1>;
        using norm_tensor_view_type = TensorView<const T, NDIM>;
        using base_type = detail::FunctionNodeBase<T, NDIM>;
        constexpr static Dimension ndim() { return NDIM; }

      private:
        struct function_metadata {
          T sum = 0.0;
          bool is_leaf = false;
          std::array<bool, Key<NDIM>::num_children()> is_child_leaf = { false };
          template<typename Archive>
          void serialize(Archive& ar){
            ar & sum;
            ar & is_leaf;
            ar & is_child_leaf;
          }
        };

        std::vector<function_metadata> m_metadata;

      public:
        FunctionsReconstructedNode() = default;

        /* constructs a node with metadata for N functions and all coefficients zero */
        FunctionsReconstructedNode(const Key<NDIM>& key, size_type N)
        : base_type(key, N)
        , m_metadata(N)
        { }

        FunctionsReconstructedNode(const Key<NDIM>& key, size_type N, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, N, K, scope)
        , m_metadata(N)
        { }


        FunctionsReconstructedNode(FunctionsReconstructedNode&& other) = default;
        FunctionsReconstructedNode(const FunctionsReconstructedNode& other) = delete;

        FunctionsReconstructedNode& operator=(FunctionsReconstructedNode&& other) = default;
        FunctionsReconstructedNode& operator=(const FunctionsReconstructedNode& other) = delete;

        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(size_type K, ttg::scope scope = ttg::scope::SyncIn) {
          base_type::allocate(K, scope);
        }

        bool has_children(size_type i) const {
          return !m_metadata[i].is_leaf;
        }

        bool any_have_children() const {
          bool result = false;
          for (size_type i = 0; i < m_metadata.size(); ++i) {
            result |= has_children(i);
          }
          return result;
        }

        void set_all_leaf(bool val) {
          for (auto& data : m_metadata) {
            data.is_leaf = val;
          }
        }

        bool is_all_leaf() const {
          bool all_leaf = true;
          for (auto& data : m_metadata) {
            all_leaf &= data.is_leaf;
          }
          return all_leaf;
        }

        bool& is_leaf(size_type i) {
          return m_metadata[i].is_leaf;
        }

        bool is_leaf(size_type i) const {
          return m_metadata[i].is_leaf;
        }

        bool& is_child_leaf(size_type i, size_type child) {
          return m_metadata[i].is_child_leaf[child];
        }

        bool is_leaf(size_type i, size_type child) const {
          return m_metadata[i].is_child_leaf[child];
        }

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
    };


    template <typename T, Dimension NDIM>
    class FunctionsCompressedNode : public ttg::TTValue<FunctionsCompressedNode<T, NDIM>>,
                                    public detail::FunctionNodeBase<T, NDIM> {
      public: // temporarily make everything public while we figure out what we are doing
        static constexpr bool is_function_node = true;
        using key_type          = Key<NDIM>;
        using view_type         = TensorView<T, NDIM>;
        using const_view_type   = TensorView<const T, NDIM>;
        using norm_tensor_type = Tensor<T, 1>;
        using norm_tensor_view_type = TensorView<const T, NDIM>;
        using base_type = detail::FunctionNodeBase<T, NDIM>;

      private:
        std::vector<std::array<bool, Key<NDIM>::num_children()>> m_is_child_leafs; //< True if that child is leaf on tree

      public:
        FunctionsCompressedNode() = default; // needed for serialization

        /* constructs a node for N functions with zero coefficients */
        FunctionsCompressedNode(const Key<NDIM>& key, size_type N)
        : base_type(key, N)
        , m_is_child_leafs(N)
        {
          set_all_child_leafs(true);
        }

        FunctionsCompressedNode(const Key<NDIM>& key, size_type N, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, N, 2*K, scope)
        , m_is_child_leafs(N)
        { }

        /**
         * Allocate space for coefficients using K.
         * The node must be empty before and will not be empty afterwards.
         */
        void allocate(size_type K, ttg::scope scope = ttg::scope::SyncIn) {
          base_type::allocate(2*K, scope);
        }

        FunctionsCompressedNode(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode(const FunctionsCompressedNode& other) = delete;

        FunctionsCompressedNode& operator=(FunctionsCompressedNode&& other) = default;
        FunctionsCompressedNode& operator=(const FunctionsCompressedNode& other) = delete;

        bool has_children(size_type i, int childindex) const {
            assert(childindex < Key<NDIM>::num_children());
            assert(i < m_is_child_leafs.size());
            return !m_is_child_leafs[i][childindex];
        }

        std::array<bool, Key<NDIM>::num_children()>& is_child_leaf(size_type i) {
          return m_is_child_leafs[i];
        }

        const std::array<bool, Key<NDIM>::num_children()>& is_child_leaf(size_type i) const {
          return m_is_child_leafs[i];
        }

        bool is_child_leaf(size_type i, size_type child) const {
          return m_is_child_leafs[i][child];
        }

        void set_all_child_leafs(bool arg = true) {
          for (auto& node : m_is_child_leafs) {
            for (auto& c : node) {
              c = arg;
            }
          }
        }

        bool is_all_child_leaf() const {
          bool result = true;
          for (const auto& node : m_is_child_leafs) {
            for (const auto& c : node) {
              result &= c;
            }
          }
          return result;
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          base_type::serialize(ar);
          ar& this->m_is_child_leafs;
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
        target.is_leaf(i) = (src.is_leaf(i) && ...);
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
          target.is_child_leaf(i)[j] = (src.is_child_leaf(i)[j] && ...);
        }
      }
    }

    template<typename T, Dimension NDIM, typename... Nodes>
    requires((std::is_same_v<FunctionsReconstructedNode<T, NDIM>, std::decay_t<Nodes>> && ...)
          && sizeof...(Nodes) == Key<NDIM>::num_children())
    void apply_leaf_info(FunctionsCompressedNode<T, NDIM>& target, Nodes&&... src) {
      for (std::size_t i = 0; i < target.count(); ++i) {
        target.is_child_leaf(i) = std::array{src.is_leaf(i)...};
      }
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsReconstructedNode<T,NDIM>& node) {
      for (size_type i = 0; i < node.count(); ++i) {
        s << "FunctionsReconstructedNode[" << i << "](" << node.key() << ", leaf " << node.is_leaf(i) << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      }
      return s;
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsCompressedNode<T,NDIM>& node) {
      for (size_type i = 0; i < node.count(); ++i) {
        s << "FunctionsCompressedNode[" << i << "](" << node.key() << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
      }
      return s;
    }



} // namespace mra

#endif // HAVE_MRA_FUNCTIONNODE_H
