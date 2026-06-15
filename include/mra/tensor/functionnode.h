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
        tensor_type m_coeffs; //< the coefficients of the node, with the first dimension corresponding to the function index. The sparsity of this tensor encodes the sparsity of the node.
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

        bool is_nonzero(size_type i) const {
          return !sparsity().is_zero(i);
        }

        bool is_zero(size_type i) const {
          return sparsity().is_zero(i);
        }

        bool is_all_zero() const {
          return !sparsity().is_any_nonzero();
        }

        bool is_any_nonzero() const {
          return sparsity().is_any_nonzero();
        }

        bool is_all_nonzero() const {
          return sparsity().is_all_nonzero();
        }

        /**
         * Sets the i'th function to zero on the host.
         * Does not automatically update any device information.
         */
        void set_zero(size_type i) {
          sparsity().set_zero(i);
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

        template<typename U>
        friend std::ostream& operator<<(std::ostream& s, const FunctionsReconstructedNode<U,NDIM>& node);

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

        bool is_leaf_or_invalid(size_type i) const {
          return m_metadata[i].status == LeafStatus::Leaf || m_metadata[i].status == LeafStatus::Invalid;
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

        friend std::ostream& operator<<(std::ostream& s, const FunctionsReconstructedNode<T,NDIM>& node) {
          std::cout << "FunctionsReconstructedNode(" << node.key() << ", leaf status " << " [";
          for (size_type i = 0; i < node.count(); ++i) {
            std::cout << node.leaf_status(i) << (i < node.count() - 1 ? ", " : "");
          }
          std::cout << "], norms ";
          if (node.coeffs().buffer().is_current_on(ttg::device::Device::host())) {
            std::cout << "[";
            for (size_type i = 0; i < node.count(); ++i) {
              if (node.is_nonzero(i)) {
                std::cout << normf(node.coeffs().host_view()(i));
              } else {
                std::cout << "Z";
              }
              std::cout << (i < node.count() - 1 ? ", " : "");
            }
            std::cout << "]";
          } else {
            std::cout << "N/A";
          }
          std::cout << node.coeffs().dims() << ", " << node.sparsity() << ")";
          return s;
        }

    };

    namespace detail {


      /**
       * Struct that carries the leaf status for each child of a node.
       */
      template<Dimension NDIM>
      struct ChildLeafInfo : public ttg::TTValue<ChildLeafInfo<NDIM>> {
      private:
        /**
         * We store the information on whether the child exists, so that we can automatically initialize
         * it to false (i.e., leaf) and only set it to true if we know it has a child.
         */
        std::vector<std::array<bool, Key<NDIM>::num_children()>> m_have_child; // shape (num_functions, num_children)

      public:

        /**
         * Default constructore needed for TTG
         */
        ChildLeafInfo() = default;

        /**
         * Copy not allowed, but move is fine
         */
        ChildLeafInfo(const ChildLeafInfo& other) = delete;
        ChildLeafInfo& operator=(const ChildLeafInfo& other) = delete;
        ChildLeafInfo(ChildLeafInfo&& other) = default;
        ChildLeafInfo& operator=(ChildLeafInfo&& other) = default;

        explicit ChildLeafInfo(size_type N)
        : m_have_child(N)
        { }

        void set_all_child_leaf(bool value) {
          if (m_have_child.size() == 0) {
            throw std::runtime_error("Function index out of bounds in set_child_leaf");
          }
          for (size_type i = 0; i < m_have_child.size(); ++i) {
            m_have_child[i] = !value;
          }
        }

        void set_child_leaf(int fn, int child, bool value = true) {
          if (m_have_child.size() <= fn) {
            throw std::runtime_error("Function index out of bounds in set_child_leaf");
          }
          m_have_child[fn][child] = !value;
        }

        void set_child_leaf(int fn, const Key<NDIM>& child, bool value = true) {
          if (m_have_child.size() <= fn) {
            throw std::runtime_error("Function index out of bounds in set_child_leaf");
          }
          m_have_child[fn][child.childindex()] = !value;
        }

        void set_all_child_leaf(int fn, bool value = true) {
          if (m_have_child.size() <= fn) {
            throw std::runtime_error("Function index out of bounds in set_child_leaf");
          }
          m_have_child[fn].fill(!value);
        }

        bool is_child_leaf(int fn, int child) const {
          return m_have_child.size() == 0 || !m_have_child[fn][child];
        }

        bool is_child_leaf(int fn, const Key<NDIM>& child) const {
          return m_have_child.size() == 0 || !m_have_child[fn][child.childindex()];
        }

        bool is_all_child_leaf(int fn) const {
          if (m_have_child.size() <= fn) {
            throw std::runtime_error("Function index out of bounds in is_child_leaf");
          }
          return !std::any_of(m_have_child[fn].begin(), m_have_child[fn].end(), [](bool has_child){
                    return has_child;
                  });
        }

        bool is_all_child_leaf(const Key<NDIM>& key) const {
          return !std::any_of(m_have_child.begin(), m_have_child.end(), [&](const auto& child_array){
                    return child_array[key.childindex()];
                  });
        }

        bool is_all_child_leaf() const {
          return !std::any_of(m_have_child.begin(), m_have_child.end(), [](const auto& child_array){
                    return std::any_of(child_array.begin(), child_array.end(), [](bool has_child){
                      return has_child;
                    });
                  });
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          ar& this->m_have_child;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }

        friend std::ostream& operator<<(std::ostream& s, const ChildLeafInfo& info) {
          std::cout << "ChildLeafInfo[";
          for (size_type i = 0; i < info.m_have_child.size(); ++i) {
            std::cout << "[";
            for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
              std::cout << !info.m_have_child[i][c] << (c < Key<NDIM>::num_children() - 1 ? ", " : "");
            }
            std::cout << "]" << (i < info.m_have_child.size() - 1 ? ", " : "");
          }
          std::cout << "]";
          return s;
        }
      };

    } // namespace detail


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
        using child_info_type = detail::ChildLeafInfo<NDIM>;

      private:
        /**
         * We need to keep track of which children are leaf nodes. This is important so
         * that in reconstruct we can mark individual functions as leafs/inner/invalid.
         * Any attempt to reduce this to a single bool per child is futile and should not
         * be attempted. Ask me how I know.
         */
        child_info_type m_child_is_leaf;
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
        , m_child_is_leaf(N)
        { }

        FunctionsCompressedNode(const Key<NDIM>& key, size_type N, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, N, 2*K, scope)
        , m_child_is_leaf(N)
        { }

        FunctionsCompressedNode(const Key<NDIM>& key, const SparsityInfo& sparsity, size_type K, ttg::scope scope = ttg::scope::SyncIn)
        : base_type(key, sparsity, 2*K, scope)
        , m_child_is_leaf(base_type::count())
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


        void set_child_leaf(int fn, int child, bool value = true) {
          m_child_is_leaf.set_child_leaf(fn, child, value);
        }

        void set_child_leaf(int fn, const Key<NDIM>& child, bool value = true) {
          m_child_is_leaf.set_child_leaf(fn, child, value);
        }

        void set_all_child_leaf(int fn, bool value = true) {
          m_child_is_leaf.set_all_child_leaf(fn, value);
        }

        bool is_child_leaf(int fn, int child) const {
          return m_child_is_leaf.is_child_leaf(fn, child);
        }

        bool is_child_leaf(int fn, const Key<NDIM>& child) const {
          return m_child_is_leaf.is_child_leaf(fn, child);
        }

        bool is_all_child_leaf(int fn) const {
          return m_child_is_leaf.is_all_child_leaf(fn);
        }

        bool is_all_child_leaf(const Key<NDIM>& key) const {
          return m_child_is_leaf.is_all_child_leaf(key);
        }

        bool is_all_child_leaf() const {
          return m_child_is_leaf.is_all_child_leaf();
        }

        const child_info_type& child_info() const {
          return m_child_is_leaf;
        }

        void set_ns(bool arg = true) {
          m_ns = arg;
        }

        bool is_ns() const {
          return m_ns;
        }

        void clear() {
          base_type::clear();
          set_all_child_leaf(true);
        }

        void make_empty() {
          base_type::clear();
        }

        template <typename Archive>
        void serialize(Archive& ar) {
          base_type::serialize(ar);
          ar& this->m_child_is_leaf;
          ar& this->m_ns;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          serialize(ar);
        }

        friend std::ostream& operator<<(std::ostream& s, const FunctionsCompressedNode<T,NDIM>& node) {
          std::cout << "FunctionsCompressedNode(" << node.key() << ", ns " << node.is_ns()
                    << ", " << node.m_child_is_leaf << ", norm ";
          if (node.coeffs().buffer().is_current_on(ttg::device::Device::host())) {
            std::cout << "[";
            for (size_type i = 0; i < node.count(); ++i) {
              if (node.is_nonzero(i)) {
                std::cout << normf(node.coeffs().host_view()(i));
              } else {
                std::cout << "Z";
              }
              std::cout << (i < node.count() - 1 ? ", " : "");
            }
            std::cout << "]";
          } else {
            std::cout << "N/A";
          }
          std::cout << ", " << node.coeffs().dims() << ", " << node.sparsity() << ")";
          return s;
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
        //std::cout << "apply_leaf_info for function " << target.key() << " index " << i
        //          << ": any_is_leaf " << any_is_leaf << ", any_is_inner " << any_is_inner << std::endl;
        if (any_is_leaf || any_is_inner) {
          target.set_leaf(i, LeafStatus::Inner);
        } else { // TODO: not sure what to set here, since we don't know what the status of the current node is
          target.set_leaf(i, LeafStatus::Invalid);
        }
      }
    }


    /**
     * Copy the leaf information the source to the target node.
     */
    template<typename T, Dimension NDIM>
    void apply_leaf_info(FunctionsReconstructedNode<T, NDIM>& target, const FunctionsReconstructedNode<T, NDIM>& src) {
      for (size_type i = 0; i < target.count(); ++i) {
        //std::cout << "apply_leaf_info for function " << target.key() << " index " << i
        //          << ": any_is_leaf " << any_is_leaf << ", any_is_inner " << any_is_inner << std::endl;
        target.set_leaf(i, src.leaf_status(i));
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
        for (int c = 0; c < Key<NDIM>::num_children(); ++c) {
          bool all_is_leaf = ((src.invalid() || src.is_child_leaf(i, c)) && ...); // actual leaf
          //std::cout << "apply_leaf_info for compressed node " << target.key() << " index " << i << " child " << c
          //          << ": any_is_child_leaf " << all_is_leaf << std::endl;
          target.set_child_leaf(i, c, all_is_leaf);
        }
      }
    }

    template<typename T, Dimension NDIM, typename... Nodes>
    requires((std::is_same_v<FunctionsReconstructedNode<T, NDIM>, std::decay_t<Nodes>> && ...)
          && sizeof...(Nodes) == Key<NDIM>::num_children())
    void apply_leaf_info(FunctionsCompressedNode<T, NDIM>& target, Nodes&&... src) {
      for (size_type i = 0; i < target.count(); ++i) {
        std::array<bool, Key<NDIM>::num_children()> is_child_leaf = {src.is_leaf_or_invalid(i)...};
        for (size_type c = 0; c < Key<NDIM>::num_children(); ++c) {
          target.set_child_leaf(i, c, is_child_leaf[c]);
          //std::cout << "apply_leaf_info for compressed node " << target.key() << " index " << i << " child " << c
          //          << ": is_child_leaf " << is_child_leaf[c] << std::endl;
        }
      }
    }

    template <typename T, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionsReconstructedNode<T,NDIM>& node) {
      std::cout << "FunctionsReconstructedNode(" << node.key() << ", leaf status " << node.is_all_leaf() << " [";
      for (size_type i = 0; i < node.count(); ++i) {
        std::cout << node.leaf_status(i) << (i < node.count() - 1 ? ", " : "");
      }
      std::cout << "], " << node.coeffs().dims() << ", " << node.sparsity() << ")";
      //for (size_type i = 0; i < node.count(); ++i) {
      //  s << "FunctionsReconstructedNode[" << i << "](" << node.key() << ", leaf " << node.is_leaf(i) << ", norm " << mra::normf(node.coeffs_view(i)) << ")";
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
