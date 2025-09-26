#ifndef MRA_CONVOLUTIONDATA_H
#define MRA_CONVOLUTIONDATA_H

#include "mra/ops/inner.h"
#include "mra/misc/gl.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/types.h"
#include "mra/misc/adquad.h"
#include "mra/misc/twoscale.h"
#include "mra/misc/platform.h"
#include "mra/misc/autocorr.h"
#include "mra/tensor/tensorview.h"

#define MRA_MAX_LX 7
namespace mra {

  template <typename T>
  struct ConvolutionData {
    Tensor<T, 2> R, S;
    T normR, normS;
  };

  /// Nonstandard form of the operator
  template <typename T, Dimension NDIM>
  struct OperatorData {
    std::array<std::shared_ptr<const ConvolutionData<T>>, NDIM> ops;
    T norm;
    T fac;

    OperatorData() : ops{}, norm(0.0), fac(1.0) {}

    OperatorData(const OperatorData&) = default;

    ~OperatorData() = default;

  };

  template <typename T, Dimension NDIM>
  class Convolution {

    private:
      using ns_type = const ConvolutionData<T>;
      size_type K;
      int npt;                                         // number of quadrature points
      T expnt;                                         // exponent for the Gaussian
      T coeff;                                         // coefficient for the Gaussian
      const T* quad_x;                                 // quadrature points
      const T* quad_w;                                 // quadrature weights
      Tensor<T, 3> c;                                  // autocorrelation coefficients
      FunctionData<T, NDIM>& functiondata;             // function data
      FunctionData<T, NDIM>& functiondata2;            // second function data for rnlp at finer level
      mutable std::map<Key<NDIM>, Tensor<T, 2>> rnlijcache;    // map for storing rnlij matrices
      mutable std::map<Key<NDIM>, Tensor<T, 1>> rnlpcache;     // map for storing rnlp matrices
      mutable std::map<Key<NDIM>, std::shared_ptr<ns_type>> nscache; // map for storing ns matrices
      mutable std::mutex cachemutex;                   // mutex for thread safety

      void autoc(){
        Tensor<T, 3> autocorrcoef(K, K, 4*K);
        auto autocorr_view = autocorrcoef.current_view();
        detail::autocorr_get<T>(K, autocorr_view);
        auto c_view = c.current_view();
        c_view = 0.0;
        std::array<Slice,NDIM> slices = {Slice(0, K), Slice(0, K), Slice(0, 2*K)};
        c_view(slices) = autocorr_view(slices);
        slices = {Slice(0, K), Slice(0, K), Slice(2*K, 4*K)};
        c_view(slices) = autocorr_view(slices);
      }



    public:

      Convolution(size_type K, int npt, T coeff, T expnt, FunctionData<T, NDIM>& functiondata, FunctionData<T, NDIM>& functiondata2)
        : K(K), npt(npt), c(K, K, 4*K), coeff(coeff), expnt(expnt), functiondata(functiondata), functiondata2(functiondata2) {
        GLget(&quad_x, &quad_w, npt);
        autoc();
      }

      Convolution(Convolution&&) = default;
      Convolution(const Convolution&) = delete;
      Convolution& operator=(Convolution&&) = default;
      Convolution& operator=(const Convolution&) = delete;

      bool issmall(Level n, Translation lx) const {
        T beta = expnt * std::pow(T(0.25), T(n));
        Translation l;
        if (l > 0) l = lx-1;
        else if (lx < 0) l = -lx-1;
        else l = 0;

        return beta*l*l > 49.0; // heuristic cutoff (empirical)
      }

      // projection of a Gaussian onto double order polynomials
      const Tensor<T, 1>& make_rnlp(const Level n, Translation lx) const {
        mra::Key<NDIM> key(n, std::array<Translation, NDIM>({lx}));
        auto it = rnlpcache.find(key);
        if (it != rnlpcache.end()) {
          const auto& r = it->second;
          return r;
        }

        Tensor<T, 1> rnlp(2*K);
        auto rnlp_view = rnlp.current_view();
        rnlp_view = 0.0;

        Translation lkeep = lx;
        if (lx < 0) lx = -lx-1;
        T scaledcoeff  = coeff*std::pow(0.5, 0.5*n);
        T beta = expnt * std::pow(T(0.25), T(n));
        T h = 1.0/std::sqrt(beta);
        size_type nbox = size_type(1.0/h);
        if (nbox < 1) nbox = 1;
        h = 1.0/nbox;
        T sch = std::abs(scaledcoeff*h);
        T argmax = std::abs(std::log(1e-22/sch));

        for (size_type box=0; box<nbox; ++box){
          T xlo = box*h + lx;
          if (beta*xlo*xlo > argmax) break;

          for (size_type i=0; i<npt; ++i){
            T* phix = new T[2*K];
            T xx = xlo + h*quad_x[i];
            T ee = scaledcoeff*std::exp(-beta*xx*xx)*quad_w[i]*h;
            legendre_scaling_functions(xx-lx, 2*K, &phix[0]);
            for (size_type p=0; p<2*K; ++p) {
              rnlp_view(p) += ee*phix[p];
            }
            delete[] phix;
          }
          if (lkeep < 0) {
            for (int p = 1; p < 2*K; p += 2) {
              rnlp_view(p) = -rnlp_view(p);
            }
          }
        }
        cachemutex.lock();
        if (rnlpcache.find(key) == rnlpcache.end()) {
          rnlpcache.emplace(key, std::move(rnlp));
        }
        it = rnlpcache.find(key);
        cachemutex.unlock();
        const auto& r = it->second;
        return r;
      }

      const Tensor<T, 1>& get_rnlp (const Level n, const Translation lx) const {
        mra::Key<NDIM> key(n, std::array<Translation, NDIM>({lx}));
        auto it = rnlpcache.find(key);
        if (it != rnlpcache.end()) {
          const auto& r = it->second;
          return r;
        }

        Tensor<T, 1> rnlp;
        Level natlev = 0.5*std::log2(expnt) + 1; // natural level for Gaussian

        if (issmall(n, lx)) {
          rnlp = Tensor<T, 1>(2*K); // initialize it to zero
          auto rnlp_view = rnlp.current_view();
          rnlp_view = 0.0;
          return rnlp; // zero
        }
        else if (n < natlev) {
          // compute at a finer level
          Tensor<T, 1> tmp(4*K), R(4*K), work(2*K);
          const auto& r1 = get_rnlp(n+1, 2*lx);
          const auto& r2 = get_rnlp(n+1, 2*lx+1);

          auto tmp_view = tmp.current_view();
          auto r1_view = r1.current_view();
          auto r2_view = r2.current_view();

          std::array<Slice,1> slice1 = {Slice(0, 2*K)};
          std::array<Slice,1> slice2 = {Slice(2*K, 4*K)};
          tmp_view(slice1) = r1_view(slice1);
          tmp_view(slice2) = r2_view(slice1);

          const auto& hgTtwo = functiondata2.get_hgT();
          auto hgTtwo_view = hgTtwo.current_view();
          auto R_view = rnlp.current_view();
          transform(tmp_view, hgTtwo_view, R_view, work.data());

          rnlp = Tensor<T, 1>(2*K);
          auto rnlp_view = rnlp.current_view();
          rnlp_view(slice1) = R_view(slice1);

          cachemutex.lock();
          if (rnlpcache.find(key) == rnlpcache.end()) {
            rnlpcache.emplace(key, std::move(rnlp));
          }
          it = rnlpcache.find(key);
          cachemutex.unlock();
          const auto& r = it->second;
          return r;
        }
        else {
          // the usual computation
          return make_rnlp(n, lx);
        }


      }

      const Tensor<T, 2>& make_rnlij (const Level n, const Translation lx) const {
        mra::Key<NDIM> key(n, std::array<Translation, NDIM>({lx}));
        cachemutex.lock();
        auto it = rnlijcache.find(key);
        cachemutex.unlock();
        if (it != rnlijcache.end()) {
          const auto& r = it->second;
          return r;
        }
        Tensor<T, 1> R(4*K);
        Tensor<T, 2> rnlij(K, K);
        auto R_view = R.current_view();

        const auto& rnlp1 = get_rnlp(n, lx-1);
        const auto& rnlp2 = get_rnlp(n, lx);
        auto rnlp1_view = rnlp1.current_view();
        auto rnlp2_view = rnlp2.current_view();

        std::cout << "For (" << n << ", " << lx << ") rnlp1 is \n" << rnlp1_view << "\n and rnlp2 is \n" << rnlp2_view << std::endl;
        // std::cout << "Function call for rnlp1 results in \n" << rnlp1_view << "\n and rnlp2 results in \n" << rnlp2_view << std::endl;
        std::array<Slice,1> slice1 = {Slice(0, 2*K)};
        R_view(slice1) = rnlp1_view(slice1);
        // std::cout << "After copying rnlp1 to R_view, R_view is \n" << R_view << std::endl;
        std::array<Slice,1> slice2 = {Slice(2*K, 4*K)};
        R_view(slice2) = rnlp2_view(slice1);
        // std::cout << "After copying rnlp2 to R_view, R_view is \n" << R_view << std::endl;

        T scale = std::pow(T(0.5), T(0.5*n));
        R_view *= scale;
        auto rnlij_view = rnlij.current_view();
        rnlij_view = 0.0;
        // std::cout << "***MRA: for (" << n << ", " << lx << ") Before inner, R_view: \n" << R_view << std::endl; //" \n and c_view: \n" << c.current_view() << std::endl;
        detail::inner(c.current_view(), R_view, rnlij_view);
        // std::cout << "***MRA: for (" << n << ", " << lx << ") After inner, rnlij_view: \n" << rnlij_view << std::endl;

        cachemutex.lock();
        if (rnlijcache.find(key) == rnlijcache.end()) {
          rnlijcache.emplace(key, std::move(rnlij));
        }

        it = rnlijcache.find(key);
        cachemutex.unlock();
        const auto& r = it->second;
        return r;
      }

      std::shared_ptr<const ConvolutionData<T>> make_nonstandard (const Level n, const Translation lx) const {
        mra::Key<NDIM> key(n, std::array<Translation, NDIM>({lx}));
        cachemutex.lock();
        auto it = nscache.find(key);
        cachemutex.unlock();
        if (it != nscache.end()) {
          const auto& r = it->second;
          return r;
        }

        Tensor<T, 2> tmp(2*K, 2*K);
        const Tensor<T, 2>& rm = make_rnlij(n+1, 2*lx-1);
        const Tensor<T, 2>& r0 = make_rnlij(n+1, 2*lx);
        const Tensor<T, 2>& rp = make_rnlij(n+1, 2*lx+1);

        auto tmp_view = tmp.current_view();
        auto rm_view = rm.current_view();
        auto r0_view = r0.current_view();
        auto rp_view = rp.current_view();

        std::array<Slice,2> slice = {Slice(0, K), Slice(0, K)};
        tmp_view(slice) = r0_view;
        slice = {Slice(0, K), Slice(K, 2*K)};
        tmp_view(slice) = rm_view;
        slice = {Slice(K, 2*K), Slice(0, K)};
        tmp_view(slice) = rp_view;
        slice = {Slice(K, 2*K), Slice(K, 2*K)};
        tmp_view(slice) = r0_view;

        const auto& hgT = functiondata.get_hgT();
        auto hgT_view = hgT.current_view();
        Tensor<T, 2> R(2*K, 2*K), work(2*K, 2*K);
        auto R_view = R.current_view();
        transform(tmp_view, hgT_view, R_view, work.data());
        Tensor<T, 2> S(K, K);
        auto S_view = S.current_view();
        slice = {Slice(0, K), Slice(0, K)};
        S_view(slice) = R_view(slice);

        // transpose
        for (size_type i = 0; i < 2*K; ++i) {
          for (size_type j = i+1; j < 2*K; ++j) {
            std::swap(R_view(i, j), R_view(j, i));
          }
        }

        for (size_type i = 0; i < K; ++i) {
          for (size_type j = i+1; j < K; ++j) {
            std::swap(S_view(i, j), S_view(j, i));
          }
        }
        auto obj = ConvolutionData<T>();
        obj.R = std::move(R);
        obj.S = std::move(S);
        obj.normR = normf(obj.R.current_view());
        obj.normS = normf(obj.S.current_view());

        cachemutex.lock();
        if (nscache.find(key) == nscache.end()) {
          auto obj_ptr = std::make_shared<const ConvolutionData<T>>(std::move(obj));
          nscache.emplace(key, std::move(obj_ptr));
        }
        it = nscache.find(key);
        cachemutex.unlock();
        const auto& r = it->second;
        return r;
      }
    };

  template <typename T, Dimension NDIM>
  class ConvolutionOperator {

  private:
    using op_type = const OperatorData<T, NDIM>;
    size_type K;
    // size_type seprank;
    Convolution<T, NDIM>& conv;                             // convolution object
    mutable std::map<Key<NDIM>, std::shared_ptr<op_type>> opdata;   // map for storing operator data
    mutable std::mutex cachemutex;                          // mutex for thread safety

    T norm_ns(Level n, std::array<std::shared_ptr<const ConvolutionData<T>>, NDIM>& ns) const {
      T norm = 1.0, sum = 0.0;

      for (size_type d = 0; d < NDIM; ++d) {
        Tensor<T, 2> ns_r(2*K, 2*K);
        const auto& ref_view = ns[d]->R.current_view();
        const auto& ns_sview = ns[d]->S.current_view();
        auto ns_rview = ns_r.current_view();
        for (size_type i = 0; i < 2*K; ++i) {
          for (size_type j = 0; j < 2*K; ++j) {
            if (i<K && j<K) ns_rview(i, j) = 0.0;
            else ns_rview(i, j) = ref_view(i, j);
          }
        }
        T rnorm = normf(ns_rview);
        T snorm = normf(ns_sview);
        T aa = std::min(rnorm, snorm);
        T bb = std::max(rnorm, snorm);
        norm *= aa;
        if (bb > 0.0) sum += aa/bb;
      }
      if (n) norm *= sum;
      return norm;
    }

  public:

    ConvolutionOperator(size_type K, size_type npt, Convolution<T, NDIM>& conv)
    : K(K), /* seprank(npt): TODO ,*/ conv(conv) {}

    ConvolutionOperator(ConvolutionOperator&&) = default;
    ConvolutionOperator(const ConvolutionOperator&) = delete;
    ConvolutionOperator& operator=(ConvolutionOperator&&) = default;
    ConvolutionOperator& operator=(const ConvolutionOperator&) = delete;

    std::shared_ptr<const OperatorData<T, NDIM>> get_op(const Key<NDIM>& key) const {
      cachemutex.lock();
      auto it = opdata.find(key);
      cachemutex.unlock();
      if (it != opdata.end()) {
        return it->second;
      }

      OperatorData<T, NDIM> data;
      for (int i = 0; i < NDIM; ++i) data.ops[i] = conv.make_nonstandard(key.level(), key.translation()[i]);
      data.norm = norm_ns(key.level(), data.ops);

      cachemutex.lock();
      if (opdata.find(key) == opdata.end()) {
        auto data_ptr = std::make_shared<const OperatorData<T, NDIM>>(std::move(data));
        opdata.emplace(key, std::move(data_ptr));
      }
      it = opdata.find(key);
      cachemutex.unlock();
      auto& r = it->second;
      return r;
    }
  };

} // namespace mra

#endif // MRA_CONVOLUTIONDATA_H
