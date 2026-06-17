#ifndef HAVE_VMRA_CONVOLUTION_H
#define HAVE_VMRA_CONVOLUTION_H

#include "madness/mra/mra.h"
#include "mra/mra.h"

namespace mra::vmra {


  /**
   * Apply the SeparatedConvolution to a vector of MADNESS functions, returning the result as a vector of MADNESS functions.
   */

  template<typename T, std::size_t NDIM>
  std::vector<madness::Function<T, NDIM>>
  apply_convolution(const madness::SeparatedConvolution<thread_local, 3>& mad_conv,
                    const std::vector<madness::Function<T, NDIM>>& in, bool print_dot = false) {
    std::size_t N = in.size();
    size_type K = in.front().get_k();
    ttg::Edge<mra::Key<NDIM>, void> control;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> load_to_compress;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> compress_to_convolution;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsCompressedNode<T, NDIM>> convolution_to_reconstruct;
    ttg::Edge<mra::Key<NDIM>, mra::FunctionsReconstructedNode<T, NDIM>> reconstruct_to_store;
    // placeholder, we do not project the functions here
    auto gaussians = make_functionset<mra::Gaussian<T, NDIM>>(N);

    // TODO: cache the functiondata between invocations
    mra::FunctionData<T, NDIM> functiondata;

    std::vector<madness::Function<T, NDIM>> result(N);


    auto op = mra::GaussianConvolutionOperator<T, NDIM>{mad_conv};

    auto start            = make_start(gaussians, control);
    auto load_tt          = mra::make_mra_load(in, load_to_compress, "load_vmra");
    auto compress_ns_tt   = mra::make_compress(gaussians, K, true, functiondata,
                                               load_to_compress, compress_to_convolution, "compress_ns_vmra");
    auto conv_tt          = mra::make_convolution(gaussians, K, op,
                                                  compress_to_convolution, convolution_to_reconstruct,
                                                  op, "convolution");
    auto reconstruct_tt   = mra::make_reconstruct(gaussians, K, functiondata, convolution_to_reconstruct,
                                                  reconstruct_to_store, "reconstruct_vmra");
    auto store_tt         = mra::make_vmra_store(result, reconstruct_to_store, "store_vmra");


    /**
     * Check we can execute and then start the execution.
     */
    auto connected = make_graph_executable(start.get());
    assert(connected);

    if (print_dot) {
      std::cout << "==== begin dot ====\n";
      std::cout << ttg::Dot(true)(start.get()) << std::endl;
      std::cout << "====  end dot  ====\n";
    }

    if (ttg::default_execution_context().rank() == 0) {
        // This kicks off the entire computation
        start->invoke();
    }
    ttg::execute();
    ttg::fence();

    return result;
  }

} // namespace mra::vmra


#endif // HAVE_VMRA_CONVOLUTION_H