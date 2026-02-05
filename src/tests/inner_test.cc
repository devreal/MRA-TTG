#include <cassert>
#include <ttg.h>
#include "mra/ops/inner.h"
#include "mra/tensor/tensor.h"

#include <madness/tensor/tensor.h>
#include <madness/world/print.h>


int main(int argc, char **argv) {

  ttg::initialize(argc, argv);

	constexpr const mra::size_type K = 3;

  {
    bool passed = true;
    madness::DenseTensor<double> nA(K, K, K), nB(K, K, K);

    mra::DenseTensor<double, 3> nodeA(K), nodeB(K);
    mra::DenseTensor<double, 4> nodeC(K);
    mra::DenseTensorView<double, 3> nodeAv = nodeA.current_view();
    mra::DenseTensorView<double, 3> nodeBv = nodeB.current_view();
    mra::DenseTensorView<double, 4> nodeCv = nodeC.current_view();

    /* set up tensors */
    for (int i=0; i<K; ++i){
      for (int j=0; j<K; ++j){
        for (int k=0; k<K; ++k){
          nA(i,j,k) = 1+i+j+k;
          nB(i,j,k) = 1-i-j-k;
          nodeAv(i,j,k) = 1+i+j+k;
          nodeBv(i,j,k) = 1-i-j-k;
        }
      }
    }

    for (int axis = 0; axis < 3; ++axis) {
      /* zero out result */
      nodeCv = 0.0;
      mra::detail::inner(nodeAv, nodeBv, nodeCv, axis, axis);
      madness::Tensor<double> nC = madness::inner(nA, nB, axis, axis);
      for (int i=0; i<K; ++i) {
        for (int j=0; j<K; ++j) {
          for (int k=0; k<K; ++k) {
            for (int l=0; l<K; ++l) {
              if (nodeCv(i,j,k,l) != nC(i,j,k,l)) {
                std::cout << "inner result at [" << i << "," << j << "," << k << "," << l << "] does not match for axis " << axis << std::endl;
                std::cout << "MRA:\n" << nodeCv << std::endl;
                std::cout << "MAD:\n" << nC << std::endl;
                passed = false;
              }
              assert(nodeCv(i,j,k,l) == nC(i,j,k,l));
            }
          }
        }
      }
    }

    if (!passed) {
      std::cout << "inner test failed" << std::endl;
    } else {
      std::cout << "inner test passed" << std::endl;
    }
  }

	ttg::execute();
  ttg::fence();

  ttg::finalize();
}
