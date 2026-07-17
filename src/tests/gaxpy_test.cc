#include <cassert>
#include <ttg.h>
#include "mra/kernels/gaxpy.h"
#include "mra/tensor/tensor.h"

int main(int argc, char **argv) {

  ttg::initialize(argc, argv);

  double scalarA = 1.0, scalarB = -1.0;
  constexpr const mra::size_type K = 3;

  mra::DenseTensor<double, 2> nodeA(K), nodeB(K), nodeC(K);
  mra::DenseTensorView<double, 2> nodeAv = nodeA.current_view();
  mra::DenseTensorView<double, 2> nodeBv = nodeB.current_view();
  mra::DenseTensorView<double, 2> nodeCv = nodeC.current_view();
  for (int i=0; i<nodeA.size(); ++i){
    nodeAv[i] = i;
    nodeBv[i] = i;
    nodeCv[i] = 100.0;
  }

  mra::detail::gaxpy_kernel_impl<double, 2>(nodeAv, nodeBv, nodeCv, scalarA, scalarB);

  for (int i=0; i<nodeA.size(); ++i){
    assert(nodeCv[i] == 0.0);
  }

  ttg::execute();
  ttg::fence();

  ttg::finalize();
}
