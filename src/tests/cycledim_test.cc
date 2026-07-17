#include <cassert>
#include <ttg.h>
#include "mra/tensor/cycledim.h"
#include "mra/tensor/tensor.h"

int main(int argc, char **argv) {

  ttg::initialize(argc, argv);

  constexpr const mra::size_type K = 10;

  mra::DenseTensor<double, 2> nodeA(K), nodeB(K), nodeC(K);
  mra::DenseTensorView<double, 2> nodeAv = nodeA.current_view();
  mra::DenseTensorView<double, 2> nodeBv = nodeB.current_view();
  mra::DenseTensorView<double, 2> nodeCv = nodeC.current_view();

  for (int i=0; i<nodeA.size(); ++i){
    nodeAv[i] = i;
    nodeBv[i] = 100.0;
    nodeCv[i] = 100.0;
  }

  mra::detail::cycledim(nodeAv, nodeBv, 1, 0, 2);
  mra::detail::cycledim(nodeBv, nodeCv, 1, 0, 2);

  for (int i=0; i<nodeA.size(); ++i){
    assert(nodeCv[i] == nodeAv[i]);
  }

  ttg::finalize();
}
