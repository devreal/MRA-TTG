#include <ttg.h>

#include "mra/mra.h"
#include "mra/misc/misc.h"
#include "mra/misc/types.h"
#include "mra/misc/convolutiondata.h"
#include <madness/mra/mra.h>
#include "mra/tensor/tensor.h"
#include <madness/world/world.h>
#include <madness/mra/twoscale.h>

void test_coeffs(int argc, char** argv) {
  mra::Convolution<double, 3> conv(4, 10, 10.0, 10.0);
  const mra::Tensor<double, 2>& rnlij = conv.make_rnlij(2, 1);

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world, argc, argv);

  madness::GaussianConvolution1D<double> conv1d(4, 10, 10, 0, 0);
  madness::Tensor<double> rnlij_mad = conv1d.rnlij(2, 1);

  // Check rnlij
  for (int i = 0; i < rnlij_mad.size(); ++i) {
    for (int j = 0; j < rnlij_mad.size(); ++j) {
        assert(std::abs(rnlij(i, j) - rnlij_mad(i, j)) < 1e-10);
    }
  }

  world.gop.fence();
}

int main(int argc, char **argv){

  ttg::initialize(argc, argv, 4);
  mra::GLinitialize();

  #if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
  #endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);

  test_coeffs(argc, argv);

  madness::finalize();
  ttg::execute();
  ttg::fence();
  ttg::finalize();
  return 0;
}
