#include <ttg.h>


#include "mra/misc/misc.h"
#include "mra/misc/types.h"
#include "mra/misc/convolutiondata.h"
#include <madness/mra/mra.h>
#include "mra/tensor/tensor.h"
#include <madness/world/world.h>
#include <madness/mra/twoscale.h>

void test_coeffs(int argc, char** argv) {
  mra::ConvolutionData<double, 3> conv_data(4, 3, 10, 2, 10.0);
  const mra::Tensor<double, 3>& mat = conv_data.get_autocorrcoef();

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world, argc, argv);

  // Example usage of madness::autoc
  madness::Tensor<double> c;
  bool success = madness::autoc(4, &c);
  if (!success) {
    std::cerr << "Failed to compute autocorrelation coefficients." << std::endl;
    return;
  }

  for (int i = 0; i < c.size(); ++i) {
    for (int j = 0; j < c.size(); ++j) {
      for (int k = 0; k < c.size(); ++k) {
        assert(std::abs(c(i, j, k) - mat(i, j, k)) < 1e-10);
      }
    }
  }
  world.gop.fence();
}

int main(int argc, char **argv){

  ttg::initialize(argc, argv, 4);

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
