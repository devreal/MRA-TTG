#include <ttg.h>

#include "mra/mra.h"
#include "mra/misc/misc.h"
#include "mra/misc/types.h"
#include "mra/misc/convolutiondata.h"
#include <madness/mra/mra.h>
#include "mra/tensor/tensor.h"
#include <madness/world/world.h>
#include <madness/mra/twoscale.h>
#include <madness/mra/convolution1d.h>

void test_coeffs(int argc, char** argv) {
  constexpr int K = 4; // wavelet order
  constexpr int npt = 10; // number of quadrature points
  constexpr double coeff = 10.0; // coefficient for the Gaussian
  constexpr double expnt = 10.0; // exponent for the Gaussian
  mra::FunctionData<double, 3> functiondata(K);



  mra::Convolution<double, 3> conv(K, npt, coeff, expnt, functiondata);
  const mra::Tensor<double, 2>& rnlij = conv.make_rnlij(2, 1);
  auto rnlij_view = rnlij.current_view();
  const mra::ConvolutionData<double>& cd = conv.make_nonstandard(2, 1);
  auto cdR_view = cd.R.current_view();
  auto cdS_view = cd.S.current_view();

  mra::ConvolutionOperator<double, 3> op(K, npt, coeff, expnt, functiondata);
  const mra::OperatorData<double, 3>& op_data = op.get_op(mra::Key<3>(2, {1, 1, 1}));

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world, argc, argv);

  madness::GaussianConvolution1D<double> conv1d(K, coeff, expnt, 0, false);
  madness::Tensor<double> rnlij_mad = conv1d.rnlij(2, 1);
  const madness::ConvolutionData1D<double>* cd_mad = conv1d.nonstandard(2, 1);

  std::cout << "rnlij_mad: " << rnlij_mad << std::endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "rnlij MRA: " << rnlij << std::endl;
  // Check rnlij
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
        assert(std::abs(rnlij_view(i, j) - rnlij_mad(i, j)) < 1e-10);
    }
  }
  // Check ConvolutionData1D
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
        assert(std::abs(cdR_view(i, j) - cd_mad->R(i, j)) < 1e-10);
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
        assert(std::abs(cdS_view(i, j) - cd_mad->T(i, j)) < 1e-10);
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
