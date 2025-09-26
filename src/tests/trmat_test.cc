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
  constexpr int K = 8; // wavelet order
  constexpr int npt = 2*K; // number of quadrature points
  constexpr double expnt = 1500.0; // exponent for the Gaussian
  static double coeff = std::pow(2.0*expnt/std::numbers::pi, 0.25*3);; // coefficient for the Gaussian
  mra::FunctionData<double, 3> functiondata(K);



  mra::Convolution<double, 3> conv(K, npt, coeff, expnt, functiondata);
  // const mra::Tensor<double, 1>& rnlp = conv.make_rnlp(2, 1);
  const mra::Tensor<double, 2>& rnlij = conv.make_rnlij(2, 1);
  auto rnlij_view = rnlij.current_view();
  // auto rnlp_view = rnlp.current_view();

  // mra::ConvolutionOperator<double, 3> op(K, npt, coeff, expnt, functiondata);
  mra::ConvolutionOperator<double, 3> op(K, npt, conv);
  std::shared_ptr<const mra::OperatorData<double, 3>> op_data = op.get_op(mra::Key<3>(1, {0, 0, 0}));

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world, argc, argv);

  madness::GaussianConvolution1D<double> conv1d(K, coeff, expnt, 0, false);
  const madness::Tensor<double>& rnlp_mad = conv1d.rnlp(2, 1);
  madness::Tensor<double> rnlij_mad = conv1d.rnlij(2, 1);
  // const madness::ConvolutionData1D<double>* cd_mad = conv1d.nonstandard(1, 0);

  // std::cout << "opdata norm: " << op_data->norm << std::endl;

  // for (int i = 0; i < op_data->ops.size(); ++i) {
  //   std::cout << "MRA op[" << i << "].R \n" << op_data->ops[i]->R.current_view() << std::endl;
  //   std::cout << "MRA op[" << i << "].S: \n" << op_data->ops[i]->S.current_view() << std::endl;
  //   std::cout << "MAD op[" << i << "].R: \n" << cd_mad->R << std::endl;
  //   std::cout << "MAD op[" << i << "].T: \n" << cd_mad->T << std::endl;
  // }

  std::cout << "\n rnlij_mad: \n" << rnlij_mad << std::endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "\n rnlij MRA: \n" << rnlij << std::endl;
  // Check rnlij
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
        assert(std::abs(rnlij_view(i, j) - rnlij_mad(i, j)) < 1e-10);
    }
  }

  // std::cout << "\n rnlp_mad: \n" << rnlp_mad << std::endl;
  // std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  // std::cout << "\n rnlp MRA: \n" << rnlp << std::endl;
  // // Check rnlp
  // for (int i = 0; i < 2*K; ++i) {
  //   assert(std::abs(rnlp_view(i) - rnlp_mad(i)) < 1e-10);
  // }

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
