#ifndef MRA_FUNCTIONDATA_H
#define MRA_FUNCTIONDATA_H

#include "mra/misc/gl.h"
#include "mra/misc/types.h"
#include "mra/misc/domain.h"
#include "mra/misc/twoscale.h"
#include "mra/tensor/tensor.h"

namespace mra {

    /// Convenient co-location of frequently used data
    template <typename T, Dimension NDIM>
    class FunctionData {

    public:
        enum class DerivOp {
            RM = 0, R0 = 1, RP = 2,             ///< Blocks of the derivative operator
            RMT = 3, R0T = 4, RPT = 5,          ///< Blocks of the derivative operator, transposed
            LEFT_RM = 6, LEFT_R0 = 7,           ///< Blocks of the derivative for the left boundary
            LEFT_RMT = 8, LEFT_R0T = 9,         ///< Blocks of the derivative for the left boundary, transposed
            RIGHT_R0 = 10, RIGHT_RP = 11,       ///< Blocks of the derivative for the right boundary
            RIGHT_R0T = 12, RIGHT_RPT = 13,     ///< Blocks of the derivative for the right boundary, transposed
            COUNT = 14                          ///< Number of blocks
            // BV_LEFT = 14, BV_RIGHT = 15,        ///< Blocks of the derivative operator for the boundary contribution --- not used
        };

        enum BCType {BC_ZERO = 0, BC_PERIODIC = 1, BC_FREE = 2, BC_DIRICHLET = 3, BC_ZERONEUMANN = 4, BC_NEUMANN = 5};
        // NOTE: In the current v, we only consider Dirichlet boundary conditions.

    private:
        size_type K;
        Tensor<T,2> phi;            // phi(mu,i) = phi(x[mu],i) --- value of scaling functions at quadrature points on level 0
        Tensor<T,2> phiT;           // transpose of phi
        Tensor<T,2> phibar;         // phibar(mu,i) = w[mu]*phi(x[mu],i)
        Tensor<T,2> HG;             // Two scale filter applied from left to scaling function coeffs
        Tensor<T,2> HGT;            // Two scale filter applied from right to scaling function coeffs
        Tensor<T,1> quad_x;         // Quadrature points on level 0
        std::unique_ptr<T[]> x, w;  // Quadrature points and weights on level 0
        Tensor<T, 2+1> operators;   // Derivative operators
        BCType bc_left, bc_right;
        // TODO: add quad_x and corresponding functions

        void make_deriv_op(){
            bc_left = BCType::BC_ZERO;
            bc_right = BCType::BC_ZERO;
            auto deriv_op_view = operators.current_view();

            // Operator blocks
            auto rm         = deriv_op_view(static_cast<int>(DerivOp::RM));
            auto r0         = deriv_op_view(static_cast<int>(DerivOp::R0));
            auto rp         = deriv_op_view(static_cast<int>(DerivOp::RP));
            auto rmt        = deriv_op_view(static_cast<int>(DerivOp::RMT));
            auto r0t        = deriv_op_view(static_cast<int>(DerivOp::R0T));
            auto rpt        = deriv_op_view(static_cast<int>(DerivOp::RPT));
            auto left_rm    = deriv_op_view(static_cast<int>(DerivOp::LEFT_RM));
            auto left_r0    = deriv_op_view(static_cast<int>(DerivOp::LEFT_R0));
            auto left_rmt   = deriv_op_view(static_cast<int>(DerivOp::LEFT_RMT));
            auto left_r0t   = deriv_op_view(static_cast<int>(DerivOp::LEFT_R0T));
            auto right_r0   = deriv_op_view(static_cast<int>(DerivOp::RIGHT_R0));
            auto right_rp   = deriv_op_view(static_cast<int>(DerivOp::RIGHT_RP));
            auto right_r0t  = deriv_op_view(static_cast<int>(DerivOp::RIGHT_R0T));
            auto right_rpt  = deriv_op_view(static_cast<int>(DerivOp::RIGHT_RPT));
            // auto bv_left    = deriv_op_view(DerivOp::BV_LEFT);
            // auto bv_right   = deriv_op_view(DerivOp::BV_RIGHT);

            double kphase = -1.0;
            if (K%2 == 0) kphase = 1.0;
            double iphase = 1.0;
            for (int i=0; i<K; ++i) {
                double jphase = 1.0;
                for (int j=0; j<K; ++j) {
                    double gammaij = sqrt(double((2*i+1)*(2*j+1)));
                    double Kij;
                    if (((i-j)>0) && (((i-j)%2)==1))
                        Kij = 2.0;
                    else
                        Kij = 0.0;

                    r0(i,j) = 0.5*(1.0 - iphase*jphase - 2.0*Kij)*gammaij;
                    rm(i,j) = 0.5*jphase*gammaij;
                    rp(i,j) =-0.5*iphase*gammaij;

                    // Constraints on the derivative
                    if (bc_left == BC_ZERONEUMANN || bc_left == BC_NEUMANN) {
                        left_rm(i, j)= jphase*gammaij*0.5*(1.0 + iphase*kphase/K);

                        double phi_tmpj_left = 0;

                        for (int l=0; l<K; ++l) {
                            double gammalj = sqrt(double((2*l+1)*(2*j+1)));
                            double Klj;

                            if (((l-j)>0) && (((l-j)%2)==1))  Klj = 2.0;
                            else   Klj = 0.0;

                            phi_tmpj_left += sqrt(double(2*l+1))*Klj*gammalj;
                        }
                        phi_tmpj_left = -jphase*phi_tmpj_left;
                        left_r0(i,j) = (0.5*(1.0 + iphase*kphase/K) - Kij)*gammaij + iphase*sqrt(double(2*i+1))*phi_tmpj_left/pow(K,2.);
                    }
                    else if (bc_left == BC_ZERO || bc_left == BC_DIRICHLET || bc_left == BC_FREE) {
                        left_rm(i,j) = rm(i,j);

                        // B.C. with a function
                        if (bc_left == BC_ZERO || bc_left == BC_DIRICHLET)
                            left_r0(i,j) = (0.5 - Kij)*gammaij;

                        // No B.C.
                        else if (bc_left == BC_FREE)
                            left_r0(i,j) = (0.5 - iphase*jphase - Kij)*gammaij;
                    }

                    // Constraints on the derivative
                    if (bc_right == BC_ZERONEUMANN || bc_right == BC_NEUMANN) {
                        right_rp(i,j) = -0.5*(iphase + kphase / K)*gammaij;

                        double phi_tmpj_right = 0;
                        for (int l=0; l<K; ++l) {
                            double gammalj = sqrt(double((2*l+1)*(2*j+1)));
                            double Klj;
                            if (((l-j)>0) && (((l-j)%2)==1))  Klj = 2.0;
                            else   Klj = 0.0;
                            phi_tmpj_right += sqrt(double(2*l+1))*Klj*gammalj;
                        }
                        right_rp(i,j) = -(0.5*jphase*(iphase+ kphase/K) + Kij)*gammaij + sqrt(double(2*i+1))*phi_tmpj_right/pow(K,2.);
                    }
                    else if (bc_right == BC_ZERO || bc_right == BC_FREE || bc_right == BC_DIRICHLET) {
                        right_rp(i,j) = rp(i,j);

                        // Zero BC
                        if (bc_right == BC_ZERO || bc_right == BC_DIRICHLET)
                            right_r0(i,j) = -(0.5*iphase*jphase + Kij)*gammaij;

                        // No BC
                        else if (bc_right == BC_FREE)
                            right_r0(i,j) = (1.0 - 0.5*iphase*jphase - Kij)*gammaij;
                    }

                    jphase = -jphase;
                }
                iphase = -iphase;
            }

            for (size_type i=0; i< K; ++i){
                for (size_type j=0; j< K; ++j){
                    rmt(i,j) = rm(j,i);
                    r0t(i,j) = r0(j,i);
                    rpt(i,j) = rp(j,i);

                    left_rmt(i,j) = left_rm(j,i);
                    left_r0t(i,j) = left_r0(j,i);

                    right_r0t(i,j) = right_r0(j,i);
                    right_rpt(i,j) = right_rp(j,i);
                }
            }


        }

        /// Set phi(mu,i) to be phi(x[mu],i)
        void make_phi() {
            /* retrieve x, w from constant memory */
            const T *x, *w;
            GLget(&x, &w, K);
            T* p = new T[K];
            auto phi_view = phi.current_view();
            for (size_type mu = 0; mu < K; ++mu) {
                legendre_scaling_functions(x[mu], K, &p[0]);
                for (size_type i = 0; i < K; ++i) {
                    phi_view(mu,i) = p[i];
                }
            }
            delete[] p;
        }

        /// Set phiT(mu,i) to be phiT(x[mu],i)
        void make_phiT() {
            /* retrieve x, w from constant memory */
            const T *x, *w;
            GLget(&x, &w, K);
            T* p = new T[K];
            auto phiT_view = phi.current_view();
            for (size_type mu = 0; mu < K; ++mu) {
                legendre_scaling_functions(x[mu], K, &p[0]);
                for (size_type i = 0; i < K; ++i) {
                    phiT_view(i, mu) = p[i];
                }
            }
            delete[] p;
        }

        /// Set phibar(mu,i) to be w[mu]*phi(x[mu],i)
        void make_phibar() {
            /* retrieve x, w from constant memory */
            const T *x, *w;
            GLget(&x, &w, K);
            T *p = new T[K];
            auto phibar_view = phibar.current_view();
            for (size_type mu = 0; mu < K; ++mu) {
                legendre_scaling_functions(x[mu], K, &p[0]);
                for (size_type i = 0; i < K; ++i) {
                    phibar_view(mu,i) = w[mu]*p[i];
                }
            }
            delete[] p;
            // FixedTensor<T,K,2> phi, r;
            // make_phi<T,K>(phi);
            // mTxmq(K, K, K, r.ptr(), phi.ptr(), phibar.ptr());
            // std::cout << r << std::endl; // should be identify matrix
        }

        void make_quad_x() {
            auto quad_x_view = quad_x.current_view();
            const T *x, *w;
            GLget(&x, &w, K);
            for (size_type i = 0; i < K; ++i) {
                quad_x_view(i) = x[i];
            }
        }

    public:

        FunctionData(size_type K)
        : K(K)
        , phi(K, K)
        , phiT(K, K)
        , phibar(K, K)
        , quad_x(K)
        , HG(2*K, 2*K)
        , HGT(2*K, 2*K)
        , operators(DerivOp::COUNT, K, K)
        {
            make_phi();
            make_phiT();
            make_phibar();
            make_quad_x();
            twoscale_get(K, HG.data());
            auto HG_view  = HG.current_view();
            auto HGT_view = HGT.current_view();
            for (size_type i = 0; i < 2*K; ++i) {
                for (size_type j = 0; j < 2*K; ++j) {
                    HGT_view(j,i) = HG_view(i,j);
                }
            }
            make_deriv_op();
        }

        FunctionData(FunctionData&&) = default;
        FunctionData(const FunctionData&) = delete;
        FunctionData& operator=(FunctionData&&) = default;
        FunctionData& operator=(const FunctionData&) = delete;

        const auto& get_phi() const {return phi;}
        const auto& get_phiT() const {return phiT;}
        const auto& get_phibar() const {return phibar;}
        const auto& get_quad_x() const {return quad_x;}
        const auto& get_hg() const {return HG;}
        const auto& get_hgT() const {return HGT;}
        const auto& get_operators() const {return operators;}
};
} // namespace mra

#endif // MRA_FUNCTIONDATA_H
