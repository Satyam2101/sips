#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "potentials/biased_kicker.hpp"
#include "potentials/inversepower_potential.hpp"



static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

/*
 * InversePower tests
 */

class SIPReciprocityTest :  public ::testing::Test
{
public:
    std::vector<double> x, radii, disp, boxv;
    size_t natoms = 100, ndim = 3;
    virtual void SetUp(){ 	
        #ifdef _OPENMP
        omp_set_num_threads(2);
        #endif
        boxv = std::vector<double>{1.0,1.0,1.0};
        x.resize(natoms*ndim);
        disp.resize(x.size());
        radii.resize(natoms);
        uniformly_fill(x,1.0);
        uniformly_fill(radii,0.1);
    }

    double displacement_sum(std::vector<double>& disp, size_t ndim){
        std::vector<double> disp_sum;
        disp_sum.resize(ndim);
        std::fill(disp_sum.begin(),disp_sum.end(),0.0);
        size_t natoms = disp.size()/ndim;
        for (size_t i = 0; i < natoms; i++){
            for (size_t k = 0; k< ndim; k++ ){
                disp_sum[k] += disp[i*ndim + k];
            }
        }
        double norm = 0.0;
        for (size_t k = 0; k< ndim; k++ ){
            norm += disp_sum[k]*disp_sum[k];
        }
        norm = sqrt(norm);
        return norm/natoms;
    }

    void uniformly_fill(std::vector<double>& x, double rng){
        UniformNoise rnd(0.0,rng);
        for (size_t i = 0 ; i < x.size(); i++){
            x[i] = rnd.rand();
        }
    }
};

TEST_F(SIPReciprocityTest, BRO){
    double kick_size = 0.1;
    ha::ReciprocalPairwiseBiasedKickerPeriodicCellLists<3> kicker(0.1,radii,boxv);
    kicker.get_kick(x,disp);
    double e = displacement_sum(disp,3);
    EXPECT_LT(e,1e-14) << "the sum of displacement is " << e;
}

TEST_F(SIPReciprocityTest, SGD){
    //test probabilistic pairwise SGD
    double pow = 2.0, eps = 1.0, lr = 1.0, prob = 0.5;
    ha::InversePowerPeriodicProbabilisticPairBatchCellLists<3> pot1(pow,eps,lr,prob,radii,boxv);
    pot1.get_batch_energy_gradient(x,disp);
    double e = displacement_sum(disp,3);
    EXPECT_LT(e,1e-14) << "the sum of displacement is " << e;
}

TEST_F(SIPReciprocityTest, Noise){
    //test reciprocal pairwise stochastic noise
    double pow = 2.0, eps = 1.0, alpha= 1.0, D0 = 0.25;
    ha::InversePowerPeriodicReciprocalPairwiseNoiseCellLists<3> pot2(pow,eps,alpha,D0,radii,boxv);
    pot2.get_stochastic_force(x,disp);
    double e = displacement_sum(disp,3);
    EXPECT_LT(e,1e-14) << "the sum of displacement is " << e;
}