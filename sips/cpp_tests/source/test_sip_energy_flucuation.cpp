#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "potentials/biased_kicker.hpp"
#include "algorithms/stochastic_algorithms.hpp"

// test the enery flucuation function
class SIPSEnergyFlucuation :  public ::testing::Test
{
public:
    std::vector<double> x = {-1.0,0.0, 0.0,1.0, 1.0,0.0}, radii = {sqrt(2),sqrt(2),sqrt(2)};
    std::vector<double> boxv = {5.0,5.0};    
    virtual void SetUp(){ 	
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
    }

};

TEST_F(SIPSEnergyFlucuation, EnergyFlucuation){
    inversepower_nonreciprocal_pairwise_sd_clist<2> a(2.0,1.0,-0.1,0.1,radii,boxv,x);
    ha::InversePowerPeriodicCellLists<2> pot(2.0,1.0,radii,boxv);
    std::vector<double> hessian;
    size_t natoms = 3;
    hessian.resize(36);
    pot.get_hessian(x,hessian);
    double e_f = a.avg_energy_flucuation(0.04,4000000);
    double laplacian = 0.0;
    for (size_t i = 0; i < x.size(); i++){
        laplacian += hessian[i*x.size() + i];
    }
    double rerr = abs(e_f*natoms - 0.5*laplacian)/(0.5*laplacian); 
    EXPECT_LT(rerr,0.2) << "the energy flucuation = " << e_f*natoms << " 1/2*laplacian = " <<0.5*laplacian;
}
