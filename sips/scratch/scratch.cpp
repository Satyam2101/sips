#include "algorithms/bro_algorithms.hpp"
#include "algorithms/sgd_algorithms.hpp"
#include "algorithms/stochastic_algorithms.hpp"
#include "utils/noise.hpp"
#include <algorithm>
#include <iostream>

int main(){
    std::vector<double> x = {-1.0,0.0, 0.0,1.0, 1.0,0.0}, radii = {sqrt(2),sqrt(2),sqrt(2)};
    std::vector<double> boxv = {5.0,5.0};    
    //pariticlewise_bro_clist<2> a(0.1,radii,boxv,x);
    inversepower_nonreciprocal_pairwise_sd_clist<2> a(2.0,1.0,-0.1,0.1,radii,boxv,x);
    double e_f = a.avg_energy_flucuation(0.05,4000000);
    ha::InversePowerPeriodicCellLists<2> pot(2.0,1.0,radii,boxv);
    std::vector<double> hessian;
    hessian.resize(36);
    pot.get_hessian(x,hessian);
    double laplacian = 0.0;
    for (size_t i = 0; i < x.size(); i++){
        laplacian += hessian[i*x.size() + i];
    }
    std::cout<<"energy flucuation = "<< e_f*3<<std::endl;
    std::cout<<"0.5*lapcian = "<<0.5*laplacian<<std::endl;
    return 0;
}