#include "potentials/inversepower_potential.hpp"
#include "potentials/biased_kicker.hpp"
#include "utils/noise.hpp"
#include <algorithm>
#include <iostream>

std::vector<double> d,d_ref,g;

double dist_vec(std::vector<double>& v1, std::vector<double>& v2){
    if (v2.size() != v1.size()){
        throw std::runtime_error("v1 and v2 must have the same size");
    }
    double sum = 0.0;
    for (size_t i=0;i<v1.size();i++){
        sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }
    return sqrt(sum);
}

double norm(std::vector<double>& v1){
    double sum = 0.0;
    for (size_t i=0;i<v1.size();i++){
        sum += v1[i]*v1[i];
    }
    return sqrt(sum);

}
void uniformly_fill(std::vector<double>& x, double rng){
    UniformNoise rnd(0.0,rng);
    for (size_t i = 0 ; i < x.size(); i++){
        x[i] = rnd.rand();
    }
}

void print_vec(std::vector<double>& v1){
    std::cout<<"{"<<v1[0];
    for (size_t i=1;i<v1.size();i++){
        std::cout<<","<<v1[i];   
    }
    std::cout<<"}";
}

void inversepower2_pbc(double mpow,double a,
                    std::vector<double> x1, std::vector<double> x2,
                    std::vector<double>& g1, std::vector<double>& g2,
                    double r1, double r2,
                    std::vector<double>& boxv){
    size_t ndim = boxv.size();
    // deal with the pbc
    for (size_t k = 0; k < ndim; k++){
        while (x1[k] - x2[k] > 0.5*boxv[k]){
            x1[k] -= boxv[k];
        }

        while (x1[k] - x2[k] < -0.5*boxv[k]){
            x1[k] += boxv[k];
        }
    }
    double r = dist_vec(x1,x2);
    if (r >= r1+r2){
        for (size_t k = 0; k < ndim; k++){
            g1[k] = 0.0;
            g2[k] = 0.0;
        }
    }
    else{
        double v = a*std::pow(1 - r/(r1+r2),mpow - 1.0)/(r1+r2);
        for (size_t k = 0; k < ndim; k++){
            g1[k] = v*(x2[k] - x1[k])/r;
            g2[k] = v*(x1[k] - x2[k])/r;
        }
    }
}
/*
int main(){
    std::vector<double> x1 = {1.0,0.0},x2={0.0,1.0};
    std::vector<double> g1 = {0.0,0.0},g2 = {0.0,0.0};
    std::vector<double> boxv = {8.0,8.0,8.0};
    double a = 4*sqrt(2), mpow = 2.0, r1 = sqrt(2), r2 = sqrt(2);
    inversepower2_pbc(mpow,a,x1,x2,g1,g2,r1,r2,boxv);
    print_vec(g1);
    std::cout<<std::endl;
    print_vec(g2);


    return 0;
}
*/

int main(){
    std::vector<double> x, radii, disp, boxv = {1.0,1.0,1.0};    
    size_t natoms = 40, ndim = 3;
    x.resize(natoms*ndim);
    d.resize(natoms*ndim);
    d_ref.resize(natoms*ndim);
    g.resize(natoms*ndim);
    radii.resize(natoms);
    uniformly_fill(x,1.0);
    //uniformly_fill(radii,0.4);
    for (auto & r:radii){
        r = 0.3;
    }
    print_vec(radii);
    //double a = 200.0, mpow = 2.0, lr = -2.0, p = 0.5;
    //ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<3> pot(mpow,a,lr,p,radii,boxv);
    double a = 4.0*sqrt(2), mpow = 2, lr = 1.0, p = 0.8; 
    ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<3> pot(mpow, a, lr, p, radii,boxv);
    ha::InversePowerPeriodicCellLists<3> pot_ref(mpow, a,radii,boxv);
    pot.get_batch_energy_gradient(x,d);
    pot_ref.get_energy_gradient(x,g);
    std::vector<size_t> batch_particles;
    pot.get_batch_particles(batch_particles);
    std::cout<<"printing pairs"<<std::endl; 
    for (auto i:batch_particles){
        std::cout<<"("<<i<<"),";
        std::cout<<" ,";
        for (size_t k = 0; k < ndim; k++){
            d_ref[i*ndim + k] = g[i*ndim + k];
        }
    }
    std::cout<<" here is the comparison" <<std::endl;
    std::cout<<" the relative error is :" << dist_vec(d,d_ref)/norm(d_ref)<<std::endl;
    print_vec(d);
    std::cout<<std::endl;
    print_vec(d_ref);
    std::cout<<std::endl;
    return 0;
}