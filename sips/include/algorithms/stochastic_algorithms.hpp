#ifndef STOCHSTIC_ALGORITHMS_HPP
#define STOCHSTIC_ALGORITHMS_HPP
#include <vector>
#include <string>
#include <fstream>
#include "utils/json.hpp"
#include "hyperalg/vecN.hpp"
#include "base_algorithm.hpp"
#include "potentials/inversepower_potential.hpp"

template<size_t ndim>
class inversepower_reciprocal_pairwise_sd_clist: public base_potential_algorithm<ha::InversePowerPeriodicReciprocalPairwiseNoiseCellLists<ndim> >{
public: 
    inversepower_reciprocal_pairwise_sd_clist(double pow, double eps, double alpha, double D0,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicReciprocalPairwiseNoiseCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicReciprocalPairwiseNoiseCellLists<ndim> >(pow,eps,alpha,D0,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
};

template<size_t ndim>
class inversepower_nonreciprocal_pairwise_sd_clist: public base_potential_algorithm<ha::InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists<ndim> >{
public: 
    inversepower_nonreciprocal_pairwise_sd_clist(double pow, double eps, double alpha, double D0,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists<ndim> >(pow,eps,alpha,D0,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
    virtual anneal(size_t n_steps){
        // annealing process: relax the system with zero noise and small learning rate
        // for n_steps
        double alpha = this->m_potential->get_alpha();
        double D0 = this->m_potential->get_D0();
        this->m_potential->set_alpha(0.1*lr);
        this->m_potential->set_D0(0.0); // D0=0.0 --> noiseless
        for (size_t i=0;i<n_steps;i++){
            one_step();
        }
        // recover the state before annealing
        this->m_potential->set_alpha(alpha);
        this->m_potential->set_D0(D0);
    }
};

template<size_t ndim>
class inversepower_particlewise_sd_clist: public base_potential_algorithm<ha::InversePowerPeriodicParticlewiseNoiseCellLists<ndim> >{
public: 
    inversepower_particlewise_sd_clist(double pow, double eps, double alpha, double D0,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicParticlewiseNoiseCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicParticlewiseNoiseCellLists<ndim> >(pow,eps,alpha,D0,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
    virtual anneal(size_t n_steps){
        // annealing process: relax the system with zero noise and small learning rate
        // for n_steps
        double alpha = this->m_potential->get_alpha();
        double D0 = this->m_potential->get_D0();
        this->m_potential->set_alpha(0.1*alpha);
        this->m_potential->set_D0(0.0); // D0=0.0 --> noiseless
        for (size_t i=0;i<n_steps;i++){
            one_step();
        }
        // recover the state before annealing
        this->m_potential->set_alpha(alpha);
        this->m_potential->set_D0(D0);
    }
};



#endif
