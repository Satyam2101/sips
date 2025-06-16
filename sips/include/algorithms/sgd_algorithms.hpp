#ifndef SGD_ALGORITHMS_HPP
#define SDG_ALGORITHMS_HPP
#include "utils/json.hpp"
#include "base_algorithm.hpp"
#include "potentials/inversepower_potential.hpp"

template<size_t ndim>
class inversepower_probabilistic_pairwise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >{
public: 
    inversepower_probabilistic_pairwise_sgd_clist(double pow, double eps, double lr, double prob,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicProbabilisticPairBatchCellLists<ndim> >(pow,eps,lr,prob,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
    virtual anneal(size_t n_steps){
        // annealing process: relax the system with zero noise and small learning rate
        // for n_steps
        double lr = this->m_potential->get_lr();
        double prob = this->m_potential->get_prob();
        this->m_potential->set_lr(0.1*lr);
        this->m_potential->set_prob(1.0); // prob=1.0 --> noiseless
        for (size_t i=0;i<n_steps;i++){
            one_step();
        }
        // recover the state before annealing
        this->m_potential->set_lr(lr);
        this->m_potential->set_prob(prob);
    }
};

template<size_t ndim>
class inversepower_probabilistic_particlewise_sgd_clist: public base_potential_algorithm<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >{
public: 
    inversepower_probabilistic_particlewise_sgd_clist(double pow, double eps, double lr, double prob,
                  std::vector<double> const radii, 
                  std::vector<double> const boxv,std::vector<double> init_coords,
                  const double ncellx_scale=1.0, const bool balance_omp=true):
            base_potential_algorithm<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >(
                    std::make_shared<ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<ndim> >(pow,eps,lr,prob,radii,boxv,ncellx_scale,balance_omp),
                     boxv,init_coords)
                {}
    virtual anneal(size_t n_steps){
        // annealing process: relax the system with zero noise and small learning rate
        // for n_steps
        double lr = this->m_potential->get_lr();
        double prob = this->m_potential->get_prob();
        this->m_potential->set_lr(0.1*lr);
        this->m_potential->set_prob(1.0); // prob=1.0 --> noiseless
        for (size_t i=0;i<n_steps;i++){
            one_step();
        }
        // recover the state before annealing
        this->m_potential->set_lr(lr);
        this->m_potential->set_prob(prob);
    }
};




#endif
