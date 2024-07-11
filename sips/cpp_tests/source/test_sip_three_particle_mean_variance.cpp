#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "potentials/biased_kicker.hpp"
#include "potentials/inversepower_potential.hpp"


/*
 * SIP:Mean and variance tests for three particles
 */

class SIPMeanVarianceTestThreeParticles :  public ::testing::Test
{
public:
    std::vector<double> x = {-1.0,0.0, 0.0,1.0, 1.0,0.0}, radii = {sqrt(2)-1,sqrt(2)+1,sqrt(2)-1};
    double tol = 0.05;
    std::vector<double> d,avg;
    std::vector<double> boxv = {20.0,20.0};
    size_t natoms = 3, ndim = 2 , n_repeat = 8000;
    virtual void SetUp(){ 	
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
        d.resize(natoms*ndim);
    }
    //-------------helper functions -------------------
    //calculate the distance between two vectors
    double dist_vec(std::vector<double>& v1, std::vector<double>& v2){
        if (v2.size() != v1.size()){
            throw std::runtime_error("dist_vec()::v1 and v2 must have the same size");
        }
        double sum = 0.0;
        for (size_t i=0;i<v1.size();i++){
            sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
        }
        return sqrt(sum);
    }
    // calculate the distance bwtween two matrices
    double dist_mat(std::vector<std::vector<double> >& m1,std::vector<std::vector<double> >& m2, size_t ndim){
        double sum = 0.0;
        for (size_t i=0;i<ndim;i++){
            for (size_t j=0;j<ndim;j++){
                sum += (m1[i][j] - m2[i][j])*(m1[i][j] - m2[i][j]);
            }
        }
        return sqrt(sum);
    }
    // add v1 = v1 + v2
    void self_add(std::vector<double>& v1, std::vector<double>& v2){
        if (v2.size() != v1.size()){
            throw std::runtime_error("self_add()::v1 and v2 must have the same size");
        }
        for (size_t i=0;i<v1.size();i++){
            v1[i] += v2[i];
        }
    }
    // print a vector
    void print_vec(std::vector<double>& v1){
        std::cout<<"{"<<v1[0];
        for (size_t i=1;i<v1.size();i++){
            std::cout<<","<<v1[i];   
        }
        std::cout<<"}";
    }

    void get_mean_cov(std::vector<std::vector<double>>& disp_vec, std::vector<double>& mean_vec, 
                  std::vector<std::vector<double>>& cov_mat){
        // disp_vec is the list of displacement
        std::vector<double> avg = {0.,0.};
        std::vector<std::vector<double> > cov = {{0.,0.},{0.,0.}};
        for (auto d:disp_vec){
            self_add(avg,d);
        }
        self_mul(avg,double(1.0/disp_vec.size()));
        for(auto d:disp_vec){
            for (size_t i = 0 ;i < ndim; i++){
                for (size_t j = 0 ;j < ndim; j++){
                    cov[i][j] += (d[i]-avg[i])*(d[j]-avg[j])/disp_vec.size();
                }
            }
        }
        mean_vec = avg;
        cov_mat = cov;
    }
    void self_mul(std::vector<double>& v1, double n){
        for (size_t i=0;i<v1.size();i++){
            v1[i] *= n;
        }
    }

    void self_mul(std::vector<std::vector<double> >& m1,double n){
        for (size_t i = 0; i < m1.size();i++){
            for (size_t j = 0; j < m1[0].size();j++){
                m1[i][j] *= n;
            }
        }
    }
};

TEST_F(SIPMeanVarianceTestThreeParticles, ReciprocalBRO){
    double eps = 0.5; // kick_size
    ha::ReciprocalPairwiseBiasedKickerPeriodicCellLists<2> kicker(eps,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // k0 = {-1/sqrt(2),-1/sqrt(2)}; mean0 = 0.5*eps*k0, cov0 = eps^2/12*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    // k2 = {1/sqrt(2),-1/sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    std::vector<double> ans_mean0 = {-1.0/sqrt(2),-1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,sqrt(2)}; 
    std::vector<double> ans_mean2 = {1.0/sqrt(2), -1.0/sqrt(2)};
    self_mul(ans_mean0,eps*0.5);
    self_mul(ans_mean1,eps*0.5);
    self_mul(ans_mean2,eps*0.5);
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,eps*eps/12.0);
    self_mul(ans_cov1,eps*eps/12.0);
    self_mul(ans_cov2,eps*eps/12.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
    //atom 2
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2 ;
}

TEST_F(SIPMeanVarianceTestThreeParticles, NonReciprocalBRO){
    double eps = 0.5; // kick_size
    ha::NonReciprocalPairwiseBiasedKickerPeriodicCellLists<2> kicker(eps,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // k0 = {-1/sqrt(2),-1/sqrt(2)}; mean0 = 0.5*eps*k0, cov0 = eps^2/12*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    // k2 = {1/sqrt(2),-1/sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    std::vector<double> ans_mean0 = {-1.0/sqrt(2),-1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,sqrt(2)}; 
    std::vector<double> ans_mean2 = {1.0/sqrt(2), -1.0/sqrt(2)};
    self_mul(ans_mean0,eps*0.5);
    self_mul(ans_mean1,eps*0.5);
    self_mul(ans_mean2,eps*0.5);
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,eps*eps/12.0);
    self_mul(ans_cov1,eps*eps/12.0);
    self_mul(ans_cov2,eps*eps/12.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
    //atom 2
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2 ;
}

TEST_F(SIPMeanVarianceTestThreeParticles, ParticlewiseBRO){
    double eps = 0.5; // kick_size
    ha::ParticlewiseBiasedKickerPeriodicCellLists<2> kicker(eps,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        kicker.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // k0 = {-1/sqrt(2),-1/sqrt(2)}; mean0 = 0.5*eps*k0, cov0 = eps^2/12*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    // k2 = {1/sqrt(2),-1/sqrt(2)}; mean1 = 0.5*eps*k1, cov1 = eps^2/12(k0*k0^T + k1*k1^T)
    std::vector<double> ans_mean0 = {-1.0/sqrt(2),-1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,sqrt(2)}; 
    std::vector<double> ans_mean2 = {1.0/sqrt(2), -1.0/sqrt(2)};
    self_mul(ans_mean0,eps*0.5);
    self_mul(ans_mean1,eps*0.5);
    self_mul(ans_mean2,eps*0.5);
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{0.0,0.0},{0.0,2.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,eps*eps/12.0);
    self_mul(ans_cov1,eps*eps/12.0);
    self_mul(ans_cov2,eps*eps/12.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}


TEST_F(SIPMeanVarianceTestThreeParticles, ReciprocalPairwiseNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -1.0, D0 = 0.5; 
    ha::InversePowerPeriodicReciprocalPairwiseNoiseCellLists<2> pot(mpow,a, alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,alpha*a/sqrt(32.0));
    self_mul(ans_mean1,alpha*a/sqrt(32.0));
    self_mul(ans_mean2,alpha*a/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}

TEST_F(SIPMeanVarianceTestThreeParticles, NonReciprocalPairwiseNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -1.0, D0 = 0.5; 
    ha::InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists<2> pot(mpow,a, alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    std::vector<double> xx = {-1.0,0.0, 0.0,1.0, 1.0,0.0};
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,alpha*a/sqrt(32.0));
    self_mul(ans_mean1,alpha*a/sqrt(32.0));
    self_mul(ans_mean2,alpha*a/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}


TEST_F(SIPMeanVarianceTestThreeParticles, ParticlewiseNoise){
    double a = 4.0*sqrt(2), mpow = 2, alpha = -0.5, D0 = 0.5; 
    ha::InversePowerPeriodicParticlewiseNoiseCellLists<2> pot(mpow,a, alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }

    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;

    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = alpha*a/(4*sqrt(2))*k0, cov0 = D0*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = alpha*a/(4*sqrt(2))*k1, cov1 = D0*a^2/32*(k0+k2)*(k0+k2)^T 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = alpha*a/(4*sqrt(2))*k0, cov2 = D0*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,alpha*a/sqrt(32.0));
    self_mul(ans_mean1,alpha*a/sqrt(32.0));
    self_mul(ans_mean2,alpha*a/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{0.0,0.0},{0.0,2.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*D0/32.0);
    self_mul(ans_cov1,a*a*D0/32.0);
    self_mul(ans_cov2,a*a*D0/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};
    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}

TEST_F(SIPMeanVarianceTestThreeParticles, ParticlewiseBatch){
    double a = 4.0*sqrt(2), mpow = 2, lr = -0.5, p = 0.5;         
    ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<2> pot(mpow,a,lr,p,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_batch_energy_gradient(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat; 
    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = lr*p*a/(4*sqrt(2))*k0, cov0 = lr*lr*p*(1-p)*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = lr*p*a/(4*sqrt(2))*k1, cov1 = lr*lr*p*(1-p)*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = lr*p*a/(4*sqrt(2))*k0, cov2 = lr*lr*p*(1-p)*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,lr*a*p/sqrt(32.0));
    self_mul(ans_mean1,lr*a*p/sqrt(32.0));
    self_mul(ans_mean2,lr*a*p/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{0.0,0.0},{0.0,2.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov1,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov2,a*a*lr*lr*p*(1-p)/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}

TEST_F(SIPMeanVarianceTestThreeParticles, PairwiseBatch){
    double a = 4.0*sqrt(2), mpow = 2, lr = -0.5, p = 0.5;         
    ha::InversePowerPeriodicProbabilisticPairBatchCellLists<2> pot(mpow,a,lr,p,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1,disp_vec2; // the collection of dispalcments for 3 atoms
    disp_vec0.clear();
    disp_vec1.clear();
    disp_vec2.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1,d2;
        d0.clear();
        d1.clear();   
        d2.clear();
        pot.get_batch_energy_gradient(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
            d2.push_back(d[2*ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
        disp_vec2.push_back(d2);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat; 
    // k0 = {1/sqrt(2),1/sqrt(2)}; mean0 = lr*p*a/(4*sqrt(2))*k0, cov0 = lr*lr*p*(1-p)*a^2/32*k0*k0^T  
    // k1 = {0.0,sqrt(2)}; mean1 = lr*p*a/(4*sqrt(2))*k1, cov1 = lr*lr*p*(1-p)*a^2/32*(k0*k0^T+ k1*k1^T) 
    // k2 = {-1/sqrt(2),1/sqrt(2)}; mean2 = lr*p*a/(4*sqrt(2))*k0, cov2 = lr*lr*p*(1-p)*a^2/32*k2*k2^T 
    std::vector<double> ans_mean0 = {1.0/sqrt(2),1.0/sqrt(2)};
    std::vector<double> ans_mean1 = {0.0,-sqrt(2)}; 
    std::vector<double> ans_mean2 = {-1.0/sqrt(2), 1.0/sqrt(2)};
    self_mul(ans_mean0,lr*a*p/sqrt(32.0));
    self_mul(ans_mean1,lr*a*p/sqrt(32.0));
    self_mul(ans_mean2,lr*a*p/sqrt(32.0));
    std::vector<std::vector<double> > ans_cov0 = {{0.5,0.5},{0.5,0.5}};
    std::vector<std::vector<double> > ans_cov1 = {{1.0,0.0},{0.0,1.0}};
    std::vector<std::vector<double> > ans_cov2 = {{0.5,-0.5},{-0.5,0.5}};
    self_mul(ans_cov0,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov1,a*a*lr*lr*p*(1-p)/32.0);
    self_mul(ans_cov2,a*a*lr*lr*p*(1-p)/32.0);
    std::vector<std::vector<double> > zero_mat = {{0.,0.},{0.,0.}};
    std::vector<double> zero_vec = {0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 0:relative err for cov_mat = " <<rerr2;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1;
    EXPECT_LT(rerr2,tol) << "atom 1:relative err for cov_mat = " <<rerr2;
    //atom 2
    get_mean_cov(disp_vec2,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean2)/dist_vec(ans_mean2,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov2,ndim)/dist_mat(ans_cov2,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 2:relative err for mean_vec = " <<rerr1; 
    EXPECT_LT(rerr2,tol) << "atom 2:relative err for cov_mat = " <<rerr2;
}

