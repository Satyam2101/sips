#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "potentials/biased_kicker.hpp"
#include "potentials/inversepower_potential.hpp"


/*
 * SIP:Mean and variance tests
 */

class SIPMeanVarianceTestTwoParticles :  public ::testing::Test
{
public:
    std::vector<double> x = {0.,0.,0.,3.,4.,5.}, radii = {6*sqrt(2),4*sqrt(2)};
    std::vector<double> d;
    std::vector<double> boxv = {50.0,50.0,50.0};
    size_t natoms = 2, ndim = 3 , n_repeat = 4800;
    double tol = 0.06;
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
            throw std::runtime_error("v1 and v2 must have the same size");
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
            throw std::runtime_error("v1 and v2 must have the same size");
        }
        for (size_t i=0;i<v1.size();i++){
            v1[i] += v2[i];
        }
    }
    // v1 = v1/ n
    void self_div(std::vector<double>& v1, double n){
        for (size_t i=0;i<v1.size();i++){
            v1[i] /= n;
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
        std::vector<double> avg = {0.,0.,0.};
        std::vector<std::vector<double> > cov = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
        for (auto d:disp_vec){
            self_add(avg,d);
        }
        self_div(avg,double(disp_vec.size()));
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
};

TEST_F(SIPMeanVarianceTestTwoParticles, ReciprocalBRO){
    double kick_size = 2.0;
    ha::ReciprocalPairwiseBiasedKickerPeriodicCellLists<3> kicker1(kick_size,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1;
    disp_vec0.clear();
    disp_vec1.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1;
        d0.clear();
        d1.clear();
        kicker1.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    std::vector<double> ans_mean0 = {-3.0/sqrt(50),-4.0/sqrt(50),-5.0/sqrt(50)};
    std::vector<double> ans_mean1 = {3.0/sqrt(50),4.0/sqrt(50),5.0/sqrt(50)};
    std::vector<std::vector<double> > ans_cov0 = {{9.0/150,12.0/150,15.0/150},
                                                 {12.0/150,16.0/150,20.0/150},
                                                 {15.0/150,20.0/150,25.0/150}};
    std::vector<std::vector<double> > ans_cov1 = {{9.0/150,12.0/150,15.0/150},
                                                 {12.0/150,16.0/150,20.0/150},
                                                 {15.0/150,20.0/150,25.0/150}};
    std::vector<std::vector<double> > zero_mat = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
    std::vector<double> zero_vec = {0.,0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;

}

TEST_F(SIPMeanVarianceTestTwoParticles, NonReciprocalPairwiseBRO){
    double kick_size = 2.0;
    ha::NonReciprocalPairwiseBiasedKickerPeriodicCellLists<3> kicker1(kick_size,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1;
    disp_vec0.clear();
    disp_vec1.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1;
        d0.clear();
        d1.clear();
        kicker1.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    std::vector<double> ans_mean0 = {-3.0/sqrt(50),-4.0/sqrt(50),-5.0/sqrt(50)};
    std::vector<double> ans_mean1 = {3.0/sqrt(50),4.0/sqrt(50),5.0/sqrt(50)};
    std::vector<std::vector<double> > ans_cov0 = {{9.0/150,12.0/150,15.0/150},
                                                 {12.0/150,16.0/150,20.0/150},
                                                 {15.0/150,20.0/150,25.0/150}};
    std::vector<std::vector<double> > ans_cov1 = {{9.0/150,12.0/150,15.0/150},
                                                 {12.0/150,16.0/150,20.0/150},
                                                 {15.0/150,20.0/150,25.0/150}};
    std::vector<std::vector<double> > zero_mat = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
    std::vector<double> zero_vec = {0.,0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
   
}

TEST_F(SIPMeanVarianceTestTwoParticles, ParticlewiseBRO){
    double kick_size = 2.0;
    ha::ParticlewiseBiasedKickerPeriodicCellLists<3> kicker1(kick_size,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1;
    disp_vec0.clear();
    disp_vec1.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1;
        d0.clear();
        d1.clear();
        kicker1.get_kick(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    std::vector<double> ans_mean0 = {-3.0/sqrt(50),-4.0/sqrt(50),-5.0/sqrt(50)};
    std::vector<double> ans_mean1 = {3.0/sqrt(50),4.0/sqrt(50),5.0/sqrt(50)};
    std::vector<std::vector<double> > ans_cov0 = {{9.0/150,12.0/150,15.0/150},
                                                 {12.0/150,16.0/150,20.0/150},
                                                 {15.0/150,20.0/150,25.0/150}};
    std::vector<std::vector<double> > ans_cov1 = {{9.0/150,12.0/150,15.0/150},
                                                 {12.0/150,16.0/150,20.0/150},
                                                 {15.0/150,20.0/150,25.0/150}};
    std::vector<std::vector<double> > zero_mat = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
    std::vector<double> zero_vec = {0.,0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
   
}

TEST_F(SIPMeanVarianceTestTwoParticles, ReciprocalPairwiseNoise){
    double a = 200.0, mpow = 2.0, alpha = -2.0, D0 = 2.0;
    ha::InversePowerPeriodicReciprocalPairwiseNoiseCellLists<3> pot(mpow,a,alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1;
    disp_vec0.clear();
    disp_vec1.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1;
        d0.clear();
        d1.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // g0 = {3,4,5}; mean0 = alpha*g0, cov0 = D0*g0*g0^T  
    // g1 = {-3,-4,-5}; mean1 = alpha*g1, cov1 = D0*g1*g1^T  
    std::vector<double> ans_mean0 = {-6.0,-8.0,-10.0};
    std::vector<double> ans_mean1 = {6.0,8.0,10.0};
    std::vector<std::vector<double> > ans_cov0 = {{18.0,24.0,30.0},
                                                  {24.0,32.0,40.0},
                                                  {30.0,40.0,50.0}};
    std::vector<std::vector<double> > ans_cov1 = {{18.0,24.0,30.0},
                                                  {24.0,32.0,40.0},
                                                  {30.0,40.0,50.0}};
    std::vector<std::vector<double> > zero_mat = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
    std::vector<double> zero_vec = {0.,0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
   
}

TEST_F(SIPMeanVarianceTestTwoParticles, NonReciprocalPairwiseNoise){
    double a = 200.0, mpow = 2.0, alpha = -2.0, D0 = 2.0;
    ha::InversePowerPeriodicNonReciprocalPairwiseNoiseCellLists<3> pot(mpow,a,alpha,D0,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1;
    disp_vec0.clear();
    disp_vec1.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1;
        d0.clear();
        d1.clear();
        pot.get_stochastic_force(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // g0 = {3,4,5}; mean0 = alpha*g0, cov0 = D0*g0*g0^T  
    // g1 = {-3,-4,-5}; mean1 = alpha*g1, cov1 = D0*g1*g1^T  
    std::vector<double> ans_mean0 = {-6.0,-8.0,-10.0};
    std::vector<double> ans_mean1 = {6.0,8.0,10.0};
    std::vector<std::vector<double> > ans_cov0 = {{18.0,24.0,30.0},
                                                  {24.0,32.0,40.0},
                                                  {30.0,40.0,50.0}};
    std::vector<std::vector<double> > ans_cov1 = {{18.0,24.0,30.0},
                                                  {24.0,32.0,40.0},
                                                  {30.0,40.0,50.0}};
    std::vector<std::vector<double> > zero_mat = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
    std::vector<double> zero_vec = {0.,0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
   
}

TEST_F(SIPMeanVarianceTestTwoParticles, ProbabilitsticPairwiseSGD){
    double a = 200.0, mpow = 2.0, lr = -2.0, p = 0.8;
    ha::InversePowerPeriodicProbabilisticPairBatchCellLists<3> pot(mpow,a,lr,p,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1;
    disp_vec0.clear();
    disp_vec1.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1;
        d0.clear();
        d1.clear();
        pot.get_batch_energy_gradient(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // g0 = {3,4,5}; mean0 = lr*p*g0, cov0 = lr*lr*(p-p*p)*g0*g0^T  
    // g1 = {-3,-4,-5}; mean1 = lr*p*g1, cov1 = lr*lr*(p-p*p)*g1*g1^T
    double s1 = lr*p, s2 = lr*lr*(p - p*p);
    std::vector<double> ans_mean0 = {3*s1,4.*s1,5.*s1};
    std::vector<double> ans_mean1 = {-3.*s1,-4.*s1,-5.*s1};
    std::vector<std::vector<double> > ans_cov0 = {{9.0*s2, 12.0*s2, 15.0*s2},
                                                 {12.0*s2, 16.0*s2, 20.0*s2},
                                                 {15.0*s2, 20.0*s2, 25.0*s2}};
    std::vector<std::vector<double> > ans_cov1 = {{9.0*s2, 12.0*s2, 15.0*s2},
                                                 {12.0*s2, 16.0*s2, 20.0*s2},
                                                 {15.0*s2, 20.0*s2, 25.0*s2}};
    std::vector<std::vector<double> > zero_mat = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
    std::vector<double> zero_vec = {0.,0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
}

TEST_F(SIPMeanVarianceTestTwoParticles, ProbabilitsticParticlewiseSGD){
    double a = 200.0, mpow = 2.0, lr = -2.0, p = 0.8;
    ha::InversePowerPeriodicProbabilisticParticleBatchCellLists<3> pot(mpow,a,lr,p,radii,boxv);
    std::vector<std::vector<double>> disp_vec0,disp_vec1;
    disp_vec0.clear();
    disp_vec1.clear();
    std::vector<double> d;
    d.resize(natoms*ndim);
    for (size_t i = 0; i < n_repeat; i++){
        std::vector<double> d0,d1;
        d0.clear();
        d1.clear();
        pot.get_batch_energy_gradient(x,d);
        for (size_t k = 0 ; k < ndim ; k++){
            d0.push_back(d[k]);
            d1.push_back(d[ndim + k]);
        }
        disp_vec0.push_back(d0);
        disp_vec1.push_back(d1);
    }
    std::vector<double> mean_vec;
    std::vector<std::vector<double> > cov_mat;
    // g0 = {3,4,5}; mean0 = lr*p*g0, cov0 = lr*lr*(p-p*p)*g0*g0^T  
    // g1 = {-3,-4,-5}; mean1 = lr*p*g1, cov1 = lr*lr*(p-p*p)*g1*g1^T
    double s1 = lr*p, s2 = lr*lr*(p - p*p);
    std::vector<double> ans_mean0 = {3*s1,4.*s1,5.*s1};
    std::vector<double> ans_mean1 = {-3.*s1,-4.*s1,-5.*s1};
    std::vector<std::vector<double> > ans_cov0 = {{9.0*s2, 12.0*s2, 15.0*s2},
                                                 {12.0*s2, 16.0*s2, 20.0*s2},
                                                 {15.0*s2, 20.0*s2, 25.0*s2}};
    std::vector<std::vector<double> > ans_cov1 = {{9.0*s2, 12.0*s2, 15.0*s2},
                                                 {12.0*s2, 16.0*s2, 20.0*s2},
                                                 {15.0*s2, 20.0*s2, 25.0*s2}};
    std::vector<std::vector<double> > zero_mat = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
    std::vector<double> zero_vec = {0.,0.,0.};

    //atom 0  
    get_mean_cov(disp_vec0,mean_vec,cov_mat);
    double rerr1 = dist_vec(mean_vec,ans_mean0)/dist_vec(ans_mean0,zero_vec);
    double rerr2 = dist_mat(cov_mat,ans_cov0,ndim)/dist_mat(ans_cov0,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 0:relative err for cov_mat = " <<rerr2 ;
    //atom 1
    get_mean_cov(disp_vec1,mean_vec,cov_mat);
    rerr1 = dist_vec(mean_vec,ans_mean1)/dist_vec(ans_mean1,zero_vec);
    rerr2 = dist_mat(cov_mat,ans_cov1,ndim)/dist_mat(ans_cov1,zero_mat,ndim);
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for mean_vec = " <<rerr1 ;
    EXPECT_LT(rerr1,tol) << "atom 1:relative err for cov_mat = " <<rerr2 ;
}
