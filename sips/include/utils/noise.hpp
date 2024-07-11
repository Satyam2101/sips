#ifndef NOISE_HPP
#define NOISE_HPP
#include <random>
#include <sstream>

template<typename distribution>
class BaseNoise{
    public:
    virtual ~BaseNoise(){}
    BaseNoise(double scale=1.0):
                m_gen(std::random_device{}()),
                m_scale(scale)
                {
                    m_state << m_gen;
                }
    // abstract function to generate one number
    virtual double rand(){
        return m_scale*m_dis(m_gen); 
    }

    virtual void set_seed(unsigned s){
        m_gen.seed(s);
    }

    virtual void recover(){
        m_state >> m_gen;
    }

    protected:
        std::mt19937 m_gen;
        distribution m_dis;
        double m_scale;
        std::stringstream m_state;
};

class GaussianNoise:public BaseNoise<std::normal_distribution<double>>{
    public:
    GaussianNoise(double mean=0.0,double sigma=1.0,double scale = 1.0):BaseNoise(scale)
    {
            m_dis.reset();
            m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }
};

class UniformNoise:public BaseNoise<std::uniform_real_distribution<double>>{
    public:
    UniformNoise(double low = 0.0,double high = 1.0,double scale = 1.0):BaseNoise(scale){
            m_dis.reset();
            m_dis.param(std::uniform_real_distribution<double>::param_type(low, high));
    }
};

class ReciprocalGaussianNoise:public BaseNoise<std::normal_distribution<double>>{
    public:
    ReciprocalGaussianNoise(double mean = 0.0,double sigma = 1.0, double scale = 1.0):BaseNoise(scale){
            m_dis.reset();
            m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }
    void rand2(double* a, double* b){
        *a = rand();
        *b = *a;
    }
};


class ReciprocalUniformNoise:public BaseNoise<std::uniform_real_distribution<double>>{
    public:
    ReciprocalUniformNoise(double low = 0.0,double high = 1.0,double scale = 1.0):BaseNoise(scale){
            m_dis.reset();
            m_dis.param(std::uniform_real_distribution<double>::param_type(low, high));
    }
    void rand2(double* a, double* b){
        *a = rand();
        *b = *a;
    }
};

class NonReciprocalGaussianNoise:public BaseNoise<std::normal_distribution<double>>{
    public:
    NonReciprocalGaussianNoise(double mean = 0.0,double sigma = 1.0, double scale = 1.0):BaseNoise(scale){
            m_dis.reset();
            m_dis.param(std::normal_distribution<double>::param_type(mean, sigma));
    }
    void rand2(double* a, double* b){
        *a = rand();
        *b = rand();
    }
};


class NonReciprocalUniformNoise:public BaseNoise<std::uniform_real_distribution<double>>{
    public:
    NonReciprocalUniformNoise(double low = 0.0,double high = 1.0,double scale = 1.0):BaseNoise(scale){
            m_dis.reset();
            m_dis.param(std::uniform_real_distribution<double>::param_type(low, high));
    }
    void rand2(double* a, double* b){
        *a = rand();
        *b = rand();
    }
};


#endif