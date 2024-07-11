import sys
import os
sys.path.append('../sips/build')
sys.path.append('./utils')
from sips import algorithms
import numpy as np
import json
import Utils
import time
import math


class Info:
    def __init__(self,input_dir):
        input_file= os.path.join(input_dir,'info.json')
        if not os.path.isfile(input_file):
            raise FileNotFoundError("The input file" + input_file + "does not exist")
        with open(input_file) as fs:
            input_dict = json.loads(fs.read())
        self.input_dict = input_dict
        self.loc = input_dir
        compulsory_keys = ["N","dim","n_steps","n_save","n_rec","box_size","phi",
                           "optimization_method","init_config"]
        #--------------general information -------------------
        print("--------Input info:--------")
        print(json.dumps(input_dict,indent=4))
        
        for k in compulsory_keys:
            if not k in input_dict:
                raise ValueError(k + " missing in the info.json")
        # update the properties
        for k in input_dict:
            setattr(self,k,input_dict[k])
        
        if self.dim != len(self.box_size):
            raise ValueError("the box size and the dimension does not match")
        # calculate the radius
        v0 = math.pow(math.pi,0.5*self.dim)/math.gamma(0.5*self.dim + 1.0)
        v_box = np.prod(self.box_size)
        self.r = math.pow(self.phi*v_box/(self.N*v0),1.0/self.dim)
        input_dict["r"] = self.r
        
        if not "save_mode" in input_dict:
            self.save_mode = "concise"
        if not "compression" in input_dict:
            self.compression = "on"
        if not "use_clist" in input_dict:
            self.use_clist = "on"

        if not "zoom_rate" in input_dict:
            self.zoom_rate = -1
        #--------------- for BRO methods ------------------------------
        if "bro" in input_dict["optimization_method"]:
            if not "eps" in input_dict:
                raise ValueError("eps(kick size) for bro is missing in the info.json")
        #--------------- for prabalistic SGD methods -------------------
        if ("sgd" in self.optimization_method) and ("probabilistic" in self.optimization_method):
            if "match_bro" in input_dict["optimization_method"]:
                self.mpow = 1.0
                self.a0 = self.r
                self.lr = -abs(self.eps/0.75) # negative lr means gradient descent
                self.prob = 0.75 
            elif not "lr" in input_dict:   
                raise ValueError("lr for sgd is missing in the info.json")
            elif not "prob" in input_dict:
                raise ValueError("For probablistic sgd methods, prob must be given") 

        #--------------- for StoDyn methods -----------------
        if "stodyn" in input_dict["optimization_method"]:
            if "match_bro" in input_dict["optimization_method"]:
                self.mpow = 1.0
                self.a0 = self.r
                self.alpha = -abs(self.eps/0.75)
                self.D0 = (0.75 - 0.75*0.75)*self.alpha*2.0*self.r
            elif "match_sgd" in input_dict["optimization_method"]:
                self.D0 = (0.75 - 0.75*0.75)*self.lr*2.0*self.r
            elif not "alpha" in input_dict:
                raise ValueError("alpha for stodyn method is missing in the info.json")
            elif not "D0" in input_dict:
                raise ValueError("D0 for stodyn method is missing in the info.json")
        #--------------- scale the learning rate/ kick size/ alpha to the unit of radius----
        if "lr" in input_dict:
            self.lr = -abs(self.lr)*2.0*self.r
        if "eps" in input_dict:
            self.eps = abs(self.eps)*2.0*self.r
        if "alpha" in input_dict:
            self.alpha = -abs(self.alpha)*2.0*self.r
        if "a0_r" in input_dict:
            self.a0 = self.a0_r*self.r 
        #--------------- for potential based mathods--------------------
        if "inversepower" in input_dict["optimization_method"]:
            if (not hasattr(self,"a0")) or (not hasattr(self,"mpow")):
                raise ValueError("a0 and mpow must be provided for inversepower potential")
        #--------------- set the cutoff-------------------------
        if not "cutoff" in input_dict:
            # the default cut_off value is 0
            # but cutoff = 1e-7 or 1e-8 are typically good for nonlinear potentials
            self.cutoff = 0.0  
        # update the input information
        with open(input_file,'w') as fs:
            fs.write(json.dumps(input_dict,indent=4))

def initialize_position(input:Info):
    x = np.zeros((input.N,input.dim))
    current_t = 0
    if input.init_config == "random":
        for d in range(input.dim):
            x[:,d] = np.random.uniform(0.0,input.box_size[d],input.N)
        x = x.reshape(input.N*input.dim)
    elif input.init_config == 'fixed':
        st0 = np.random.get_state() #store the random seed
        np.random.seed(314) # set random seed to a lucky number
        x = np.zeros((input.N,input.dim))
        for d in range(input.dim):
            x[:,d] = np.random.uniform(0.0,input.box_size[d],input.N)
        x = x.reshape(input.N*input.dim)
        np.random.set_state(st0) #recover the old random seed
    elif input.init_config == 'appending':
        dl = Utils.DataLoader(input.loc,mode='last_existed')
        iters,x_list = dl.get_data_list('x')
        current_t = iters[-1]
        x = np.array(x_list[-1])
    return current_t,x

def construct_algorithm(input:Info,x0:np.array):
    boxv = input.box_size
    optimization_method = input.optimization_method.replace("match_bro","")
    if input.use_clist == "on":
        if "particlewise_bro" == optimization_method:
            if input.dim == 2:
                return algorithms.ParticlewiseBROCList2D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.ParticlewiseBROCList3D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.ParticlewiseBROCList4D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
        elif "reciprocal_pairwise_bro" == optimization_method:
            if input.dim == 2:
                return algorithms.ReciprocalPairwiseBROCList2D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.ReciprocalPairwiseBROCList3D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.ReciprocalPairwiseBROCList4D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
        elif "nonreciprocal_pairwise_bro" == optimization_method:
            if input.dim == 2:
                return algorithms.NonReciprocalPairwiseBROCList2D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.NonReciprocalPairwiseBROCList3D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.NonReciprocalPairwiseBROCList4D(input.eps,np.full(input.N,input.r),boxv,x0,1.0,True)
        elif "inversepower_probabilistic_particlewise_sgd" == optimization_method:
            if input.dim == 2:
                return algorithms.InversePowerProbParticlewiseSGDCList2D(input.mpow,input.a0,input.lr,input.prob,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.InversePowerProbParticlewiseSGDCList3D(input.mpow,input.a0,input.lr,input.prob,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.InversePowerProbParticlewiseSGDCList4D(input.mpow,input.a0,input.lr,input.prob,np.full(input.N,input.r),boxv,x0,1.0,True)
        elif "inversepower_probabilistic_pairwise_sgd" == optimization_method:
            if input.dim == 2:
                return algorithms.InversePowerProbPairwiseSGDCList2D(input.mpow,input.a0,input.lr,input.prob,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.InversePowerProbPairwiseSGDCList3D(input.mpow,input.a0,input.lr,input.prob,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.InversePowerProbPairwiseSGDCList4D(input.mpow,input.a0,input.lr,input.prob,np.full(input.N,input.r),boxv,x0,1.0,True)
        elif "inversepower_particlewise_stodyn" == optimization_method:
            if input.dim == 2:
                return algorithms.InversePowerParticlewiseStoDynCList2D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.InversePowerParticlewiseStoDynCList3D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.InversePowerParticlewiseStoDynCList4D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
        elif "inversepower_reciprocal_pairwise_stodyn" == optimization_method:
            if input.dim == 2:
                return algorithms.InversePowerReciprocalPairwiseStoDynCList2D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.InversePowerReciprocalPairwiseStoDynCList3D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.InversePowerReciprocalPairwiseStoDynCList4D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
        elif "inversepower_nonreciprocal_pairwise_stodyn" == optimization_method:
            if input.dim == 2:
                return algorithms.InversePowerNonReciprocalPairwiseStoDynCList2D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 3:
                return algorithms.InversePowerNonReciprocalPairwiseStoDynCList3D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
            elif input.dim == 4:
                return algorithms.InversePowerNonReciprocalPairwiseStoDynCList4D(input.mpow,input.a0,input.alpha,input.D0,np.full(input.N,input.r),boxv,x0,1.0,True)
    else:
        raise NotImplementedError("non-cell list methods have not been implemented yet")


# Inverse Potential
# E(xi,xj) = a0/a*(1-dij/(ri+rj))^a, if(dij<ri+rj) where dij = sqrt((xi-xj)^2) 
#          = 0 ,if dij>=ri+rj
# xi and xj are the coordinates of atom i and j
# the input for get_energy is x=[x1,y1,z1,x2,y2,z2 ...]
# x1,y1,z1 is the coordinate for the first atom in 3D
# return value e is the energy
# return value g is the jacobian 
# where dE/dxi_alpha = g[i*dim + alpha] i=0..N-1, alpha = 0..dim-1
#pot = potentials.InversePower(a=mpow,eps=a0,boxv=boxv,radii = np.full(N,r),
#                        ncellx_scale = None,enable_batch = True,method=Distance.PERIODIC,balance_omp=True)

# set to set the number of cores for NUMBA using environment variables 
# export NUMBA_NUM_THREADS=2


#----------------------------the main code starts------------------------------------------------
loc = sys.argv[1]     
# get the input information
info = Info(loc)
# initialization
t0,x = initialize_position(info)
# construct the potential
algo = construct_algorithm(info,x)
#algo.set_zoom_steps(0,100,2)
if info.zoom_rate > 0:
    algo.set_zoom_steps(starting_step = info.zoom_start,ending_step = info.zoom_end,
                        zoom_rate = info.zoom_rate)
print(algo)
#---------------------------the simulation starts------------------------------------------
print("--------Simulation starts--------")
t1 = time.time()
algo.run(n_steps = info.n_steps, n_save = info.n_save, n_rec = info.n_rec,
         starting_step = t0, cutoff = 1.0 - info.cutoff,
         output_dir = loc,save_mode = info.save_mode,
         compression = (info.compression=="on"))
t2 = time.time()
print("--------Simulation ends--------")
print("This simulation took " + str(t2-t1) + " secs")
