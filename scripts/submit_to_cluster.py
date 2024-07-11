import sys
import os
import matplotlib.pyplot as plt
import itertools
import subprocess
import json
import time
import numpy as np
import copy


class BatchJobManager:
    def __init__(self, input_config, comp_config,root_dir,rm = False):
        self.input_config = input_config
        self._input_config = input_config
        self.comp_config = comp_config
        self.root_dir = root_dir
        self.check_nthreads(float(comp_config["NTHREADS"]),input_config["N"],input_config["phi"])
        self.count = 0
        self.check_input()
        if os.path.isdir(root_dir) and rm:
            subprocess.run(['rm','-r',root_dir])
        subprocess.run(['mkdir',root_dir])
        
    def check_nthreads(self,Nth,N,phi):
        if Nth*Nth*Nth > 3.14159*N/(48*phi):
            raise RuntimeError("#Threads check fails \n Increase the number of particles or decrease the number of threads ")
    def check_input(self):
        if type(self.input_config['n_save']) == str:
            raise RuntimeError("n_save should be a int ")
    #scan two parameters
    def scan_params(self,scan_dict,parent_param = None,n_repeat = 1):
        '''
        root_dir--root directory to store the data
        scan_dict--the dictionary for parameters e,g {"phi":[0.1,0.2],"N":[100,200],"batch_szie":[1,2]}
        n_repeat--runs n_repeat computaion(with the same input parameters but different initial configuration) 
        parent_param -- a new directory will be created for each value of the parent_param 
        '''
        if parent_param:
            parent_param_values = scan_dict[parent_param]
            scan_dict.pop(parent_param)
            for ppv in parent_param_values:
                ppv_str = self.parameters2str({parent_param:ppv})
                sub_dir_name = os.path.join(self.root_dir,ppv_str)
                subprocess.run(['mkdir',sub_dir_name])
                count = 0
                for run_id in range(n_repeat):
                    self.input_config["runID"] = run_id
                    for pv_dict in self.get_unzip_dict(scan_dict):
                        data_dir_name = "dir{}#".format(count) + self.parameters2str(pv_dict) + "#runID_"+str(run_id) 
                        subprocess.run(['mkdir',os.path.join(sub_dir_name,data_dir_name)])
                        self.input_config[parent_param] = ppv
                        self.update_input_config(pv_dict)
                        self.write_input(os.path.join(sub_dir_name,data_dir_name))
                        self.submit_config(os.path.join(sub_dir_name,data_dir_name))
                        self.restore_input_config()
                        count += 1
        else:
            count = 0
            for run_id in range(n_repeat):
                self.input_config["runID"] = run_id
                for pv_dict in self.get_unzip_dict(scan_dict):
                    data_dir_name = "dir{}#".format(count) + self.parameters2str(pv_dict) + "#runID_"+str(run_id) 
                    subprocess.run(['mkdir',os.path.join(self.root_dir,data_dir_name)])
                    self.update_input_config(pv_dict)
                    self.write_input(os.path.join(self.root_dir,data_dir_name))
                    self.submit_config(os.path.join(self.root_dir,data_dir_name))
                    self.restore_input_config()
                    count += 1
            
                    
    def parameters2str(self,pv_dict):
        rtn = ""
        for p in pv_dict.keys():
            if type(pv_dict[p]) == float or type(pv_dict[p]) == np.float64:
                rtn +="{}_{:.6E}#".format(p,pv_dict[p])
            else:
                rtn +="{}_{}#".format(p,pv_dict[p])
        return rtn[:-1]
        
        
    def get_unzip_dict(self,scan_dict):
        '''
        scan_dict = {'a':[a1,a2..an],'b':[b1,b2..bn]...}
                  where a and b are parameter names in input_config.keys()
        return unzipped dicts, {'a':a1,'b':b1 ...},{'a':a1,'b':b2 ...}.
        '''
        value_lists=  list(itertools.product(*scan_dict.values())) 
        param_name_list = list(scan_dict.keys())
        for v_list in value_lists:
            param_value_dict = {param_name_list[i]:v_list[i] for i in range(len(param_name_list))}
            yield param_value_dict

    def update_input_config(self,config):
        for k in config.keys():
            self.input_config[k] = config[k]

    def restore_input_config(self):
        self.input_config = self._input_config

    def write_input(self,loc):
        with open(os.path.join(loc,"info.json"),'w') as fs:
            fs.write(json.dumps(self.input_config,indent=4))
    
    def submit_config(self,args,nsleep = 0.05):
        with open('./submit_batch.INI', 'r') as file:
            fstring = file.read()
        for key in self.comp_config:
            fstring = fstring.replace(key, self.comp_config[key])
        
        if type(args) != str:
            raise NotImplementedError("args must be a string")
        fstring = fstring.replace("ARGS", args)
        with open('./submit_batch.sbatch', 'w') as file:
            file.write(fstring)
        subprocess.run(["sbatch", "submit_batch.sbatch"])
        time.sleep(nsleep)
        self.count += 1
        subprocess.run(["rm", "submit_batch.sbatch"])
    

input_config3D = {
    "runID": 0,
    "dim": 3,
    "N": 64,
    "box_size": [
        100.0,
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":40,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}

input_config2D = {
    "runID": 0,
    "dim": 3,
    "N": 64,
    "box_size": [
        100.0,
        100.0
    ],
    "phi": 0.15,
    "n_steps": 100000,
    "n_save": 1000,
    "n_rec":40,
    "save_mode": "concise",
    "optimization_method": "particlewise_bro",
    "init_config": "random",
}


comp_config = {
    "NTHREADS":"2",
    "PYTHON_EXE":"run.py",
    "MEMORY":"4GB",
    "TIME":"50:00:00",
    "CONDA_ENV":"hyperalg"
    
}


# rmsd comparison
root_dir = '/scratch/gz2241/sig-python/sips/scripts/rmsd_cmp_steps/'
subprocess.run(['mkdir',root_dir])

# 1 particlewise BRO
job_root_dir = os.path.join(root_dir,'particlewise_bro')
input_config = copy.deepcopy(input_config3D)
input_config["optimization_method"] = "particlewise_bro"
input_config["eps"] = 0.001
input_config["N"] = 40000
input_config["n_steps"] = 180000
input_config["n_save"] = 3600
input_config["init_config"] = "fixed"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 14400
input_config["zoom_rate"] = 20
scan_dict = {'phi':[0.63,0.65]}
comp_config["TIME"] = "48:00:00"
comp_config["NTHREADS"] = "1"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 10)

# 2 reciprocal pairwise BRO
job_root_dir = os.path.join(root_dir,'reciprocal_particlewise_bro')
input_config = copy.deepcopy(input_config3D)
input_config["optimization_method"] = "reciprocal_pairwise_bro"
input_config["eps"] = 0.01
input_config["N"] = 40000
input_config["n_steps"] = 180000
input_config["n_save"] = 3600
input_config["init_config"] = "fixed"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 14400
input_config["zoom_rate"] = 20
scan_dict = {'phi':[0.63,0.65]}
comp_config["TIME"] = "48:00:00"
comp_config["NTHREADS"] = "1"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 10)

# 3 particlewise stodyn match bro
job_root_dir = os.path.join(root_dir,'inversepower_particlewise_stodyn_match_bro')
input_config = copy.deepcopy(input_config3D)
input_config["optimization_method"] = "inversepower_particlewise_stodyn_match_bro"
input_config["eps"] = 0.01
input_config["N"] = 40000
input_config["n_steps"] = 180000
input_config["n_save"] = 3600
input_config["init_config"] = "fixed"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 14400
input_config["zoom_rate"] = 20
scan_dict = {'phi':[0.63,0.65]}
comp_config["TIME"] = "48:00:00"
comp_config["NTHREADS"] = "1"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 10)

# 4 reciprocal pairwise stodyn match bro
job_root_dir = os.path.join(root_dir,'inversepower_reciprocal_particlewise_stodyn_match_bro')
input_config = copy.deepcopy(input_config3D)
input_config["optimization_method"] = "inversepower_reciprocal_particlewise_stodyn_match_bro"
input_config["eps"] = 0.01
input_config["N"] = 40000
input_config["n_steps"] = 180000
input_config["n_save"] = 3600
input_config["init_config"] = "fixed"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 14400
input_config["zoom_rate"] = 20
scan_dict = {'phi':[0.63,0.65]}
comp_config["TIME"] = "48:00:00"
comp_config["NTHREADS"] = "1"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 10)

# 5 particlewise sgd match bro
job_root_dir = os.path.join(root_dir,'prob_particlewise_sgd match_bro')
input_config = copy.deepcopy(input_config3D)
input_config["optimization_method"] = "inversepower_probabilistic_particlewise_sgd_match_bro"
input_config["eps"] = 0.01
input_config["N"] = 40000
input_config["n_steps"] = 180000
input_config["n_save"] = 3600
input_config["init_config"] = "fixed"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 14400
input_config["zoom_rate"] = 20
scan_dict = {'phi':[0.63,0.65]}
comp_config["TIME"] = "48:00:00"
comp_config["NTHREADS"] = "1"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 10)

# 6 pairwise sgd match bro
job_root_dir = os.path.join(root_dir,'prob_pairwise_sgd match_bro')
input_config = copy.deepcopy(input_config3D)
input_config["optimization_method"] = "inversepower_probabilistic_pairwise_sgd_match_bro"
input_config["eps"] = 0.01
input_config["N"] = 40000
input_config["n_steps"] = 180000
input_config["n_save"] = 3600
input_config["init_config"] = "fixed"
input_config["zoom_start"] = 0
input_config["zoom_end"] = 14400
input_config["zoom_rate"] = 20
scan_dict = {'phi':[0.63,0.65]}
comp_config["TIME"] = "48:00:00"
comp_config["NTHREADS"] = "1"
jobs = BatchJobManager(input_config,comp_config,job_root_dir)
jobs.scan_params(scan_dict,"phi",n_repeat= 10)






