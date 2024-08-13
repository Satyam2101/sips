import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

#use latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.rcParams.update(plt.rcParamsDefault)

plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=14)

data_dirs = ["/home/richard/HPC-Scratch/sig-python/sips/scripts/energy_flucuation2/prob_particlewise_sgd/prob_1.000000E+00",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/energy_flucuation2/prob_particlewise_sgd/prob_2.500000E-01"]

#data_dirs = ["/home/richard/HPC-Scratch/sig-python/sips/scripts/absorbing_transition_particlewise_sgd/prob_0.2/N_25000"]     

colors = ['red','blue','green','orange','black']

counter = 0
for dir_name in data_dirs:
    print(dir_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(111) 
    # convert epsilon to string and use them as the keys
    for simulation_dir in os.listdir(dir_name):
        if counter > 200:
            continue
        print("In file loop")
        counter += 1
        data = dict()
        if simulation_dir.find('run') < 0:
            continue
        print(simulation_dir)
        json_file = os.path.join(dir_name,simulation_dir,'stat_data.json')
        print(json_file)
        if os.path.isfile(json_file):
            with open(json_file,'r') as fs:
                data = json.loads(fs.read())
            with open(os.path.join(dir_name,simulation_dir,"info.json"),'r') as fs:
                info = json.loads(fs.read())
            f_active = data['activities']
            phi = info['phi']
            if phi > 0.72:
                continue
            t = data['time_steps'] 
            #eps = data['eps']
            #eps = data['batch_lr']
        else:
            dl = Utils.DataLoader(os.path.join(dir_name,simulation_dir),mode='all')
            method = dl.info['optimization_method']
            N,dim = dl.info['N'],dl.info['dim']
            #eps = dl.info['eps']  
            #eps = data['batch_lr']          
            t,f_active = dl.get_data_list('activity')
            phi = dl.info['phi']
            data["phi"] = dl.info['phi']
            data["iter"] = t
            data["f_active"] = f_active
            #data["eps"] = dl.info['eps'] 
            with open(os.path.join(dir_name,simulation_dir,'fa_time.json'),'w') as fs:
                fs.write(json.dumps(data,indent=4))
        #ax1.loglog(np.array(t),f_active,'.-',label="phi={:.4f}".format(phi))
        ax1.semilogx(np.array(t),f_active,'.-',label="phi={:.4f}".format(phi))

    #ax1.legend()
plt.show()
