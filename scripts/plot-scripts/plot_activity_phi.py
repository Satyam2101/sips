import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

data_dir_name = "/home/richard/HPC-Scratch/sig-python/sips/scripts/absorbing_transition2/prob_particlewise_sgd/prob_2.000000E-01"


phi_fa_pair = dict()
phi_list = []
fa_list = []
for simulation_dir in os.listdir(data_dir_name):
    if simulation_dir.find('run') < 0:
            continue
    print(simulation_dir)
    data_json = os.path.join(data_dir_name,simulation_dir,'stat_data.json')
    info_json = os.path.join(data_dir_name,simulation_dir,'info.json')
    if not os.path.isfile(data_json):
        continue
    with open(data_json,'r') as fs:
        data = json.loads(fs.read())
    with open(info_json,'r') as fs:
        info = json.loads(fs.read())

    f_active = data["activities"]
    phi = info['phi']
    fa = f_active[-1]
    phi_list.append(phi)
    fa_list.append(fa)


phi_arr = np.array(phi_list)
fa_arr = np.array(fa_list)
indices = np.argsort(phi_arr)

phi = phi_arr[indices]
f_active = fa_arr[indices]
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(phi,f_active,marker='o',facecolors='none',edgecolors='blue')


ax.tick_params(axis='x',labelsize = 14)
ax.tick_params(axis = 'y', which = 'both', labelsize = 14)
ax.set_xlabel(r"$\phi$",fontsize=14)
ax.set_ylabel("$f_a$",fontsize=15)

plt.show()



    
    
