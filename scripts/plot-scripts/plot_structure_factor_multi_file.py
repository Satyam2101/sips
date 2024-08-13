import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats

def save_data(k,s_k,output_file):
    if isinstance(k,np.ndarray):
        k = k.tolist()
    if isinstance(s_k,np.ndarray):
        s_k = s_k.tolist()
    data = {'k':k,'s_k':s_k}
    with open(output_file,'w') as fs:
        fs.write(json.dumps(data))

def read_data(loc):
    if not os.path.isfile(loc):
        return None
    with open(loc,'r') as fs:
        data = json.loads(fs.read())
    k = np.array(data['k'])
    s_k = np.array(data['s_k'])
    return (k,s_k)

def get_radial_sk(simulation_dir:str,N = 400,n_bins = 280):
    dl = Utils.DataLoader(simulation_dir,mode='last_exist')
    iters,x_list = dl.get_data_list('x')
    x = x_list[0]
    if dl.info['dim'] == 2:
        kx,ky,s = dl.get_strucutre_factor(x)
        k,sk = Utils.get_radial_structure_factor2d(kx,ky,s)
    elif dl.info['dim'] == 3:
        kx,ky,kz,s = dl.get_strucutre_factor(x,N)
        k,sk = Utils.get_radial_structure_factor3d(kx,ky,kz,s,n_bins)
    return k,sk
    


data_dir_list = [
    "/home/richard/HPC-Scratch/sig-python/hyperalg/scripts/3d_linear_pot_hyperuniform/batch_size1.0/N_100000"]


npts = 32
power = 0.25
fig = plt.figure()
ax = fig.add_subplot(111)
colors = ['red','blue','orange','green']
force_calculation = False
c = 0
for data_dir in data_dir_list:
    print(data_dir)
    for _dir in os.listdir(data_dir):
        if _dir.find('run') < 0:
            continue
        print(_dir)
        simulation_dir = os.path.join(data_dir,_dir)
        sk_list = []
        output_file = os.path.join(simulation_dir,'structure_factor.json')
        if os.path.isfile(output_file) and (not force_calculation):
            k,s_k = read_data(output_file)
            sk_list.append(s_k)
            dl = Utils.DataLoader(simulation_dir,mode='last_exist')
        else:
            k,sk = get_radial_sk(simulation_dir)
            sk_list.append(sk)
    avg_sk = np.zeros_like(sk_list[0])
    for _s_k in sk_list:
        avg_sk += _s_k
    avg_sk /= len(sk_list)
    
    log_k,log_sk = np.log(k[0:npts]),np.log(avg_sk[0:npts])
    res = stats.linregress(log_k, log_sk)
    print(res)
    linear_fit = {'slope':res.slope,'std_err':res.stderr}
    
    method_str = dl.info['optimization_method']
    print(linear_fit['slope'])
    k *= dl.info["r"]*2.0/(2.0*np.pi)
    ax.plot(k,avg_sk,'.-',label = method_str,c=colors[c],alpha=0.2)
    c+=1

ax.plot(k[0:npts],k[0]*(k[0:npts]**power),'--',c='black',alpha=0.9)
ax.tick_params(axis='x',labelsize = 14)
ax.tick_params(axis = 'y', which = 'both', labelsize = 14)
ax.set_xlabel(r"$qd/2\pi$",fontsize=14)
ax.set_ylabel("$S(q)$",fontsize=14)      
ax.legend()
plt.show()

