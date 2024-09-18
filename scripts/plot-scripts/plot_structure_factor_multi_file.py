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

def get_radial_sk(simulation_dir:str,N = 300,n_bins = 200,n_frames=5):
    dl = Utils.DataLoader(simulation_dir,mode="range",rng=[-n_frames,-1])
    iters,x_list = dl.get_data_list('x')
    x = [dl.get_pbc_position(np.array(_x)) for _x in x_list]
    sk_list = []
    for _x in x:
        if dl.info['dim'] == 2:
            kx,ky,s = dl.get_strucutre_factor(_x,N)
            #k,sk = Utils.get_radial_structure_factor2d(kx,ky,s,n_bins)
            k,sk = Utils.get_radial_profile(s)
        elif dl.info['dim'] == 3:
            kx,ky,kz,s = dl.get_strucutre_factor(_x,N)
            #k,sk = Utils.get_radial_structure_factor3d(kx,ky,kz,s,n_bins)
            k,sk = Utils.get_radial_profile(s)
        sk_list.append(sk)
        
    k = k.astype(float)*np.max(kx)/np.max(k)*dl.info["r"]*2.0/(2.0*np.pi)
    avg_sk = np.mean(np.array(sk_list),axis=0)
    method_str = dl.info['optimization_method']
    if method_str == "inversepower_probabilistic_pairwise_sgd":
        method_str = "pairwise SGD, phi=" + str(dl.info['phi'])
    elif method_str == "inversepower_probabilistic_particlewise_sgd":
        method_str = "particle-wise SGD, phi=" + str(dl.info['phi'])
    return k,avg_sk,method_str
    


data_dir_list = ["/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_pairwise_sgd2/lr_0.01/phi_6.325000E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_pairwise_sgd2/lr_0.01/phi_6.332500E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_pairwise_sgd2/lr_0.01/phi_6.340000E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_pairwise_sgd2/lr_0.01/phi_6.347500E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_pairwise_sgd2/lr_0.01/phi_6.355000E-01"
]


data_dir_list = [
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_particlewise_sgd2/lr_0.01/phi_6.315000E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_particlewise_sgd2/lr_0.01/phi_6.322500E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_particlewise_sgd2/lr_0.01/phi_6.330000E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_particlewise_sgd2/lr_0.01/phi_6.337500E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_particlewise_sgd2/lr_0.01/phi_6.345000E-01"]


data_dir_list = [
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_pairwise_sgd2/lr_0.01/phi_6.355000E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/hyperuniformity_particlewise_sgd2/lr_0.01/phi_6.345000E-01"]
npts = 12
power = 0.25
figure_size = [4.5,3.0]
fig = plt.figure()
fig.set_size_inches(figure_size[0], figure_size[1])
ax = fig.add_subplot(111)
colors = ['red','blue','orange','green','black','brown']
force_calculation = False
c = 0
for data_dir in data_dir_list:
    print(data_dir)
    sk_list = []
    for _dir in os.listdir(data_dir):
        print(_dir)
        if _dir.find('run') < 0:
            continue

        simulation_dir = os.path.join(data_dir,_dir)
        output_file = os.path.join(simulation_dir,'structure_factor.json')
        if os.path.isfile(output_file) and (not force_calculation):
            k,s_k = read_data(output_file)
            sk_list.append(s_k)        
        else:
            k,sk,label = get_radial_sk(simulation_dir)
            sk_list.append(sk)
    
    sk_arr = np.array(sk_list)
    avg_sk = np.mean(sk_arr,axis=0)
    err_sk = np.std(sk_arr,axis=0)
    
    #log_k,log_sk = np.log(k[0:npts]),np.log(avg_sk[0:npts])
    #res = stats.linregress(log_k, log_sk)
    #print(res)
    #linear_fit = {'slope':res.slope,'std_err':res.stderr}
    
    #method_str = dl.info['optimization_method']
    #print(linear_fit['slope'])
    #k *= dl.info["r"]*2.0/(2.0*np.pi)
    print(k)
    print(avg_sk)
    print(err_sk)
    #ax.errorbar(k[1:],avg_sk[1:],yerr=err_sk[1:],fmt='.-',c=colors[c],alpha=0.6,label=label)
    ax.plot(k[1:],avg_sk[1:],'.-',c=colors[c],alpha=0.6,label=label)
    c+=1

ax.plot(k[1:npts],(1.1*avg_sk[4]/(k[4]**0.25))*k[1:npts]**power,'--',lw=3,c='black',alpha=0.5)
ax.tick_params(axis='x',labelsize = 14)
ax.tick_params(axis = 'y', which = 'both', labelsize = 14)
ax.set_xlabel(r"$qd/2\pi$",fontsize=14)
ax.set_ylabel("$S(q)$",fontsize=14)      
ax.legend()
fig.tight_layout()
fig.savefig('strucutre_factor.png', dpi=200)
plt.show()

