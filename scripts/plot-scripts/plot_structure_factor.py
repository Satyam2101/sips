import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats

def save_data(k,s_k,linear_fit,output_file):
    if isinstance(k,np.ndarray):
        k = k.tolist()
    if isinstance(s_k,np.ndarray):
        s_k = s_k.tolist()
    data = {'k':k,'s_k':s_k,'linear_fit':linear_fit}
    with open(output_file,'w') as fs:
        fs.write(json.dumps(data))

def read_data(loc):
    if not os.path.isfile(loc):
        return None
    with open(loc,'r') as fs:
        data = json.loads(fs.read())
    k = np.array(data['k'])
    s_k = np.array(data['s_k'])
    linear_fit = data['linear_fit']
    return (k,s_k,linear_fit)


data_dir_list = [
    "/home/richard/HPC-Scratch/sig-python/hyperalg/scripts/3d_linear_pot_hyperuniform1/bro_smsc/N_1000000/dir1#phi_6.350000E-01#runID_1"
]


npts = 24
fig = plt.figure()
ax = fig.add_subplot(111)
colors = ['red','blue','orange','green']
force_calculation = True
c = 0
for data_dir in data_dir_list:
    print(data_dir)
    output_file = os.path.join(data_dir,'stracture_factor.json')
    if os.path.isfile(output_file) and (not force_calculation):
        k,s_k,linear_fit = read_data(output_file)
        dl = Utils.DataLoader(data_dir,mode='last_existed')
    else:
        dl = Utils.DataLoader(data_dir,mode='last_existed')
        iters,x = dl.get_data_list('x')
        '''
        print(iters)
        dl = Utils.DataLoader(data_dir,mode='range',
                         rng=[iters[-1]- 0.1*dl.info['n_save'],iters[-1]])
        iters,x_list = dl.get_data_list('x')
        sk_list = []
        for x in x_list:
            if dl.info['dim'] == 2:
                kx,ky,s = dl.get_strucutre_factor(x)
                k,s_k = Utils.get_radial_structure_factor2d(kx,ky,s)
                X,Y = np.meshgrid(kx,ky)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.contourf(X,Y,np.log(s),10)
            elif dl.info['dim'] == 3:
                kx,ky,kz,s = dl.get_strucutre_factor(x,N=400)
                k,s_k = Utils.get_radial_structure_factor3d(kx,ky,kz,s,n_bins=400)
            sk_list.append(s_k)
            print(len(sk_list))
        
        print(len(sk_list))
        s_k = np.zeros_like(sk_list[0])
        for _s_k in sk_list:
            s_k += _s_k
        s_k /= len(sk_list)
        '''
        x= x[-1]
        if dl.info['dim'] == 2:
            kx,ky,s = dl.get_strucutre_factor(x)
            k,s_k = Utils.get_radial_structure_factor2d(kx,ky,s)
            X,Y = np.meshgrid(kx,ky)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.contourf(X,Y,np.log(s),10)
        elif dl.info['dim'] == 3:
            kx,ky,kz,s = dl.get_strucutre_factor(x,N=400)
            k,s_k = Utils.get_radial_structure_factor3d(kx,ky,kz,s,n_bins=400)

        log_k,log_sk = np.log(k[2:npts]),np.log(s_k[2:npts])
        res = stats.linregress(log_k, log_sk)
        print(res)
        linear_fit = {'slope':res.slope,'std_err':res.stderr}
        save_data(k,s_k,linear_fit,output_file)
    method = dl.info['optimization_method']
    if 'sgd' in method:
        fraction = dl.info['fraction']
        method_str = '$MSG,b_f = ' + str(fraction) + '$'
    else:
        method_str = '$BRO$' 
    print(linear_fit['slope'])
    k *= dl.info["r"]*2.0/(2.0*np.pi)
    ax.plot(k,s_k,'.-',label = method_str,c=colors[c],alpha=0.2)
    #ax.plot(k,s_k,'.',c=colors[c],alpha=0.5)
    c+=1

ax.plot(k[2:20],(s_k[2]/(k[2]**0.25))*k[2:20]**0.25,'--',c='black',alpha=0.8)
ax.tick_params(axis='x',labelsize = 14)
ax.tick_params(axis = 'y', which = 'both', labelsize = 14)
ax.set_xlabel(r"$qd/2\pi$",fontsize=14)
ax.set_ylabel("$S(q)$",fontsize=14)      
ax.legend()
plt.show()

