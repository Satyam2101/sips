import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
# default figure size in matplotlib [6.4,4.8]
figure_size = [4.5,3.0]

plt.rcParams["font.family"] = "Times New Roman"

data_dirs = ["/home/richard/HPC-Scratch/sig-python/sips/scripts/rmsd_cmp_eps1/inversepower_particlewise_stodyn_match_bro/phi_6.300000E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/rmsd_cmp_eps1/particlewise_bro/phi_6.300000E-01",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/rmsd_cmp_eps1/prob_particlewise_sgd_match_bro/phi_6.300000E-01"
]


colors = ['red','blue','green','orange','purple','black']
markers = ['.','v','H','s','*','P']

#plot different rmsd in figure 1
fig1 = plt.figure()
fig1.set_size_inches(figure_size[0], figure_size[1])
ax1 = fig1.add_subplot(111)

'''
# plot the number of active particles 
fig2 = plt.figure()
fig2.set_size_inches(figure_size[0], figure_size[1])
ax2 = fig2.add_subplot(111)
custom_xticks = [0, 30000, 60000, 90000, 120000, 150000]
ax2.set_xticks(custom_xticks)
'''
def read_data(f):
    f_data = os.path.join(f,"msd_eps_data.json")
    f_info = os.path.join(f,"msd_eps_info.json")
    with open(f_data,'r') as fs:
        data_dict = json.loads(fs.read())
    with open(f_info,'r') as fs:
        info = json.loads(fs.read())
    for k in data_dict:
        if isinstance(data_dict[k],list):
            data_dict[k] = np.array(data_dict[k])
    return data_dict,info

def write_data(f,data,info):
    d = copy.deepcopy(data)
    for k in d:
        if isinstance(d[k],np.ndarray):
            d[k] = d[k].tolist()
    f_data = os.path.join(f,"msd_eps_data.json")
    f_info = os.path.join(f,"msd_eps_info.json")
    with open(f_data,'w') as fs:
        fs.write(json.dumps(d,indent=4))
    with open(f_info,'w') as fs:
        fs.write(json.dumps(info,indent=4))

def get_label(method):
    if "particlewise_bro" == method:
        return "nonreciprocal particle-wise BRO"
    elif "reciprocal_pairwise_bro" == method:
        return "reciprocal pairwise BRO"
    elif "inversepower_probabilistic_particlewise_sgd" in method:
        return "particlewise SGD"
    elif "inversepower_probabilistic_pairwise_sgd" in method:
        return "pairwise SGD"
    elif "inversepower_particlewise_stodyn" in method:
        #return "stochastic approximation for particle-wise BRO and SGD)"
        return "stochastic approximation"
    elif "inversepower_reciprocal_pairwise_stodyn" in method:
        #return "stochastic approximation for reciprocal pairwise BRO and SGD)"
        return "stochastic approxiamtion"
    

def msd(x,boxv):
    '''
    x is the list of numpy array
    '''
    nsteps = len(x);
    _msd = np.zeros(nsteps)
    x = np.array(x)
    x0 = x[0,:]
    N,dim = x.shape[1]//len(boxv),len(boxv)
    for k in range(nsteps):
        if k == 0:
            dx = x[k,:] - x0
            acc_dx = np.zeros((N,dim))
        else:
            dx = (x[k,:] - x[k-1,:]).reshape(N,dim)
            # for periodic boundary contition
            for d in range(dim):
                dx[np.where(dx[:,d] > boxv[d]*0.5)] -= boxv[d]
                dx[np.where(dx[:,d] < -boxv[d]*0.5)] += boxv[d]
            # acc_dx = pbc(x[k]-x[0])
            acc_dx += dx
        _msd[k] = np.mean(np.linalg.norm(acc_dx,axis=1)**2)
    return np.array(_msd)

def get_array(data_dict):
    eps_list,avg_msd,msd_err = [],[],[]
    for eps in data_dict:
        eps_list.append(float(eps))
        avg_msd.append(np.mean(data_dict[eps]))
        msd_err.append(np.std(data_dict[eps]))
    eps_arr = np.array(eps_list)
    sorted_ind = np.argsort(eps_arr)
    eps_arr = eps_arr[sorted_ind]
    avg_msd = np.array(avg_msd)[sorted_ind]
    msd_err = np.array(msd_err)[sorted_ind]
    return eps_arr,avg_msd,msd_err

def float2str(num:float):
    num = round(num,6)
    return format(num,f".{6}f")

count = 0
for dir in data_dirs:
    print(dir)
    data_dict = dict()
    info = {"dir":dir,"last_iteration":[]}
    # if the output file exists 
    if os.path.isfile(os.path.join(dir,"msd_eps_info.json")):
        data_dict,info = read_data(dir)
        eps,avg_msd,msd_err = get_array(data_dict)
        ax1.plot(eps,avg_msd,'.-',marker=markers[count],markersize = 4.0,
             label=get_label(info['optimization_method']),c=colors[count],alpha=0.5)
        ax1.fill_between(eps, avg_msd - msd_err, avg_msd + msd_err, color=colors[count],alpha=0.2)
        count += 1
        continue
    for simulation_dir in os.listdir(dir):
        print(simulation_dir)
        if simulation_dir.find('run') > 0:
            dl = Utils.DataLoader(os.path.join(dir,simulation_dir),mode='all')
            N,dim = dl.info['N'],dl.info['dim']
            info['phi'] = dl.info['phi']
            info['optimization_method'] = dl.info['optimization_method']
            steps,x = dl.get_data_list('x')
            if float2str(dl.info['eps']) in data_dict:
                data_dict[float2str(dl.info['eps'])].append(msd(x,dl.info["box_size"])[-1])
            else:
                data_dict[float2str(dl.info['eps'])] = [msd(x,dl.info["box_size"])[-1]]
    eps,avg_msd,msd_err = get_array(data_dict)
    print(type(eps[0]))
    ax1.plot(eps,avg_msd,'.-',marker=markers[count],markersize = 4.0,
             label=get_label(info['optimization_method']),c=colors[count],alpha=0.5)
    ax1.fill_between(eps, avg_msd - msd_err, avg_msd + msd_err, color=colors[count],alpha=0.2)
    count += 1
    write_data(dir,data_dict,info)


ax1.tick_params(axis='x',labelsize = 14)
ax1.tick_params(axis = 'y', which = 'both', labelsize = 14)
ax1.set_xlabel(r"$\epsilon$",fontsize=14)
ax1.set_ylabel("MSD",fontsize=14)
ax1.legend(fontsize=10)
fig1.tight_layout()
fig1.savefig('msd_eps.png', dpi=200)
plt.show()
