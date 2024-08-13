import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats


plt.rcParams['font.family'] = 'Times New Roman'

data_dirs = [
"/home/richard/HPC-Scratch/sig-python/sips/scripts/phic/prob_pairwise_sgd/prob_0.2",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/phic/prob_pairwise_sgd/prob_0.6",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/phic/prob_pairwise_sgd/prob_1.0",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/phic/prob_particlewise_sgd/prob_0.2",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/phic/prob_particlewise_sgd/prob_0.6",
"/home/richard/HPC-Scratch/sig-python/sips/scripts/phic/prob_particlewise_sgd/prob_1.0"

]

colors = ['red','brown','green','orange','black','cyan','blue']
marker_list = ['o','v','H','s','*','P','.']
eps_list = [1e-3,1e-2,5e-2,1e-1,2e-1,3e-1]
f_th = 0.1
fm = Utils.FileManager()
overwrite = False

def float2str(num:float,digits = 5):
    num = round(num,digits)
    return format(num,f".{5}f")

def get_data_arrays(data_dict):
    x,y = [],[]
    for k in data_dict:
        x.append(float(k))
        y.append(data_dict[k])
    x = np.array(x)
    y = np.array(y)
    args = np.argsort(x)
    x = x[args]
    y = y[args]
    return x,y

def get_legend_str(info_dict):
    if 'particlewise_sgd' in info_dict['optimization_method']:
        method = r'particle-wise SGD, p={}'.format(info_dict['prob'])
    elif 'pairwise_sgd' in info_dict['optimization_method']:
        method = r'pairwise SGD, p={}'.format(info_dict['prob'])
    return method

def plot_line(ax, x_min,x_max,slope,intercept,color):
    x = np.arange(x_min,x_max,0.01)
    y = slope*x + intercept
    ax.plot(x,y,'-',c=color,alpha = 0.5 )


fig1 = plt.figure()
fig1.set_size_inches(4.5, 3.0)
ax1 = fig1.add_subplot(111)
count = 0
# add all the data to a single list
x_all = []
y_all = []
for _dir in data_dirs:
    print(_dir)
    # if the output file exists 
    f_data = os.path.join(_dir,"phic_data.json")
    f_info = os.path.join(_dir,"phic_info.json")
    if os.path.isfile(f_data) and (not overwrite):
        with open(f_data,'r') as fs:
            data_dict = json.loads(fs.read())
        with open(f_info,'r') as fs:
            info_dict = json.loads(fs.read())
    else:
        fm.update_folder_list_by_root_dir(_dir)
        if "sgd" in _dir:
            var_name = "lr"
        elif "bro" in _dir:
            var_name = "eps"
        elif "stodyn" in _dir:
            var_name = "alpha"
    
        simulation_dirs = fm.sorted_by_value(var_name)
        data_dict = dict()
        info_dict = dict()
        for var_val in simulation_dirs:
            data_dict[var_val] = 0.0
            for data_dir in simulation_dirs[var_val]:
                print(data_dir)
                dl = Utils.DataLoader(data_dir,mode='last_existed')
                method = dl.info['optimization_method']
                phi = dl.info['phi']      
                iters,activity = dl.get_data_list('activity')
                f_active = activity[-1]
                if f_active < f_th and data_dict[var_val] < phi:
                    data_dict[var_val] = phi
                # add additional informations
                info_dict["optimization_method"] = dl.info['optimization_method']
                info_dict["var"] = var_name
                if 'sgd' in dl.info['optimization_method']:
                    info_dict["prob"] = dl.info['prob']
                elif 'stodyn' in dl.info['optimization_method']:
                    info_dict["D0"] = dl.info["D0"]
        print(data_dict)
        with open(f_data,'w') as fs:
            fs.write(json.dumps(data_dict,indent=4))
        with open(f_info,'w') as fs:
            fs.write(json.dumps(info_dict,indent=4))

    x,y = get_data_arrays(data_dict)
    res = stats.linregress(x, y)
    info_dict["slope"] = res.slope
    info_dict["intercept"] = res.intercept
    info_dict["slope_stderr"] = res.stderr
    info_dict["intercept_stderr"] = res.intercept_stderr
    # save data
    with open(f_data,'w') as fs:
        fs.write(json.dumps(data_dict,indent=4))
    with open(f_info,'w') as fs:
        fs.write(json.dumps(info_dict,indent=4))
    #plot_line(ax1,x_min=0.0,x_max=x[-1]+0.1,slope=info_dict["slope"],intercept=info_dict["intercept"],color=colors[count])
    ax1.scatter(x,y,marker = marker_list[count],c=colors[count],alpha = 0.7,label = get_legend_str(info_dict))
    # add all the data points into a single list
    for _x in x.tolist():
        x_all.append(_x)
    for _y in y.tolist():
        y_all.append(_y)
    count += 1 
    
x_all,y_all = np.array(x_all),np.array(y_all)
arg = np.argsort(x_all)
x_all = x_all[arg]
y_all = y_all[arg]
res = stats.linregress(x_all, y_all)
plot_line(ax1,x_min=0.0,x_max=x_all[-1]+0.1,slope=res.slope,intercept=res.intercept,color=colors[count])

x = np.linspace(-0.0,0.25,10)
y = np.array([0.64]*10)
ax1.plot(x,y,'--')

ax1.tick_params(axis='x',labelsize = 12)
ax1.tick_params(axis = 'y', which = 'both', labelsize = 12)
ax1.set_xlabel(r"$\alpha$",fontsize=12)
ax1.set_ylabel(r"$\phi_c$",fontsize=12)
ax1.legend(fontsize=8.5)
fig1.tight_layout()
fig1.savefig('sgd.png', dpi=200)
plt.show()
