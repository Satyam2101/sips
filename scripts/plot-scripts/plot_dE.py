import sys
import os
sys.path.append('../utils')
import Utils
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import ticker
import json

plt.rcParams['font.family'] = 'Times New Roman'

# input info
prob_list = [0.25,0.5,0.75,1.0]
fm = Utils.FileManager()
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)

def float2str(num:float,digits = 5):
    num = round(num,digits)
    return format(num,f".{5}f")

def get_array(data_dict):
    phi_list,avg_dE,std_dE = [],[],[]
    for phi_str in data_dict:
        phi_list.append(float(phi_str))
        avg_dE.append(np.mean(data_dict[phi_str]))
        std_dE.append(np.std(data_dict[phi_str]))
    phi_arr = np.array(phi_list)
    sorted_ind = np.argsort(phi_arr)
    phi_arr = phi_arr[sorted_ind]
    avg_dE = np.array(avg_dE)[sorted_ind]
    std_dE = np.array(std_dE)[sorted_ind]
    return phi_arr,avg_dE,std_dE

def calculate_and_plot(root_dir,bf_list,ax,l_str,colors,fmt = '.'):
    data_dict = {}
    for count,p in enumerate(prob_list):
        fm.update_folder_list_by_root_dir(root_dir)
        simulation_dirs = fm.filter_by_dict({"prob":p},update_list=True)
        data_dict = dict()
        for sim_dir in simulation_dirs:
            print(sim_dir)
            phi = Utils.extract_value_from_filename(sim_dir,"phi")
            phi_str = float2str(phi)
            if os.path.isfile(os.path.join(sim_dir,"stat_data.json")):
                with open(os.path.join(sim_dir,"stat_data.json"),"r") as f:
                    stat_data_dict = json.loads(f.read())
                dE = stat_data_dict["energy_flucuation"]/np.power(phi,0.3333333)
                if phi_str in data_dict:
                    data_dict[phi_str].append(dE)
                else:
                    data_dict[phi_str] = [dE]
        phi,avg_dE,dE_err = get_array(data_dict)
        simulation_dirs = fm.sorted_by_value("phi")
        (plotline, _, _) = ax.errorbar(phi,avg_dE,yerr=dE_err,fmt=fmt,color=colors[count],markersize=6,label=l_str + ",$p = {:2}$".format(p))
        plotline.set_markerfacecolor('none')
        #ax.scatter(phi,dE,s=8,marker=fmt,color=colors[count])
          

#cmap = get_cmap('cool')
cmap = get_cmap('gist_heat')


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

# Using contourf to provide my colorbar info, then clearing the figure
mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',[cmap(i/4.0) for i in range(4)])
Z = [[0,0],[0,0]]
CS3 = ax1.contourf(Z, levels=[0.25,0.5,0.75,1.0,1.25], cmap=mymap)
ax1.clear()

root_dir1 = "/home/richard/HPC-Scratch/sig-python/sips/scripts/energy_flucuation2/prob_pairwise_sgd"
calculate_and_plot(root_dir1,[0.25,0.5,0.75,1.0],ax1,'pairwise SGD',[cmap(i/4.0) for i in range(4)],'-s')


root_dir2 = "/home/richard/HPC-Scratch/sig-python/sips/scripts/energy_flucuation2/prob_particlewise_sgd"
calculate_and_plot(root_dir2,[0.25,0.5,0.75,1.0],ax1,'particle-wise SGD',[cmap(i/4.0) for i in range(4)],'-^')



ax1.set_xticks(ax1.get_xticks()[::4])
ax1.legend()
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
ax1.tick_params(axis='x',labelsize = 17)
ax1.tick_params(axis = 'y', which = 'both', labelsize = 17)
ax1.set_xlabel(r"$\phi$")
ax1.set_ylabel(r"$\Delta V$")

#ax1.legend(fontsize=15)
cb = plt.colorbar(CS3,ax=ax1) # using the colorbar info I got from contourf
cb.set_ticks([])
cb.outline.set_visible(False)
plt.show()
