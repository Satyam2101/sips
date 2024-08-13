import sys
import os
sys.path.append('../utils')
import Utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
dl = Utils.DataLoader('/home/richard/HPC-Scratch/sig-python/hyperalg/scripts/RO_BRO_SME_2D_20000_sme/D_noise_0.000000E+00/dir1#phi_9.000000E-01#runID_0',mode = 'last_exist')
splot = Utils.SpherePlot(dl)
splot.active_color = 'r'
fig = plt.figure()


def my_plot(iter):
    print(iter)
    fig.clear()
    ax = fig.add_subplot(111)
    splot.plot_atoms(ax,splot.loader.data[iter])
    #splot.plot_batch(ax,splot.loader.data[iter])
    #splot.plot_gradient(ax,splot.loader.data[iter],scale = 0.2,alpha = 1.0)
    ax.set_title("iteration={}".format(splot.loader.data[iter]['iter']))
    ax.set_xlim([0,splot.loader.info['box_size'][0]])
    ax.set_ylim([0,splot.loader.info['box_size'][1]])
    ax.set_aspect('equal')

ani = animation.FuncAnimation(fig, my_plot,frames=np.array([-1]),
                             interval=1, blit=False)
#ani = animation.FuncAnimation(fig, my_plot,frames=np.arange(0,99),
#                             interval=1, blit=False)

#plt.show()
splot.save(ani,output_file = "./test.gif",fps = 10,dpi = 1000)
