import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import copy
import scipy
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from pickle import TRUE
from datetime import datetime
from time import perf_counter as time
import random
import corner


def plot_his(epoch_his,loss_his, loss_his_test = None, save_path = None,name = 'his'):
    # print(f'denoise setting {best_epoch}_{epoch_his}_s_{self.number_of_sam}_{self.check_point["best_loss"]:.6f}')
    plt.figure(2)
    plt.plot(epoch_his,loss_his, label = 'train_loss')
    if loss_his_test:
        plt.plot(epoch_his,loss_his_test, label = 'test_loss')
    plt.legend()
    plt.title(f'loss history')
    # plt.suptitle(f'rho {self.stars_dataset.sector["rho"]*u.kpc}, phi {self.stars_dataset.sector["phi"]*u.deg}, z {self.stars_dataset.sector["z"]*u.kpc}')
    plt.xlabel(f'number of epoch')
    plt.ylabel(f'loss')
    # plt.text(.01, .99, f'best_epoch = {best_epoch}, best_loss={best_loss:.5f}', ha='left', va='top')
    if save_path:
        plt.savefig(os.path.join(save_path,name), format='png', bbox_inches = 'tight' , dpi = 600)
    plt.show()






def plot_velocity(data_list,data_text,title, n_norm = 0,save_path = None,file_name = 'velocity'):
    nbins=1
    vrange = [[-150, 150.0],[50.0, 350], [-150, 150.0]]
    bins = [(nbins*round(vrange[0][1] - vrange[0][0]) , round(nbins*vrange[1][1] - vrange[1][0])),(nbins*round(vrange[0][1] - vrange[0][0]) , round(nbins*vrange[2][1] - vrange[2][0])),(nbins*round(vrange[2][1] - vrange[2][0]) , round(nbins*vrange[1][1] - vrange[1][0]))]
    extent =  ([vrange[0][0],vrange[0][1],vrange[1][0],vrange[1][1]],[vrange[0][0],vrange[0][1],vrange[2][0],vrange[2][1]],[vrange[2][0],vrange[2][1],vrange[1][0],vrange[1][1]])


    def gen_mask(data,vrange):
        return    (
        (data[:,0] > vrange[0][0]) & (data[:,0] < vrange[0][1])
        & (data[:,1] > vrange[1][0]) & (data[:,1] < vrange[1][1])
        & (data[:,2] > vrange[2][0]) & (data[:,2] < vrange[2][1])
        )


    fig, axs = plt.subplots(nrows=len(data_list), ncols=3 ,figsize=(3*4,len(data_list)*4))
    vmax = 0

    con = gen_mask(data_list[n_norm],vrange)

    vr = data_list[n_norm][con][:,0]
    vphi = data_list[n_norm][con][:,1]
    vz = data_list[n_norm][con][:,2]

    stats1,_,_,_ = scipy.stats.binned_statistic_2d(vr,vphi,vz,statistic='count', bins=bins[0], range = vrange[:2] )
    stats2,_,_,_ = scipy.stats.binned_statistic_2d(vr,vz,vphi,statistic='count', bins=bins[1], range = [vrange[0],vrange[2]] )
    stats3,_,_,_ = scipy.stats.binned_statistic_2d(vz,vphi,vr,statistic='count', bins=bins[2], range = [vrange[2],vrange[1]] )

    vmax = [stats1.max(),stats2.max(),stats3.max()]





    for i in range(len(data_list)):

        con = gen_mask(data_list[i],vrange)

        vr = data_list[i][con][:,0]
        vphi = data_list[i][con][:,1]
        vz = data_list[i][con][:,2]

        stats1,_,_,_ = scipy.stats.binned_statistic_2d(vr,vphi,vz,statistic='count', bins=bins[0], range = vrange[:2] )
        stats2,_,_,_ = scipy.stats.binned_statistic_2d(vr,vz,vphi,statistic='count', bins=bins[1], range = [vrange[0],vrange[2]] )
        stats3,_,_,_ = scipy.stats.binned_statistic_2d(vz,vphi,vr,statistic='count', bins=bins[2], range = [vrange[2],vrange[1]] )

        im = axs[i,0].imshow(stats1.T,cmap=cm.jet, origin='lower', extent = extent[0] , interpolation= 'none',
                            norm = cm.colors.LogNorm(vmin = 0.01, vmax = vmax[0]))
        axs[i,0].set_title(f'{title} v_r vs v_phi')
        axs[i,0].text(.01, .99, f'{data_text[i]}', ha='left', va='top', transform=axs[i,0].transAxes)
        plt.colorbar(im,ax=axs[i,0],pad = .015)

        im = axs[i,1].imshow(stats2.T,cmap=cm.jet, origin='lower', extent = extent[1] , interpolation= 'none',
                            norm = cm.colors.LogNorm(vmin = 0.01, vmax = vmax[1]))
        axs[i,1].set_title(f'{title} v_r vs v_z')
        axs[i,1].text(.01, .99, f'{data_text[i]}', ha='left', va='top', transform=axs[i,1].transAxes)
        plt.colorbar(im,ax=axs[i,1],pad = .015)

        im = axs[i,2].imshow(stats3.T,cmap=cm.jet, origin='lower', extent = extent[0] , interpolation= 'none',
                            norm = cm.colors.LogNorm(vmin = 0.01, vmax = vmax[2]))
        axs[i,2].set_title(f'{title} v_z vs v_phi')
        axs[i,2].text(.01, .99, f'{data_text[i]}', ha='left', va='top', transform=axs[i,2].transAxes)
        plt.colorbar(im,ax=axs[i,2],pad = .015)


    if save_path:
        plt.savefig(os.path.join(save_path,file_name), format='png', bbox_inches = 'tight' , dpi = 1000)
    plt.show()



def plot_velocity_z(data_list,data_text,title,save_path = None,file_name = 'velocity_z'):
    bin_size = 0.5
    vrange =[[-60,60],[160,220]]
    bins = (round((vrange[0][1] - vrange[0][0])/bin_size),round((vrange[1][1] - vrange[1][0])/bin_size))
    extent = ([vrange[0][0],vrange[0][1],vrange[1][0],vrange[1][1]])


    def gen_mask(data,vrange):
        return    (
        (data[:,0] > vrange[0][0]) & (data[:,0] < vrange[0][1])
        & (data[:,1] > vrange[1][0]) & (data[:,1] < vrange[1][1])
        )


    fig, axs = plt.subplots(nrows=1, ncols=len(data_list) ,figsize=(len(data_list)*5,4))
    vmax = 0
    stats_list=[]
    for i in range(len(data_list)):

        con = gen_mask(data_list[i],vrange)

        vr = data_list[i][con][:,0]
        vphi = data_list[i][con][:,1]

        stats,_,_,_ = scipy.stats.binned_statistic_2d(vr,vphi,None,statistic='count', bins=bins, range = vrange )
        if i ==0:
            vmax = stats.max()
            print(vmax)


        im = axs[i].imshow(stats.T,cmap=cm.jet, origin='lower', extent = extent , interpolation= 'none',
                        #    aspect='auto',
                            norm = cm.colors.LogNorm(vmin = 0.01, vmax = vmax))
        axs[i].set_title(f'{title} v_r vs v_phi')
        axs[i].text(.01, .99, f'{data_text[i]}', ha='left', va='bottom', transform=axs[i].transAxes)
        plt.colorbar(im,ax=axs[i],pad = .015)
        stats_list.append(stats.T)

    if save_path:
        plt.savefig(os.path.join(save_path,file_name), format='png', bbox_inches = 'tight' , dpi = 1000)
    plt.show()



def plot_spiral(data_list,data_text,name,save_path = None):

    vrange = {
        'z' : [-1, 1],
        'v_z' : [-60.0,60.0],
        'v_r' : [-10.0,10.0],
        'v_phi' : [200.0, 240.0 ]
    }
    bins = (round((vrange['z'][1] - vrange['z'][0])/0.02) , round(vrange['v_z'][1] - vrange['v_z'][0]))

    fig, axs = plt.subplots(nrows=2, ncols=len(data_list) ,figsize=(len(data_list)*5,2*4))

    for i in range(len(data_list)):

        z = data_list[i][:,3]
        vz = data_list[i][:,2]
        vr = data_list[i][:,0]
        vphi = data_list[i][:,1]

        stats_vr,x_edge,y_edge,_ = scipy.stats.binned_statistic_2d(z,vz,vr,statistic='median', bins=bins, range = [vrange['z'],vrange['v_z']] )
        stats_vphi,x_edge,y_edge,_ = scipy.stats.binned_statistic_2d(z,vz,vphi,statistic='median', bins=bins, range = [vrange['z'],vrange['v_z']] )
        XX, YY = np.meshgrid(x_edge, y_edge)


        im = axs[0,i].pcolormesh(XX,YY,stats_vr.T, vmin = vrange['v_r'][0], vmax=vrange['v_r'][1] , cmap=cm.bwr)

        axs[0,i].set_title(f'{name} z vs vz vs vr')
        axs[0,i].text(.01, .99, f'{data_text[i]}', ha='left', va='bottom', transform=axs[0,i].transAxes)
        plt.colorbar(im,ax=axs[0,i],pad = .015)

        im = axs[1,i].pcolormesh(XX,YY,stats_vphi.T, vmin = vrange['v_phi'][0], vmax=vrange['v_phi'][1] , cmap=cm.jet)
        axs[1,i].set_title(f'{name} z vs vz vs vphi')
        axs[1,i].text(.01, .99, f'{data_text[i]}', ha='left', va='bottom', transform=axs[1,i].transAxes)
        plt.colorbar(im,ax=axs[1,i],pad = .015)

    if save_path:
        plt.savefig(os.path.join(save_path,'spiral'), format='png', bbox_inches = 'tight' , dpi = 1000)
    plt.show()



def plot_1d(result,save_path = None,file_name = '1d'):
    if result['model_parameters']['flow_setting']['features'] == 3:
        vrange = [[-200.0, 200.0],[0.0, 400.0], [-200.0, 200.0]]
    elif result['model_parameters']['flow_setting']['features'] == 6:
        vrange = [[-200.0, 200.0],[0.0, 400.0], [-200.0, 200.0],[-2.5,2.5],[8.1-0.2,8.1+0.2],[180-2,180+2]]



    def gen_mask(data,vrange):
        return    (
        (data[:,0] > vrange[0][0]) & (data[:,0] < vrange[0][1])
        & (data[:,1] > vrange[1][0]) & (data[:,1] < vrange[1][1])
        & (data[:,2] > vrange[2][0]) & (data[:,2] < vrange[2][1])
        & (data[:,3] > vrange[3][0]) & (data[:,3] < vrange[3][1])
        )

    con_flow = gen_mask(result['flow_sam'],vrange)
    con_train = gen_mask(result['train_cyl'],vrange)
    con_mock = gen_mask(result['mock_cyl'],vrange)


    a = pd.DataFrame(result['flow_sam'][con_flow],columns=[r'$v_r$','$v_{{\phi}}$','$v_z$','$z$','$\rho$','$\phi$'])
    a.insert(0,'type','flow')
    b = pd.DataFrame(result['train_cyl'][con_train],columns=[r'$v_r$','$v_{{\phi}}$','$v_z$','$z$','$\rho$','$\phi$'])
    b.insert(0,'type','gaia')
    c = pd.DataFrame(result['mock_cyl'][con_mock],columns=[r'$v_r$','$v_{{\phi}}$','$v_z$','$z$','$\rho$','$\phi$'])
    c.insert(0,'type','mock')
    data_frame = pd.concat([a,b,c])


    for x in ['$v_r$','$v_{{\phi}}$','$v_z$','$z$']:
        sns_plot = sns.displot(data_frame,x=x,hue='type',kind='kde')
        plt.savefig(os.path.join(save_path,file_name + x + '.pdf'), format='pdf', bbox_inches = 'tight' , dpi = 300)
        plt.show()

def plot_corner_6d_large(datasets, labels_list, colors, max_points=100000, save_path=None, file_name="corner_plot",):
    """
    Generates a 6D corner plot optimized for large datasets with multiple overlays.

    Parameters:
    - datasets: list of ndarrays or DataFrames with 6 columns ('$v_r$', '$v_{\phi}$', '$v_z$', '$z$', '$\rho$', '$\phi$')
    - labels_list: list of labels for each dataset
    - colors: list of colors for each dataset
    - max_points: int (default=100000) - Maximum number of points per dataset to plot to avoid memory issues.
    - save_path: str (optional) - If provided, saves the figure to this path.
    - file_name: str (optional) - Name of the output file (without extension).

    Returns:
    - A corner plot visualizing pairwise relationships between the 6 parameters with overlays.
    """

    # Column labels for LaTeX formatting
    labels = [r"$v_r$ [km/s]", r"$v_{\phi}$ [km/s]", r"$v_z$ [km/s]", r"$z$ [kpc]", r"$\rho$ [kpc]", r"$\phi$ [deg]"]

    # Process and sample each dataset
    processed_datasets = []
    for data in datasets:
        if data.shape[0] > max_points:
            idx = np.random.choice(data.shape[0], max_points, replace=False)  # Random subset
            data = data[idx, :]
        processed_datasets.append(data)

    # Generate the corner plot with the first dataset
    fig = corner.corner(
        processed_datasets[0],
        labels=labels,
        range=[0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
        levels = (0.393,0.864),
        show_titles=True,
        title_kwargs={"fontsize": 14},
        label_kwargs={"fontsize": 14},
        bins=50,
        hist_kwargs={"density": True, "alpha": 0.5, "color": colors[0]},
        smooth=0.7,
        plot_contours=True,
        plot_datapoints=False,
        fill_contours=False,
        contour_kwargs={"colors": colors[0]}
    )

    # Overlay the additional datasets
    for i, data in enumerate(processed_datasets[1:]):
        corner.corner(
            data,
            labels=labels,
            range=[0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
            levels = (0.393,0.864),
            fig=fig,
            bins=50,
            hist_kwargs={"density": True, "alpha": 0.5, "color": colors[i+1]},
            smooth=0.7,
            plot_contours=True,
            plot_datapoints=False,
            fill_contours=False,
            contour_kwargs={"colors": colors[i+1]}
        )

    # Add legend
    plt.legend(handles=[plt.Line2D([0], [0], color=colors[i], lw=2, label=labels_list[i]) for i in range(len(datasets))],
               loc='lower right', fontsize=12)

    # Adjust layout for readability
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(f"{save_path}/{file_name}.pdf", format="pdf", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()



