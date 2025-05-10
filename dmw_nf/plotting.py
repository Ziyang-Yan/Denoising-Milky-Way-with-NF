from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import os
import corner
from .flow import load_and_gen_sample

def plot_his(epoch_his,loss_his, loss_his_test = None, save_path = None,name = 'his'):
    """
    Plots the training and testing loss history over epochs.
    Optionally saves the plot to a specified path.
    """
    plt.figure(2)
    plt.plot(epoch_his,loss_his, label = 'train_loss')
    if loss_his_test:
        plt.plot(epoch_his,loss_his_test, label = 'test_loss')
    plt.legend()
    plt.title(f'loss history')
    plt.xlabel(f'number of epoch')
    plt.ylabel(f'loss')
    if save_path:
        plt.savefig(os.path.join(save_path,name), format='png', bbox_inches = 'tight' , dpi = 600)
    plt.show()






def plot_velocity(data_list,data_text,title, n_norm = 0,save_path = None,file_name = 'velocity'):
    """
    Plots 2D velocity histograms for multiple datasets.
    Includes comparisons of v_r vs v_phi, v_r vs v_z, and v_z vs v_phi.
    """
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
    """
    Plots 2D histograms of v_r vs v_phi for multiple datasets.
    Focuses on a specific velocity range.
    """
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
    """
    Plots the spiral structure of the galaxy using z vs v_z.
    Includes median values of v_r and v_phi.
    """
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

def plot_velocity_residual(
    data_list, row_a=0, row_b=2, data_labels=None,
    save_path=None, file_name='velocity_residual', zvalue=None
):
    """
    Computes and plots residuals between two datasets in velocity space.
    Residuals are normalized by the combined counts in bins.
    """
    # Define velocity range and bins
    vrange = [[-150, 150.0], [50.0, 350], [-150, 150.0]]
    nbins = 1
    bins = [
        (nbins * round(vrange[0][1] - vrange[0][0]), round(nbins * vrange[1][1] - vrange[1][0])),
        (nbins * round(vrange[0][1] - vrange[0][0]), round(nbins * vrange[2][1] - vrange[2][0])),
        (nbins * round(vrange[2][1] - vrange[2][0]), round(nbins * vrange[1][1] - vrange[1][0]))
    ]
    extent = (
        [vrange[0][0], vrange[0][1], vrange[1][0], vrange[1][1]],
        [vrange[0][0], vrange[0][1], vrange[2][0], vrange[2][1]],
        [vrange[2][0], vrange[2][1], vrange[1][0], vrange[1][1]]
    )

    def gen_mask(data, vrange):
        return (
            (data[:, 0] > vrange[0][0]) & (data[:, 0] < vrange[0][1]) &
            (data[:, 1] > vrange[1][0]) & (data[:, 1] < vrange[1][1]) &
            (data[:, 2] > vrange[2][0]) & (data[:, 2] < vrange[2][1])
        )

    def extract(data, mask):
        return data[mask][:, 0], data[mask][:, 1], data[mask][:, 2]

    # Apply masks and extract components
    mask_a = gen_mask(data_list[row_a], vrange)
    mask_b = gen_mask(data_list[row_b], vrange)
    vr_a, vphi_a, vz_a = extract(data_list[row_a], mask_a)
    vr_b, vphi_b, vz_b = extract(data_list[row_b], mask_b)

    # Compute 2D histograms
    stat_a1, _, _, _ = scipy.stats.binned_statistic_2d(vr_a, vphi_a, vz_a, statistic='count', bins=bins[0], range=vrange[:2])
    stat_b1, _, _, _ = scipy.stats.binned_statistic_2d(vr_b, vphi_b, vz_b, statistic='count', bins=bins[0], range=vrange[:2])

    stat_a2, _, _, _ = scipy.stats.binned_statistic_2d(vr_a, vz_a, vphi_a, statistic='count', bins=bins[1], range=[vrange[0], vrange[2]])
    stat_b2, _, _, _ = scipy.stats.binned_statistic_2d(vr_b, vz_b, vphi_b, statistic='count', bins=bins[1], range=[vrange[0], vrange[2]])

    stat_a3, _, _, _ = scipy.stats.binned_statistic_2d(vz_a, vphi_a, vr_a, statistic='count', bins=bins[2], range=[vrange[2], vrange[1]])
    stat_b3, _, _, _ = scipy.stats.binned_statistic_2d(vz_b, vphi_b, vr_b, statistic='count', bins=bins[2], range=[vrange[2], vrange[1]])

    # Residuals (Z-score-like: difference over sqrt(count))
    res1 = np.where(stat_a1 + stat_b1 !=0,(stat_a1 - stat_b1) / np.sqrt(stat_a1 + stat_b1),0)
    res2 = np.where(stat_a2 + stat_b2 !=0,(stat_a2 - stat_b2) / np.sqrt(stat_a2 + stat_b2),0)
    res3 = np.where(stat_a3 + stat_b3 !=0,(stat_a3 - stat_b3) / np.sqrt(stat_a3 + stat_b3),0)


    # Symmetrical color scale (linear)
    vmax = np.abs([res1, res2, res3]).max()
    print(res1.max(),res2.max(),res3.max())
    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    titles = [r'$v_r$ vs $v_{\phi}$', r'$v_r$ vs $v_z$', r'$v_z$ vs $v_{\phi}$']
    extents = [extent[0], extent[1], extent[2]]
    data_pairs = [res1, res2, res3]
    labels = [(r'$v_r$', r'$v_{\phi}$'), (r'$v_r$', r'$v_z$'), (r'$v_z$', r'$v_{\phi}$')]

    for i in range(3):
        im = axs[i].imshow(
            data_pairs[i].T,
            cmap='seismic',
            origin='lower',
            extent=extents[i],
            interpolation='none',
            vmin=-vmax,
            vmax=vmax
        )
        # axs[i].set_title(f'Residual: {titles[i]}', fontsize=12)
        axs[i].set_xlabel(labels[i][0] + ' [km/s]', fontsize=12)
        axs[i].set_ylabel(labels[i][1] + ' [km/s]', fontsize=12)

    # Colorbar
    fig.subplots_adjust(left=0.07, right=0.89, top=0.9, bottom=0.12, wspace=0.20)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.015, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'Normalised Residual Counts in Bin', fontsize=12)


    # Save as PDF
    if save_path:
        plt.savefig(os.path.join(save_path, file_name + '.pdf'), format='pdf', bbox_inches='tight')

    plt.show()






def plot_1d(result, save_path=None, file_name='1d'):
    """
    Plots 1D distributions of velocity and spatial parameters.
    Uses kernel density estimation for smooth curves.
    """
    # Define velocity range based on feature count
    if result['model_parameters']['flow_setting']['features'] == 3:
        vrange = [[-200.0, 200.0], [0.0, 400.0], [-200.0, 200.0]]
    elif result['model_parameters']['flow_setting']['features'] == 6:
        vrange = [[-200.0, 200.0], [0.0, 400.0], [-200.0, 200.0], [-2.5, 2.5], [7.9, 8.3], [178, 182]]
    else:
        raise ValueError("Unsupported feature count in model parameters.")
    
    def gen_mask(data,vrange):
        return    (
        (data[:,0] > vrange[0][0]) & (data[:,0] < vrange[0][1])
        & (data[:,1] > vrange[1][0]) & (data[:,1] < vrange[1][1])
        & (data[:,2] > vrange[2][0]) & (data[:,2] < vrange[2][1])
        & (data[:,3] > vrange[3][0]) & (data[:,3] < vrange[3][1])
        )
    # Apply masks
    con_flow = gen_mask(result['flow_sam'], vrange)
    con_train = gen_mask(result['train_cyl'], vrange)
    con_mock = gen_mask(result['mock_cyl'], vrange)

    # Build combined DataFrame
    a = pd.DataFrame(result['flow_sam'][con_flow], columns=[r'$v_r$', '$v_{{\phi}}$', '$v_z$', '$z$', '$\rho$', '$\phi$'])
    a.insert(0, 'type', 'flow')
    b = pd.DataFrame(result['train_cyl'][con_train], columns=[r'$v_r$', '$v_{{\phi}}$', '$v_z$', '$z$', '$\rho$', '$\phi$'])
    b.insert(0, 'type', 'gaia')
    c = pd.DataFrame(result['mock_cyl'][con_mock], columns=[r'$v_r$', '$v_{{\phi}}$', '$v_z$', '$z$', '$\rho$', '$\phi$'])
    c.insert(0, 'type', 'mock')
    data_frame = pd.concat([a, b, c])

    # Line styles, colors, and alpha
    style_map = {
        'gaia': {'label': 'Gaia', 'linestyle': '-', 'color': 'blue', 'alpha': 0.6,'linewidth' : 4.5},
        'mock': {'label': 'Mock', 'linestyle': '--', 'color': 'black', 'alpha': 0.8, 'linewidth' : 2.5},
        'flow': {'label': 'Flow', 'linestyle': ':', 'color': 'red', 'alpha': 0.8, 'linewidth' : 2.5}
    }

    # Plot each variable
    for x in [r'$v_r$', r'$v_{{\phi}}$', r'$v_z$', r'$z$']:
        plt.figure()
        for type_key, props in style_map.items():
            subset = data_frame[data_frame['type'] == type_key][x].dropna()
            sns.kdeplot(
                subset, label=props['label'], linestyle=props['linestyle'],
                color=props['color'], alpha=props['alpha'], linewidth=props['linewidth']
            )
        plt.xlabel(x, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Distribution of {x}', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, file_name + x + '.pdf'), format='pdf', bbox_inches='tight', dpi=300)
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


def mask_gen(input_data,dim, boundaries):
    """
    Generates masks for filtering data based on specified boundaries.
    """
    mask =[]
    for i in range(len(boundaries)):
        mask.append(
            (input_data[:,dim] > boundaries[i][0]) &  (input_data[:,dim] < boundaries[i][1])
        )
    return mask


def load_and_plot(path, tf=0.7):
    """
    Loads model results and generates various plots for analysis.
    Includes velocity, spiral, and corner plots.
    """
    result = load_and_gen_sample(path,tf)
    train_cyl = result['train_cyl']
    mock_cyl = result['mock_cyl']
    flow_sam = result['flow_sam']
    model_parameters = result['model_parameters']


    plot_his(model_parameters['best_epoch'], model_parameters['best_loss'],
    model_parameters['epoch_his'],model_parameters['loss_his'],model_parameters['loss_his_test'],
    model_parameters['path'])

    data_list = [train_cyl,mock_cyl,flow_sam]
    data_text = ['gaia','mock','flow']

    plot_velocity(data_list,data_text,f'', 1,path, f'velocity')
    plot_velocity_z(data_list,data_text,f'',path,f'velocity_z')

    boundaries = [[-1.0,1.0],[-0.3,0.3],[-0.1,0.1],[-0.05,0.05]]
    train_mask = mask_gen(train_cyl, 3 , boundaries)
    mock_mask = mask_gen(mock_cyl, 3 , boundaries)
    flow_mask = mask_gen(flow_sam, 3 , boundaries)
    data_text = ['gaia','mock','flow']
    if model_parameters['flow_setting']['features'] != 3:
        plot_spiral(data_list,data_text,'',path)
    for i in range(len(boundaries)):
        data_list = [train_cyl[train_mask[i]],mock_cyl[mock_mask[i]],flow_sam[flow_mask[i]]]
        plot_velocity(data_list,data_text,f'{boundaries[i][0]}<z<{boundaries[i][1]}', 1,path, f'{boundaries[i][0]}<z<{boundaries[i][1]}_velocity')
        plot_velocity_z(data_list,data_text,f'{boundaries[i][0]}<z<{boundaries[i][1]}',path,f'{boundaries[i][0]}<z<{boundaries[i][1]}_velocity_z')
    plot_1d(result,path,f'1d')

    plot_corner_6d_large(
    datasets=data_list,
    labels_list=data_text,
    colors=["blue", "red", "green"],
    max_points=1000000,
    save_path=path,
    file_name='corner')
