from typing import *
import numpy as np
from astropy import units as u
from astropy.coordinates import (SkyCoord, Distance, Galactic)
import astropy.coordinates as coord
from astropy import uncertainty as unc
from pickle import TRUE
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import *
import random
from .plotting import plot_his,plot_velocity,plot_velocity_z,plot_spiral,plot_1d,plot_corner_6d_large
from .flow import load_and_gen_sample

def select_stars(table, sector):
    """
    Filters stars from the input astropy Qtable (with units) based on the specified sector.
    Supports Cartesian and Galactocentric coordinate systems.
    """
    
    gaia_stars = table
    parallax = gaia_stars['parallax'].value
    sindx = np.where( (parallax > 0)
                     & ((gaia_stars['parallax_error']/gaia_stars['parallax']).value < 0.5)
    )
    gaia_stars = gaia_stars[sindx]

    stars_data = SkyCoord(
        ra = gaia_stars['ra'],
        dec = gaia_stars['dec'],
        distance = Distance(parallax = gaia_stars['parallax'])
    )

    if sector['flag'] == 'cartesian':
        con = ((np.sqrt((stars_data.cartesian.x - sector['center_x']*u.kpc)**2 + (stars_data.cartesian.y - sector['center_y']*u.kpc)**2) < sector['radius']*u.kpc)
            & (abs(stars_data.cartesian.z) < sector['z_limit']*u.kpc))
    elif sector['flag'] == 'galcen':
        stars_data = stars_data.transform_to(coord.builtin_frames.Galactocentric())
        stars_data.representation_type = 'cylindrical'
        stars_data.phi.wrap_at('360d', inplace=True)
        con = (abs(stars_data.rho - sector['rho_0']*u.kpc) <  sector['d_rho']*u.kpc) & (abs(stars_data.phi - sector['phi_0']*u.deg) < sector['d_phi']*u.deg) & (abs(stars_data.cartesian.z) < sector['z_limit']*u.kpc)
    else:
        raise Exception('flag should be cartesian or galcen')

    gaia_stars = gaia_stars[con]

    # print(f'number of stars select = {gaia_stars["ra"].size}')
    return gaia_stars, sector

def make_mock_data(input_data, input_error, amp, training_fraction=0.7, seed=42):
    """
    Generates mock data by adding Gaussian noise to the training data.
    Splits the input data into training and testing sets.
    """
    np.random.seed(seed)
    random.seed(seed)
    n_star = len(input_data)
    indices = np.random.permutation(n_star)
    training_idx, test_idx = indices[:int(n_star * training_fraction)], indices[int(n_star * training_fraction):]
    training_data, test_data = input_data[training_idx], input_data[test_idx]
    training_error, test_error = input_error[training_idx], input_error[test_idx]

    mock_error = training_error * amp
    mock_data = np.array([np.random.normal(training_data[:,i], mock_error[:,i]) for i in range(6)]).T
    return mock_data, mock_error, test_data,training_data



def sampling_function(data, error, N, seed=None):
    """
    Generates N samples for each star using Gaussian distributions.
    Ensures positive parallax values.
    """
    np.random.seed(seed)
    random.seed(seed)
    # uncorelated guassian distribution
    sample = np.stack([unc.normal(center= data[:,i], std=error[:,i], n_samples=N).distribution for i in range(6)],axis=2)
    parallax_factor = np.ones_like(sample[:,:,2]).reshape(-1,1)
    parallax_factor[sample[:,:,2].reshape(-1,1)<0] = 0
    return sample, parallax_factor.reshape(sample[:,:,2].shape)

def sam_tran(data, error, N, seed=None):
    """
    Transforms sampled data into Galactocentric cylindrical coordinates.
    Returns the transformed data and parallax correction factors.
    """
    sample,parallax_factor = sampling_function(data,error,N,seed)
    
    mas_per_yr = u.mas/u.yr
    km_per_s = u.km/u.s

    stars_sample = SkyCoord(
        ra = sample[:,:,0] * u.degree,
        dec = sample[:,:,1] * u.degree,
        distance = Distance(parallax = abs(sample[:,:,2])*u.mas), # ensure positive parallax
        pm_ra_cosdec = sample[:,:,3] * mas_per_yr,
        pm_dec = sample[:,:,4] * mas_per_yr,
        radial_velocity = sample[:,:,5] *km_per_s
    )
    stars_sample = stars_sample.transform_to(coord.builtin_frames.Galactocentric())
    stars_sample.representation_type = 'cylindrical'
    stars_sample.phi.wrap_at('360d', inplace=True)


    stars_data_cyl = {
        'rho' : stars_sample.rho.to(u.kpc).value,
        'phi' : stars_sample.phi.value,
        'z' : stars_sample.z.to(u.kpc).value,
        'v_r' : stars_sample.d_rho.to(u.km/u.s).value,
        'v_phi' : -(stars_sample.rho * stars_sample.d_phi).to(u.km/u.s, equivalencies = u.dimensionless_angles()).value,
        'v_z' : stars_sample.d_z.to(u.km/u.s).value
    }
    return stars_data_cyl,parallax_factor,stars_sample


def selection_func(stars_sam, sam_max, sam_min=1, sig_v=1):
    """
    Determines the number of samples per star based on velocity variance.
    Returns a selection factor tensor for sampling.
    """
    number_of_sam = sam_max
    sam_per_star = (np.ceil((np.maximum(stars_sam['v_r'].var(axis=1),
                        stars_sam['v_phi'].var(axis=1),
                        stars_sam['v_z'].var(axis=1)  ) + sig_v**2)/sig_v**2)).astype(int)
    sam_per_star[sam_per_star <= sam_min] = sam_min
    sam_per_star[sam_per_star >= sam_max] = sam_max
    sam_per_star = torch.tensor(sam_per_star)

    selection_factor = torch.zeros((len(sam_per_star),number_of_sam))
    for i in range(len(sam_per_star)):
        selection_factor[i, 0:sam_per_star[i]] = 1

    return selection_factor.reshape(-1,1), sam_per_star

def dict_to_tensor(input_data,tensor_order = None):
    """
    Converts a dictionary of star data into a tensor.
    Orders the data based on the specified tensor order.
    """
    if tensor_order != None:
        output_data = np.hstack([input_data[key].reshape(-1,1) for key in tensor_order])
    else:
        output_data = np.hstack((input_data['v_r'].reshape(-1,1), input_data['v_phi'].reshape(-1,1),
                input_data['v_z'].reshape(-1,1), input_data['z'].reshape(-1,1 ),
                input_data['rho'].reshape(-1,1), input_data['phi'].reshape(-1,1 )))
    return output_data

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



