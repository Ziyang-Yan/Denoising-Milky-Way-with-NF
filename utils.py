from typing import *
import matplotlib as mpl
import numpy as np
import copy
import scipy
from astropy import units as u
from astropy.coordinates import (SkyCoord, Distance, Galactic)
import astropy.coordinates as coord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from astropy import uncertainty as unc
import os
from pathlib import Path
from pickle import TRUE
from datetime import datetime
from time import perf_counter as time
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing
from tqdm import *


import random

from .utils import load_and_gen_sample
from .plotting import plot_his,plot_velocity,plot_velocity_z,plot_spiral,plot_1d,plot_corner_6d_large

def select_stars(table , sector):
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

def make_mock_data(input_data,input_error,amp,training_fraction=0.7,seed=42):
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

def sam_tran(data,error,N,seed= None):
    np.random.seed(seed)
    random.seed(seed)

    mas_per_yr = u.mas/u.yr
    km_per_s = u.km/u.s

    parallax_sample = unc.normal(center= data[:,2], std=error[:,2], n_samples=N).distribution *u.mas
    parallax_factor = np.ones_like(parallax_sample.value.reshape(-1,1))
    parallax_factor[parallax_sample.value.reshape(-1,1)<0] = 0
    parallax_factor = parallax_factor.reshape(parallax_sample.value.shape)
    stars_sample = SkyCoord(
        ra = unc.normal(center= data[:,0], std=error[:,0], n_samples=N).distribution * u.degree,
        dec = unc.normal(center= data[:,1], std=error[:,1], n_samples=N).distribution * u.degree,
        distance = Distance(parallax = abs(parallax_sample)),
        pm_ra_cosdec = unc.normal(center= data[:,3], std=error[:,3], n_samples=N).distribution * mas_per_yr,
        pm_dec = unc.normal(center= data[:,4], std=error[:,4], n_samples=N).distribution * mas_per_yr,
        radial_velocity = unc.normal(center= data[:,5], std=error[:,5], n_samples=N).distribution *km_per_s
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
    return stars_data_cyl, parallax_factor,stars_sample


def selection_func(stars_sam,sam_max, sam_min = 1, sig_v = 1 ):
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
    if tensor_order != None:
        output_data = np.hstack([input_data[key].reshape(-1,1) for key in tensor_order])
    else:
        output_data = np.hstack((input_data['v_r'].reshape(-1,1), input_data['v_phi'].reshape(-1,1),
                input_data['v_z'].reshape(-1,1), input_data['z'].reshape(-1,1 ),
                input_data['rho'].reshape(-1,1), input_data['phi'].reshape(-1,1 )))
    return output_data

def mask_gen(input_data,dim, boundaries):
    mask =[]
    for i in range(len(boundaries)):
        mask.append(
            (input_data[:,dim] > boundaries[i][0]) &  (input_data[:,dim] < boundaries[i][1])
        )
    return mask

def load_and_plot(path,tf=0.7):
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



