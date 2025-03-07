from typing import *
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

import zuko
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader as DL
import sklearn.preprocessing
from tqdm import *

from .utils import select_stars, make_mock_data,sam_tran,selection_func,dict_to_tensor

import random



class RQS_Flow(zuko.flows.Flow):
    def __init__(self, features: int, transforms: int, context: int = 0, hidden_features: Sequence[int] = (64, 64),flow_type = "RQS" ,bins = 8,eps=1e-3,LU= False):
        if flow_type == 'RQS':
            transform_type = zuko.transforms.MonotonicRQSTransform
            transform_shape = [(bins,), (bins,), (bins - 1,)]
        # elif flow_type == 'BP':
        #     transform_type = zuko.transforms.BoundedBernsteinTransform
        #     transform_shape = [(bins+1,)]
        transforms = [
            zuko.flows.MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features),
                hidden_features = hidden_features,
                univariate=transform_type,
                shapes=transform_shape,

            )
            for i in range(transforms)
        ]

        if LU:
            for i in reversed(range(1, len(transforms))):
                transforms.insert(
                    i,
                    zuko.flows.UnconditionalTransform(
                        zuko.transforms.LULinearTransform,
                        torch.eye(features)*(np.log(np.exp(1 - eps) - 1)),
                    ),
                )
        transforms.insert(0,zuko.flows.UnconditionalTransform(
            zuko.transforms.SoftclipTransform,
            4.0
        ))
        base = zuko.flows.UnconditionalDistribution(
            zuko.distributions.DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)






def train_flow(flow,optimizer,train_set,train_error,test_set,sector,flow_setting,n_epoch,batch_size,number_of_sam,save_path = None,group_name=None, flow_name='',device= torch.device('cpu'),
              denoise = False, amp = None,training_fraction = 0.7, pre_train = True,pre_train_epoch = 50,num_workers = 0,clip_norm = True,scheduler_setting = None,test_no_noise = False):
    if flow_setting['features'] == 3:
        tensor_order = ['v_r','v_phi','v_z']
    elif flow_setting['features'] == 4:
        tensor_order = ['v_r','v_phi','v_z','z']
    else:
        tensor_order = None

    train_set_cyl, _,_ = sam_tran(train_set,np.zeros_like(train_set),1)
    training_tensor = dict_to_tensor(train_set_cyl,tensor_order)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(training_tensor)
    scaler.scale_
    if test_set is not None:
        test_set_cyl, _ ,_= sam_tran(test_set,np.zeros_like(test_set),1)
        test_tensor = dict_to_tensor(test_set_cyl,tensor_order)

    norm_factor = 1.0
    for i in range(len(scaler.scale_)):
        norm_factor *= scaler.scale_[i]
    epoch_his = []
    pre_loss_his =[]
    pre_loss_his_test = []
    loss_his = []
    loss_his_test = []
    best_epoch = -1
    best_loss = np.inf
    best_loss_test = np.inf
    report = 'Report \n'
    report += f"flow_dim = {flow_setting['features']}"
    best_model = copy.deepcopy(flow)
    t0 = time()
    # print(f'Pre_train = {pre_train}, pre_train_epoch = {pre_train_epoch}')
    report += f'Pre_train = {pre_train}, pre_train_epoch = {pre_train_epoch}\n'
    # pre_train_size = min(int(len(train_set)/5 -1),batch_size*number_of_sam)
    pre_train_size = int(len(train_set)/5 -1)
    if pre_train:
        for epoch in tqdm(range(0,pre_train_epoch)):
            total_samples = 0
            active_samples = 0
            n_stars = 0
            loss_epoch = 0
            flow.train()
            loader = DL(list(zip(train_set,train_error)), batch_size=pre_train_size, shuffle=True, drop_last= True)

            loss_epoch = 0
            for batch_i,[data_batch, error_batch] in enumerate(loader):
                optimizer.zero_grad()
                stars_data_cyl, parallax_factor,_ = sam_tran(data_batch,torch.zeros_like(data_batch),1)
                tensor_batch = scaler.transform(dict_to_tensor(stars_data_cyl,tensor_order))
                tensor_batch = torch.tensor(tensor_batch,dtype=torch.float32).to(device)

                loss = -flow().log_prob(tensor_batch).sum()
                loss.backward()
                loss_epoch += loss.detach()
                n_stars += len(data_batch)
                optimizer.step()

            loss_epoch = (loss_epoch.item()/n_stars) + torch.log(torch.tensor(norm_factor))
            pre_loss_his.append(loss_epoch)


            flow.eval()
            loader = DL(test_set, batch_size=pre_train_size, shuffle=True, drop_last= True)
            n_stars = 0
            loss_epoch_test = 0
            for batch_i,data_batch in enumerate(loader):
                stars_data_cyl, parallax_factor,_ = sam_tran(data_batch,np.zeros_like(data_batch),1)
                tensor_batch = torch.tensor(scaler.transform(dict_to_tensor(stars_data_cyl,tensor_order)),dtype=torch.float32).to(device)

                n_stars+=len(data_batch)
                loss = -flow().log_prob(tensor_batch).sum().detach()
                loss_epoch_test += loss
            loss_epoch_test = (loss_epoch_test.cpu()/n_stars) + torch.log(torch.tensor(norm_factor))
            pre_loss_his_test.append(loss_epoch_test)

            # print(f'epoch = {epoch},pre_loss_epoch = {loss_epoch:.5f},pre_loss_epoch_test = {loss_epoch_test:.5f}')
            report += f'epoch = {epoch},pre_loss_epoch = {loss_epoch:.5f},pre_loss_epoch_test = {loss_epoch_test:.5f}\n'



        # print(f'pre train finish \n')
        report += f'pre train finish \n'
        report += f'final pre_train_loss = {loss_epoch:.5f}\n'
        # print(f'final pre_train_loss = {loss_epoch:.5f}')
        # print(f'clip norm = {clip_norm}')
        report += f'clip norm = {clip_norm}\n'
        # print(f'pre_train_size = {pre_train_size}')
        report += f'pre_train_size = {pre_train_size}\n'
        flow.eval()

        loader = DL(test_set, batch_size=pre_train_size, shuffle=True, drop_last= True)
        n_stars = 0
        loss_epoch_test = 0
        for batch_i,data_batch in enumerate(loader):
            stars_data_cyl, parallax_factor,_ = sam_tran(data_batch,np.zeros_like(data_batch),1)
            tensor_batch = torch.tensor(scaler.transform(dict_to_tensor(stars_data_cyl,tensor_order)),dtype=torch.float32).to(device)

            n_stars+=len(data_batch)
            loss = -flow().log_prob(tensor_batch).sum().detach()

            loss_epoch_test += loss

        loss_epoch_test = (loss_epoch_test.cpu()/n_stars) + torch.log(torch.tensor(norm_factor))
        print(f'final_pre_loss_epoch_test = {loss_epoch_test:.5f}')
        report += f'final_pre_loss_epoch_test = {loss_epoch_test:.5f}\n'
        print(f'LR = {optimizer.param_groups[0]["lr"]}')


        epoch_his.append(0)
        loss_his.append(loss_epoch)
        loss_his_test.append(loss_epoch_test.item())



    if scheduler_setting['type'] == 'StepLR':
        scheduler = StepLR(optimizer, step_size=scheduler_setting['step_size'], gamma=scheduler_setting['gamma'],verbose = scheduler_setting['verbose'])

    elif scheduler_setting['type'] == 'ExpLR':
        scheduler = ExponentialLR(optimizer, gamma = scheduler_setting['gamma'],verbose = scheduler_setting['verbose'])
    elif scheduler_setting['type'] == 'ReduceLR':
        scheduler = ReduceLR(optimizer, factor = scheduler_setting['factor'],
                            patience = scheduler_setting['patience'],
                            threshold=scheduler_setting['threshold'],
                            cooldown=scheduler_setting['cooldown'],
                            min_lr = scheduler_setting['min_lr'],
                            verbose = scheduler_setting['verbose'])
    else:
        scheduler = None


    # print(f'num_workers = {num_workers}')
    # print(f'batch_size = {batch_size}')
    # print(f'n_epoch = {n_epoch}')
    report += f'num_workers = {num_workers}\n'
    report += f'batch_size = {batch_size}\n'
    report += f'n_epoch = {n_epoch}\n'

    for epoch in tqdm(range(1,n_epoch+1)):



        flow.train()
        loader = DL(list(zip(train_set,train_error)), batch_size=batch_size, shuffle=True, drop_last= True,num_workers = num_workers)

        total_samples = 0
        active_sam = 0
        n_stars = 0
        loss_epoch = 0
        # n_parallax = 0
        for batch_i,[data_batch, error_batch] in enumerate(loader):
            optimizer.zero_grad()
            if test_no_noise:
                stars_data_cyl, parallax_factor,_ = sam_tran(data_batch,np.zeros_like(error_batch),number_of_sam)
            else:
                stars_data_cyl, parallax_factor,_ = sam_tran(data_batch,error_batch,number_of_sam)
            selection_factor, sam_per_stars = selection_func(stars_data_cyl,number_of_sam,number_of_sam)
            n_stars += len(data_batch)
            tensor_batch = scaler.transform(dict_to_tensor(stars_data_cyl,tensor_order))
            total_samples += len(tensor_batch)
            tensor_batch = torch.tensor(tensor_batch,dtype=torch.float32).to(device)
            parallax_factor = parallax_factor.reshape(-1,1)


            log_prob = torch.full([len(selection_factor)], -float('inf'), device = device)
            con = (((parallax_factor)*selection_factor.numpy()) == 1).T.squeeze()

            # flow.abcsd = [log_prob,tensor_batch,con]
            log_prob[con] = flow().log_prob(tensor_batch[con])
            loss = (torch.nan_to_num(-log_prob.reshape(-1,number_of_sam).logsumexp(dim=1), posinf=0.0) +
                    torch.log(sam_per_stars.to(device))).sum()
                                                    #.mean() + torch.log(torch.tensor(self.norm_factor))

            loss.backward()
            loss_epoch += loss.detach()
            # n_parallax += parallax_factor.sum()
            active_sam += con.sum()
            if clip_norm:
                norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), 100.0)
            # print(f"norm = {norm}")
            optimizer.step()
        loss_epoch = (loss_epoch.item()/n_stars) + torch.log(torch.tensor(norm_factor))
        # print(f'n_stars = {n_stars}')
        # print(f'n_parallax = {n_parallax}')
        # print(f'active_sam = {active_sam}')
        # print(f'reject_sam = {number_of_sam*n_stars - active_sam}')



        flow.eval()
        loss_epoch_test = 0
        test_size = min(int(len(test_set)/5 -1 ),batch_size*16)
        loader = DL(test_set, batch_size=test_size, shuffle=True, drop_last= True)
        n_stars = 0

        for batch_i,data_batch in enumerate(loader):
            stars_data_cyl, parallax_factor,_ = sam_tran(data_batch,np.zeros_like(data_batch),1)
            tensor_batch = torch.tensor(scaler.transform(dict_to_tensor(stars_data_cyl,tensor_order)),dtype=torch.float32).to(device)

            n_stars+=len(data_batch)
            loss = -flow().log_prob(tensor_batch).sum().detach()

            loss_epoch_test += loss

        loss_epoch_test = (loss_epoch_test.cpu()/n_stars) + torch.log(torch.tensor(norm_factor))

        # print(f'loss_epoch = {loss_epoch:.5f}')
        # print(f'loss_epoch_test = {loss_epoch_test:.5f}')


        if loss_epoch < best_loss:
            best_loss = loss_epoch
            best_epoch = epoch
            best_loss_test = loss_epoch_test
            best_model = copy.deepcopy(flow.state_dict())
        epoch_his.append(epoch)
        loss_his.append(loss_epoch)
        loss_his_test.append(loss_epoch_test.item())
        report += f'epoch = {epoch}\n'
        report += f'loss_epoch = {loss_epoch:.5f}\n'
        report += f'loss_epoch_test = {loss_epoch_test:.5f}\n'





        if scheduler:
            scheduler.step()
            # print(f'LR = {optimizer.param_groups[0]["lr"]}')
            report += f'LR = {optimizer.param_groups[0]["lr"]}'

    path_prefix = save_path
    if group_name:
        path_prefix = os.path.join(path_prefix,group_name)
    dt = datetime.now().strftime("%m%d%Y_%H%M%S")
    path = os.path.join(path_prefix, flow_name+dt)
    Path(path).mkdir(parents=True, exist_ok=True)

    model_parameters ={
        'best_model': flow.state_dict(),
        'denoise': denoise,
        'amp': amp,
        'training_fraction': training_fraction,
        'flow_setting':flow_setting,
        'num_workers':num_workers,
        'number_of_sam':number_of_sam,
        'pre_train' : pre_train,
        'pre_train_epoch': pre_train_epoch,
        'clip_norm' : clip_norm,
        'scheduler_setting' : scheduler_setting,
        'test_no_noise' : test_no_noise,
        'scaler_mean' : scaler.mean_,
        'scaler_scale' : scaler.scale_,
        'sector' : sector,
        'epoch_his': epoch_his,
        'loss_his': loss_his,
        'loss_his_test': loss_his_test,
        'pre_loss_his': pre_loss_his,
        'pre_loss_his_test': pre_loss_his_test,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'best_loss_test':best_loss_test,
        'path' : path
        }


    for key, value in sector.items():
        report += f'{key} : {value}\n'

    for key, value in flow_setting.items():
        report += f'{key} : {value}\n'

    report += f'scaler_mean = {scaler.mean_}\n'
    report += f'scaler_scale = {scaler.scale_}\n'

    report += f'denoise = {denoise}\n'
    report += f'amp = {amp}\n'
    report += f'number_of_sample = {number_of_sam}\n'
    report += f'training_fraction = {training_fraction}\n'
    report += f'test_no_noise = {test_no_noise}\n'
    report += f'best_epoch = {best_epoch}\n'
    report += f'best_loss = {best_loss}\n'
    report += f'best_loss_test = {best_loss_test}\n'
    with open(os.path.join(path, "report.md"), "a") as report_file:
          report_file.write(report)
    torch.save(model_parameters, os.path.join(path,f'model_parameters.pth'))

    flow.eval()
    return model_parameters


def load_flow(state_dict, flow_setting,device):
    flow = RQS_Flow(**flow_setting).to(device)
    flow.load_state_dict(state_dict)
    flow.eval()
    return flow


def sample_from_flow(flow,scaler,num_samples,batch_size):
    flow.eval()
    torch.cuda.empty_cache()
    num_batches = num_samples // batch_size
    num_leftover = num_samples % batch_size
    samples = [flow().sample((batch_size,)).cpu() for _ in range(num_batches)]
    if num_leftover > 0:
        samples.append(flow().sample((num_leftover,)).cpu())
    return scaler.inverse_transform(torch.cat(samples, dim=0))


def load_and_gen_sample(path):
    model_parameters = torch.load(os.path.join(path,'model_parameters.pth'))

    print(model_parameters['best_epoch'])
    print(model_parameters['best_loss'])
    print(model_parameters['best_loss_test'])


    scaler = sklearn.preprocessing.StandardScaler()
    scaler.mean_ = model_parameters['scaler_mean']
    scaler.scale_ = model_parameters['scaler_scale']





    gaia_data, sector = select_stars(QTable.read('gdrive/MyDrive/Colab Notebooks/normalizing_flow_gaia/gaiadr3_rv.fits')
                    ,model_parameters['sector']
                    )

    gaia_stars = np.array([
        gaia_data['ra'].value,
        gaia_data['dec'].value,
        gaia_data['parallax'].value,
        gaia_data['pmra'].value,
        gaia_data['pmdec'].value,
        gaia_data['radial_velocity'].value
        ]).T

    gaia_error = np.array([
        np.zeros_like(gaia_data['ra'].value),
        np.zeros_like(gaia_data['ra'].value),
        gaia_data['parallax_error'].value,
        gaia_data['pmra_error'].value,
        gaia_data['pmdec_error'].value,
        gaia_data['radial_velocity_error'].value,
        ]).T

    mock_data, mock_error, test_data,training_data = make_mock_data(gaia_stars,gaia_error,model_parameters['amp'])


    flow = load_flow(model_parameters['best_model'],model_parameters['flow_setting'])
    if model_parameters['flow_setting']['features'] == 3:
        tensor_order = ['v_r','v_phi','v_z']
    else:
        tensor_order = None

    train_cyl, _,train_object = sam_tran(training_data,np.zeros_like(training_data),1)
    train_cyl = dict_to_tensor(train_cyl,tensor_order)



    mock_cyl, _,mock_object = sam_tran(mock_data,np.zeros_like(mock_data),1)
    mock_cyl = dict_to_tensor(mock_cyl,tensor_order)


    flow.eval()
    flow_sam = sample_from_flow(flow,scaler,len(train_cyl),2**16)

    return {
        'model_parameters' : model_parameters,
        'train_cyl' : train_cyl,
        'mock_cyl' : mock_cyl,
        'flow_sam' : flow_sam
    }