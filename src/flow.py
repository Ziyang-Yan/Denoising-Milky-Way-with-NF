from typing import *
import numpy as np
import copy
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
        # elif flow_type == 'MAF':
        #     transform_type = zuko.transforms.AffineAutoregressiveTransform  # MAF
        #     transform_shape = [(features,), (features,)]
        else:
            raise ValueError(f"Unknown flow_type: {flow_type}")
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






def pre_train_one_epoch(
    flow,
    optimizer,
    train_set,
    train_error,
    pre_train_size,
    device,
    norm_factor,
    tensor_order,
    scaler
):
    """
    Run one epoch of *pre-training* on 'flow' given training data and errors.
    Return the average negative log likelihood loss (plus log(norm_factor)).
    """
    flow.train()
    loader = DL(
        list(zip(train_set, train_error)), 
        batch_size=pre_train_size, 
        shuffle=True, 
        drop_last=True
    )
    
    loss_epoch_val = 0.0
    n_stars = 0
    
    for _, (data_batch, error_batch) in enumerate(loader):
        optimizer.zero_grad()
        
        # For pre-training, we typically don't add noise: use torch.zeros_like(...)
        stars_data_cyl, _, _ = sam_tran(data_batch, torch.zeros_like(data_batch), 1)
        tensor_batch = scaler.transform(dict_to_tensor(stars_data_cyl, tensor_order))
        tensor_batch = torch.tensor(tensor_batch, dtype=torch.float32).to(device)

        loss = -flow().log_prob(tensor_batch).sum()
        loss.backward()
        loss_epoch_val += loss.detach().item()
        n_stars += len(data_batch)

        optimizer.step()

    # Normalize by number of stars and add log(norm_factor)
    loss_epoch = (loss_epoch_val / n_stars) + torch.log(torch.tensor(norm_factor))
    return loss_epoch.item()


def pre_train_one_epoch_test(
    flow,
    test_set,
    pre_train_size,
    device,
    norm_factor,
    tensor_order,
    scaler
):
    """
    Evaluate the log probability on the test set (single epoch of test for pre-training).
    Return the average negative log likelihood (plus log(norm_factor)).
    """
    flow.eval()
    loader = DL(test_set, batch_size=pre_train_size, shuffle=True, drop_last=True)

    loss_epoch_test_val = 0.0
    n_stars = 0

    for _, data_batch in enumerate(loader):
        # Also use no noise here, consistent with pre-training approach
        stars_data_cyl, _, _ = sam_tran(data_batch, torch.zeros_like(data_batch), 1)
        tensor_batch = scaler.transform(dict_to_tensor(stars_data_cyl, tensor_order))
        tensor_batch = torch.tensor(tensor_batch, dtype=torch.float32).to(device)

        loss = -flow().log_prob(tensor_batch).sum().detach()
        loss_epoch_test_val += loss.item()
        n_stars += len(data_batch)

    loss_epoch_test = (loss_epoch_test_val / n_stars) + torch.log(torch.tensor(norm_factor))
    return loss_epoch_test.item()


def train_one_epoch(
    flow,
    optimizer,
    train_set,
    train_error,
    batch_size,
    num_workers,
    device,
    norm_factor,
    tensor_order,
    scaler,
    clip_norm,
    number_of_sam,
    test_no_noise
):
    """
    Run one epoch of *main training* on 'flow'.
    Return the average negative log likelihood (plus log(norm_factor)).
    """
    flow.train()
    
    loader = DL(
        list(zip(train_set, train_error)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    loss_epoch_val = 0.0
    n_stars = 0

    for _, (data_batch, error_batch) in enumerate(loader):
        optimizer.zero_grad()

        # Decide whether or not to use noise
        if test_no_noise:
            stars_data_cyl, parallax_factor, _ = sam_tran(
                data_batch, torch.zeros_like(error_batch), number_of_sam
            )
        else:
            stars_data_cyl, parallax_factor, _ = sam_tran(
                data_batch, error_batch, number_of_sam
            )

        # Evaluate selection function
        selection_factor, sam_per_stars = selection_func(
            stars_data_cyl, number_of_sam, number_of_sam
        )
        n_stars += len(data_batch)

        # Scale inputs
        tensor_batch = scaler.transform(dict_to_tensor(stars_data_cyl, tensor_order))
        tensor_batch = torch.tensor(tensor_batch, dtype=torch.float32).to(device)
        parallax_factor = parallax_factor.reshape(-1, 1)

        # We only compute log probability where con is True
        log_prob = torch.full([len(selection_factor)], -float('inf'), device=device)
        con = ((parallax_factor * selection_factor.numpy()) == 1).T.squeeze()

        log_prob[con] = flow().log_prob(tensor_batch[con])

        loss = (
            torch.nan_to_num(
                -log_prob.reshape(-1, number_of_sam).logsumexp(dim=1),
                posinf=0.0
            )
            + torch.log(sam_per_stars.to(device))
        ).sum()

        loss.backward()
        loss_epoch_val += loss.detach().item()

        if clip_norm:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 100.0)

        optimizer.step()

    # Normalize by number of stars and add log(norm_factor)
    loss_epoch = (loss_epoch_val / n_stars) + torch.log(torch.tensor(norm_factor))
    return loss_epoch.item()


def test_one_epoch(
    flow,
    test_set,
    batch_size,
    device,
    norm_factor,
    tensor_order,
    scaler
):
    """
    Run a single epoch of testing on 'flow'.
    Return the average negative log likelihood (plus log(norm_factor)).
    """
    flow.eval()
    loader_test = DL(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    loss_epoch_test_val = 0.0
    n_stars_test = 0

    for _, data_batch in enumerate(loader_test):
        stars_data_cyl, _, _ = sam_tran(data_batch, np.zeros_like(data_batch), 1)
        tensor_batch = scaler.transform(dict_to_tensor(stars_data_cyl, tensor_order))
        tensor_batch = torch.tensor(tensor_batch, dtype=torch.float32).to(device)

        loss_t = -flow().log_prob(tensor_batch).sum().detach()
        loss_epoch_test_val += loss_t.item()
        n_stars_test += len(data_batch)

    loss_epoch_test = (loss_epoch_test_val / n_stars_test) + torch.log(torch.tensor(norm_factor))
    return loss_epoch_test.item()


################################################################################
# Main Training Function
################################################################################

def train_flow(
    flow,
    optimizer,
    train_set,
    train_error,
    test_set,
    sector,
    flow_setting,
    n_epoch,
    batch_size,
    number_of_sam,
    save_path=None,
    group_name=None,
    flow_name='',
    device=torch.device('cpu'),
    denoise=False,
    amp=None,
    training_fraction=0.7,
    pre_train=True,
    pre_train_epoch=50,
    pre_train_size=None,
    num_workers=0,
    clip_norm=True,
    scheduler_setting=None,
    test_no_noise=False
):
    """
    Main function orchestrating:
      - (Optionally) Pre-training
      - Main training epochs
      - (Optionally) Scheduling
      - Saving best model & metadata

    Dependencies (assumed to exist):
      - sam_tran(data_batch, error_batch, number_of_sam)
      - dict_to_tensor(stars_data_cyl, tensor_order)
      - selection_func(stars_data_cyl, number_of_sam1, number_of_sam2)
    """


    if flow_setting['features'] == 3:
        tensor_order = ['v_r','v_phi','v_z']
    elif flow_setting['features'] == 4:
        tensor_order = ['v_r','v_phi','v_z','z']
    else:
        tensor_order = None


    train_set_cyl, _, _ = sam_tran(train_set, np.zeros_like(train_set), 1)
    training_tensor = dict_to_tensor(train_set_cyl, tensor_order)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(training_tensor)

    if test_set is not None:
        test_set_cyl, _, _ = sam_tran(test_set, np.zeros_like(test_set), 1)
        test_tensor = dict_to_tensor(test_set_cyl, tensor_order)

    # Compute product of the scaling factors
    norm_factor = 1.0
    for s_val in scaler.scale_:
        norm_factor *= s_val

    # Bookkeeping
    epoch_his = []
    pre_loss_his = []
    pre_loss_his_test = []
    loss_his = []
    loss_his_test = []
    best_epoch = -1
    best_loss = np.inf
    best_loss_test = np.inf
    best_model = copy.deepcopy(flow.state_dict())

    report = 'Report \n'
    report += f"flow_dim = {flow_setting['features']}\n"
    t0 = time()


    report += f'Pre_train = {pre_train}, pre_train_epoch = {pre_train_epoch}\n'

    
    if pre_train:
        if pre_train_size is None:
            pre_train_size = int(len(train_set)/5 - 1)
        for epoch in tqdm(range(pre_train_epoch)):
            loss_epoch = pre_train_one_epoch(
                flow=flow,
                optimizer=optimizer,
                train_set=train_set,
                train_error=train_error,
                pre_train_size=pre_train_size,
                device=device,
                norm_factor=norm_factor,
                tensor_order=tensor_order,
                scaler=scaler
            )
            loss_epoch_test = pre_train_one_epoch_test(
                flow=flow,
                test_set=test_set,
                pre_train_size=pre_train_size,
                device=device,
                norm_factor=norm_factor,
                tensor_order=tensor_order,
                scaler=scaler
            )

            pre_loss_his.append(loss_epoch)
            pre_loss_his_test.append(loss_epoch_test)
            report += (f"Pre-train Epoch {epoch}: "
                       f"pre_loss = {loss_epoch:.5f}, "
                       f"pre_loss_test = {loss_epoch_test:.5f}\n")

        report += f"Pre-train finished.\n"
        report += f"Final pre_train_loss = {loss_epoch:.5f}\n"
        flow.eval()

        # Evaluate on test one more time after pre-training
        loss_epoch_test = pre_train_one_epoch_test(
            flow=flow,
            test_set=test_set,
            pre_train_size=pre_train_size,
            device=device,
            norm_factor=norm_factor,
            tensor_order=tensor_order,
            scaler=scaler
        )
        report += f"Final pre_train_loss_test = {loss_epoch_test:.5f}\n"

        # We append a "dummy" epoch index = 0 to keep consistent with plotting
        epoch_his.append(0)
        loss_his.append(loss_epoch)
        loss_his_test.append(loss_epoch_test)


    if scheduler_setting is None:
        scheduler_setting = {'type': None}  # In case it's omitted
        scheduler = None
    else:
        if scheduler_setting['type'] == 'StepLR':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_setting['step_size'],
                gamma=scheduler_setting['gamma'],
                verbose=scheduler_setting['verbose']
            )
        elif scheduler_setting['type'] == 'ExpLR':
            scheduler = ExponentialLR(
                optimizer,
                gamma=scheduler_setting['gamma'],
                verbose=scheduler_setting['verbose']
            )
        elif scheduler_setting['type'] == 'ReduceLR':
            scheduler = ReduceLR(
                optimizer,
                factor=scheduler_setting['factor'],
                patience=scheduler_setting['patience'],
                threshold=scheduler_setting['threshold'],
                cooldown=scheduler_setting['cooldown'],
                min_lr=scheduler_setting['min_lr'],
                verbose=scheduler_setting['verbose']
            )
        else:
            scheduler = None

    report += f'num_workers = {num_workers}\n'
    report += f'batch_size = {batch_size}\n'
    report += f'n_epoch = {n_epoch}\n'

    for epoch in tqdm(range(1, n_epoch + 1)):
        # Single training epoch
        loss_epoch = train_one_epoch(
            flow=flow,
            optimizer=optimizer,
            train_set=train_set,
            train_error=train_error,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            norm_factor=norm_factor,
            tensor_order=tensor_order,
            scaler=scaler,
            clip_norm=clip_norm,
            number_of_sam=number_of_sam,
            test_no_noise=test_no_noise
        )

        # Single testing epoch
        test_size = min(int(len(test_set)/5 - 1), batch_size * 16)
        loss_epoch_test = test_one_epoch(
            flow=flow,
            test_set=test_set,
            batch_size=test_size,
            device=device,
            norm_factor=norm_factor,
            tensor_order=tensor_order,
            scaler=scaler
        )

        # Check for best model
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            best_epoch = epoch
            best_loss_test = loss_epoch_test
            best_model = copy.deepcopy(flow.state_dict())

        epoch_his.append(epoch)
        loss_his.append(loss_epoch)
        loss_his_test.append(loss_epoch_test)

        report += f"Epoch = {epoch}\n"
        report += f"loss_epoch = {loss_epoch:.5f}\n"
        report += f"loss_epoch_test = {loss_epoch_test:.5f}\n"

        # Step the scheduler if any
        if scheduler:
            scheduler.step()
            report += f"LR = {optimizer.param_groups[0]['lr']}\n"

    path_prefix = save_path
    if group_name:
        path_prefix = os.path.join(path_prefix, group_name)
    dt = datetime.now().strftime("%m%d%Y_%H%M%S")
    path = os.path.join(path_prefix, flow_name + dt)
    Path(path).mkdir(parents=True, exist_ok=True)

    model_parameters = {
        'best_model': flow.state_dict(),
        'denoise': denoise,
        'amp': amp,
        'training_fraction': training_fraction,
        'flow_setting': flow_setting,
        'num_workers': num_workers,
        'number_of_sam': number_of_sam,
        'pre_train': pre_train,
        'pre_train_epoch': pre_train_epoch,
        'pre_train_size': pre_train_size,
        'clip_norm': clip_norm,
        'scheduler_setting': scheduler_setting,
        'test_no_noise': test_no_noise,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'sector': sector,
        'epoch_his': epoch_his,
        'loss_his': loss_his,
        'loss_his_test': loss_his_test,
        'pre_loss_his': pre_loss_his,
        'pre_loss_his_test': pre_loss_his_test,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'best_loss_test': best_loss_test,
        'path': path
    }

    # Append sector info
    for key, value in sector.items():
        report += f'{key} : {value}\n'
    # Append flow_setting info
    for key, value in flow_setting.items():
        report += f'{key} : {value}\n'

    # Scaler and other info
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

    # Write the report out
    with open(os.path.join(path, "report.md"), "a") as report_file:
        report_file.write(report)

    # Save parameters
    torch.save(model_parameters, os.path.join(path, 'model_parameters.pth'))

    # Final eval mode
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


def load_and_gen_sample(model_path,data_path):
    model_parameters = torch.load(os.path.join(model_path,'model_parameters.pth'))

    print(model_parameters['best_epoch'])
    print(model_parameters['best_loss'])
    print(model_parameters['best_loss_test'])


    scaler = sklearn.preprocessing.StandardScaler()
    scaler.mean_ = model_parameters['scaler_mean']
    scaler.scale_ = model_parameters['scaler_scale']





    gaia_data, sector = select_stars(QTable.read(data_path)
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