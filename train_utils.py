"""
Some functions and classes we want to use during training.
"""

import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim

import startup
import settings
from data_loader import DataModes

class Loss(nn.MSELoss) :
    """
    simply change the base class (and possibly its __init__ args)
    for different functionality

    The D3M paper (which has a somewhat similar objective) uses the MSELoss.

    Note that it may be useful to include some dependency on delta_L,
    since the particles in high delta_L move a lot further than in low delta_L
    and this may cause the network to put most of its efforts towards getting the
    high delta_L right. (on the other hand, it may be argued that if it corectly
    handles high delta_L, lower values should be no problem...)
    This needs to be tested.
    """
#{{{
    def __init__(self) :
        super().__init__()
#}}}

class Optimizer(torch.optim.Adam) :
    """
    simply change the base class (and possibly its __init__ args)
    for different functionality
    """
#{{{
    def __init__(self, params) :
        super().__init__(params, **settings.OPTIMIZER_ARGS)
#}}}

def do_diagnostic_output(training_loss, validation_loss,
                         training_loss_guess, validation_loss_guess,
                         Nepochs, epoch_len) :
    """
    saves the various losses to disk
    """
#{{{
    training_times = np.linspace(0, Nepochs, num=Nepochs*epoch_len).repeat(settings.WORLD_SIZE)
    validation_times = np.linspace(1, Nepochs, num=Nepochs).repeat(settings.WORLD_SIZE)

    np.savez(settings.LOSS_FILE, training_times=training_times, validation_times=validation_times,
                                 training_loss=training_loss,
                                 validation_loss=validation_loss,
                                 training_loss_guess=training_loss_guess,
                                 validation_loss_guess=validation_loss_guess)
#}}}


def save_model(model, optimizer) :
    """
    this function will only be called from the rank=0 process

    It serializes the model and the optimizer state.
    """
#{{{
    assert settings.RANK == 0
    torch.save(dict(model_state_dict=model.module.state_dict(),
                    optimizer_state_dict=optimizer.state_dict()),
               settings.MODEL_FILE)
#}}}


def load_model(model, optimizer=None) :
    """
    This function will be called from any process, we need to make sure we map
    the tensors to the correct devices

    It can also be used for inference, in which case the second argument is not passed
    """
#{{{
    try :
        # NOTE : since only the RANK=0 process writes to file, we can safely assume that all CUDA tensors
        #        were written from the cuda:0 device
        checkpoint = torch.load(settings.MODEL_FILE, map_location={'cuda:0': 'cuda:%d'%settings.DEVICE_IDX})
        if settings.RANK == 0 :
            print('Found saved model, continuing from there.')
    except FileNotFoundError :
        if settings.RANK == 0 :
            print('No saved model found, continuing with freshly initialized one.')
        return

    model.module.load_state_dict(checkpoint['model_state_dict'], strict=not settings.WARMSTART)
    
    if optimizer is not None :
        if settings.RANK == 0 :
            print('Loading optimizer state dict (lr, betas are updated if necessary).')
        opt_state_dict = checkpoint['optimizer_state_dict']
        # there are some aspects of this state that we may want to alter:
        opt_state_dict['param_groups'][0]['lr'] = settings.OPTIMIZER_ARGS['lr']
        opt_state_dict['param_groups'][0]['betas'] = settings.OPTIMIZER_ARGS['betas']
        
        try :
            optimizer.load_state_dict(opt_state_dict)
        except ValueError :
            if settings.WARMSTART :
                if settings.RANK == 0 :
                    print('Could not load optimizer state from disk as for a different model, default initializing')
            else :
                raise ValueError('saved optimizer not compatible with model, and settings.WARMSTART=False.')

    else :
        if settings.RANK == 0 :
            print('Not loading optimizer state as not requested.')
#}}}

def load_loss() :
    """
    Tries to load existing loss curves from disk so we get a nice continous
    loss curve even if we have to interrupt training at some point.
    """
#{{{
    assert settings.RANK == 0

    try :
        with np.load(settings.LOSS_FILE) as f :
            start_epoch = int(f['validation_times'][-1])
            training_loss = f['training_loss']
            validation_loss = f['validation_loss']
            training_loss_guess = f['training_loss_guess']
            validation_loss_guess = f['validation_loss_guess']
        print('starting at epoch %d'%start_epoch)
        return start_epoch, training_loss, validation_loss, training_loss_guess, validation_loss_guess
    except OSError :
        return 0, np.empty(0), np.empty(0), np.empty(0), np.empty(0)
#}}}


def setup_process(rank) :
    """
    to be called at the beginning of a child process
    rank passed is the local rank

    This function establishes the distributed environment for pytorch.
    """
#{{{

    # new process needs to get a consistent view of the settings
    startup.main(DataModes.TRAINING, rank)

    os.environ['MASTER_ADDR'] = settings.MASTER_ADDR
    os.environ['MASTER_PORT'] = settings.MASTER_PORT

    # note : the `rank' kw here is NOT equal the `rank' argument, since it refers
    #        to the entire world
    torch.distributed.init_process_group('nccl',
                                         rank=settings.RANK,
                                         world_size=settings.WORLD_SIZE)

    # the global pytorch switch -- not sure if it's important here but cannot hurt.
    torch.cuda.set_device(settings.DEVICE_IDX)
#}}}


def cleanup_process() :
    """
    to be called at the end of a child process
    """
#{{{
    torch.distributed.destroy_process_group()
#}}}
