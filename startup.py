from sys import argv
import os
import os.path
import warnings
import tempfile
import argparse
import numpy as np

import torch

import settings
from settings import ToSet
import data_loader # import whole module to avoid circular import
import simulation_run # import whole module to avoid circular import

class ArgParser :
    """
    upon construction, this will parse the command line.
    Add arguments that should be parsed to the `populate_parser' method.
    """
#{{{
    def populate_parser(self) :

        self.parser.add_argument('--id', type=str,
                                 help='settings.ID : string, identifies data associated with this run')
        self.parser.add_argument('--batchsize', type=int,
                                 help='settings.BATCH_SIZE : integer, real batch size is times num_GPUs')
        self.parser.add_argument('--naugment', type=int,
                                 help='settings.N_AUGMENTATIONS : integer, number of augmentations per epoch')


    def __init__(self) :

        # construct the argument parser
        self.parser = argparse.ArgumentParser()

        # load the switches we would like to parse
        self.populate_parser()


    def parse_and_set(self) :

        args = self.parser.parse_args()

        if hasattr(args, 'id') :
            settings.ID = settings.ID.set(args.id)
        else :
            # this is a special case, as we need the ID for other things
            settings.ID = settings.ID.set()
        if hasattr(args, 'batchsize') :
            settings.BATCH_SIZE = settings.BATCH_SIZE.set(args.batchsize)
        if hasattr(args, 'naugment') :
            settings.N_AUGMENTATIONS = settings.N_AUGMENTATIONS.set(args.naugment)
#}}}


def populate_settings_from_cl() :
    """
    populates some settings that can be done directly from the command line
    """
#{{{
    args = ArgParser()
    args.parse_and_set()
#}}}


def load_normalizations() :
    """
    populates the normalization function dicts
    """
#{{{
    # note that we have to *capture* in the lambda,
    # otherwise we'll always get the same output
    # regardless of mode

    settings.DENSITY_NORMALIZATIONS \
        = settings.DENSITY_NORMALIZATIONS.set(dict())
    settings.DISPLACEMENT_NORMALIZATIONS \
        = settings.DISPLACEMENT_NORMALIZATIONS.set(dict())
    with np.load(settings.NORMALIZATION_FILE) as f :
        for mode in data_loader.DataModes :
            sigma_displacement, sigma_density, A, B = f[str(mode)]
            settings.DENSITY_NORMALIZATIONS[mode] \
                = lambda x, s=sigma_density, a=A, b=B : a * ( np.log1p(x/s) - b )
            settings.DISPLACEMENT_NORMALIZATIONS[mode] \
                = lambda x, s=sigma_displacement : x/s

    settings.DELTA_L_NORMALIZATIONS \
        = settings.DELTA_L_NORMALIZATIONS.set(dict())
    for mode in data_loader.DataModes :
        run_pairs = simulation_run.get_runs(mode)
        delta_L_arr = np.array([run_pair[1].delta_L for run_pair in run_pairs])
        mean = np.mean(delta_L_arr)
        stddev = np.std(delta_L_arr)
        settings.DELTA_L_NORMALIZATIONS[mode] \
            = lambda x, m=mean, s=stddev : (x-m) / s
#}}}


def set_mp_env(rank) :
    """
    sets all variables referring to multiprocessing which can be inferred
    without knowledge of a specific rank

    and if rank is not None also the others

    rank refers to the `local rank', i.e. the one within a specific MPI process
    """
#{{{
    try :
        settings.MPI_WORLD_SIZE = settings.MPI_WORLD_SIZE.set(int(os.environ['OMPI_COMM_WORLD_SIZE']))
        settings.MPI_RANK = settings.MPI_RANK.set(int(os.environ['OMPI_COMM_WORLD_RANK']))
        # note that this one should go last
        settings.MPI_NODENAME = settings.MPI_NODENAME.set(os.environ['HOSTNAME'])
    except KeyError :
        settings.MPI_WORLD_SIZE = settings.MPI_WORLD_SIZE.set()
        settings.MPI_RANK = settings.MPI_RANK.set()
        settings.MPI_NODENAME = settings.MPI_NODENAME.set()

    if settings.MPI_WORLD_SIZE != 1 :
        # we have multiple nodes, need to figure out which one is the root
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        assert comm.Get_rank() == settings.MPI_RANK
        if settings.MPI_RANK == 0 :
            root_name = settings.MPI_NODENAME
        else :
            root_name = None
        root_name = comm.bcast(root_name, root=0)
        settings.MASTER_ADDR = settings.MASTER_ADDR.set(root_name)

    # we assume that each process has the same number of GPUs available
    settings.NUM_GPUS = settings.NUM_GPUS.set(torch.cuda.device_count())
    settings.WORLD_SIZE = settings.WORLD_SIZE.set(settings.NUM_GPUS * settings.MPI_WORLD_SIZE)

    if rank is not None :
        # we are in a specific process
        settings.LOCAL_RANK = settings.LOCAL_RANK.set(rank)
        settings.RANK = settings.RANK.set(settings.MPI_RANK * settings.NUM_GPUS + settings.LOCAL_RANK)
        settings.DEVICE_IDX = settings.DEVICE_IDX.set(settings.RANK % settings.NUM_GPUS)
        assert settings.LOCAL_RANK == settings.DEVICE_IDX

    # ID must have been set before!
    settings.SHARE_FILE = settings.SHARE_FILE.set('/scratch/gpfs/lthiele/torch_share_files/'
                                                  + settings.ID)

    if settings.VERBOSE and rank is not None :
        # give some diagnostic output
        print(f'On NODENAME {settings.MPI_NODENAME}: '\
              f'On MPI_RANK {settings.MPI_RANK+1} / {settings.MPI_WORLD_SIZE}: '\
              f'On LOCAL_RANK {settings.LOCAL_RANK+1} / {settings.NUM_GPUS}: '\
              f'RANK = {settings.RANK}, '\
              f'DEVICE_IDX = {settings.DEVICE_IDX}, '\
              f'MASTER_ADDR = {settings.MASTER_ADDR}')
#}}}


def set_filenames() :
    """
    populates the input/output filenames
    """
#{{{
    settings.LOSS_FILE = settings.LOSS_FILE.set(os.path.join(settings.RESULTS_PATH, 'loss_%s.npz'%settings.ID))
    settings.MODEL_FILE = settings.MODEL_FILE.set(os.path.join(settings.RESULTS_PATH, 'model_%s.pt'%settings.ID))
#}}}


def set_remaining() :
    """
    set ToSet instances to their default values if they have not already been set
    """
#{{{
    for name in dir(settings) :
        if name.startswith('__') :
            # this is an internal variable that we don't care about
            continue
        this_setting = settings.__dict__[name]

        if isinstance(this_setting, ToSet) :
            settings.__dict__[name] = this_setting.set()

        try :
            # dict-like
            for k, v in this_setting.items() :
                if isinstance(v, ToSet) :
                    this_setting[k] = v.set()
        except AttributeError :
            pass

        try :
            # iterable
            for ii, v in enumerate(this_setting) :
                if isinstance(v, ToSet) :
                    this_setting[ii] = v.set()
        except TypeError :
            pass

        try :
            # custom class
            for k, v in vars(this_setting).items() :
                if k.startswith('__') :
                    continue
                if isinstance(v, ToSet) :
                    this_setting.__dict__[k] = v.set()
        except TypeError :
            pass
#}}}


def main(mode, rank=None) :
    """
    this is the only function that should be called from this module.
    `mode' refers to the global mode of execution.

    `rank' is the process-local rank (i.e. within an MPI process)
    """
    assert isinstance(mode, data_loader.DataModes)

    assert not settings.STARTUP_CALLED

    # tell anyone using the settings module that it
    # is in the correct state
    # (all global variables have been updated)
    settings.STARTUP_CALLED = True

    # this should be called right at the beginning so subsequent
    # action can make use of this information if necessary
    # and we don't have to pass the `mode' argument around
    settings.MODE = settings.MODE.set(mode)

    populate_settings_from_cl()
    set_mp_env(rank)
    set_filenames()
    load_normalizations()
    set_remaining()
