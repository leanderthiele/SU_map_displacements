"""
The `main' function in this module populates the `settings' module
with global variables.

The most important thing is that main needs to be called whenever a new
process starts up since python does not consistently copy state into
forked/spawned processes.
"""


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
        self.parser.add_argument('--nodensity', action='store_true',
                                 help='~ settings.USE_DENSITY : disables the use of the density field as an additional channel')
        self.parser.add_argument('--residualconcat', action='store_true',
                                 help='~ settings.RESIDUAL_ADD : enables concatenation instead of addition')
        self.parser.add_argument('--skipconcat', action='store_true',
                                 help='~ settings.SKIP_ADD : enables concatenation instead of addition')
        self.parser.add_argument('--batchsize', type=int,
                                 help='settings.BATCH_SIZE : integer, real batch size is times num_GPUs')
        self.parser.add_argument('--naugment', type=int,
                                 help='settings.N_AUGMENTATIONS : integer, number of augmentations per epoch')
        self.parser.add_argument('--nlayers', type=int,
                                 help='settings.DEFAULT_NLAYERS : integer, typical number of layers in a block')
        self.parser.add_argument('--nlevels', type=int,
                                 help='settings.NLEVELS : integer, number of UNet levels')
        self.parser.add_argument('--learningrate', type=float,
                                 help='settings.OPTIMIZER_ARGS[lr] : float, learning rate')
        self.parser.add_argument('--deltaLmax', type=float,
                                 help='settings.DELTA_L_BOUNDS[1]: float, maximum target Delta_L')


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
        if hasattr(args, 'nodensity') :
            settings.USE_DENSITY = settings.USE_DENSITY.set(not args.nodensity)
        if hasattr(args, 'nlayers') :
            settings.DEFAULT_NLAYERS = settings.DEFAULT_NLAYERS.set(args.nlayers)
        if hasattr(args, 'nlevels') :
            settings.NLEVELS = settings.NLEVELS.set(args.nlevels)
        if hasattr(args, 'learningrate') :
            settings.OPTIMIZER_ARGS['lr'] = settings.OPTIMIZER_ARGS['lr'].set(args.learningrate)
        if hasattr(args, 'residualconcat') :
            settings.RESIDUAL_ADD = settings.RESIDUAL_ADD.set(not args.residualconcat)
        if hasattr(args, 'skipconcat') :
            settings.SKIP_ADD = settings.SKIP_ADD.set(not args.skipconcat)
        if hasattr(args, 'deltaLmax') :
            settings.DELTA_L_BOUNDS[1] = settings.DELTA_L_BOUNDS[1].set(args.deltaLmax)
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


def set_mp_env() :
    """
    sets all variables referring to multiprocessing which can be inferred
    without knowledge of a specific rank

    We write this function flexibly enough to accomodate the following cases:

        1) we are seeing 1 GPU (in that case, our life is easy)
           SINGLEGPU
                In that case, we assign this GPU to our rank
                and launch training without spawning
                --> assert rank == 0 if set

        2) we are seeing multiple GPUs, and there are multiple ranks on our node
           MULTIGPU_MULTIRANK
                In that case, we assign each rank one of the GPUs
                and launch training without spawning
                --> assert rank == 0 if set

        3) we are seeing multiple GPUs, and there is one rank on our node
           MULTIGPU_SINGLERANK
                In that case, we launch training with spawning
                and assign each sub-process one of the GPUs
           [this case may be impossible to run on multiple nodes because OpenMPI does not like
            it if we fork within one process.
            However, in case we're using only one node this should be the preferred setup.]
    """
#{{{
    if 'SLURM_SRUN_COMM_HOST' in os.environ :
        # we were launched using srun and can use its environment variables
        # we check for an environment variable that we are fairly sure will not be set by sbatch

        settings.MPI_WORLD_SIZE = settings.MPI_WORLD_SIZE\
            .set(int(os.environ['SLURM_NTASKS']))
        settings.MPI_RANK = settings.MPI_RANK\
            .set(int(os.environ['SLURM_PROCID']))
        settings.MPI_LOCAL_WORLD_SIZE = settings.MPI_LOCAL_WORLD_SIZE\
            .set(len(os.environ['SLURM_GTIDS'].split(',')))
        settings.MPI_LOCAL_RANK = settings.MPI_LOCAL_RANK\
            .set(int(os.environ['SLURM_LOCALID']))
        settings.MPI_NODENAME = settings.MPI_NODENAME\
            .set(os.environ['SLURMD_NODENAME'])
        settings.MASTER_ADDR = settings.MASTER_ADDR\
            .set(os.environ['SLURM_SRUN_COMM_HOST'])

    elif 'OMPI_COMM_WORLD_SIZE' in os.environ :
        # we were launched using mpirun and can use its environment variables

        settings.MPI_WORLD_SIZE = settings.MPI_WORLD_SIZE\
            .set(int(os.environ['OMPI_COMM_WORLD_SIZE']))
        settings.MPI_RANK = settings.MPI_RANK\
            .set(int(os.environ['OMPI_COMM_WORLD_RANK']))
        settings.MPI_LOCAL_WORLD_SIZE = settings.MPI_LOCAL_WORLD_SIZE\
            .set(int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE']))
        settings.MPI_LOCAL_RANK = settings.MPI_LOCAL_RANK\
            .set(int(os.environ['OMPI_COMM_WORLD_NODE_RANK']))

        # unfortunately, mpirun does not set a global root, so we have to figure
        # this one out ourselves
        settings.MPI_NODENAME = settings.MPI_NODENAME\
            .set(os.environ['HOSTNAME'])

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        assert comm.Get_rank() == settings.MPI_RANK
        if settings.MPI_RANK == 0 :
            root_name = settings.MPI_NODENAME
        else :
            root_name = None
        root_name = comm.bcast(root_name, root=0)
        settings.MASTER_ADDR = settings.MASTER_ADDR.set(root_name)

    else :
        # we were not launched using either srun or mpirun (i.e. this is a single-node job)

        settings.MPI_WORLD_SIZE = settings.MPI_WORLD_SIZE.set()
        settings.MPI_RANK = settings.MPI_RANK.set()
        settings.MPI_LOCAL_WORLD_SIZE = settings.MPI_LOCAL_WORLD_SIZE.set()
        settings.MPI_LOCAL_RANK = settings.MPI_LOCAL_RANK.set()
        settings.MPI_NODENAME = settings.MPI_NODENAME.set()
        settings.MASTER_ADDR = settings.MASTER_ADDR.set()

    # we assume that each process has the same number of GPUs available
    settings.VISIBLE_GPUS = settings.VISIBLE_GPUS.set(torch.cuda.device_count())

    if settings.VISIBLE_GPUS == 1 :
        settings.MPI_ENV_TYPE = settings.MPI_ENV_TYPE.set(settings.MPIEnvTypes.SINGLEGPU)
        settings.WORLD_SIZE = settings.WORLD_SIZE.set(settings.MPI_WORLD_SIZE)

    elif settings.VISIBLE_GPUS > 1 and settings.MPI_LOCAL_WORLD_SIZE > 1 :
        assert settings.MPI_LOCAL_WORLD_SIZE <= settings.VISIBLE_GPUS
        settings.MPI_ENV_TYPE = settings.MPI_ENV_TYPE.set(settings.MPIEnvTypes.MULTIGPU_MULTIRANK)
        settings.WORLD_SIZE = settings.WORLD_SIZE.set(settings.MPI_WORLD_SIZE)

    elif settings.VISIBLE_GPUS > 1 and settings.MPI_LOCAL_WORLD_SIZE == 1 :
        assert settings.MPI_LOCAL_RANK == 0
        settings.MPI_ENV_TYPE = settings.MPI_ENV_TYPE.set(settings.MPIEnvTypes.MULTIGPU_SINGLERANK)
        settings.WORLD_SIZE = settings.WORLD_SIZE.set(settings.VISIBLE_GPUS * settings.MPI_WORLD_SIZE)

    else :
        raise RuntimeError('Invalid MPI environment')
#}}}


def set_mp_env_for_rank(rank) :
    """
    things we need to do once we know the rank we're on
    The argument passed is the local rank, i.e. within a single MPI process
    the rank that we get from spawning the training function
    """
#{{{
    assert not settings.THIS_RANK_SET

    if (settings.MPI_ENV_TYPE is settings.MPIEnvTypes.SINGLEGPU) \
       or (settings.MPI_ENV_TYPE is settings.MPIEnvTypes.MULTIGPU_MULTIRANK) :
        assert rank == 0

    # we are in a specific process
    settings.LOCAL_RANK = rank

    if (settings.MPI_ENV_TYPE is settings.MPIEnvTypes.SINGLEGPU) \
       or (settings.MPI_ENV_TYPE is settings.MPIEnvTypes.MULTIGPU_MULTIRANK) :
        settings.RANK = settings.MPI_RANK

    elif settings.MPI_ENV_TYPE is settings.MPIEnvTypes.MULTIGPU_SINGLERANK :
        settings.RANK = settings.MPI_RANK * settings.VISIBLE_GPUS + settings.LOCAL_RANK


    if settings.MPI_ENV_TYPE is settings.MPIEnvTypes.SINGLEGPU :
        settings.DEVICE_IDX = 0

    elif settings.MPI_ENV_TYPE is settings.MPIEnvTypes.MULTIGPU_MULTIRANK :
        settings.DEVICE_IDX = settings.MPI_LOCAL_RANK

    elif settings.MPI_ENV_TYPE is settings.MPIEnvTypes.MULTIGPU_SINGLERANK :
        settings.DEVICE_IDX = settings.LOCAL_RANK

    if settings.VERBOSE :
        # give some diagnostic output
        print(f'On NODENAME {settings.MPI_NODENAME}: '\
              f'On MPI_RANK {settings.MPI_RANK+1}: '\
              f'On LOCAL_RANK {settings.LOCAL_RANK+1}: '\
              f'RANK = {settings.RANK}, '\
              f'DEVICE_IDX = {settings.DEVICE_IDX}, '\
              f'MASTER_ADDR = {settings.MASTER_ADDR}')

    settings.THIS_RANK_SET = True
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
    `mode' refers to the global mode of execution (i.e. training, validation, or testing).

    `rank' is the process-local rank (i.e. within an MPI process)
    """

    assert isinstance(mode, data_loader.DataModes)

    if settings.STARTUP_CALLED :

        if rank is not None and not settings.THIS_RANK_SET :
            # this is the case when the main process has called startup
            # without a specific rank and then the training process
            # calls it again with knowledge of its rank, but without spawning
            assert rank == 0
            set_mp_env_for_rank(rank)

        return

    # this should be called right at the beginning so subsequent
    # action can make use of this information if necessary
    # and we don't have to pass the `mode' argument around
    settings.MODE = settings.MODE.set(mode)

    populate_settings_from_cl()
    set_mp_env()
    if rank is not None :
        set_mp_env_for_rank(rank)
    set_filenames()
    load_normalizations()
    set_remaining()

    # tell anyone using the settings module that it
    # is in the correct state
    # (all global variables have been updated)
    settings.STARTUP_CALLED = True
