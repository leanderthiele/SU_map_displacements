"""
This file contains some global switches etc. that we'd like to
use in different parts of the code.

We adopt the convention that only variables that are initialized
as instances of the ToSet class are allowed to be changed by the startup.main
method.
This allows us to easily distinguish between real hardcoded values and default
values (all of which can then be found somewhere in startup.py)

This can be consistently implemented by only assigning to variables in this module
through expressions of the form
    `variable = variable.set(value)'
[ objects that are not instances of ToSet won't implement this interface so we'd get an error ]
"""

from enum import Enum, auto

class ToSet :
    """
    settings that are required to be set by startup.main() are initialized
    as instances of this class.

    Implements the set(value) method, which at the moment is trivial but can
    potentially be used to implement some logic later

    At the moment, there's no further logic in this class, but it can be added
    if needed (e.g. with a `required' switch)
    """
#{{{
    def __init__(self, default) :
        self.default = default

    def set(self, value=None) :
        if value is None :
            if self.default is None :   
                raise RuntimeError('default-less ToSet instance not set')
            return self.default
        return value
#}}}

# time in minutes for slurm output files to be considered as belonging to a finished job,
# used in remove_locks.py
# this should be safely longer than the interval we typically write into our output files
# with and can be quite large since it is only a fallback option in case the output file
# for a finished job does not end with some sort of error message
SLURM_TIME_DIFF = 10.0

# this one is simply for debugging purposes, functions that use variables from
# this module can check whether they have a correct view
STARTUP_CALLED = False

# whether to give some more detailed print output
VERBOSE = True

# the global mode of execution
MODE = ToSet(None)

# the global ID of this run
# we can use it, for example, to produce unique output file names
# and identify which files we should load as input during testing
ID = ToSet('debug')

# whether we want to map only from the zero DC mode to others
# or, if False, from any DC mode to any other
ONLY_FROM_ZERO = True

# wether we want to include the density field as a separate input channel
USE_DENSITY = ToSet(True)

# whether we want to work in little h-units internally
H_UNITS = False

# in which range we want to train (can restrict initially to make task simpler)
DELTA_L_BOUNDS = [ToSet(0.0), ToSet(100)]

# whether to attempt warm starting when loading a model from disk that doesn't
# exactly fit the network architecture
WARMSTART = ToSet(True)

# specify the redshift that we're working at using this switch
# TODO we can generalize this into a list and map between redshifts.
#      In that case, redshift has to be an additional style
SNAP_IDX = 14

# particles and cells per side
# NOTE this can't be a ToSet instance as we use it in remove_locks.py
NSIDE = 128

# sidelength of the glass files -- we need this to reorder the particles properly
GLASS_SIDE = 64

# where we find the simulations
# the file structure is
# DATA_PATH/
#   seed<SEED>/
#      <abs(DELTA)><sgn(DELTA)>/ (with DELTA formatted as %.8f)
#        G4/
#          snap_<SNAP_IDX>.hdf5 (with SNAP_IDX formatted as %.3d)
#          snap_<SNAP_IDX>_postprocessing/
#            density_<NSIDE>.npz/
#              'density' (in h-units, np.float32)
#              'h' (hubble parameter)
# NOTE this can't be a ToSet instance as we use it in remove_locks.py
DATA_PATH = '/projects/QUIJOTE/Leander/SU/ML_fixed_cosmo_DMonly_128'

# how many seconds to wait for a file to become unlocked
# we use file locks to make sure we don't read from hdf5, npy, npz files concurrently
# -- I'm not sure if this is an issue at all but I experienced problems with this
#    in the DM_to_electrons project (in that case we used MPI to synchronize concurrent hdf5 reads,
#                                    but this is perhaps not the best solution here as parallelism
#                                    is rather complicated in this code, there are two levels, one from
#                                    srun and the other from potentially multiple workers)
LOCK_EXTENSION = 'LCK'
LOCK_TIMEOUT = 10.0

# note : the actual batch size is num_GPUs * DATALOADER_ARGS[batch_size] * BATCH_SIZE
#        in practice, the dataloader batch size can only be 1 because otherwise
#        we'll run out of GPU memory
#
# TODO the only benefit of this being >1 would be the batch normalization.
#      Unfortunately, the SyncBatchNorm module has a synchronization hardcoded
#      in the forward pass, which makes it impossible to do this.
#      Thus, for now we'll have to live with batch size = num_GPUs,
#      if it gets really bad we can try to hack the SyncBatchNorm code
#      (torch/nn/modules/_functions.py)
#
# note : there's virtually no runtime benefit in setting this to values larger than
#        one (the reason is probably that the forward() call is always synchronizing)
BATCH_SIZE = ToSet(1)

# Note: since we are using distributed training, the actual number of CPUs
#       used for the workers will be num_workers * num_GPUs
DATALOADER_ARGS = dict(batch_size=1,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=True,
                       prefetch_factor=1)

# some integer
DATASET_SHUFFLING_SEED = 137

# these must be consistent between runs
NSEEDS_TESTING = 2
NSEEDS_VALIDATION = 4

# how many data augmentations to do per sample per epoch
# must be <= 48 and divisible by the number of GPUs
# (note that all 48 augmentations will still occur if this is set
#  to less than 48, just not every epoch for every sample)
N_AUGMENTATIONS = ToSet(4)

# N_layers argument for most blocks (except the collapsing one at the end)
DEFAULT_NLAYERS = ToSet(6)

# can only increase this if USE_DENSITY=False, otherwise memory runs out
NLEVELS = ToSet(5)

# combination mode for the residual connections within blocks
# (if False, concatenation is used)
RESIDUAL_ADD = ToSet(True)

# combination mode for the skip connections within levels
# (if False, concatenation is used)
SKIP_ADD = ToSet(True)

# leaky relu slope -- default is what I tried first
LEAKYRELU_SLOPE = ToSet(0.1)

# number of styles, 1 for delta_L only
NSTYLES = 1

# where we store the parameters required for the normalization functions 
# NOTE this can't be a ToSet instance as we use it in remove_locks.py
NORMALIZATION_FILE = 'normalization_128.npz'

# dicts that store lambda functions for the normalization
DENSITY_NORMALIZATIONS = ToSet(None)
DISPLACEMENT_NORMALIZATIONS = ToSet(None)
DELTA_L_NORMALIZATIONS = ToSet(None)

# set this to a high number and train until time is up
EPOCHS = 1000

# note that these are somewhat dependent on the specific optimizer chosen in train_utils.py
OPTIMIZER_ARGS = dict(lr=ToSet(1e-3),
                      betas=(0.5, 0.999), # according to Paco, this may be better than the default (0.9, 0.999)
                      eps=1e-8, # default value from pytorch docs
                      weight_decay=0.0, # L2 penalty
                      amsgrad=False)

# where all data products will go
RESULTS_PATH = 'results'

# where we store the training and validation loss
LOSS_FILE = ToSet(RESULTS_PATH+'/loss.npz')

# where to store the model during training
MODEL_FILE = ToSet(RESULTS_PATH+'/model.pt')

# multiprocessing environment -- these can be set from anywhere
MPI_WORLD_SIZE = ToSet(1)
MPI_RANK = ToSet(0)
MPI_LOCAL_WORLD_SIZE = ToSet(1)
MPI_LOCAL_RANK = ToSet(0)
MPI_NODENAME = ToSet('localhost')

# categorize the type of environment we're in
class MPIEnvTypes(Enum) :
    SINGLEGPU = auto() # there's one GPU visible
    MULTIGPU_MULTIRANK = auto() # there are multiple GPUs visible and there are multiple ranks on this node
    MULTIGPU_SINGLERANK = auto() # there are multiple GPUs visible and there is a single rank on this node

MPI_ENV_TYPE = ToSet(None)

MASTER_ADDR = ToSet('localhost')
MASTER_PORT = '12355'
VISIBLE_GPUS = ToSet(None)
WORLD_SIZE = ToSet(None) # size of the entire team

# multiprocessing environment -- these can be set only on a specific rank
# the default values are not usable
# Note that these are special in that we can't make them instances of ToSet because
# otherwise they'd be set to default in the main thread but then the training process
# wouldn't be able to override them (no .set method)
THIS_RANK_SET = False
LOCAL_RANK = -1 # refers to rank within one MPI process
RANK = -1 # refers to rank within the entire team
DEVICE_IDX = -1
