
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
    def __init__(self) :
        pass
    def set(self, value) :
        return value
#}}}

# this one is simply for debugging purposes, functions that use variables from
# this module can check whether they have a correct view
STARTUP_CALLED = False

# the global mode of execution
MODE = ToSet()

# the global ID of this run
# we can use it, for example, to produce unique output file names
# and identify which files we should load as input during testing
ID = ToSet()

# whether we want to map only from the zero DC mode to others
# or, if False, from any DC mode to any other
ONLY_FROM_ZERO = True

# wether we want to include the density field as a separate input channel
USE_DENSITY = True

# whether we want to work in little h-units internally
H_UNITS = False

# specify the redshift that we're working at using this switch
# The redshifts are :
#           [0] z = 126
#           [1] z = 3
#           [2] z = 2
#           [3] z = 1
#           [4] z = 0.5
#           [5] z = 0
# FIXME this will be different with the 128 runs
# TODO we can generalize this into a list and map between redshifts.
#      In that case, redshift has to be an additional style
SNAP_IDX = 5

# particles and cells per side
NSIDE = 256

# sidelength of the glass files
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
DATA_PATH = '/projects/QUIJOTE/Leander/SU/ML_fixed_cosmo_DMonly'

# NOTE since we are using distributed training, the actual number of CPUs
#      used for the workers will be num_workers * num_GPUs
DATALOADER_ARGS = dict(batch_size=1,
                       shuffle=True,
                       num_workers=2,
                       pin_memory=True,
                       prefetch_factor=1)

# some integer
DATASET_SHUFFLING_SEED = 137

# these must be consistent between runs
NSEEDS_TESTING = 2
NSEEDS_VALIDATION = 8

# how many data augmentations to do per sample per epoch
# must be <= 48 and divisible by the number of GPUs
# (note that all 48 augmentations will still occur if this is set
#  to less than 48, just not every epoch for every sample)
N_AUGMENTATIONS = 4

# number of styles, 1 for delta_L only
NSTYLES = 1

# where we store the parameters required for the normalization functions 
NORMALIZATION_FILE = 'normalization.npz'

# dicts that store lambda functions for the normalization
DENSITY_NORMALIZATIONS = ToSet()
DISPLACEMENT_NORMALIZATIONS = ToSet()

# set this to a high number (but not too high since we allocate some buffers proportionally)
EPOCHS = 1000

# note that these are somewhat dependent on the specific optimizer chosen in train_utils.py
OPTIMIZER_ARGS = dict(lr=1e-3,
                      betas=(0.9, 0.999), # TODO according to Paco, this may not be the best choice
                      eps=1e-8, # default value from pytorch docs
                      weight_decay=0.0, # L2 penalty
                      amsgrad=False)

# where all data files will go
RESULTS_PATH = 'results'

# where we store the training and validation loss
LOSS_FILE = ToSet()

# where to store the model during training
MODEL_FILE = ToSet()
