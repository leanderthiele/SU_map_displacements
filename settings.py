
"""
This file contains some global switches etc. that we'd like to
use in different parts of the code.
"""

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

# should be set from the main thread
DEVICE = None

DATALOADER_ARGS = dict(batch_size=1,
                       shuffle=True,
                       num_workers=4,
                       pin_memory=True,
                       prefetch_factor=1)

DATASET_SHUFFLING_SEED = 137
NSAMPLES_TESTING = 5
NSAMPLES_VALIDATION = 10

NSTYLES = 1
