from glob import glob
from os.path import splitext
from os import system

import numpy as np
import h5py

from voxelize import Voxelize

BOX_N = 256

PATH = '/projects/QUIJOTE/Leander/SU/ML_fixed_cosmo_DMonly/seed*'
CPU_ONLY = False

def get_out_fname(fname) :
    # converts a filename .../snap_+++.hdf5 into a .npz filename for the density field
    directory = splitext(fname)[0]+'_postprocessing/'
    system('mkdir -p '+directory)
    return directory + 'density_%d.npz'%BOX_N

FNAMES = glob(PATH+'/**/snap_%s.hdf5'%('[0-9]' * 3), recursive=True)

with Voxelize(use_gpu=not CPU_ONLY) as v :

    for fname in FNAMES :

        with h5py.File(fname, 'r') as f :
            BoxSize = f['Header'].attrs['BoxSize']
            DesNumNgb = f['Parameters'].attrs['DesNumNgb']
            HubbleParam = f['Parameters'].attrs['HubbleParam']

            coordinates = f['PartType1/Coordinates'][...]
            radii = f['PartType1/SubfindHsml'][...] / np.cbrt(DesNumNgb+1)
            density = f['PartType1/SubfindDensity'][...]

        box = v(BoxSize, coordinates, radii, density, BOX_N)

        out_fname = get_out_fname(fname)
        print('Writing file %s ...'%out_fname)
        np.savez(out_fname,
                 density=box.astype(np.float32),
                 h=HubbleParam,
                 README=np.string_("""
                                   Density is in h-units [i.e. Msun/h / (kpc/h)**3].
                                   Multiply by h**2 to get a density
                                   that is comparable between different DC modes.
                                   """)
                )
