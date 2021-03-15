# call as python create_density_fields.py RANK WORLD_SIZE


from glob import glob
from os.path import splitext
from sys import argv
from os import system

import numpy as np
import h5py

from voxelize import Voxelize

import settings

PATH = settings.DATA_PATH+'/seed*'
CPU_ONLY = False

RANK = int(argv[1])
WORLD_SIZE = int(argv[2])

def get_out_fname(fname) :
    # converts a filename .../snap_+++.hdf5 into a .npz filename for the density field
    directory = splitext(fname)[0]+'_postprocessing/'
    system('mkdir -p '+directory)
    return directory + 'density_%d.npz'%settings.NSIDE

FNAMES = glob(PATH+'/**/snap_%s.hdf5'%('[0-9]' * 3), recursive=True)[RANK::WORLD_SIZE]

with Voxelize(use_gpu=not CPU_ONLY) as v :

    for fname in FNAMES :

        with h5py.File(fname, 'r') as f :
            BoxSize = f['Header'].attrs['BoxSize']
            DesNumNgb = f['Parameters'].attrs['DesNumNgb']
            HubbleParam = f['Parameters'].attrs['HubbleParam']

            coordinates = f['PartType1/Coordinates'][...]
            radii = f['PartType1/SubfindHsml'][...] / np.cbrt(DesNumNgb+1)
            density = f['PartType1/SubfindDensity'][...]

        box = v(BoxSize, coordinates, radii, density, settings.NSIDE)

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
