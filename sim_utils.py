from filelock import FileLock
from time import time

import numpy as np
import h5py

import settings



def get_displacement(x1, x2, box_size) :
    """
    Computes the displacement field x2-x1, taking into account
    periodic boundary conditions

    In detail, we have to deal with the following configurations
    (where B is the box size):
    
    1) 0<-----x1-----------------------x2------>B
       with x2-x1 > +B/2
        --> answer = x2 - (x1+B) = (x2-x1) - B

    2) 0<-----x2-----------------------x1------>B
       with x2-x1 < -B/2
       --> answer = (x2+B) - x1 = (x2-x1) + B

    In all other cases (i.e. |x2-x1| < B/2), we can simply take the signed difference.

    The resulting field is in the interval [ -1/2, 1/2 ] * BoxSize
    """
#{{{
    x2 -= x1

    x2[x2 > +0.5*box_size] -= box_size
    x2[x2 < -0.5*box_size] += box_size

    return x2
#}}}


def validate_part_pos(x, box_size, msg=None, check=False, save=False) :
    """
    a debugging helper function
    x are the computed particle positions (a 3D numpy array)

    The check function tests that the standard deviation of particle positions from grid points
    as well as the mean of this field are reasonably small, this strictly only makes sense for
    the high-z reference snaps.
    The checks are not very strict but very likely to fail if there is some indexing problem.

    If save is True, the array x will be saved to a file (the filename is made unique using the time function).
    This functionality should only be used in something resembling interactive mode,
    since otherwise A LOT of files will be generated.
    """
#{{{
    assert x.shape[0] == x.shape[1] == x.shape[2]
    assert x.shape[3] == 3

    if msg is not None :
        assert isinstance(msg, str)
    
    # these are the `left' cell edges
    cell_pos_1d = np.linspace(0, box_size, num=x.shape[0], endpoint=False)

    cell_pos_3d = np.stack(np.meshgrid(*[cell_pos_1d,]*3, indexing='ij'),
                           axis=-1)

    # periodic distance
    d = get_displacement(x, cell_pos_3d, box_size)

    if msg is not None :
        print('***DEBUGGING (validate_part_pos) %s: mean(d)=%f, mean(|d|)=%f, std(d)=%f, max(|d|)=%f'%(msg, np.mean(d), np.mean(np.fabs(d)), np.std(d), np.max(np.fabs(d))))

    if save :
        np.save(f'test_{time()}.npy', x)

    if check :
        assert np.std(d) < 0.1 * box_size
        assert np.max(np.fabs(d)) < 0.1 * box_size
        assert np.mean(np.fabs(d)) < 0.1 * box_size
#}}}


def load_particles(fname) :
    """
    Returns dark matter particle positions from the file fname.
    The positions are ordered in such a way that the output of
    shape [NSIDE, NSIDE, NSIDE, 3] is in the correct
    order (i.e. the individual particles are associated with their cells,
    which undoes the tiling performed by 2lpt).
    """
#{{{ 
    with FileLock('%s.%s'%(fname, settings.LOCK_EXTENSION),
                  timeout=settings.LOCK_TIMEOUT) :
        with h5py.File(fname, 'r') as f :
            x = f['PartType1/Coordinates'][...][np.argsort(f['PartType1/ParticleIDs'][...]), :]

    assert x.shape[0] == settings.NSIDE**3
    assert x.shape[1] == 3

    factor = int(settings.NSIDE / settings.GLASS_SIDE)
    assert factor * settings.GLASS_SIDE == settings.NSIDE

    x = x.reshape((*[factor,]*3, *[settings.GLASS_SIDE,]*3, 3))
    out = np.empty((*[settings.NSIDE,]*3, 3))

    for ii in range(factor) :
        for jj in range(factor) :
            for kk in range(factor) :
                out[ii*settings.GLASS_SIDE : (ii+1)*settings.GLASS_SIDE,
                    jj*settings.GLASS_SIDE : (jj+1)*settings.GLASS_SIDE,
                    kk*settings.GLASS_SIDE : (kk+1)*settings.GLASS_SIDE, ...] = x[ii, jj, kk, ...]

    del x
    return out
#}}}
