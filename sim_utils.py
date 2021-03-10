import numpy as np
import h5py

import settings

def load_particles(fname) :
    """
    Returns dark matter particle positions from the file fname.
    The positions are ordered in such a way that the output of
    shape [NSIDE, NSIDE, NSIDE, 3] is in the correct
    order (i.e. the individual particles are associated with their cells,
    which undoes the tiling performed by 2lpt).
    """
#{{{ 
    with h5py.File(fname, 'r') as f :
        x = f['PartType1/Coordinates'][np.argsort(f['PartType1/ParticleIDs'][...]), :]

    assert x.shape[0] == settings.NSIDE**3
    assert x.shape[1] == 3

    factor = int(settings.NSIDE / settings.GLASS_SIDE)
    assert factor * settings.GLASS_SIDE == settings.NSIDE

    x = x.reshape((*[factor,]*3, *[settings.GLASS_SIDE,]*3, 3))
    out = np.empty((*[settings.NSIDE,]*3))

    for ii in range(factor) :
        for jj in range(factor) :
            for kk in range(factor) :
                out[ii*settings.GLASS_SIDE : (ii+1)*settings.GLASS_SIDE,
                    jj*settings.GLASS_SIDE : (jj+1)*settings.GLASS_SIDE,
                    kk*settings.GLASS_SIDE : (kk+1)*settings.GLASS_SIDE, :] = x[ii, jj, kk, ...]

    del x
    return out
#}}}

def get_displacement(x1, x2, box_size) :
    """
    Computes the displacement field x2-x1, taking into account
    periodic boundary conditions
    """
#{{{
    x2 -= x1

    x2[x2 < -0.5*box_size] *= -1.0
    x2[x2 > +0.5*box_size] -= box_size

    return x2
#}}}
