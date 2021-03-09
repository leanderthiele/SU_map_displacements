import numpy as np
import h5py


class SimUtils :
    """
    Simply a namespace-like construct so that we don't have
    to write an extra module for this stuff.
    """

    @staticmethod
    def load_particles(fname, part_side=256, glass_side=64) :
        """
        Returns dark matter particle positions from the file fname.
        The positions are ordered in such a way that the output of
        shape [part_side, part_side, part_side, 3] is in the correct
        order (i.e. the individual particles are associated with their cells,
        which undoes the tiling performed by 2lpt).
        """
    #{{{ 
        with h5py.File(fname, 'r') as f :
            x = f['PartType1/Coordinates'][np.argsort(f['PartType1/ParticleIDs'][...]), :]
        
        assert(x.shape[0] == part_side**3)
        assert(x.shape[1] == 3)

        factor = int(part_side / glass_side)
        assert(factor * glass_side == part_side)

        x = x.reshape((factor, factor, factor, glass_side, glass_side, glass_side, 3))
        out = np.empty((part_side, part_side, part_side))

        for ii in range(factor) :
            for jj in range(factor) :
                for kk in range(factor) :
                    out[ii*glass_side : (ii+1)*glass_side,
                        jj*glass_side : (jj+1)*glass_side,
                        kk*glass_side : (kk+1)*glass_side, :] = x[ii, jj, kk, ...]

        del x
        return out
    #}}}

    @staticmethod
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
