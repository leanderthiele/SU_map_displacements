import os.path

import settings

class SimulationRun :
    """
    represents a single simulation run and implements my canonical directory structure
    """
#{{{
    def __init__(self, dirname) :
        self.dirname = dirname
        seed_root, run_root = os.path.split(os.path.normpath(self.dirname).rstrip('/'))

        self.seed = int(seed_root.split('seed')[-1])

        self.delta_L = float(run_root.strip('p').strip('m'))
        if run_root[-1] == 'm' :
            self.delta_L *= -1.0
        else :
            assert run_root[-1] == 'p'

    def snap_fname(self) :
        # returns the filename of the hdf5 particle catalog
        return os.path.join(self.dirname, 'G4',
                            'snap_%.3d.hdf5'%settings.SNAP_IDX)

    def reference_snap_fname(self) :
        # returns the filename of the high-redshift hdf5 particle catalog
        return os.path.join(self.dirname, 'G4',
                            'snap_%.3d.hdf5'%0)

    def density_fname(self) :
        # returns the filename of the density field
        return os.path.join(self.dirname, 'G4',
                            'snap_%.3d_postprocessing'%settings.SNAP_IDX,
                            'density_%d.npz'%settings.NSIDE)

    def is_zero(self) :
        # returns whether this is a zero delta_L run
        return abs(self.delta_L) < 1e-10
#}}}
